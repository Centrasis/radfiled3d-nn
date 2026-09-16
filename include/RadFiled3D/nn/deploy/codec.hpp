// Byte-level primitives for the RF3M container.
//
// Everything in the format is built from these, and every type that goes into the container
// implements one `encode(Writer&)` / `decode(Reader&)` pair, exactly once. That is the fix for
// defect D1 in requirements.md: the original writes the layout in `save_to_memory` and re-walks it
// by hand in three separate readers, so a field added in one place and forgotten in another
// desynchronises the parse silently.
//
// All integers are little-endian. A string is `[u32 len][utf-8 bytes]`, not NUL-terminated. A blob
// is `[u64 len][bytes]`. Nothing here depends on the host's endianness or on any struct layout.
#pragma once

#include <RadFiled3D/nn/exception.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace RadFiled3D::nn::deploy {

using bytes = std::vector<std::uint8_t>;
using byte_view = std::span<const std::uint8_t>;

/// A cursor over a package's bytes that cannot read past the end.
///
/// Everything it returns for a variable-length field is a *view* into the underlying bytes, so a
/// 400 MB ONNX payload is not copied until the caller decides to own it (R-F1, fixing D4).
class Reader {
public:
    explicit Reader(byte_view bytes) : bytes_(bytes) {}

    std::size_t remaining() const noexcept { return bytes_.size() - pos_; }
    bool empty() const noexcept { return remaining() == 0; }

    std::uint8_t u8(std::string_view ctx);
    std::uint32_t u32(std::string_view ctx);
    std::uint64_t u64(std::string_view ctx);
    float f32(std::string_view ctx);
    double f64(std::string_view ctx);
    bool boolean(std::string_view ctx);
    std::string string(std::string_view ctx);

    /// `[u64 len][bytes]`, returned as a view.
    byte_view blob(std::string_view ctx);

    /// Borrow `n` bytes outright — for a region whose length was read separately.
    byte_view take(std::size_t n, std::string_view ctx);

    /// A `[u32 len]`-prefixed region, handed to `f` as its own reader. Whatever `f` leaves unread
    /// is skipped, which is what makes a payload whose *kind* this build does not understand
    /// harmless.
    template <typename F>
    auto sized(std::string_view ctx, F&& f) {
        const std::size_t len = u32(ctx);
        Reader inner(take(len, ctx));
        return f(inner);
    }

    /// The `[u64 len]` form, for regions that can hold a payload: a block may carry hundreds of
    /// megabytes, which a u32 could not describe.
    template <typename F>
    auto sized_u64(std::string_view ctx, F&& f) {
        const std::size_t len = static_cast<std::size_t>(u64(ctx));
        Reader inner(take(len, ctx));
        return f(inner);
    }

    /// `[u32 count]` followed by that many items, each produced by `f(reader)`.
    template <typename F>
    auto list(std::string_view ctx, F&& f) {
        using T = decltype(f(*this));
        const std::uint32_t n = u32(ctx);
        std::vector<T> out;
        out.reserve(n < 1024 ? n : 1024);
        for (std::uint32_t i = 0; i < n; ++i) out.push_back(f(*this));
        return out;
    }

    /// Everything left, consumed.
    byte_view rest() noexcept;

private:
    byte_view bytes_;
    std::size_t pos_ = 0;
};

/// An append-only byte sink. Mirrors `Reader` one method at a time; the pairing is what keeps a
/// round trip honest.
class Writer {
public:
    std::size_t size() const noexcept { return buf_.size(); }
    bool empty() const noexcept { return buf_.empty(); }
    const bytes& data() const noexcept { return buf_; }
    bytes take() noexcept { return std::move(buf_); }

    void u8(std::uint8_t v);
    void u32(std::uint32_t v);
    void u64(std::uint64_t v);
    void f32(float v);
    void f64(double v);
    void boolean(bool v);
    void string(std::string_view v);
    void blob(byte_view v);
    void raw(byte_view v);

    /// The write side of `Reader::sized`: encodes into a scratch writer to learn the length, then
    /// emits `[u32 len][payload]`.
    template <typename F>
    void sized(F&& f) {
        Writer inner;
        f(inner);
        u32(static_cast<std::uint32_t>(inner.size()));
        raw(inner.data());
    }

    /// The write side of `Reader::sized_u64`.
    template <typename F>
    void sized_u64(F&& f) {
        Writer inner;
        f(inner);
        u64(inner.size());
        raw(inner.data());
    }

    /// `[u32 count]` then `f(writer, item)` per item.
    template <typename Range, typename F>
    void list(const Range& items, F&& f) {
        u32(static_cast<std::uint32_t>(items.size()));
        for (const auto& item : items) f(*this, item);
    }

private:
    bytes buf_;
};

}  // namespace RadFiled3D::nn::deploy
