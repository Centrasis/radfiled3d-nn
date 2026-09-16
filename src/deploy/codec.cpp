#include <RadFiled3D/nn/deploy/codec.hpp>

#include <bit>
#include <cstring>

namespace RadFiled3D::nn::deploy {

byte_view Reader::take(std::size_t n, std::string_view ctx) {
    if (remaining() < n) throw Exception::truncated(ctx, n, remaining());
    const byte_view out = bytes_.subspan(pos_, n);
    pos_ += n;
    return out;
}

std::uint8_t Reader::u8(std::string_view ctx) { return take(1, ctx)[0]; }

std::uint32_t Reader::u32(std::string_view ctx) {
    const byte_view b = take(4, ctx);
    return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
           (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

std::uint64_t Reader::u64(std::string_view ctx) {
    const byte_view b = take(8, ctx);
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | b[static_cast<std::size_t>(i)];
    return v;
}

float Reader::f32(std::string_view ctx) { return std::bit_cast<float>(u32(ctx)); }
double Reader::f64(std::string_view ctx) { return std::bit_cast<double>(u64(ctx)); }
bool Reader::boolean(std::string_view ctx) { return u8(ctx) != 0; }

std::string Reader::string(std::string_view ctx) {
    const std::size_t len = u32(ctx);
    const byte_view raw = take(len, ctx);
    return std::string(reinterpret_cast<const char*>(raw.data()), raw.size());
}

byte_view Reader::blob(std::string_view ctx) {
    const std::size_t len = static_cast<std::size_t>(u64(ctx));
    return take(len, ctx);
}

byte_view Reader::rest() noexcept {
    const byte_view out = bytes_.subspan(pos_);
    pos_ = bytes_.size();
    return out;
}

void Writer::u8(std::uint8_t v) { buf_.push_back(v); }

void Writer::u32(std::uint32_t v) {
    for (int i = 0; i < 4; ++i) buf_.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}

void Writer::u64(std::uint64_t v) {
    for (int i = 0; i < 8; ++i) buf_.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}

void Writer::f32(float v) { u32(std::bit_cast<std::uint32_t>(v)); }
void Writer::f64(double v) { u64(std::bit_cast<std::uint64_t>(v)); }
void Writer::boolean(bool v) { u8(v ? 1 : 0); }

void Writer::string(std::string_view v) {
    u32(static_cast<std::uint32_t>(v.size()));
    buf_.insert(buf_.end(), v.begin(), v.end());
}

void Writer::blob(byte_view v) {
    u64(v.size());
    raw(v);
}

void Writer::raw(byte_view v) { buf_.insert(buf_.end(), v.begin(), v.end()); }

}  // namespace RadFiled3D::nn::deploy
