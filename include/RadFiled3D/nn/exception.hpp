// The library's one exception type.
//
// Two rules shape it, both from requirements.md:
//
//   * A disabled backend keeps its full API surface and throws `ErrorKind::FeatureDisabled` naming
//     the CMake option that would enable it (R-B3, R-G3). It never falls back to a slower path —
//     that turns a configuration mistake into a performance mystery.
//   * Nothing is guessed. A malformed package, a missing tensor, an unknown enum value: each gets a
//     distinct kind carrying what was expected and what was found.
//
// Exceptions stop at two boundaries: the C ABI (src/capi) turns them into status codes, and the
// Python bindings turn them into Python exceptions. Nothing else catches them.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace RadFiled3D::nn {

enum class ErrorKind : std::uint8_t {
    /// The requested capability was compiled out. `feature()` names the option to enable.
    FeatureDisabled,
    /// The byte stream is not an RF3M package, or its header is unusable.
    BadMagic,
    /// A container version this build cannot read.
    UnsupportedVersion,
    /// A read ran past the end of the buffer.
    Truncated,
    /// The payload digest does not match the one recorded in the header.
    DigestMismatch,
    /// A field held a value outside its defined set.
    InvalidValue,
    /// The package is structurally readable but does not describe a usable model.
    InvalidPackage,
    /// A tensor, channel or layer was addressed by a name the object does not declare.
    NotFound,
    /// A caller-supplied argument is unusable (an unknown backend name, a degenerate geometry).
    InvalidArgument,
    /// The graphics API and the compute backend cannot share memory with each other.
    UnsupportedInterop,
    /// A file could not be read or written.
    Io,
};

std::string_view to_string(ErrorKind kind) noexcept;

class Exception : public std::runtime_error {
public:
    Exception(ErrorKind kind, std::string message)
        : std::runtime_error(std::move(message)), kind_(kind) {}

    ErrorKind get_kind() const noexcept { return kind_; }

    /// For `FeatureDisabled`: the CMake option / compile define that would enable the capability.
    const std::string& get_feature() const noexcept { return feature_; }

    // ── constructors for the kinds that carry structure ─────────────────────────────────────────

    static Exception feature_disabled(std::string_view feature, std::string_view what);
    static Exception truncated(std::string_view context, std::size_t needed, std::size_t available);
    static Exception unsupported_version(std::uint32_t found, std::uint32_t supported);
    static Exception invalid_value(std::string_view what, std::uint64_t value);
    static Exception invalid_package(std::string message);
    static Exception not_found(std::string_view role, std::string_view name);
    static Exception invalid_argument(std::string message);
    static Exception unsupported_interop(std::string_view graphics, std::string_view compute);
    static Exception io(std::string_view path, std::string_view what);
    static Exception bad_magic(std::string message);
    static Exception digest_mismatch();

private:
    ErrorKind kind_;
    std::string feature_;
};

}  // namespace RadFiled3D::nn
