#include <RadFiled3D/nn/exception.hpp>

#include <cctype>

namespace RadFiled3D::nn {

std::string_view to_string(ErrorKind kind) noexcept {
    switch (kind) {
        case ErrorKind::FeatureDisabled: return "feature disabled";
        case ErrorKind::BadMagic: return "bad magic";
        case ErrorKind::UnsupportedVersion: return "unsupported version";
        case ErrorKind::Truncated: return "truncated";
        case ErrorKind::DigestMismatch: return "digest mismatch";
        case ErrorKind::InvalidValue: return "invalid value";
        case ErrorKind::InvalidPackage: return "invalid package";
        case ErrorKind::NotFound: return "not found";
        case ErrorKind::InvalidArgument: return "invalid argument";
        case ErrorKind::UnsupportedInterop: return "unsupported interop";
        case ErrorKind::Io: return "io";
    }
    return "unknown";
}

Exception Exception::feature_disabled(std::string_view feature, std::string_view what) {
    Exception err(ErrorKind::FeatureDisabled,
              std::string(what) + " requires the `" + std::string(feature) +
                  "` feature (CMake option RFNN_WITH_" + [&] {
                      std::string upper(feature);
                      for (auto& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                      return upper;
                  }() + "), which this build was compiled without");
    err.feature_ = std::string(feature);
    return err;
}

Exception Exception::truncated(std::string_view context, std::size_t needed, std::size_t available) {
    return Exception(ErrorKind::Truncated, "truncated RF3M package: " + std::string(context) + " needs " +
                                           std::to_string(needed) + " more bytes, " +
                                           std::to_string(available) + " available");
}

Exception Exception::unsupported_version(std::uint32_t found, std::uint32_t supported) {
    return Exception(ErrorKind::UnsupportedVersion,
                 "unsupported RF3M container version " + std::to_string(found) + " (this build reads " +
                     std::to_string(supported) + ")");
}

Exception Exception::invalid_value(std::string_view what, std::uint64_t value) {
    return Exception(ErrorKind::InvalidValue,
                 "invalid value " + std::to_string(value) + " for " + std::string(what));
}

Exception Exception::invalid_package(std::string message) {
    return Exception(ErrorKind::InvalidPackage, "invalid package: " + std::move(message));
}

Exception Exception::not_found(std::string_view role, std::string_view name) {
    return Exception(ErrorKind::NotFound, "no " + std::string(role) + " named `" + std::string(name) + "`");
}

Exception Exception::invalid_argument(std::string message) {
    return Exception(ErrorKind::InvalidArgument, std::move(message));
}

Exception Exception::unsupported_interop(std::string_view graphics, std::string_view compute) {
    return Exception(ErrorKind::UnsupportedInterop, std::string(graphics) + " memory cannot be shared with the " +
                                                    std::string(compute) + " execution backend");
}

Exception Exception::io(std::string_view path, std::string_view what) {
    return Exception(ErrorKind::Io, "io error on " + std::string(path) + ": " + std::string(what));
}

Exception Exception::bad_magic(std::string message) {
    return Exception(ErrorKind::BadMagic, "not an RF3M package: " + std::move(message));
}

Exception Exception::digest_mismatch() {
    return Exception(ErrorKind::DigestMismatch,
                 "RF3M payload digest mismatch: the package is corrupt or was modified");
}

}  // namespace RadFiled3D::nn
