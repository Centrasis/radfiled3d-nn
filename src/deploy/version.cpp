#include <RadFiled3D/nn/deploy/version.hpp>

#include <RadFiled3D/nn/deploy/v1.hpp>

#include <algorithm>

namespace RadFiled3D::nn::deploy {

std::string_view to_string(FormatVersion version) noexcept {
    switch (version) {
        case FormatVersion::V1: return "v1";
    }
    return "?";
}

FormatVersion peek_version(byte_view bytes) {
    if (bytes.size() < 12)
        throw Exception::bad_magic(std::to_string(bytes.size()) + " bytes is too short for any RF3M preamble");
    if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin()))
        throw Exception::bad_magic("magic is not \"RF3M\"");

    // The version word decides, and nothing else does. A word this build does not know is reported
    // as such rather than guessed at — the digest is there to prove a package is intact, never to
    // identify which layout it is in.
    Reader r(bytes.subspan(kMagic.size()));
    const std::uint32_t version = r.u32("format version");
    if (version != static_cast<std::uint32_t>(FormatVersion::V1))
        throw Exception::unsupported_version(version, static_cast<std::uint32_t>(kWriteVersion));
    return FormatVersion::V1;
}

const Serializer& get_serializer_by(FormatVersion version) {
    static const v1::Serializer kV1;
    switch (version) {
        case FormatVersion::V1: return kV1;
    }
    throw Exception::unsupported_version(static_cast<std::uint32_t>(version),
                                         static_cast<std::uint32_t>(kWriteVersion));
}

}  // namespace RadFiled3D::nn::deploy
