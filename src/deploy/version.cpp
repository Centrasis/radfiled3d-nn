#include <RadFiled3D/nn/deploy/version.hpp>

#include <RadFiled3D/nn/deploy/v1.hpp>

#include <algorithm>

namespace RadFiled3D::nn::deploy {

std::string_view to_string(FormatVersion version) noexcept {
    switch (version) {
        case FormatVersion::V1: return "v1";
        case FormatVersion::V2_renumbered: return "v2";
    }
    return "?";
}

FormatVersion peek_version(byte_view bytes) {
    if (bytes.size() < 12)
        throw Exception::bad_magic(std::to_string(bytes.size()) + " bytes is too short for any RF3M preamble");
    if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin()))
        throw Exception::bad_magic("magic is not \"RF3M\"");

    // The version word decides, and nothing else does — the digest proves a package is intact,
    // never which layout it is in. Peeking REPORTS the word, including one this build cannot serve:
    // refusing it is the parser's job (`get_serializer_by`), so a tool can name the version of a
    // package it is unable to read.
    Reader r(bytes.subspan(kMagic.size()));
    return static_cast<FormatVersion>(r.u32("format version"));
}

const Serializer& get_serializer_by(FormatVersion version) {
    static const v1::Serializer kV1;
    switch (version) {
        // One layout, two numbers: the renumbered word differs, the bytes it introduces do not.
        case FormatVersion::V1:
        case FormatVersion::V2_renumbered: return kV1;
    }
    throw Exception::unsupported_version(static_cast<std::uint32_t>(version),
                                         static_cast<std::uint32_t>(kWriteVersion));
}

}  // namespace RadFiled3D::nn::deploy
