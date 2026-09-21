// Format versions, and the serializer that implements each one.
//
// `deploy` mirrors RadFiled3D's own versioning, and RadFiled3D's is polymorphic rather than a
// switch: `Storage::` declares the abstract contracts, `Storage::V1::` derives a concrete one from
// each, and `FieldStore::get_store_by(StoreVersion)` hands back the selected implementation. Same
// shape here — `Package::read` / `read_metadata` / `to_bytes` are a façade over `peek_version` +
// `get_serializer_by`, and a caller never names a version namespace.
//
// ONLY THE BYTE WALK IS VERSIONED. `Package`, `Metadata`, `TensorDescriptor` and the rest are one
// in-memory model that every version produces, which is what makes "users only work in ::deploy"
// true. If a future version carries something `Package` cannot express, `Package` grows an optional
// field; it never forks into a v3::Package.
#pragma once

#include <RadFiled3D/nn/deploy/codec.hpp>
#include <RadFiled3D/nn/deploy/package.hpp>

#include <cstdint>

namespace RadFiled3D::nn::deploy {

/// The enumerator value IS the version word this library writes for that layout. (Do not copy
/// RadFiled3D here: its `StoreVersion::V1 = 0`, a 0-based enum matching no byte in its files.)
enum class FormatVersion : std::uint32_t {
    /// The layout: magic, version, digest, skippable metadata, typed blocks.
    V1 = 1,
    /// The SAME layout, under the number it carried before the renumber. Everything after the
    /// version word is byte-identical to V1, so such a package is read by the V1 serializer and
    /// rewritten with `kWriteVersion` — that rewrite is what `rf3m convert` is for. Never written.
    V2_renumbered = 2,
};

std::string_view to_string(FormatVersion version) noexcept;

/// What the writer emits.
inline constexpr FormatVersion kWriteVersion = FormatVersion::V1;

/// What every format version must be able to do. One implementation per version namespace.
class Serializer {
public:
    virtual ~Serializer() = default;

    virtual FormatVersion get_version() const noexcept = 0;
    /// Whether this version can be produced as well as consumed. True for every version that
    /// currently exists; the hook is here for the first one that becomes read-only.
    virtual bool can_write() const noexcept = 0;

    /// Full parse, payloads included.
    virtual Package read(byte_view bytes) const = 0;
    /// Scan: descriptors and the block directory, without touching a payload.
    virtual Metadata read_metadata(byte_view bytes) const = 0;
    /// Throws `UnsupportedVersion` when `can_write()` is false.
    virtual bytes write(const Package& package) const = 0;
};

/// Which layout these bytes are in, read from the preamble alone.
///
/// One version exists, so this checks the magic and the version word and nothing more. It is not
/// redundant: it is the seam a second layout is added at, and keeping the façade expressed in terms
/// of it means adding one costs a namespace and a table entry rather than a rewrite of `Package`.
FormatVersion peek_version(byte_view bytes);

/// The serializer for a version. Throws `UnsupportedVersion` for one this build cannot handle.
const Serializer& get_serializer_by(FormatVersion version);

}  // namespace RadFiled3D::nn::deploy
