#include <RadFiled3D/nn/deploy/v1.hpp>

#include <blake3.h>

#include <algorithm>
#include <map>

namespace RadFiled3D::nn::deploy::v1 {

// ── one decoder per part, shared by every entry point ─────────────────────────────────────────────

namespace {

struct MetadataParts {
    Provenance provenance;
    std::vector<TensorDescriptor> io;
    Geometry geometry;
    std::map<std::string, double> metrics;
    std::optional<Rf3Metadata> rf3_metadata;
};

MetadataParts decode_metadata(Reader& r) {
    MetadataParts parts;
    parts.provenance = Provenance::decode(r);
    parts.io = r.list("io descriptors", [](Reader& r) { return TensorDescriptor::decode(r); });
    parts.geometry = Geometry::decode(r);
    for (auto& [key, value] : r.list("metrics", [](Reader& r) {
             std::string key = r.string("metric name");
             return std::pair<std::string, double>(std::move(key), r.f64("metric value"));
         }))
        parts.metrics.emplace(std::move(key), value);
    if (r.boolean("has rf3 metadata"))
        parts.rf3_metadata = r.sized("rf3 metadata", [](Reader& r) { return Rf3Metadata::decode(r); });
    return parts;
}

std::array<std::uint8_t, kDigestBytes> digest_of(byte_view body) {
    std::array<std::uint8_t, kDigestBytes> out{};
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, body.data(), body.size());
    blake3_hasher_finalize(&hasher, out.data(), out.size());
    return out;
}

/// Check magic, version and digest; return the metadata region and a reader positioned at the
/// block count.
std::pair<byte_view, Reader> split(byte_view bytes) {
    if (bytes.size() < kHeaderBytes)
        throw Exception::bad_magic(std::to_string(bytes.size()) + " bytes is shorter than the " +
                               std::to_string(kHeaderBytes) + "-byte header");
    if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) throw Exception::bad_magic("magic is not \"RF3M\"");
    Reader r(bytes.subspan(4));
    const std::uint32_t version = r.u32("file version");
    // There is one container version, so anything else is a package this build cannot read: a
    // newer producer, or a header corrupt past the magic. Reported by number rather than misparsed
    // — the digest below would catch the corruption anyway, but not say what was wrong.
    if (version != kVersion) throw Exception::unsupported_version(version, kVersion);
    const byte_view digest = r.take(kDigestBytes, "digest");
    const auto computed = digest_of(bytes.subspan(kHeaderBytes - 8));
    if (!std::equal(computed.begin(), computed.end(), digest.begin())) throw Exception::digest_mismatch();

    const std::size_t metadata_bytes = static_cast<std::size_t>(r.u64("metadata byte count"));
    const byte_view metadata = r.take(metadata_bytes, "metadata region");
    return {metadata, r};
}

/// Walk the blocks, keeping the payloads.
std::vector<Block> decode_blocks(Reader& r) {
    return r.list("blocks", [](Reader& r) {
        return r.sized_u64("block", [](Reader& r) {
            Block b;
            b.kind = BlockKind::from_name(r.string("block kind"));
            b.name = r.string("block name");
            b.target_arch = r.string("block target arch");
            const byte_view payload = r.rest();
            b.payload.assign(payload.begin(), payload.end());
            return b;
        });
    });
}

/// Walk the blocks, skipping the payloads. This is the scan the format exists to make cheap.
///
/// With ONE exception: a composition block is decoded. It is a few hundred bytes describing how the
/// model fits together rather than being part of it, and a listing that showed which graphs a
/// package carries without showing how they connect would answer the less useful half of the
/// question. Every other payload is still skipped by its length.
std::vector<BlockInfo> scan_blocks(Reader& r, std::optional<Composition>& composition) {
    return r.list("blocks", [&composition](Reader& r) {
        return r.sized_u64("block", [&composition](Reader& r) {
            BlockInfo b;
            b.kind = BlockKind::from_name(r.string("block kind"));
            b.name = r.string("block name");
            b.target_arch = r.string("block target arch");
            b.payload_bytes = r.remaining();
            if (b.kind == BlockKind::Composition) composition = Composition::decode(r);
            return b;
        });
    });
}

}  // namespace

// ── reading ───────────────────────────────────────────────────────────────────────────────────────

Package Serializer::read(byte_view bytes) const {
    auto [metadata, blocks] = split(bytes);
    Reader r(metadata);
    MetadataParts parts = decode_metadata(r);
    Package pkg;
    pkg.provenance = std::move(parts.provenance);
    pkg.io = std::move(parts.io);
    pkg.geometry = parts.geometry;
    pkg.metrics = std::move(parts.metrics);
    pkg.rf3_metadata = std::move(parts.rf3_metadata);
    pkg.blocks = decode_blocks(blocks);
    pkg.validate();
    return pkg;
}

Metadata Serializer::read_metadata(byte_view bytes) const {
    auto [metadata, blocks] = split(bytes);
    Reader r(metadata);
    MetadataParts parts = decode_metadata(r);
    Metadata md;
    md.provenance = std::move(parts.provenance);
    md.io = std::move(parts.io);
    md.geometry = parts.geometry;
    md.metrics = std::move(parts.metrics);
    md.rf3_metadata = std::move(parts.rf3_metadata);
    md.blocks = scan_blocks(blocks, md.composition);
    return md;
}

deploy::bytes Serializer::write(const Package& package) const {
    package.validate();

    Writer metadata;
    package.provenance.encode(metadata);
    metadata.list(package.io, [](Writer& w, const TensorDescriptor& d) { d.encode(w); });
    package.geometry.encode(metadata);
    metadata.list(package.metrics, [](Writer& w, const auto& kv) {
        w.string(kv.first);
        w.f64(kv.second);
    });
    // Optional and length-prefixed, so a reader without it skips exactly the right bytes.
    if (!package.rf3_metadata) {
        metadata.boolean(false);
    } else {
        metadata.boolean(true);
        metadata.sized([&](Writer& w) { package.rf3_metadata->encode(w); });
    }

    Writer body;
    body.u64(metadata.size());
    body.raw(metadata.data());
    body.list(package.blocks, [](Writer& w, const Block& block) {
        // The length covers everything after itself, so a reader that does not want this block
        // seeks past it without parsing a single field.
        w.sized_u64([&](Writer& w) {
            w.string(block.kind.get_name());
            w.string(block.name);
            w.string(block.target_arch);
            w.raw(block.payload);
        });
    });

    Writer out;
    out.raw(kMagic);
    out.u32(kVersion);
    out.raw(digest_of(body.data()));
    out.raw(body.data());
    return out.take();
}


}  // namespace RadFiled3D::nn::deploy::v1
