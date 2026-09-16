// Conversion between RadFiled3D's metadata object and the container's typed form.
//
// This is the only code that knows both `RadFiled3D::Storage::V1::RadiationFieldMetadata` and
// `deploy::Rf3Metadata`, which is exactly where that knowledge belongs. The fixed header is a
// `#pragma pack(4)` POD; it is read and written FIELD BY FIELD through its members, never copied as
// bytes, so nothing durable ever carries a struct image (CLAUDE.md rule 1b). The dynamic layers
// have no typed form and travel as the bytes RadFiled3D's own serialiser produces — the tail after
// the fixed header, because the header is already carried typed.
#pragma once

#include <RadFiled3D/nn/deploy/package.hpp>

#include <RadFiled3D/storage/Types.hpp>

#include <memory>

namespace RadFiled3D::nn {

/// The portable form the container stores.
deploy::Rf3Metadata to_package_metadata(const RadFiled3D::Storage::V1::RadiationFieldMetadata& metadata);

/// Rebuild a RadFiled3D metadata object from the container's typed form.
std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> from_package_metadata(
    const deploy::Rf3Metadata& metadata);

/// The bytes RadFiled3D's serialiser wrote AFTER the fixed header: `serialize()` emits
/// `[header block][fixed header][dynamic tail]`, and only the tail is opaque.
deploy::bytes dynamic_tail(const RadFiled3D::Storage::V1::RadiationFieldMetadata& metadata);

}  // namespace RadFiled3D::nn
