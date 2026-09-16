// The current `.rf3m` layout. Reader and writer.
//
// The byte layout is documented once, on `Package` in package.hpp. This is the only code that
// walks it.
#pragma once

#include <RadFiled3D/nn/deploy/version.hpp>

namespace RadFiled3D::nn::deploy::v1 {

class Serializer final : public deploy::Serializer {
public:
    FormatVersion get_version() const noexcept override { return FormatVersion::V1; }
    bool can_write() const noexcept override { return true; }

    Package read(byte_view bytes) const override;
    Metadata read_metadata(byte_view bytes) const override;
    deploy::bytes write(const Package& package) const override;
};

}  // namespace RadFiled3D::nn::deploy::v1
