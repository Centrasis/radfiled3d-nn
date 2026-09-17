#include <RadFiled3D/nn/core/field.hpp>

#include <RadFiled3D/storage/RadiationFieldStore.hpp>

#include <cmath>
#include <cstring>
#include <limits>

namespace RadFiled3D::nn {

// ── CartesianFieldGeometry ─────────────────────────────────────────────────────────────────────────────────

CartesianFieldGeometry CartesianFieldGeometry::make(std::array<std::uint32_t, 3> voxel_counts, std::array<float, 3> field_dimensions_m) {
    for (const auto c : voxel_counts)
        if (c == 0) throw Exception::invalid_argument("voxel counts must all be positive");
    // `<= 0` alone would let a NaN through.
    for (const float d : field_dimensions_m)
        if (!std::isfinite(d) || d <= 0.f) throw Exception::invalid_argument("field dimensions must all be positive");
    return CartesianFieldGeometry{voxel_counts, field_dimensions_m};
}

CartesianFieldGeometry CartesianFieldGeometry::cubic(std::uint32_t resolution, float field_box_m) {
    return make({resolution, resolution, resolution}, {field_box_m, field_box_m, field_box_m});
}

std::array<float, 3> CartesianFieldGeometry::get_voxel_dimensions_m() const noexcept {
    return {field_dimensions_m[0] / static_cast<float>(voxel_counts[0]),
            field_dimensions_m[1] / static_cast<float>(voxel_counts[1]),
            field_dimensions_m[2] / static_cast<float>(voxel_counts[2])};
}

std::uint64_t CartesianFieldGeometry::get_voxel_count() const noexcept {
    return static_cast<std::uint64_t>(voxel_counts[0]) * voxel_counts[1] * voxel_counts[2];
}

glm::vec3 CartesianFieldGeometry::get_field_dimensions() const noexcept {
    return {field_dimensions_m[0], field_dimensions_m[1], field_dimensions_m[2]};
}

glm::vec3 CartesianFieldGeometry::get_voxel_dimensions() const noexcept {
    const auto v = get_voxel_dimensions_m();
    return {v[0], v[1], v[2]};
}

glm::uvec3 CartesianFieldGeometry::get_voxel_counts() const noexcept { return {voxel_counts[0], voxel_counts[1], voxel_counts[2]}; }

}  // namespace RadFiled3D::nn

// ── GPUCartesianRadiationField ─────────────────────────────────────────────────────────────────────────────
// Declared in RadFiled3D itself (field.hpp), so its members are defined here. The helpers below sit
// in RadFiled3D's anonymous namespace: lookup from RadFiled3D::nn walks up, so the host helpers
// further down reach them unqualified.

namespace RadFiled3D {

namespace {
// Upstream's count rule verbatim (RadiationField.cpp): the epsilon before the divide keeps an exact
// box/voxel ratio from losing a voxel to float truncation.
glm::uvec3 derive_counts(const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions) {
    return glm::uvec3((field_dimensions + glm::vec3(std::numeric_limits<float>::epsilon())) / voxel_dimensions);
}

void check_positive(const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions) {
    // A zero voxel edge would divide by zero inside the count computation, so it is rejected here
    // rather than producing a degenerate field.
    for (int i = 0; i < 3; ++i)
        if (!(field_dimensions[i] > 0.f) || !(voxel_dimensions[i] > 0.f))
            throw nn::Exception::invalid_argument("field and voxel dimensions must all be positive");
}

void check_derived_counts(const glm::uvec3& derived, const nn::CartesianFieldGeometry& geometry) {
    if (derived != geometry.get_voxel_counts())
        throw nn::Exception::invalid_argument("grid [" + std::to_string(geometry.voxel_counts[0]) + ", " +
                                      std::to_string(geometry.voxel_counts[1]) + ", " +
                                      std::to_string(geometry.voxel_counts[2]) + "] does not tile the box: RadFiled3D derived [" +
                                      std::to_string(derived.x) + ", " + std::to_string(derived.y) + ", " +
                                      std::to_string(derived.z) + "]");
}
}  // namespace

GPUCartesianRadiationField::GPUCartesianRadiationField(const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions)
    : voxel_dimensions_(voxel_dimensions),
      voxel_counts_((check_positive(field_dimensions, voxel_dimensions), derive_counts(field_dimensions, voxel_dimensions))),
      field_dimensions_(field_dimensions) {}

GPUCartesianRadiationField::GPUCartesianRadiationField(const nn::CartesianFieldGeometry& geometry)
    : GPUCartesianRadiationField(geometry.get_field_dimensions(), geometry.get_voxel_dimensions()) {
    check_derived_counts(voxel_counts_, geometry);
}

std::shared_ptr<VoxelBuffer> GPUCartesianRadiationField::add_channel(const std::string& channel_name) {
    const auto found = this->channels.find(channel_name);
    if (found != this->channels.end()) return found->second;
    auto buffer = std::make_shared<GPUVoxelBuffer>(this->get_voxel_count());
    this->channels[channel_name] = buffer;
    return buffer;
}

std::shared_ptr<IRadiationField> GPUCartesianRadiationField::copy() const {
    auto field = std::make_shared<GPUCartesianRadiationField>(field_dimensions_, voxel_dimensions_);
    for (const auto& [name, buffer] : this->channels) {
        auto* copied = static_cast<GPUVoxelBuffer*>(buffer->copy());
        field->channels[name] = std::shared_ptr<GPUVoxelBuffer>(copied);
    }
    return field;
}

nn::CartesianFieldGeometry GPUCartesianRadiationField::get_geometry() const {
    return nn::CartesianFieldGeometry{{voxel_counts_.x, voxel_counts_.y, voxel_counts_.z},
                             {field_dimensions_.x, field_dimensions_.y, field_dimensions_.z}};
}

std::shared_ptr<CartesianRadiationField> GPUCartesianRadiationField::to_host_field() const {
    auto host = std::make_shared<CartesianRadiationField>(field_dimensions_, voxel_dimensions_);
    for (const auto& [channel_name, buffer] : this->channels) {
        auto target = host->add_channel(channel_name);
        for (const auto& layer_name : buffer->get_layers()) {
            const VoxelLayer& layer = buffer->get_layer(layer_name);
            if (buffer->get_voxel_count() == 0) continue;
            // Replicate the layer's voxel type from a template voxel, then byte-copy the data. This
            // works for any voxel type — scalar, histogram, angular — because get_bytes() is the
            // per-voxel data size whatever the type.
            const IVoxel* tmpl = layer.get_voxel_flat_raw(0);
            target->add_custom_layer_unsafe(layer_name, tmpl, layer.get_unit());
            const std::size_t bytes = tmpl->get_bytes() * buffer->get_voxel_count();
            std::memcpy(target->get_layer<char>(layer_name), layer.get_raw_data(), bytes);
        }
    }
    return host;
}

}  // namespace RadFiled3D

namespace RadFiled3D::nn {

// ── host helpers ──────────────────────────────────────────────────────────────────────────────────

std::shared_ptr<HostRadiationField> allocate_host_field(const CartesianFieldGeometry& geometry) {
    auto field = std::make_shared<HostRadiationField>(geometry.get_field_dimensions(), geometry.get_voxel_dimensions());
    check_derived_counts(field->get_voxel_counts(), geometry);
    // RadFiled3D stores the voxel size verbatim, so this compares exactly; a difference would mean
    // the two sides disagree about what a voxel index maps to in metres.
    if (field->get_voxel_dimensions() != geometry.get_voxel_dimensions())
        throw Exception::invalid_argument("voxel size disagreement between the requested geometry and RadFiled3D");
    return field;
}

std::shared_ptr<GPUCartesianRadiationField> allocate_gpu_field(const CartesianFieldGeometry& geometry) {
    return std::make_shared<GPUCartesianRadiationField>(geometry);
}

CartesianFieldGeometry get_geometry_of(const HostRadiationField& field) {
    const glm::uvec3 c = field.get_voxel_counts();
    const glm::vec3 v = field.get_voxel_dimensions();
    return CartesianFieldGeometry::make({c.x, c.y, c.z},
                               {static_cast<float>(c.x) * v.x, static_cast<float>(c.y) * v.y, static_cast<float>(c.z) * v.z});
}

std::shared_ptr<HostRadiationField> load_host_field(const std::filesystem::path& path) {
    std::shared_ptr<RadFiled3D::IRadiationField> loaded;
    try {
        loaded = RadFiled3D::Storage::FieldStore::load(path.string());
    } catch (const std::exception& e) {
        throw Exception::io(path.string(), e.what());
    }
    auto field = std::dynamic_pointer_cast<HostRadiationField>(loaded);
    // A polar field is a valid .rf3 but not something this module's Cartesian API can hold, so it
    // is refused here rather than silently reinterpreted.
    if (!field) throw Exception::invalid_argument("`" + path.string() + "` is not a Cartesian RadFiled3D field");
    return field;
}

void store_host_field(std::shared_ptr<HostRadiationField> field, const std::filesystem::path& path,
                      const StoreProvenance& provenance) {
    using namespace RadFiled3D::Storage;
    if (!field) throw Exception::invalid_argument("store_host_field: null field");
    auto metadata = std::make_shared<V1::RadiationFieldMetadata>(
        FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(),
        FiledTypes::V1::RadiationFieldMetadataHeader::Software(provenance.software, provenance.version,
                                                                provenance.repository, provenance.commit));
    try {
        FieldStore::store(field, metadata, path.string(), StoreVersion::V1);
    } catch (const std::exception& e) {
        throw Exception::io(path.string(), e.what());
    }
}

}  // namespace RadFiled3D::nn
