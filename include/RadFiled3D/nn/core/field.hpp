// Field geometry, the GPU-resident radiation fields, and the host field.
//
// There is no wrapper type for a host field: a host field IS `RadFiled3D::CartesianRadiationField`,
// used through RadFiled3D's own API (`add_channel`, `get_channel`, `add_layer<float>`,
// `get_layer<float>`, `FieldStore::store`). This header adds the two things RadFiled3D does not
// have — a geometry value type that enforces the box/resolution rule, and a family of field types
// whose layers can carry a GPU mirror — and a few helpers that construct and check them.
#pragma once

#include <RadFiled3D/nn/exception.hpp>
#include <RadFiled3D/nn/memory/memory_ref.hpp>

#include <RadFiled3D/RadiationField.hpp>
#include <RadFiled3D/VoxelBuffer.hpp>
#include <glm/vec3.hpp>

#include <RadFiled3D/nn/core/types.hpp>

// The field types belong to RadFiled3D itself, beside CartesianRadiationField and
// PolarRadiationField: a generated field IS a RadFiled3D field (R-C1), so it is declared where
// every other field type lives. Everything else this module owns stays in RadFiled3D::nn, and
// every file this module ships stays under RadFiled3D/nn/ so ownership is legible.
namespace RadFiled3D {

/// A host `VoxelBuffer` whose layers can additionally carry a GPU mirror.
///
/// The host data stays the source of truth; the mirror is what a compute or graphics backend
/// writes into. It is a `MemoryRef`, so this header pulls in no GPU SDK and a consumer that only
/// wants the geometry never pays for CUDA or Vulkan.
class GPUVoxelBuffer : public VoxelBuffer {
public:
    explicit GPUVoxelBuffer(std::size_t voxel_count) : VoxelBuffer(voxel_count) {}

    void set_device_memory(const std::string& layer, std::shared_ptr<nn::memory::MemoryRef> memory) {
        device_memory_[layer] = std::move(memory);
    }

    /// The GPU mirror for a layer, or null when it has none.
    std::shared_ptr<nn::memory::MemoryRef> get_device_memory(const std::string& layer) const {
        const auto found = device_memory_.find(layer);
        return found == device_memory_.end() ? nullptr : found->second;
    }

private:
    std::map<std::string, std::shared_ptr<nn::memory::MemoryRef>> device_memory_;
};

/// The GPU-aware field base: a `RadiationField<BufferT>` whose layers carry a device mirror.
///
/// Why this derives from the TEMPLATE `RadiationField<BufferT>` rather than from
/// `CartesianRadiationField`: upstream hardcodes `CartesianRadiationField : RadiationField<
/// VoxelGridBuffer>`, so the buffer type is not substitutable there. Deriving from the template with
/// our own buffer keeps every existing layer/voxel accessor and the `FieldStore` serialiser working,
/// and adds the one thing a generated field needs.
///
/// The geometry stays in the concrete subclass, exactly as upstream splits `RadiationField<BufferT>`
/// from `CartesianRadiationField` and `PolarRadiationField`, so a polar GPU field costs one class
/// and no change here.
///
/// Ownership: `IRadiationField` declares NO virtual destructor, so deleting one through a base
/// pointer — `delete base`, or a `unique_ptr<IRadiationField>` — is undefined behaviour. Construct
/// with `std::make_shared`, keep it in `shared_ptr`, and hand that to RadFiled3D, which is also what
/// its own API hands out everywhere.
template <typename BufferT>
class GPURadiationField : public RadiationField<BufferT> {
public:
    /// Attach the GPU mirror of a layer — the buffer inference writes into. Creates the channel if
    /// it does not exist yet, so a caller can bind before filling.
    void set_device_memory(const std::string& channel, const std::string& layer,
                           std::shared_ptr<nn::memory::MemoryRef> memory) {
        this->add_channel(channel);
        this->get_channel(channel)->set_device_memory(layer, std::move(memory));
    }

    /// The GPU mirror of a layer, or null when it has none — including when the channel itself does
    /// not exist, because "no mirror" is the honest answer either way.
    std::shared_ptr<nn::memory::MemoryRef> get_device_memory(const std::string& channel,
                                                             const std::string& layer) const {
        const auto found = this->channels.find(channel);
        if (found == this->channels.end()) return nullptr;
        return found->second->get_device_memory(layer);
    }
};

/// A Cartesian radiation field backed by `GPUVoxelBuffer` channels.
///
/// Mirrors `CartesianRadiationField`'s geometry API exactly, so it is a drop-in for any read path
/// that only asks for counts and dimensions.
class GPUCartesianRadiationField : public GPURadiationField<GPUVoxelBuffer> {
public:
    explicit GPUCartesianRadiationField(const nn::CartesianFieldGeometry& geometry);
    GPUCartesianRadiationField(const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions);

    const std::string& get_typename() const override {
        static const std::string name = "GPUCartesianRadiationField";
        return name;
    }

    std::shared_ptr<VoxelBuffer> add_channel(const std::string& channel_name) override;
    std::shared_ptr<IRadiationField> copy() const override;

    const glm::vec3& get_voxel_dimensions() const noexcept { return voxel_dimensions_; }
    const glm::uvec3& get_voxel_counts() const noexcept { return voxel_counts_; }
    const glm::vec3& get_field_dimensions() const noexcept { return field_dimensions_; }
    std::size_t get_voxel_count() const noexcept {
        return static_cast<std::size_t>(voxel_counts_.x) * voxel_counts_.y * voxel_counts_.z;
    }
    nn::CartesianFieldGeometry get_geometry() const;

    /// Copy every channel and layer into a plain host `CartesianRadiationField`, which
    /// `FieldStore::store` writes as `.rf3`. GPU mirrors are not transferred: the host data is the
    /// source of truth, so a device-only result must be synced back through its backend first.
    std::shared_ptr<CartesianRadiationField> to_host_field() const;

private:
    const glm::vec3 voxel_dimensions_;
    const glm::uvec3 voxel_counts_;
    const glm::vec3 field_dimensions_;
};

}  // namespace RadFiled3D

namespace RadFiled3D::nn {

/// The host field type. Not a wrapper: this is RadFiled3D's own class.
using HostRadiationField = CartesianRadiationField;

/// Allocate a host field of exactly this geometry.
///
/// RadFiled3D derives its own voxel counts from `box / voxel size`; if they disagree with the
/// requested counts the grid does not tile the box and every voxel index would be shifted, so the
/// field is refused rather than returned.
std::shared_ptr<HostRadiationField> allocate_host_field(const CartesianFieldGeometry& geometry);
std::shared_ptr<GPUCartesianRadiationField> allocate_gpu_field(const CartesianFieldGeometry& geometry);

/// The geometry a RadFiled3D field actually has.
CartesianFieldGeometry get_geometry_of(const HostRadiationField& field);

/// Load a `.rf3` through RadFiled3D's own `FieldStore`. Refuses a polar field, which this Cartesian
/// API cannot hold.
std::shared_ptr<HostRadiationField> load_host_field(const std::filesystem::path& path);

/// Who wrote a `.rf3`; goes into the mandatory software half of RadFiled3D's metadata header.
struct StoreProvenance {
    std::string software = "radfiled3d-nn";
    std::string version;
    std::string repository;
    std::string commit;
};

/// Serialise as `.rf3` through RadFiled3D's own `FieldStore`. A generated field has no Monte-Carlo
/// simulation behind it, so the simulation half of the header stays at its defaults.
void store_host_field(std::shared_ptr<HostRadiationField> field, const std::filesystem::path& path,
                      const StoreProvenance& provenance);

}  // namespace RadFiled3D::nn
