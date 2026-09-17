#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <glm/vec3.hpp>

namespace RadFiled3D::nn {

/// Channel the runtime writes its prediction into, and the layers inside it. Shared constants so
/// a producer and a consumer never have to agree on a name out of band.
inline constexpr std::string_view kPredictionChannel = "prediction";
inline constexpr std::string_view kFluxLayer = "flux";
inline constexpr std::string_view kSpectrumLayer = "spectrum";

struct FieldGeometry {
    enum class CoordinateSystem : unsigned char {
        Cartesian,
        Polar
    };

    virtual FieldGeometry::CoordinateSystem coordinate_system() const noexcept = 0;
};

/// A Cartesian voxel grid over a metric box.
///
/// The rule inherited from the training runtime: THE BOX IS METRIC AND THE VOXEL SIZE FOLLOWS FROM
/// THE RESOLUTION, never the other way round. Constructing from a voxel size invites a grid that
/// does not tile the box. `make` and `cubic` are the only constructors and both validate.
struct CartesianFieldGeometry : public FieldGeometry {
    std::array<std::uint32_t, 3> voxel_counts{};
    std::array<float, 3> field_dimensions_m{};

    CartesianFieldGeometry(const std::array<std::uint32_t, 3>& voxel_counts, const std::array<float, 3>& field_dimensions_m) {
        this->voxel_counts = voxel_counts;
        this->field_dimensions_m = field_dimensions_m;
    }

    /// Throws `InvalidArgument` on a zero count or a non-positive / non-finite edge. A NaN edge
    /// would otherwise silently poison every voxel coordinate derived from it.
    static CartesianFieldGeometry make(std::array<std::uint32_t, 3> voxel_counts,
                              std::array<float, 3> field_dimensions_m);
    /// A cubic box, the common case for a trained field.
    static CartesianFieldGeometry cubic(std::uint32_t resolution, float field_box_m);

    virtual FieldGeometry::CoordinateSystem coordinate_system() const noexcept override final { return CoordinateSystem::Cartesian; };

    std::array<float, 3> get_voxel_dimensions_m() const noexcept;
    std::uint64_t get_voxel_count() const noexcept;

    glm::vec3 get_field_dimensions() const noexcept;
    glm::vec3 get_voxel_dimensions() const noexcept;
    glm::uvec3 get_voxel_counts() const noexcept;

    bool operator==(const CartesianFieldGeometry&) const = default;
};

}  // namespace RadFiled3D::nn