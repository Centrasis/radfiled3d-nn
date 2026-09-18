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

    /// VIRTUAL, because this base has virtual functions and so may one day be held by pointer. The
    /// same hazard as `IRadiationField`, which does NOT have one — destroying through a base
    /// pointer there is undefined behaviour and ownership has to be `shared_ptr` everywhere as a
    /// result. There is no reason to inherit that problem here.
    virtual ~FieldGeometry() = default;

    virtual FieldGeometry::CoordinateSystem get_coordinate_system() const noexcept = 0;
};

/// A Cartesian voxel grid over a metric box.
///
/// The rule inherited from the training runtime: THE BOX IS METRIC AND THE VOXEL SIZE FOLLOWS FROM
/// THE RESOLUTION, never the other way round. Constructing from a voxel size invites a grid that
/// does not tile the box. `make` and `cubic` are the only constructors and both validate.
struct CartesianFieldGeometry : public FieldGeometry {
    std::array<std::uint32_t, 3> voxel_counts{};
    std::array<float, 3> field_dimensions_m{};

    /// Throws `InvalidArgument` on a zero count or a non-positive / non-finite edge. A NaN edge
    /// would otherwise silently poison every voxel coordinate derived from it.
    static CartesianFieldGeometry make(std::array<std::uint32_t, 3> voxel_counts,
                              std::array<float, 3> field_dimensions_m);
    /// A cubic box, the common case for a trained field.
    static CartesianFieldGeometry cubic(std::uint32_t resolution, float field_box_m);

    FieldGeometry::CoordinateSystem get_coordinate_system() const noexcept override final {
        return CoordinateSystem::Cartesian;
    }

    std::array<float, 3> get_voxel_dimensions_m() const noexcept;
    std::uint64_t get_voxel_count() const noexcept;

    glm::vec3 get_field_dimensions() const noexcept;
    glm::vec3 get_voxel_dimensions() const noexcept;
    glm::uvec3 get_voxel_counts() const noexcept;

    /// Compared field by field rather than `= default`, and the base deliberately gets no
    /// `operator==` of its own. `FieldGeometry` declares no members, so an equality on it would
    /// report two geometries of DIFFERENT coordinate systems as equal whenever they are compared
    /// through a base reference — a wrong answer reachable silently. Each concrete geometry
    /// compares what it actually has.
    bool operator==(const CartesianFieldGeometry& other) const noexcept {
        return voxel_counts == other.voxel_counts && field_dimensions_m == other.field_dimensions_m;
    }

private:
    /// PRIVATE, so `make` and `cubic` really are the only ways in and the validation above cannot be
    /// stepped around. A public one let `{0, 0, 0}` counts through, and `get_voxel_dimensions_m()`
    /// then divides by zero.
    CartesianFieldGeometry(const std::array<std::uint32_t, 3>& voxel_counts,
                           const std::array<float, 3>& field_dimensions_m)
        : voxel_counts(voxel_counts), field_dimensions_m(field_dimensions_m) {}
};

}  // namespace RadFiled3D::nn