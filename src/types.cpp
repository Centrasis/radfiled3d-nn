// The geometry vocabulary, compiled into `rfnn::deploy`.
//
// It lives HERE rather than in `src/core/` because the container needs it: `PackageBuilder::
// field_geometry` writes a geometry into a package and `Geometry::get_field_geometry` reads one
// back, so `rfnn_deploy` must resolve these symbols on its own. With the definitions in
// `src/core/field.cpp` it could not, and the link was closed by making `rfnn_deploy` depend on
// `rfnn` — which, since `rfnn` already depends on `rfnn_deploy`, is a cycle.
//
// Nothing in this file touches RadFiled3D or ONNX Runtime: it is arithmetic over `std::array` and
// glm, and the one exception it throws is already in this target. That is what lets the dependency
// run one way, `rfnn` -> `rfnn_deploy`, and keeps the container readable without building either.
// Same arrangement as `src/exception.cpp`, for the same reason.
#include <RadFiled3D/nn/types.hpp>

#include <RadFiled3D/nn/exception.hpp>

#include <cmath>

namespace RadFiled3D::nn {

CartesianFieldGeometry CartesianFieldGeometry::make(std::array<std::uint32_t, 3> voxel_counts,
                                                    std::array<float, 3> field_dimensions_m) {
    for (const auto c : voxel_counts)
        if (c == 0) throw Exception::invalid_argument("voxel counts must all be positive");
    // `<= 0` alone would let a NaN through.
    for (const float d : field_dimensions_m)
        if (!std::isfinite(d) || d <= 0.f)
            throw Exception::invalid_argument("field dimensions must all be positive");
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

glm::uvec3 CartesianFieldGeometry::get_voxel_counts() const noexcept {
    return {voxel_counts[0], voxel_counts[1], voxel_counts[2]};
}

}  // namespace RadFiled3D::nn
