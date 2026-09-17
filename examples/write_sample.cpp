// Write a sample `.rf3m` so `rf3m` has something to inspect.
//
// It is the shape a TPBRFNet export produces — a per-voxel implicit field over beam parameters and
// a patient translation — plus one output whose semantic this library does not know, to show that
// such a model needs no change here.
#include <RadFiled3D/nn/deploy.hpp>

#include <cstdio>
#include <numbers>

using namespace RadFiled3D::nn::deploy;
namespace rfnn = RadFiled3D::nn;

int main() {
    constexpr double pi = std::numbers::pi;
    const bytes onnx_trunk = {'<', 'o', 'n', 'n', 'x', ' ', 't', 'r', 'u', 'n', 'k', '>'};
    const bytes onnx_encoder = {'<', 'o', 'n', 'n', 'x', ' ', 'e', 'n', 'c', 'o', 'd', 'e', 'r', '>'};
    const bytes ptx = {'<', 'p', 't', 'x', '>'};

    PackageBuilder b;
    b.provenance("xray-scatter-v3", "radfield3d-nn 2.1", "G4EmStandardPhysics_option4")
        .created("2026-08-31T08:49:10Z")
        //.field_geometry(rfnn::CartesianFieldGeometry::cubic(50, 1.f))
        .field_dimensions_m({1.f, 1.f, 1.f})
        .input("position", Semantic::Position, {3}).unit("m").normalizer(Linear01{0.0, 1.0}).done()
        .input("beam_direction", Semantic::BeamDirection, {2}).unit("rad")
            .range(MinMax{-pi, pi}).normalizer(LinearSym{-pi, pi}).done()
        .input("source_distance", Semantic::SourceDistance, {1}).unit("m").range(MinMax{0.5, 3.0}).done()
        .input("patient_translation", Semantic::PatientTranslation, {3}).unit("m").range(MinMax{-0.5, 0.5}).done()
        .output("flux", Semantic::Flux, {1}).normalizer(LogScale{1e-12, 30.0}).done()
        .output("spectrum", Semantic::Spectrum, {128}).unit("eV").done()
        .output("direction_distribution", Semantic::from_name("direction_distribution"), {16, 32}).unit("sr^-1").done()
        .graph("trunk", onnx_trunk)
        .graph("beam_encoder", onnx_encoder)
        // A specialised form of the same trunk. The loader prefers it on a matching device and
        // falls back to the ONNX graph everywhere else.
        .block(BlockKind::CudaPtx, "trunk", "", ptx)
        .metric("air_kerma_smape", 0.0431)
        .metric("gamma_pass_rate", 0.981);

    try {
        const Package package = b.build();
        package.write_file("sample.rf3m");
        std::printf("wrote sample.rf3m (%zu bytes)\n", package.to_bytes().size());
        return 0;
    } catch (const std::exception& err) {
        std::fprintf(stderr, "write_sample: %s\n", err.what());
        return 1;
    }
}
