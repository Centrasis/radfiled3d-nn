// `rf3m` — inspect a deployment package.
//
// Deliberately trivial, and deliberately useful: it links only `RadFiled3D::nn::deploy`, so it runs on a
// machine with no GPU, no CUDA, no ONNX Runtime and no RadFiled3D. That it CAN is the practical
// proof that extracting the format from the training framework was worth doing.
#include <RadFiled3D/nn/deploy.hpp>
#include <RadFiled3D/nn/deploy/version.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>

using namespace RadFiled3D::nn::deploy;

namespace {

const char* or_dash(const std::string& s) { return s.empty() ? "-" : s.c_str(); }

std::string shape_of(const std::vector<std::uint32_t>& shape) {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) out += (i ? ", " : "") + std::to_string(shape[i]);
    return out + "]";
}

void print(const std::string& path, const Metadata& md) {
    if (!path.empty()) std::printf("%s\n", path.c_str());
    std::printf("  dataset   %s\n", or_dash(md.provenance.dataset));
    std::printf("  software  %s\n", or_dash(md.provenance.software));
    std::printf("  physics   %s\n", or_dash(md.provenance.physics));
    std::printf("  created   %s\n", or_dash(md.provenance.created));
    const auto& box = md.geometry.field_dimensions_m;
    std::printf("  field box [%g, %g, %g] m\n", box[0], box[1], box[2]);
    if (md.geometry.voxelization) {
        const auto& c = md.geometry.voxelization->voxel_counts;
        std::printf("  voxelized [%u, %u, %u] voxels\n", c[0], c[1], c[2]);
    }

    for (const Role role : {Role::Input, Role::Output}) {
        bool any = false;
        for (const auto& d : md.io) any |= d.role == role;
        if (!any) continue;
        std::printf("  %ss\n", std::string(to_string(role)).c_str());
        for (const auto& d : md.io) {
            if (d.role != role) continue;
            const std::string unit = d.unit.empty() ? "" : " [" + d.unit + "]";
            const char* unknown = d.semantic.is_known() ? "" : "  (semantic unknown to this build)";
            std::printf("    %-24s %s%-8s %s / %s%s\n", d.name.c_str(), shape_of(d.shape).c_str(), unit.c_str(),
                        std::string(d.semantic.get_name()).c_str(), std::string(get_name(d.normalizer)).c_str(), unknown);
        }
    }

    if (!md.blocks.empty()) {
        // The scan: kind, name and size for every block, read without touching one payload byte.
        std::printf("  blocks\n");
        for (const auto& b : md.blocks) {
            const std::string arch = b.target_arch.empty() ? "any" : b.target_arch;
            const char* unknown = b.kind.is_known() ? "" : "  (kind unknown to this build)";
            std::printf("    %-12s %-16s arch %-8s %10llu bytes%s\n", std::string(b.kind.get_name()).c_str(),
                        b.name.c_str(), arch.c_str(), static_cast<unsigned long long>(b.payload_bytes), unknown);
        }
    }
    if (md.composition) {
        // The one payload a scan reads, because a listing that showed a package's code without
        // showing how it is driven would answer the less useful half of the question.
        std::printf("  composition\n");
        for (const auto& b : md.composition->buffers)
            std::printf("    buffer  %-16s %llu x %s per row\n", b.name.c_str(),
                        static_cast<unsigned long long>(b.elements),
                        std::string(to_string(b.dtype)).c_str());
        for (const auto& s : md.composition->stages) {
            std::printf("    stage   %-16s %s\n", s.name.c_str(), std::string(to_string(s.invocation)).c_str());
            for (const auto& r : s.reads)
                std::printf("      reads   %-18s <- %s\n", r.tensor.c_str(), r.buffer.c_str());
            for (const auto& w : s.writes)
                std::printf("      writes  %-18s -> %s\n", w.tensor.c_str(), w.buffer.c_str());
            if (s.weights.elements > 0)
                std::printf("      weights %llu x %s%s\n", static_cast<unsigned long long>(s.weights.elements),
                            std::string(to_string(s.weights.dtype)).c_str(),
                            s.weights.onnx_external_file.empty()
                                ? ""
                                : (" (onnx external `" + s.weights.onnx_external_file + "`)").c_str());
            // Which blocks implement this stage, and therefore which hardware can run it. A stage
            // with no `onnx` among them is the one that makes a package refuse to load elsewhere.
            std::string forms;
            bool portable = false;
            for (const auto& b : md.blocks) {
                if (b.name != s.name || !b.kind.is_executable()) continue;
                portable = portable || b.kind == BlockKind::Onnx;
                forms += (forms.empty() ? "" : ", ") + std::string(b.kind.get_name()) +
                         (b.target_arch.empty() ? "" : "(" + b.target_arch + ")");
            }
            std::printf("      runs as %s%s\n", forms.empty() ? "nothing in this package" : forms.c_str(),
                        portable ? "" : "   <- no portable form: this stage needs matching hardware");
            for (const auto& [ref, launch] : s.launches)
                std::printf("      launch  %-10s %s  block %ux%ux%u\n", ref.kind.c_str(),
                            launch.entry_point.c_str(), launch.block_size[0], launch.block_size[1],
                            launch.block_size[2]);
        }
    }
    if (md.rf3_metadata) {
        const auto& rf3 = *md.rf3_metadata;
        std::printf("  trained on\n");
        if (!rf3.simulation.physics_list.empty()) std::printf("    physics   %s\n", rf3.simulation.physics_list.c_str());
        if (!rf3.simulation.geometry.empty()) std::printf("    geometry  %s\n", rf3.simulation.geometry.c_str());
        if (rf3.simulation.tube.max_energy_ev > 0.f) std::printf("    tube      %g eV max\n", rf3.simulation.tube.max_energy_ev);
    }
    if (!md.metrics.empty()) {
        std::printf("  metrics\n");
        for (const auto& [k, v] : md.metrics) std::printf("    %-24s %g\n", k.c_str(), v);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string usage = "usage: rf3m <package.rf3m>\n       rf3m convert <in.rf3m> <out.rf3m>\n"
                            "       rf3m extract <package.rf3m> <graph> <out.onnx>\n";

    // `convert` reads whatever version the input is and writes the current one. One version exists,
    // so today it is a rewrite — it earns its place as the migration path the moment a second one
    // does, because only the byte walk is versioned and the `Package` value in between is shared.
    if (argc == 4 && std::string(argv[1]) == "convert") {
        const std::string in = argv[2], out = argv[3];
        try {
            const bytes raw = read_all(in);
            const FormatVersion from = peek_version(raw);
            const Package package = Package::read(raw);
            package.write_file(out);
            std::printf("%s (%s) -> %s (%s), %zu -> %zu bytes\n", in.c_str(),
                        std::string(to_string(from)).c_str(), out.c_str(),
                        std::string(to_string(kWriteVersion)).c_str(), raw.size(),
                        package.to_bytes().size());
            return 0;
        } catch (const std::exception& err) {
            std::fprintf(stderr, "rf3m: converting %s: %s\n", in.c_str(), err.what());
            return 1;
        }
    }

    // `extract` writes one block's payload out as-is, which is what makes a converted package
    // checkable against its source with cmp(1) rather than on trust.
    if (argc == 5 && std::string(argv[1]) == "extract") {
        try {
            const Package package = Package::read_file(argv[2]);
            const std::optional<byte_view> graph = package.get_graph(argv[3]);
            if (!graph) {
                std::fprintf(stderr, "rf3m: %s has no graph named `%s`\n", argv[2], argv[3]);
                return 1;
            }
            std::ofstream out(argv[4], std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(graph->data()), static_cast<std::streamsize>(graph->size()));
            if (!out) {
                std::fprintf(stderr, "rf3m: cannot write %s\n", argv[4]);
                return 1;
            }
            std::printf("%s: wrote %zu bytes of graph `%s`\n", argv[4], graph->size(), argv[3]);
            return 0;
        } catch (const std::exception& err) {
            std::fprintf(stderr, "rf3m: %s\n", err.what());
            return 1;
        }
    }

    if (argc != 2) {
        std::fprintf(stderr, "%s", usage.c_str());
        return 1;
    }
    const std::string path = argv[1];
    try {
        const bytes raw = read_all(path);
        std::printf("%s\n  format    %s\n", path.c_str(), std::string(to_string(peek_version(raw))).c_str());
        print("", Package::read_metadata(raw));
        return 0;
    } catch (const std::exception& err) {
        std::fprintf(stderr, "rf3m: %s: %s\n", path.c_str(), err.what());
        return 1;
    }
}
