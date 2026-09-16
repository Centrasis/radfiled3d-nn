// What a model consumes and produces.
//
// This is the answer to defect D3 in requirements.md. The original describes a model's interface
// with a bitflag enum plus a hardcoded "known bits" mask that *rejects* anything it does not
// recognise, and stores each output's resolution as a named field of a fixed struct
// (`spectrum_bins`, `angular_phi_segments`, ...). Adding one new output — a direction distribution,
// say — therefore costs a new bit, a new mask value, new struct fields, a new validation clause and
// a release, and every older consumer refuses the file.
//
// Here a model declares an ordered list of `TensorDescriptor`s instead. One design rule makes it
// extensible and is applied consistently:
//
//     Names on the wire, types in the API, unknown names preserved.
//
// A `Semantic` or a `Normalizer` this build has never heard of decodes into its `Custom` form,
// survives a round trip byte-for-byte, and can still be reported to a user. Resolution lives in the
// descriptor's own `shape`, so a quantity and its size can no longer disagree.
#pragma once

#include <RadFiled3D/nn/deploy/codec.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace RadFiled3D::nn::deploy {

/// Which side of the model a tensor is on.
enum class Role : std::uint8_t { Input = 0, Output = 1 };

std::string_view to_string(Role role) noexcept;

/// The physical quantity a tensor carries.
///
/// The known set covers what the training framework produces today. Anything else is `Custom` — a
/// model emitting a direction distribution needs no change to this library, which is the whole
/// point of the redesign. The wire name is the identity; the enum is a convenience over it.
struct Semantic {
    enum Kind : std::uint8_t {
        // inputs evaluated per query point
        Position,
        QueryDirection,  // for NIRC-like models where the query is per direction
        // inputs describing the beam / the scene, constant over a field
        TubeSpectrum,
        BeamDirection,
        SourceDistance,
        SourceOrigin,
        BeamCollimation,
        PatientTranslation,
        PatientRotation,
        GeometryMap,
        AnodeAngle,
        // outputs
        Flux,
        Spectrum,
        AngularFlux,
        AirKerma,
        Uncertainty,
        /// Any quantity this build does not know. Round-trips verbatim.
        Custom,
    };

    Kind kind = Custom;
    std::string custom_name;  // only meaningful when kind == Custom

    Semantic() = default;
    Semantic(Kind k) : kind(k) {}  // NOLINT(google-explicit-constructor): `Semantic::Flux` reads as one

    /// Resolve a wire name; an unknown name becomes `Custom`.
    static Semantic from_name(std::string_view name);

    std::string_view get_name() const noexcept;

    /// Whether this build recognises the quantity. A consumer can use it to warn ("this package
    /// declares an output I do not understand") instead of failing.
    bool is_known() const noexcept { return kind != Custom; }

    bool operator==(const Semantic&) const = default;

    void encode(Writer& w) const;
    static Semantic decode(Reader& r);
};

/// Element type of a tensor. Unlike semantics this set is closed: a dtype the runtime cannot bind
/// is a hard failure, not something to report and skip.
enum class DType : std::uint8_t { F32 = 0, F16 = 1, I32 = 2, U8 = 3 };

std::size_t size_bytes(DType dtype) noexcept;
std::string_view to_string(DType dtype) noexcept;

/// The dtype's wire form, declared once (rule 2) because more than one type carries one: a tensor
/// descriptor and a composition buffer both do, and two hand-written walks over the same byte would
/// be two chances to disagree about what a `3` means.
void encode(Writer& w, DType dtype);
DType decode_dtype(Reader& r, const char* what);

// ── ranges ────────────────────────────────────────────────────────────────────────────────────────
//
// The interval of values a model was trained on — its validity domain. A query outside it is out of
// distribution and the runtime must be able to say so rather than quietly extrapolating.
//
// Every variant is written inside a length-prefixed region, so a reader that meets a kind it does
// not know skips the payload and keeps the entry as `UnknownRange` instead of failing.

struct NoRange {
    bool operator==(const NoRange&) const = default;
};
struct MinMax {
    double min = 0, max = 0;
    bool operator==(const MinMax&) const = default;
};
/// A histogram axis: `bins = round((max - min) / bin_width)`. Used for the tube spectrum.
struct Histogram {
    double min = 0, max = 0, bin_width = 0;
    bool operator==(const Histogram&) const = default;
};
struct Categorical {
    std::vector<std::string> labels;
    bool operator==(const Categorical&) const = default;
};
struct RangeMap;
/// A kind written by a newer producer. Preserved verbatim so a rewrite loses nothing.
struct UnknownRange {
    std::uint8_t kind = 0;
    bytes payload;
    bool operator==(const UnknownRange&) const = default;
};

using Range = std::variant<NoRange, MinMax, Histogram, Categorical, RangeMap, UnknownRange>;

/// A named group, for a parameter that decomposes (collimation → width, height).
struct RangeMap {
    std::vector<std::pair<std::string, Range>> children;
    bool operator==(const RangeMap&) const;
};

/// Whether `value` lies inside the recorded range. `NoRange` and unknown kinds cannot judge, so
/// they answer `nullopt` rather than pretending.
std::optional<bool> contains(const Range& range, double value) noexcept;

void encode(Writer& w, const Range& range);
Range decode_range(Reader& r);

// ── normalizers ───────────────────────────────────────────────────────────────────────────────────
//
// How a metric value becomes what the graph expects. A caller of a loaded model passes metres,
// degrees and electronvolts; the package says how to map them, and the runtime applies it. That is
// what makes two models with different normalisations interchangeable to their caller (R-F2.2).
//
// One wire form — `[str name][u32 n][f64 params]` — serves every variant, so a normalizer added by
// a newer producer decodes as `CustomNormalizer` and survives a rewrite.

/// Pass the value through untouched.
struct Identity {
    bool operator==(const Identity&) const = default;
};
/// `(v - min) / (max - min)` into `[0, 1]`.
struct Linear01 {
    double min = 0, max = 1;
    bool operator==(const Linear01&) const = default;
};
/// `2 (v - min) / (max - min) - 1` into `[-1, 1]`.
struct LinearSym {
    double min = -1, max = 1;
    bool operator==(const LinearSym&) const = default;
};
/// `log(v + epsilon) / scale`. For quantities spanning decades, like flux.
struct LogScale {
    double epsilon = 0, scale = 1;
    bool operator==(const LogScale&) const = default;
};
/// `asinh(v / scale)`. A signed log that stays finite at zero.
struct Asinh {
    double scale = 1;
    bool operator==(const Asinh&) const = default;
};
struct CustomNormalizer {
    std::string name;
    std::vector<double> params;
    bool operator==(const CustomNormalizer&) const = default;
};

using Normalizer = std::variant<Identity, Linear01, LinearSym, LogScale, Asinh, CustomNormalizer>;

std::string_view get_name(const Normalizer& n) noexcept;
std::vector<double> get_params(const Normalizer& n);

/// Apply the forward transform: metric value → graph value. A degenerate parameterisation, a
/// normalizer this build does not implement, or a result that is not finite answers `nullopt`: the
/// caller must be told, not handed a silently wrong number. A NaN out of `log` of a negative value
/// is such a number, which is why the finiteness check is here and not left to the caller.
std::optional<double> apply(const Normalizer& n, double v) noexcept;

/// The reverse transform: graph value → metric value. What a model's OUTPUT needs, so a field layer
/// labelled `eV` holds electronvolts rather than whatever the network was trained to emit.
///
/// Exactly `apply`'s contract in reverse, including its refusals, and one more that only arises
/// going this way: `LogScale` inverts through `exp`, which overflows to infinity for a large enough
/// graph value. An infinite flux is not a reading, so it answers `nullopt` too.
///
/// `invert(n, *apply(n, v)) == v` to within floating-point tolerance for every implemented
/// normalizer; that round trip is what the pair is tested on.
std::optional<double> invert(const Normalizer& n, double v) noexcept;

/// Whether this build can transform values for `n` in both directions. A `CustomNormalizer` cannot:
/// its name is preserved and reportable (rule 4), but nothing here knows its arithmetic.
bool is_invertible(const Normalizer& n) noexcept;

/// Build from the wire form. A known name with the wrong arity is a producer bug, not a new
/// normalizer; keeping it as `CustomNormalizer` preserves the bytes and lets validation report it.
Normalizer normalizer_from_wire(std::string name, std::vector<double> params);

void encode(Writer& w, const Normalizer& n);
Normalizer decode_normalizer(Reader& r);

// ── the descriptor ────────────────────────────────────────────────────────────────────────────────

/// One tensor of the model's interface.
///
/// `name` is the graph's tensor name and the key a caller binds against; `semantic` is what the
/// numbers mean; `shape` is the per-query shape and therefore the only home of a resolution
/// (spectrum bins are `shape == {bins}`, an angular quantity is `{phi, theta}`).
struct TensorDescriptor {
    std::string name;
    Role role = Role::Input;
    Semantic semantic;
    std::vector<std::uint32_t> shape;
    DType dtype = DType::F32;
    /// Physical unit of the *metric* value: "m", "deg", "rad", "eV", or empty.
    std::string unit;
    Range range = NoRange{};
    Normalizer normalizer = Identity{};
    /// Whether the runtime may leave this tensor unbound.
    bool optional = false;

    /// Elements per query point — the product of the shape.
    std::uint64_t get_elements() const noexcept;
    /// Bytes per query point.
    std::uint64_t get_bytes() const noexcept { return get_elements() * size_bytes(dtype); }

    bool operator==(const TensorDescriptor&) const = default;

    void encode(Writer& w) const;
    static TensorDescriptor decode(Reader& r);
};

}  // namespace RadFiled3D::nn::deploy
