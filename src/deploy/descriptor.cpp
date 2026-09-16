#include <RadFiled3D/nn/deploy/descriptor.hpp>

#include <cmath>

namespace RadFiled3D::nn::deploy {

std::string_view to_string(Role role) noexcept {
    return role == Role::Input ? "input" : "output";
}

// ── Semantic ──────────────────────────────────────────────────────────────────────────────────────

namespace {
constexpr std::pair<Semantic::Kind, std::string_view> kSemanticNames[] = {
    {Semantic::Position, "position"},
    {Semantic::QueryDirection, "query_direction"},
    {Semantic::TubeSpectrum, "tube_spectrum"},
    {Semantic::BeamDirection, "beam_direction"},
    {Semantic::SourceDistance, "source_distance"},
    {Semantic::SourceOrigin, "source_origin"},
    {Semantic::BeamCollimation, "beam_collimation"},
    {Semantic::PatientTranslation, "patient_translation"},
    {Semantic::PatientRotation, "patient_rotation"},
    {Semantic::GeometryMap, "geometry_map"},
    {Semantic::AnodeAngle, "anode_angle"},
    {Semantic::Flux, "flux"},
    {Semantic::Spectrum, "spectrum"},
    {Semantic::AngularFlux, "angular_flux"},
    {Semantic::AirKerma, "air_kerma"},
    {Semantic::Uncertainty, "uncertainty"},
};
}  // namespace

Semantic Semantic::from_name(std::string_view name) {
    for (const auto& [kind, known] : kSemanticNames)
        if (known == name) return Semantic(kind);
    Semantic s;
    s.custom_name = std::string(name);
    return s;
}

std::string_view Semantic::get_name() const noexcept {
    for (const auto& [kind, known] : kSemanticNames)
        if (kind == this->kind) return known;
    return custom_name;
}

void Semantic::encode(Writer& w) const { w.string(get_name()); }
Semantic Semantic::decode(Reader& r) { return from_name(r.string("tensor semantic")); }

// ── DType ─────────────────────────────────────────────────────────────────────────────────────────

std::size_t size_bytes(DType dtype) noexcept {
    switch (dtype) {
        case DType::F32:
        case DType::I32: return 4;
        case DType::F16: return 2;
        case DType::U8: return 1;
    }
    return 0;
}

std::string_view to_string(DType dtype) noexcept {
    switch (dtype) {
        case DType::F32: return "f32";
        case DType::F16: return "f16";
        case DType::I32: return "i32";
        case DType::U8: return "u8";
    }
    return "?";
}

void encode(Writer& w, DType dtype) { w.u8(static_cast<std::uint8_t>(dtype)); }

DType decode_dtype(Reader& r, const char* what) {
    const std::uint8_t dtype = r.u8(what);
    // Closed set, unlike `Semantic`: a dtype the runtime cannot bind is a hard failure, because
    // there is no way to carry an unknown element type through a buffer of a known size.
    if (dtype > static_cast<std::uint8_t>(DType::U8)) throw Exception::invalid_value(what, dtype);
    return static_cast<DType>(dtype);
}

// ── Range ─────────────────────────────────────────────────────────────────────────────────────────

bool RangeMap::operator==(const RangeMap& other) const { return children == other.children; }

std::optional<bool> contains(const Range& range, double value) noexcept {
    if (const auto* mm = std::get_if<MinMax>(&range)) return value >= mm->min && value <= mm->max;
    if (const auto* h = std::get_if<Histogram>(&range)) return value >= h->min && value <= h->max;
    return std::nullopt;
}

namespace {
std::uint8_t range_kind(const Range& range) {
    return std::visit(
        [](const auto& r) -> std::uint8_t {
            using T = std::decay_t<decltype(r)>;
            if constexpr (std::is_same_v<T, NoRange>) return 0;
            else if constexpr (std::is_same_v<T, MinMax>) return 1;
            else if constexpr (std::is_same_v<T, Histogram>) return 2;
            else if constexpr (std::is_same_v<T, Categorical>) return 3;
            else if constexpr (std::is_same_v<T, RangeMap>) return 4;
            else return r.kind;
        },
        range);
}
}  // namespace

void encode(Writer& w, const Range& range) {
    w.u8(range_kind(range));
    w.sized([&](Writer& w) {
        std::visit(
            [&](const auto& r) {
                using T = std::decay_t<decltype(r)>;
                if constexpr (std::is_same_v<T, NoRange>) {
                } else if constexpr (std::is_same_v<T, MinMax>) {
                    w.f64(r.min);
                    w.f64(r.max);
                } else if constexpr (std::is_same_v<T, Histogram>) {
                    w.f64(r.min);
                    w.f64(r.max);
                    w.f64(r.bin_width);
                } else if constexpr (std::is_same_v<T, Categorical>) {
                    w.list(r.labels, [](Writer& w, const std::string& l) { w.string(l); });
                } else if constexpr (std::is_same_v<T, RangeMap>) {
                    w.list(r.children, [](Writer& w, const auto& child) {
                        w.string(child.first);
                        encode(w, child.second);
                    });
                } else {
                    w.raw(r.payload);
                }
            },
            range);
    });
}

Range decode_range(Reader& r) {
    const std::uint8_t kind = r.u8("range kind");
    return r.sized("range payload", [kind](Reader& r) -> Range {
        switch (kind) {
            case 0: return NoRange{};
            case 1: {
                MinMax mm;
                mm.min = r.f64("range min");
                mm.max = r.f64("range max");
                return mm;
            }
            case 2: {
                Histogram h;
                h.min = r.f64("range min");
                h.max = r.f64("range max");
                h.bin_width = r.f64("range bin width");
                return h;
            }
            case 3:
                return Categorical{r.list("range labels", [](Reader& r) { return r.string("range label"); })};
            case 4: {
                RangeMap map;
                map.children = r.list("range children", [](Reader& r) {
                    std::string name = r.string("range child name");
                    return std::pair<std::string, Range>(std::move(name), decode_range(r));
                });
                return map;
            }
            default: {
                const byte_view rest = r.rest();
                return UnknownRange{kind, bytes(rest.begin(), rest.end())};
            }
        }
    });
}

// ── Normalizer ────────────────────────────────────────────────────────────────────────────────────

std::string_view get_name(const Normalizer& n) noexcept {
    return std::visit(
        [](const auto& v) -> std::string_view {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, Identity>) return "identity";
            else if constexpr (std::is_same_v<T, Linear01>) return "linear0_1";
            else if constexpr (std::is_same_v<T, LinearSym>) return "linear-1_1";
            else if constexpr (std::is_same_v<T, LogScale>) return "log_scale";
            else if constexpr (std::is_same_v<T, Asinh>) return "asinh";
            else return v.name;
        },
        n);
}

std::vector<double> get_params(const Normalizer& n) {
    return std::visit(
        [](const auto& v) -> std::vector<double> {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, Identity>) return {};
            else if constexpr (std::is_same_v<T, Linear01> || std::is_same_v<T, LinearSym>) return {v.min, v.max};
            else if constexpr (std::is_same_v<T, LogScale>) return {v.epsilon, v.scale};
            else if constexpr (std::is_same_v<T, Asinh>) return {v.scale};
            else return v.params;
        },
        n);
}

namespace {

/// A transform's result, or nothing when it is not a number a caller may act on.
///
/// `log` of a negative value is NaN and `exp` of a large one is infinity; both are the "plausible
/// number in place of an answer" the design forbids (rule 9), and neither is the arithmetic's
/// fault — they mark a value outside the transform's domain. Every branch below funnels through
/// here so that judgement is made once.
std::optional<double> finite(double result) noexcept {
    return std::isfinite(result) ? std::optional<double>(result) : std::nullopt;
}

}  // namespace

std::optional<double> apply(const Normalizer& n, double value) noexcept {
    return std::visit(
        [value](const auto& v) -> std::optional<double> {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, Identity>) return finite(value);
            else if constexpr (std::is_same_v<T, Linear01>) {
                if (!(v.max > v.min)) return std::nullopt;
                return finite((value - v.min) / (v.max - v.min));
            } else if constexpr (std::is_same_v<T, LinearSym>) {
                if (!(v.max > v.min)) return std::nullopt;
                return finite(2.0 * (value - v.min) / (v.max - v.min) - 1.0);
            } else if constexpr (std::is_same_v<T, LogScale>) {
                if (v.scale == 0.0) return std::nullopt;
                return finite(std::log(value + v.epsilon) / v.scale);
            } else if constexpr (std::is_same_v<T, Asinh>) {
                if (v.scale == 0.0) return std::nullopt;
                return finite(std::asinh(value / v.scale));
            } else {
                return std::nullopt;
            }
        },
        n);
}

std::optional<double> invert(const Normalizer& n, double value) noexcept {
    return std::visit(
        [value](const auto& v) -> std::optional<double> {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, Identity>) return finite(value);
            else if constexpr (std::is_same_v<T, Linear01>) {
                // The same degeneracy check as forward, and for the same reason: a zero-width range
                // has no inverse either, it just fails as a multiplication by zero instead of a
                // division, which would return a plausible `min` for every input.
                if (!(v.max > v.min)) return std::nullopt;
                return finite(value * (v.max - v.min) + v.min);
            } else if constexpr (std::is_same_v<T, LinearSym>) {
                if (!(v.max > v.min)) return std::nullopt;
                return finite((value + 1.0) * 0.5 * (v.max - v.min) + v.min);
            } else if constexpr (std::is_same_v<T, LogScale>) {
                if (v.scale == 0.0) return std::nullopt;
                return finite(std::exp(value * v.scale) - v.epsilon);
            } else if constexpr (std::is_same_v<T, Asinh>) {
                if (v.scale == 0.0) return std::nullopt;
                return finite(std::sinh(value) * v.scale);
            } else {
                return std::nullopt;
            }
        },
        n);
}

bool is_invertible(const Normalizer& n) noexcept {
    // Every implemented normalizer has a closed-form inverse; only a name this build does not know
    // has none. Degenerate PARAMETERS are a per-value refusal from `invert`, not a property of the
    // kind, so they are deliberately not consulted here.
    return !std::holds_alternative<CustomNormalizer>(n);
}

Normalizer normalizer_from_wire(std::string name, std::vector<double> params) {
    if (name == "identity") return Identity{};
    if (name == "linear0_1" && params.size() == 2) return Linear01{params[0], params[1]};
    if (name == "linear-1_1" && params.size() == 2) return LinearSym{params[0], params[1]};
    if (name == "log_scale" && params.size() == 2) return LogScale{params[0], params[1]};
    if (name == "asinh" && params.size() == 1) return Asinh{params[0]};
    return CustomNormalizer{std::move(name), std::move(params)};
}

void encode(Writer& w, const Normalizer& n) {
    w.string(get_name(n));
    w.list(get_params(n), [](Writer& w, double p) { w.f64(p); });
}

Normalizer decode_normalizer(Reader& r) {
    std::string name = r.string("normalizer name");
    auto params = r.list("normalizer params", [](Reader& r) { return r.f64("normalizer param"); });
    return normalizer_from_wire(std::move(name), std::move(params));
}

// ── TensorDescriptor ──────────────────────────────────────────────────────────────────────────────

std::uint64_t TensorDescriptor::get_elements() const noexcept {
    std::uint64_t n = 1;
    for (const auto d : shape) n *= d;
    return n;
}

void TensorDescriptor::encode(Writer& w) const {
    w.string(name);
    w.u8(static_cast<std::uint8_t>(role));
    semantic.encode(w);
    w.list(shape, [](Writer& w, std::uint32_t d) { w.u32(d); });
    deploy::encode(w, dtype);
    w.string(unit);
    deploy::encode(w, range);
    deploy::encode(w, normalizer);
    w.boolean(optional);
}

TensorDescriptor TensorDescriptor::decode(Reader& r) {
    TensorDescriptor d;
    d.name = r.string("tensor name");
    const std::uint8_t role = r.u8("tensor role");
    if (role > 1) throw Exception::invalid_value("tensor role", role);
    d.role = static_cast<Role>(role);
    d.semantic = Semantic::decode(r);
    d.shape = r.list("tensor shape", [](Reader& r) { return r.u32("tensor dim"); });
    d.dtype = decode_dtype(r, "tensor dtype");
    d.unit = r.string("tensor unit");
    d.range = decode_range(r);
    d.normalizer = decode_normalizer(r);
    d.optional = r.boolean("tensor optional");
    return d;
}

}  // namespace RadFiled3D::nn::deploy
