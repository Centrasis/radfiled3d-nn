#include <RadFiled3D/nn/core/rf3_metadata.hpp>

#include <algorithm>
#include <cstring>
#include <sstream>

namespace RadFiled3D::nn {

namespace {
using Header = RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader;
using HeaderBlock = RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeaderBlock;

/// Read a NUL-padded fixed-size char array, stopping at the first NUL. Upstream fills these with
/// `strncpy_s`, which does NOT NUL-terminate when the value exactly fills the array, so the whole
/// array is the fallback terminator.
template <std::size_t N>
std::string c_array(const char (&field)[N]) {
    const auto end = std::find(field, field + N, '\0');
    return std::string(field, end);
}

/// Fill a fixed-size, NUL-padded char array, truncating rather than overflowing.
template <std::size_t N>
void set_c_array(char (&field)[N], const std::string& value) {
    std::memset(field, 0, N);
    std::memcpy(field, value.data(), std::min(value.size(), N));
}

std::array<float, 3> to_array(const glm::vec3& v) { return {v.x, v.y, v.z}; }
glm::vec3 to_vec3(const std::array<float, 3>& a) { return {a[0], a[1], a[2]}; }

deploy::bytes serialize(const RadFiled3D::Storage::V1::RadiationFieldMetadata& metadata) {
    std::ostringstream stream(std::ios::binary);
    metadata.serialize(stream);
    const std::string s = stream.str();
    return deploy::bytes(s.begin(), s.end());
}
}  // namespace

deploy::bytes dynamic_tail(const RadFiled3D::Storage::V1::RadiationFieldMetadata& metadata) {
    deploy::bytes stream = serialize(metadata);
    constexpr std::size_t prefix = sizeof(HeaderBlock) + sizeof(Header);
    // A metadata object with no dynamic layers still writes the prefix and nothing more.
    if (stream.size() <= prefix) return {};
    stream.erase(stream.begin(), stream.begin() + static_cast<std::ptrdiff_t>(prefix));
    return stream;
}

deploy::Rf3Metadata to_package_metadata(const RadFiled3D::Storage::V1::RadiationFieldMetadata& metadata) {
    const Header& h = metadata.get_header();
    deploy::Rf3Metadata out;
    out.store_version = 1;
    out.simulation.primary_particle_count = h.simulation.primary_particle_count;
    out.simulation.geometry = c_array(h.simulation.geometry);
    out.simulation.physics_list = c_array(h.simulation.physics_list);
    out.simulation.tube.radiation_direction = to_array(h.simulation.tube.radiation_direction);
    out.simulation.tube.radiation_origin = to_array(h.simulation.tube.radiation_origin);
    out.simulation.tube.max_energy_ev = h.simulation.tube.max_energy_eV;
    out.simulation.tube.tube_id = c_array(h.simulation.tube.tube_id);
    out.software.name = c_array(h.software.name);
    out.software.version = c_array(h.software.version);
    out.software.repository = c_array(h.software.repository);
    out.software.commit = c_array(h.software.commit);
    out.software.doi = c_array(h.software.doi);
    out.dynamic = dynamic_tail(metadata);
    return out;
}

std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> from_package_metadata(
    const deploy::Rf3Metadata& metadata) {
    Header h{};
    h.simulation.primary_particle_count = metadata.simulation.primary_particle_count;
    set_c_array(h.simulation.geometry, metadata.simulation.geometry);
    set_c_array(h.simulation.physics_list, metadata.simulation.physics_list);
    h.simulation.tube.radiation_direction = to_vec3(metadata.simulation.tube.radiation_direction);
    h.simulation.tube.radiation_origin = to_vec3(metadata.simulation.tube.radiation_origin);
    h.simulation.tube.max_energy_eV = metadata.simulation.tube.max_energy_ev;
    set_c_array(h.simulation.tube.tube_id, metadata.simulation.tube.tube_id);
    set_c_array(h.software.name, metadata.software.name);
    set_c_array(h.software.version, metadata.software.version);
    set_c_array(h.software.repository, metadata.software.repository);
    set_c_array(h.software.commit, metadata.software.commit);
    set_c_array(h.software.doi, metadata.software.doi);

    auto out = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>();
    out->set_header(h);
    if (metadata.dynamic.empty()) return out;

    // The dynamic layers can only be rebuilt by RadFiled3D's own deserialiser, which expects the
    // whole stream: `[block][header][tail]`. It is reassembled here, in memory, in the host's own
    // layout — which is host-specific by nature, and precisely why it never happens in a file.
    HeaderBlock block{};
    block.dynamic_metadata_size = metadata.dynamic.size();
    std::string stream;
    stream.append(reinterpret_cast<const char*>(&block), sizeof(block));
    stream.append(reinterpret_cast<const char*>(&h), sizeof(h));
    stream.append(reinterpret_cast<const char*>(metadata.dynamic.data()), metadata.dynamic.size());
    std::istringstream in(stream, std::ios::binary);
    out->deserialize(in);
    return out;
}

}  // namespace RadFiled3D::nn
