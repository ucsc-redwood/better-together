#include "conf.hpp"

#include <nlohmann/json.hpp>
#include <stdexcept>

#include "device_specs_embedded.hpp"

// The device registry is now data-driven: it parses the per-device specs under
// devices/*.json (embedded at build time into device_specs_embedded.hpp via
// scripts/embed_device_specs.py, so there is no runtime file dependency). This
// replaces the old hand-maintained C++ table and is the single source of truth
// for core topology, pinnability, and GPU subgroup size. To add/change a device,
// edit devices/<id>.json (validate with scripts/validate_devices.py); the build
// regenerates the embedded header automatically (CMake bt_device_specs target).

namespace {
ProcessorType parse_core_type(const std::string& s) {
  if (s == "little") return ProcessorType::kLittleCore;
  if (s == "medium") return ProcessorType::kMediumCore;
  if (s == "big") return ProcessorType::kBigCore;
  if (s == "super") return ProcessorType::kSuperCore;
  throw std::runtime_error("device spec: unknown core type '" + s + "'");
}
}  // namespace

DeviceRegistry::DeviceRegistry() {
  for (const auto spec : bt::device_specs::kEmbedded) {
    const auto j = nlohmann::json::parse(std::string(spec));

    const std::string id = j.at("id").get<std::string>();

    std::vector<Core> cores;
    cores.reserve(j.at("cores").size());
    for (const auto& c : j.at("cores")) {
      cores.push_back(Core{c.at("id").get<int>(),
                           parse_core_type(c.at("type").get<std::string>()),
                           c.at("pinnable").get<bool>()});
    }

    int subgroup = 0;
    if (j.contains("gpu") && j["gpu"].contains("subgroup_size")) {
      subgroup = j["gpu"]["subgroup_size"].get<int>();
    }

    devices_.emplace(id, Device(id, std::move(cores), subgroup));
  }
}
