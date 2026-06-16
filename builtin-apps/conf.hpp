#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// 1. Define an enum for core types.
enum class ProcessorType {
  kLittleCore = 0,
  kMediumCore = 1,
  kBigCore = 2,
  kVulkan = 3,
  kCuda = 4,
  kSuperCore = 5,
};

inline std::string CoreTypeName(const ProcessorType core_type) {
  switch (core_type) {
    case ProcessorType::kLittleCore:
      return "little";
    case ProcessorType::kMediumCore:
      return "medium";
    case ProcessorType::kBigCore:
      return "big";
    case ProcessorType::kSuperCore:
      return "super";
    default:
      return "unknown";
  }
}

// 2. Define a struct for a Core.
struct Core {
  int id;              // The OS/core id.
  ProcessorType type;  // Type of the core (LITTLE, MEDIUM, BIG).
  bool pinnable;       // Whether this core is available for pinning.
};

// 3. Create a Device class that holds a list of cores.
class Device {
 public:
  // Construct a device with a name, a list of cores, and (optionally) the GPU
  // subgroup/warp size (0 = unknown / no GPU spec).
  Device(std::string name, std::vector<Core> cores, int gpu_subgroup_size = 0)
      : name_(std::move(name)),
        cores_(std::move(cores)),
        gpu_subgroup_size_(gpu_subgroup_size) {}

  // GPU subgroup (warp) size from the device spec; 0 if unspecified.
  [[nodiscard]] int gpu_subgroup_size() const { return gpu_subgroup_size_; }

  // Get all cores.
  const std::vector<Core>& getCores() const { return cores_; }

  // Get all cores of a specific type.
  std::vector<Core> getCores(ProcessorType type) const {
    std::vector<Core> result;
    for (const auto& core : cores_) {
      if (core.type == type) {
        result.push_back(core);
      }
    }
    return result;
  }

  // Get pinnable cores of a specific type. NOTE: no default -- a missing arg used
  // to silently mean "little cores only", which is easily mistaken for
  // getAllPinnableCores() (all types). Callers must name the tier explicitly.
  std::vector<Core> getPinnableCores(ProcessorType type) const {
    std::vector<Core> result;
    for (const auto& core : cores_) {
      if (core.pinnable && core.type == type) result.push_back(core);
    }
    return result;
  }

  // Get all pinnable cores regardless of type.
  std::vector<Core> getAllPinnableCores() const {
    std::vector<Core> result;
    for (const auto& core : cores_) {
      if (core.pinnable) result.push_back(core);
    }
    return result;
  }

 private:
  std::string name_;
  std::vector<Core> cores_;
  int gpu_subgroup_size_ = 0;
};

// 4. Create a device registry for easy lookup.
class DeviceRegistry {
 public:
  DeviceRegistry();

  // Retrieve a device configuration by its id.
  const Device& getDevice(const std::string& deviceId) const {
    auto it = devices_.find(deviceId);
    if (it != devices_.end()) return it->second;
    throw std::runtime_error("Device not found: " + deviceId);
  }

 private:
  std::unordered_map<std::string, Device> devices_;
};

inline DeviceRegistry& GlobalDeviceRegistry() {
  static DeviceRegistry instance;
  return instance;
}
