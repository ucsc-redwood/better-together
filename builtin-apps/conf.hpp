#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// 1. Define an enum for core types.
enum class ProcessorType {
  kLittleCore,
  kMediumCore,
  kBigCore,
};

// 2. Define a struct for a Core.
struct Core {
  int id;              // The OS/core id.
  ProcessorType type;  // Type of the core (LITTLE, MEDIUM, BIG).
  bool pinnable;       // Whether this core is available for pinning.
};

// 3. Create a Device class that holds a list of cores.
class Device {
 public:
  // Construct a device with a name and a list of cores.
  Device(std::string name, std::vector<Core> cores)
      : name_(std::move(name)), cores_(std::move(cores)) {}

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

  // Get all pinnable cores (optionally filtered by type).
  std::vector<Core> getPinnableCores(ProcessorType type = ProcessorType::kLittleCore) const {
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
