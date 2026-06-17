#pragma once

#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <optional>
#include <string>
#include <vector>

#include "conf.hpp"
#include "pipeline/schedule.hpp"  // Schedule, first_unavailable_pu

inline std::string g_device_id;
inline std::string g_spdlog_log_level;

// cores
inline std::vector<int> g_lit_cores;
inline std::vector<int> g_med_cores;
inline std::vector<int> g_big_cores;
inline std::vector<int> g_sup_cores;

[[nodiscard]] static inline bool has_lit_cores() { return !g_lit_cores.empty(); }
[[nodiscard]] static inline bool has_med_cores() { return !g_med_cores.empty(); }
[[nodiscard]] static inline bool has_big_cores() { return !g_big_cores.empty(); }
[[nodiscard]] static inline bool has_sup_cores() { return !g_sup_cores.empty(); }

// First CPU tier the device actually has -- use for portable warmups/defaults so
// a hardcoded tier (e.g. Little) never throws on a device that lacks it (the
// Big-only MiniPC has no Little cores).
[[nodiscard]] static inline ProcessorType first_present_cpu_type() {
  if (has_big_cores()) return ProcessorType::kBigCore;
  if (has_med_cores()) return ProcessorType::kMediumCore;
  return ProcessorType::kLittleCore;  // little, or the only remaining choice
}

// Is this PU type present on the device we parsed (g_*_cores filled by PARSE_ARGS)?
// GPU PUs (kVulkan/kCuda) are implied present: each executor binary is backend-specific
// and only ever dispatches its own GPU chunks, so reaching one means the device has it.
[[nodiscard]] static inline bool device_has_pu(const ProcessorType pt) {
  switch (pt) {
    case ProcessorType::kLittleCore:
      return has_lit_cores();
    case ProcessorType::kMediumCore:
      return has_med_cores();
    case ProcessorType::kBigCore:
      return has_big_cores();
    case ProcessorType::kSuperCore:
      return has_sup_cores();
    case ProcessorType::kVulkan:
    case ProcessorType::kCuda:
      return true;
  }
  return false;
}

// Up-front executor guard: nullopt if the schedule is runnable on THIS device, else
// the reason its first offending chunk can't run (so the caller can skip+warn instead
// of letting an absent-PU chunk throw inside an unguarded worker thread). See
// first_unavailable_pu() in pipeline/schedule.hpp.
[[nodiscard]] static inline std::optional<std::string> schedule_unrunnable_reason(
    const Schedule& schedule) {
  return first_unavailable_pu(schedule, device_has_pu);
}

[[nodiscard]] static inline std::vector<int>& get_cores_by_type(const ProcessorType core_type) {
  switch (core_type) {
    case ProcessorType::kLittleCore:
      return g_lit_cores;
    case ProcessorType::kMediumCore:
      return g_med_cores;
    case ProcessorType::kBigCore:
      return g_big_cores;
    case ProcessorType::kSuperCore:
      return g_sup_cores;
    default:
      throw std::invalid_argument("Invalid core type");
  }
}

[[nodiscard]] static inline std::optional<std::vector<int>> get_cpu_cores_by_type(
    const ProcessorType core_type) {
  switch (core_type) {
    case ProcessorType::kLittleCore:
      return g_lit_cores;
    case ProcessorType::kMediumCore:
      return g_med_cores;
    case ProcessorType::kBigCore:
      return g_big_cores;
    case ProcessorType::kSuperCore:
      return g_sup_cores;
    default:
      return std::nullopt;
  }
}

// Define macros for clearer test code
#define LITTLE_CORES g_lit_cores, g_lit_cores.size()
#define MEDIUM_CORES g_med_cores, g_med_cores.size()
#define BIG_CORES g_big_cores, g_big_cores.size()
#define SUPER_CORES g_sup_cores, g_sup_cores.size()

[[nodiscard]] size_t get_vulkan_warp_size();

#define PARSE_ARGS_BEGIN CLI::App app{"default"};

// this way we can add app.add_option() before PARSE_ARGS_END to add additional options

#define PARSE_ARGS_END                                                                    \
  app.add_option("-d,--device", g_device_id, "Device ID")->required();                    \
  app.add_option("-l,--log-level", g_spdlog_log_level, "Log level")->default_val("info"); \
  app.allow_extras();                                                                     \
  CLI11_PARSE(app, argc, argv);                                                           \
  if (g_device_id.empty()) {                                                              \
    throw std::runtime_error("Device ID is required");                                    \
    exit(1);                                                                              \
  }                                                                                       \
  auto& registry = GlobalDeviceRegistry();                                                \
  try {                                                                                   \
    const Device& device = registry.getDevice(g_device_id);                               \
    auto littleCores = device.getCores(ProcessorType::kLittleCore);                       \
    auto mediumCores = device.getCores(ProcessorType::kMediumCore);                       \
    auto bigCores = device.getCores(ProcessorType::kBigCore);                             \
    auto supCores = device.getCores(ProcessorType::kSuperCore);                           \
    std::string little_cores_str;                                                         \
    for (const auto& core : littleCores) {                                                \
      little_cores_str += std::to_string(core.id) + " ";                                  \
      g_lit_cores.push_back(core.id);                                                     \
    }                                                                                     \
    spdlog::info("Pinable Lit cores: {}", little_cores_str);                              \
    std::string medium_cores_str;                                                         \
    for (const auto& core : mediumCores) {                                                \
      medium_cores_str += std::to_string(core.id) + " ";                                  \
      g_med_cores.push_back(core.id);                                                     \
    }                                                                                     \
    spdlog::info("Pinable Med cores: {}", medium_cores_str);                              \
    std::string big_cores_str;                                                            \
    for (const auto& core : bigCores) {                                                   \
      big_cores_str += std::to_string(core.id) + " ";                                     \
      g_big_cores.push_back(core.id);                                                     \
    }                                                                                     \
    spdlog::info("Pinable Big cores: {}", big_cores_str);                                 \
    std::string sup_cores_str;                                                            \
    for (const auto& core : supCores) {                                                   \
      sup_cores_str += std::to_string(core.id) + " ";                                     \
      g_sup_cores.push_back(core.id);                                                     \
    }                                                                                     \
    spdlog::info("Pinable Sup cores: {}", sup_cores_str);                                 \
  } catch (const std::exception& e) {                                                     \
    std::cerr << e.what() << std::endl;                                                   \
    return 1;                                                                             \
  }

int parse_args(int argc, char** argv);

// Non-fatal variant for unit tests. Resolves --device when present and fills the
// g_*_cores from the registry when the device is known, but NEVER exit()s or
// returns non-zero on a missing/unknown device — tests that need specific cores
// or a GPU guard themselves with GTEST_SKIP(). This decouples CI from the
// hardcoded device allow-list: a board absent from the registry no longer aborts
// the whole test binary before a single test runs.
int parse_args_test(int argc, char** argv);