#pragma once

#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <vector>

#include "conf.hpp"

inline std::string g_device_id;
inline std::string g_spdlog_log_level;
inline std::vector<int> g_little_cores;
inline std::vector<int> g_medium_cores;
inline std::vector<int> g_big_cores;
inline bool g_debug_filelogger = false;

[[nodiscard]] size_t get_vulkan_warp_size();

#define PARSE_ARGS_BEGIN CLI::App app{"default"};

// this way we can add app.add_option() before PARSE_ARGS_END to add additional options

#define PARSE_ARGS_END                                                                    \
  app.add_option("-d,--device", g_device_id, "Device ID")->required();                    \
  app.add_option("-l,--log-level", g_spdlog_log_level, "Log level")->default_val("info"); \
  app.add_flag("--debug_filelogger", g_debug_filelogger, "Debug filelogger");             \
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
    std::string little_cores_str;                                                         \
    for (const auto& core : littleCores) {                                                \
      little_cores_str += std::to_string(core.id) + " ";                                  \
      g_little_cores.push_back(core.id);                                                  \
    }                                                                                     \
    spdlog::info("Little cores: {}", little_cores_str);                                   \
    std::string medium_cores_str;                                                         \
    for (const auto& core : mediumCores) {                                                \
      medium_cores_str += std::to_string(core.id) + " ";                                  \
      g_medium_cores.push_back(core.id);                                                  \
    }                                                                                     \
    spdlog::info("Medium cores: {}", medium_cores_str);                                   \
    std::string big_cores_str;                                                            \
    for (const auto& core : bigCores) {                                                   \
      big_cores_str += std::to_string(core.id) + " ";                                     \
      g_big_cores.push_back(core.id);                                                     \
    }                                                                                     \
    spdlog::info("Big cores: {}", big_cores_str);                                         \
  } catch (const std::exception& e) {                                                     \
    std::cerr << e.what() << std::endl;                                                   \
    return 1;                                                                             \
  }

int parse_args(int argc, char** argv);