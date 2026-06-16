#include "app.hpp"

#include <unordered_map>

size_t get_vulkan_warp_size() {
  assert(g_device_id.empty() == false);

  // Map of device IDs to their corresponding warp sizes
  static const std::unordered_map<std::string, size_t> device_warp_sizes = {
      {"3A021JEHN02756", 16},        // Mali-G710
      {"9b034f1b", 64},              // Adreno (TM) 740
      {"R9TR30814KJ", 64},           // Adreno (TM) 610
      {"ce0717178d7758b00b7e", 32},  // Adreno (TM) 540
      {"minipc", 64},                // AMD Radeon 780M
      {"pc", 32},                    // Intel UHD Graphics 770
      {"jetson", 32},                // NVIDIA Tegra Orin (nvgpu)
      {"jetsonlowpower", 32},        // NVIDIA Tegra Orin (nvgpu)
      {"mba", 32},                   // Apple M4
      {"R5CY21Y3VEV", 32},           // Samsung Galaxy S24 (SM-S926B), Xclipse 940
  };

  auto it = device_warp_sizes.find(g_device_id);
  if (it != device_warp_sizes.end()) {
    return it->second;
  }

  throw std::runtime_error("Invalid device ID. [" + g_device_id + "] " + std::string(__FILE__) +
                           ":" + std::to_string(__LINE__));
}

bool check_device_arg(const int argc, char** argv) {
  for (int i = 0; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.find("--device=") != std::string::npos) {
      return true;
    }
    if (arg == "--device" && i + 1 < argc) {
      return true;
    }
  }
  std::cerr << "Error: --device argument is required\n";
  std::exit(1);
  return false;
}

int parse_args(int argc, char** argv) {
  if (!check_device_arg(argc, argv)) {
    std::exit(1);
  }
  PARSE_ARGS_BEGIN
  PARSE_ARGS_END
  return 0;
}

int parse_args_test(int argc, char** argv) {
  CLI::App app{"unit-test"};
  app.add_option("-d,--device", g_device_id, "Device ID");  // not ->required()
  app.add_option("-l,--log-level", g_spdlog_log_level, "Log level")->default_val("info");
  app.allow_extras();  // let gtest's own flags pass through
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError&) {
    // Ignore CLI parse problems in the test harness; never abort the binary.
  }

  if (g_device_id.empty()) {
    spdlog::warn("No --device given; core-pinning / device-specific tests will self-skip.");
    return 0;
  }

  // Populate the pinnable-core lists when the device is known. An unknown device
  // is a warning, not a fatal error: the relevant tests GTEST_SKIP() on empty
  // core lists / absent accelerators instead of aborting the whole run.
  try {
    const Device& device = GlobalDeviceRegistry().getDevice(g_device_id);
    for (const auto& core : device.getCores(ProcessorType::kLittleCore)) g_lit_cores.push_back(core.id);
    for (const auto& core : device.getCores(ProcessorType::kMediumCore)) g_med_cores.push_back(core.id);
    for (const auto& core : device.getCores(ProcessorType::kBigCore)) g_big_cores.push_back(core.id);
    for (const auto& core : device.getCores(ProcessorType::kSuperCore)) g_sup_cores.push_back(core.id);
  } catch (const std::exception& e) {
    spdlog::warn("Unknown device '{}': {}. Hardware-specific tests will self-skip.", g_device_id,
                 e.what());
  }
  return 0;
}