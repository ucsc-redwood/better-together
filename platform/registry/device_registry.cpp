#include "platform/registry/device_registry.hpp"

// Cross-check only: the Vulkan engine takes the subgroup size from the running device.
// devices/<id>.json -> gpu.subgroup_size is informational; a disagreement is a warning.
bool check_vulkan_subgroup_size(const uint32_t engine_subgroup_size) {
  if (g_device_id.empty()) return true;

  const auto& devices = GlobalDeviceRegistry();
  if (!devices.hasDevice(g_device_id)) return true;
  const int registry = devices.getDevice(g_device_id).gpu_subgroup_size();
  if (registry <= 0 || static_cast<uint32_t>(registry) == engine_subgroup_size) return true;

  spdlog::warn(
      "devices/{}.json declares gpu.subgroup_size {} but the Vulkan device reports {}; "
      "using {}",
      g_device_id,
      registry,
      engine_subgroup_size,
      engine_subgroup_size);
  return false;
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
    for (const auto& core : device.getCores(ProcessorType::kLittleCore))
      g_lit_cores.push_back(core.id);
    for (const auto& core : device.getCores(ProcessorType::kMediumCore))
      g_med_cores.push_back(core.id);
    for (const auto& core : device.getCores(ProcessorType::kBigCore))
      g_big_cores.push_back(core.id);
    for (const auto& core : device.getCores(ProcessorType::kSuperCore))
      g_sup_cores.push_back(core.id);
  } catch (const std::exception& e) {
    spdlog::warn(
        "Unknown device '{}': {}. Hardware-specific tests will self-skip.", g_device_id, e.what());
  }
  return 0;
}