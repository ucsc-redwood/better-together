#include "app.hpp"

// Vulkan subgroup (warp) size, data-driven from the device registry
// (devices/<id>.json -> gpu.subgroup_size). Adding a board is now dropping a JSON
// spec, not editing a hardcoded map here. Throws if the device isn't registered
// or its spec has no GPU subgroup size.
size_t get_vulkan_warp_size() {
  assert(g_device_id.empty() == false);

  const int subgroup = GlobalDeviceRegistry().getDevice(g_device_id).gpu_subgroup_size();
  if (subgroup <= 0) {
    throw std::runtime_error("device [" + g_device_id +
                             "] has no gpu.subgroup_size in its devices/*.json spec");
  }
  return static_cast<size_t>(subgroup);
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