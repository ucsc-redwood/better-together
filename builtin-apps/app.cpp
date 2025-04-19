#include "app.hpp"

#include <unordered_map>

size_t get_vulkan_warp_size() {
  assert(g_device_id.empty() == false);

  // Map of device IDs to their corresponding warp sizes
  static const std::unordered_map<std::string, size_t> device_warp_sizes = {
      {"3A021JEHN02756", 16},  // Mali-G710
      {"9b034f1b", 64},        // Adreno (TM) 740
      {"R9TR30814KJ", 64},     // Adreno (TM) 610
      {"ce0717178d7758b00b7e", 32},
      {"minipc", 64},  // AMD Radeon 780M
      {"pc", 32},
      {"jetson", 32},          // NVIDIA Tegra Orin (nvgpu)
      {"jetsonlowpower", 32},  // NVIDIA Tegra Orin (nvgpu)
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