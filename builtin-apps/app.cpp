#include "app.hpp"

[[nodiscard]] size_t get_vulkan_warp_size() {
  if (g_device_id == "3A021JEHN02756") {
    return 16;
  } else if (g_device_id == "9b034f1b") {
    return 64;
  } else if (g_device_id == "ce0717178d7758b00b7e") {
    return 32;
  } else if (g_device_id == "minipc") {
    return 64;
  } else if (g_device_id == "pc" || g_device_id == "jetson") {
    return 32;
  } else if (g_device_id == "jetsonlowpower") {
    return 32;
  }
  throw std::runtime_error("Invalid device ID. " + std::string(__FILE__) + ":" +
                           std::to_string(__LINE__));
}

[[nodiscard]] inline bool check_device_arg(const int argc, char** argv) {
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