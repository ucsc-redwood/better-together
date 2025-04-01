#pragma once

#include <filesystem>

namespace helpers {

constexpr const char* kBenchmarkResultsPath = "data/raw_bm_results";
constexpr const char* kLogsPath = "data/logs";

// ----------------------------------------------------------------------------
// Helper function to get the path to the resources directory
// Based on the platform, this will be different.
// ----------------------------------------------------------------------------

// We assume you will run the program using "xmake run XXX" from the $(project_root) instead of
// directly running the binary.

[[nodiscard]] inline std::filesystem::path get_project_root_path() {
  auto cwd = std::filesystem::current_path();

#if defined(__ANDROID__)
  return "/data/local/tmp/";
#else

  // check current dir see if it contains "build", if so, we are already in the project root
  // get an list of all files in the current dir
  for (const auto& entry : std::filesystem::directory_iterator(cwd)) {
    if (entry.path().filename() == "build") {
      return cwd;
    }
  }

  // build
  // └── linux
  //     └── x86_64
  //         ├── debug
  //         │   ├── bm-cifar-dense
  // resources
  // data
  return std::filesystem::current_path().parent_path().parent_path().parent_path().parent_path();
#endif
}

[[nodiscard]] inline std::filesystem::path get_resource_base_path() {
  return get_project_root_path() / "resources";
}

// where to put the benchmark results?
// On Linux/Windows:
//  just $(project_root)/data/raw_bm_results
// On Android:
//  /data/local/tmp/
//  and then we have to use adb pull to get the file to
//  $(project_root)/data/raw_bm_results
// using a different script

[[nodiscard]] inline std::filesystem::path get_benchmark_storage_location() {
#if defined(__ANDROID__)
  return "/data/local/tmp/";
#else
  return get_project_root_path() / kBenchmarkResultsPath;
#endif
}

[[nodiscard]] inline std::filesystem::path get_log_storage_location() {
#if defined(__ANDROID__)
  return "/data/local/tmp/";
#else
  return get_project_root_path() / kLogsPath;
#endif
}

}  // namespace helpers
