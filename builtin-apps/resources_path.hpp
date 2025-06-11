#pragma once

#include <filesystem>

namespace helpers {

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

}  // namespace helpers
