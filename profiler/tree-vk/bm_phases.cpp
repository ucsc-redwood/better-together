// Per-chunk phase timing (design D1) for tree x Vulkan: see ../bm_phases_vulkan.hpp.
#include <cstdio>
#include <exception>

#include "const.hpp"
#include "profiler/bm_phases_vulkan.hpp"

int main(int argc, char** argv) {
  try {
    return bt_phases::run(
        argc, argv, [](const std::vector<int>& cores, size_t n, AppDataT& app, int lo, int hi) {
          tree::omp::dispatch_multi_stage(cores, static_cast<int>(n), app, lo, hi);
        });
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bm-phases: %s\n", e.what());
    return 1;
  }
}
