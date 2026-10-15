// Per-chunk phase timing (design D1) for tree x Vulkan: see ../bm_phases_vulkan.hpp.
#include "const.hpp"
#include "profiler/bm_phases_vulkan.hpp"

int main(int argc, char** argv) {
  return bt_phases::run(
      argc, argv, [](const std::vector<int>& cores, size_t n, AppDataT& app, int lo, int hi) {
        tree::omp::dispatch_multi_stage(cores, n, app, lo, hi);
      });
}
