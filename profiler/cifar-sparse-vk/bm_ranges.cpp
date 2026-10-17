// GPU stage-range records (gpu-chunk-cost-model D2) for cifar-sparse x Vulkan: see
// ../bm_ranges_vulkan.hpp.
#include <cstdio>
#include <exception>

#include "const.hpp"
#include "profiler/bm_ranges_vulkan.hpp"

int main(int argc, char** argv) {
  try {
    return bt_ranges::run(
        argc,
        argv,
        "cifar-sparse",
        [](const std::vector<int>& cores, size_t n, AppDataT& app, int lo, int hi) {
          cifar_sparse::omp::dispatch_multi_stage(cores, static_cast<int>(n), app, lo, hi);
        });
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bm-ranges: %s\n", e.what());
    return 1;
  }
}
