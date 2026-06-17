// Normal-vs-fully per-stage benchmark for CIFAR-dense x VK. The per-PU thread/timing
// logic + main() live in ../bm_fully_common.hpp; this file supplies the cell types
// (const.hpp, included first) and the OMP dispatch closure + GPU backend selection.
#include "const.hpp"

#include "../bm_fully_common.hpp"

int main(int argc, char** argv) {
  return bt_fully::run<bt_fully::WallTimer>(
      argc, argv, ProcessorType::kVulkan, 3, "Vulkan",
      [](const std::vector<int>& cores, size_t n, AppDataT& app, int lo, int hi) {
        cifar_dense::omp::dispatch_multi_stage(cores, n, app, lo, hi);
      });
}
