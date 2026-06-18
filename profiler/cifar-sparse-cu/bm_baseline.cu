// Single-PU "no framework" baselines for CIFAR-Sparse x CUDA. All registration +
// the main() live in ../bm_baseline_common.hpp; this file only supplies the cell
// types (const.hpp) and the per-backend dispatch closures.
#include "profiler/bm_baseline_common.hpp"
#include "const.hpp"

int main(int argc, char** argv) {
  return bt_baseline::run<AppDataT, DispatcherT>(
      argc, argv, static_cast<int>(kNumStages), "CIFAR-Sparse", "CUDA",
      [](AppDataT& a, int lo, int hi) { cifar_sparse::omp::dispatch_multi_stage(a, lo, hi); },
      [](std::vector<int>& c, size_t n, AppDataT& a, int lo, int hi) {
        cifar_sparse::omp::dispatch_multi_stage(c, n, a, lo, hi);
      },
      [](DispatcherT& d, AppDataT& a, int lo, int hi) { d.dispatch_multi_stage(a, lo, hi); });
}
