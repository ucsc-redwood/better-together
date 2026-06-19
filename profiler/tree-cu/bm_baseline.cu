// Single-PU "no framework" baselines for Tree x CUDA. All registration +
// the main() live in ../bm_baseline_common.hpp; this file only supplies the cell
// types (const.hpp) and the per-backend dispatch closures.
#include "const.hpp"
#include "profiler/bm_baseline_common.hpp"

int main(int argc, char** argv) {
  return bt_baseline::run<AppDataT, DispatcherT>(
      argc,
      argv,
      static_cast<int>(kNumStages),
      "Tree",
      "CUDA",
      [](AppDataT& a, int lo, int hi) { tree::omp::dispatch_multi_stage(a, lo, hi); },
      [](std::vector<int>& c, size_t n, AppDataT& a, int lo, int hi) {
        tree::omp::dispatch_multi_stage(c, n, a, lo, hi);
      },
      [](DispatcherT& d, AppDataT& a, int lo, int hi) { d.dispatch_multi_stage(a, lo, hi); });
}
