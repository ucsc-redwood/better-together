// Normal-vs-fully per-stage benchmark for Tree x CUDA. The per-PU thread/timing logic
// + main() live in ../bm_fully_common.hpp; this file supplies the cell types
// (const.hpp, included first) and the OMP dispatch closure + GPU backend selection.
// The cu cell times the GPU window with a cudaEvent (CudaEventTimer).
#include "const.hpp"

#include "../bm_fully_common.hpp"

int main(int argc, char** argv) {
  return bt_fully::run<bt_fully::CudaEventTimer>(
      argc, argv, ProcessorType::kCuda, 4, "CUDA",
      [](const std::vector<int>& cores, size_t n, AppDataT& app, int lo, int hi) {
        tree::omp::dispatch_multi_stage(cores, n, app, lo, hi);
      });
}
