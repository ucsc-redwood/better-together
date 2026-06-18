// ---------------------------------------------------------------------------
// bm_prof -- canonical-JSONL profiler for cifar-dense x CUDA (isolated).
//
// One bm_prof binary per (app, backend) cell. The whole driver -- env knobs,
// per-cell registration, the calibrated measured loop, interference load, and
// JSONL emission -- lives in ../bm_prof_common.hpp (run_bm_prof); the CUDA
// GPU-timer policy (cudaEvent path + wall-clock fallback) lives in
// ../bm_prof_cuda.cuh. This cell supplies only its three points of variation:
// the app/backend identity, the CUDA timer policy, and the app's OMP token.
// ---------------------------------------------------------------------------

#include "profiler/bm_prof_cuda.cuh"
#include "const.hpp"  // DispatcherT (CudaDispatcher), AppDataT, kNumStages; pulls in CheckCuda

int main(int argc, char** argv) {
  return bt_prof::run_bm_prof<DispatcherT, AppDataT>(
      argc, argv, "cifar-dense", "cuda", "cuda", static_cast<int>(kNumStages),
      [](DispatcherT&) { return bt_prof::CudaTimer<DispatcherT, AppDataT>{}; },
      [](auto&& cores, size_t n, AppDataT& app, int a, int b) {
        cifar_dense::omp::dispatch_multi_stage(cores, n, app, a, b);
      });
}
