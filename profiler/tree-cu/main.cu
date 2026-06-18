// Two-stage concurrent-pipeline demo for tree x CUDA. The runner lives in
// ../bm_demo_common.hpp; this cell supplies only the stage-1 OMP token.
#include "const.hpp"  // DispatcherT, AppDataT, QueueT, kNumToProcess; pulls in worker/make_dataset
#include "profiler/bm_demo_common.hpp"

int main(int argc, char** argv) {
  return bt_demo::run_main<DispatcherT, AppDataT, AppDataPtr, QueueT>(
      argc, argv, kNumToProcess,
      [](AppDataT* app) { tree::omp::dispatch_stage(*app, 1); });
}
