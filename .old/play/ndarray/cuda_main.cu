#include <nvtx3/nvToolsExt.h>
#include <spdlog/spdlog.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/common/cuda/cu_mem_resource.cuh"
#include "cuda/dispatchers.cuh"

void run_cuda_pinned() {
  cuda::CudaDispatcher<cuda::CudaPinnedResource> disp;
  cifar_dense::AppDataBatch batched_appdata(&disp.get_mr());

  nvtxRangePushA("Compute");

  disp.dispatch_multi_stage(batched_appdata, 1, 9);

  nvtxRangePop();
  cifar_dense::print_batch_predictions(batched_appdata, 10);
}

void run_cuda_managed() {
  cuda::CudaDispatcher<cuda::CudaManagedResource> disp;
  cifar_dense::AppDataBatch batched_appdata(&disp.get_mr());

  nvtxRangePushA("Compute");

  disp.dispatch_multi_stage(batched_appdata, 1, 9);

  nvtxRangePop();
  cifar_dense::print_batch_predictions(batched_appdata, 10);
}

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN;
  bool pinned = false;
  app.add_flag("--pinned", pinned, "Run the CUDA code with pinned memory");
  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (pinned) {
    run_cuda_pinned();
  } else {
    run_cuda_managed();
  }

  spdlog::info("Done");
  return 0;
}
