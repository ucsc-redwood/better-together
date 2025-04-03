#include <nvtx3/nvToolsExt.h>
#include <spdlog/spdlog.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/common/cuda/cu_mem_resource.cuh"
#include "cuda/dispatchers.cuh"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  {
    cuda::CudaDispatcher<cuda::CudaPinnedResource> disp;
    cifar_dense::AppDataBatch batched_appdata(&disp.get_mr());

    nvtxRangePushA("Compute");

    disp.dispatch_multi_stage(batched_appdata, 1, 9);

    nvtxRangePop();
    cifar_dense::print_batch_predictions(batched_appdata, 10);
  }

  spdlog::info("Done");
  return 0;
}
