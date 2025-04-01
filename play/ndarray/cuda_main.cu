#include <nvtx3/nvToolsExt.h>
#include <spdlog/spdlog.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/common/cuda/cu_mem_resource.cuh"
#include "cuda/dispatchers.cuh"

#define PREPARE_DATA                              \
  cuda::CudaManagedResource mr;                   \
  cifar_dense::AppDataBatch batched_appdata(&mr); \
  const cuda::DeviceModelData d_model_data(cifar_dense::AppDataBatch::get_model());

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  PREPARE_DATA;

  nvtxRangePushA("Compute");

  cuda::dispatch_multi_stage(batched_appdata, d_model_data, 1, 9);
  CheckCuda(cudaDeviceSynchronize());

  nvtxRangePop();

  cifar_dense::print_batch_predictions(batched_appdata, 10);

  return 0;
}
