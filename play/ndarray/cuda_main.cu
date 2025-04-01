#include <nvtx3/nvToolsExt.h>
#include <spdlog/spdlog.h>

#include <memory>

#include "builtin-apps/app.hpp"
#include "builtin-apps/common/cuda/manager.cuh"
#include "cuda/dispatchers.cuh"

#define PREPARE_DATA                                        \
  cuda::CudaManager mgr;                                    \
  cifar_dense::AppDataBatch batched_appdata(&mgr.get_mr()); \
  const cuda::DeviceModelData d_model_data(cifar_dense::AppDataBatch::get_model());

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  // cuda::CudaManager mgr;
  // const cuda::DeviceModelData d_model_data(cifar_dense::AppDataBatch::get_model());

  // cifar_dense::AppDataBatch batched_appdata(&mgr.get_mr());

  PREPARE_DATA;

  nvtxRangePushA("Compute");
  cuda::dispatch_multi_stage(batched_appdata, d_model_data, 1, 9, mgr);
  CheckCuda(cudaDeviceSynchronize());

  nvtxRangePop();

  cifar_dense::print_batch_predictions(batched_appdata, 10);

  return 0;
}
