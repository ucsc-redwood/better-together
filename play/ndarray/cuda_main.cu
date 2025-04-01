#include <nvtx3/nvToolsExt.h>
#include <spdlog/spdlog.h>

#include <memory>

#include "builtin-apps/app.hpp"
#include "builtin-apps/common/cuda/manager.cuh"
#include "cuda/dispatchers.cuh"
#include "omp/dispatchers.hpp"

// Global CUDA objects to ensure proper lifetime and thread visibility
cuda::CudaManager g_cuda_mgr;
std::unique_ptr<cuda::DeviceModelData> g_device_model_data;

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  // static cuda::CudaManager mgr;
  // static cuda::DeviceModelData d_model_data(cifar_dense::AppDataBatch::get_model());
  g_device_model_data =
      std::make_unique<cuda::DeviceModelData>(cifar_dense::AppDataBatch::get_model());

  cifar_dense::AppDataBatch batched_appdata(&g_cuda_mgr.get_mr());
  nvtxRangePushA("Compute");

  omp::dispatch_multi_stage<ProcessorType::kLittleCore>(4, batched_appdata, 1, 3);
  cuda::dispatch_multi_stage(batched_appdata, *g_device_model_data, 4, 9, g_cuda_mgr);
  nvtxRangePop();

  cifar_dense::print_batch_predictions(batched_appdata, 10);

  return 0;
}
