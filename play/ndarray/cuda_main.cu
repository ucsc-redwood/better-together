#include <spdlog/spdlog.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/common/cuda/manager.cuh"
#include "cuda/dispatchers.cuh"
#include "omp/dispatchers.hpp"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  cuda::CudaManager mgr;
  const cuda::DeviceModelData d_model_data(cifar_dense::AppDataBatch::get_model());

  cifar_dense::AppDataBatch batched_appdata(&mgr.get_mr());

  omp::dispatch_multi_stage<ProcessorType::kLittleCore>(4, batched_appdata, 1, 3);
  cuda::dispatch_multi_stage(batched_appdata, d_model_data, 4, 6, mgr);
  // omp::dispatch_multi_stage<ProcessorType::kBigCore>(2, batched_appdata, 7, 9);

  cifar_dense::print_batch_predictions(batched_appdata, 10);

  return 0;
}
