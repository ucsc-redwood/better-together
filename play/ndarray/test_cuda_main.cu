#include <spdlog/spdlog.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/common/cuda/manager.cuh"
#include "cuda/dispatchers.cuh"
#include "cuda/model_data.cuh"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  cuda::CudaManager mgr;

  cifar_dense::AppDataBatch batched_appdata(&mgr.get_mr());
  const cuda::DeviceModelData d_model_data(cifar_dense::AppDataBatch::get_model());

  cuda::run_stage_1(batched_appdata, d_model_data, mgr);
  cuda::run_stage_2(batched_appdata, d_model_data, mgr);
  cuda::run_stage_3(batched_appdata, d_model_data, mgr);
  cuda::run_stage_4(batched_appdata, d_model_data, mgr);
  cuda::run_stage_5(batched_appdata, d_model_data, mgr);
  cuda::run_stage_6(batched_appdata, d_model_data, mgr);
  cuda::run_stage_7(batched_appdata, d_model_data, mgr);
  cuda::run_stage_8(batched_appdata, d_model_data, mgr);
  cuda::run_stage_9(batched_appdata, d_model_data, mgr);

  // Print result
  int predicted_class = cifar_dense::arg_max(batched_appdata.linear_out.raw());
  cifar_dense::print_prediction(predicted_class);

  return 0;
}
