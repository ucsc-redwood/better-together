#include <spdlog/spdlog.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/common/cuda/manager.cuh"
#include "cuda/dispatchers.cuh"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  cuda::CudaManager mgr;

  cifar_dense::AppDataBatch batched_appdata(&mgr.get_mr());

  cuda::run_stage_1(batched_appdata, mgr);

  // Print result
  int predicted_class = cifar_dense::arg_max(batched_appdata.linear_out.raw());
  cifar_dense::print_prediction(predicted_class);

  return 0;
}
