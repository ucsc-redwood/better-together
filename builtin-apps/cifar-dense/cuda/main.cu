#include "../../app.hpp"
#include "dispatchers.cuh"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  cifar_dense::cuda::CudaDispatcher disp;

  cifar_dense::AppData appdata(&disp.get_mr());

  disp.run_stage_1_async(appdata);
  disp.run_stage_2_async(appdata);
  disp.run_stage_3_async(appdata);
  disp.run_stage_4_async(appdata);
  disp.run_stage_5_async(appdata);
  disp.run_stage_6_async(appdata);
  disp.run_stage_7_async(appdata);
  disp.run_stage_8_async(appdata);
  disp.run_stage_9_async(appdata);

  return 0;
}
