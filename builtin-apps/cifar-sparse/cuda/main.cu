#include "../../app.hpp"
#include "../../hex_dump.hpp"
#include "dispatchers.cuh"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  cifar_sparse::cuda::CudaDispatcher disp;

  cifar_sparse::AppData appdata(&disp.get_mr());

  dumpCompressed(appdata.u_conv1_out.pmr_vec());

  disp.run_stage_1_async(appdata);

  dumpCompressed(appdata.u_conv1_out.pmr_vec());

  disp.run_stage_2_async(appdata);
  dumpCompressed(appdata.u_pool1_out.pmr_vec());

  disp.run_stage_3_async(appdata);
  dumpCompressed(appdata.u_conv2_out.pmr_vec());

  disp.run_stage_4_async(appdata);
  dumpCompressed(appdata.u_pool2_out.pmr_vec());

  disp.run_stage_5_async(appdata);
  dumpCompressed(appdata.u_conv3_out.pmr_vec());

  disp.run_stage_6_async(appdata);
  dumpCompressed(appdata.u_conv4_out.pmr_vec());

  disp.run_stage_7_async(appdata);
  dumpCompressed(appdata.u_conv5_out.pmr_vec());

  disp.run_stage_8_async(appdata);
  dumpCompressed(appdata.u_conv5_out.pmr_vec());

  disp.run_stage_9_async(appdata);
  dumpCompressed(appdata.u_conv5_out.pmr_vec());

  return 0;
}
