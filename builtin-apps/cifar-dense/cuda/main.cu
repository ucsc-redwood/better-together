#include "../../app.hpp"
#include "../../hex_dump.hpp"
#include "dispatchers.cuh"

#include <omp.h>

int main(int argc, char** argv) {
  parse_args(argc, argv);

  // print number of omp threads
  printf("Number of OMP threads: %d\n", omp_get_max_threads());
  #pragma omp parallel
  {
    printf("Thread ID: %d\n", omp_get_thread_num());
  }

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  cifar_dense::cuda::CudaDispatcher disp;

  cifar_dense::AppData appdata(&disp.get_mr());

  dumpCompressed(appdata.u_conv1_out.pmr_vec());

  disp.dispatch_stage(appdata, 1);

  dumpCompressed(appdata.u_conv1_out.pmr_vec());

  disp.dispatch_stage(appdata, 2);
  dumpCompressed(appdata.u_pool1_out.pmr_vec());

  disp.dispatch_stage(appdata, 3);
  dumpCompressed(appdata.u_conv2_out.pmr_vec());

  disp.dispatch_stage(appdata, 4);
  dumpCompressed(appdata.u_pool2_out.pmr_vec());

  disp.dispatch_stage(appdata, 5);
  dumpCompressed(appdata.u_conv3_out.pmr_vec());

  disp.dispatch_stage(appdata, 6);
  dumpCompressed(appdata.u_conv4_out.pmr_vec());

  disp.dispatch_stage(appdata, 7);
  dumpCompressed(appdata.u_conv5_out.pmr_vec());

  disp.dispatch_stage(appdata, 8);
  dumpCompressed(appdata.u_conv5_out.pmr_vec());

  disp.dispatch_stage(appdata, 9);
  dumpCompressed(appdata.u_conv5_out.pmr_vec());

  return 0;
}
