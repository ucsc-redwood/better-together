#include <omp.h>

#include "../../app.hpp"
#include "../../hex_dump.hpp"
#include "dispatchers.cuh"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  // print number of omp threads
  printf("Number of OMP threads: %d\n", omp_get_max_threads());
#pragma omp parallel
  {
    printf("Thread ID: %d\n", omp_get_thread_num());
  }

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  tree::cuda::CudaDispatcher disp;
  tree::SafeAppData appdata(&disp.get_mr());

  return 0;
}
