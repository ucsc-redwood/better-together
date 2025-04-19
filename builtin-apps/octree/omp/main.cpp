#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../../app.hpp"
#include "../appdata.hpp"
#include "dispatchers.hpp"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  auto mr = std::pmr::new_delete_resource();
  octree::AppData appdata(mr);

  

#pragma omp parallel num_threads(g_little_cores.size())
  {
    printf("Thread %d/%d\n", omp_get_thread_num(), omp_get_num_threads());
    bind_thread_to_cores(g_little_cores);

    // octree::omp::run_stage_1(appdata);
    // octree::omp::run_stage_2(appdata);
  }

  // print first 10 points
  for (size_t i = 0; i < 10; i++) {
    std::cout << "[pos " << i << "]\t" << appdata.u_positions[i].x << ", "
              << appdata.u_positions[i].y << ", " << appdata.u_positions[i].z << ", "
              << appdata.u_positions[i].w << std::endl;
  }
  std::cout << std::endl;

  // print first 10 morton codes
  for (size_t i = 0; i < 10; i++) {
    std::cout << "[morton " << i << "]\t" << appdata.u_morton_codes[i] << std::endl;
  }
  std::cout << std::endl;

  //

  return 0;
}
