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
  spdlog::info("Using {} little cores, {} medium cores, {} big cores",
               g_lit_cores.size(), g_med_cores.size(), g_big_cores.size());

  auto mr = std::pmr::new_delete_resource();
  octree::AppData appdata(mr);

  // Use little cores if available, or big cores as fallback
  const std::vector<int>& cores_to_use = !g_lit_cores.empty() ? g_lit_cores : g_big_cores;
  const size_t num_threads = cores_to_use.size();
  
  if (num_threads == 0) {
    spdlog::error("No cores available for execution");
    return 1;
  }
  
  spdlog::info("Using {} cores for execution", num_threads);

#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    
    // Bind thread to appropriate core
    bind_thread_to_cores(cores_to_use);
    
    spdlog::info("Thread {}/{} running on core set", tid, nth);
    
    // Run stages in sequence with proper synchronization
    octree::omp::run_stage_1(appdata);
    
#pragma omp barrier
    octree::omp::run_stage_2(appdata);
    
#pragma omp barrier
    // Additional stages would go here if needed
  }

  // Print first 10 points
  for (size_t i = 0; i < std::min(size_t(10), appdata.n); i++) {
    std::cout << "[pos " << i << "]\t" << appdata.u_positions[i].x << ", "
              << appdata.u_positions[i].y << ", " << appdata.u_positions[i].z << ", "
              << appdata.u_positions[i].w << std::endl;
  }
  std::cout << std::endl;

  // Print first 10 morton codes (input)
  for (size_t i = 0; i < std::min(size_t(10), appdata.n); i++) {
    std::cout << "[morton_alt " << i << "]\t" << appdata.u_morton_codes_alt[i] << std::endl;
  }
  std::cout << std::endl;

  // Print first 10 morton codes (sorted)
  for (size_t i = 0; i < std::min(size_t(10), appdata.n); i++) {
    std::cout << "[morton " << i << "]\t" << appdata.u_morton_codes[i] << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
