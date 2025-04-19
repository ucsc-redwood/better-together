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
    // Bind thread to appropriate core
    bind_thread_to_cores(cores_to_use);

    octree::omp::run_stage_1(appdata);
    octree::omp::run_stage_2(appdata);
    octree::omp::run_stage_3(appdata);
    octree::omp::run_stage_4(appdata);
    octree::omp::run_stage_5(appdata);
    octree::omp::run_stage_6(appdata);
    octree::omp::run_stage_7(appdata);
  }

  // Verify the sort worked correctly
  bool is_sorted =
      std::is_sorted(appdata.u_morton_codes.begin(), appdata.u_morton_codes.begin() + appdata.m);
  if (!is_sorted) {
    spdlog::error("Morton codes are not properly sorted!");
  } else {
    spdlog::info("Morton codes successfully sorted");
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

  appdata.print_radix_tree(appdata, 10);
  appdata.print_octree_nodes(appdata);

  return 0;
}
