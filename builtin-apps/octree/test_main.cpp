#include "../app.hpp"
#include "appdata.hpp"
#include "omp/dispatchers.hpp"

#define LITTLE_CORES g_little_cores, g_little_cores.size()
#define MEDIUM_CORES g_medium_cores, g_medium_cores.size()
#define BIG_CORES g_big_cores, g_big_cores.size()

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  auto mr = std::pmr::new_delete_resource();
  octree::AppData app(mr);

  octree::omp::dispatch_multi_stage(BIG_CORES, app, 1, 7);

  std::cout << "Input points:      " << app.n << "\n";
  std::cout << "Unique Morton codes: " << app.m << "\n";
  std::cout << "Total children refs: " << app.total_children << "\n";

  // // peek first 10 ... last 10 morton codes
  // for (int i = 0; i < 10; i++) {
  //   spdlog::debug("morton code {} = {}", i, app.u_morton_codes[i]);
  // }
  // spdlog::debug("...");
  // for (int i = app.n - 10; i < app.n; i++) {
  //   spdlog::debug("morton code {} = {}", i, app.u_morton_codes[i]);
  // }

  // // print number of unique morton codes
  // spdlog::info("number of unique morton codes = {} / {}", app.u_morton_codes.size(), app.n);

  // app.print_radix_tree(app, 10);

  // // peek edge count
  // for (int i = 0; i < 10; i++) {
  //   spdlog::debug("edge count {} = {}", i, app.u_edge_count[i]);
  // }
  // spdlog::debug("...");
  // for (int i = app.m - 10; i < app.m; i++) {
  //   spdlog::debug("edge count {} = {}", i, app.u_edge_count[i]);
  // }

  // // peek offsets
  // for (int i = 0; i < 10; i++) {
  //   spdlog::debug("offsets {} = {}", i, app.u_offsets[i]);
  // }
  // spdlog::debug("...");
  // for (int i = app.m - 10; i < app.m; i++) {
  //   spdlog::debug("offsets {} = {}", i, app.u_offsets[i]);
  // }

  return 0;
}
