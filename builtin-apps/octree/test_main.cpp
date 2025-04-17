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
  octree::AppData appdata(mr);

  octree::omp::dispatch_multi_stage(BIG_CORES, appdata, 1, 7);

  // peek first 10 ... last 10 morton codes
  for (int i = 0; i < 10; i++) {
    spdlog::debug("morton code {} = {}", i, appdata.u_morton_codes[i]);
  }
  spdlog::debug("...");
  for (int i = appdata.n_input - 10; i < appdata.n_input; i++) {
    spdlog::debug("morton code {} = {}", i, appdata.u_morton_codes[i]);
  }

  // print number of unique morton codes
  spdlog::info(
      "number of unique morton codes = {} / {}", appdata.u_morton_codes.size(), appdata.n_input);

  


  return 0;
}
