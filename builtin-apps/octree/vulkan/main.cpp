#include <spdlog/spdlog.h>

#include "../../app.hpp"
#include "dispatchers.hpp"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  octree::vulkan::VulkanDispatcher disp;
  octree::AppData appdata(disp.get_mr());

  disp.dispatch_multi_stage(appdata, 1, 1);

  // print first 100 morton codes
  for (size_t i = 0; i < 100; ++i) {
    std::cout << appdata.u_morton_codes[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
