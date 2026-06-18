#include <gtest/gtest.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "platform/registry/device_registry.hpp"
#include "apps/tree/tree_diff_oracle.hpp"
#include "dispatchers.hpp"
#include "vk_appdata.hpp"

// ----------------------------------------------------------------------------
// Tree × Vulkan differential oracle. The Vulkan backend operates on
// VkAppData_Safe (a subclass of SafeAppData), so the OMP golden and _out buffers
// the checks read are inherited. Same BT_DECLARE expansion as the OMP/CUDA
// suites, only the Runner + its AppData type differ. Run on an integrated-GPU
// box / Jetson / phone (the engine hard-selects an integrated GPU).
// ----------------------------------------------------------------------------

namespace {
struct VulkanTreeRunner {
  using AppData = tree::vulkan::VkAppData_Safe;
  tree::vulkan::VulkanDispatcher disp;
  // The engine constructor selects an integrated GPU and throws if none exists;
  // only run `ctest -L vulkan` on a target that has one.
  static bool Available() { return true; }
  kiss_vk::VulkanMemoryResource::memory_resource* Mr() { return disp.get_mr(); }
  void RunStage(AppData& a, int stage) { disp.dispatch_stage(a, stage); }
};
}  // namespace

BT_DECLARE_TREE_DIFF_TESTS(TreeDiffVulkan, VulkanTreeRunner)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  // Host-side golden uses OMP; pin single-threaded so the radix-tree/octree
  // reference is deterministic (matches the OMP/CUDA suites).
  omp_set_num_threads(1);
  return RUN_ALL_TESTS();
}
