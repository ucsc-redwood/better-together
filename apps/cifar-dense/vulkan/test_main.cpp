#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <memory_resource>

#include "platform/registry/device_registry.hpp"
#include "apps/cifar-dense/cifar_dense_diff_oracle.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// cifar-dense × Vulkan differential oracle. Vulkan uses the same AppData as
// OMP/CUDA, so the harness is identical — only the Runner differs. Each stage is
// checked against the independent double-precision conv/pool/linear reference.
// Run on an integrated-GPU box / Jetson / phone.
// ----------------------------------------------------------------------------

namespace {
struct VulkanRunner {
  cifar_dense::vulkan::VulkanDispatcher disp;
  static bool Available() { return kiss_vk::has_integrated_gpu(); }  // skip if no iGPU
  std::pmr::memory_resource* Mr() { return disp.get_mr(); }
  void RunStage(cifar_dense::AppData& a, int stage) { disp.dispatch_stage(a, stage); }
};
}  // namespace

BT_DECLARE_CIFAR_DENSE_DIFF_TESTS(CifarDenseDiffVulkan, VulkanRunner)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
