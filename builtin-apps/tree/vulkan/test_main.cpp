#include <gtest/gtest.h>

#include <cstdint>
#include <queue>

#include "../../app.hpp"
#include "../omp/dispatchers.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// test Stage 1
// ----------------------------------------------------------------------------

TEST(Stage1Test, Basic) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  const std::vector<float> before(appdata.u_morton_keys_s1_out.begin(),
                                  appdata.u_morton_keys_s1_out.end());

  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1)) << "Stage 1 should not throw";

  const std::vector<float> after(appdata.u_morton_keys_s1_out.begin(),
                                 appdata.u_morton_keys_s1_out.end());

  const bool is_different = !std::ranges::equal(before, after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 2
// ----------------------------------------------------------------------------

TEST(Stage2Test, Basic) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  // Run stage 1 first
  disp.dispatch_stage(appdata, 1);

  const std::vector<uint32_t> before(appdata.u_morton_keys_sorted_s2_out.begin(),
                                     appdata.u_morton_keys_sorted_s2_out.end());

  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 2)) << "Stage 2 should not throw";

  const std::vector<uint32_t> after(appdata.u_morton_keys_sorted_s2_out.begin(),
                                    appdata.u_morton_keys_sorted_s2_out.end());

  const bool is_different = !std::ranges::equal(before, after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 3
// ----------------------------------------------------------------------------

TEST(Stage3Test, Basic) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  // Run previous stages
  disp.dispatch_multi_stage(appdata, 1, 2);

  const std::vector<uint32_t> before(appdata.u_morton_keys_unique_s3_out.begin(),
                                     appdata.u_morton_keys_unique_s3_out.end());

  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 3)) << "Stage 3 should not throw";

  const std::vector<uint32_t> after(appdata.u_morton_keys_unique_s3_out.begin(),
                                    appdata.u_morton_keys_unique_s3_out.end());

  const bool is_different = !std::ranges::equal(before, after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 4
// ----------------------------------------------------------------------------

TEST(Stage4Test, Basic) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  // Run previous stages
  disp.dispatch_multi_stage(appdata, 1, 3);

  const std::vector<uint8_t> before(appdata.u_brt_prefix_n_s4_out.begin(),
                                    appdata.u_brt_prefix_n_s4_out.end());

  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 4)) << "Stage 4 should not throw";

  const std::vector<uint8_t> after(appdata.u_brt_prefix_n_s4_out.begin(),
                                   appdata.u_brt_prefix_n_s4_out.end());

  const bool is_different = !std::ranges::equal(before, after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 5
// ----------------------------------------------------------------------------

TEST(Stage5Test, Basic) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  // Run previous stages
  disp.dispatch_multi_stage(appdata, 1, 4);

  const std::vector<int32_t> before(appdata.u_edge_count_s5_out.begin(),
                                    appdata.u_edge_count_s5_out.end());

  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 5)) << "Stage 5 should not throw";

  const std::vector<int32_t> after(appdata.u_edge_count_s5_out.begin(),
                                   appdata.u_edge_count_s5_out.end());

  const bool is_different = !std::ranges::equal(before, after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 6
// ----------------------------------------------------------------------------

TEST(Stage6Test, Basic) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  // Run previous stages
  disp.dispatch_multi_stage(appdata, 1, 5);

  const std::vector<int32_t> before(appdata.u_edge_offset_s6_out.begin(),
                                    appdata.u_edge_offset_s6_out.end());

  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 6)) << "Stage 6 should not throw";

  const std::vector<int32_t> after(appdata.u_edge_offset_s6_out.begin(),
                                   appdata.u_edge_offset_s6_out.end());

  const bool is_different = !std::ranges::equal(before, after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}
// ----------------------------------------------------------------------------
// test Stage 7
// ----------------------------------------------------------------------------

TEST(Stage7Test, Basic) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  // Run previous stages
  disp.dispatch_multi_stage(appdata, 1, 6);

  const std::vector<float> before(appdata.u_oct_cell_size_s7_out.begin(),
                                  appdata.u_oct_cell_size_s7_out.end());

  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 7)) << "Stage 7 should not throw";

  const std::vector<float> after(appdata.u_oct_cell_size_s7_out.begin(),
                                 appdata.u_oct_cell_size_s7_out.end());

  const bool is_different = !std::ranges::equal(before, after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// Test Mixing Omp and Vulkan
// ----------------------------------------------------------------------------

TEST(MixingTest, VulkanThenOmp) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1));
  EXPECT_NO_THROW(tree::omp::run_stage_2(appdata));
}

TEST(MixingTest, OmpThenVulkan) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  EXPECT_NO_THROW(tree::omp::run_stage_1(appdata));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 2));
}

TEST(MixingTest, MultipleStages) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  // Run first 3 stages with CUDA
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 2));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 3));

  // Run next 2 stages with OMP
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 4));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 5));

  // Run final stages with CUDA
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 6));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 7));
}

TEST(MixingTest, AlternatingStages) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  // Alternate between CUDA and OMP for each stage
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1));
  EXPECT_NO_THROW(tree::omp::dispatch_stage(appdata, 2));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 3));
  EXPECT_NO_THROW(tree::omp::dispatch_stage(appdata, 4));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 5));
  EXPECT_NO_THROW(tree::omp::dispatch_stage(appdata, 6));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 7));
}

TEST(MixingTest, MixedBatch) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  // Run first half with CUDA
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 2));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 3));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 4));

  // Run second half with OMP
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 5));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 6));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 7));
}

// ----------------------------------------------------------------------------
// Test Queue environment
// ----------------------------------------------------------------------------

TEST(QueueTest, Basic) {
  tree::vulkan::VulkanDispatcher disp;

  std::vector<std::shared_ptr<tree::vulkan::VkAppData_Safe>> appdatas;
  appdatas.reserve(10);
  for (int i = 0; i < 10; i++) {
    appdatas.push_back(std::make_shared<tree::vulkan::VkAppData_Safe>(disp.get_mr()));
  }

  std::queue<std::shared_ptr<tree::vulkan::VkAppData_Safe>> queue;
  for (auto& appdata : appdatas) {
    queue.push(appdata);
  }

  while (!queue.empty()) {
    auto appdata = queue.front();
    queue.pop();

    EXPECT_NO_THROW(disp.dispatch_multi_stage(*appdata, 1, 7));
  }

  EXPECT_TRUE(queue.empty());
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
