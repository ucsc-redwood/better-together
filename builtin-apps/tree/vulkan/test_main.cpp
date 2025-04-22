#include <gtest/gtest.h>

#include <cstdint>

#include "../../app.hpp"
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

int main(int argc, char **argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
