#include <gtest/gtest.h>

#include "../../app.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// test Stage 1
// ----------------------------------------------------------------------------

TEST(Stage1Test, Basic) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  const std::vector<float> morton_before(appdata.u_morton_keys_s1_out.begin(),
                                         appdata.u_morton_keys_s1_out.end());

  EXPECT_NO_THROW(disp.run_stage_1(appdata)) << "Stage 1 should not throw";

  const std::vector<float> morton_after(appdata.u_morton_keys_s1_out.begin(),
                                        appdata.u_morton_keys_s1_out.end());

  const bool is_different = !std::ranges::equal(morton_before, morton_after);

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

  EXPECT_NO_THROW(disp.run_stage_2(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 3
// ----------------------------------------------------------------------------

TEST(Stage3Test, Basic) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  // Run previous stages
  disp.run_stage_1(appdata);
  disp.run_stage_2(appdata);

  // Run stage 3
  EXPECT_NO_THROW(disp.run_stage_3(appdata));

  // Check no throw
  EXPECT_NO_THROW(disp.run_stage_3(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 4
// ----------------------------------------------------------------------------

TEST(Stage4Test, Basic) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  // Run previous stages
  disp.run_stage_1(appdata);
  disp.run_stage_2(appdata);
  disp.run_stage_3(appdata);

  // Run stage 4
  EXPECT_NO_THROW(disp.run_stage_4(appdata));

  // Check no throw
  EXPECT_NO_THROW(disp.run_stage_4(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 5
// ----------------------------------------------------------------------------

TEST(Stage5Test, Basic) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  // Run previous stages
  disp.run_stage_1(appdata);
  disp.run_stage_2(appdata);
  disp.run_stage_3(appdata);
  disp.run_stage_4(appdata);

  // Run stage 5
  EXPECT_NO_THROW(disp.run_stage_5(appdata));

  // Check no throw
  EXPECT_NO_THROW(disp.run_stage_5(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 6
// ----------------------------------------------------------------------------

TEST(Stage6Test, Basic) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  // Run previous stages
  disp.run_stage_1(appdata);
  disp.run_stage_2(appdata);
  disp.run_stage_3(appdata);
  disp.run_stage_4(appdata);
  disp.run_stage_5(appdata);

  // Run stage 6
  EXPECT_NO_THROW(disp.run_stage_6(appdata));

  // Check no throw
  EXPECT_NO_THROW(disp.run_stage_6(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 7
// ----------------------------------------------------------------------------

TEST(Stage7Test, Basic) {
  tree::vulkan::VulkanDispatcher disp;
  tree::vulkan::VkAppData_Safe appdata(disp.get_mr());

  // Run previous stages
  disp.run_stage_1(appdata);
  disp.run_stage_2(appdata);
  disp.run_stage_3(appdata);
  disp.run_stage_4(appdata);
  disp.run_stage_5(appdata);
  disp.run_stage_6(appdata);

  // Run stage 7
  EXPECT_NO_THROW(disp.run_stage_7(appdata));

  // Check no throw
  EXPECT_NO_THROW(disp.run_stage_7(appdata));
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
