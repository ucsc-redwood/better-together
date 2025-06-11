#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <memory_resource>
#include <queue>

#include "../../app.hpp"
#include "../appdata.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// Test Vulkan Context Setup
// ----------------------------------------------------------------------------

constexpr int kTestBatchSize = 128;

class CIFARDenseVulkanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up Vulkan context here if needed
    vk_dispatcher = std::make_unique<cifar_dense::vulkan::VulkanDispatcher>();
    mr = vk_dispatcher->get_mr();
  }

  void TearDown() override {
    // Clean up Vulkan context here if needed
    vk_dispatcher.reset();
  }

  std::unique_ptr<cifar_dense::vulkan::VulkanDispatcher> vk_dispatcher;
  std::pmr::memory_resource* mr;
};

// ----------------------------------------------------------------------------
// test Stage 1: Conv1
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage1) {
  cifar_dense::AppData appdata(mr);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv1_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv1_out.d1(), 16);
  EXPECT_EQ(appdata.u_conv1_out.d2(), 32);
  EXPECT_EQ(appdata.u_conv1_out.d3(), 32);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> conv1_out_before(appdata.u_conv1_out.pmr_vec().begin(),
                                            appdata.u_conv1_out.pmr_vec().end());

  // Run stage 1
  EXPECT_NO_THROW(vk_dispatcher->run_stage_1(appdata)) << "Stage 1 should not throw";

  const std::vector<float> conv1_out_after(appdata.u_conv1_out.pmr_vec().begin(),
                                           appdata.u_conv1_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < conv1_out_before.size(); ++i) {
    if (conv1_out_before[i] != conv1_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 2: MaxPool1
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage2) {
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(vk_dispatcher->run_stage_1(appdata));

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool1_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_pool1_out.d1(), 16);
  EXPECT_EQ(appdata.u_pool1_out.d2(), 16);
  EXPECT_EQ(appdata.u_pool1_out.d3(), 16);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> pool1_out_before(appdata.u_pool1_out.pmr_vec().begin(),
                                            appdata.u_pool1_out.pmr_vec().end());

  // Run stage 2
  EXPECT_NO_THROW(vk_dispatcher->run_stage_2(appdata)) << "Stage 2 should not throw";

  const std::vector<float> pool1_out_after(appdata.u_pool1_out.pmr_vec().begin(),
                                           appdata.u_pool1_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < pool1_out_before.size(); ++i) {
    if (pool1_out_before[i] != pool1_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 3: Conv2
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage3) {
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(vk_dispatcher->run_stage_1(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_2(appdata));

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv2_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv2_out.d1(), 32);
  EXPECT_EQ(appdata.u_conv2_out.d2(), 16);
  EXPECT_EQ(appdata.u_conv2_out.d3(), 16);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> conv2_out_before(appdata.u_conv2_out.pmr_vec().begin(),
                                            appdata.u_conv2_out.pmr_vec().end());

  // Run stage 3
  EXPECT_NO_THROW(vk_dispatcher->run_stage_3(appdata)) << "Stage 3 should not throw";

  const std::vector<float> conv2_out_after(appdata.u_conv2_out.pmr_vec().begin(),
                                           appdata.u_conv2_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < conv2_out_before.size(); ++i) {
    if (conv2_out_before[i] != conv2_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 4: MaxPool2
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage4) {
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(vk_dispatcher->run_stage_1(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_2(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_3(appdata));

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool2_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_pool2_out.d1(), 32);
  EXPECT_EQ(appdata.u_pool2_out.d2(), 8);
  EXPECT_EQ(appdata.u_pool2_out.d3(), 8);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> pool2_out_before(appdata.u_pool2_out.pmr_vec().begin(),
                                            appdata.u_pool2_out.pmr_vec().end());

  // Run stage 4
  EXPECT_NO_THROW(vk_dispatcher->run_stage_4(appdata)) << "Stage 4 should not throw";

  const std::vector<float> pool2_out_after(appdata.u_pool2_out.pmr_vec().begin(),
                                           appdata.u_pool2_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < pool2_out_before.size(); ++i) {
    if (pool2_out_before[i] != pool2_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 5: Conv3
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage5) {
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(vk_dispatcher->run_stage_1(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_2(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_3(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_4(appdata));

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv3_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv3_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv3_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv3_out.d3(), 8);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> conv3_out_before(appdata.u_conv3_out.pmr_vec().begin(),
                                            appdata.u_conv3_out.pmr_vec().end());

  // Run stage 5
  EXPECT_NO_THROW(vk_dispatcher->run_stage_5(appdata)) << "Stage 5 should not throw";

  const std::vector<float> conv3_out_after(appdata.u_conv3_out.pmr_vec().begin(),
                                           appdata.u_conv3_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < conv3_out_before.size(); ++i) {
    if (conv3_out_before[i] != conv3_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 6: Conv4
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage6) {
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(vk_dispatcher->run_stage_1(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_2(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_3(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_4(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_5(appdata));

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv4_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv4_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv4_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv4_out.d3(), 8);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> conv4_out_before(appdata.u_conv4_out.pmr_vec().begin(),
                                            appdata.u_conv4_out.pmr_vec().end());

  // Run stage 6
  EXPECT_NO_THROW(vk_dispatcher->run_stage_6(appdata)) << "Stage 6 should not throw";

  const std::vector<float> conv4_out_after(appdata.u_conv4_out.pmr_vec().begin(),
                                           appdata.u_conv4_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < conv4_out_before.size(); ++i) {
    if (conv4_out_before[i] != conv4_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 7: Conv5
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage7) {
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(vk_dispatcher->run_stage_1(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_2(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_3(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_4(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_5(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_6(appdata));

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv5_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv5_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv5_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv5_out.d3(), 8);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> conv5_out_before(appdata.u_conv5_out.pmr_vec().begin(),
                                            appdata.u_conv5_out.pmr_vec().end());

  // Run stage 7
  EXPECT_NO_THROW(vk_dispatcher->run_stage_7(appdata)) << "Stage 7 should not throw";

  const std::vector<float> conv5_out_after(appdata.u_conv5_out.pmr_vec().begin(),
                                           appdata.u_conv5_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < conv5_out_before.size(); ++i) {
    if (conv5_out_before[i] != conv5_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 8: MaxPool3
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage8) {
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(vk_dispatcher->run_stage_1(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_2(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_3(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_4(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_5(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_6(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_7(appdata));

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool3_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_pool3_out.d1(), 64);
  EXPECT_EQ(appdata.u_pool3_out.d2(), 4);
  EXPECT_EQ(appdata.u_pool3_out.d3(), 4);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> pool3_out_before(appdata.u_pool3_out.pmr_vec().begin(),
                                            appdata.u_pool3_out.pmr_vec().end());

  // Run stage 8
  EXPECT_NO_THROW(vk_dispatcher->run_stage_8(appdata)) << "Stage 8 should not throw";

  const std::vector<float> pool3_out_after(appdata.u_pool3_out.pmr_vec().begin(),
                                           appdata.u_pool3_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < pool3_out_before.size(); ++i) {
    if (pool3_out_before[i] != pool3_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 9: Linear
// ----------------------------------------------------------------------------

TEST_F(CIFARDenseVulkanTest, Stage9) {
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(vk_dispatcher->run_stage_1(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_2(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_3(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_4(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_5(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_6(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_7(appdata));
  EXPECT_NO_THROW(vk_dispatcher->run_stage_8(appdata));

  // Check output dimensions
  EXPECT_EQ(appdata.u_linear_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_linear_out.d1(), 10);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> linear_out_before(appdata.u_linear_out.pmr_vec().begin(),
                                             appdata.u_linear_out.pmr_vec().end());

  // Run stage 9
  EXPECT_NO_THROW(vk_dispatcher->run_stage_9(appdata)) << "Stage 9 should not throw";

  const std::vector<float> linear_out_after(appdata.u_linear_out.pmr_vec().begin(),
                                            appdata.u_linear_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < linear_out_before.size(); ++i) {
    if (linear_out_before[i] != linear_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// Test Queue environment
// ----------------------------------------------------------------------------

TEST(VkQueueTest, Basic) {
  auto vk_dispatcher = std::make_unique<cifar_dense::vulkan::VulkanDispatcher>();
  auto mr = vk_dispatcher->get_mr();

  std::vector<std::shared_ptr<cifar_dense::AppData>> appdatas;
  appdatas.reserve(10);
  for (int i = 0; i < 10; i++) {
    appdatas.push_back(std::make_shared<cifar_dense::AppData>(mr));
  }

  std::queue<std::shared_ptr<cifar_dense::AppData>> queue;
  for (auto& appdata : appdatas) {
    queue.push(appdata);
  }

  while (!queue.empty()) {
    auto appdata = queue.front();
    queue.pop();

    EXPECT_NO_THROW(vk_dispatcher->run_stage_1(*appdata));
    EXPECT_NO_THROW(vk_dispatcher->run_stage_2(*appdata));
    EXPECT_NO_THROW(vk_dispatcher->run_stage_3(*appdata));
    EXPECT_NO_THROW(vk_dispatcher->run_stage_4(*appdata));
    EXPECT_NO_THROW(vk_dispatcher->run_stage_5(*appdata));
    EXPECT_NO_THROW(vk_dispatcher->run_stage_6(*appdata));
    EXPECT_NO_THROW(vk_dispatcher->run_stage_7(*appdata));
    EXPECT_NO_THROW(vk_dispatcher->run_stage_8(*appdata));
    EXPECT_NO_THROW(vk_dispatcher->run_stage_9(*appdata));
  }

  EXPECT_TRUE(queue.empty());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  return RUN_ALL_TESTS();
}
