#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "../../app.hpp"
#include "../appdata.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// test Stage 1
// ----------------------------------------------------------------------------

TEST(Stage1Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv1_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv1_out.d1(), 16);
  EXPECT_EQ(appdata.u_conv1_out.d2(), 32);
  EXPECT_EQ(appdata.u_conv1_out.d3(), 32);

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 1));
}

// ----------------------------------------------------------------------------
// test Stage 2
// ----------------------------------------------------------------------------

TEST(Stage2Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run stage 2
  cifar_dense::omp::dispatch_stage(appdata, 1);

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool1_out.d0(), 128);
  EXPECT_EQ(appdata.u_pool1_out.d1(), 16);
  EXPECT_EQ(appdata.u_pool1_out.d2(), 16);
  EXPECT_EQ(appdata.u_pool1_out.d3(), 16);

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 2));
}

// ----------------------------------------------------------------------------
// test Stage 3
// ----------------------------------------------------------------------------

TEST(Stage3Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run stage 3
  cifar_dense::omp::dispatch_multi_stage(appdata, 1, 2);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv2_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv2_out.d1(), 32);
  EXPECT_EQ(appdata.u_conv2_out.d2(), 16);
  EXPECT_EQ(appdata.u_conv2_out.d3(), 16);

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 3));
}

// ----------------------------------------------------------------------------
// test Stage 4
// ----------------------------------------------------------------------------

TEST(Stage4Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run stage 4
  cifar_dense::omp::dispatch_multi_stage(appdata, 1, 3);

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool2_out.d0(), 128);
  EXPECT_EQ(appdata.u_pool2_out.d1(), 32);
  EXPECT_EQ(appdata.u_pool2_out.d2(), 8);
  EXPECT_EQ(appdata.u_pool2_out.d3(), 8);

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 4));
}

// ----------------------------------------------------------------------------
// test Stage 5
// ----------------------------------------------------------------------------

TEST(Stage5Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run stage 5
  cifar_dense::omp::dispatch_multi_stage(appdata, 1, 4);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv3_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv3_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv3_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv3_out.d3(), 8);

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 5));
}

// ----------------------------------------------------------------------------
// test Stage 6
// ----------------------------------------------------------------------------

TEST(Stage6Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run stage 6
  cifar_dense::omp::dispatch_multi_stage(appdata, 1, 5);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv4_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv4_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv4_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv4_out.d3(), 8);

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 6));
}

// ----------------------------------------------------------------------------
// test Stage 7
// ----------------------------------------------------------------------------

TEST(Stage7Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run stage 7
  cifar_dense::omp::dispatch_multi_stage(appdata, 1, 6);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv5_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv5_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv5_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv5_out.d3(), 8);

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 7));
}

// ----------------------------------------------------------------------------
// test Stage 8
// ----------------------------------------------------------------------------

TEST(Stage8Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run stage 8
  cifar_dense::omp::dispatch_multi_stage(appdata, 1, 7);

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool3_out.d0(), 128);
  EXPECT_EQ(appdata.u_pool3_out.d1(), 64);
  EXPECT_EQ(appdata.u_pool3_out.d2(), 4);
  EXPECT_EQ(appdata.u_pool3_out.d3(), 4);

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 8));
}

// ----------------------------------------------------------------------------
// test Stage 9
// ----------------------------------------------------------------------------

TEST(Stage9Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  cifar_dense::omp::dispatch_multi_stage(appdata, 1, 8);

  // Check output dimensions
  EXPECT_EQ(appdata.u_linear_out.d0(), 128);
  EXPECT_EQ(appdata.u_linear_out.d1(), 10);

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 9));
}

int main(int argc, char** argv) {
  // Initialize Google Test
  ::testing::InitGoogleTest(&argc, argv);

  // Parse command-line arguments
  parse_args(argc, argv);

  // Set logging level to off
  spdlog::set_level(spdlog::level::off);

  // Run the tests
  return RUN_ALL_TESTS();
}
