#include <gtest/gtest.h>

#include "../sparse_appdata.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// test appdata initialization
// ----------------------------------------------------------------------------

TEST(AppdataTest, Initialization) {
  auto mr = std::pmr::new_delete_resource();
  cifar_sparse::v2::AppData appdata(mr);

  EXPECT_EQ(appdata.conv1_sparse.rows, 16);
  EXPECT_EQ(appdata.conv1_sparse.cols, 27);
}

// ----------------------------------------------------------------------------
// test Stage 1
// ----------------------------------------------------------------------------

TEST(Stage1Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_sparse::v2::AppData appdata(mr);

  // Run stage 1
  cifar_sparse::omp::v2::run_stage_1(appdata);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv1_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv1_out.d1(), 16);
  EXPECT_EQ(appdata.u_conv1_out.d2(), 32);
  EXPECT_EQ(appdata.u_conv1_out.d3(), 32);

  // Check no throw
  EXPECT_NO_THROW(cifar_sparse::omp::v2::run_stage_1(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 2
// ----------------------------------------------------------------------------

TEST(Stage2Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_sparse::v2::AppData appdata(mr);

  // Run stage 2
  cifar_sparse::omp::v2::run_stage_2(appdata);

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool1_out.d0(), 128);
  EXPECT_EQ(appdata.u_pool1_out.d1(), 16);
  EXPECT_EQ(appdata.u_pool1_out.d2(), 16);
  EXPECT_EQ(appdata.u_pool1_out.d3(), 16);

  // Check no throw
  EXPECT_NO_THROW(cifar_sparse::omp::v2::run_stage_2(appdata));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
