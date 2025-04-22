#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <memory>

#include "../../app.hpp"
#include "../safe_tree_appdata.hpp"
#include "builtin-apps/tree/tree_appdata.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// test appdata initialization
// ----------------------------------------------------------------------------

TEST(AppdataTest, Initialization) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  EXPECT_EQ(appdata.get_n_input(), tree::kDefaultInputSize);

  EXPECT_LT(appdata.get_n_octree_nodes() / appdata.get_n_input(), tree::kMemoryRatio);
}

// ----------------------------------------------------------------------------
// test Stage 1
// ----------------------------------------------------------------------------

TEST(Stage1Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // Run stage 1
  tree::omp::run_stage_1(appdata);

  // Check no throw
  EXPECT_NO_THROW(tree::omp::run_stage_1(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 2
// ----------------------------------------------------------------------------

TEST(Stage2Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // Run stage 1 first
  tree::omp::run_stage_1(appdata);

  // Run stage 2
  tree::omp::run_stage_2(appdata);

  // Check no throw
  EXPECT_NO_THROW(tree::omp::run_stage_2(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 3
// ----------------------------------------------------------------------------

TEST(Stage3Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // Run previous stages
  tree::omp::run_stage_1(appdata);
  tree::omp::run_stage_2(appdata);

  // Run stage 3
  tree::omp::run_stage_3(appdata);

  // Check no throw
  EXPECT_NO_THROW(tree::omp::run_stage_3(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 4
// ----------------------------------------------------------------------------

TEST(Stage4Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // Run previous stages
  tree::omp::run_stage_1(appdata);
  tree::omp::run_stage_2(appdata);
  tree::omp::run_stage_3(appdata);

  // Run stage 4
  tree::omp::run_stage_4(appdata);

  // Check no throw
  EXPECT_NO_THROW(tree::omp::run_stage_4(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 5
// ----------------------------------------------------------------------------

TEST(Stage5Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // Run previous stages
  tree::omp::run_stage_1(appdata);
  tree::omp::run_stage_2(appdata);
  tree::omp::run_stage_3(appdata);
  tree::omp::run_stage_4(appdata);

  // Run stage 5
  tree::omp::run_stage_5(appdata);

  // Check no throw
  EXPECT_NO_THROW(tree::omp::run_stage_5(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 6
// ----------------------------------------------------------------------------

TEST(Stage6Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // Run previous stages
  tree::omp::run_stage_1(appdata);
  tree::omp::run_stage_2(appdata);
  tree::omp::run_stage_3(appdata);
  tree::omp::run_stage_4(appdata);
  tree::omp::run_stage_5(appdata);

  // Run stage 6
  tree::omp::run_stage_6(appdata);

  // Check no throw
  EXPECT_NO_THROW(tree::omp::run_stage_6(appdata));
}

// ----------------------------------------------------------------------------
// test Stage 7
// ----------------------------------------------------------------------------

TEST(Stage7Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // Run previous stages
  tree::omp::run_stage_1(appdata);
  tree::omp::run_stage_2(appdata);
  tree::omp::run_stage_3(appdata);
  tree::omp::run_stage_4(appdata);
  tree::omp::run_stage_5(appdata);
  tree::omp::run_stage_6(appdata);

  // Run stage 7
  tree::omp::run_stage_7(appdata);

  // Check no throw
  EXPECT_NO_THROW(tree::omp::run_stage_7(appdata));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Parse command-line arguments
  parse_args(argc, argv);

  // Set logging level to off
  spdlog::set_level(spdlog::level::off);

  return RUN_ALL_TESTS();
}
