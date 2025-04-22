#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "../../app.hpp"
#include "../safe_tree_appdata.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// test Stage 1
// ----------------------------------------------------------------------------

TEST(Stage1Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  const std::vector<float> morton_before(appdata.u_morton_keys_s1_out.begin(),
                                         appdata.u_morton_keys_s1_out.end());

  EXPECT_NO_THROW(tree::omp::dispatch_stage(appdata, 1)) << "Stage 1 should not throw";

  const std::vector<float> morton_after(appdata.u_morton_keys_s1_out.begin(),
                                        appdata.u_morton_keys_s1_out.end());

  const bool is_different = !std::ranges::equal(morton_before, morton_after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 2
// ----------------------------------------------------------------------------

TEST(Stage2Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // Run previous stages
  tree::omp::dispatch_stage(appdata, 1);

  EXPECT_NO_THROW(tree::omp::dispatch_stage(appdata, 2)) << "Stage 2 should not throw";
}

// ----------------------------------------------------------------------------
// test Stage 3
// ----------------------------------------------------------------------------

TEST(Stage3Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // Run previous stages
  tree::omp::dispatch_multi_stage(appdata, 1, 2);

  EXPECT_NO_THROW(tree::omp::dispatch_stage(appdata, 3)) << "Stage 3 should not throw";
}

// ----------------------------------------------------------------------------
// test Stage 4
// ----------------------------------------------------------------------------

TEST(Stage4Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // Run previous stages
  tree::omp::dispatch_multi_stage(appdata, 1, 3);

  EXPECT_NO_THROW(tree::omp::dispatch_stage(appdata, 4)) << "Stage 4 should not throw";
}

// ----------------------------------------------------------------------------
// test Stage 5
// ----------------------------------------------------------------------------

TEST(Stage5Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // Run previous stages
  tree::omp::dispatch_multi_stage(appdata, 1, 4);

  EXPECT_NO_THROW(tree::omp::dispatch_stage(appdata, 5)) << "Stage 5 should not throw";
}

// ----------------------------------------------------------------------------
// test Stage 6
// ----------------------------------------------------------------------------

TEST(Stage6Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // Run previous stages
  tree::omp::dispatch_multi_stage(appdata, 1, 5);

  EXPECT_NO_THROW(tree::omp::dispatch_stage(appdata, 6)) << "Stage 6 should not throw";
}

// ----------------------------------------------------------------------------
// test Stage 7
// ----------------------------------------------------------------------------

TEST(Stage7Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  tree::omp::dispatch_multi_stage(appdata, 1, 6);

  EXPECT_NO_THROW(tree::omp::dispatch_stage(appdata, 7)) << "Stage 7 should not throw";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  return RUN_ALL_TESTS();
}
