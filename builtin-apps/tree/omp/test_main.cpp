#include <gtest/gtest.h>

#include "../safe_tree_appdata.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// test appdata initialization
// ----------------------------------------------------------------------------

TEST(AppdataTest, Initialization) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  EXPECT_EQ(appdata.get_n_input(), 3840 * 2160);
  EXPECT_EQ(appdata.get_n_unique(), 3840 * 2160);
  EXPECT_EQ(appdata.get_n_brt_nodes(), 3840 * 2160 - 1);
  EXPECT_EQ(appdata.get_n_octree_nodes(), 3840 * 2160);
}

// ----------------------------------------------------------------------------
// test Stage 1
// ----------------------------------------------------------------------------

TEST(Stage1Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  tree::SafeAppData appdata(mr);

  // Run stage 1
  tree::omp::run_stage_1(appdata);

  // Check output dimensions
  EXPECT_EQ(appdata.u_morton_keys_s1.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_morton_keys_s1_out.size(), 3840 * 2160);

  // Check no throw
  EXPECT_NO_THROW(tree::omp::run_stage_1(appdata));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
