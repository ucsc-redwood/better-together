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
  EXPECT_EQ(appdata.get_n_unique(), 8262775);
  EXPECT_EQ(appdata.get_n_brt_nodes(), 8262774);
  EXPECT_EQ(appdata.get_n_octree_nodes(), 3841043);

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

  // Check output dimensions
  EXPECT_EQ(appdata.u_morton_keys_s1.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_morton_keys_s1_out.size(), 3840 * 2160);

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

  // Check output dimensions
  EXPECT_EQ(appdata.u_morton_keys_sorted_s2.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_morton_keys_sorted_s2_out.size(), 3840 * 2160);

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

  // Check output dimensions
  EXPECT_EQ(appdata.u_morton_keys_unique_s3.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_morton_keys_unique_s3_out.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_num_selected_out.size(), 1);

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

  // Check output dimensions
  EXPECT_EQ(appdata.u_brt_prefix_n_s4.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_brt_has_leaf_left_s4.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_brt_has_leaf_right_s4.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_brt_left_child_s4.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_brt_parents_s4.size(), 3840 * 2160);

  // Check output dimensions for out buffers
  EXPECT_EQ(appdata.u_brt_prefix_n_s4_out.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_brt_has_leaf_left_s4_out.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_brt_has_leaf_right_s4_out.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_brt_left_child_s4_out.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_brt_parents_s4_out.size(), 3840 * 2160);

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

  // Check output dimensions
  EXPECT_EQ(appdata.u_edge_count_s5.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_edge_count_s5_out.size(), 3840 * 2160);

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

  // Check output dimensions
  EXPECT_EQ(appdata.u_edge_offset_s6.size(), 3840 * 2160);
  EXPECT_EQ(appdata.u_edge_offset_s6_out.size(), 3840 * 2160);

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

  // Check output dimensions
  EXPECT_EQ(appdata.u_oct_children_s7.size(), 3840 * 2160 * 8 * tree::kMemoryRatio);
  EXPECT_EQ(appdata.u_oct_corner_s7.size(), 3840 * 2160 * tree::kMemoryRatio);
  EXPECT_EQ(appdata.u_oct_cell_size_s7.size(), 3840 * 2160 * tree::kMemoryRatio);
  EXPECT_EQ(appdata.u_oct_child_node_mask_s7.size(), 3840 * 2160 * tree::kMemoryRatio);
  EXPECT_EQ(appdata.u_oct_child_leaf_mask_s7.size(), 3840 * 2160 * tree::kMemoryRatio);

  // Check no throw
  EXPECT_NO_THROW(tree::omp::run_stage_7(appdata));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
