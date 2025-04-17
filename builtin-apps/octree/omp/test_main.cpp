#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <memory>

#include "../../app.hpp"
#include "../appdata.hpp"
#include "dispatchers.hpp"

// Setup helper to initialize the app data
std::unique_ptr<octree::AppData> setupAppData(size_t n_input = 1000) {
  auto mr = std::pmr::new_delete_resource();
  return std::make_unique<octree::AppData>(mr, n_input);
}

// Test fixture for octree dispatcher tests
class OctreeDispatcherTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize app data with a smaller number of points for faster tests
    app_data = setupAppData(1000);
  }

  std::unique_ptr<octree::AppData> app_data;
};

// Test each stage individually
TEST_F(OctreeDispatcherTest, Stage1_ComputeMortonCodes) {
  octree::omp::dispatch_multi_stage(BIG_CORES, *app_data, 1, 1);

  // Verify some basic expectations after running stage 1
  // Morton codes should be computed for all input points
  bool all_non_zero = true;
  for (size_t i = 0; i < app_data->n; ++i) {
    if (app_data->u_morton_codes[i] == 0) {
      all_non_zero = false;
      break;
    }
  }
  EXPECT_TRUE(all_non_zero) << "Some morton codes were not computed";
}

TEST_F(OctreeDispatcherTest, Stage2_SortMortonCodes) {
  // Run stage 1 first to generate morton codes
  octree::omp::dispatch_multi_stage(BIG_CORES, *app_data, 1, 1);

  // Then run stage 2 to sort them
  octree::omp::dispatch_multi_stage(BIG_CORES, *app_data, 2, 2);

  // Verify that the morton codes are sorted
  for (size_t i = 1; i < app_data->n; ++i) {
    EXPECT_LE(app_data->u_morton_codes[i - 1], app_data->u_morton_codes[i])
        << "Morton codes are not sorted at index " << i;
  }
}

TEST_F(OctreeDispatcherTest, Stage3_UniqueMortonCodes) {
  // Run stages 1-2 first
  octree::omp::dispatch_multi_stage(BIG_CORES, *app_data, 1, 2);

  // Then run stage 3 to find unique morton codes
  octree::omp::dispatch_multi_stage(BIG_CORES, *app_data, 3, 3);

  // Verify that m is now set and less than or equal to n
  EXPECT_LE(app_data->m, app_data->n) << "Number of unique morton codes exceeds input size";
  EXPECT_NE(app_data->m, std::numeric_limits<size_t>::max()) << "m was not updated";

  // Verify that unique morton codes are sorted and unique
  for (size_t i = 1; i < app_data->m; ++i) {
    EXPECT_LT(app_data->u_morton_codes[i - 1], app_data->u_morton_codes[i])
        << "Unique morton codes are not strictly sorted at index " << i;
  }
}

TEST_F(OctreeDispatcherTest, Stage4_BuildRadixTree) {
  // Run stages 1-3 first
  octree::omp::dispatch_multi_stage(BIG_CORES, *app_data, 1, 3);

  // Then run stage 4 to build the radix tree
  octree::omp::dispatch_multi_stage(BIG_CORES, *app_data, 4, 4);

  // Basic validation of the radix tree structure
  // For each internal node, check some basic properties
  for (size_t i = 0; i < app_data->m; ++i) {
    // If it's not a leaf, it should have valid child indices
    if (!app_data->u_has_leaf_left[i]) {
      int child_idx = app_data->u_left_child[i];
      EXPECT_GE(child_idx, 0) << "Invalid left child index at node " << i;
      EXPECT_LT(child_idx, static_cast<int>(app_data->m))
          << "Left child index out of range at node " << i;
    }

    // Prefix length should be less than or equal to 32 (for 32-bit morton codes)
    EXPECT_LE(app_data->u_prefix_length[i], 32) << "Invalid prefix length at node " << i;
  }
}

TEST_F(OctreeDispatcherTest, Stage5_EdgeCount) {
  // Run stages 1-4 first
  octree::omp::dispatch_multi_stage(BIG_CORES, *app_data, 1, 4);

  // Then run stage 5 to compute edge counts
  octree::omp::dispatch_multi_stage(BIG_CORES, *app_data, 5, 5);

  // Verify edge counts are valid (between 0 and 8 for octree)
  for (size_t i = 0; i < app_data->m; ++i) {
    EXPECT_GE(app_data->u_edge_count[i], 0) << "Negative edge count at node " << i;
    EXPECT_LE(app_data->u_edge_count[i], 8) << "Edge count exceeds octant limit at node " << i;
  }
}

TEST_F(OctreeDispatcherTest, Stage6_ComputeOffsets) {
  // Run stages 1-5 first
  octree::omp::dispatch_multi_stage(BIG_CORES, *app_data, 1, 5);

  // Then run stage 6 to compute offsets
  octree::omp::dispatch_multi_stage(BIG_CORES, *app_data, 6, 6);

  // Verify that offsets are correct
  EXPECT_EQ(app_data->u_offsets[0], 0) << "First offset should be 0";

  // Each offset should be the sum of previous edge counts
  for (size_t i = 1; i < app_data->m; ++i) {
    EXPECT_EQ(app_data->u_offsets[i], app_data->u_offsets[i - 1] + app_data->u_edge_count[i - 1])
        << "Offset calculation is incorrect at index " << i;
  }

  // Verify total_children is computed correctly
  size_t expected_total =
      app_data->u_offsets[app_data->m - 1] + app_data->u_edge_count[app_data->m - 1];
  EXPECT_EQ(app_data->total_children, expected_total) << "total_children calculation is incorrect";
}

TEST_F(OctreeDispatcherTest, Stage7_BuildOctreeNodes) {
  // Run stages 1-6 first
  octree::omp::dispatch_multi_stage(BIG_CORES, *app_data, 1, 6);

  // Then run stage 7 to build octree nodes
  octree::omp::dispatch_multi_stage(BIG_CORES, *app_data, 7, 7);

  // Verify that children indices are valid
  for (size_t i = 0; i < app_data->m; ++i) {
    int offset = app_data->u_offsets[i];
    int count = app_data->u_edge_count[i];

    for (int j = 0; j < count; ++j) {
      int child_idx = app_data->u_children[offset + j];
      EXPECT_GE(child_idx, 0) << "Invalid child index at node " << i << ", child " << j;
      EXPECT_LT(child_idx, static_cast<int>(app_data->m))
          << "Child index out of range at node " << i << ", child " << j;
    }
  }
}

// Test all stages together
TEST_F(OctreeDispatcherTest, AllStages) {
  // Run all stages in one dispatch call
  octree::omp::dispatch_multi_stage(BIG_CORES, *app_data, 1, 7);

  // Verify final state
  EXPECT_NE(app_data->m, std::numeric_limits<size_t>::max()) << "m was not updated";
  EXPECT_NE(app_data->total_children, std::numeric_limits<size_t>::max())
      << "total_children was not updated";

  // Print some statistics about the octree construction
  std::cout << "Input points:      " << app_data->n << "\n";
  std::cout << "Unique Morton codes: " << app_data->m << "\n";
  std::cout << "Total children refs: " << app_data->total_children << "\n";
}

// Test dispatching with different core configurations
TEST_F(OctreeDispatcherTest, DifferentCoreConfigurations) {
  // Try with little cores
  if (!g_little_cores.empty()) {
    EXPECT_NO_THROW(octree::omp::dispatch_multi_stage(LITTLE_CORES, *app_data, 1, 7))
        << "Failed to dispatch with little cores";
  }

  // Try with medium cores
  if (!g_medium_cores.empty()) {
    EXPECT_NO_THROW(octree::omp::dispatch_multi_stage(MEDIUM_CORES, *app_data, 1, 7))
        << "Failed to dispatch with medium cores";
  }

  // Already tested with big cores in other tests
}

int main(int argc, char** argv) {
  // Initialize Google Test
  ::testing::InitGoogleTest(&argc, argv);

  // Parse command-line arguments
  parse_args(argc, argv);

  // Set logging level
  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  // Run the tests
  return RUN_ALL_TESTS();
}
