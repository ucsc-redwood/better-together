#include "dispatchers.hpp"

#include "../../app.hpp"
#include "../../debug_logger.hpp"

namespace octree::vulkan {

// // push‐constant: one uint + two floats + pad → 16 bytes total
// layout(push_constant) uniform PC {
//   uint n;           // offset 0
//   float min_coord;  // offset 4
//   float range;      // offset 8
//   float pad;        // offset 12 (unused, just for alignment)
// }
// pc;

struct MortonPushConstants {
  uint32_t n;
  float min_coord;
  float range;
  float pad;
};

static_assert(sizeof(MortonPushConstants) == 16);

struct InputSizePushConstantsUnsigned {
  uint32_t n;
};
static_assert(sizeof(InputSizePushConstantsUnsigned) == 4);

// layout(local_size_x = 256) in;

// layout(std430, set = 0, binding = 0) readonly buffer Codes { uint codes[]; };
// layout(std430, set = 0, binding = 1) readonly buffer LeftChild { int left_child[]; };
// layout(std430, set = 0, binding = 2) readonly buffer PrefixLength { int prefix_length[]; };
// layout(std430, set = 0, binding = 3) writeonly buffer EdgeCount { uint edge_count[]; };

// layout(push_constant) uniform PC5 { uint n; }
// pc5;

struct BuildEdgeCountPushConstants {
  uint n;
};
static_assert(sizeof(BuildEdgeCountPushConstants) == 4);

// layout(local_size_x = 256) in;

// layout(std430, set = 0, binding = 0) readonly buffer Codes { uint codes[]; };
// layout(std430, set = 0, binding = 1) buffer Parents { int parents[]; };
// layout(std430, set = 0, binding = 2) buffer LeftChild { int left_child[]; };
// layout(std430, set = 0, binding = 3) buffer HasLeafLeft { uint has_leaf_left[]; };
// layout(std430, set = 0, binding = 4) buffer HasLeafRight { uint has_leaf_right[]; };
// layout(std430, set = 0, binding = 5) buffer PrefixLength { int prefix_length[]; };

// layout(push_constant) uniform PC3 { uint n; }
// pc3;

struct BuildRadixTreePushConstants {
  uint n;
};
static_assert(sizeof(BuildRadixTreePushConstants) == 4);

// layout(local_size_x = 256) in;

// layout(std430, set = 0, binding = 0) readonly buffer Codes { uint codes[]; };
// layout(std430, set = 0, binding = 1) readonly buffer LeftChild { int left_child[]; };
// layout(std430, set = 0, binding = 2) readonly buffer PrefixLength { int prefix_length[]; };
// layout(std430, set = 0, binding = 3) readonly buffer EdgeCount { uint edge_count[]; };
// layout(std430, set = 0, binding = 4) readonly buffer Offsets { uint offsets[]; };
// layout(std430, set = 0, binding = 5) writeonly buffer Children { uint children[]; };

// layout(push_constant) uniform PC7 { uint n; }
// pc7;

struct BuildOctreeNodesPushConstants {
  uint n;
};
static_assert(sizeof(BuildOctreeNodesPushConstants) == 4);

VulkanDispatcher::VulkanDispatcher() : engine(), seq(engine.make_seq()) {
  spdlog::debug("VulkanDispatcher::VulkanDispatcher(), Initializing VulkanDispatcher");

  // *** Stage 1 ***

  auto morton_algo = engine.make_algo("octree_morton")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)
                         ->num_buffers(2)
                         ->push_constant<MortonPushConstants>()
                         ->build();

  cached_algorithms.try_emplace("morton", std::move(morton_algo));

  // *** Stage 2 ***

  auto radixsort_algo =
      engine.make_algo("tmp_single_radixsort_warp" + std::to_string(get_vulkan_warp_size()))
          ->work_group_size(256, 1, 1)
          ->num_sets(1)
          ->num_buffers(2)
          ->push_constant<InputSizePushConstantsUnsigned>()
          ->build();

  cached_algorithms.try_emplace("radixsort", std::move(radixsort_algo));

  // *** Stage 4 ***

  auto build_radix_tree_algo = engine.make_algo("octree_build_radix_tree")
                                   ->work_group_size(256, 1, 1)
                                   ->num_sets(1)
                                   ->num_buffers(6)
                                   ->push_constant<BuildRadixTreePushConstants>()
                                   ->build();

  cached_algorithms.try_emplace("octree_build_radix_tree", std::move(build_radix_tree_algo));

  // *** Stage 5 ***

  auto build_edge_count_algo = engine.make_algo("octree_edge_count")
                                   ->work_group_size(256, 1, 1)
                                   ->num_sets(1)
                                   ->num_buffers(5)
                                   ->push_constant<BuildEdgeCountPushConstants>()
                                   ->build();

  cached_algorithms.try_emplace("octree_edge_count", std::move(build_edge_count_algo));

  // *** Stage 7 ***

  auto build_octree_nodes_algo = engine.make_algo("octree_build_octree_nodes")
                                     ->work_group_size(256, 1, 1)
                                     ->num_sets(1)
                                     ->num_buffers(6)
                                     ->push_constant<BuildOctreeNodesPushConstants>()
                                     ->build();

  cached_algorithms.try_emplace("octree_build_octree_nodes", std::move(build_octree_nodes_algo));
}

// ----------------------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_1(AppData& app) {
  LOG_KERNEL(LogKernelType::kVK, 1, &app);

  auto algo = cached_algorithms.at("morton").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app.u_positions),
                                  engine.get_buffer_info(app.u_morton_codes_alt),
                              });

  algo->update_push_constant(MortonPushConstants{
      .n = static_cast<uint32_t>(app.n),
      .min_coord = kMinCoord,
      .range = kMaxCoord - kMinCoord,
      .pad = 0,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(kiss_vk::div_ceil(app.n, 256)), 1, 1});
  seq->cmd_end();

  seq->reset_fence();
  seq->submit();
  seq->wait_for_fence();

  // // verify that all morton are non-zero
  // for (size_t i = 0; i < app.n; ++i) {
  //   std::cout << app.u_morton_codes_alt[i] << "\n";
  //   if (app.u_morton_codes_alt[i] == 0) {
  //     spdlog::error("Morton code is zero at index {}", i);
  //     exit(1);
  //   }
  // }
}

// ----------------------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_2(AppData& app) {
  LOG_KERNEL(LogKernelType::kVK, 2, &app);

  auto algo = cached_algorithms.at("radixsort").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app.u_morton_codes_alt),
                                  engine.get_buffer_info(app.u_morton_codes),
                              });

  algo->update_push_constant(InputSizePushConstantsUnsigned{
      .n = static_cast<uint32_t>(app.n),
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(), {1, 1, 1});  // Special case: single workgroup
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();

  // Verify that the morton codes are sorted
  for (size_t i = 1; i < app.n; ++i) {
    if (app.u_morton_codes[i - 1] > app.u_morton_codes[i]) {
      spdlog::error("Morton codes are not sorted at index {}", i);
      exit(1);
    }
  }

  exit(0);
}

// ----------------------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_3(AppData& app) {
  LOG_KERNEL(LogKernelType::kVK, 3, &app);

  auto end_it = std::unique(app.u_morton_codes.begin(), app.u_morton_codes.begin() + app.n);
  app.m = std::distance(app.u_morton_codes.begin(), end_it);

  assert(size_t(app.m) <= app.reserved_n);
}

// ----------------------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_4(AppData& app) { LOG_KERNEL(LogKernelType::kVK, 4, &app); }

// ----------------------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_5(AppData& app) { LOG_KERNEL(LogKernelType::kVK, 5, &app); }

// ----------------------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_6(AppData& app) { LOG_KERNEL(LogKernelType::kVK, 6, &app); }

// ----------------------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_7(AppData& app) { LOG_KERNEL(LogKernelType::kVK, 7, &app); }

}  // namespace octree::vulkan
