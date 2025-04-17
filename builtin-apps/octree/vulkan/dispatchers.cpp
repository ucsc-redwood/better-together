#include "dispatchers.hpp"

#include "../../debug_logger.hpp"

namespace octree::vulkan {

// layout(local_size_x = 256) in;

// layout(std430, set = 0, binding = 0) readonly buffer Positions { vec4 positions[]; };
// layout(std430, set = 0, binding = 1) writeonly buffer MortonKeys { uint morton_keys[]; };

// layout(push_constant) uniform PC {
//   uint n;
//   vec3 bounds_min;
//   vec3 bounds_max;
// }
// pc;

// must be exactly 16 bytes
struct MortonPushConstants {
  uint32_t n;       // offset 0
  float min_coord;  // offset 4
  float range;      // offset 8
  float pad;        // offset 12 → ensures total size is 16
};
static_assert(sizeof(MortonPushConstants) == 16, "Push‐constant block must be 16 bytes");

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

VulkanDispatcher::VulkanDispatcher() : engine(), seq(engine.make_seq()) {
  spdlog::debug("VulkanDispatcher::VulkanDispatcher(), Initializing VulkanDispatcher");

  auto morton_algo = engine.make_algo("octree_morton")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)
                         ->num_buffers(2)
                         ->push_constant<MortonPushConstants>()
                         ->build();

  cached_algorithms.try_emplace("octree_morton", std::move(morton_algo));

  auto build_edge_count_algo = engine.make_algo("octree_edge_count")
                                   ->work_group_size(256, 1, 1)
                                   ->num_sets(1)
                                   ->num_buffers(5)
                                   ->push_constant<BuildEdgeCountPushConstants>()
                                   ->build();

  cached_algorithms.try_emplace("octree_edge_count", std::move(build_edge_count_algo));

  auto build_radix_tree_algo = engine.make_algo("octree_build_radix_tree")
                                   ->work_group_size(256, 1, 1)
                                   ->num_sets(1)
                                   ->num_buffers(6)
                                   ->push_constant<BuildRadixTreePushConstants>()
                                   ->build();

  cached_algorithms.try_emplace("octree_build_radix_tree", std::move(build_radix_tree_algo));

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

void VulkanDispatcher::run_stage_1(AppData& appdata) {
  auto algo = cached_algorithms.at("octree_morton").get();

  LOG_KERNEL(LogKernelType::kVK, 1, &appdata);

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(appdata.u_positions),
                                  engine.get_buffer_info(appdata.u_morton_codes),
                              });

  algo->update_push_constant(MortonPushConstants{
      .n = static_cast<uint32_t>(appdata.n),
      .min_coord = kMinCoord,
      .range = kMaxCoord - kMinCoord,
      .pad = 0,
  });

  uint32_t grid_size_x = static_cast<uint32_t>(kiss_vk::div_ceil(appdata.n, 256));

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(), {grid_size_x, 1, 1});
  seq->cmd_end();

  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();
}

// ----------------------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_2(AppData& appdata) {
  LOG_KERNEL(LogKernelType::kVK, 2, &appdata);
}

// ----------------------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_3(AppData& appdata) {
  LOG_KERNEL(LogKernelType::kVK, 3, &appdata);
}

// ----------------------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_4(AppData& appdata) {
  LOG_KERNEL(LogKernelType::kVK, 4, &appdata);
}

// ----------------------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_5(AppData& appdata) {
  LOG_KERNEL(LogKernelType::kVK, 5, &appdata);
}

// ----------------------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_6(AppData& appdata) {
  LOG_KERNEL(LogKernelType::kVK, 6, &appdata);
}

// ----------------------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_7(AppData& appdata) {
  LOG_KERNEL(LogKernelType::kVK, 7, &appdata);
}

}  // namespace octree::vulkan
