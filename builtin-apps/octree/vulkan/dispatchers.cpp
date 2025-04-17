#include "dispatchers.hpp"

#include <glm/vec3.hpp>

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

struct MortonPushConstants {
  uint n;             // offset 0
  uint32_t _pad0[3];  // offsets 4,8,12

  // now at offset 16:
  glm::vec3 bounds_min;  // 12 bytes of real data
  float _pad1;           // pad to 16 bytes (offset 28)

  // at offset 32:
  glm::vec3 bounds_max;  // 12 bytes of real data
  float _pad2;           // pad to 16 bytes (offset 44)
};

static_assert(sizeof(MortonPushConstants) == 48,
              "Must be exactly 48 bytes to match GLSL std140 layout");

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
}

}  // namespace octree::vulkan
