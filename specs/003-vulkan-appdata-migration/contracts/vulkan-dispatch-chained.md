# Contract: Chained Vulkan Dispatch API

New `VulkanDispatcher` member overloads, added alongside the existing `VkAppData_Safe`
ones (unmodified). Mirrors the shape of `tree::cuda::CudaDispatcher`'s
`tree::AppData` overloads (`apps/tree/cuda/dispatchers.cuh`), adjusted for Vulkan's
command-buffer/fence model.

```cpp
namespace tree::vulkan {

class VulkanDispatcher {
 public:
  // ... existing VkAppData_Safe members unchanged ...

  void run_stage_1(VkAppData& appdata);
  void run_stage_2(VkAppData& appdata);
  void run_stage_3(VkAppData& appdata);
  void run_stage_4(VkAppData& appdata);
  void run_stage_5(VkAppData& appdata);
  void run_stage_6(VkAppData& appdata);
  void run_stage_7(VkAppData& appdata);

  void dispatch_stage(VkAppData& appdata, int stage);

  // Behavioral difference from the VkAppData_Safe overload: internally splits
  // [start_stage, end_stage] into sub-batches at the stage3|4 and stage6|7
  // boundaries (each ending in its own submit + wait_for_fence + host readback
  // of n_unique/n_brt_nodes or n_octree_nodes), single batch otherwise. Callers
  // (bm_* profilers, pipeline_runner's dispatch_multi_stage call sites) see no
  // API difference -- same signature as the VkAppData_Safe overload.
  void dispatch_multi_stage(VkAppData& appdata, int start_stage, int end_stage);

 private:
  void record_stage_1(VkAppData& appdata, vk::CommandBuffer cmd);
  void record_stage_2(VkAppData& appdata, vk::CommandBuffer cmd);
  void record_stage_3(VkAppData& appdata, vk::CommandBuffer cmd);
  void record_stage_4(VkAppData& appdata, vk::CommandBuffer cmd);
  void record_stage_5(VkAppData& appdata, vk::CommandBuffer cmd);
  void record_stage_6(VkAppData& appdata, vk::CommandBuffer cmd);
  void record_stage_7(VkAppData& appdata, vk::CommandBuffer cmd);
};

}  // namespace tree::vulkan
```

## Pre/post conditions per stage (chained struct)

| Stage | Reads (own instance, plain field name) | Writes (plain field name, no `_out`) | Host readback required before next stage? |
|-------|------------------------------------------|----------------------------------------|---------------------------------------------|
| 1 | `u_input_points_s0` | `u_morton_keys_s1` | no |
| 2 | `u_morton_keys_s1` + scratch | `u_morton_keys_sorted_s2` | no |
| 3 | `u_morton_keys_sorted_s2` + scratch | `u_morton_keys_unique_s3`, `u_num_selected_out`\* | **yes** — `n_unique`/`n_brt_nodes` needed to size stage 4's dispatch |
| 4 | `u_morton_keys_unique_s3` | `u_brt_prefix_n_s4`, `u_brt_has_leaf_left_s4`, `u_brt_has_leaf_right_s4`, `u_brt_left_child_s4`, `u_brt_parents_s4` | no |
| 5 | `u_brt_prefix_n_s4`, `u_brt_parents_s4` | `u_edge_count_s5` | no |
| 6 | `u_edge_count_s5` + scratch | `u_edge_offset_s6` | **yes** — `n_octree_nodes` bookkeeping value read after this stage |
| 7 | `u_edge_offset_s6`, `u_edge_count_s5`, `u_morton_keys_unique_s3`, `u_brt_prefix_n_s4`, `u_brt_parents_s4`, `u_brt_left_child_s4`, `u_brt_has_leaf_left_s4`, `u_brt_has_leaf_right_s4` | `u_oct_children_s7`, `u_oct_corner_s7`, `u_oct_cell_size_s7`, `u_oct_child_node_mask_s7`, `u_oct_child_leaf_mask_s7` | no |

\* `u_num_selected_out` is the dedup-count scratch counter (already present on
`tree::AppData`, used identically to CUDA's own stage-3 readback).

## Test harness contract (no changes needed — see research.md Finding 4)

```cpp
// apps/tree/vulkan/test_main.cpp -- new suite, added alongside the existing one
struct VulkanTreeRunner {                 // existing, SafeAppData -- untouched
  using AppData = tree::vulkan::VkAppData_Safe;
  ...
};
BT_DECLARE_TREE_DIFF_TESTS(TreeDiffVulkan, VulkanTreeRunner)

struct VulkanChainedTreeRunner {          // new
  using AppData = tree::vulkan::VkAppData;
  tree::vulkan::VulkanDispatcher disp;
  static bool Available() { return kiss_vk::has_integrated_gpu(); }
  auto Mr() { return disp.get_mr(); }
  void RunStage(AppData& a, int stage) { disp.dispatch_stage(a, stage); }
};
BT_DECLARE_TREE_DIFF_TESTS_APPDATA(TreeDiffVulkanChained, VulkanChainedTreeRunner)
```
