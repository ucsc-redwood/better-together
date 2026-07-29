#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <memory_resource>
#include <optional>
#include <vector>

#include "apps/tree/omp/dispatchers.hpp"
#include "dispatchers.hpp"  // tree::vulkan::VulkanDispatcher
#include "platform/registry/device_registry.hpp"
#include "runtime/pipeline_runner.hpp"  // run_pipeline
#include "runtime/schedule.hpp"
#include "runtime/spsc_queue.hpp"
#include "vk_appdata.hpp"  // tree::vulkan::VkAppData

// ----------------------------------------------------------------------------
// EXPERIMENTAL, standalone proof-of-correctness for the genuinely-chained
// tree::vulkan::VkAppData dispatch path on a HYBRID OMP+Vulkan schedule -- an
// OMP CPU chunk and a Vulkan GPU chunk over the SAME pool, so the CPU writes
// one item's buffers while the GPU reads/writes another's concurrently. NOT
// part of the differential/oracle suite (test-tree-vk) or the
// concurrency-mechanics suite (test-pipeline-e2e-vk), which both keep using
// tree::vulkan::VkAppData_Safe unchanged. Mirrors
// apps/tree/cuda/test_pipeline_chained_cu.cu.
//
// What this proves: VulkanDispatcher::dispatch_multi_stage's VkAppData
// overload (its internal stage3|4 host-readback split) is sufficient for a
// hybrid schedule's GPU<->CPU handoff to see genuinely-chained (not
// golden-const) data correctly -- each pooled item's final octree matches a
// straightforward single-threaded OMP reference pass over the same input.
//
// Deliberately excluded from ctest -L vulkan (LABELS overridden to
// "experimental" below) -- a one-off verification, not a maintained gate.
// ----------------------------------------------------------------------------

namespace {

using QueueT = SPSCQueue<tree::vulkan::VkAppData*, 64>;  // pow2 >= pool_size(32) + free slot

constexpr size_t kPoolSize = 32;
constexpr size_t kNumToProcess = 100;
constexpr int kNumStages = 7;

void OmpDispatch(const std::vector<int>& /*cores*/,
                 int /*n*/,
                 tree::vulkan::VkAppData& app,
                 int start,
                 int end) {
  tree::omp::dispatch_multi_stage(app, start, end);
}

// Reference: a fresh tree::AppData seeded with the SAME input points as
// `item`, run through the extracted OMP stage functions sequentially (the
// oracle -- no pool/ring/GPU), then diff every terminal buffer against what
// the hybrid pool/ring (OMP stages + Vulkan stages) produced.
void CheckItemChained(tree::vulkan::VkAppData& item) {
  auto mr = std::pmr::new_delete_resource();
  tree::AppData ref(mr, item.get_n_input());
  ref.u_input_points_s0 = item.u_input_points_s0;

  tree::omp::run_stage_1(ref);
  tree::omp::run_stage_2(ref);
  tree::omp::run_stage_3(ref);
  tree::omp::run_stage_4(ref);
  tree::omp::run_stage_5(ref);
  tree::omp::run_stage_6(ref);
  tree::omp::run_stage_7(ref);

  ASSERT_EQ(item.get_n_unique(), ref.get_n_unique());
  ASSERT_EQ(item.get_n_brt_nodes(), ref.get_n_brt_nodes());
  ASSERT_EQ(item.get_n_octree_nodes(), ref.get_n_octree_nodes());

  const auto n = item.get_n_octree_nodes();
  bool all_zero = n > 0;
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(item.u_oct_child_node_mask_s7[i], ref.u_oct_child_node_mask_s7[i]) << "at node " << i;
    EXPECT_EQ(item.u_oct_child_leaf_mask_s7[i], ref.u_oct_child_leaf_mask_s7[i]) << "at node " << i;
    if (item.u_oct_child_node_mask_s7[i] != 0) all_zero = false;
  }
  EXPECT_FALSE(all_zero) << "octree node_mask is all-zero -- GPU->CPU visibility symptom";
}

// Hybrid CPU+GPU: OMP stages 1-3 (CPU), Vulkan stages 4-7 (GPU), over the SAME
// genuinely-chained VkAppData pool -- the concurrent visibility path tested
// with real chaining instead of golden-const inputs.
TEST(PipelineChainedVk, HybridOmpVulkan) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  Schedule sched;
  sched.uid = "chained-omp-vk";
  sched.chunks = {
      {ExecutionModel::kOMP, 1, 3, first_present_cpu_type()},
      {ExecutionModel::kVulkan, 4, 7, std::nullopt},
  };
  validate_schedule_coverage(sched, kNumStages);

  tree::vulkan::VulkanDispatcher disp;
  bt_pipe_test::run_pipeline<tree::vulkan::VkAppData, tree::vulkan::VulkanDispatcher, QueueT>(
      sched, kPoolSize, kNumToProcess, ExecutionModel::kVulkan, OmpDispatch, CheckItemChained);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
