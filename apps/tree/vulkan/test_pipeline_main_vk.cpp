#include <gtest/gtest.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <functional>
#include <memory>
#include <memory_resource>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "apps/tree/omp/dispatchers.hpp"
#include "apps/tree/tree_diff_oracle.hpp"  // tree::testing::CheckStage7
#include "dispatchers.hpp"                 // tree::vulkan::VulkanDispatcher
#include "platform/registry/device_registry.hpp"
#include "runtime/record.hpp"
#include "runtime/spsc_queue.hpp"
#include "vk_appdata.hpp"  // tree::vulkan::VkAppData, VkAppData_Safe

// ----------------------------------------------------------------------------
// Framework runtime test, VULKAN: drive the tree app through the REAL concurrent
// worker/SPSC ring with a HYBRID schedule -- an OMP CPU chunk and a Vulkan GPU
// chunk, both operating on the SAME unified-memory (UMA) AppData pool. The OMP
// thread (stages 1-3) and the GPU thread (stages 4-7) run CONCURRENTLY on
// different pooled items, so the CPU writes one item's buffers while the GPU
// reads/writes another's -- the concurrent CPU+GPU + unified-memory-coherency
// path (§1 CUDA managed-mem / §7 Mali HOST_CACHED family) that the sequential
// per-stage oracle never exercises. Reuses run_pipeline() + the CheckStage7
// differential oracle. Runs on an integrated-GPU box (rocky-ryzen) / Jetson /
// the Mali phones (the engine hard-selects an integrated GPU).
// ----------------------------------------------------------------------------

#include "runtime/pipeline_runner.hpp"  // run_runtime_test, AppTraits

// AppTraits for the Vulkan tree runtime-test cell (keyed on its real GPU dispatcher;
// mirrors the constants in pipe/tree-vk/const.hpp). Real Vulkan dispatcher + the
// genuinely-chained tree::vulkan::VkAppData (Phase 3 of the Vulkan migration --
// mirrors apps/tree/cuda/test_pipeline_main_cu.cu's equivalent switch; this
// specialization can only exist once per binary, so it's an in-place switch, not
// additive -- VkAppData_Safe remains untouched for TreeDiffVulkan in test_main.cpp).
template <>
struct AppTraits<tree::vulkan::VulkanDispatcher> {
  using AppData = tree::vulkan::VkAppData;
  using Queue = SPSCQueue<AppData*, 32>;  // pow2 >= kPoolSize(16) with a free slot
  static constexpr int kNumStages = bt::vocab::kTreeStages;
  static constexpr std::size_t kPoolSize = 16;
  static constexpr std::size_t kNumToProcess = 100;
  static constexpr ExecutionModel kGpuExecModel = ExecutionModel::kVulkan;
  static void omp_dispatch(const std::vector<int>& cores, int n, AppData& app, int start, int end) {
    tree::omp::dispatch_multi_stage(cores, n, app, start, end);
  }
};

namespace {

using bt_pipe_test::run_runtime_test;
using AppDataT = AppTraits<tree::vulkan::VulkanDispatcher>::AppData;
constexpr int kNumStages = AppTraits<tree::vulkan::VulkanDispatcher>::kNumStages;

// Per-item check: build a fresh OMP-computed `ref` chain from `a`'s own input
// points (the same pattern test_pipeline_chained_vk.cpp's CheckItemChained uses),
// then diff `a` (what the concurrent ring actually produced) against it -- the
// full stage-7 differential oracle plus the §1/§7 subset-zero detector (a
// partial-visibility / stale-read regression leaves the octree mask all-zero on
// a subset of items).
void CheckItem(AppDataT& a) {
  tree::AppData ref(std::pmr::new_delete_resource(), a.get_n_input());
  ref.u_input_points_s0 = a.u_input_points_s0;
  tree::omp::dispatch_multi_stage(ref, 1, 7);

  tree::testing::CheckStage7(ref, a);
  const auto n = a.get_n_octree_nodes();
  bool all_zero = n > 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (a.u_oct_child_node_mask_s7[i] != 0) {
      all_zero = false;
      break;
    }
  }
  EXPECT_FALSE(all_zero) << "octree node_mask is all-zero -- the §1/§7 visibility symptom";
}

// Hybrid CPU+GPU: OMP stages 1-3 (CPU writes UMA), Vulkan stages 4-7 (GPU reads the
// CPU writes + writes back). The two chunks are two concurrent worker threads over the
// shared pool -> the concurrent unified-memory-visibility gate.
TEST(PipelineE2EVk, HybridOmpVulkan) {
  Schedule sched;
  sched.uid = "e2e-omp-vk";
  sched.chunks = {
      {ExecutionModel::kOMP, 1, 3, first_present_cpu_type()},
      {ExecutionModel::kVulkan, 4, 7, std::nullopt},
  };
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<tree::vulkan::VulkanDispatcher>(sched, CheckItem);
}

// All-Vulkan through the ring: every stage on the GPU chunk -- the GPU runtime path
// (engine reuse + SPSC handoff + pool recycle) with no CPU chunk.
TEST(PipelineE2EVk, AllVulkan) {
  Schedule sched;
  sched.uid = "e2e-all-vk";
  sched.chunks = {{ExecutionModel::kVulkan, 1, 7, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<tree::vulkan::VulkanDispatcher>(sched, CheckItem);
}

// Submission spec, "Depth 2 preserves order and results" (design D3): with two items in
// flight on each Vulkan chunk, every item still matches the oracle, and run_pipeline checks
// that items reach the last chunk in the order they entered.
TEST(PipelineE2EVk, InflightDepth2KeepsOrderAndResults) {
  const char* old = std::getenv("BT_VK_INFLIGHT");
  const std::string saved = old != nullptr ? old : "";
  setenv("BT_VK_INFLIGHT", "2", 1);
  for (const auto& chunks : std::vector<std::vector<ChunkConfig>>{
           {{ExecutionModel::kVulkan, 1, 7, std::nullopt}},
           {{ExecutionModel::kOMP, 1, 3, first_present_cpu_type()},
            {ExecutionModel::kVulkan, 4, 7, std::nullopt}},
           {{ExecutionModel::kVulkan, 1, 2, std::nullopt},
            {ExecutionModel::kOMP, 3, 5, first_present_cpu_type()},
            {ExecutionModel::kVulkan, 6, 7, std::nullopt}},
       }) {
    Schedule sched{"e2e-depth2", chunks};
    validate_schedule_coverage(sched, kNumStages);
    run_runtime_test<tree::vulkan::VulkanDispatcher>(sched, CheckItem);
  }
  if (old != nullptr) {
    setenv("BT_VK_INFLIGHT", saved.c_str(), 1);
  } else {
    unsetenv("BT_VK_INFLIGHT");
  }
}

// Runtime gpu-chunks spec, "Two Vulkan chunks around a CPU chunk": the GPU serves two
// chunks of one schedule, each with its own Sequence, concurrently with the CPU chunk
// between them.
TEST(PipelineE2EVk, TwoVulkanChunksAroundCpu) {
  Schedule sched;
  sched.uid = "e2e-vk-omp-vk";
  sched.chunks = {
      {ExecutionModel::kVulkan, 1, 2, std::nullopt},
      {ExecutionModel::kOMP, 3, 5, first_present_cpu_type()},
      {ExecutionModel::kVulkan, 6, 7, std::nullopt},
  };
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<tree::vulkan::VulkanDispatcher>(sched, CheckItem);
}

// The finer hybrid that alternates the GPU across THREE chunks (VK 1-3, OMP 4, VK 5,
// OMP 6, VK 7). Before per-chunk Sequences it reproducibly crashed: all Vulkan chunks
// shared one command buffer, so two worker threads recorded into it at once
// (VUID-vkBeginCommandBuffer-commandBuffer-00049 -> device loss; diagnosed 2026-06-17 with
// GPU-assisted validation on rocky/RADV), and first_concurrent_gpu_chunk() rejected it.
// Now each chunk has its own Sequence and the schedule must run and match the oracle.
TEST(PipelineE2EVk, ThreeVulkanChunksAlternating) {
  Schedule sched;
  sched.uid = "e2e-alt";
  sched.chunks = {
      {ExecutionModel::kVulkan, 1, 3, std::nullopt},
      {ExecutionModel::kOMP, 4, 4, first_present_cpu_type()},
      {ExecutionModel::kVulkan, 5, 5, std::nullopt},
      {ExecutionModel::kOMP, 6, 6, first_present_cpu_type()},
      {ExecutionModel::kVulkan, 7, 7, std::nullopt},
  };
  validate_schedule_coverage(sched, kNumStages);
  EXPECT_FALSE(first_concurrent_gpu_chunk(sched).has_value());
  run_runtime_test<tree::vulkan::VulkanDispatcher>(sched, CheckItem);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  // Single-threaded host golden (matches the differential suites) so the per-item
  // reference built at AppData construction is deterministic.
  omp_set_num_threads(1);
  return RUN_ALL_TESTS();
}
