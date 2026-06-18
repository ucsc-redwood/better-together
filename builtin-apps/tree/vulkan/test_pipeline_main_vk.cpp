#include <gtest/gtest.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <vector>

#include "../../app.hpp"
#include "../../pipeline/record.hpp"
#include "../../pipeline/spsc_queue.hpp"
#include "../omp/dispatchers.hpp"
#include "../tree_diff_oracle.hpp"  // tree::testing::CheckStage7
#include "dispatchers.hpp"          // tree::vulkan::VulkanDispatcher
#include "vk_appdata.hpp"           // tree::vulkan::VkAppData_Safe

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
// mirrors the constants in pipe/tree-vk/const.hpp). Real Vulkan dispatcher + UMA AppData.
template <>
struct AppTraits<tree::vulkan::VulkanDispatcher> {
  using AppData = tree::vulkan::VkAppData_Safe;
  using Queue = SPSCQueue<AppData*, 32>;  // pow2 >= kPoolSize(16) with a free slot
  static constexpr int kNumStages = 7;
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

// Per-item check: the full stage-7 differential oracle + the §1/§7 subset-zero
// detector (a partial-visibility / stale-read regression leaves the octree mask
// all-zero on a subset of items).
void CheckItem(AppDataT& a) {
  tree::testing::CheckStage7(a);
  const auto n = a.get_n_octree_nodes();
  bool all_zero = n > 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (a.u_oct_child_node_mask_s7_out[i] != 0) {
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

// Finer hybrid that ALTERNATES the GPU across multiple chunks (VK 1-3, OMP 4, VK 5,
// OMP 6, VK 7 -> THREE Vulkan chunks). FINDING (2026-06-17): reproducibly SIGSEGVs.
// ROOT CAUSE (diagnosed with GPU-assisted validation on rocky/RADV) = a concurrent
// command-buffer race, NOT octree re-entry (the earlier "stale stage-7 count" triage
// was wrong -- VkAppData_Safe's counts are const-correct). Each chunk runs on its own
// worker thread, but all Vulkan chunks share ONE VulkanDispatcher -> one Sequence /
// command buffer / fence. With ≥2 Vulkan chunks, two threads record into that one
// buffer at once: vkBeginCommandBuffer on a buffer still in another thread's recording
// state (VUID-vkBeginCommandBuffer-commandBuffer-00049) -> corruption -> device loss.
// Crash frequency tracks the NUMBER of concurrent Vulkan chunks; a single contiguous
// GPU chunk ({OMP 1-3, VK 4-7}, {VK 1-7}) never races. The z3 solver assigns one
// contiguous chunk per PU, so it never emits a multi-GPU-chunk schedule -- the
// framework now REJECTS one via first_concurrent_gpu_chunk() rather than racing the
// GPU. This test asserts that rejection (no GPU needed; runs on any target).
TEST(PipelineE2EVk, RejectsMultiGpuChunkSchedule) {
  Schedule sched;
  sched.uid = "e2e-alt";
  sched.chunks = {
      {ExecutionModel::kVulkan, 1, 3, std::nullopt},
      {ExecutionModel::kOMP, 4, 4, ProcessorType::kBigCore},
      {ExecutionModel::kVulkan, 5, 5, std::nullopt},
      {ExecutionModel::kOMP, 6, 6, ProcessorType::kBigCore},
      {ExecutionModel::kVulkan, 7, 7, std::nullopt},
  };
  validate_schedule_coverage(sched, kNumStages);  // coverage is fine; the GPU re-use is not
  EXPECT_TRUE(first_concurrent_gpu_chunk(sched).has_value())
      << "a schedule with 3 Vulkan chunks must be rejected (shared command-buffer race)";
  // The single-GPU-chunk schedules the real tests use must NOT be rejected.
  EXPECT_FALSE(first_concurrent_gpu_chunk(Schedule{"ok-hybrid",
      {{ExecutionModel::kOMP, 1, 3, ProcessorType::kBigCore},
       {ExecutionModel::kVulkan, 4, 7, std::nullopt}}}).has_value());
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
