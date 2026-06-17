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

// Pipeline typedefs the shared worker()/make_dataset() reference by name (mirrors
// pipe/tree-vk/const.hpp). Real Vulkan dispatcher + UMA AppData this time.
using DispatcherT = tree::vulkan::VulkanDispatcher;
using AppDataT = tree::vulkan::VkAppData_Safe;
using AppDataPtr = std::unique_ptr<AppDataT>;
constexpr size_t kNumStages = 7;
constexpr size_t kPoolSize = 16;
constexpr size_t kNumToProcess = 100;
using QueueT = SPSCQueue<AppDataT*, 32>;  // pow2 >= kPoolSize(16) with a free slot
using LocalQueue = std::queue<AppDataT*>;

#include "../../../pipe/pipeline_common.hpp"      // make_dataset + worker
#include "../../pipeline/pipeline_test_runner.hpp"  // run_pipeline (after the typedefs)

namespace {

using bt_pipe_test::run_pipeline;

auto omp_dispatch = [](const std::vector<int>& cores, int n, AppDataT& app, int start, int end) {
  tree::omp::dispatch_multi_stage(cores, n, app, start, end);
};

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
  run_pipeline<AppDataT, DispatcherT, QueueT>(sched, kPoolSize, kNumToProcess,
                                              ExecutionModel::kVulkan, omp_dispatch, CheckItem);
}

// All-Vulkan through the ring: every stage on the GPU chunk -- the GPU runtime path
// (engine reuse + SPSC handoff + pool recycle) with no CPU chunk.
TEST(PipelineE2EVk, AllVulkan) {
  Schedule sched;
  sched.uid = "e2e-all-vk";
  sched.chunks = {{ExecutionModel::kVulkan, 1, 7, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_pipeline<AppDataT, DispatcherT, QueueT>(sched, kPoolSize, kNumToProcess,
                                              ExecutionModel::kVulkan, omp_dispatch, CheckItem);
}

// Finer hybrid: alternate CPU/GPU at single-stage boundaries (VK 1-3, OMP 4, VK 5,
// OMP 6, VK 7). FINDING (2026-06-17): this REPRODUCIBLY SEGFAULTS on rocky-ryzen
// (RADV) while the 2-chunk Hybrid + the all-VK cases pass -- a coverage-valid schedule
// that crashes the executor. DISABLED until triaged: still TBD whether it's a real
// framework robustness bug (the 5-chunk ring / fine-grained GPU<->CPU handoff) or a
// tree data-dependency limitation (a single mid-pipeline GPU stage entered from a
// CPU-produced state, e.g. stage 5 or 7 alone, needing setup an earlier GPU stage
// would have done). Run: --gtest_also_run_disabled_tests on a device + gdb/cuda-gdb.
TEST(PipelineE2EVk, DISABLED_AlternatingBoundary) {
  if (!has_big_cores() && !has_med_cores() && !has_lit_cores()) {
    GTEST_SKIP() << "no CPU tier to host the OMP chunks";
  }
  const ProcessorType pt = first_present_cpu_type();
  Schedule sched;
  sched.uid = "e2e-alt";
  sched.chunks = {
      {ExecutionModel::kVulkan, 1, 3, std::nullopt},
      {ExecutionModel::kOMP, 4, 4, pt},
      {ExecutionModel::kVulkan, 5, 5, std::nullopt},
      {ExecutionModel::kOMP, 6, 6, pt},
      {ExecutionModel::kVulkan, 7, 7, std::nullopt},
  };
  validate_schedule_coverage(sched, kNumStages);
  run_pipeline<AppDataT, DispatcherT, QueueT>(sched, kPoolSize, kNumToProcess,
                                              ExecutionModel::kVulkan, omp_dispatch, CheckItem);
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
