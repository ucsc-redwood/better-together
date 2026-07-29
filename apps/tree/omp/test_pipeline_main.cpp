#include <gtest/gtest.h>
#include <omp.h>
#include <sched.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <memory_resource>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include "apps/tree/tree_diff_oracle.hpp"
#include "dispatchers.hpp"
#include "platform/registry/device_registry.hpp"
#include "runtime/pipeline_runner.hpp"  // run_runtime_test, OmpStubDispatcher, AppTraits
#include "runtime/spsc_queue.hpp"

// AppTraits for the OMP-only tree runtime-test cell (keyed on the OMP stub dispatcher:
// host memory, no GPU chunk -- its dispatch_multi_stage is never reached here). Uses
// tree::AppData (genuinely chained) -- see the Phase 3 note in tree_diff_oracle.hpp.
template <>
struct AppTraits<bt_pipe_test::OmpStubDispatcher<tree::AppData>> {
  using AppData = tree::AppData;
  using Queue = SPSCQueue<tree::AppData*, 64>;  // pow2 >= kPoolSize(32) with a free slot
  static constexpr int kNumStages = bt::vocab::kTreeStages;
  static constexpr std::size_t kPoolSize = 32;
  static constexpr std::size_t kNumToProcess = 100;
  static constexpr ExecutionModel kGpuExecModel = ExecutionModel::kCuda;  // unused (OMP-only)
  static void omp_dispatch(
      const std::vector<int>& cores, int n, tree::AppData& app, int start, int end) {
    tree::omp::dispatch_multi_stage(cores, n, app, start, end);
  }
};

// ----------------------------------------------------------------------------
// FIRST framework runtime-correctness test: drive the tree app through the REAL
// concurrent worker/SPSC ring (runtime/pipeline_runner.hpp run_runtime_test(),
// reusing runtime/pipeline.hpp worker()) with a multi-chunk OMP schedule whose
// chunks run on DIFFERENT CPU tiers, and assert each item's final _out matches its
// own OMP golden (the Contract §1 differential, seed 114514). This is the harness
// every later category (visibility/SPSC/robustness) reuses by swapping the
// Schedule/Runner; here it proves the concurrent multi-chunk ring preserves the
// functional-equivalence invariant -- coverage the per-stage tests never touch
// (worker loop, SPSC handoff, AppData pool reuse, cross-tier OMP-barrier
// visibility, affinity-pinned dispatch). OMP-only, locally verifiable on pc.
//
// ORACLE HARDENING (the adversarial review's "two load-bearing corrections"):
//   1. Distinguishable items -- ALREADY satisfied: tree_appdata.cpp seeds points from a
//      STATIC std::mt19937(114514) that advances per construction, so each pool item
//      has DISTINCT input -> a DISTINCT golden. A wrong-item write lands a value that
//      mismatches the victim's golden -> caught. (The critique assumed a fixed per-item
//      seed without checking the generator; the code already distinguishes items.)
//   2. Completion-edge assertion -- DONE in run_pipeline(): the last chunk records every
//      finished item; the runner asserts the count == n_items and all pool objects
//      reached it, so a later-cycle drop (whose stale _out still matches its golden)
//      can't pass silently.
// NB on x86-TSO OMP->OMP the parallel-region barrier makes this path insensitive to
// handoff ORDERING -- ordering/visibility is gated on the GPU tests (vk/cu).
// ----------------------------------------------------------------------------

namespace {

using bt_pipe_test::run_runtime_test;

// This OMP-only cell's dispatcher key + stage count (used by validate_schedule_coverage).
using OmpTreeDispatcher = bt_pipe_test::OmpStubDispatcher<tree::AppData>;
constexpr int kNumStages = AppTraits<OmpTreeDispatcher>::kNumStages;

// Per-item check: build a fresh OMP-computed `ref` chain from `a`'s own input
// points (the same pattern test_pipeline_chained.cpp's CheckItemChained uses),
// then diff `a` (what the concurrent ring actually produced) against it -- the
// full stage-7 differential oracle plus the interior stage-4/6 bracket, plus
// the §1/§7 subset-zero detector (a partial-visibility / stale-read regression
// leaves a stage output all-zero on a subset of items).
void CheckItem(tree::AppData& a) {
  tree::AppData ref(std::pmr::new_delete_resource(), a.get_n_input());
  ref.u_input_points_s0 = a.u_input_points_s0;
  tree::omp::dispatch_multi_stage(ref, 1, 7);

  // Assert interior hand-off too, not just the terminal stage: a mid-pipeline buffer
  // corrupted by a bad chunk hand-off can be numerically swamped before stage 7 (review #9).
  // Stage 4 (radix tree) + stage 6 (edge offset) bracket the interior at exact tolerance.
  tree::testing::CheckStage4(ref, a);
  tree::testing::CheckStage6(ref, a);
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

// ----------------------------------------------------------------------------
// Case 1.1 -- OMP 2-chunk (big | little) e2e through the real ring.
// ----------------------------------------------------------------------------
TEST(PipelineE2EOmp, TwoChunkBigLittle) {
  if (!has_big_cores() || !has_lit_cores()) {
    GTEST_SKIP() << "device lacks a distinct big+little tier (need both to pin two chunks)";
  }
  Schedule sched;
  sched.uid = "e2e-big-little";
  sched.chunks = {
      {ExecutionModel::kOMP, 1, 4, ProcessorType::kBigCore},
      {ExecutionModel::kOMP, 5, 7, ProcessorType::kLittleCore},
  };
  validate_schedule_coverage(sched, kNumStages);

  run_runtime_test<OmpTreeDispatcher>(sched, CheckItem);
}

// ----------------------------------------------------------------------------
// Case 1.2 -- 7 single-stage chunks alternating big/little: the deepest ring (7
// queues, 7 worker threads), maximal cross-chunk OMP-barrier handoffs.
// ----------------------------------------------------------------------------
TEST(PipelineE2EOmp, PerStageSevenChunks) {
  if (!has_big_cores() || !has_lit_cores()) {
    GTEST_SKIP() << "device lacks a distinct big+little tier";
  }
  Schedule sched;
  sched.uid = "e2e-per-stage";
  for (int s = 1; s <= static_cast<int>(kNumStages); ++s) {
    const ProcessorType tier = (s % 2 == 1) ? ProcessorType::kBigCore : ProcessorType::kLittleCore;
    sched.chunks.push_back({ExecutionModel::kOMP, s, s, tier});
  }
  validate_schedule_coverage(sched, kNumStages);

  run_runtime_test<OmpTreeDispatcher>(sched, CheckItem);
}

// ----------------------------------------------------------------------------
// Case 2.4 -- 3-chunk big|medium|little e2e: exercises the MEDIUM tier, which only
// the mobile SoCs have (pc is big+little only -> skips here). Proves a medium chunk
// binds to its JSON tier and the concurrent ring preserves the §1 differential
// across all three tiers handing AppData between them.
// ----------------------------------------------------------------------------
TEST(PipelineE2EOmp, ThreeChunkBigMediumLittle) {
  if (!has_big_cores() || !has_med_cores() || !has_lit_cores()) {
    GTEST_SKIP() << "device lacks a distinct big+medium+little tier (mobile SoC only)";
  }
  Schedule sched;
  sched.uid = "e2e-big-med-little";
  sched.chunks = {
      {ExecutionModel::kOMP, 1, 3, ProcessorType::kBigCore},
      {ExecutionModel::kOMP, 4, 5, ProcessorType::kMediumCore},
      {ExecutionModel::kOMP, 6, 7, ProcessorType::kLittleCore},
  };
  validate_schedule_coverage(sched, kNumStages);

  run_runtime_test<OmpTreeDispatcher>(sched, CheckItem);
}

// ----------------------------------------------------------------------------
// Case 2.x -- affinity / tier binding. A standalone OMP region pinned to a tier:
// each thread samples sched_getcpu() inside the region; assert every sampled core
// is in the intended tier's set. Symmetric for big and little.
// ----------------------------------------------------------------------------
std::set<int> SampleRunningCores(const std::vector<int>& cores) {
  std::set<int> seen;
  std::mutex m;
#pragma omp parallel num_threads(static_cast<int>(cores.size()))
  {
    bind_thread_to_cores(cores);
    // Force a reschedule so a no-op setaffinity (thread left on its old core) is
    // exposed rather than masked by the thread simply not having migrated yet.
    sched_yield();
    const int cpu = sched_getcpu();
    {
      std::lock_guard<std::mutex> lk(m);
      seen.insert(cpu);
    }
  }
  return seen;
}

TEST(PipelineE2EOmp, AffinityTierBinding) {
  if (!has_big_cores() && !has_med_cores() && !has_lit_cores()) {
    GTEST_SKIP() << "device has no pinnable CPU tier";
  }
  if (has_med_cores()) {
    const std::set<int> med_set(g_med_cores.begin(), g_med_cores.end());
    const std::set<int> ran = SampleRunningCores(g_med_cores);
    EXPECT_FALSE(ran.empty());
    for (const int cpu : ran) {
      EXPECT_TRUE(med_set.count(cpu)) << "medium-pinned thread ran on off-tier core " << cpu;
    }
  }
  if (has_big_cores()) {
    const std::set<int> big_set(g_big_cores.begin(), g_big_cores.end());
    const std::set<int> ran = SampleRunningCores(g_big_cores);
    EXPECT_FALSE(ran.empty());
    for (const int cpu : ran) {
      EXPECT_TRUE(big_set.count(cpu)) << "big-pinned thread ran on off-tier core " << cpu;
    }
  }
  if (has_lit_cores()) {
    const std::set<int> lit_set(g_lit_cores.begin(), g_lit_cores.end());
    const std::set<int> ran = SampleRunningCores(g_lit_cores);
    EXPECT_FALSE(ran.empty());
    for (const int cpu : ran) {
      EXPECT_TRUE(lit_set.count(cpu)) << "little-pinned thread ran on off-tier core " << cpu;
    }
  }
}

// ----------------------------------------------------------------------------
// Case 2.2 -- bind + sched_getaffinity read-back: after binding to a tier, the
// thread's affinity MASK must equal exactly that tier's device-JSON core set (no
// extra, no missing). Necessary-but-not-sufficient companion to the running-core
// sampling above (a setaffinity can succeed yet the thread keep running on its old
// core until the next reschedule); this asserts the mask the kernel actually holds.
// Run on a dedicated thread so the test thread's own affinity is left untouched.
std::set<int> ReadBackMask(const std::vector<int>& cores) {
  std::set<int> got;
  std::thread([&] {
    bind_thread_to_cores(cores);
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
      for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &set)) got.insert(c);
    }
  }).join();
  return got;
}

TEST(PipelineE2EOmp, AffinityMaskReadback) {
  if (!has_big_cores() && !has_med_cores() && !has_lit_cores()) {
    GTEST_SKIP() << "device has no pinnable CPU tier";
  }
  const std::vector<std::pair<bool, const std::vector<int>*>> tiers = {
      {has_big_cores(), &g_big_cores},
      {has_med_cores(), &g_med_cores},
      {has_lit_cores(), &g_lit_cores}};
  for (const auto& [present, cores] : tiers) {
    if (!present) continue;
    const std::set<int> want(cores->begin(), cores->end());
    EXPECT_EQ(ReadBackMask(*cores), want)
        << "sched_getaffinity read-back mask != the device-JSON tier core set";
  }
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);  // fills g_big_cores / g_lit_cores from --device pc
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
