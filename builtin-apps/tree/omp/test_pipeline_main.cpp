#include <gtest/gtest.h>
#include <omp.h>
#include <sched.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <set>
#include <vector>

#include "../../app.hpp"
#include "../../pipeline/pipeline_test_executor.hpp"
#include "../tree_diff_oracle.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// FIRST framework runtime-correctness test: drive the tree app through the REAL
// concurrent worker/SPSC ring (builtin-apps/pipeline/pipeline_test_executor.hpp,
// reusing pipe/pipeline_common.hpp worker()) with a multi-chunk OMP schedule whose
// chunks run on DIFFERENT CPU tiers, and assert each item's final _out matches its
// own OMP golden (the Contract §1 differential, seed 114514). This is the harness
// every later category (visibility/SPSC/robustness) reuses by swapping the
// Schedule/Runner; here it proves the concurrent multi-chunk ring preserves the
// functional-equivalence invariant -- coverage the per-stage tests never touch
// (worker loop, SPSC handoff, AppData pool reuse, cross-tier OMP-barrier
// visibility, affinity-pinned dispatch). OMP-only, locally verifiable on pc.
//
// SCOPE / KNOWN HARDENING TODO (see docs/reports-for-human/runtime-test-suite-plan.md,
// Category 1, "two load-bearing corrections"). This first cut asserts each pool item's
// _out against a self-golden from the SAME seed 114514 and checks EVERY pool object
// after drain. The adversarial review flagged two blind spots to close before this
// becomes the GPU ordering/visibility gate (Category 3):
//   1. Distinguishable items: seed pool item i with 114514+i and assert output id ==
//      input id, so a wrong-item write (pool aliasing / stale mapped ptr surviving
//      reset) can't pass by writing a correct value into the wrong item.
//   2. Completion-edge assertion: record what the LAST chunk completed and assert the
//      multiset == kNumToProcess (catches drop/dup/orphan), not the static pool vector.
//      NB on x86-TSO OMP->OMP the parallel-region barrier makes this insensitive to
//      handoff ORDERING -- ordering is gated on GPU only.
// ----------------------------------------------------------------------------

namespace {

using bt_pipe_test::run_pipeline;

// The omp_dispatch closure: forward to the tree OMP affinity-pinned dispatcher.
auto omp_dispatch = [](const std::vector<int>& cores, int n, tree::SafeAppData& app, int start,
                       int end) { tree::omp::dispatch_multi_stage(cores, n, app, start, end); };

// Per-item check: the full stage-7 differential oracle, plus the §1/§7 subset-zero
// detector (a partial-visibility / stale-read regression leaves a stage output
// all-zero on a subset of items).
void CheckItem(tree::SafeAppData& a) {
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

  run_pipeline<tree::SafeAppData, bt_pipe_test::OmpDispatcher, QueueT>(
      sched, kPoolSize, kNumToProcess, ExecutionModel::kCuda, omp_dispatch, CheckItem);
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

  run_pipeline<tree::SafeAppData, bt_pipe_test::OmpDispatcher, QueueT>(
      sched, kPoolSize, kNumToProcess, ExecutionModel::kCuda, omp_dispatch, CheckItem);
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
  if (!has_big_cores() && !has_lit_cores()) {
    GTEST_SKIP() << "device has no pinnable CPU tier";
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

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);  // fills g_big_cores / g_lit_cores from --device pc
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
