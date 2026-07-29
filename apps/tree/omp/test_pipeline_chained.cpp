#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <memory_resource>
#include <vector>

#include "apps/tree/omp/dispatchers.hpp"
#include "platform/registry/device_registry.hpp"
#include "runtime/pipeline_runner.hpp"  // run_pipeline, OmpStubDispatcher
#include "runtime/schedule.hpp"
#include "runtime/spsc_queue.hpp"

// ----------------------------------------------------------------------------
// EXPERIMENTAL, standalone proof-of-correctness for the new genuinely-chained
// tree::AppData dispatch path (apps/tree/omp/dispatchers.{hpp,cpp}'s
// run_stage_N(AppData&) overloads) -- NOT part of the differential/oracle test
// suite (test-tree-omp) or the concurrency-mechanics suite
// (test-pipeline-e2e-omp), which both keep using tree::SafeAppData unchanged.
// See .claude plan "enumerated-mapping-storm" / the chat session that produced
// it for full context.
//
// What this proves: pooling plain tree::AppData through the REAL SPSC
// ring/worker machinery (make_dataset/worker/run_pipeline -- the same
// machinery test-pipeline-e2e-omp uses) and running a multi-chunk OMP
// schedule produces, for each pooled item, EXACTLY the same octree a
// straightforward single-threaded stage-1..7 pass over that item's own input
// would -- i.e. genuine end-to-end chaining survives the pool/queue/thread
// handoff, not just a single sequential call.
//
// Deliberately excluded from ctest -L omp (LABELS overridden to
// "experimental" below) -- this is a one-off verification, not a maintained
// regression gate.
// ----------------------------------------------------------------------------

namespace {

using DispatcherT = bt_pipe_test::OmpStubDispatcher<tree::AppData>;
using QueueT = SPSCQueue<tree::AppData*, 64>;  // pow2 >= pool_size(32) with a free slot

constexpr size_t kPoolSize = 32;
constexpr size_t kNumToProcess = 100;
constexpr int kNumStages = 7;

void OmpDispatch(
    const std::vector<int>& /*cores*/, int /*n*/, tree::AppData& app, int start, int end) {
  // Core-pinning intentionally unsupported for this proof-of-correctness --
  // see apps/tree/omp/dispatchers.hpp's AppData overload comment.
  tree::omp::dispatch_multi_stage(app, start, end);
}

// Reference: a fresh AppData seeded with the SAME input points as `item`, run
// through the extracted stage functions sequentially (no pool/ring/threads),
// then diff every terminal buffer against what the pool/ring produced.
void CheckItemChained(tree::AppData& item) {
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
  EXPECT_FALSE(all_zero) << "octree node_mask is all-zero -- pool/ring produced no real work";
}

TEST(PipelineChainedOmp, TwoChunkBigLittle) {
  if (!has_big_cores() || !has_lit_cores()) {
    GTEST_SKIP() << "device lacks a distinct big+little tier (need both to pin two chunks)";
  }
  Schedule sched;
  sched.uid = "chained-big-little";
  sched.chunks = {
      {ExecutionModel::kOMP, 1, 4, ProcessorType::kBigCore},
      {ExecutionModel::kOMP, 5, 7, ProcessorType::kLittleCore},
  };
  validate_schedule_coverage(sched, kNumStages);

  DispatcherT disp;
  bt_pipe_test::run_pipeline<tree::AppData, DispatcherT, QueueT>(
      sched, kPoolSize, kNumToProcess, ExecutionModel::kCuda, OmpDispatch, CheckItemChained);
}

TEST(PipelineChainedOmp, SingleChunkAllStages) {
  Schedule sched;
  sched.uid = "chained-single-chunk";
  sched.chunks = {{ExecutionModel::kOMP, 1, 7, first_present_cpu_type()}};
  validate_schedule_coverage(sched, kNumStages);

  DispatcherT disp;
  bt_pipe_test::run_pipeline<tree::AppData, DispatcherT, QueueT>(
      sched, kPoolSize, kNumToProcess, ExecutionModel::kCuda, OmpDispatch, CheckItemChained);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
