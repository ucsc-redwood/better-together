#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <vector>

#include "platform/registry/device_registry.hpp"
#include "runtime/record.hpp"
#include "runtime/spsc_queue.hpp"
#include "apps/tree/omp/dispatchers.hpp"
#include "apps/tree/tree_diff_oracle.hpp"  // tree::testing::CheckStage7
#include "dispatchers.cuh"          // tree::cuda::CudaDispatcher

// ----------------------------------------------------------------------------
// Framework runtime test, CUDA: drive the tree app through the REAL concurrent
// worker/SPSC ring with a HYBRID schedule -- an OMP CPU chunk and a CUDA GPU chunk
// over the SAME zero-copy pinned (UMA) AppData pool. The OMP thread (stages 1-3)
// and the GPU thread (stages 4-7) run CONCURRENTLY on different pooled items, so the
// CPU writes one item's pinned buffers while the GPU reads/writes another's -- the
// concurrent CPU+GPU + unified-memory path. This is the first test that exercises §1
// (the cudaMallocManaged-vs-pinned visibility defect, fixed by the zero-copy pinned
// switch in commit 4161664) under CONCURRENT execution rather than the sequential
// per-stage oracle. Cross-built in bt-cross:6.1, run on the Jetson.
// ----------------------------------------------------------------------------

#include "runtime/pipeline_runner.hpp"  // run_runtime_test, AppTraits

// AppTraits for this (tree, Cuda) runtime-test cell, keyed on its dispatcher.
template <>
struct AppTraits<tree::cuda::CudaDispatcher> {
  using AppData = tree::SafeAppData;
  using Queue = SPSCQueue<tree::SafeAppData*, 64>;
  static constexpr int kNumStages = 7;
  static constexpr std::size_t kPoolSize = 32;
  static constexpr std::size_t kNumToProcess = 100;
  static constexpr ExecutionModel kGpuExecModel = ExecutionModel::kCuda;
  static void omp_dispatch(const std::vector<int>& cores, int n, tree::SafeAppData& app, int start,
                           int end) {
    tree::omp::dispatch_multi_stage(cores, n, app, start, end);
  }
};

namespace {

using bt_pipe_test::run_runtime_test;
using AppDataT = AppTraits<tree::cuda::CudaDispatcher>::AppData;
constexpr int kNumStages = AppTraits<tree::cuda::CudaDispatcher>::kNumStages;

bool CudaAvailable() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

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
  EXPECT_FALSE(all_zero) << "octree node_mask is all-zero -- the §1 visibility symptom";
}

// Hybrid CPU+GPU: OMP stages 1-3 (CPU writes pinned UMA), CUDA stages 4-7 (GPU reads
// the CPU writes + writes back). Two concurrent worker threads over the shared pool ->
// the concurrent §1 (zero-copy pinned) visibility gate.
TEST(PipelineE2ECu, HybridOmpCuda) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device";
  Schedule sched;
  sched.uid = "e2e-omp-cu";
  sched.chunks = {
      {ExecutionModel::kOMP, 1, 3, first_present_cpu_type()},
      {ExecutionModel::kCuda, 4, 7, std::nullopt},
  };
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<tree::cuda::CudaDispatcher>(sched, CheckItem);
}

// All-CUDA through the ring: the GPU runtime path (dispatcher reuse + SPSC handoff +
// pool recycle) with no CPU chunk.
TEST(PipelineE2ECu, AllCuda) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device";
  Schedule sched;
  sched.uid = "e2e-all-cu";
  sched.chunks = {{ExecutionModel::kCuda, 1, 7, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<tree::cuda::CudaDispatcher>(sched, CheckItem);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  omp_set_num_threads(1);  // deterministic single-thread host golden at construction
  return RUN_ALL_TESTS();
}
