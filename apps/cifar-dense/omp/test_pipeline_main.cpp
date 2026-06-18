// Framework runtime test, cifar-dense × OMP: tree's harness applied to cifar on the
// CPU-only path. A big|little 2-chunk schedule through the REAL concurrent worker/SPSC
// ring -- OMP stages 1-5 on the big tier ∥ OMP stages 6-9 on the little tier, handing
// AppData across the SPSC handoff. Completes the OMP column of the hetero-pipeline
// matrix (the vk/cu variants cover the GPU columns). per-item check = the end-to-end
// CheckFinalPipeline (NearEqual on the final logits). Locally verifiable on pc; the
// medium tier (mobile) is covered by the tree ThreeChunkBigMediumLittle case.
#include <gtest/gtest.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <memory_resource>
#include <queue>
#include <stdexcept>
#include <vector>

#include "platform/registry/device_registry.hpp"
#include "runtime/record.hpp"
#include "runtime/spsc_queue.hpp"
#include "apps/cifar-dense/appdata.hpp"
#include "apps/cifar-dense/cifar_dense_diff_oracle.hpp"  // cifar_dense::testing::CheckFinalPipeline
#include "apps/cifar-dense/omp/dispatchers.hpp"

#include "runtime/pipeline_runner.hpp"  // run_runtime_test, OmpStubDispatcher, AppTraits

// AppTraits for this (cifar_dense, OMP) runtime-test cell, keyed on its dispatcher.
template <>
struct AppTraits<bt_pipe_test::OmpStubDispatcher<cifar_dense::AppData>> {
  using AppData = cifar_dense::AppData;
  using Queue = SPSCQueue<cifar_dense::AppData*, 16>;
  static constexpr int kNumStages = 9;
  static constexpr std::size_t kPoolSize = 8;
  static constexpr std::size_t kNumToProcess = 32;
  static constexpr ExecutionModel kGpuExecModel = ExecutionModel::kCuda;  // unused (OMP-only)
  static void omp_dispatch(const std::vector<int>& cores, int n, cifar_dense::AppData& app, int start,
                           int end) {
    cifar_dense::omp::dispatch_multi_stage(cores, n, app, start, end);
  }
};

namespace {

using bt_pipe_test::run_runtime_test;
using AppDataT = AppTraits<bt_pipe_test::OmpStubDispatcher<cifar_dense::AppData>>::AppData;
constexpr int kNumStages = AppTraits<bt_pipe_test::OmpStubDispatcher<cifar_dense::AppData>>::kNumStages;

void CheckItem(AppDataT& a) {
  // Assert a couple of interior stages too (review #9): an intermediate buffer corrupted by
  // a bad chunk hand-off can be swamped before the final logits pass at the loose e2e
  // tolerance. CheckStage uses the tighter per-stage bound on the actual upstream buffer.
  cifar_dense::testing::CheckStage(a, 3);
  cifar_dense::testing::CheckStage(a, 6);
  cifar_dense::testing::CheckFinalPipeline(a);
}

TEST(PipelineE2ECifarDenseOmp, TwoChunkBigLittle) {
  if (!has_big_cores() || !has_lit_cores()) {
    GTEST_SKIP() << "device lacks a distinct big+little tier (need both to pin two chunks)";
  }
  Schedule sched;
  sched.uid = "cifar-dense-big-little";
  sched.chunks = {{ExecutionModel::kOMP, 1, 5, ProcessorType::kBigCore},
                  {ExecutionModel::kOMP, 6, 9, ProcessorType::kLittleCore}};
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<bt_pipe_test::OmpStubDispatcher<cifar_dense::AppData>>(sched, CheckItem);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
