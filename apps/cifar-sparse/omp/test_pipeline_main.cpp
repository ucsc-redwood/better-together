// Framework runtime test, cifar-sparse × OMP: the CPU-only hetero-pipeline path for the
// sparse app. A big|little 2-chunk schedule through the REAL concurrent worker/SPSC ring
// -- OMP stages 1-5 on the big tier ∥ OMP stages 6-9 on the little tier. Completes the
// OMP column of the hetero-pipeline matrix. per-item check = the end-to-end
// CheckFinalPipeline (NearEqual on the final logits). Locally verifiable on pc.
#include <gtest/gtest.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <memory_resource>
#include <queue>
#include <stdexcept>
#include <vector>

#include "apps/cifar-sparse/appdata.hpp"
#include "apps/cifar-sparse/cifar_sparse_diff_oracle.hpp"  // cifar_sparse::testing::CheckFinalPipeline
#include "apps/cifar-sparse/omp/dispatchers.hpp"
#include "platform/registry/device_registry.hpp"
#include "runtime/pipeline_runner.hpp"  // run_runtime_test, OmpStubDispatcher, AppTraits
#include "runtime/record.hpp"
#include "runtime/spsc_queue.hpp"

// AppTraits for this (cifar_sparse, OMP) runtime-test cell, keyed on its dispatcher.
template <>
struct AppTraits<bt_pipe_test::OmpStubDispatcher<cifar_sparse::AppData>> {
  using AppData = cifar_sparse::AppData;
  using Queue = SPSCQueue<cifar_sparse::AppData*, 16>;
  static constexpr int kNumStages = bt::vocab::kCifarSparseStages;
  static constexpr std::size_t kPoolSize = 8;
  static constexpr std::size_t kNumToProcess = 32;
  static constexpr ExecutionModel kGpuExecModel = ExecutionModel::kCuda;  // unused (OMP-only)
  static void omp_dispatch(
      const std::vector<int>& cores, int n, cifar_sparse::AppData& app, int start, int end) {
    cifar_sparse::omp::dispatch_multi_stage(cores, n, app, start, end);
  }
};

namespace {

using bt_pipe_test::run_runtime_test;
using AppDataT = AppTraits<bt_pipe_test::OmpStubDispatcher<cifar_sparse::AppData>>::AppData;
constexpr int kNumStages =
    AppTraits<bt_pipe_test::OmpStubDispatcher<cifar_sparse::AppData>>::kNumStages;

void CheckItem(AppDataT& a) {
  // Assert a couple of interior stages too (review #9): an intermediate buffer corrupted by
  // a bad chunk hand-off can be swamped before the final logits pass at the loose e2e
  // tolerance. CheckStage uses the tighter per-stage bound on the actual upstream buffer.
  cifar_sparse::testing::CheckStage(a, 3, 1e-5f, 1e-4f);  // OMP: tight IEEE-fp32 bound
  cifar_sparse::testing::CheckStage(a, 6, 1e-5f, 1e-4f);
  cifar_sparse::testing::CheckFinalPipeline(a, 1e-3f, 1e-3f);
}

TEST(PipelineE2ECifarSparseOmp, TwoChunkBigLittle) {
  if (!has_big_cores() || !has_lit_cores()) {
    GTEST_SKIP() << "device lacks a distinct big+little tier (need both to pin two chunks)";
  }
  Schedule sched;
  sched.uid = "cifar-sparse-big-little";
  sched.chunks = {{ExecutionModel::kOMP, 1, 5, ProcessorType::kBigCore},
                  {ExecutionModel::kOMP, 6, 9, ProcessorType::kLittleCore}};
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<bt_pipe_test::OmpStubDispatcher<cifar_sparse::AppData>>(sched, CheckItem);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
