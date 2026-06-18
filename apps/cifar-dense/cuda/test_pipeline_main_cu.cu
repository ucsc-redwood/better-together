// Framework runtime test, cifar-dense × CUDA: hybrid OMP|CUDA through the REAL
// concurrent ring -- OMP stages 1-4 ∥ CUDA stages 5-9 on zero-copy pinned UMA. cifar
// is feed-forward (no data-dependent counts). per-item check = CheckFinalPipeline
// (NearEqual on the final logits). Cross-built in bt-cross:6.1, run on the Jetson.
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
#include "apps/cifar-dense/appdata.hpp"
#include "apps/cifar-dense/cifar_dense_diff_oracle.hpp"  // cifar_dense::testing::CheckFinalPipeline
#include "apps/cifar-dense/omp/dispatchers.hpp"
#include "dispatchers.cuh"  // cifar_dense::cuda::CudaDispatcher
#include "runtime/pipeline_runner.hpp"  // run_runtime_test, AppTraits

// AppTraits for this (cifar_dense, Cuda) runtime-test cell, keyed on its dispatcher.
template <>
struct AppTraits<cifar_dense::cuda::CudaDispatcher> {
  using AppData = cifar_dense::AppData;
  using Queue = SPSCQueue<cifar_dense::AppData*, 16>;
  static constexpr int kNumStages = bt::vocab::kCifarDenseStages;
  static constexpr std::size_t kPoolSize = 8;
  static constexpr std::size_t kNumToProcess = 32;
  static constexpr ExecutionModel kGpuExecModel = ExecutionModel::kCuda;
  static void omp_dispatch(const std::vector<int>& cores, int n, cifar_dense::AppData& app, int start,
                           int end) {
    cifar_dense::omp::dispatch_multi_stage(cores, n, app, start, end);
  }
};

namespace {

using bt_pipe_test::run_runtime_test;
using AppDataT = AppTraits<cifar_dense::cuda::CudaDispatcher>::AppData;
constexpr int kNumStages = AppTraits<cifar_dense::cuda::CudaDispatcher>::kNumStages;

bool CudaAvailable() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

void CheckItem(AppDataT& a) { cifar_dense::testing::CheckFinalPipeline(a); }

TEST(PipelineE2ECifarDenseCu, HybridOmpCuda) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device";
  Schedule sched;
  sched.uid = "cifar-dense-omp-cu";
  sched.chunks = {{ExecutionModel::kOMP, 1, 4, first_present_cpu_type()},
                  {ExecutionModel::kCuda, 5, 9, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<cifar_dense::cuda::CudaDispatcher>(sched, CheckItem);
}

TEST(PipelineE2ECifarDenseCu, AllCuda) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device";
  Schedule sched;
  sched.uid = "cifar-dense-all-cu";
  sched.chunks = {{ExecutionModel::kCuda, 1, 9, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<cifar_dense::cuda::CudaDispatcher>(sched, CheckItem);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
