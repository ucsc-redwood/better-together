// Framework runtime test, cifar-sparse × CUDA: hybrid OMP|CUDA through the REAL
// concurrent ring -- OMP stages 1-4 ∥ CUDA stages 5-11 on zero-copy pinned UMA.
// per-item check = CheckFinalPipeline. Cross-built in bt-cross:6.1, run on the Jetson.
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <vector>

#include "apps/cifar-sparse/appdata.hpp"
#include "apps/cifar-sparse/cifar_sparse_diff_oracle.hpp"  // cifar_sparse::testing::CheckFinalPipeline
#include "apps/cifar-sparse/omp/dispatchers.hpp"
#include "dispatchers.cuh"  // cifar_sparse::cuda::CudaDispatcher
#include "platform/registry/device_registry.hpp"
#include "runtime/pipeline_runner.hpp"  // run_runtime_test, AppTraits
#include "runtime/record.hpp"
#include "runtime/spsc_queue.hpp"

// AppTraits for this (cifar_sparse, Cuda) runtime-test cell, keyed on its dispatcher.
template <>
struct AppTraits<cifar_sparse::cuda::CudaDispatcher> {
  using AppData = cifar_sparse::AppData;
  using Queue = SPSCQueue<cifar_sparse::AppData*, 16>;
  static constexpr int kNumStages = bt::vocab::kCifarSparseStages;
  static constexpr std::size_t kPoolSize = 8;
  static constexpr std::size_t kNumToProcess = 32;
  static constexpr ExecutionModel kGpuExecModel = ExecutionModel::kCuda;
  static void omp_dispatch(
      const std::vector<int>& cores, int n, cifar_sparse::AppData& app, int start, int end) {
    cifar_sparse::omp::dispatch_multi_stage(cores, n, app, start, end);
  }
};

namespace {

using bt_pipe_test::run_runtime_test;
using AppDataT = AppTraits<cifar_sparse::cuda::CudaDispatcher>::AppData;
constexpr int kNumStages = AppTraits<cifar_sparse::cuda::CudaDispatcher>::kNumStages;

bool CudaAvailable() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

void CheckItem(AppDataT& a) {
  cifar_sparse::testing::CheckFinalPipeline(a, 1e-3f, 1e-3f);
}  // OMP+CUDA: tight

TEST(PipelineE2ECifarSparseCu, HybridOmpCuda) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device";
  Schedule sched;
  sched.uid = "cifar-sparse-omp-cu";
  sched.chunks = {{ExecutionModel::kOMP, 1, 4, first_present_cpu_type()},
                  {ExecutionModel::kCuda, 5, 11, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<cifar_sparse::cuda::CudaDispatcher>(sched, CheckItem);
}

TEST(PipelineE2ECifarSparseCu, AllCuda) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device";
  Schedule sched;
  sched.uid = "cifar-sparse-all-cu";
  sched.chunks = {{ExecutionModel::kCuda, 1, 11, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<cifar_sparse::cuda::CudaDispatcher>(sched, CheckItem);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
