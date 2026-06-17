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

#include "../../app.hpp"
#include "../../pipeline/record.hpp"
#include "../../pipeline/spsc_queue.hpp"
#include "../appdata.hpp"
#include "../cifar_dense_diff_oracle.hpp"  // cifar_dense::testing::CheckFinalPipeline
#include "../omp/dispatchers.hpp"
#include "dispatchers.cuh"  // cifar_dense::cuda::CudaDispatcher

using DispatcherT = cifar_dense::cuda::CudaDispatcher;
using AppDataT = cifar_dense::AppData;
using AppDataPtr = std::unique_ptr<AppDataT>;
constexpr size_t kNumStages = 9;
constexpr size_t kPoolSize = 8;
constexpr size_t kNumToProcess = 32;
using QueueT = SPSCQueue<AppDataT*, 16>;  // pow2 >= kPoolSize(8) with a free slot
using LocalQueue = std::queue<AppDataT*>;

#include "../../../pipe/pipeline_common.hpp"
#include "../../pipeline/pipeline_test_runner.hpp"

namespace {

using bt_pipe_test::run_pipeline;

bool CudaAvailable() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

auto omp_dispatch = [](const std::vector<int>& cores, int n, AppDataT& app, int s, int e) {
  cifar_dense::omp::dispatch_multi_stage(cores, n, app, s, e);
};

void CheckItem(AppDataT& a) { cifar_dense::testing::CheckFinalPipeline(a); }

TEST(PipelineE2ECifarDenseCu, HybridOmpCuda) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device";
  Schedule sched;
  sched.uid = "cifar-dense-omp-cu";
  sched.chunks = {{ExecutionModel::kOMP, 1, 4, first_present_cpu_type()},
                  {ExecutionModel::kCuda, 5, 9, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_pipeline<AppDataT, DispatcherT, QueueT>(sched, kPoolSize, kNumToProcess,
                                              ExecutionModel::kCuda, omp_dispatch, CheckItem);
}

TEST(PipelineE2ECifarDenseCu, AllCuda) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device";
  Schedule sched;
  sched.uid = "cifar-dense-all-cu";
  sched.chunks = {{ExecutionModel::kCuda, 1, 9, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_pipeline<AppDataT, DispatcherT, QueueT>(sched, kPoolSize, kNumToProcess,
                                              ExecutionModel::kCuda, omp_dispatch, CheckItem);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
