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

#include "../../app.hpp"
#include "../../pipeline/record.hpp"
#include "../../pipeline/spsc_queue.hpp"
#include "../appdata.hpp"
#include "../cifar_sparse_diff_oracle.hpp"  // cifar_sparse::testing::CheckFinalPipeline
#include "../omp/dispatchers.hpp"

namespace {
// Trivial OMP "dispatcher": make_dataset() only needs get_mr() (host memory on the OMP
// path). The GPU branch of run_pipeline() is never reached for an OMP-only schedule.
struct OmpDispatcher {
  static std::pmr::memory_resource* get_mr() { return std::pmr::new_delete_resource(); }
  void dispatch_multi_stage(cifar_sparse::AppData&, int, int) {
    throw std::logic_error("OmpDispatcher has no GPU dispatch path");
  }
};
}  // namespace

using DispatcherT = OmpDispatcher;
using AppDataT = cifar_sparse::AppData;
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

auto omp_dispatch = [](const std::vector<int>& cores, int n, AppDataT& app, int s, int e) {
  cifar_sparse::omp::dispatch_multi_stage(cores, n, app, s, e);
};

void CheckItem(AppDataT& a) { cifar_sparse::testing::CheckFinalPipeline(a); }

TEST(PipelineE2ECifarSparseOmp, TwoChunkBigLittle) {
  if (!has_big_cores() || !has_lit_cores()) {
    GTEST_SKIP() << "device lacks a distinct big+little tier (need both to pin two chunks)";
  }
  Schedule sched;
  sched.uid = "cifar-sparse-big-little";
  sched.chunks = {{ExecutionModel::kOMP, 1, 5, ProcessorType::kBigCore},
                  {ExecutionModel::kOMP, 6, 9, ProcessorType::kLittleCore}};
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
