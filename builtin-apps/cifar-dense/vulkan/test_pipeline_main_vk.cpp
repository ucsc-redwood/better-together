// Framework runtime test, cifar-dense × VULKAN: tree's harness applied to a second
// app. Hybrid OMP|Vulkan schedule through the REAL concurrent worker/SPSC ring --
// OMP stages 1-4 ∥ Vulkan stages 5-9 over the shared UMA pool, the concurrent CPU+GPU
// visibility path. cifar is feed-forward (fixed stage shapes, no data-dependent
// counts), so unlike tree it has no GPU-re-entry landmine. per-item check = the
// end-to-end CheckFinalPipeline (NearEqual on the final logits). Runs on rocky / Mali.
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
#include "dispatchers.hpp"  // cifar_dense::vulkan::VulkanDispatcher

using DispatcherT = cifar_dense::vulkan::VulkanDispatcher;
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

auto omp_dispatch = [](const std::vector<int>& cores, int n, AppDataT& app, int s, int e) {
  cifar_dense::omp::dispatch_multi_stage(cores, n, app, s, e);
};

void CheckItem(AppDataT& a) { cifar_dense::testing::CheckFinalPipeline(a); }

TEST(PipelineE2ECifarDenseVk, HybridOmpVulkan) {
  Schedule sched;
  sched.uid = "cifar-dense-omp-vk";
  sched.chunks = {{ExecutionModel::kOMP, 1, 4, first_present_cpu_type()},
                  {ExecutionModel::kVulkan, 5, 9, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_pipeline<AppDataT, DispatcherT, QueueT>(sched, kPoolSize, kNumToProcess,
                                              ExecutionModel::kVulkan, omp_dispatch, CheckItem);
}

TEST(PipelineE2ECifarDenseVk, AllVulkan) {
  Schedule sched;
  sched.uid = "cifar-dense-all-vk";
  sched.chunks = {{ExecutionModel::kVulkan, 1, 9, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_pipeline<AppDataT, DispatcherT, QueueT>(sched, kPoolSize, kNumToProcess,
                                              ExecutionModel::kVulkan, omp_dispatch, CheckItem);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
