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

#include "apps/cifar-dense/appdata.hpp"
#include "apps/cifar-dense/cifar_dense_diff_oracle.hpp"  // cifar_dense::testing::CheckFinalPipeline
#include "apps/cifar-dense/omp/dispatchers.hpp"
#include "dispatchers.hpp"  // cifar_dense::vulkan::VulkanDispatcher
#include "platform/registry/device_registry.hpp"
#include "runtime/pipeline_runner.hpp"  // run_runtime_test, AppTraits
#include "runtime/record.hpp"
#include "runtime/spsc_queue.hpp"

// AppTraits for this (cifar_dense, Vulkan) runtime-test cell, keyed on its dispatcher.
template <>
struct AppTraits<cifar_dense::vulkan::VulkanDispatcher> {
  using AppData = cifar_dense::AppData;
  using Queue = SPSCQueue<cifar_dense::AppData*, 16>;
  static constexpr int kNumStages = bt::vocab::kCifarDenseStages;
  static constexpr std::size_t kPoolSize = 8;
  static constexpr std::size_t kNumToProcess = 32;
  static constexpr ExecutionModel kGpuExecModel = ExecutionModel::kVulkan;
  static void omp_dispatch(
      const std::vector<int>& cores, int n, cifar_dense::AppData& app, int start, int end) {
    cifar_dense::omp::dispatch_multi_stage(cores, n, app, start, end);
  }
};

namespace {

using bt_pipe_test::run_runtime_test;
using AppDataT = AppTraits<cifar_dense::vulkan::VulkanDispatcher>::AppData;
constexpr int kNumStages = AppTraits<cifar_dense::vulkan::VulkanDispatcher>::kNumStages;

void CheckItem(AppDataT& a) {
  cifar_dense::testing::CheckFinalPipeline(a, 5e-3f, 5e-3f);
}  // OMP+Vulkan: relaxed

TEST(PipelineE2ECifarDenseVk, HybridOmpVulkan) {
  Schedule sched;
  sched.uid = "cifar-dense-omp-vk";
  sched.chunks = {{ExecutionModel::kOMP, 1, 4, first_present_cpu_type()},
                  {ExecutionModel::kVulkan, 5, 9, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<cifar_dense::vulkan::VulkanDispatcher>(sched, CheckItem);
}

TEST(PipelineE2ECifarDenseVk, AllVulkan) {
  Schedule sched;
  sched.uid = "cifar-dense-all-vk";
  sched.chunks = {{ExecutionModel::kVulkan, 1, 9, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<cifar_dense::vulkan::VulkanDispatcher>(sched, CheckItem);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
