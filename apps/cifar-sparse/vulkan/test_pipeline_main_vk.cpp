// Framework runtime test, cifar-sparse × VULKAN: hybrid OMP|Vulkan through the REAL
// concurrent ring -- OMP stages 1-4 ∥ Vulkan stages 5-9 over the shared UMA pool (the
// concurrent CPU+GPU visibility path). per-item check = CheckFinalPipeline. Runs on
// rocky / Mali.
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
#include "dispatchers.hpp"  // cifar_sparse::vulkan::VulkanDispatcher
#include "platform/registry/device_registry.hpp"
#include "runtime/pipeline_runner.hpp"  // run_runtime_test, AppTraits
#include "runtime/record.hpp"
#include "runtime/spsc_queue.hpp"

// AppTraits for this (cifar_sparse, Vulkan) runtime-test cell, keyed on its dispatcher.
template <>
struct AppTraits<cifar_sparse::vulkan::VulkanDispatcher> {
  using AppData = cifar_sparse::AppData;
  using Queue = SPSCQueue<cifar_sparse::AppData*, 16>;
  static constexpr int kNumStages = bt::vocab::kCifarSparseStages;
  static constexpr std::size_t kPoolSize = 8;
  static constexpr std::size_t kNumToProcess = 32;
  static constexpr ExecutionModel kGpuExecModel = ExecutionModel::kVulkan;
  static void omp_dispatch(
      const std::vector<int>& cores, int n, cifar_sparse::AppData& app, int start, int end) {
    cifar_sparse::omp::dispatch_multi_stage(cores, n, app, start, end);
  }
};

namespace {

using bt_pipe_test::run_runtime_test;
using AppDataT = AppTraits<cifar_sparse::vulkan::VulkanDispatcher>::AppData;
constexpr int kNumStages = AppTraits<cifar_sparse::vulkan::VulkanDispatcher>::kNumStages;

void CheckItem(AppDataT& a) {
  cifar_sparse::testing::CheckFinalPipeline(a, 5e-3f, 5e-3f);
}  // OMP+Vulkan: relaxed

TEST(PipelineE2ECifarSparseVk, HybridOmpVulkan) {
  Schedule sched;
  sched.uid = "cifar-sparse-omp-vk";
  sched.chunks = {{ExecutionModel::kOMP, 1, 4, first_present_cpu_type()},
                  {ExecutionModel::kVulkan, 5, 9, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<cifar_sparse::vulkan::VulkanDispatcher>(sched, CheckItem);
}

TEST(PipelineE2ECifarSparseVk, AllVulkan) {
  Schedule sched;
  sched.uid = "cifar-sparse-all-vk";
  sched.chunks = {{ExecutionModel::kVulkan, 1, 9, std::nullopt}};
  validate_schedule_coverage(sched, kNumStages);
  run_runtime_test<cifar_sparse::vulkan::VulkanDispatcher>(sched, CheckItem);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
