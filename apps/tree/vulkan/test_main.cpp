#include <gtest/gtest.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "apps/tree/tree_diff_oracle.hpp"
#include "dispatchers.hpp"
#include "platform/registry/device_registry.hpp"
#include "runtime/tests/placement_check.hpp"
#include "vk_appdata.hpp"

// ----------------------------------------------------------------------------
// Tree × Vulkan differential oracle. The Vulkan backend operates on
// VkAppData_Safe (a subclass of SafeAppData), so the OMP golden and _out buffers
// the checks read are inherited. Same BT_DECLARE expansion as the OMP/CUDA
// suites, only the Runner + its AppData type differ. Run on an integrated-GPU
// box / Jetson / phone (the engine hard-selects an integrated GPU).
// ----------------------------------------------------------------------------

namespace {
struct VulkanTreeRunner {
  using AppData = tree::vulkan::VkAppData_Safe;
  tree::vulkan::VulkanDispatcher disp;
  // Skip (don't crash) on a box with no integrated GPU -- mirrors the CUDA suites' probe.
  static bool Available() { return kiss_vk::has_integrated_gpu(); }
  kiss_vk::VulkanMemoryResource::memory_resource* Mr() { return disp.get_mr(); }
  void RunStage(AppData& a, int stage) { disp.dispatch_stage(a, stage); }
};
}  // namespace

BT_DECLARE_TREE_DIFF_TESTS(TreeDiffVulkan, VulkanTreeRunner)

// Object-lifetime spec, "Repeated engine construction": construct a dispatcher (and so its
// Engine and every stage's Algorithm), run the full chain, destroy it, and do it all again
// in the same process. Under BT_VK_VALIDATION=1 an object outliving its device shows up
// here as VUID-vkDestroyDevice-device-05137.
TEST(VulkanDispatcherLifetime, ConstructDestroyReconstruct) {
  for (int round = 0; round < 2; ++round) {
    SCOPED_TRACE(round);
    tree::testing::RunAndCheckStage<VulkanTreeRunner>(7);
  }
}

// Runtime gpu-chunks spec / design C2: two threads, each with its own Sequence, record and
// submit full chunks (stages 1-7) through ONE dispatcher at the same time -- 1000 chunks in
// all, on two items per thread. Every item must still match the oracle; under
// BT_VK_VALIDATION=1 the gate also requires 0 validation errors (incl. queue
// external-synchronization ones, which the Engine's submit mutex prevents).
TEST(VulkanConcurrentSequences, TwoThreadsOneDispatcher) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  tree::vulkan::VulkanDispatcher disp;
  constexpr int kThreads = 2;
  constexpr int kItemsPerThread = 2;
  constexpr int kChunksPerThread = 500;
  std::vector<std::unique_ptr<tree::vulkan::VkAppData_Safe>> items;
  items.reserve(kThreads * kItemsPerThread);
  for (int i = 0; i < kThreads * kItemsPerThread; ++i) {
    items.push_back(std::make_unique<tree::vulkan::VkAppData_Safe>(disp.get_mr()));
  }

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&disp, &items, t] {
      const auto seq = disp.make_seq();
      for (int i = 0; i < kChunksPerThread; ++i) {
        disp.dispatch_multi_stage(*items[t * kItemsPerThread + i % kItemsPerThread], 1, 7, *seq);
      }
    });
  }
  for (auto& th : threads) th.join();

  for (size_t i = 0; i < items.size(); ++i) {
    SCOPED_TRACE("item " + std::to_string(i));
    tree::testing::CheckStage1(*items[i]);
    tree::testing::CheckStage2(*items[i]);
    tree::testing::CheckStage3(*items[i]);
    tree::testing::CheckStage4(*items[i]);
    tree::testing::CheckStage5(*items[i]);
    tree::testing::CheckStage6(*items[i]);
    tree::testing::CheckStage7(*items[i]);
  }
}

// Submission spec, "Steady-state execution" (design D2): the same pooled item through the
// same chunk a second time records no command buffer and still matches the oracle; another
// item (other buffers) records its own. On the transient-pool fallback every pass records.
TEST(VulkanPrerecord, SameItemSecondPassRecordsNothing) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  tree::vulkan::VulkanDispatcher disp;
  const auto seq = disp.make_seq();
  tree::vulkan::VkAppData_Safe a(disp.get_mr());
  tree::vulkan::VkAppData_Safe b(disp.get_mr());
  const uint64_t again = seq->reuses_recordings() ? 0 : 1;

  const auto pass = [&](tree::vulkan::VkAppData_Safe& item) {
    const uint64_t before = seq->recording_count();
    disp.dispatch_multi_stage(item, 1, 7, *seq);
    return seq->recording_count() - before;
  };
  EXPECT_EQ(pass(a), 1u);
  EXPECT_EQ(pass(a), again);
  EXPECT_EQ(pass(b), 1u);
  EXPECT_EQ(pass(a), again);
  tree::testing::CheckStage7(a);
  tree::testing::CheckStage7(b);
}

// ----------------------------------------------------------------------------
// Tree x Vulkan differential oracle, genuinely-chained path (tree::vulkan::VkAppData,
// no golden/_out split) -- added alongside TreeDiffVulkan above (unchanged). Same
// BT_DECLARE_TREE_DIFF_TESTS_APPDATA expansion as the OMP/CUDA chained suites; see
// apps/tree/tree_diff_oracle.hpp and this feature's research.md/contracts.
// ----------------------------------------------------------------------------

namespace {
struct VulkanChainedTreeRunner {
  using AppData = tree::vulkan::VkAppData;
  tree::vulkan::VulkanDispatcher disp;
  static bool Available() { return kiss_vk::has_integrated_gpu(); }
  kiss_vk::VulkanMemoryResource::memory_resource* Mr() { return disp.get_mr(); }
  void RunStage(AppData& a, int stage) { disp.dispatch_stage(a, stage); }
};
}  // namespace

BT_DECLARE_TREE_DIFF_TESTS_APPDATA(TreeDiffVulkanChained, VulkanChainedTreeRunner)

// Morton-codes spec, "Boundary input on all backends" (see tree_diff_oracle.hpp).
TEST(TreeDiffVulkanChained, Stage1_MortonBoundary) {
  tree::testing::RunAndCheckMortonBoundary<VulkanChainedTreeRunner>();
}

// Octree-build spec, "Many leaves linked into one node concurrently": stage 7 (both
// phases) 100 times on an input where 4 brt nodes link 8 leaves into each node.
TEST(TreeLeafRaceVulkan, ConcurrentLeafLinksKeepEveryBit) {
  if (!VulkanChainedTreeRunner::Available()) GTEST_SKIP() << "no integrated GPU";
  VulkanChainedTreeRunner runner;
  EXPECT_EQ(tree::testing::CountLostLeafBits(runner, 100), 0u)
      << "leaf-mask bits lost over 100 runs";
}

// Memory-placement spec, design B3: the stages each VkAppData buffer declares (what the
// schedule-driven placement trusts) are exactly the stages whose dispatches bind it. Stages
// run in order, so the dispatcher's read-backs (n_unique after stage 3) happen as in a pipeline.
TEST(Placement, DeclaredStagesMatchBindings) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  tree::vulkan::VulkanDispatcher disp;
  auto* vk_mr = static_cast<kiss_vk::VulkanMemoryResource*>(disp.get_mr());
  bt_placement_test::RecordingDispatcher rec(vk_mr);
  tree::vulkan::VkAppData app(rec.placement(all_stages(7)));

  for (int s = 1; s <= 7; ++s) {
    disp.dispatch_multi_stage(app, s, s);
    const auto& touched = disp.get_seq()->touched_buffers();
    std::set<VkBuffer> bound(touched.begin(), touched.end());
    std::set<VkBuffer> declared;
    for (const auto& [ptr, alloc] : rec.log) {
      if (std::ranges::find(alloc.stages, s) != alloc.stages.end()) {
        declared.insert(vk_mr->find_buffer(ptr));
      }
    }
    EXPECT_EQ(bound, declared) << "stage " << s;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  // Host-side golden uses OMP; pin single-threaded so the radix-tree/octree
  // reference is deterministic (matches the OMP/CUDA suites).
  omp_set_num_threads(1);
  return RUN_ALL_TESTS();
}
