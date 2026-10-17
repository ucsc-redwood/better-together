// The choice of cores for the thread that drives a Vulkan GPU chunk (runtime/gpu_host_thread.hpp).
// Hardware-free: the topology is a table here, not the device's.
#include <gtest/gtest.h>

#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "runtime/gpu_host_thread.hpp"

namespace {

using Cores = std::vector<int>;

using Topology = std::map<ProcessorType, Cores>;

// Pixel 7a: 4 little, 2 medium, 2 big. Samsung: 4 little, 3 medium, 2 big and one super core.
// A single-tier CPU (a Jetson, a PC): everything is "big". Functions, not globals: building a
// std::map can throw, which a static initializer cannot report.
Topology pixel() {
  return {{ProcessorType::kLittleCore, {0, 1, 2, 3}},
          {ProcessorType::kMediumCore, {4, 5}},
          {ProcessorType::kBigCore, {6, 7}}};
}
Topology samsung() {
  return {{ProcessorType::kLittleCore, {0, 1, 2, 3}},
          {ProcessorType::kMediumCore, {4, 5, 6}},
          {ProcessorType::kBigCore, {7, 8}},
          {ProcessorType::kSuperCore, {9}}};
}
Topology one_tier() { return {{ProcessorType::kBigCore, {0, 1, 2, 3, 4, 5}}}; }

Cores choose(const Topology& topology, const std::set<ProcessorType>& busy) {
  return bt::choose_gpu_host_cores(busy, [&](const ProcessorType t) {
    const auto it = topology.find(t);
    return it == topology.end() ? Cores{} : it->second;
  });
}

}  // namespace

TEST(GpuHostCores, PrefersMediumThenBig) {
  EXPECT_EQ(choose(pixel(), {}), (Cores{4, 5}));
  EXPECT_EQ(choose(pixel(), {ProcessorType::kBigCore}), (Cores{4, 5}));
  EXPECT_EQ(choose(pixel(), {ProcessorType::kMediumCore}), (Cores{6, 7}));
  EXPECT_EQ(choose(samsung(), {ProcessorType::kBigCore}), (Cores{4, 5, 6}));
}

// Pinned to a little core the Pixel's tree pipeline measured 71 % slower than unpinned, so
// with the medium and big tiers taken the thread stays unpinned.
TEST(GpuHostCores, NeverALittleCore) {
  EXPECT_EQ(choose(pixel(), {ProcessorType::kMediumCore, ProcessorType::kBigCore}), Cores{});
  const Topology little_only = {{ProcessorType::kLittleCore, {0, 1}}};
  EXPECT_EQ(choose(little_only, {}), Cores{});
}

TEST(GpuHostCores, NeverTheSuperCoreAndNothingWhenEveryTierIsBusy) {
  EXPECT_EQ(choose(samsung(), {ProcessorType::kMediumCore, ProcessorType::kBigCore}), Cores{});
  // The super core is not offered even when it is the only free one.
  const Topology super_only = {{ProcessorType::kSuperCore, {9}}};
  EXPECT_EQ(choose(super_only, {}), Cores{});
}

TEST(GpuHostCores, SingleTierCpuPinsOnlyWhenNoCpuChunkUsesIt) {
  EXPECT_EQ(choose(one_tier(), {}), (Cores{0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(choose(one_tier(), {ProcessorType::kBigCore}), Cores{});
}

TEST(GpuHostCores, CoreListParsing) {
  EXPECT_EQ(bt::parse_core_list("4"), (Cores{4}));
  EXPECT_EQ(bt::parse_core_list("4,5"), (Cores{4, 5}));
  EXPECT_EQ(bt::parse_core_list("none"), Cores{});
  EXPECT_EQ(bt::parse_core_list(""), Cores{});
  EXPECT_THROW((void)bt::parse_core_list("4,x"), std::exception);
  EXPECT_THROW((void)bt::parse_core_list("4,"), std::exception);
  EXPECT_THROW((void)bt::parse_core_list("-1"), std::exception);
}

namespace {
// Sets (or, with nullptr, unsets) BT_GPU_THREAD_CORES for one scope.
class ScopedCoresEnv {
 public:
  explicit ScopedCoresEnv(const char* value) {
    if (const char* old = std::getenv("BT_GPU_THREAD_CORES")) old_ = old;
    value != nullptr ? setenv("BT_GPU_THREAD_CORES", value, 1) : unsetenv("BT_GPU_THREAD_CORES");
  }
  ~ScopedCoresEnv() {
    old_ ? setenv("BT_GPU_THREAD_CORES", old_->c_str(), 1) : unsetenv("BT_GPU_THREAD_CORES");
  }
  ScopedCoresEnv(const ScopedCoresEnv&) = delete;
  ScopedCoresEnv& operator=(const ScopedCoresEnv&) = delete;
  ScopedCoresEnv(ScopedCoresEnv&&) = delete;
  ScopedCoresEnv& operator=(ScopedCoresEnv&&) = delete;

 private:
  std::optional<std::string> old_;
};

Schedule vulkan_schedule() {
  Schedule sched;
  sched.chunks = {ChunkConfig{ExecutionModel::kOMP, 1, 3, ProcessorType::kBigCore},
                  ChunkConfig{ExecutionModel::kVulkan, 4, 7, std::nullopt}};
  return sched;
}
}  // namespace

// Pinning is opt-in: it measured slower on the Pixel and unstable on the Samsung.
TEST(GpuHostCores, UnsetLeavesTheThreadToTheOs) {
  const ScopedCoresEnv env(nullptr);
  EXPECT_TRUE(bt::gpu_host_cores(vulkan_schedule(), ExecutionModel::kVulkan).empty());
}

TEST(GpuHostCores, EnvironmentVariablePinsExactlyOrNot) {
  {
    const ScopedCoresEnv env("2,3");
    EXPECT_EQ(bt::gpu_host_cores(vulkan_schedule(), ExecutionModel::kVulkan), (Cores{2, 3}));
  }
  {
    const ScopedCoresEnv env("none");
    EXPECT_TRUE(bt::gpu_host_cores(vulkan_schedule(), ExecutionModel::kVulkan).empty());
  }
  {
    const ScopedCoresEnv env("garbage");  // warns and stays unpinned
    EXPECT_TRUE(bt::gpu_host_cores(vulkan_schedule(), ExecutionModel::kVulkan).empty());
  }
}

TEST(GpuHostCores, OnlyVulkanChunksAreSteered) {
  const ScopedCoresEnv env("2,3");
  Schedule sched = vulkan_schedule();
  EXPECT_TRUE(bt::gpu_host_cores(sched, ExecutionModel::kCuda).empty());
}
