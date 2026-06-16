#include <gtest/gtest.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <memory_resource>

#include "../../app.hpp"
#include "../tree_diff_oracle.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// Tree × OMP differential oracle. OMP is the reference, so this run is a
// self-consistency check: dispatching each stage must reproduce the golden
// computed at SafeAppData construction (same kernels), proving the harness and
// the per-stage comparisons before CUDA/Vulkan adopt the identical BT_DECLARE
// expansion. Always available (every target has a CPU).
// ----------------------------------------------------------------------------

namespace {
struct OmpTreeRunner {
  static constexpr bool Available() { return true; }
  static std::pmr::memory_resource* Mr() { return std::pmr::new_delete_resource(); }
  void RunStage(tree::SafeAppData& a, int stage) { tree::omp::dispatch_stage(a, stage); }
};
}  // namespace

BT_DECLARE_TREE_DIFF_TESTS(TreeDiffOmp, OmpTreeRunner)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  // Run the oracle deterministically: the radix-tree (stage 4) and octree
  // (stage 7) builds have cross-node writes whose result is order-sensitive
  // under parallel `omp for`, so a multi-threaded reference is not bitwise
  // stable run-to-run. A correctness oracle must be deterministic; speed is not
  // under test here. (Cross-backend GPU comparison of these structural stages
  // will still need invariant/canonical checks — see docs/TESTING.md.)
  omp_set_num_threads(1);
  return RUN_ALL_TESTS();
}
