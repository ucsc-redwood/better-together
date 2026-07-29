#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "apps/tree/tree_diff_oracle.hpp"
#include "dispatchers.cuh"
#include "platform/registry/device_registry.hpp"

// ----------------------------------------------------------------------------
// Tree × CUDA differential oracle. CUDA dispatches each stage into its own
// tree::AppData chain (`out`); a freshly-built OMP chain on the same input
// (`ref`, built by RunAndCheckStageAppData) is the reference. Same
// BT_DECLARE_TREE_DIFF_TESTS_APPDATA expansion as the OMP suite, only the
// Runner differs. GTEST_SKIPs when no CUDA device is present (e.g. desktop CI
// box).
// ----------------------------------------------------------------------------

namespace {
struct CudaTreeRunner {
  using AppData = tree::AppData;
  tree::cuda::CudaDispatcher disp;
  static bool Available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
  }
  std::pmr::memory_resource* Mr() { return &disp.get_mr(); }
  void RunStage(tree::AppData& a, int stage) { disp.dispatch_stage(a, stage); }
};
}  // namespace

BT_DECLARE_TREE_DIFF_TESTS_APPDATA(TreeDiffCuda, CudaTreeRunner)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  // The host-side golden uses OMP for the radix-tree / octree builds, which are
  // order-nondeterministic in parallel; pin it single-threaded so the reference
  // is stable (matches the OMP suite). CUDA execution is unaffected.
  omp_set_num_threads(1);
  return RUN_ALL_TESTS();
}
