#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "../../app.hpp"
#include "../cifar_sparse_diff_oracle.hpp"
#include "dispatchers.cuh"

// ----------------------------------------------------------------------------
// cifar-sparse × CUDA differential oracle. Each stage output is checked against
// the densified-CSR double-precision reference (same harness as the OMP suite,
// CudaRunner instead of OmpRunner). A valid CSR is built in-test (the shipped
// AppData leaves it empty). GTEST_SKIPs when no CUDA device is present.
// ----------------------------------------------------------------------------

namespace {
struct CudaRunner {
  cifar_sparse::cuda::CudaDispatcher disp;
  static bool Available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
  }
  std::pmr::memory_resource* Mr() { return &disp.get_mr(); }
  void RunStage(cifar_sparse::AppData& a, int stage) { disp.dispatch_stage(a, stage); }
};
}  // namespace

BT_DECLARE_CIFAR_SPARSE_DIFF_TESTS(CifarSparseDiffCuda, CudaRunner)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
