#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <memory_resource>
#include <numeric>

#include "platform/registry/device_registry.hpp"
#include "apps/cifar-sparse/cifar_sparse_diff_oracle.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// cifar-sparse × OMP differential oracle. Each stage output is compared against
// an independent double-precision reference computed from the densified CSR, so
// this is real numerical correctness, not a "buffer changed" smoke test. A valid
// CSR is built in-test (the shipped AppData leaves it empty — see the guard test
// below). CUDA/Vulkan adopt the same BT_DECLARE expansion with their own Runner.
// ----------------------------------------------------------------------------

namespace {
struct OmpRunner {
  static constexpr bool Available() { return true; }
  static std::pmr::memory_resource* Mr() { return std::pmr::new_delete_resource(); }
  void RunStage(cifar_sparse::AppData& a, int stage) { cifar_sparse::omp::dispatch_stage(a, stage); }
};
}  // namespace

BT_DECLARE_CIFAR_SPARSE_DIFF_TESTS(CifarSparseDiffOmp, OmpRunner)

// Shipped-path regression guard: the shipped AppData must build real CSR indices
// (CSRMatrix::build_from_dense in the ctor). A regression to the old behavior --
// leaving row_ptr/col_idx all-zero (nnz=0) -- would make the sparse pipeline
// compute all zeros while the differential tests (which run the same shipped
// ctor) silently still pass against a zero reference. This catches that (§5).
TEST(CifarSparseAppData, ShippedCsrIndicesAreNonEmpty) {
  cifar_sparse::AppData a(std::pmr::new_delete_resource());
  const auto& rptr = a.conv1_sparse.row_ptr;
  const long row_ptr_sum = std::accumulate(rptr.begin(), rptr.end(), 0L);
  EXPECT_GT(row_ptr_sum, 0)
      << "cifar_sparse::AppData shipped with an empty CSR (nnz=0) -> the sparse pipeline "
         "outputs all zeros. CSRMatrix::build_from_dense must run in the ctor.";
  EXPECT_GT(a.conv1_sparse.nnz, 0) << "conv1 CSR has no nonzeros";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
