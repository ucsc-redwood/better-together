#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <memory_resource>
#include <numeric>

#include "../../app.hpp"
#include "../cifar_sparse_diff_oracle.hpp"
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

// Shipped-path assertion (NOT a skip): the shipped AppData must build real CSR
// indices. Today it leaves row_ptr/col_idx all-zero (nnz=0), so the shipped sparse
// pipeline computes all zeros — the differential tests work around it with an
// in-test CSR. This FAILS by design until that's fixed, so the green suite can no
// longer hide that the SHIPPED application is broken (BUGS-FOUND §5). The fix is to
// build the CSR in AppData::initialize() (which needs nnz to be non-const).
TEST(CifarSparseAppData, ShippedCsrIndicesAreNonEmpty) {
  cifar_sparse::AppData a(std::pmr::new_delete_resource());
  const auto& rptr = a.conv1_sparse.row_ptr;
  const long row_ptr_sum = std::accumulate(rptr.begin(), rptr.end(), 0L);
  EXPECT_GT(row_ptr_sum, 0)
      << "cifar_sparse::AppData leaves the CSR empty (nnz=0) -> the shipped sparse pipeline "
         "outputs all zeros. Build the CSR in AppData::initialize() for a faithful sparse run.";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
