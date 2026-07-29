#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory_resource>
#include <numeric>
#include <vector>

#include "apps/cifar-sparse/cifar_sparse_diff_oracle.hpp"
#include "dispatchers.hpp"
#include "platform/registry/device_registry.hpp"
#include "platform/util/npy_loader.hpp"

// ----------------------------------------------------------------------------
// cifar-sparse × OMP differential oracle. Each stage output is compared against
// an independent double-precision reference computed from the densified CSR, so
// this is real numerical correctness, not a "buffer changed" smoke test. The
// shipped AppData builds a real CSR in its ctor (see the regression guard below).
// CUDA/Vulkan adopt the same BT_DECLARE expansion with their own Runner.
// ----------------------------------------------------------------------------

namespace {
struct OmpRunner {
  // IEEE fp32 against a double-precision reference: hold tight. The e2e bound is
  // looser than per-stage because float error accumulates across all 11 stages,
  // but still far tighter than Vulkan's.
  static constexpr float kRtol = 1e-5f;
  static constexpr float kAtol = 1e-4f;
  static constexpr float kE2eRtol = 1e-3f;
  static constexpr float kE2eAtol = 1e-3f;
  static constexpr bool Available() { return true; }
  static std::pmr::memory_resource* Mr() { return std::pmr::new_delete_resource(); }
  void RunStage(cifar_sparse::AppData& a, int stage) {
    cifar_sparse::omp::dispatch_stage(a, stage);
  }
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

// End-task accuracy on the REAL pruned weights + real normalized CIFAR-10 test
// batch (04-alexnet-cifar-spec.md §7). Skips unless BT_WEIGHTS_DIR is set, so
// hermetic runs are unaffected. The PyTorch sparse reference hits 90.58% on
// this batch; assert a safe >= 85%.
TEST(CifarSparseAppData, RealWeights_EndTaskAccuracy) {
  const char* dir = std::getenv("BT_WEIGHTS_DIR");
  if (dir == nullptr) {
    GTEST_SKIP() << "BT_WEIGHTS_DIR not set (deploy via scripts/deploy-weights.sh)";
  }
  cifar_sparse::AppData a(std::pmr::new_delete_resource());  // ctor loads real weights + batch
  for (int s = 1; s <= 11; ++s) cifar_sparse::omp::dispatch_stage(a, s);

  const int n = a.u_fc3_out.d0();
  const int k = a.u_fc3_out.d1();
  std::vector<int32_t> labels(n);
  bt::npy::load(
      std::string(dir) + "/test_labels.npy", "<i4", {static_cast<size_t>(n)}, labels.data());

  int correct = 0;
  for (int i = 0; i < n; ++i) {
    const float* row = a.u_fc3_out.data() + static_cast<size_t>(i) * k;
    int argmax = 0;
    for (int j = 1; j < k; ++j) {
      if (row[j] > row[argmax]) argmax = j;
    }
    if (argmax == labels[i]) ++correct;
  }
  const double acc = static_cast<double>(correct) / n;
  std::printf("cifar-sparse real-weight accuracy: %.4f (%d/%d)\n", acc, correct, n);
  EXPECT_GE(acc, 0.85) << "end-task accuracy regressed on the real weights";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
