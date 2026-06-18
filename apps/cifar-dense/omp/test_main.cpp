#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <memory_resource>

#include "platform/registry/device_registry.hpp"
#include "apps/cifar-dense/cifar_dense_diff_oracle.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// cifar-dense × OMP differential oracle. Each stage output is compared against
// an independent double-precision conv/pool/linear reference (not the kernel's
// own code), so this is a real numerical-correctness check, not a "buffer
// changed" smoke test. CUDA/Vulkan adopt the same BT_DECLARE expansion with
// their own Runner. All cifar kernels parallelize over independent output
// elements, so the result is deterministic regardless of thread count.
// ----------------------------------------------------------------------------

namespace {
struct OmpRunner {
  // IEEE fp32 against a double-precision reference: hold tight. The e2e bound is
  // looser than per-stage because float error accumulates across all 9 stages,
  // but still far tighter than Vulkan's.
  static constexpr float kRtol = 1e-5f;
  static constexpr float kAtol = 1e-4f;
  static constexpr float kE2eRtol = 1e-3f;
  static constexpr float kE2eAtol = 1e-3f;
  static constexpr bool Available() { return true; }
  static std::pmr::memory_resource* Mr() { return std::pmr::new_delete_resource(); }
  void RunStage(cifar_dense::AppData& a, int stage) { cifar_dense::omp::dispatch_stage(a, stage); }
};
}  // namespace

BT_DECLARE_CIFAR_DENSE_DIFF_TESTS(CifarDenseDiffOmp, OmpRunner)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
