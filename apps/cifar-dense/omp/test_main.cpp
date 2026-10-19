#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory_resource>
#include <vector>

#include "apps/cifar-dense/cifar_dense_diff_oracle.hpp"
#include "dispatchers.hpp"
#include "platform/registry/device_registry.hpp"
#include "platform/util/npy_loader.hpp"
#include "platform/util/testing/input_purity.hpp"
#include "runtime/tests/placement_check.hpp"

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
  // looser than per-stage because float error accumulates across all 11 stages,
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

// Memory-placement spec, "Boundary buffers keep host-cached placement" (design B3): under
// every schedule with one GPU stage range, each buffer a CPU stage or host code touches is
// placed in the boundary class; the rest go to the weights / GPU-local classes.
// Hardware-free: the classes are recording host resources.
TEST(Placement, CpuChunkBuffersStayBoundary) {
  bt_placement_test::ExpectCpuBuffersStayBoundary<cifar_dense::AppData>(11);
}

// Memory-placement spec, "Undeclared host access keeps today's placement": no declaration here
// uses bt::kHostSparse, so every buffer gets the class it had before the marker existed.
TEST(Placement, UndeclaredHostAccessKeepsPlacement) {
  bt_placement_test::ExpectPlacementUnchanged<cifar_dense::AppData>(11);
}

// End-task accuracy on the REAL trained weights + real normalized CIFAR-10 test
// batch (04-alexnet-cifar-spec.md §7). Skips unless BT_WEIGHTS_DIR is set, so
// hermetic runs are unaffected. The PyTorch dense reference is ~92% on the full
// exported batch; at BATCH_SIZE=16 the sample is small, so assert >= 0.75
// (binomial: P(<12/16 correct | p=0.92) is far below 1%).
TEST(CifarDenseAppData, RealWeights_EndTaskAccuracy) {
  const char* dir = std::getenv("BT_WEIGHTS_DIR");
  if (dir == nullptr) {
    GTEST_SKIP() << "BT_WEIGHTS_DIR not set (deploy via scripts/deploy-weights.sh)";
  }
  cifar_dense::AppData a(std::pmr::new_delete_resource());  // ctor loads real weights + batch
  for (int s = 1; s <= 11; ++s) cifar_dense::omp::dispatch_stage(a, s);

  const int n = a.u_fc3_out.d0();
  const int k = a.u_fc3_out.d1();
  std::vector<int32_t> labels(n);
  bt::npy::load_prefix(
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
  std::printf("cifar-dense real-weight accuracy: %.4f (%d/%d)\n", acc, correct, n);
  EXPECT_GE(acc, 0.75) << "end-task accuracy regressed on the real weights";
}

// profiling/cost-table spec, "A stage is profiled on unmodified input": each stage, run
// twice on one AppData, leaves every buffer it does not write unchanged and writes the same
// output both times. Stage k writes the k-th activation after the input, nothing else.
namespace {
std::vector<bt::testing::Region> Regions(const cifar_dense::AppData& a) {
  using bt::testing::Buf;
  std::vector<bt::testing::Region> r = {
      Buf("u_input", a.u_input.pmr_vec()),         Buf("u_conv1_out", a.u_conv1_out.pmr_vec()),
      Buf("u_pool1_out", a.u_pool1_out.pmr_vec()), Buf("u_conv2_out", a.u_conv2_out.pmr_vec()),
      Buf("u_pool2_out", a.u_pool2_out.pmr_vec()), Buf("u_conv3_out", a.u_conv3_out.pmr_vec()),
      Buf("u_conv4_out", a.u_conv4_out.pmr_vec()), Buf("u_conv5_out", a.u_conv5_out.pmr_vec()),
      Buf("u_pool3_out", a.u_pool3_out.pmr_vec()), Buf("u_fc1_out", a.u_fc1_out.pmr_vec()),
      Buf("u_fc2_out", a.u_fc2_out.pmr_vec()),     Buf("u_fc3_out", a.u_fc3_out.pmr_vec()),
      Buf("u_conv1_w", a.u_conv1_w.pmr_vec()),     Buf("u_conv1_b", a.u_conv1_b.pmr_vec()),
      Buf("u_conv2_w", a.u_conv2_w.pmr_vec()),     Buf("u_conv2_b", a.u_conv2_b.pmr_vec()),
      Buf("u_conv3_w", a.u_conv3_w.pmr_vec()),     Buf("u_conv3_b", a.u_conv3_b.pmr_vec()),
      Buf("u_conv4_w", a.u_conv4_w.pmr_vec()),     Buf("u_conv4_b", a.u_conv4_b.pmr_vec()),
      Buf("u_conv5_w", a.u_conv5_w.pmr_vec()),     Buf("u_conv5_b", a.u_conv5_b.pmr_vec()),
      Buf("u_fc1_w", a.u_fc1_w.pmr_vec()),         Buf("u_fc1_b", a.u_fc1_b.pmr_vec()),
      Buf("u_fc2_w", a.u_fc2_w.pmr_vec()),         Buf("u_fc2_b", a.u_fc2_b.pmr_vec()),
      Buf("u_fc3_w", a.u_fc3_w.pmr_vec()),         Buf("u_fc3_b", a.u_fc3_b.pmr_vec())};
  return r;
}
}  // namespace

TEST(InputPurityOmp, CifarDense) {
  const char* outputs[] = {"u_conv1_out",
                           "u_pool1_out",
                           "u_conv2_out",
                           "u_pool2_out",
                           "u_conv3_out",
                           "u_conv4_out",
                           "u_conv5_out",
                           "u_pool3_out",
                           "u_fc1_out",
                           "u_fc2_out",
                           "u_fc3_out"};
  cifar_dense::AppData app(std::pmr::new_delete_resource());
  for (int s = 1; s <= 11; ++s) {
    bt::testing::ExpectStageInputPure(
        app,
        s,
        [](cifar_dense::AppData& a, const int st) { cifar_dense::omp::dispatch_stage(a, st); },
        Regions,
        {outputs[s - 1]});
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
