#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory_resource>
#include <random>
#include <vector>

#include "apps/cifar-dense/cifar_dense_diff_oracle.hpp"
#include "dispatchers.hpp"
#include "platform/registry/device_registry.hpp"
#include "platform/util/fc_weights.hpp"
#include "platform/util/npy_loader.hpp"
#include "platform/util/testing/cpu_pinning.hpp"
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

// cifar-gpu-kernels F2: the INT8 copy of FC weights decodes back within the per-element bound
// of platform/util/fc_weights.hpp: on a matrix spanning 2^-30..2^15 with a zero row and an
// outlier row, and on the app's fc1/fc2 (the synthetic weights, or the real export with
// BT_WEIGHTS_DIR). By default no copy is made.
namespace {
void ExpectFcWeightRoundTrip(const float* w, const int rows, const int cols, const char* name) {
  bt::FcWeightCopy q(bt::FcWeights::kInt8, rows, cols, std::pmr::new_delete_resource());
  q.quantize(w);
  double max_ratio = 0.0;  // largest error relative to its bound
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < cols; ++j) {
      const float x = w[static_cast<size_t>(r) * cols + j];
      // scale / 2, plus the float rounding of scale * q (<= 127 scale 2^-24).
      const double bound = q.scale(r) * (0.5 + 1e-5);
      const double err = std::abs(x - q.dequantized(r, j));
      ASSERT_LE(err, bound) << name << " [" << r << ", " << j << "] = " << x;
      if (bound > 0.0) max_ratio = std::max(max_ratio, err / bound);
    }
  }
  std::printf("%s: int8 max error / bound %.3f\n", name, max_ratio);
}
}  // namespace

TEST(FcWeights, QuantizeRoundTripWithinBound) {
  constexpr int kRows = 5, kCols = 1000;
  std::mt19937 rng(114514);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  std::uniform_real_distribution<float> log_mag(-30.0f, 15.0f);    // 2^-30 .. 2^15
  std::vector<float> w(static_cast<size_t>(kRows) * kCols, 0.0f);  // row 0 stays zero
  for (int j = 0; j < kCols; ++j) {
    w[1 * kCols + j] = 0.02f * u(rng);                                  // synthetic FC
    w[2 * kCols + j] = std::copysign(std::exp2(log_mag(rng)), u(rng));  // every range
    w[3 * kCols + j] = std::ldexp(u(rng), -16);                         // tiny
    w[4 * kCols + j] = j == 7 ? 3.0f : 0.01f * u(rng);                  // one outlier
  }
  ExpectFcWeightRoundTrip(w.data(), kRows, kCols, "mixed");

  const cifar_dense::AppData app(std::pmr::new_delete_resource());
  ExpectFcWeightRoundTrip(app.u_fc1_w.data(), app.u_fc1_w.d0(), app.u_fc1_w.d1(), "fc1");
  ExpectFcWeightRoundTrip(app.u_fc2_w.data(), app.u_fc2_w.d0(), app.u_fc2_w.d1(), "fc2");
  if (std::getenv("BT_VK_FC_WEIGHTS") == nullptr) {  // the default: FP32 only, no copies
    EXPECT_EQ(app.fc_weights, bt::FcWeights::kFp32);
    EXPECT_TRUE(app.u_fc1_wq.words.empty());
    EXPECT_TRUE(app.u_fc2_wq.words.empty());
    // In INT8 mode the AppData quantizes its own loaded weights.
    setenv("BT_VK_FC_WEIGHTS", "int8", 1);
    const cifar_dense::AppData q(std::pmr::new_delete_resource());
    unsetenv("BT_VK_FC_WEIGHTS");
    ASSERT_EQ(q.u_fc2_wq.words.size(), size_t{4096} * 4096 / 4 + 4096);
    bt::FcWeightCopy want(bt::FcWeights::kInt8, 4096, 4096, std::pmr::new_delete_resource());
    want.quantize(q.u_fc2_w.data());
    EXPECT_EQ(q.u_fc2_wq.words, want.words);
  }
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
      Buf("u_fc3_w", a.u_fc3_w.pmr_vec()),         Buf("u_fc3_b", a.u_fc3_b.pmr_vec()),
      Buf("u_fc1_wq", a.u_fc1_wq.words),           Buf("u_fc2_wq", a.u_fc2_wq.words)};
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

// runtime/cpu-chunks spec, "Every app": stages 1 to 11 through the pinned dispatch run only on
// the given cores, with at most the given thread count, and compute what the unpinned run does.
TEST(CpuPinningOmp, CifarDense) {
  cifar_dense::AppData app(std::pmr::new_delete_resource());
  bt::testing::ExpectChunkPinned(
      app,
      [](cifar_dense::AppData& a) { cifar_dense::omp::dispatch_multi_stage(a, 1, 11); },
      [](const std::vector<int>& cores, const int n, cifar_dense::AppData& a) {
        cifar_dense::omp::dispatch_multi_stage(cores, n, a, 1, 11);
      },
      Regions);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
