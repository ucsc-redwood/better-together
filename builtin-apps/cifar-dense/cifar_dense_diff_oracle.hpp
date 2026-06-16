#pragma once
// ----------------------------------------------------------------------------
// cifar-dense differential-oracle harness (backend-parametrized).
//
// Each stage's output is checked against an independent double-precision
// reference (bt::testing::cnn) recomputed from the stage's ACTUAL upstream
// buffer, so every stage is validated in isolation (an upstream bug does not
// cascade into a confusing wall of downstream failures). Float tolerance. Each
// backend's test_main defines a Runner and expands BT_DECLARE_CIFAR_DENSE_*.
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "appdata.hpp"
#include "builtin-apps/common/testing/cnn_ref.hpp"
#include "builtin-apps/common/testing/oracle.hpp"

namespace cifar_dense::testing {

namespace ref = bt::testing::cnn;

// fp32 conv with up to 64*3*3=576 accumulations per output; tolerance covers the
// float-vs-double accumulation gap (tighten/loosen per backend in CUDA/Vulkan).
inline constexpr float kRtol = 1e-3f;
inline constexpr float kAtol = 1e-4f;

// End-to-end (L2a): the whole 9-stage pipeline runs on one backend and the FINAL
// logits are compared against a full double-precision reference chained from the
// seed. Tolerance is looser than per-stage because float-vs-double error
// accumulates across all 9 stages.
inline constexpr float kE2eRtol = 5e-3f;
inline constexpr float kE2eAtol = 5e-3f;

inline void CheckConv(const char* name, const Ndarray4D& in, const Ndarray4D& w,
                      const Ndarray1D& b, const Ndarray4D& out) {
  const auto r = ref::Conv2dRef(in.data(), w.data(), b.data(), in.d0(), in.d1(), in.d2(), in.d3(),
                                w.d0(), w.d2(), w.d3(), out.d2(), out.d3(), cifar_dense::kStride,
                                cifar_dense::kPadding, cifar_dense::kRelu);
  EXPECT_TRUE(bt::testing::NearEqual(r, out.pmr_vec(), kRtol, kAtol, name));
}

inline void CheckPool(const char* name, const Ndarray4D& in, const Ndarray4D& out) {
  const auto r = ref::MaxPool2dRef(in.data(), in.d0(), in.d1(), in.d2(), in.d3(), out.d2(), out.d3(),
                                   cifar_dense::kPoolSize, cifar_dense::kPoolStride);
  EXPECT_TRUE(bt::testing::NearEqual(r, out.pmr_vec(), kRtol, kAtol, name));
}

inline void CheckLinear(const char* name, const Ndarray4D& in, const Ndarray2D& w,
                        const Ndarray1D& b, const Ndarray2D& out) {
  const int in_features = in.d1() * in.d2() * in.d3();  // flatten (N,C,H,W) -> (N, C*H*W)
  const auto r = ref::LinearRef(in.data(), w.data(), b.data(), in.d0(), in_features, w.d0());
  EXPECT_TRUE(bt::testing::NearEqual(r, out.pmr_vec(), kRtol, kAtol, name));
}

inline void CheckStage(const AppData& a, int s) {
  switch (s) {
    case 1: CheckConv("cifar-dense conv1", a.u_input, a.u_conv1_w, a.u_conv1_b, a.u_conv1_out); break;
    case 2: CheckPool("cifar-dense pool1", a.u_conv1_out, a.u_pool1_out); break;
    case 3: CheckConv("cifar-dense conv2", a.u_pool1_out, a.u_conv2_w, a.u_conv2_b, a.u_conv2_out); break;
    case 4: CheckPool("cifar-dense pool2", a.u_conv2_out, a.u_pool2_out); break;
    case 5: CheckConv("cifar-dense conv3", a.u_pool2_out, a.u_conv3_w, a.u_conv3_b, a.u_conv3_out); break;
    case 6: CheckConv("cifar-dense conv4", a.u_conv3_out, a.u_conv4_w, a.u_conv4_b, a.u_conv4_out); break;
    case 7: CheckConv("cifar-dense conv5", a.u_conv4_out, a.u_conv5_w, a.u_conv5_b, a.u_conv5_out); break;
    case 8: CheckPool("cifar-dense pool3", a.u_conv5_out, a.u_pool3_out); break;
    case 9: CheckLinear("cifar-dense linear", a.u_pool3_out, a.u_linear_w, a.u_linear_b, a.u_linear_out); break;
    default: FAIL() << "no such cifar-dense stage: " << s;
  }
}

template <class Runner>
inline void RunAndCheckStage(int s) {
  if (!Runner::Available()) {
    GTEST_SKIP() << "backend device not available on this target";
  }
  Runner runner;
  AppData a(runner.Mr());
  for (int i = 1; i <= s; ++i) runner.RunStage(a, i);
  CheckStage(a, s);
}

// ----------------------------------------------------------------------------
// L2a: end-to-end. Run the whole pipeline (stages 1..9) on the backend, then
// compare the FINAL output against an independent full-pipeline double-precision
// reference chained from the seeded input + weights. Unlike CheckStage (which
// recomputes each stage's reference from the ACTUAL upstream buffer, isolating
// stages), this lets cross-stage cumulative / interface errors surface.
// ----------------------------------------------------------------------------
inline void CheckFinalPipeline(const AppData& a) {
  const int N = a.u_input.d0();
  auto c1 = ref::Conv2dRef(a.u_input.data(), a.u_conv1_w.data(), a.u_conv1_b.data(),
                           N, a.u_input.d1(), a.u_input.d2(), a.u_input.d3(),
                           a.u_conv1_out.d1(), a.u_conv1_w.d2(), a.u_conv1_w.d3(),
                           a.u_conv1_out.d2(), a.u_conv1_out.d3(), kStride, kPadding, kRelu);
  auto p1 = ref::MaxPool2dRef(c1.data(), N, a.u_conv1_out.d1(), a.u_conv1_out.d2(),
                              a.u_conv1_out.d3(), a.u_pool1_out.d2(), a.u_pool1_out.d3(),
                              kPoolSize, kPoolStride);
  auto c2 = ref::Conv2dRef(p1.data(), a.u_conv2_w.data(), a.u_conv2_b.data(),
                           N, a.u_pool1_out.d1(), a.u_pool1_out.d2(), a.u_pool1_out.d3(),
                           a.u_conv2_out.d1(), a.u_conv2_w.d2(), a.u_conv2_w.d3(),
                           a.u_conv2_out.d2(), a.u_conv2_out.d3(), kStride, kPadding, kRelu);
  auto p2 = ref::MaxPool2dRef(c2.data(), N, a.u_conv2_out.d1(), a.u_conv2_out.d2(),
                              a.u_conv2_out.d3(), a.u_pool2_out.d2(), a.u_pool2_out.d3(),
                              kPoolSize, kPoolStride);
  auto c3 = ref::Conv2dRef(p2.data(), a.u_conv3_w.data(), a.u_conv3_b.data(),
                           N, a.u_pool2_out.d1(), a.u_pool2_out.d2(), a.u_pool2_out.d3(),
                           a.u_conv3_out.d1(), a.u_conv3_w.d2(), a.u_conv3_w.d3(),
                           a.u_conv3_out.d2(), a.u_conv3_out.d3(), kStride, kPadding, kRelu);
  auto c4 = ref::Conv2dRef(c3.data(), a.u_conv4_w.data(), a.u_conv4_b.data(),
                           N, a.u_conv3_out.d1(), a.u_conv3_out.d2(), a.u_conv3_out.d3(),
                           a.u_conv4_out.d1(), a.u_conv4_w.d2(), a.u_conv4_w.d3(),
                           a.u_conv4_out.d2(), a.u_conv4_out.d3(), kStride, kPadding, kRelu);
  auto c5 = ref::Conv2dRef(c4.data(), a.u_conv5_w.data(), a.u_conv5_b.data(),
                           N, a.u_conv4_out.d1(), a.u_conv4_out.d2(), a.u_conv4_out.d3(),
                           a.u_conv5_out.d1(), a.u_conv5_w.d2(), a.u_conv5_w.d3(),
                           a.u_conv5_out.d2(), a.u_conv5_out.d3(), kStride, kPadding, kRelu);
  auto p3 = ref::MaxPool2dRef(c5.data(), N, a.u_conv5_out.d1(), a.u_conv5_out.d2(),
                              a.u_conv5_out.d3(), a.u_pool3_out.d2(), a.u_pool3_out.d3(),
                              kPoolSize, kPoolStride);
  const int in_features = a.u_pool3_out.d1() * a.u_pool3_out.d2() * a.u_pool3_out.d3();
  auto logits = ref::LinearRef(p3.data(), a.u_linear_w.data(), a.u_linear_b.data(),
                               N, in_features, a.u_linear_w.d0());
  EXPECT_TRUE(bt::testing::NearEqual(logits, a.u_linear_out.pmr_vec(), kE2eRtol, kE2eAtol,
                                     "cifar-dense end-to-end logits"));
}

template <class Runner>
inline void RunFullAndCheckFinal() {
  if (!Runner::Available()) {
    GTEST_SKIP() << "backend device not available on this target";
  }
  Runner runner;
  AppData a(runner.Mr());
  for (int i = 1; i <= 9; ++i) runner.RunStage(a, i);
  CheckFinalPipeline(a);
}

}  // namespace cifar_dense::testing

#define BT_DECLARE_CIFAR_DENSE_DIFF_TESTS(SUITE, RUNNER)                                  \
  TEST(SUITE, Stage1_Conv1)  { cifar_dense::testing::RunAndCheckStage<RUNNER>(1); }       \
  TEST(SUITE, Stage2_Pool1)  { cifar_dense::testing::RunAndCheckStage<RUNNER>(2); }       \
  TEST(SUITE, Stage3_Conv2)  { cifar_dense::testing::RunAndCheckStage<RUNNER>(3); }       \
  TEST(SUITE, Stage4_Pool2)  { cifar_dense::testing::RunAndCheckStage<RUNNER>(4); }       \
  TEST(SUITE, Stage5_Conv3)  { cifar_dense::testing::RunAndCheckStage<RUNNER>(5); }       \
  TEST(SUITE, Stage6_Conv4)  { cifar_dense::testing::RunAndCheckStage<RUNNER>(6); }       \
  TEST(SUITE, Stage7_Conv5)  { cifar_dense::testing::RunAndCheckStage<RUNNER>(7); }       \
  TEST(SUITE, Stage8_Pool3)  { cifar_dense::testing::RunAndCheckStage<RUNNER>(8); }       \
  TEST(SUITE, Stage9_Linear) { cifar_dense::testing::RunAndCheckStage<RUNNER>(9); }            \
  TEST(SUITE, EndToEnd_FinalLogits) { cifar_dense::testing::RunFullAndCheckFinal<RUNNER>(); }
