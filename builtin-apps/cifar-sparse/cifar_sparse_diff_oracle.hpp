#pragma once
// ----------------------------------------------------------------------------
// cifar-sparse differential-oracle harness (backend-parametrized).
//
// IMPORTANT: the shipped cifar_sparse::AppData fills only CSRMatrix::values and
// leaves row_ptr / col_idx all-zero, so the sparse kernels iterate empty rows
// and the whole pipeline computes zeros (verified). To get real ground truth we
// build a deterministic *valid* CSR from the seeded values in-test (PopulateCsr)
// before running any stage — this does not touch shipped code. The shipped
// empty-CSR defect is surfaced separately by a guard test in test_main.
//
// Each stage's output is then checked against an independent double-precision
// reference (bt::testing::cnn) computed from the SAME CSR (densified) and the
// stage's actual upstream buffer. Because the kernel does sparse conv directly
// and the reference densifies-then-dense-convs, agreement is a genuine check of
// the kernel's CSR decode + accumulation, not a tautology.
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "appdata.hpp"
#include "builtin-apps/common/testing/cnn_ref.hpp"
#include "builtin-apps/common/testing/oracle.hpp"

namespace cifar_sparse::testing {

namespace ref = bt::testing::cnn;

inline constexpr float kRtol = 1e-3f;
inline constexpr float kAtol = 1e-4f;

// End-to-end (L2a): full 9-stage pipeline on one backend vs a full double-precision
// reference chained from the seed (densified CSR). Looser tol — error accumulates.
inline constexpr float kE2eRtol = 5e-3f;
inline constexpr float kE2eAtol = 5e-3f;

// Build a deterministic valid CSR in place from the dense `values` the AppData
// seeded. Keeps a fixed ~3/4 subset of columns per row (so rows have varying,
// non-trivial nnz that exercises the kernel's index decode), compacting the kept
// weights into values[0..nnz) with matching col_idx / row_ptr.
inline void PopulateCsr(cifar_sparse::CSRMatrix& m) {
  auto& vals = m.values_pmr_vec();
  auto& rptr = m.row_ptr_pmr_vec();
  auto& cidx = m.col_idx_pmr_vec();
  std::vector<float> kept;
  kept.reserve(static_cast<std::size_t>(m.rows) * m.cols);
  int nz = 0;
  rptr[0] = 0;
  for (int r = 0; r < m.rows; ++r) {
    for (int c = 0; c < m.cols; ++c) {
      if (((r * 131 + c * 17) % 4) != 0) {  // deterministic keep pattern
        cidx[nz] = c;
        kept.push_back(vals[static_cast<std::size_t>(r) * m.cols + c]);
        ++nz;
      }
    }
    rptr[r + 1] = nz;
  }
  for (int i = 0; i < nz; ++i) vals[i] = kept[i];  // compact (aliasing-safe via `kept`)
}

inline void PopulateAllCsr(cifar_sparse::AppData& a) {
  PopulateCsr(a.conv1_sparse);
  PopulateCsr(a.conv2_sparse);
  PopulateCsr(a.conv3_sparse);
  PopulateCsr(a.conv4_sparse);
  PopulateCsr(a.conv5_sparse);
  PopulateCsr(a.linear_sparse);
}

// Densify a CSR into a row-major (rows x cols) dense matrix.
inline std::vector<float> Densify(const cifar_sparse::CSRMatrix& m) {
  std::vector<float> dense(static_cast<std::size_t>(m.rows) * m.cols, 0.0f);
  const int* rptr = m.row_ptr_data();
  const int* cidx = m.col_idx_data();
  const float* vals = m.values_data();
  for (int r = 0; r < m.rows; ++r)
    for (int nz = rptr[r]; nz < rptr[r + 1]; ++nz)
      dense[static_cast<std::size_t>(r) * m.cols + cidx[nz]] = vals[nz];
  return dense;
}

inline void CheckConv(const char* name, const Ndarray4D& in, const cifar_sparse::CSRMatrix& w,
                      const Ndarray1D& b, const Ndarray4D& out) {
  const auto dense_w = Densify(w);  // (outC, inC*kH*kW)
  const auto r = ref::Conv2dRef(in.data(), dense_w.data(), b.data(), in.d0(), in.d1(), in.d2(),
                                in.d3(), w.rows, cifar_sparse::kKernelSize, cifar_sparse::kKernelSize,
                                out.d2(), out.d3(), cifar_sparse::kStride, cifar_sparse::kPadding,
                                cifar_sparse::kRelu);
  EXPECT_TRUE(bt::testing::NearEqual(r, out.pmr_vec(), kRtol, kAtol, name));
}

inline void CheckPool(const char* name, const Ndarray4D& in, const Ndarray4D& out) {
  const auto r = ref::MaxPool2dRef(in.data(), in.d0(), in.d1(), in.d2(), in.d3(), out.d2(), out.d3(),
                                   cifar_sparse::kPoolSize, cifar_sparse::kPoolStride);
  EXPECT_TRUE(bt::testing::NearEqual(r, out.pmr_vec(), kRtol, kAtol, name));
}

inline void CheckLinear(const char* name, const Ndarray4D& in, const cifar_sparse::CSRMatrix& w,
                        const Ndarray1D& b, const Ndarray2D& out) {
  const int in_features = in.d1() * in.d2() * in.d3();
  const auto dense_w = Densify(w);  // (out_neurons, in_features)
  const auto r = ref::LinearRef(in.data(), dense_w.data(), b.data(), in.d0(), in_features, w.rows);
  EXPECT_TRUE(bt::testing::NearEqual(r, out.pmr_vec(), kRtol, kAtol, name));
}

inline void CheckStage(const AppData& a, int s) {
  switch (s) {
    case 1: CheckConv("cifar-sparse conv1", a.u_input, a.conv1_sparse, a.u_conv1_b, a.u_conv1_out); break;
    case 2: CheckPool("cifar-sparse pool1", a.u_conv1_out, a.u_pool1_out); break;
    case 3: CheckConv("cifar-sparse conv2", a.u_pool1_out, a.conv2_sparse, a.u_conv2_b, a.u_conv2_out); break;
    case 4: CheckPool("cifar-sparse pool2", a.u_conv2_out, a.u_pool2_out); break;
    case 5: CheckConv("cifar-sparse conv3", a.u_pool2_out, a.conv3_sparse, a.u_conv3_b, a.u_conv3_out); break;
    case 6: CheckConv("cifar-sparse conv4", a.u_conv3_out, a.conv4_sparse, a.u_conv4_b, a.u_conv4_out); break;
    case 7: CheckConv("cifar-sparse conv5", a.u_conv4_out, a.conv5_sparse, a.u_conv5_b, a.u_conv5_out); break;
    case 8: CheckPool("cifar-sparse pool3", a.u_conv5_out, a.u_pool3_out); break;
    case 9: CheckLinear("cifar-sparse linear", a.u_pool3_out, a.linear_sparse, a.u_linear_b, a.u_linear_out); break;
    default: FAIL() << "no such cifar-sparse stage: " << s;
  }
}

template <class Runner>
inline void RunAndCheckStage(int s) {
  if (!Runner::Available()) {
    GTEST_SKIP() << "backend device not available on this target";
  }
  Runner runner;
  AppData a(runner.Mr());
  PopulateAllCsr(a);  // shipped AppData leaves the CSR empty; build a valid one in-test
  for (int i = 1; i <= s; ++i) runner.RunStage(a, i);
  CheckStage(a, s);
}

// L2a end-to-end: run all 9 stages on the backend, compare FINAL logits against a
// full double-precision reference chained from the seed (densified CSR weights).
inline void CheckFinalPipeline(const AppData& a) {
  const int N = a.u_input.d0();
  const auto w1 = Densify(a.conv1_sparse);
  auto c1 = ref::Conv2dRef(a.u_input.data(), w1.data(), a.u_conv1_b.data(),
                           N, a.u_input.d1(), a.u_input.d2(), a.u_input.d3(),
                           a.u_conv1_out.d1(), kKernelSize, kKernelSize,
                           a.u_conv1_out.d2(), a.u_conv1_out.d3(), kStride, kPadding, kRelu);
  auto p1 = ref::MaxPool2dRef(c1.data(), N, a.u_conv1_out.d1(), a.u_conv1_out.d2(),
                              a.u_conv1_out.d3(), a.u_pool1_out.d2(), a.u_pool1_out.d3(),
                              kPoolSize, kPoolStride);
  const auto w2 = Densify(a.conv2_sparse);
  auto c2 = ref::Conv2dRef(p1.data(), w2.data(), a.u_conv2_b.data(),
                           N, a.u_pool1_out.d1(), a.u_pool1_out.d2(), a.u_pool1_out.d3(),
                           a.u_conv2_out.d1(), kKernelSize, kKernelSize,
                           a.u_conv2_out.d2(), a.u_conv2_out.d3(), kStride, kPadding, kRelu);
  auto p2 = ref::MaxPool2dRef(c2.data(), N, a.u_conv2_out.d1(), a.u_conv2_out.d2(),
                              a.u_conv2_out.d3(), a.u_pool2_out.d2(), a.u_pool2_out.d3(),
                              kPoolSize, kPoolStride);
  const auto w3 = Densify(a.conv3_sparse);
  auto c3 = ref::Conv2dRef(p2.data(), w3.data(), a.u_conv3_b.data(),
                           N, a.u_pool2_out.d1(), a.u_pool2_out.d2(), a.u_pool2_out.d3(),
                           a.u_conv3_out.d1(), kKernelSize, kKernelSize,
                           a.u_conv3_out.d2(), a.u_conv3_out.d3(), kStride, kPadding, kRelu);
  const auto w4 = Densify(a.conv4_sparse);
  auto c4 = ref::Conv2dRef(c3.data(), w4.data(), a.u_conv4_b.data(),
                           N, a.u_conv3_out.d1(), a.u_conv3_out.d2(), a.u_conv3_out.d3(),
                           a.u_conv4_out.d1(), kKernelSize, kKernelSize,
                           a.u_conv4_out.d2(), a.u_conv4_out.d3(), kStride, kPadding, kRelu);
  const auto w5 = Densify(a.conv5_sparse);
  auto c5 = ref::Conv2dRef(c4.data(), w5.data(), a.u_conv5_b.data(),
                           N, a.u_conv4_out.d1(), a.u_conv4_out.d2(), a.u_conv4_out.d3(),
                           a.u_conv5_out.d1(), kKernelSize, kKernelSize,
                           a.u_conv5_out.d2(), a.u_conv5_out.d3(), kStride, kPadding, kRelu);
  auto p3 = ref::MaxPool2dRef(c5.data(), N, a.u_conv5_out.d1(), a.u_conv5_out.d2(),
                              a.u_conv5_out.d3(), a.u_pool3_out.d2(), a.u_pool3_out.d3(),
                              kPoolSize, kPoolStride);
  const auto wl = Densify(a.linear_sparse);
  const int in_features = a.u_pool3_out.d1() * a.u_pool3_out.d2() * a.u_pool3_out.d3();
  auto logits = ref::LinearRef(p3.data(), wl.data(), a.u_linear_b.data(),
                               N, in_features, a.u_linear_out.d1());
  EXPECT_TRUE(bt::testing::NearEqual(logits, a.u_linear_out.pmr_vec(), kE2eRtol, kE2eAtol,
                                     "cifar-sparse end-to-end logits"));
}

template <class Runner>
inline void RunFullAndCheckFinal() {
  if (!Runner::Available()) {
    GTEST_SKIP() << "backend device not available on this target";
  }
  Runner runner;
  AppData a(runner.Mr());
  PopulateAllCsr(a);
  for (int i = 1; i <= 9; ++i) runner.RunStage(a, i);
  CheckFinalPipeline(a);
}

}  // namespace cifar_sparse::testing

#define BT_DECLARE_CIFAR_SPARSE_DIFF_TESTS(SUITE, RUNNER)                                 \
  TEST(SUITE, Stage1_Conv1)  { cifar_sparse::testing::RunAndCheckStage<RUNNER>(1); }      \
  TEST(SUITE, Stage2_Pool1)  { cifar_sparse::testing::RunAndCheckStage<RUNNER>(2); }      \
  TEST(SUITE, Stage3_Conv2)  { cifar_sparse::testing::RunAndCheckStage<RUNNER>(3); }      \
  TEST(SUITE, Stage4_Pool2)  { cifar_sparse::testing::RunAndCheckStage<RUNNER>(4); }      \
  TEST(SUITE, Stage5_Conv3)  { cifar_sparse::testing::RunAndCheckStage<RUNNER>(5); }      \
  TEST(SUITE, Stage6_Conv4)  { cifar_sparse::testing::RunAndCheckStage<RUNNER>(6); }      \
  TEST(SUITE, Stage7_Conv5)  { cifar_sparse::testing::RunAndCheckStage<RUNNER>(7); }      \
  TEST(SUITE, Stage8_Pool3)  { cifar_sparse::testing::RunAndCheckStage<RUNNER>(8); }      \
  TEST(SUITE, Stage9_Linear) { cifar_sparse::testing::RunAndCheckStage<RUNNER>(9); }       \
  TEST(SUITE, EndToEnd_FinalLogits) { cifar_sparse::testing::RunFullAndCheckFinal<RUNNER>(); }
