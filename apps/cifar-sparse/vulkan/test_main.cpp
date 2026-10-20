#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory_resource>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "apps/cifar-sparse/cifar_sparse_diff_oracle.hpp"
#include "apps/cifar-sparse/omp/all_kernels.hpp"
#include "dispatchers.hpp"
#include "platform/engine/vulkan/shaders/conv_k3s1p1_contract.hpp"
#include "platform/engine/vulkan/shaders/linear_bt_contract.hpp"
#include "platform/registry/device_registry.hpp"
#include "runtime/tests/placement_check.hpp"

// ----------------------------------------------------------------------------
// cifar-sparse × Vulkan differential oracle. Vulkan uses the same AppData as
// OMP/CUDA; the shipped AppData builds a real CSR in its ctor
// (CSRMatrix::build_from_dense).
// Each stage is checked against the densified-CSR double-precision reference.
// Run on an integrated-GPU box / Jetson / phone.
// ----------------------------------------------------------------------------

namespace {
struct VulkanRunner {
  // Relaxed-precision fp32 shaders: keep the looser bound.
  static constexpr float kRtol = 1e-3f;
  static constexpr float kAtol = 1e-4f;
  static constexpr float kE2eRtol = 5e-3f;
  static constexpr float kE2eAtol = 5e-3f;
  cifar_sparse::vulkan::VulkanDispatcher disp;
  static bool Available() { return kiss_vk::has_integrated_gpu(); }  // skip if no iGPU
  std::pmr::memory_resource* Mr() { return disp.get_mr(); }
  // cifar-sparse's Vulkan dispatcher exposes only dispatch_multi_stage; run one
  // stage as a width-1 range.
  void RunStage(cifar_sparse::AppData& a, int stage) { disp.dispatch_multi_stage(a, stage, stage); }
};
}  // namespace

BT_DECLARE_CIFAR_SPARSE_DIFF_TESTS(CifarSparseDiffVulkan, VulkanRunner)

// Memory-placement spec, design B3: the stages each AppData buffer declares (what the
// schedule-driven placement trusts) are exactly the stages whose dispatches bind it. A conv
// declares both its CSR col_idx (OMP, CUDA, the Mali path) and its Vulkan taps; a Vulkan
// conv binds exactly one of the two.
TEST(Placement, DeclaredStagesMatchBindings) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  cifar_sparse::vulkan::VulkanDispatcher disp;
  auto* vk_mr = static_cast<kiss_vk::VulkanMemoryResource*>(disp.get_mr());
  bt_placement_test::RecordingDispatcher rec(vk_mr);
  cifar_sparse::AppData app(rec.placement(all_stages(11)));
  const std::map<int, std::pair<const int*, const int*>> csr_index = {
      {1, {app.conv1_sparse.col_idx.data(), app.conv1_taps.data()}},
      {3, {app.conv2_sparse.col_idx.data(), app.conv2_taps.data()}},
      {5, {app.conv3_sparse.col_idx.data(), app.conv3_taps.data()}},
      {6, {app.conv4_sparse.col_idx.data(), app.conv4_taps.data()}},
      {7, {app.conv5_sparse.col_idx.data(), app.conv5_taps.data()}}};

  for (int s = 1; s <= 11; ++s) {
    disp.dispatch_multi_stage(app, s, s);
    const auto& touched = disp.get_seq()->touched_buffers();
    std::set<VkBuffer> bound(touched.begin(), touched.end());
    std::set<VkBuffer> declared;
    for (const auto& [ptr, alloc] : rec.log) {
      if (std::ranges::find(alloc.stages, s) != alloc.stages.end()) {
        declared.insert(vk_mr->find_buffer(ptr));
      }
    }
    if (const auto it = csr_index.find(s); it != csr_index.end()) {
      const VkBuffer col_idx = vk_mr->find_buffer(it->second.first);
      const VkBuffer taps = vk_mr->find_buffer(it->second.second);
      EXPECT_NE(bound.contains(col_idx), bound.contains(taps)) << "stage " << s;
      declared.erase(bound.contains(col_idx) ? taps : col_idx);
    }
    EXPECT_EQ(bound, declared) << "stage " << s;
  }
}

// new_cifar_sparse_conv2d_k3s1p1 is the Mali path (PX adjacent pixels of a row per invocation),
// which cifar-sparse runs below subgroup 32. It does not depend on the subgroup size, so every GPU
// runs these tests, not just the Pixel. 6x6 and 8x8 maps, and 7x5 for an odd width that is not a
// multiple of PX (a masked tail). A hand-made CSR with an empty row and a row whose nonzeros all
// sit in the kernel's right column (at the right edge every tap reads the padded border), checked
// against a double-precision reference.
namespace {
struct ConvPc {  // same layout as Conv2dPushConstants in dispatchers.cpp
  int32_t batch_size, in_channels, in_height, in_width, out_channels, out_height, out_width,
      kernel_size, stride, padding, relu, bias_size;
};

void ExpectSparseConvMatchesReference(const int h, const int w) {
  constexpr int kN = 3, kC = 4, kK = 9, kEmptyRow = 4, kBorderRow = 6;
  std::mt19937 rng(114514);
  std::uniform_real_distribution<float> val(-1.0f, 1.0f);

  kiss_vk::Engine engine;
  auto* mr = engine.get_mr();
  std::pmr::vector<float> in(static_cast<size_t>(kN) * kC * h * w, 0.0f, mr);
  std::pmr::vector<float> out(static_cast<size_t>(kN) * kK * h * w, -1.0f, mr);
  std::pmr::vector<float> bias(kK, 0.0f, mr);
  std::pmr::vector<float> values(mr);
  std::pmr::vector<int> row_ptr(1, 0, mr);
  std::pmr::vector<int> col_idx(mr);
  for (auto& x : in) x = val(rng);
  for (auto& x : bias) x = val(rng);
  for (int k = 0; k < kK; ++k) {
    for (int flat = 0; flat < kC * 9 && k != kEmptyRow; ++flat) {
      // kBorderRow keeps only kx == 2; the others ~40% density.
      if (k == kBorderRow ? flat % 3 != 2 : val(rng) > 0.2f) continue;
      values.push_back(val(rng));
      col_idx.push_back(flat);
    }
    row_ptr.push_back(static_cast<int>(values.size()));
  }

  auto algo = engine.make_algo("new_cifar_sparse_conv2d_k3s1p1")
                  ->work_group_size(256, 1, 1)
                  ->num_buffers(6)
                  ->push_constant<ConvPc>()
                  ->build();
  const ConvPc pc{kN, kC, h, w, kK, h, w, 3, 1, 1, /*relu=*/1, kK};
  auto seq = engine.make_seq();
  seq->cmd_begin();
  algo->record_bind(*seq,
                    {engine.get_buffer_info(in),
                     engine.get_buffer_info(out),
                     engine.get_buffer_info(values),
                     engine.get_buffer_info(row_ptr),
                     engine.get_buffer_info(col_idx),
                     engine.get_buffer_info(bias)});
  algo->record_push_constants(*seq, pc);
  const size_t invocations = kiss_vk::sparse_conv_k3s1p1_mali_invocations(kN, kK, h, w);
  algo->record_dispatch(*seq, {static_cast<uint32_t>(kiss_vk::div_ceil(invocations, 256)), 1, 1});
  seq->cmd_end();
  seq->wait(seq->submit());

  for (int n = 0; n < kN; ++n) {
    for (int k = 0; k < kK; ++k) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          double acc = bias[k];
          for (int nz = row_ptr[k]; nz < row_ptr[k + 1]; ++nz) {
            const int c = col_idx[nz] / 9, ky = (col_idx[nz] % 9) / 3, kx = col_idx[nz] % 3;
            const int iy = y + ky - 1, ix = x + kx - 1;
            if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
            acc += static_cast<double>(values[nz]) *
                   in[((static_cast<size_t>(n) * kC + c) * h + iy) * w + ix];
          }
          // Rows that contribute nothing here: bias plus ReLU only.
          if (k == kEmptyRow || (k == kBorderRow && x == w - 1)) {
            ASSERT_EQ(acc, static_cast<double>(bias[k]));
          }
          const double want = std::max(acc, 0.0);
          const float got = out[((static_cast<size_t>(n) * kK + k) * h + y) * w + x];
          ASSERT_NEAR(got, want, 1e-4 + 1e-3 * std::abs(want))
              << h << "x" << w << " n " << n << " k " << k << " y " << y << " x " << x;
        }
      }
    }
  }
}
}  // namespace

TEST(SparseConvK3S1P1, Map6x6) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectSparseConvMatchesReference(6, 6);
}

TEST(SparseConvK3S1P1, Map8x8) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectSparseConvMatchesReference(8, 8);
}

TEST(SparseConvK3S1P1, RaggedOddWidthMap) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectSparseConvMatchesReference(7, 5);
}

// cifar-gpu-kernels spec "Empty and border-only rows": the per-channel kernel
// (new_cifar_sparse_conv2d_k3s1p1_oc, the subgroup >= 32 path) against the OMP kernel, on a CSR
// with an empty row, a row whose taps all sit in the kernel's right column and one whose taps all
// sit in its top row: at the right edge, resp. the top edge, every tap of those rows reads the
// padded border, so the output there is bias plus ReLU only. 13x11 maps need two workgroups per
// channel, the last one ragged. Row 8 keeps every tap (1152 nonzeros), more than one STAGE_ROW
// chunk; each map runs with STAGE_ROW off and on.
namespace {
void ExpectPerChannelConvMatchesOmp(const int h, const int w, const bool stage_row) {
  constexpr int kN = 3, kC = 128, kK = 9, kEmptyRow = 4, kRightRow = 6, kTopRow = 7, kFullRow = 8;
  std::mt19937 rng(114514);
  std::uniform_real_distribution<float> val(-1.0f, 1.0f);

  kiss_vk::Engine engine;
  auto* mr = engine.get_mr();
  std::pmr::vector<float> in(static_cast<size_t>(kN) * kC * h * w, 0.0f, mr);
  std::pmr::vector<float> out(static_cast<size_t>(kN) * kK * h * w, -1.0f, mr);
  std::pmr::vector<float> bias(kK, 0.0f, mr);
  std::pmr::vector<float> values(mr);
  std::pmr::vector<int> row_ptr(1, 0, mr);
  std::pmr::vector<int> col_idx(mr);
  for (auto& x : in) x = val(rng);
  for (auto& x : bias) x = val(rng);
  for (int k = 0; k < kK; ++k) {
    for (int flat = 0; flat < kC * 9 && k != kEmptyRow; ++flat) {
      const int tap = flat % 9;
      const bool keep = k == kRightRow  ? tap % 3 == 2
                        : k == kTopRow  ? tap / 3 == 0
                        : k == kFullRow ? true
                                        : val(rng) > 0.2f;
      if (!keep) continue;
      values.push_back(val(rng));
      col_idx.push_back(flat);
    }
    row_ptr.push_back(static_cast<int>(values.size()));
  }
  std::pmr::vector<int> taps(col_idx.size(), 0, mr);
  cifar_sparse::pack_k3s1p1_taps(
      col_idx.data(), static_cast<int>(col_idx.size()), h, w, taps.data());

  auto algo = engine.make_algo("new_cifar_sparse_conv2d_k3s1p1_oc")
                  ->work_group_size(256, 1, 1)
                  ->spec_constant(1, stage_row)
                  ->num_buffers(6)
                  ->push_constant<ConvPc>()
                  ->build();
  const ConvPc pc{kN, kC, h, w, kK, h, w, 3, 1, 1, /*relu=*/1, kK};
  auto seq = engine.make_seq();
  seq->cmd_begin();
  algo->record_bind(*seq,
                    {engine.get_buffer_info(in),
                     engine.get_buffer_info(out),
                     engine.get_buffer_info(values),
                     engine.get_buffer_info(row_ptr),
                     engine.get_buffer_info(taps),
                     engine.get_buffer_info(bias)});
  algo->record_push_constants(*seq, pc);
  algo->record_dispatch(
      *seq,
      {static_cast<uint32_t>(kiss_vk::div_ceil(kN * h * w, 256)), static_cast<uint32_t>(kK), 1});
  seq->cmd_end();
  seq->wait(seq->submit());

  std::vector<float> ref(out.size());
  cifar_sparse::omp::conv2d_omp_batched(in.data(),
                                        kN,
                                        kC,
                                        h,
                                        w,
                                        values.data(),
                                        row_ptr.data(),
                                        col_idx.data(),
                                        kK,
                                        bias.data(),
                                        kK,
                                        3,
                                        1,
                                        1,
                                        /*relu=*/true,
                                        ref.data());
  for (int n = 0; n < kN; ++n) {
    for (int k = 0; k < kK; ++k) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const size_t i = ((static_cast<size_t>(n) * kK + k) * h + y) * w + x;
          if (k == kEmptyRow || (k == kRightRow && x == w - 1) || (k == kTopRow && y == 0)) {
            ASSERT_EQ(out[i], std::max(bias[k], 0.0f))
                << h << "x" << w << " stage_row " << stage_row << " n " << n << " k " << k << " y "
                << y << " x " << x;
          }
          ASSERT_NEAR(out[i], ref[i], 1e-4 + 1e-3 * std::abs(ref[i]))
              << h << "x" << w << " stage_row " << stage_row << " n " << n << " k " << k << " y "
              << y << " x " << x;
        }
      }
    }
  }
}
}  // namespace

TEST(SparseConvPerChannel, EmptyAndBorderRows8x8) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  for (const bool stage_row : {false, true}) ExpectPerChannelConvMatchesOmp(8, 8, stage_row);
}

TEST(SparseConvPerChannel, EmptyAndBorderRowsOddWidth) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  for (const bool stage_row : {false, true}) ExpectPerChannelConvMatchesOmp(7, 5, stage_row);
}

TEST(SparseConvPerChannel, EmptyAndBorderRowsTwoWorkgroups) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  for (const bool stage_row : {false, true}) ExpectPerChannelConvMatchesOmp(13, 11, stage_row);
}

// The Vulkan conv taps (cifar-gpu-kernels S2) decode back to their CSR's col_idx for every
// nonzero of the five convs, and each layer's largest offset fits the reserved width. The
// weights are the synthetic ones, or the real export when BT_WEIGHTS_DIR is set.
TEST(SparseConvTaps, DecodeToColIdx) {
  const cifar_sparse::AppData app(std::pmr::new_delete_resource());
  const struct {
    const char* name;
    const cifar_sparse::CSRMatrix& csr;
    const std::pmr::vector<int>& taps;
    int hw;
  } layers[] = {{"conv1", app.conv1_sparse, app.conv1_taps, 32},
                {"conv2", app.conv2_sparse, app.conv2_taps, 16},
                {"conv3", app.conv3_sparse, app.conv3_taps, 8},
                {"conv4", app.conv4_sparse, app.conv4_taps, 8},
                {"conv5", app.conv5_sparse, app.conv5_taps, 8}};
  std::printf("taps of the %s weights:\n", std::getenv("BT_WEIGHTS_DIR") ? "real" : "synthetic");
  for (const auto& l : layers) {
    ASSERT_GT(l.csr.nnz, 0) << l.name;
    int max_offset = 0;
    for (int nz = 0; nz < l.csr.nnz; ++nz) {
      const int tap = l.taps[nz] & ((1 << cifar_sparse::kTapBits) - 1);
      const int offset = l.taps[nz] >> cifar_sparse::kTapBits;  // arithmetic: offsets go negative
      const int plane = offset - (tap / 3 - 1) * l.hw - (tap % 3 - 1);
      ASSERT_LT(tap, 9) << l.name << " nz " << nz;
      ASSERT_EQ(plane % (l.hw * l.hw), 0) << l.name << " nz " << nz;
      ASSERT_EQ(plane / (l.hw * l.hw) * 9 + tap, l.csr.col_idx[nz]) << l.name << " nz " << nz;
      max_offset = std::max(max_offset, std::abs(offset));
    }
    std::printf("  %s: nnz %d, max |offset| %d < 2^%d\n",
                l.name,
                l.csr.nnz,
                max_offset,
                cifar_sparse::kTapOffsetBits);
    EXPECT_LT(max_offset, 1 << cifar_sparse::kTapOffsetBits) << l.name;
  }
}

// new_cifar_dense_linear_bt.comp (shared with cifar-dense) computes kLinearBtFeaturesPerGroup
// output features and kLinearBtBatchTile images per workgroup. The model's layers are
// multiples of both (4096 and 10 features at N = 16 or 128); this shape is not: an odd
// output-feature count, a ragged last batch tile and a ragged last input chunk.
TEST(LinearBt, RaggedFeatureBatchAndChunkEdges) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  constexpr int kN = 21, kIn = 100, kOut = 37;  // 25 input vec4s: less than one 32-vec4 chunk
  std::mt19937 rng(114514);
  std::uniform_real_distribution<float> val(-1.0f, 1.0f);

  kiss_vk::Engine engine;
  auto* mr = engine.get_mr();
  std::pmr::vector<float> in(static_cast<size_t>(kN) * kIn, 0.0f, mr);
  std::pmr::vector<float> weight(static_cast<size_t>(kOut) * kIn, 0.0f, mr);
  std::pmr::vector<float> bias(kOut, 0.0f, mr);
  std::pmr::vector<float> out(static_cast<size_t>(kN) * kOut, -1.0f, mr);
  for (auto& x : in) x = val(rng);
  for (auto& x : weight) x = val(rng);
  for (auto& x : bias) x = val(rng);

  struct Pc {  // layout of the shader's push constants
    int32_t N, in_features, out_features, apply_relu;
  };
  auto algo = engine.make_algo("new_cifar_dense_linear_bt")
                  ->work_group_size(256, 1, 1)
                  ->num_buffers(4)
                  ->push_constant<Pc>()
                  ->build();
  auto seq = engine.make_seq();
  seq->cmd_begin();
  algo->record_bind(*seq,
                    {engine.get_buffer_info(in),
                     engine.get_buffer_info(weight),
                     engine.get_buffer_info(bias),
                     engine.get_buffer_info(out)});
  algo->record_push_constants(*seq, Pc{kN, kIn, kOut, /*apply_relu=*/1});
  algo->record_dispatch(
      *seq,
      {static_cast<uint32_t>(kiss_vk::div_ceil(kOut, kiss_vk::kLinearBtFeaturesPerGroup)),
       static_cast<uint32_t>(kiss_vk::div_ceil(kN, kiss_vk::kLinearBtBatchTile)),
       1});
  seq->cmd_end();
  seq->wait(seq->submit());

  for (int n = 0; n < kN; ++n) {
    for (int o = 0; o < kOut; ++o) {
      double acc = bias[o];
      for (int i = 0; i < kIn; ++i) {
        acc += static_cast<double>(weight[static_cast<size_t>(o) * kIn + i]) *
               in[static_cast<size_t>(n) * kIn + i];
      }
      const double want = std::max(acc, 0.0);
      ASSERT_NEAR(out[static_cast<size_t>(n) * kOut + o], want, 1e-4 + 1e-3 * std::abs(want))
          << "n " << n << " o " << o;
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
