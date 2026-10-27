#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory_resource>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "apps/cifar-sparse/cifar_sparse_diff_oracle.hpp"
#include "apps/cifar-sparse/omp/all_kernels.hpp"
#include "dispatchers.hpp"
#include "platform/engine/vulkan/shaders/conv_k3s1p1_contract.hpp"
#include "platform/engine/vulkan/shaders/linear_bt_contract.hpp"
#include "platform/registry/device_registry.hpp"
#include "platform/util/fc_weights.hpp"
#include "platform/util/npy_loader.hpp"
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
  // Against the FP32 oracle with INT8 fc1/fc2 weights (cifar-gpu-kernels F2,
  // BT_VK_FC_WEIGHTS=int8), which are off by up to half their row's scale (max |w| / 254):
  // looser bounds for fc1, fc2 and the end-to-end logits.
  static constexpr float kInt8Rtol = 2e-2f;
  static constexpr float kInt8Atol = 1.5e-2f;
  static constexpr float kInt8E2eRtol = 2e-2f;
  static constexpr float kInt8E2eAtol = 2e-2f;
  static bool Int8() { return bt::fc_weights_from_env() == bt::FcWeights::kInt8; }
  static std::pair<float, float> StageTolerance(const int s) {
    if ((s == 9 || s == 10) && Int8()) return {kInt8Rtol, kInt8Atol};
    return {kRtol, kAtol};
  }
  static std::pair<float, float> E2eTolerance() {
    if (Int8()) return {kInt8E2eRtol, kInt8E2eAtol};
    return {kE2eRtol, kE2eAtol};
  }
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
  const std::map<int, std::pair<const void*, const void*>> fc_weights = {
      {9, {app.u_fc1_w.data(), app.u_fc1_wq.words.data()}},
      {10, {app.u_fc2_w.data(), app.u_fc2_wq.words.data()}}};

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
      VkBuffer col_idx = vk_mr->find_buffer(it->second.first);
      VkBuffer taps = vk_mr->find_buffer(it->second.second);
      EXPECT_NE(bound.contains(col_idx), bound.contains(taps)) << "stage " << s;
      declared.erase(bound.contains(col_idx) ? taps : col_idx);
    }
    // fc1 and fc2 declare their FP32 weights and the reduced-precision copy (empty, so never
    // allocated, in the default FP32 mode); a Vulkan stage binds exactly one of the two.
    if (const auto it = fc_weights.find(s);
        it != fc_weights.end() && it->second.second != nullptr) {
      VkBuffer fp32 = vk_mr->find_buffer(it->second.first);
      VkBuffer reduced = vk_mr->find_buffer(it->second.second);
      EXPECT_NE(bound.contains(fp32), bound.contains(reduced)) << "stage " << s;
      declared.erase(bound.contains(fp32) ? reduced : fp32);
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

// new_cifar_dense_linear_bt{,_portable}.comp (shared with cifar-dense) compute kLinearBtRows x FPR
// output features and kLinearBtBatchTile images per workgroup. The model's layers are
// multiples of both (4096 and 10 features at N = 16 or 128); this shape is not: an odd
// output-feature count, a ragged last batch tile and a ragged last input chunk. Both modules:
// the clustered reduction (what every fleet GPU runs) and the portable one (shared-memory
// reduction, what a device without clustered subgroup operations gets; it runs anywhere, so every
// GPU tests it), and both weight precisions (cifar-gpu-kernels F2), against a reference that uses
// the weights as the kernel decodes them (bt::FcWeightCopy::dequantized), so the FP32 bound holds
// for both.
namespace {
void ExpectLinearBtMatchesReference(const bool clustered,
                                    const bt::FcWeights weights = bt::FcWeights::kFp32,
                                    const int fpr = 2) {
  constexpr int kN = 21, kIn = 100, kOut = 37;  // 25 input vec4s: less than one 32-vec4 chunk
  std::mt19937 rng(114514);
  std::uniform_real_distribution<float> val(-1.0f, 1.0f);

  kiss_vk::Engine engine;
  if (clustered && !engine.compute_limits().subgroup_clustered) {
    GTEST_SKIP() << "no clustered subgroup operations (the dispatchers use the portable module)";
  }
  auto* mr = engine.get_mr();
  std::pmr::vector<float> in(static_cast<size_t>(kN) * kIn, 0.0f, mr);
  std::pmr::vector<float> weight(static_cast<size_t>(kOut) * kIn, 0.0f, mr);
  std::pmr::vector<float> bias(kOut, 0.0f, mr);
  std::pmr::vector<float> out(static_cast<size_t>(kN) * kOut, -1.0f, mr);
  for (auto& x : in) x = val(rng);
  for (auto& x : weight) x = val(rng);
  for (auto& x : bias) x = val(rng);
  bt::FcWeightCopy copy(weights, kOut, kIn, mr);
  copy.quantize(weight.data());
  const auto w = [&](const int o, const int i) {
    return weights == bt::FcWeights::kFp32 ? weight[static_cast<size_t>(o) * kIn + i]
                                           : copy.dequantized(o, i);
  };

  struct Pc {  // layout of the shader's push constants
    int32_t N, in_features, out_features, apply_relu;
  };
  auto algo = engine.make_algo(kiss_vk::linear_bt_shader(clustered))->work_group_size(256, 1, 1);
  if (clustered) algo = algo->subgroup_size_dependent();
  algo = algo->spec_constant(2, static_cast<uint32_t>(weights))
             ->spec_constant(3, static_cast<uint32_t>(fpr))
             ->num_buffers(4)
             ->push_constant<Pc>()
             ->build();
  auto seq = engine.make_seq();
  seq->cmd_begin();
  algo->record_bind(*seq,
                    {engine.get_buffer_info(in),
                     weights == bt::FcWeights::kFp32 ? engine.get_buffer_info(weight)
                                                     : engine.get_buffer_info(copy.words),
                     engine.get_buffer_info(bias),
                     engine.get_buffer_info(out)});
  algo->record_push_constants(*seq, Pc{kN, kIn, kOut, /*apply_relu=*/1});
  algo->record_dispatch(
      *seq,
      {static_cast<uint32_t>(kiss_vk::div_ceil(kOut, kiss_vk::kLinearBtRows * fpr)),
       static_cast<uint32_t>(kiss_vk::div_ceil(kN, kiss_vk::kLinearBtBatchTile)),
       1});
  seq->cmd_end();
  seq->wait(seq->submit());

  for (int n = 0; n < kN; ++n) {
    for (int o = 0; o < kOut; ++o) {
      double acc = bias[o];
      for (int i = 0; i < kIn; ++i) {
        acc += static_cast<double>(w(o, i)) * in[static_cast<size_t>(n) * kIn + i];
      }
      const double want = std::max(acc, 0.0);
      ASSERT_NEAR(out[static_cast<size_t>(n) * kOut + o], want, 1e-4 + 1e-3 * std::abs(want))
          << bt::to_string(weights) << " fpr " << fpr << " n " << n << " o " << o;
    }
  }
}
}  // namespace

TEST(LinearBt, RaggedFeatureBatchAndChunkEdges) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectLinearBtMatchesReference(/*clustered=*/true);
}

TEST(LinearBt, RaggedFeatureBatchAndChunkEdgesPortableModule) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectLinearBtMatchesReference(/*clustered=*/false);
}

TEST(LinearBt, Int8Weights) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectLinearBtMatchesReference(/*clustered=*/true, bt::FcWeights::kInt8);
}

// What INT8 runs on subgroups of 32 and more: 4 features per row (it runs on every GPU here).
TEST(LinearBt, Int8WeightsFourFeaturesPerRow) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectLinearBtMatchesReference(/*clustered=*/true, bt::FcWeights::kInt8, /*fpr=*/4);
}

TEST(LinearBt, Int8WeightsPortableModule) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectLinearBtMatchesReference(/*clustered=*/false, bt::FcWeights::kInt8);
}

TEST(LinearBt, Int8WeightsFourFeaturesPerRowPortableModule) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectLinearBtMatchesReference(/*clustered=*/false, bt::FcWeights::kInt8, /*fpr=*/4);
}

// The module a device without clustered subgroup operations gets must be valid there: its
// SPIR-V declares no capability but Shader and has no subgroup instruction (and no subgroup-size
// constant, so its pipeline is not subgroup-size dependent). The clustered module does declare
// GroupNonUniformClustered, which is why the dispatchers choose before creating a module.
// Hardware-free: reads the baked SPIR-V.
TEST(LinearBt, PortableModuleHasNoSubgroupCapability) {
  struct Spirv {
    std::set<uint32_t> capabilities;
    bool group_ops = false;  // any OpGroupNonUniform* instruction
  };
  const auto scan = [](const char* name) {
    const auto words = kiss_vk::load_shader_spirv(name);
    Spirv r;
    for (size_t i = 5; i < words.size();) {  // after the 5-word header
      const uint32_t count = words[i] >> 16, opcode = words[i] & 0xffffu;
      if (opcode == 17) r.capabilities.insert(words[i + 1]);   // OpCapability
      if (opcode >= 333 && opcode <= 366) r.group_ops = true;  // OpGroupNonUniformElect..
      i += count > 0 ? count : 1;
    }
    return r;
  };
  constexpr uint32_t kShader = 1, kGroupNonUniformClustered = 67;
  const Spirv portable = scan(kiss_vk::linear_bt_shader(false));
  EXPECT_EQ(portable.capabilities, std::set<uint32_t>{kShader});
  EXPECT_FALSE(portable.group_ops);
  const auto portable_spec_ids =
      kiss_vk::reflect_spirv(kiss_vk::load_shader_spirv(kiss_vk::linear_bt_shader(false))).spec_ids;
  EXPECT_FALSE(portable_spec_ids.contains(kiss_vk::kSubgroupSizeSpecId));
  EXPECT_TRUE(portable_spec_ids.contains(2) && portable_spec_ids.contains(3));  // WEIGHTS, FPR

  const Spirv clustered = scan(kiss_vk::linear_bt_shader(true));
  EXPECT_TRUE(clustered.capabilities.contains(kGroupNonUniformClustered));
  EXPECT_TRUE(clustered.group_ops);
}

// cifar-gpu-kernels F2, spec "Reduced-precision weights are bounded and opt-in": end-task accuracy
// of the Vulkan pipeline on the real weights (BT_WEIGHTS_DIR) over every exported test image,
// with FP32 and INT8 fc1/fc2 weights (BT_VK_FC_WEIGHTS, set here per AppData). INT8 must stay
// within 0.5 points of FP32 on the same images. The export has 128 images,
// so one image is 0.78 points and the bound asks for the same count as FP32. Skips without
// BT_WEIGHTS_DIR.
namespace {
// Fraction of the exported test images (one batch of 128) the Vulkan pipeline classifies
// correctly, fc weights in `mode`.
double SparseVkAccuracy(cifar_sparse::vulkan::VulkanDispatcher& disp,
                        const char* mode,
                        const std::vector<int32_t>& labels) {
  setenv("BT_VK_FC_WEIGHTS", mode, 1);
  cifar_sparse::AppData app(disp.get_mr());  // loads the real weights and the 128 images
  disp.dispatch_multi_stage(app, 1, 11);
  const int k = app.u_fc3_out.d1();
  int correct = 0;
  for (int i = 0; i < app.u_fc3_out.d0(); ++i) {
    const float* row = app.u_fc3_out.data() + static_cast<size_t>(i) * k;
    if (std::max_element(row, row + k) - row == labels[i]) ++correct;
  }
  return static_cast<double>(correct) / app.u_fc3_out.d0();
}
}  // namespace

TEST(CifarSparseVk, RealWeights_EndTaskAccuracy) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  const char* dir = std::getenv("BT_WEIGHTS_DIR");
  if (dir == nullptr) GTEST_SKIP() << "BT_WEIGHTS_DIR not set";
  constexpr size_t kImages = cifar_sparse::AppData::BATCH_SIZE;  // the whole exported test batch
  std::vector<int32_t> labels(kImages);
  bt::npy::load_prefix(std::string(dir) + "/test_labels.npy", "<i4", {kImages}, labels.data());

  const char* saved = std::getenv("BT_VK_FC_WEIGHTS");
  const std::string restore = saved != nullptr ? saved : "";
  cifar_sparse::vulkan::VulkanDispatcher disp;
  const double fp32 = SparseVkAccuracy(disp, "fp32", labels);
  const double int8 = SparseVkAccuracy(disp, "int8", labels);
  if (saved != nullptr) {
    setenv("BT_VK_FC_WEIGHTS", restore.c_str(), 1);
  } else {
    unsetenv("BT_VK_FC_WEIGHTS");
  }
  std::printf(
      "cifar-sparse Vulkan real-weight accuracy over %zu images: fp32 %.4f, int8 %.4f (%+.2f "
      "points)\n",
      kImages,
      fp32,
      int8,
      100 * (int8 - fp32));
  EXPECT_GE(fp32, 0.85);
  EXPECT_LE(std::abs(int8 - fp32), 0.005) << "int8 fc weights";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
