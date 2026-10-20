#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory_resource>
#include <random>
#include <set>
#include <utility>

#include "apps/cifar-dense/cifar_dense_diff_oracle.hpp"
#include "dispatchers.hpp"
#include "platform/engine/vulkan/shaders/conv_k3s1p1_contract.hpp"
#include "platform/registry/device_registry.hpp"
#include "runtime/tests/placement_check.hpp"

// ----------------------------------------------------------------------------
// cifar-dense × Vulkan differential oracle. Vulkan uses the same AppData as
// OMP/CUDA, so the harness is identical — only the Runner differs. Each stage is
// checked against the independent double-precision conv/pool/linear reference.
// Run on an integrated-GPU box / Jetson / phone.
// ----------------------------------------------------------------------------

namespace {
struct VulkanRunner {
  // Relaxed-precision fp32 shaders: keep the looser bound.
  static constexpr float kRtol = 1e-3f;
  static constexpr float kAtol = 1e-4f;
  static constexpr float kE2eRtol = 5e-3f;
  static constexpr float kE2eAtol = 5e-3f;
  cifar_dense::vulkan::VulkanDispatcher disp;
  static bool Available() { return kiss_vk::has_integrated_gpu(); }  // skip if no iGPU
  std::pmr::memory_resource* Mr() { return disp.get_mr(); }
  void RunStage(cifar_dense::AppData& a, int stage) { disp.dispatch_stage(a, stage); }
};
}  // namespace

BT_DECLARE_CIFAR_DENSE_DIFF_TESTS(CifarDenseDiffVulkan, VulkanRunner)

// Memory-placement spec, design B3: the stages each AppData buffer declares (what the
// schedule-driven placement trusts) are exactly the stages whose dispatches bind it.
TEST(Placement, DeclaredStagesMatchBindings) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  cifar_dense::vulkan::VulkanDispatcher disp;
  auto* vk_mr = static_cast<kiss_vk::VulkanMemoryResource*>(disp.get_mr());
  bt_placement_test::RecordingDispatcher rec(vk_mr);
  cifar_dense::AppData app(rec.placement(all_stages(11)));
  const std::map<int, std::pair<const void*, const void*>> fc_weights = {
      {9, {app.u_fc1_w.data(), app.u_fc1_wq.words.data()}},
      {10, {app.u_fc2_w.data(), app.u_fc2_wq.words.data()}}};

  for (int s = 1; s <= 11; ++s) {
    disp.dispatch_stage(app, s);
    const auto& touched = disp.get_seq()->touched_buffers();
    std::set<VkBuffer> bound(touched.begin(), touched.end());
    std::set<VkBuffer> declared;
    for (const auto& [ptr, alloc] : rec.log) {
      if (std::ranges::find(alloc.stages, s) != alloc.stages.end()) {
        declared.insert(vk_mr->find_buffer(ptr));
      }
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

// new_cifar_dense_conv2d_k3s1p1 (subgroup >= 32) reads its weights as scalar work when every
// lane of a subgroup has the same output-channel block, and with a per-lane base otherwise;
// on narrower subgroups (Mali) it runs the previous branchy kernel, which these also cover. The
// model's maps (32, 16, 8) are all subgroup multiples, so the differential suite only reaches
// the uniform case; 6x6 maps (36 outputs per block, which no subgroup size divides) force the
// per-lane one, and 7x5 maps (odd width, 35 outputs per block) a ragged row as well. Checked
// against a double-precision reference, relu on.
namespace {
struct DenseConvPc {  // layout of the shader's push constants
  int32_t N, C, H, W, K, R, S, stride, padding, apply_relu;
};

// tile: new_cifar_dense_conv2d_k3s1p1_oc with that tile (cifar-gpu-kernels C1-C3, what cifar-dense
// runs on subgroups of 32 and more) instead of new_cifar_dense_conv2d_k3s1p1.
void ExpectDenseConvMatchesReference(const int h,
                                     const int w,
                                     const kiss_vk::DenseConvTile* tile = nullptr) {
  // K % OC_TILE(4) == 0: four blocks. N = 6 so every shape needs more than one 256-wide workgroup
  // on the Mali path too (4 pixels per invocation): a contract that over-counts outputs per
  // invocation then launches too few workgroups, and the missing outputs keep their -1 fill.
  // tile: N = 24, so that a channel block's invocations span more than one workgroup at up to 4
  // pixels each, and C = 5 leaves a short last pass for every staged channel count but 1 and 5.
  constexpr int kC = 5, kK = 16;
  const int kN = tile != nullptr ? 24 : 6;
  std::mt19937 rng(114514);
  std::uniform_real_distribution<float> val(-1.0f, 1.0f);

  kiss_vk::Engine engine;
  auto* mr = engine.get_mr();
  std::pmr::vector<float> in(static_cast<size_t>(kN) * kC * h * w, 0.0f, mr);
  std::pmr::vector<float> weight(static_cast<size_t>(kK) * kC * 9, 0.0f, mr);
  std::pmr::vector<float> bias(kK, 0.0f, mr);
  std::pmr::vector<float> out(static_cast<size_t>(kN) * kK * h * w, -1.0f, mr);
  for (auto& x : in) x = val(rng);
  for (auto& x : weight) x = val(rng);
  for (auto& x : bias) x = val(rng);

  auto algo = engine
                  .make_algo(tile != nullptr ? "new_cifar_dense_conv2d_k3s1p1_oc"
                                             : "new_cifar_dense_conv2d_k3s1p1")
                  ->work_group_size(256, 1, 1);
  if (tile != nullptr) {
    const int lds_floats =
        tile->c > 0 ? tile->c * kiss_vk::dense_conv_staged_floats_per_channel(tile->px) : 1;
    algo = algo->spec_constant(1, tile->oc)
               ->spec_constant(2, tile->px)
               ->spec_constant(3, tile->c)
               ->spec_constant(4, lds_floats);
  } else {
    algo = algo->subgroup_size_dependent();
  }
  algo = algo->num_buffers(4)->push_constant<DenseConvPc>()->build();
  const DenseConvPc pc{kN, kC, h, w, kK, 3, 3, 1, 1, /*apply_relu=*/1};
  auto seq = engine.make_seq();
  seq->cmd_begin();
  algo->record_bind(*seq,
                    {engine.get_buffer_info(in),
                     engine.get_buffer_info(weight),
                     engine.get_buffer_info(bias),
                     engine.get_buffer_info(out)});
  algo->record_push_constants(*seq, pc);
  if (tile != nullptr) {
    const auto grid = kiss_vk::dense_conv_k3s1p1_oc_grid(kN, kK, h, w, *tile);
    algo->record_dispatch(*seq, {grid.x, grid.y, 1});
  } else {
    const size_t invocations =
        kiss_vk::dense_conv_k3s1p1_invocations(kN, kK, h, w, engine.subgroup_size());
    algo->record_dispatch(*seq, {static_cast<uint32_t>(kiss_vk::div_ceil(invocations, 256)), 1, 1});
  }
  seq->cmd_end();
  seq->wait(seq->submit());

  for (int n = 0; n < kN; ++n) {
    for (int k = 0; k < kK; ++k) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          double acc = bias[k];
          for (int c = 0; c < kC; ++c) {
            for (int ky = 0; ky < 3; ++ky) {
              for (int kx = 0; kx < 3; ++kx) {
                const int iy = y + ky - 1, ix = x + kx - 1;
                if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                acc += static_cast<double>(weight[((k * kC + c) * 3 + ky) * 3 + kx]) *
                       in[((static_cast<size_t>(n) * kC + c) * h + iy) * w + ix];
              }
            }
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

TEST(DenseConvK3S1P1, PerLaneWeightsWhenSubgroupSpansBlocks) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectDenseConvMatchesReference(6, 6);
}

TEST(DenseConvK3S1P1, UniformWeightsWhenMapIsSubgroupMultiple) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectDenseConvMatchesReference(8, 8);
}

TEST(DenseConvK3S1P1, RaggedOddWidthMap) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectDenseConvMatchesReference(7, 5);
}

// The _oc kernel does not depend on the subgroup size, so these run on every GPU, for tiles of
// 4, 8 and 16 channels, 1, 2 and 4 pixels, and 0 (no staging), 2, 3 and 4 staged channels. The 7x5
// map ends every row in a masked tail at 2 and 4 pixels; at 1 and 2 pixels (5 and 3 invocations
// per row) workgroups also start mid-row, so a staged workgroup's first and last rows are partial.
namespace {
constexpr kiss_vk::DenseConvTile kDenseConvTiles[] = {
    {4, 1, 0}, {8, 2, 0}, {8, 4, 0}, {4, 1, 3}, {8, 2, 2}, {8, 4, 2}, {16, 2, 4}};

void ExpectDenseConvTilesMatchReference(const int h, const int w) {
  for (const auto& tile : kDenseConvTiles) {
    SCOPED_TRACE(::testing::Message() << "tile " << tile.oc << "x" << tile.px << "x" << tile.c);
    ExpectDenseConvMatchesReference(h, w, &tile);
  }
}
}  // namespace

TEST(DenseConvTiled, Map6x6) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectDenseConvTilesMatchReference(6, 6);
}

TEST(DenseConvTiled, Map8x8) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectDenseConvTilesMatchReference(8, 8);
}

TEST(DenseConvTiled, RaggedOddWidthMap) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectDenseConvTilesMatchReference(7, 5);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
