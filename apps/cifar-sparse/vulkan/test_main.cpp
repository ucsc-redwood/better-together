#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <memory_resource>
#include <random>
#include <set>
#include <vector>

#include "apps/cifar-sparse/cifar_sparse_diff_oracle.hpp"
#include "dispatchers.hpp"
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
// schedule-driven placement trusts) are exactly the stages whose dispatches bind it.
TEST(Placement, DeclaredStagesMatchBindings) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  cifar_sparse::vulkan::VulkanDispatcher disp;
  auto* vk_mr = static_cast<kiss_vk::VulkanMemoryResource*>(disp.get_mr());
  bt_placement_test::RecordingDispatcher rec(vk_mr);
  cifar_sparse::AppData app(rec.placement(all_stages(11)));

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
    EXPECT_EQ(bound, declared) << "stage " << s;
  }
}

// new_cifar_sparse_conv2d_k3s1p1 has a fast path (every lane of a subgroup shares one
// output channel; row bounds are subgroup-uniform) and a fallback for a subgroup that
// spans channels. The model's maps (32, 16, 8) are all subgroup multiples, so the
// differential suite only reaches the fast path; 6x6 maps (36 outputs per channel, which
// no subgroup size divides) force the fallback. A hand-made CSR with an empty row, checked
// against a double-precision reference.
namespace {
struct ConvPc {  // same layout as Conv2dPushConstants in dispatchers.cpp
  int32_t batch_size, in_channels, in_height, in_width, out_channels, out_height, out_width,
      kernel_size, stride, padding, relu, bias_size;
};

void ExpectSparseConvMatchesReference(const int hw) {
  constexpr int kN = 3, kC = 4, kK = 9, kEmptyRow = 4;
  std::mt19937 rng(114514);
  std::uniform_real_distribution<float> val(-1.0f, 1.0f);

  kiss_vk::Engine engine;
  auto* mr = engine.get_mr();
  std::pmr::vector<float> in(static_cast<size_t>(kN) * kC * hw * hw, 0.0f, mr);
  std::pmr::vector<float> out(static_cast<size_t>(kN) * kK * hw * hw, -1.0f, mr);
  std::pmr::vector<float> bias(kK, 0.0f, mr);
  std::pmr::vector<float> values(mr);
  std::pmr::vector<int> row_ptr(1, 0, mr);
  std::pmr::vector<int> col_idx(mr);
  for (auto& x : in) x = val(rng);
  for (auto& x : bias) x = val(rng);
  for (int k = 0; k < kK; ++k) {
    for (int flat = 0; flat < kC * 9 && k != kEmptyRow; ++flat) {
      if (val(rng) > 0.2f) continue;  // ~40% density
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
  const ConvPc pc{kN, kC, hw, hw, kK, hw, hw, 3, 1, 1, /*relu=*/1, kK};
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
  algo->record_dispatch(*seq, {static_cast<uint32_t>(kiss_vk::div_ceil(out.size(), 256)), 1, 1});
  seq->cmd_end();
  seq->wait(seq->submit());

  for (int n = 0; n < kN; ++n) {
    for (int k = 0; k < kK; ++k) {
      for (int y = 0; y < hw; ++y) {
        for (int x = 0; x < hw; ++x) {
          double acc = bias[k];
          for (int nz = row_ptr[k]; nz < row_ptr[k + 1]; ++nz) {
            const int c = col_idx[nz] / 9, ky = (col_idx[nz] % 9) / 3, kx = col_idx[nz] % 3;
            const int iy = y + ky - 1, ix = x + kx - 1;
            if (iy < 0 || iy >= hw || ix < 0 || ix >= hw) continue;
            acc += static_cast<double>(values[nz]) *
                   in[((static_cast<size_t>(n) * kC + c) * hw + iy) * hw + ix];
          }
          const double want = std::max(acc, 0.0);
          const float got = out[((static_cast<size_t>(n) * kK + k) * hw + y) * hw + x];
          ASSERT_NEAR(got, want, 1e-4 + 1e-3 * std::abs(want))
              << "hw " << hw << " n " << n << " k " << k << " y " << y << " x " << x;
        }
      }
    }
  }
}
}  // namespace

TEST(SparseConvK3S1P1, FallbackWhenSubgroupSpansChannels) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectSparseConvMatchesReference(6);
}

TEST(SparseConvK3S1P1, FastPathWhenMapIsSubgroupMultiple) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  ExpectSparseConvMatchesReference(8);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
