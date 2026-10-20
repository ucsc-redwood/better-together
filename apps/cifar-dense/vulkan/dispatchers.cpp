#include "dispatchers.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "platform/engine/vulkan/env.hpp"
#include "platform/engine/vulkan/shaders/conv_k3s1p1_contract.hpp"
#include "platform/engine/vulkan/shaders/linear_bt_contract.hpp"
#include "platform/util/debug_logger.hpp"

namespace cifar_dense::vulkan {

// // Convolution parameters
// constexpr int kKernelSize = 3;
// constexpr int kStride = 1;
// constexpr int kPadding = 1;

// // Pooling parameters
// constexpr int kPoolSize = 2;
// constexpr int kPoolStride = 2;

// constexpr bool kRelu = true;

// Push constants for the conv2d shader
struct Conv2dPushConstants_v2 {
  int32_t N;           // Batch size
  int32_t C;           // Number of input channels
  int32_t H;           // Input height
  int32_t W;           // Input width
  int32_t K;           // Number of output channels
  int32_t R;           // Kernel height
  int32_t S;           // Kernel width
  int32_t stride;      // Convolution stride
  int32_t padding;     // Convolution padding
  int32_t apply_relu;  // 1 to apply ReLU, 0 otherwise
};

// Push constants for the maxpool shader
struct MaxpoolPushConstants_v2 {
  int32_t N;       // Batch size
  int32_t C;       // Number of channels
  int32_t H;       // Input height
  int32_t W;       // Input width
  int32_t pool_h;  // Pooling kernel height
  int32_t pool_w;  // Pooling kernel width
  int32_t stride;  // Pooling stride
};

// Push constants for the linear shader
struct LinearPushConstants_v2 {
  int32_t N;             // Batch size
  int32_t in_features;   // Number of input features
  int32_t out_features;  // Number of output features
  int32_t apply_relu;    // 1 to apply ReLU, 0 otherwise
};

// ----------------------------------------------------------------------------
// Constructor (v2)
// ----------------------------------------------------------------------------

namespace {

using ConvTiles = std::array<std::optional<kiss_vk::DenseConvTile>, 5>;

// The tile of new_cifar_dense_conv2d_k3s1p1_oc per conv, conv1..conv5; nullopt keeps the previous
// kernel. Picked by timing on the Samsung's Xclipse 940 (cifar-gpu-kernels 2.6): no tile was
// faster on conv1, staging was 2-3.5x slower, and the governor clocks the most efficient tiles
// down, so each conv has the tile that was fastest in wall time.
constexpr ConvTiles kDefaultConvTiles = {std::nullopt,
                                         kiss_vk::DenseConvTile{8, 4, 0},
                                         kiss_vk::DenseConvTile{16, 4, 0},
                                         kiss_vk::DenseConvTile{16, 4, 0},
                                         kiss_vk::DenseConvTile{8, 1, 0}};

std::string tile_name(const kiss_vk::DenseConvTile& t) {
  return std::to_string(t.oc) + "x" + std::to_string(t.px) + "x" + std::to_string(t.c);
}

// BT_VK_DENSE_CONV_TILE=<t> or <t1>,<t2>,<t3>,<t4>,<t5> (one tile for every conv, or one each for
// conv1..conv5), where a tile is <oc>x<px>x<c> (output channels and pixels per invocation, input
// channels staged per pass, 0: none) or flat (the previous kernel). Unset: kDefaultConvTiles.
ConvTiles dense_conv_tiles_from_env() {
  const auto v = kiss_vk::env::get("BT_VK_DENSE_CONV_TILE");
  if (!v) return kDefaultConvTiles;
  const std::string value(*v);
  const auto fail = [&value]() {
    return std::runtime_error("BT_VK_DENSE_CONV_TILE=" + value +
                              ": expected one or five comma-separated tiles, each "
                              "<oc>x<px>x<c> (oc and px >= 1, c >= 0) or flat");
  };
  std::vector<std::optional<kiss_vk::DenseConvTile>> tiles;
  size_t pos = 0;
  while (true) {
    const size_t comma = value.find(',', pos);
    const std::string s = value.substr(pos, comma == std::string::npos ? comma : comma - pos);
    if (s == "flat") {
      tiles.emplace_back(std::nullopt);
    } else {
      kiss_vk::DenseConvTile t;
      int end = 0;
      if (std::sscanf(s.c_str(), "%dx%dx%d%n", &t.oc, &t.px, &t.c, &end) != 3 ||
          end != static_cast<int>(s.size()) || t.oc < 1 || t.px < 1 || t.c < 0) {
        throw fail();
      }
      tiles.emplace_back(t);
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  if (tiles.size() == 1) return {tiles[0], tiles[0], tiles[0], tiles[0], tiles[0]};
  if (tiles.size() != 5) throw fail();
  return {tiles[0], tiles[1], tiles[2], tiles[3], tiles[4]};
}

}  // namespace

VulkanDispatcher::VulkanDispatcher()
    : engine(), default_seq(engine.make_seq()), conv_tiles(dense_conv_tiles_from_env()) {
  spdlog::debug("VulkanDispatcher::VulkanDispatcher(), Initializing VulkanDispatcher");

  // conv2d is reused by stages 1,3,5,6,7 and maxpool by stages 2,4,8; record_bind
  // captures each stage's buffers per dispatch, so they can share one command buffer.
  auto conv2d_algo = engine.make_algo("new_cifar_dense_conv2d")
                         ->work_group_size(256, 1, 1)
                         ->num_buffers(4)  // Input, Weight, Bias, Output
                         ->push_constant<Conv2dPushConstants_v2>()
                         ->build();

  cached_algorithms.try_emplace("conv2d", std::move(conv2d_algo));

  // 3x3/stride-1/pad-1 conv specialization (every conv in AlexNetCIFAR): fully
  // unrolled taps with hoisted edge masks, register-blocked OC_TILE=4 output
  // channels per thread so the 9 input taps are reused across 4 weight rows.
  // Selected per stage when R==S==3, stride==1, padding==1 and K % 4 == 0.
  // subgroup_size_dependent: the shader picks its path from the subgroup size (spec const 0).
  auto conv2d_k3s1p1_algo = engine.make_algo("new_cifar_dense_conv2d_k3s1p1")
                                ->work_group_size(256, 1, 1)
                                ->subgroup_size_dependent()
                                ->num_buffers(4)  // Input, Weight, Bias, Output
                                ->push_constant<Conv2dPushConstants_v2>()
                                ->build();

  cached_algorithms.try_emplace("conv2d_k3s1p1", std::move(conv2d_k3s1p1_algo));

  // The same conv on subgroups of 32 and more, one block of output channels per workgroup
  // (cifar-gpu-kernels C1), one pipeline per tile the convs use (C2/C3).
  const uint32_t max_shared =
      engine.get_physical_device().getProperties().limits.maxComputeSharedMemorySize;
  for (size_t i = 0; i < conv_tiles.size(); ++i) {
    const auto& slot = conv_tiles[i];
    if (!slot.has_value()) continue;
    const auto& tile = *slot;
    const std::string name = "conv2d_k3s1p1_oc/" + tile_name(tile);
    if (!cached_algorithms.contains(name)) {
      const int lds_floats =
          tile.c > 0 ? tile.c * kiss_vk::dense_conv_staged_floats_per_channel(tile.px) : 1;
      if (static_cast<uint32_t>(lds_floats) * sizeof(float) > max_shared) {
        throw std::runtime_error("BT_VK_DENSE_CONV_TILE: " + tile_name(tile) + " needs " +
                                 std::to_string(lds_floats * sizeof(float)) +
                                 " bytes of shared memory, the device has " +
                                 std::to_string(max_shared));
      }
      cached_algorithms.try_emplace(name,
                                    engine.make_algo("new_cifar_dense_conv2d_k3s1p1_oc")
                                        ->work_group_size(256, 1, 1)
                                        ->spec_constant(1, tile.oc)
                                        ->spec_constant(2, tile.px)
                                        ->spec_constant(3, tile.c)
                                        ->spec_constant(4, lds_floats)
                                        ->num_buffers(4)  // Input, Weight, Bias, Output
                                        ->push_constant<Conv2dPushConstants_v2>()
                                        ->build());
    }
    conv_oc_algos[i] = cached_algorithms.at(name).get();
  }

  // Create algorithm for maxpool
  auto maxpool_algo = engine.make_algo("new_cifar_dense_maxpool")
                          ->work_group_size(256, 1, 1)
                          ->num_buffers(2)  // Input, Output
                          ->push_constant<MaxpoolPushConstants_v2>()
                          ->build();

  cached_algorithms.try_emplace("maxpool", std::move(maxpool_algo));

  // Create algorithm for linear
  auto linear_algo = engine.make_algo("new_cifar_dense_linear")
                         ->work_group_size(256, 1, 1)
                         ->num_buffers(4)  // Input, Weight, Bias, Output
                         ->push_constant<LinearPushConstants_v2>()
                         ->build();

  cached_algorithms.try_emplace("linear", std::move(linear_algo));

  // Batch-tiled linear: a 16-lane workgroup row owns TWO output features, strides their
  // weight row coalesced, and accumulates ALL N (<= 16) images while the row is
  // streamed ONCE -- the generic thread-per-(n,of) shader re-streams the whole weight
  // matrix once per image and each lane scans a different 16 KB row (uncoalesced).
  // Selected per stage when N <= 16 and in_features % 4 == 0.
  // Rows reduce with subgroupClusteredAdd where the device has it (cifar-gpu-kernels F3).
  auto linear_bt_algo = engine.make_algo("new_cifar_dense_linear_bt")
                            ->work_group_size(256, 1, 1)
                            ->subgroup_size_dependent()
                            ->spec_constant(1, engine.compute_limits().subgroup_clustered ? 1 : 0)
                            ->num_buffers(4)  // Input, Weight, Bias, Output
                            ->push_constant<LinearPushConstants_v2>()
                            ->build();

  cached_algorithms.try_emplace("linear_bt", std::move(linear_bt_algo));

  // Prototype layouts of cifar-gpu-kernels F1, BT_VK_FC_LAYOUT=ko|bs (a: K outer, batch tiles
  // inner; b: batch split); one workgroup per 32 features covers the whole batch (N <= 128).
  if (const auto layout = kiss_vk::env::get("BT_VK_FC_LAYOUT")) fc_layout = *layout;
  cached_algorithms.try_emplace("linear_bt_ko",
                                engine.make_algo("new_cifar_dense_linear_bt_ko")
                                    ->work_group_size(256, 1, 1)
                                    ->subgroup_size_dependent()
                                    ->num_buffers(4)
                                    ->push_constant<LinearPushConstants_v2>()
                                    ->build());
  cached_algorithms.try_emplace("linear_bt_nr",
                                engine.make_algo("new_cifar_dense_linear_bt_nr")
                                    ->work_group_size(256, 1, 1)
                                    ->subgroup_size_dependent()
                                    ->spec_constant(1, 1)
                                    ->num_buffers(4)
                                    ->push_constant<LinearPushConstants_v2>()
                                    ->build());
  cached_algorithms.try_emplace("linear_bt_tk",
                                engine.make_algo("new_cifar_dense_linear_bt_tk")
                                    ->work_group_size(256, 1, 1)
                                    ->subgroup_size_dependent()
                                    ->num_buffers(4)
                                    ->push_constant<LinearPushConstants_v2>()
                                    ->build());
  cached_algorithms.try_emplace("linear_bt_bs",
                                engine.make_algo("new_cifar_dense_linear_bt_bs")
                                    ->work_group_size(256, 1, 1)
                                    ->num_buffers(4)
                                    ->push_constant<LinearPushConstants_v2>()
                                    ->build());
}

// ----------------------------------------------------------------------------
// run_stage_k: per-stage single-submit wrappers (bm_main per-stage device timing).
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_1(AppData& appdata) { dispatch_multi_stage(appdata, 1, 1); }
void VulkanDispatcher::run_stage_2(AppData& appdata) { dispatch_multi_stage(appdata, 2, 2); }
void VulkanDispatcher::run_stage_3(AppData& appdata) { dispatch_multi_stage(appdata, 3, 3); }
void VulkanDispatcher::run_stage_4(AppData& appdata) { dispatch_multi_stage(appdata, 4, 4); }
void VulkanDispatcher::run_stage_5(AppData& appdata) { dispatch_multi_stage(appdata, 5, 5); }
void VulkanDispatcher::run_stage_6(AppData& appdata) { dispatch_multi_stage(appdata, 6, 6); }
void VulkanDispatcher::run_stage_7(AppData& appdata) { dispatch_multi_stage(appdata, 7, 7); }
void VulkanDispatcher::run_stage_8(AppData& appdata) { dispatch_multi_stage(appdata, 8, 8); }
void VulkanDispatcher::run_stage_9(AppData& appdata) { dispatch_multi_stage(appdata, 9, 9); }
void VulkanDispatcher::run_stage_10(AppData& appdata) { dispatch_multi_stage(appdata, 10, 10); }
void VulkanDispatcher::run_stage_11(AppData& appdata) { dispatch_multi_stage(appdata, 11, 11); }

// ----------------------------------------------------------------------------
// Conv stages 1, 3, 5, 6, 7
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_conv(kiss_vk::Sequence& seq,
                                   Ndarray4D& in,
                                   Ndarray4D& weight,
                                   Ndarray1D& bias,
                                   Ndarray4D& out,
                                   const int conv) {
  const int batch_size = in.d0();
  const int in_channels = in.d1();
  const int in_height = in.d2();
  const int in_width = in.d3();

  const int out_channels = weight.d0();
  const int out_height = (in_height + 2 * kPadding - kKernelSize) / kStride + 1;
  const int out_width = (in_width + 2 * kPadding - kKernelSize) / kStride + 1;

  // 3x3/s1/p1 register-blocked specialization when shapes allow; generic fallback. With a
  // subgroup of 32 or more it runs one block of output channels per workgroup, tiled
  // (cifar-gpu-kernels C1-C3), unless the conv's tile is flat; below 32, the Mali path.
  const bool spec = kKernelSize == 3 && kStride == 1 && kPadding == 1 &&
                    out_channels % kiss_vk::kDenseConvChannelsPerInvocation == 0;
  const auto& tile = conv_tiles[conv - 1];
  const bool per_block = spec && tile && engine.subgroup_size() >= 32 &&
                         out_channels % tile->oc == 0 &&
                         (tile->c == 0 || out_width <= kiss_vk::kDenseConvStagedMaxWidth);
  auto* algo = per_block ? conv_oc_algos[conv - 1]
                         : cached_algorithms.at(spec ? "conv2d_k3s1p1" : "conv2d").get();

  const auto algo_pc = Conv2dPushConstants_v2{
      .N = batch_size,
      .C = in_channels,
      .H = in_height,
      .W = in_width,
      .K = out_channels,
      .R = kKernelSize,
      .S = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .apply_relu = kRelu ? 1 : 0,
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(in.pmr_vec()),
                        engine.get_buffer_info(weight.pmr_vec()),
                        engine.get_buffer_info(bias.pmr_vec()),
                        engine.get_buffer_info(out.pmr_vec()),
                    });
  algo->record_push_constants(seq, algo_pc);

  if (per_block) {
    const auto grid =
        kiss_vk::dense_conv_k3s1p1_oc_grid(batch_size, out_channels, out_height, out_width, *tile);
    algo->record_dispatch(seq, {grid.x, grid.y, 1});
    return;
  }
  // spec: invocations from the conv contract (OC_TILE channels, and on the Mali path PX pixels).
  const size_t total_output =
      spec ? kiss_vk::dense_conv_k3s1p1_invocations(
                 batch_size, out_channels, out_height, out_width, engine.subgroup_size())
           : static_cast<size_t>(batch_size) * out_channels * out_height * out_width;
  algo->record_dispatch(seq, {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 1 (v2) - Conv1
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_1(AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 1, &appdata);
  record_conv(seq,
              appdata.u_input,
              appdata.u_conv1_w,
              appdata.u_conv1_b,
              appdata.u_conv1_out,
              /*conv=*/1);
}

// ----------------------------------------------------------------------------
// Stage 2 (v2) - MaxPool1
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_2(AppData& appdata, kiss_vk::Sequence& seq) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 2, &appdata);

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv1_out.d0();  // Expected: 128
  const int channels = appdata.u_conv1_out.d1();    // Expected: 64
  const int in_height = appdata.u_conv1_out.d2();   // Expected: 32
  const int in_width = appdata.u_conv1_out.d3();    // Expected: 32

  const int out_height = (in_height - kPoolSize) / kPoolStride + 1;  // Expected: 16
  const int out_width = (in_width - kPoolSize) / kPoolStride + 1;    // Expected: 16

  const int total_output = batch_size * channels * out_height * out_width;

  const auto algo_pc = MaxpoolPushConstants_v2{
      .N = batch_size,
      .C = channels,
      .H = in_height,
      .W = in_width,
      .pool_h = kPoolSize,
      .pool_w = kPoolSize,
      .stride = kPoolStride,
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_conv1_out.pmr_vec()),
                        engine.get_buffer_info(appdata.u_pool1_out.pmr_vec()),
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(seq, {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 3 (v2) - Conv2
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_3(AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 3, &appdata);
  record_conv(seq,
              appdata.u_pool1_out,
              appdata.u_conv2_w,
              appdata.u_conv2_b,
              appdata.u_conv2_out,
              /*conv=*/2);
}

// ----------------------------------------------------------------------------
// Stage 4 (v2) - MaxPool2
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_4(AppData& appdata, kiss_vk::Sequence& seq) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 4, &appdata);

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv2_out.d0();  // Expected: 128
  const int channels = appdata.u_conv2_out.d1();    // Expected: 192
  const int in_height = appdata.u_conv2_out.d2();   // Expected: 16
  const int in_width = appdata.u_conv2_out.d3();    // Expected: 16

  const int out_height = (in_height - kPoolSize) / kPoolStride + 1;  // Expected: 8
  const int out_width = (in_width - kPoolSize) / kPoolStride + 1;    // Expected: 8

  const int total_output = batch_size * channels * out_height * out_width;

  const auto algo_pc = MaxpoolPushConstants_v2{
      .N = batch_size,
      .C = channels,
      .H = in_height,
      .W = in_width,
      .pool_h = kPoolSize,
      .pool_w = kPoolSize,
      .stride = kPoolStride,
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_conv2_out.pmr_vec()),
                        engine.get_buffer_info(appdata.u_pool2_out.pmr_vec()),
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(seq, {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 5 (v2) - Conv3
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_5(AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 5, &appdata);
  record_conv(seq,
              appdata.u_pool2_out,
              appdata.u_conv3_w,
              appdata.u_conv3_b,
              appdata.u_conv3_out,
              /*conv=*/3);
}

// ----------------------------------------------------------------------------
// Stage 6 (v2) - Conv4
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_6(AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 6, &appdata);
  record_conv(seq,
              appdata.u_conv3_out,
              appdata.u_conv4_w,
              appdata.u_conv4_b,
              appdata.u_conv4_out,
              /*conv=*/4);
}

// ----------------------------------------------------------------------------
// Stage 7 (v2) - Conv5
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_7(AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 7, &appdata);
  record_conv(seq,
              appdata.u_conv4_out,
              appdata.u_conv5_w,
              appdata.u_conv5_b,
              appdata.u_conv5_out,
              /*conv=*/5);
}

// ----------------------------------------------------------------------------
// Stage 8 (v2) - MaxPool3
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_8(AppData& appdata, kiss_vk::Sequence& seq) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 8, &appdata);

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv5_out.d0();  // Expected: 128
  const int channels = appdata.u_conv5_out.d1();    // Expected: 256
  const int in_height = appdata.u_conv5_out.d2();   // Expected: 8
  const int in_width = appdata.u_conv5_out.d3();    // Expected: 8

  const int out_height = (in_height - kPoolSize) / kPoolStride + 1;  // Expected: 4
  const int out_width = (in_width - kPoolSize) / kPoolStride + 1;    // Expected: 4

  const int total_output = batch_size * channels * out_height * out_width;

  const auto algo_pc = MaxpoolPushConstants_v2{
      .N = batch_size,
      .C = channels,
      .H = in_height,
      .W = in_width,
      .pool_h = kPoolSize,
      .pool_w = kPoolSize,
      .stride = kPoolStride,
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_conv5_out.pmr_vec()),
                        engine.get_buffer_info(appdata.u_pool3_out.pmr_vec()),
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(seq, {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 9 (v2) - FC1
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_9(AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 9, &appdata);

  // Calculate flattened input size (total number of features per sample)
  const int batch_size = appdata.u_pool3_out.d0();       // Expected: 16
  const int channels = appdata.u_pool3_out.d1();         // Expected: 256
  const int height = appdata.u_pool3_out.d2();           // Expected: 4
  const int width = appdata.u_pool3_out.d3();            // Expected: 4
  const int input_features = channels * height * width;  // 256 * 4 * 4 = 4096

  // Number of output features from the fc1 layer
  const int out_features = appdata.u_fc1_w.d0();  // Expected: 4096

  // Batch-tiled shader when the batch fits its accumulator file; generic fallback.
  const bool proto = fc_layout != "bt" && batch_size <= 128 && input_features % 4 == 0;
  const bool bt = batch_size <= kiss_vk::kLinearBtBatchTile && input_features % 4 == 0;
  auto algo = cached_algorithms.at(proto ? "linear_bt_" + fc_layout
                                   : bt  ? "linear_bt"
                                         : "linear")
                  .get();

  // bt: a 16-lane row owns two output features, 16 rows (32 features) per workgroup.
  const size_t num_groups =
      bt || proto ? kiss_vk::div_ceil(out_features, kiss_vk::kLinearBtFeaturesPerGroup)
                  : kiss_vk::div_ceil(batch_size * out_features, 256);

  const auto algo_pc = LinearPushConstants_v2{
      .N = batch_size,
      .in_features = input_features,
      .out_features = out_features,
      .apply_relu = kRelu ? 1 : 0,
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_pool3_out.pmr_vec()),
                        engine.get_buffer_info(appdata.u_fc1_w.pmr_vec()),
                        engine.get_buffer_info(appdata.u_fc1_b.pmr_vec()),
                        engine.get_buffer_info(appdata.u_fc1_out.pmr_vec()),
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(seq, {static_cast<uint32_t>(num_groups), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 10 (v2) - FC2
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_10(AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 10, &appdata);

  const int batch_size = appdata.u_fc1_out.d0();      // Expected: 16
  const int input_features = appdata.u_fc1_out.d1();  // Expected: 4096

  // Number of output features from the fc2 layer
  const int out_features = appdata.u_fc2_w.d0();  // Expected: 4096

  // Batch-tiled shader when the batch fits its accumulator file; generic fallback.
  const bool proto = fc_layout != "bt" && batch_size <= 128 && input_features % 4 == 0;
  const bool bt = batch_size <= kiss_vk::kLinearBtBatchTile && input_features % 4 == 0;
  auto algo = cached_algorithms.at(proto ? "linear_bt_" + fc_layout
                                   : bt  ? "linear_bt"
                                         : "linear")
                  .get();

  // bt: a 16-lane row owns two output features, 16 rows (32 features) per workgroup.
  const size_t num_groups =
      bt || proto ? kiss_vk::div_ceil(out_features, kiss_vk::kLinearBtFeaturesPerGroup)
                  : kiss_vk::div_ceil(batch_size * out_features, 256);

  const auto algo_pc = LinearPushConstants_v2{
      .N = batch_size,
      .in_features = input_features,
      .out_features = out_features,
      .apply_relu = kRelu ? 1 : 0,
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_fc1_out.pmr_vec()),
                        engine.get_buffer_info(appdata.u_fc2_w.pmr_vec()),
                        engine.get_buffer_info(appdata.u_fc2_b.pmr_vec()),
                        engine.get_buffer_info(appdata.u_fc2_out.pmr_vec()),
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(seq, {static_cast<uint32_t>(num_groups), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 11 (v2) - FC3
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_11(AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 11, &appdata);

  const int batch_size = appdata.u_fc2_out.d0();      // Expected: 16
  const int input_features = appdata.u_fc2_out.d1();  // Expected: 4096

  // Number of output features from the fc3 layer
  const int out_features = appdata.u_fc3_w.d0();  // Expected: 10

  // Batch-tiled shader when the batch fits its accumulator file; generic fallback.
  const bool proto = fc_layout != "bt" && batch_size <= 128 && input_features % 4 == 0;
  const bool bt = batch_size <= kiss_vk::kLinearBtBatchTile && input_features % 4 == 0;
  auto algo = cached_algorithms.at(proto ? "linear_bt_" + fc_layout
                                   : bt  ? "linear_bt"
                                         : "linear")
                  .get();

  // bt: a 16-lane row owns two output features, 16 rows (32 features) per workgroup.
  const size_t num_groups =
      bt || proto ? kiss_vk::div_ceil(out_features, kiss_vk::kLinearBtFeaturesPerGroup)
                  : kiss_vk::div_ceil(batch_size * out_features, 256);

  const auto algo_pc = LinearPushConstants_v2{
      .N = batch_size,
      .in_features = input_features,
      .out_features = out_features,
      .apply_relu = 0,  // FC3 emits raw logits: no ReLU
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_fc2_out.pmr_vec()),
                        engine.get_buffer_info(appdata.u_fc3_w.pmr_vec()),
                        engine.get_buffer_info(appdata.u_fc3_b.pmr_vec()),
                        engine.get_buffer_info(appdata.u_fc3_out.pmr_vec()),
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(seq, {static_cast<uint32_t>(num_groups), 1, 1});
}

}  // namespace cifar_dense::vulkan
