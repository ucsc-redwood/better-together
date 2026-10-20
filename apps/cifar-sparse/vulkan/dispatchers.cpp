#include "dispatchers.hpp"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

#include "platform/engine/vulkan/shaders/conv_k3s1p1_contract.hpp"
#include "platform/engine/vulkan/shaders/linear_bt_contract.hpp"
#include "platform/util/debug_logger.hpp"

namespace cifar_sparse::vulkan {

// // Convolution parameters
// constexpr int kKernelSize = 3;
// constexpr int kStride = 1;
// constexpr int kPadding = 1;

// // Pooling parameters
// constexpr int kPoolSize = 2;
// constexpr int kPoolStride = 2;

// constexpr bool kRelu = true;

// // Push constants holding all kernel parameters.
// layout(push_constant) uniform Params {
//   int batch_size;    // Number of batches.
//   int in_channels;   // Input channels.
//   int in_height;     // Input height.
//   int in_width;      // Input width.
//   int out_channels;  // Output channels.
//   int out_height;    // Output height.
//   int out_width;     // Output width.
//   int kernel_size;   // Kernel size (assumed square: kernel_size x kernel_size).
//   int stride;        // Convolution stride.
//   int padding;       // Padding.
//   int relu;          // Flag for ReLU activation (nonzero true).
//   int bias_size;     // Bias vector size (0 if bias is not used).
// }
// params;

struct Conv2dPushConstants {
  int32_t batch_size;
  int32_t in_channels;
  int32_t in_height;
  int32_t in_width;
  int32_t out_channels;
  int32_t out_height;
  int32_t out_width;
  int32_t kernel_size;
  int32_t stride;
  int32_t padding;
  int32_t relu;
  int32_t bias_size;
};

// // Push constant block to pass convolution parameters.
// layout(push_constant) uniform Params {
//   int batch_size;  // Number of batches.
//   int channels;    // Number of channels.
//   int in_height;   // Input height.
//   int in_width;    // Input width.
//   int out_height;  // Output height, computed as (in_height - pool_size) / stride + 1.
//   int out_width;   // Output width, computed as (in_width - pool_size) / stride + 1.
//   int pool_size;   // Size of the pooling window (assumed square).
//   int stride;      // Stride of the pooling operation.
// }
// params;

struct MaxpoolPushConstants {
  int32_t batch_size;
  int32_t channels;
  int32_t in_height;
  int32_t in_width;
  int32_t out_height;
  int32_t out_width;
  int32_t pool_size;
  int32_t stride;
};

// // Push constants for kernel parameters (dense FC head; only the convs are
// // pruned -- same contract as new_cifar_dense_linear).
// layout(push_constant) uniform Params {
//   int N;             // Batch size
//   int in_features;   // Number of input features
//   int out_features;  // Number of output features
//   int apply_relu;    // 1 to apply ReLU, 0 otherwise
// }
// params;

struct LinearPushConstants {
  int32_t N;             // Batch size
  int32_t in_features;   // Number of input features
  int32_t out_features;  // Number of output features
  int32_t apply_relu;    // 1 to apply ReLU, 0 otherwise
};

// The linear_bt pipeline for a weight precision.
static std::string linear_bt_name(const bt::FcWeights w) {
  return w == bt::FcWeights::kFp32 ? "linear_bt" : std::string("linear_bt/") + bt::to_string(w);
}

// ----------------------------------------------------------------------------
// Constructor (v2)
// ----------------------------------------------------------------------------

VulkanDispatcher::VulkanDispatcher() : engine(), default_seq(engine.make_seq()) {
  spdlog::debug("VulkanDispatcher::VulkanDispatcher(), Initializing VulkanDispatcher");

  // conv2d is reused by stages 1,3,5,6,7 and maxpool by stages 2,4,8; record_bind
  // captures each stage's buffers per dispatch, so they can share one command buffer.
  auto conv2d_algo = engine.make_algo("new_cifar_sparse_conv2d")
                         ->work_group_size(256, 1, 1)
                         ->num_buffers(6)
                         ->push_constant<Conv2dPushConstants>()
                         ->build();

  cached_algorithms.try_emplace("conv2d", std::move(conv2d_algo));

  // 3x3/s1/p1 CSR specializations (every pruned conv in the model), selected per stage when
  // kernel_size==3, stride==1, padding==1, by the device's subgroup size. Below 32, the Mali path
  // (mali-conv-kernels): PX adjacent pixels per invocation, one decode and weight load per nonzero.
  auto conv2d_k3s1p1_algo = engine.make_algo("new_cifar_sparse_conv2d_k3s1p1")
                                ->work_group_size(256, 1, 1)
                                ->num_buffers(6)
                                ->push_constant<Conv2dPushConstants>()
                                ->build();

  cached_algorithms.try_emplace("conv2d_k3s1p1", std::move(conv2d_k3s1p1_algo));

  // 32 and above: one output channel per workgroup row on the packed taps (cifar-gpu-kernels S1,
  // S2). BT_VK_SPARSE_STAGE_ROW=1 stages each CSR row in shared memory (S3; off by default, it was
  // not faster on the Samsung).
  const char* stage_row = std::getenv("BT_VK_SPARSE_STAGE_ROW");
  auto conv2d_k3s1p1_oc_algo =
      engine.make_algo("new_cifar_sparse_conv2d_k3s1p1_oc")
          ->work_group_size(256, 1, 1)
          ->spec_constant(1, stage_row && std::string_view(stage_row) == "1")
          ->num_buffers(6)
          ->push_constant<Conv2dPushConstants>()
          ->build();

  cached_algorithms.try_emplace("conv2d_k3s1p1_oc", std::move(conv2d_k3s1p1_oc_algo));

  auto maxpool_algo = engine.make_algo("new_cifar_sparse_maxpool")
                          ->work_group_size(256, 1, 1)
                          ->num_buffers(2)
                          ->push_constant<MaxpoolPushConstants>()
                          ->build();

  cached_algorithms.try_emplace("maxpool", std::move(maxpool_algo));

  auto linear_algo = engine.make_algo("new_cifar_sparse_linear")
                         ->work_group_size(256, 1, 1)
                         ->num_buffers(4)  // Input, Weight, Bias, Output
                         ->push_constant<LinearPushConstants>()
                         ->build();

  cached_algorithms.try_emplace("linear", std::move(linear_algo));

  // Batch-tiled linear, shared with cifar-dense (the FC head is dense; identical
  // descriptor layout): a 16-lane workgroup row owns TWO output features, strides their
  // weight row coalesced, and accumulates a 16-image batch tile while the row is
  // streamed ONCE -- the generic thread-per-(n,of) shader re-streams the whole weight
  // matrix once per image and each lane scans a different 16 KB row (uncoalesced).
  // grid.y tiles sparse's N=128 batch in groups of 16 (weights read 8x, not 128x).
  // Selected per stage when in_features % 4 == 0.
  // Rows reduce with subgroupClusteredAdd where the device has it (cifar-gpu-kernels F3).
  // One pipeline per weight precision (cifar-gpu-kernels F2): an AppData built with
  // BT_VK_FC_WEIGHTS=int8 carries the fc1/fc2 copies its stages bind instead. INT8 rows own 4
  // features on subgroups of 32 and more (2 otherwise, as FP32): faster on the Samsung, where 2
  // left INT8 bound by the shared-memory reads; Mali spills with 4 (task 3.8).
  fc_int8_fpr = engine.subgroup_size() >= 32 ? 4 : 2;
  for (const auto w : {bt::FcWeights::kFp32, bt::FcWeights::kInt8}) {
    cached_algorithms.try_emplace(
        linear_bt_name(w),
        engine.make_algo("new_cifar_dense_linear_bt")
            ->work_group_size(256, 1, 1)
            ->subgroup_size_dependent()
            ->spec_constant(1, engine.compute_limits().subgroup_clustered ? 1 : 0)
            ->spec_constant(2, static_cast<uint32_t>(w))
            ->spec_constant(3, static_cast<uint32_t>(fc_features_per_row(w)))
            ->num_buffers(4)  // Input, Weight, Bias, Output
            ->push_constant<LinearPushConstants>()
            ->build());
  }
}

// ----------------------------------------------------------------------------
// run_stage_k: per-stage single-submit wrappers (bm_main per-stage device timing).
// ----------------------------------------------------------------------------

void VulkanDispatcher::run_stage_1(cifar_sparse::AppData& appdata) {
  dispatch_multi_stage(appdata, 1, 1);
}
void VulkanDispatcher::run_stage_2(cifar_sparse::AppData& appdata) {
  dispatch_multi_stage(appdata, 2, 2);
}
void VulkanDispatcher::run_stage_3(cifar_sparse::AppData& appdata) {
  dispatch_multi_stage(appdata, 3, 3);
}
void VulkanDispatcher::run_stage_4(cifar_sparse::AppData& appdata) {
  dispatch_multi_stage(appdata, 4, 4);
}
void VulkanDispatcher::run_stage_5(cifar_sparse::AppData& appdata) {
  dispatch_multi_stage(appdata, 5, 5);
}
void VulkanDispatcher::run_stage_6(cifar_sparse::AppData& appdata) {
  dispatch_multi_stage(appdata, 6, 6);
}
void VulkanDispatcher::run_stage_7(cifar_sparse::AppData& appdata) {
  dispatch_multi_stage(appdata, 7, 7);
}
void VulkanDispatcher::run_stage_8(cifar_sparse::AppData& appdata) {
  dispatch_multi_stage(appdata, 8, 8);
}
void VulkanDispatcher::run_stage_9(cifar_sparse::AppData& appdata) {
  dispatch_multi_stage(appdata, 9, 9);
}
void VulkanDispatcher::run_stage_10(cifar_sparse::AppData& appdata) {
  dispatch_multi_stage(appdata, 10, 10);
}
void VulkanDispatcher::run_stage_11(cifar_sparse::AppData& appdata) {
  dispatch_multi_stage(appdata, 11, 11);
}

// ----------------------------------------------------------------------------
// The convs (stages 1, 3, 5, 6, 7)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_conv(kiss_vk::Sequence& seq,
                                   Ndarray4D& in,
                                   Ndarray4D& out,
                                   CSRMatrix& csr,
                                   std::pmr::vector<int>& taps,
                                   Ndarray1D& bias) {
  // 3x3/s1/p1 CSR specialization when shapes allow; generic fallback. With a subgroup of 32 or
  // more the specialization runs one output channel per workgroup row on the packed taps
  // (cifar-gpu-kernels S1, S2); below 32, the Mali path.
  const bool spec = kKernelSize == 3 && kStride == 1 && kPadding == 1;
  const bool per_channel = spec && engine.subgroup_size() >= 32;
  auto algo =
      cached_algorithms.at(per_channel ? "conv2d_k3s1p1_oc"
                           : spec      ? "conv2d_k3s1p1"
                                       : "conv2d")
          .get();

  const int batch_size = in.d0();
  const int in_channels = in.d1();
  const int in_height = in.d2();
  const int in_width = in.d3();

  const int out_channels = csr.rows;

  const int out_height = (in_height + 2 * kPadding - kKernelSize) / kStride + 1;
  const int out_width = (in_width + 2 * kPadding - kKernelSize) / kStride + 1;

  const auto algo_pc = Conv2dPushConstants{
      .batch_size = batch_size,
      .in_channels = in_channels,
      .in_height = in_height,
      .in_width = in_width,
      .out_channels = out_channels,
      .out_height = out_height,
      .out_width = out_width,
      .kernel_size = kKernelSize,
      .stride = kStride,
      .padding = kPadding,
      .relu = kRelu,
      .bias_size = bias.d0(),
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(in.pmr_vec()),
                        engine.get_buffer_info(out.pmr_vec()),
                        engine.get_buffer_info(csr.values_pmr_vec()),
                        engine.get_buffer_info(csr.row_ptr_pmr_vec()),
                        engine.get_buffer_info(per_channel ? taps : csr.col_idx_pmr_vec()),
                        engine.get_buffer_info(bias.pmr_vec()),
                    });
  algo->record_push_constants(seq, algo_pc);

  const size_t pixels = static_cast<size_t>(batch_size) * out_height * out_width;
  if (per_channel) {
    algo->record_dispatch(seq,
                          {static_cast<uint32_t>(kiss_vk::div_ceil(pixels, 256)),
                           static_cast<uint32_t>(out_channels),
                           1});
    return;
  }
  // spec: the Mali path's invocations, PX pixels each, from the conv contract.
  const size_t total_output = spec ? kiss_vk::sparse_conv_k3s1p1_mali_invocations(
                                         batch_size, out_channels, out_height, out_width)
                                   : pixels * out_channels;
  algo->record_dispatch(seq, {static_cast<uint32_t>(kiss_vk::div_ceil(total_output, 256)), 1, 1});
}

// ----------------------------------------------------------------------------
// Stage 1 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_1(cifar_sparse::AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 1, &appdata);
  record_conv(seq,
              appdata.u_input,
              appdata.u_conv1_out,
              appdata.conv1_sparse,
              appdata.conv1_taps,
              appdata.u_conv1_b);
}

// ----------------------------------------------------------------------------
// Stage 2 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_2(cifar_sparse::AppData& appdata, kiss_vk::Sequence& seq) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 2, &appdata);

  // Extract dimensions from the convolution output NDArray4D.
  const int batch_size = appdata.u_conv1_out.d0();  // Expected: 128
  const int channels = appdata.u_conv1_out.d1();    // Expected: 64
  const int in_height = appdata.u_conv1_out.d2();   // Expected: 32
  const int in_width = appdata.u_conv1_out.d3();    // Expected: 32

  const int out_height = (in_height - kPoolSize) / kPoolStride + 1;
  const int out_width = (in_width - kPoolSize) / kPoolStride + 1;

  const int total_output = batch_size * channels * out_height * out_width;

  const auto algo_pc = MaxpoolPushConstants{
      .batch_size = batch_size,
      .channels = channels,
      .in_height = in_height,
      .in_width = in_width,
      .out_height = out_height,
      .out_width = out_width,
      .pool_size = kPoolSize,
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
// Stage 3 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_3(cifar_sparse::AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 3, &appdata);
  record_conv(seq,
              appdata.u_pool1_out,
              appdata.u_conv2_out,
              appdata.conv2_sparse,
              appdata.conv2_taps,
              appdata.u_conv2_b);
}

// ----------------------------------------------------------------------------
// Stage 4 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_4(cifar_sparse::AppData& appdata, kiss_vk::Sequence& seq) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 4, &appdata);

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv2_out.d0();
  const int channels = appdata.u_conv2_out.d1();
  const int in_height = appdata.u_conv2_out.d2();
  const int in_width = appdata.u_conv2_out.d3();

  const int out_height = (in_height - kPoolSize) / kPoolStride + 1;
  const int out_width = (in_width - kPoolSize) / kPoolStride + 1;

  const int total_output = batch_size * channels * out_height * out_width;

  const auto algo_pc = MaxpoolPushConstants{
      .batch_size = batch_size,
      .channels = channels,
      .in_height = in_height,
      .in_width = in_width,
      .out_height = out_height,
      .out_width = out_width,
      .pool_size = kPoolSize,
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
// Stage 5 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_5(cifar_sparse::AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 5, &appdata);
  record_conv(seq,
              appdata.u_pool2_out,
              appdata.u_conv3_out,
              appdata.conv3_sparse,
              appdata.conv3_taps,
              appdata.u_conv3_b);
}

// ----------------------------------------------------------------------------
// Stage 6 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_6(cifar_sparse::AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 6, &appdata);
  record_conv(seq,
              appdata.u_conv3_out,
              appdata.u_conv4_out,
              appdata.conv4_sparse,
              appdata.conv4_taps,
              appdata.u_conv4_b);
}

// ----------------------------------------------------------------------------
// Stage 7 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_7(cifar_sparse::AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 7, &appdata);
  record_conv(seq,
              appdata.u_conv4_out,
              appdata.u_conv5_out,
              appdata.conv5_sparse,
              appdata.conv5_taps,
              appdata.u_conv5_b);
}

// ----------------------------------------------------------------------------
// Stage 8 (v2)
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_8(cifar_sparse::AppData& appdata, kiss_vk::Sequence& seq) {
  auto algo = cached_algorithms.at("maxpool").get();

  LOG_KERNEL(LogKernelType::kVK, 8, &appdata);

  // Extract dimensions from the convolution output NDArray4D
  const int batch_size = appdata.u_conv5_out.d0();
  const int channels = appdata.u_conv5_out.d1();
  const int in_height = appdata.u_conv5_out.d2();
  const int in_width = appdata.u_conv5_out.d3();

  const int out_height = (in_height - kPoolSize) / kPoolStride + 1;
  const int out_width = (in_width - kPoolSize) / kPoolStride + 1;

  const int total_output = batch_size * channels * out_height * out_width;

  const auto algo_pc = MaxpoolPushConstants{
      .batch_size = batch_size,
      .channels = channels,
      .in_height = in_height,
      .in_width = in_width,
      .out_height = out_height,
      .out_width = out_width,
      .pool_size = kPoolSize,
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

void VulkanDispatcher::record_stage_9(cifar_sparse::AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 9, &appdata);

  // Calculate flattened input size (total number of features per sample)
  const int batch_size = appdata.u_pool3_out.d0();       // Expected: 128
  const int channels = appdata.u_pool3_out.d1();         // Expected: 256
  const int height = appdata.u_pool3_out.d2();           // Expected: 4
  const int width = appdata.u_pool3_out.d3();            // Expected: 4
  const int input_features = channels * height * width;  // 256 * 4 * 4 = 4096

  // Number of output features from the fc1 layer
  const int out_features = appdata.u_fc1_w.d0();  // Expected: 4096

  // Batch-tiled shader (vec4 weight loads); generic fallback.
  const bool bt = input_features % 4 == 0;
  // fc1/fc2 read the AppData's reduced-precision copy in its BT_VK_FC_WEIGHTS mode.
  const bool reduced = bt && appdata.fc_weights != bt::FcWeights::kFp32;
  const bt::FcWeights weights = reduced ? appdata.fc_weights : bt::FcWeights::kFp32;
  auto algo = cached_algorithms.at(bt ? linear_bt_name(weights) : "linear").get();

  // bt: a 16-lane row owns FPR output features, 16 rows per workgroup; grid.y walks
  // the batch in 16-image tiles.
  const uint32_t gx = static_cast<uint32_t>(
      bt ? kiss_vk::div_ceil(out_features, kiss_vk::kLinearBtRows * fc_features_per_row(weights))
         : kiss_vk::div_ceil(batch_size * out_features, 256));
  const uint32_t gy =
      bt ? static_cast<uint32_t>(kiss_vk::div_ceil(batch_size, kiss_vk::kLinearBtBatchTile)) : 1u;

  const auto algo_pc = LinearPushConstants{
      .N = batch_size,
      .in_features = input_features,
      .out_features = out_features,
      .apply_relu = kRelu ? 1 : 0,
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_pool3_out.pmr_vec()),
                        reduced ? engine.get_buffer_info(appdata.u_fc1_wq.words)
                                : engine.get_buffer_info(appdata.u_fc1_w.pmr_vec()),
                        engine.get_buffer_info(appdata.u_fc1_b.pmr_vec()),
                        engine.get_buffer_info(appdata.u_fc1_out.pmr_vec()),
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(seq, {gx, gy, 1});
}

// ----------------------------------------------------------------------------
// Stage 10 (v2) - FC2
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_10(cifar_sparse::AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 10, &appdata);

  const int batch_size = appdata.u_fc1_out.d0();      // Expected: 128
  const int input_features = appdata.u_fc1_out.d1();  // Expected: 4096

  // Number of output features from the fc2 layer
  const int out_features = appdata.u_fc2_w.d0();  // Expected: 4096

  // Batch-tiled shader (vec4 weight loads); generic fallback.
  const bool bt = input_features % 4 == 0;
  // fc1/fc2 read the AppData's reduced-precision copy in its BT_VK_FC_WEIGHTS mode.
  const bool reduced = bt && appdata.fc_weights != bt::FcWeights::kFp32;
  const bt::FcWeights weights = reduced ? appdata.fc_weights : bt::FcWeights::kFp32;
  auto algo = cached_algorithms.at(bt ? linear_bt_name(weights) : "linear").get();

  // bt: a 16-lane row owns FPR output features, 16 rows per workgroup; grid.y walks
  // the batch in 16-image tiles.
  const uint32_t gx = static_cast<uint32_t>(
      bt ? kiss_vk::div_ceil(out_features, kiss_vk::kLinearBtRows * fc_features_per_row(weights))
         : kiss_vk::div_ceil(batch_size * out_features, 256));
  const uint32_t gy =
      bt ? static_cast<uint32_t>(kiss_vk::div_ceil(batch_size, kiss_vk::kLinearBtBatchTile)) : 1u;

  const auto algo_pc = LinearPushConstants{
      .N = batch_size,
      .in_features = input_features,
      .out_features = out_features,
      .apply_relu = kRelu ? 1 : 0,
  };

  algo->record_bind(seq,
                    {
                        engine.get_buffer_info(appdata.u_fc1_out.pmr_vec()),
                        reduced ? engine.get_buffer_info(appdata.u_fc2_wq.words)
                                : engine.get_buffer_info(appdata.u_fc2_w.pmr_vec()),
                        engine.get_buffer_info(appdata.u_fc2_b.pmr_vec()),
                        engine.get_buffer_info(appdata.u_fc2_out.pmr_vec()),
                    });
  algo->record_push_constants(seq, algo_pc);
  algo->record_dispatch(seq, {gx, gy, 1});
}

// ----------------------------------------------------------------------------
// Stage 11 (v2) - FC3
// ----------------------------------------------------------------------------

void VulkanDispatcher::record_stage_11(cifar_sparse::AppData& appdata, kiss_vk::Sequence& seq) {
  LOG_KERNEL(LogKernelType::kVK, 11, &appdata);

  const int batch_size = appdata.u_fc2_out.d0();      // Expected: 128
  const int input_features = appdata.u_fc2_out.d1();  // Expected: 4096

  // Number of output features from the fc3 layer
  const int out_features = appdata.u_fc3_w.d0();  // Expected: 10

  // Batch-tiled shader (vec4 weight loads); generic fallback.
  const bool bt = input_features % 4 == 0;
  auto algo = cached_algorithms.at(bt ? "linear_bt" : "linear").get();

  // bt: a 16-lane row owns FPR output features, 16 rows per workgroup; grid.y walks
  // the batch in 16-image tiles.
  const uint32_t gx = static_cast<uint32_t>(
      bt ? kiss_vk::div_ceil(out_features,
                             kiss_vk::kLinearBtRows * fc_features_per_row(bt::FcWeights::kFp32))
         : kiss_vk::div_ceil(batch_size * out_features, 256));
  const uint32_t gy =
      bt ? static_cast<uint32_t>(kiss_vk::div_ceil(batch_size, kiss_vk::kLinearBtBatchTile)) : 1u;

  const auto algo_pc = LinearPushConstants{
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
  algo->record_dispatch(seq, {gx, gy, 1});
}

}  // namespace cifar_sparse::vulkan
