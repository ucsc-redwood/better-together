#pragma once

#include "../appdata.hpp"
#include "builtin-apps/common/kiss-vk/engine.hpp"
#include "builtin-apps/debug_logger.hpp"

namespace cifar_dense::vulkan {

// ----------------------------------------------------------------------------
// layout(push_constant) uniform Params {
//   int N;           // Batch size
//   int C;           // Number of input channels
//   int H;           // Input height
//   int W;           // Input width
//   int K;           // Number of output channels
//   int R;           // Kernel height
//   int S;           // Kernel width
//   int stride;      // Convolution stride
//   int padding;     // Convolution padding
//   int apply_relu;    // 1 to apply ReLU, 0 otherwise
// }
// params;
// ----------------------------------------------------------------------------

struct Conv2dPushConstants {
  int32_t N;
  int32_t C;
  int32_t H;
  int32_t W;
  int32_t K;
  int32_t R;
  int32_t S;
  int32_t stride;
  int32_t padding;
  int32_t apply_relu;  // should be an bool, but use int for now
};

// ----------------------------------------------------------------------------
// layout(push_constant) uniform Params {
//   int N;        // Batch size
//   int C;        // Number of channels
//   int H;        // Input height
//   int W;        // Input width
//   int pool_h;   // Pooling kernel height
//   int pool_w;   // Pooling kernel width
//   int stride;   // Pooling stride
//   int padding;  // Pooling padding
// }
// params;
// ----------------------------------------------------------------------------

struct MaxpoolPushConstants {
  int32_t N;
  int32_t C;
  int32_t H;
  int32_t W;
  int32_t pool_h;
  int32_t pool_w;
  int32_t stride;
  int32_t padding;
};

// ----------------------------------------------------------------------------
// layout(push_constant) uniform Params {
//   int N;             // Batch size
//   int in_features;   // Number of input features
//   int out_features;  // Number of output features
// }
// params;
// ----------------------------------------------------------------------------

struct LinearPushConstants {
  int32_t N;
  int32_t in_features;
  int32_t out_features;
};

// ----------------------------------------------------------------------------
// Dispatcher
// ----------------------------------------------------------------------------

class VulkanDispatcher final {
 public:
  explicit VulkanDispatcher() : engine(), seq(engine.make_seq()) {
    spdlog::debug("VulkanDispatcher::VulkanDispatcher(), Initializing VulkanDispatcher");

    // conv2d

    auto conv2d_algo = engine.make_algo("new_cifar_dense_conv2d")
                           ->work_group_size(256, 1, 1)
                           ->num_sets(1)
                           ->num_buffers(4)
                           ->push_constant<Conv2dPushConstants>()
                           ->build();

    cached_algorithms.try_emplace("conv2d", std::move(conv2d_algo));

    // maxpool2d

    auto maxpool2d_algo = engine.make_algo("new_cifar_dense_maxpool")
                              ->work_group_size(256, 1, 1)
                              ->num_sets(1)
                              ->num_buffers(2)
                              ->push_constant<MaxpoolPushConstants>()
                              ->build();

    cached_algorithms.try_emplace("maxpool2d", std::move(maxpool2d_algo));

    // linear

    auto linear_algo = engine.make_algo("new_cifar_dense_linear")
                           ->work_group_size(256, 1, 1)
                           ->num_sets(1)
                           ->num_buffers(4)
                           ->push_constant<LinearPushConstants>()
                           ->build();

    cached_algorithms.try_emplace("linear", std::move(linear_algo));
  }

  kiss_vk::VulkanMemoryResource::memory_resource* get_mr() { return engine.get_mr(); }

  // ----------------------------------------------------------------------------
  // Stage 1:
  // ----------------------------------------------------------------------------

  void run_stage_1(cifar_dense::AppDataBatch& appdata) {
    auto algo = cached_algorithms.at("conv2d").get();

    algo->update_descriptor_set(0,
                                {
                                    engine.get_buffer_info(appdata.u_input.vec()),
                                    engine.get_buffer_info(appdata.u_conv1_w.vec()),
                                    engine.get_buffer_info(appdata.u_conv1_b.vec()),
                                    engine.get_buffer_info(appdata.u_conv1_out.vec()),
                                });

    const auto& in_shape = appdata.u_input.shape();       // [128, 3, 32, 32]
    const auto& w_shape = appdata.u_conv1_w.shape();      // [16, 3, 3, 3]
    const auto& out_shape = appdata.u_conv1_out.shape();  // [128, 16, 30, 30]

    const int32_t N = in_shape[0];   // batch, 128
    const int32_t C = in_shape[1];   // in channels, 3
    const int32_t H = in_shape[2];   // in height, 32
    const int32_t W = in_shape[3];   // in width, 32
    const int32_t R = w_shape[2];    // kernel height, 3
    const int32_t S = w_shape[3];    // kernel width, 3
    const int32_t K = out_shape[1];  // out channels, 16

    constexpr int32_t padding = 0;
    constexpr int32_t stride = 1;
    constexpr int32_t apply_relu = 1;
    const int32_t P = (H + 2 * padding - R) / stride + 1;
    const int32_t Q = (W + 2 * padding - S) / stride + 1;
    const int32_t PQ = P * Q;

    algo->update_push_constant(Conv2dPushConstants{
        .N = N,
        .C = C,
        .H = H,
        .W = W,
        .K = K,
        .R = R,
        .S = S,
        .stride = stride,
        .padding = padding,
        .apply_relu = apply_relu,
    });

    LOG_KERNEL(LogKernelType::kVK, 1, &appdata);

    // const dim3 blockDim(256);
    // const dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, K, N);
    seq->cmd_begin();
    algo->record_bind_core(seq->get_handle(), 0);
    algo->record_bind_push(seq->get_handle());
    algo->record_dispatch(seq->get_handle(),
                          {static_cast<uint32_t>(kiss_vk::div_ceil(PQ, 256)),
                           static_cast<uint32_t>(K),
                           static_cast<uint32_t>(N)});
    seq->cmd_end();

    seq->reset_fence();
    seq->submit();
    seq->wait_for_fence();
  }

 private:
  kiss_vk::Engine engine;
  std::shared_ptr<kiss_vk::Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<kiss_vk::Algorithm>> cached_algorithms;
};

}  // namespace cifar_dense::vulkan