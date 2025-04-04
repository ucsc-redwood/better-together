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
//   int apply_relu;  // 1 to apply ReLU, 0 otherwise
// }
// params;
// ----------------------------------------------------------------------------

struct Conv2dPushConstants {
  uint32_t N;
  uint32_t C;
  uint32_t H;
  uint32_t W;
  uint32_t K;
  uint32_t R;
  uint32_t S;
  uint32_t stride;
  uint32_t padding;
  uint32_t apply_relu;  // should be an bool, but use int for now
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
  uint32_t N;
  uint32_t C;
  uint32_t H;
  uint32_t W;
  uint32_t pool_h;
  uint32_t pool_w;
  uint32_t stride;
  uint32_t padding;
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
  uint32_t N;
  uint32_t in_features;
  uint32_t out_features;
};

// ----------------------------------------------------------------------------
// Dispatcher
// ----------------------------------------------------------------------------

class VulkanDispatcher final {
 public:
  explicit VulkanDispatcher() : engine(), seq(engine.make_seq()) {
    spdlog::debug("VulkanDispatcher::VulkanDispatcher(), Initializing VulkanDispatcher");

    // conv2d

    auto conv2d_algo = engine.make_algo("cifar_conv2d")
                           ->work_group_size(256, 1, 1)
                           ->num_sets(1)
                           ->num_buffers(4)
                           ->push_constant<Conv2dPushConstants>()
                           ->build();

    cached_algorithms.try_emplace("conv2d", std::move(conv2d_algo));

    // maxpool2d

    auto maxpool2d_algo = engine.make_algo("cifar_maxpool2d")
                              ->work_group_size(256, 1, 1)
                              ->num_sets(1)
                              ->num_buffers(2)
                              ->push_constant<MaxpoolPushConstants>()
                              ->build();

    cached_algorithms.try_emplace("maxpool2d", std::move(maxpool2d_algo));

    // linear

    auto linear_algo = engine.make_algo("cifar_linear")
                           ->work_group_size(256, 1, 1)
                           ->num_sets(1)
                           ->num_buffers(4)
                           ->push_constant<LinearPushConstants>()
                           ->build();

    cached_algorithms.try_emplace("linear", std::move(linear_algo));
  }

  kiss_vk::VulkanMemoryResource::memory_resource* get_mr() { return engine.get_mr(); }

  void run_stage_1(cifar_dense::AppDataBatch& app_data) {
    auto algo = cached_algorithms.at("conv2d").get();

    const auto& model = cifar_dense::AppDataBatch::get_model();

    algo->update_descriptor_set(0,
                                {
                                    engine.get_buffer_info(app_data.input),
                                    engine.get_buffer_info(model.h_conv1  _w),
                                    engine.get_buffer_info(model.h_conv1_b),
                                    engine.get_buffer_info(app_data.conv1_out),
                                });
  }

 private:
  kiss_vk::Engine engine;
  std::shared_ptr<kiss_vk::Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<kiss_vk::Algorithm>> cached_algorithms;
};

}  // namespace cifar_dense::vulkan