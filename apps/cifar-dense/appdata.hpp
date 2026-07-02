#pragma once

#include <algorithm>
#include <memory_resource>
#include <random>

#include "platform/util/base_appdata.hpp"
// #include "../load_npy.hpp"
#include "platform/util/ndarray.hpp"

namespace cifar_dense {

// Convolution parameters
constexpr int kKernelSize = 3;
constexpr int kStride = 1;
constexpr int kPadding = 1;

// Pooling parameters
constexpr int kPoolSize = 2;
constexpr int kPoolStride = 2;

constexpr bool kRelu = true;

struct AppData final : public BaseAppData {
  // static constexpr size_t BATCH_SIZE = 1;
  // static constexpr size_t BATCH_SIZE = 32;
  static constexpr size_t BATCH_SIZE = 128;

  // conv1: 64 output channels, 3×3×3 kernel = 27 inputs
  // conv2: 192 output channels, 64×3×3 kernel = 576 inputs
  // conv3: 384 output channels, 192×3×3 kernel = 1728 inputs
  // conv4: 256 output channels, 384×3×3 kernel = 3456 inputs
  // conv5: 256 output channels, 256×3×3 kernel = 2304 inputs
  // fc1: 4096 output channels, 4096 inputs
  // fc2: 4096 output channels, 4096 inputs
  // fc3: 10 output channels, 4096 inputs

  explicit AppData(std::pmr::memory_resource* mr)
      : BaseAppData(),
        u_input(BATCH_SIZE, 3, 32, 32, mr),
        u_conv1_out(BATCH_SIZE, 64, 32, 32, mr),
        u_pool1_out(BATCH_SIZE, 64, 16, 16, mr),
        u_conv2_out(BATCH_SIZE, 192, 16, 16, mr),
        u_pool2_out(BATCH_SIZE, 192, 8, 8, mr),
        u_conv3_out(BATCH_SIZE, 384, 8, 8, mr),
        u_conv4_out(BATCH_SIZE, 256, 8, 8, mr),
        u_conv5_out(BATCH_SIZE, 256, 8, 8, mr),
        u_pool3_out(BATCH_SIZE, 256, 4, 4, mr),
        u_fc1_out(BATCH_SIZE, 4096, mr),
        u_fc2_out(BATCH_SIZE, 4096, mr),
        u_fc3_out(BATCH_SIZE, 10, mr),
        u_conv1_w(64, 3, 3, 3, mr),
        u_conv1_b(64, mr),
        u_conv2_w(192, 64, 3, 3, mr),
        u_conv2_b(192, mr),
        u_conv3_w(384, 192, 3, 3, mr),
        u_conv3_b(384, mr),
        u_conv4_w(256, 384, 3, 3, mr),
        u_conv4_b(256, mr),
        u_conv5_w(256, 256, 3, 3, mr),
        u_conv5_b(256, mr),
        u_fc1_w(4096, 4096, mr),
        u_fc1_b(4096, mr),
        u_fc2_w(4096, 4096, mr),
        u_fc2_b(4096, mr),
        u_fc3_w(10, 4096, mr),
        u_fc3_b(10, mr) {
    std::mt19937 gen(114514);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::ranges::generate(u_input.pmr_vec(), [&]() { return dis(gen); });

    // Keep the synthetic weights small so 11 chained stages stay well inside float
    // range: convs ±0.05, the 4096-wide FCs ±0.02. Biases are 0.0f (folded-BN
    // convention: the real bias lands in the exported b' = γ·(b−mean)/√(var+ε) + β).
    std::uniform_real_distribution<float> conv_weight_dis(-0.05f, 0.05f);
    std::ranges::generate(u_conv1_w.pmr_vec(), [&]() { return conv_weight_dis(gen); });
    std::ranges::generate(u_conv2_w.pmr_vec(), [&]() { return conv_weight_dis(gen); });
    std::ranges::generate(u_conv3_w.pmr_vec(), [&]() { return conv_weight_dis(gen); });
    std::ranges::generate(u_conv4_w.pmr_vec(), [&]() { return conv_weight_dis(gen); });
    std::ranges::generate(u_conv5_w.pmr_vec(), [&]() { return conv_weight_dis(gen); });

    std::uniform_real_distribution<float> fc_weight_dis(-0.02f, 0.02f);
    std::ranges::generate(u_fc1_w.pmr_vec(), [&]() { return fc_weight_dis(gen); });
    std::ranges::generate(u_fc2_w.pmr_vec(), [&]() { return fc_weight_dis(gen); });
    std::ranges::generate(u_fc3_w.pmr_vec(), [&]() { return fc_weight_dis(gen); });

    std::ranges::fill(u_conv1_b.pmr_vec(), 0.0f);
    std::ranges::fill(u_conv2_b.pmr_vec(), 0.0f);
    std::ranges::fill(u_conv3_b.pmr_vec(), 0.0f);
    std::ranges::fill(u_conv4_b.pmr_vec(), 0.0f);
    std::ranges::fill(u_conv5_b.pmr_vec(), 0.0f);
    std::ranges::fill(u_fc1_b.pmr_vec(), 0.0f);
    std::ranges::fill(u_fc2_b.pmr_vec(), 0.0f);
    std::ranges::fill(u_fc3_b.pmr_vec(), 0.0f);

    // assert(npy_loader::load_npy_to_ndarray("cifar/u_conv1_w.npy", u_conv1_w));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_conv1_b.npy", u_conv1_b));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_conv2_w.npy", u_conv2_w));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_conv2_b.npy", u_conv2_b));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_conv3_w.npy", u_conv3_w));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_conv3_b.npy", u_conv3_b));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_conv4_w.npy", u_conv4_w));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_conv4_b.npy", u_conv4_b));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_conv5_w.npy", u_conv5_w));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_conv5_b.npy", u_conv5_b));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_fc1_w.npy", u_fc1_w));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_fc1_b.npy", u_fc1_b));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_fc2_w.npy", u_fc2_w));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_fc2_b.npy", u_fc2_b));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_fc3_w.npy", u_fc3_w));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_fc3_b.npy", u_fc3_b));
  }

  // Input and intermediate outputs
  Ndarray4D u_input;      // (128, 3, 32, 32)
  Ndarray4D u_conv1_out;  // (128, 64, 32, 32)
  Ndarray4D u_pool1_out;  // (128, 64, 16, 16)
  Ndarray4D u_conv2_out;  // (128, 192, 16, 16)
  Ndarray4D u_pool2_out;  // (128, 192, 8, 8)
  Ndarray4D u_conv3_out;  // (128, 384, 8, 8)
  Ndarray4D u_conv4_out;  // (128, 256, 8, 8)
  Ndarray4D u_conv5_out;  // (128, 256, 8, 8)
  Ndarray4D u_pool3_out;  // (128, 256, 4, 4)

  // Flatten would be (128, 4096), stored or created on-the-fly
  Ndarray2D u_fc1_out;  // (128, 4096)
  Ndarray2D u_fc2_out;  // (128, 4096)
  Ndarray2D u_fc3_out;  // shape = (128, 10) for final classification

  // Model parameters
  Ndarray4D u_conv1_w;  // (64, 3, 3, 3)
  Ndarray1D u_conv1_b;  // (64)
  Ndarray4D u_conv2_w;  // (192, 64, 3, 3)
  Ndarray1D u_conv2_b;  // (192)
  Ndarray4D u_conv3_w;  // (384, 192, 3, 3)
  Ndarray1D u_conv3_b;  // (384)
  Ndarray4D u_conv4_w;  // (256, 384, 3, 3)
  Ndarray1D u_conv4_b;  // (256)
  Ndarray4D u_conv5_w;  // (256, 256, 3, 3)
  Ndarray1D u_conv5_b;  // (256)
  Ndarray2D u_fc1_w;    // (4096, 4096)
  Ndarray1D u_fc1_b;    // (4096)
  Ndarray2D u_fc2_w;    // (4096, 4096)
  Ndarray1D u_fc2_b;    // (4096)
  Ndarray2D u_fc3_w;    // (10, 4096)
  Ndarray1D u_fc3_b;    // (10)
};

}  // namespace cifar_dense
