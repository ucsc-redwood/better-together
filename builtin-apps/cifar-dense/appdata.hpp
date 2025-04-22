#pragma once

#include <algorithm>
#include <memory_resource>
#include <random>

#include "../base_appdata.hpp"
#include "../load_npy.hpp"
#include "../ndarray.hpp"

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
  static constexpr size_t BATCH_SIZE = 128;

  // conv1: 16 output channels, 3×3×3 kernel = 27 inputs
  // conv2: 32 output channels, 16×3×3 kernel = 144 inputs
  // conv3: 64 output channels, 32×3×3 kernel = 288 inputs
  // conv4: 64 output channels, 64×3×3 kernel = 576 inputs
  // conv5: 64 output channels, 64×3×3 kernel = 576 inputs
  // linear: 10 output channels, 1024 inputs

  explicit AppData(std::pmr::memory_resource* mr)
      : BaseAppData(),
        u_input(BATCH_SIZE, 3, 32, 32, mr),
        u_conv1_out(BATCH_SIZE, 16, 32, 32, mr),
        u_pool1_out(BATCH_SIZE, 16, 16, 16, mr),
        u_conv2_out(BATCH_SIZE, 32, 16, 16, mr),
        u_pool2_out(BATCH_SIZE, 32, 8, 8, mr),
        u_conv3_out(BATCH_SIZE, 64, 8, 8, mr),
        u_conv4_out(BATCH_SIZE, 64, 8, 8, mr),
        u_conv5_out(BATCH_SIZE, 64, 8, 8, mr),
        u_pool3_out(BATCH_SIZE, 64, 4, 4, mr),
        u_linear_out(BATCH_SIZE, 10, mr),
        u_conv1_w(16, 3, 3, 3, mr),
        u_conv1_b(16, mr),
        u_conv2_w(32, 16, 3, 3, mr),
        u_conv2_b(32, mr),
        u_conv3_w(64, 32, 3, 3, mr),
        u_conv3_b(64, mr),
        u_conv4_w(64, 64, 3, 3, mr),
        u_conv4_b(64, mr),
        u_conv5_w(64, 64, 3, 3, mr),
        u_conv5_b(64, mr),
        u_linear_w(10, 1024, mr),
        u_linear_b(10, mr) {
    std::mt19937 gen(114514);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::ranges::generate(u_input.pmr_vec(), [&]() { return dis(gen); });

    std::uniform_real_distribution<float> weight_dis(-0.1f, 0.1f);
    std::ranges::generate(u_conv1_w.pmr_vec(), [&]() { return weight_dis(gen); });
    std::ranges::generate(u_conv2_w.pmr_vec(), [&]() { return weight_dis(gen); });
    std::ranges::generate(u_conv3_w.pmr_vec(), [&]() { return weight_dis(gen); });
    std::ranges::generate(u_conv4_w.pmr_vec(), [&]() { return weight_dis(gen); });
    std::ranges::generate(u_conv5_w.pmr_vec(), [&]() { return weight_dis(gen); });
    std::ranges::generate(u_linear_w.pmr_vec(), [&]() { return weight_dis(gen); });

    std::ranges::fill(u_conv1_b.pmr_vec(), 1.0f);
    std::ranges::fill(u_conv2_b.pmr_vec(), 1.0f);
    std::ranges::fill(u_conv3_b.pmr_vec(), 1.0f);
    std::ranges::fill(u_conv4_b.pmr_vec(), 1.0f);
    std::ranges::fill(u_conv5_b.pmr_vec(), 1.0f);
    std::ranges::fill(u_linear_b.pmr_vec(), 1.0f);

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
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_linear_w.npy", u_linear_w));
    // assert(npy_loader::load_npy_to_ndarray("cifar/u_linear_b.npy", u_linear_b));
  }

  // Input and intermediate outputs
  Ndarray4D u_input;      // (128, 3, 32, 32)
  Ndarray4D u_conv1_out;  // (128, 16, 32, 32)
  Ndarray4D u_pool1_out;  // (128, 16, 16, 16)
  Ndarray4D u_conv2_out;  // (128, 32, 16, 16)
  Ndarray4D u_pool2_out;  // (128, 32, 8, 8)
  Ndarray4D u_conv3_out;  // (128, 64, 8, 8)
  Ndarray4D u_conv4_out;  // (128, 64, 8, 8)
  Ndarray4D u_conv5_out;  // (128, 64, 8, 8)
  Ndarray4D u_pool3_out;  // (128, 64, 4, 4)

  // Flatten would be (128, 1024), stored or created on-the-fly
  Ndarray2D u_linear_out;  // shape = (128, 10) for final classification

  // Model parameters
  Ndarray4D u_conv1_w;   // (16, 3, 3, 3)
  Ndarray1D u_conv1_b;   // (16)
  Ndarray4D u_conv2_w;   // (32, 16, 3, 3)
  Ndarray1D u_conv2_b;   // (32)
  Ndarray4D u_conv3_w;   // (64, 32, 3, 3)
  Ndarray1D u_conv3_b;   // (64)
  Ndarray4D u_conv4_w;   // (64, 64, 3, 3)
  Ndarray1D u_conv4_b;   // (64)
  Ndarray4D u_conv5_w;   // (64, 64, 3, 3)
  Ndarray1D u_conv5_b;   // (64)
  Ndarray2D u_linear_w;  // (10, 1024)
  Ndarray1D u_linear_b;  // (10)
};

}  // namespace cifar_dense
