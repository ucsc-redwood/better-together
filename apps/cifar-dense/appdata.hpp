#pragma once

#include <algorithm>
#include <cstdlib>
#include <memory_resource>
#include <random>
#include <string>

#include "platform/util/base_appdata.hpp"
#include "platform/util/ndarray.hpp"
#include "platform/util/npy_loader.hpp"

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
  // Batch is a FRAMEWORK workload knob, not a model property. The 11-stage
  // AlexNetCIFAR is ~27x heavier per image than the old SmallAlexNet; at batch
  // 128 a dense task took ~700 ms on the Jetson GPU and the pipeline-e2e suite
  // ran 12+ minutes. 16 keeps per-task cost in the old model's ballpark while
  // the 4096-wide FC GEMMs still see reasonable GPU utilization. (The paper ran
  // dense at 1 image/task; sparse keeps 128 -- its per-image cost is far lower.)
  static constexpr size_t BATCH_SIZE = 16;

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
    // BT_WEIGHTS_DIR set -> the real trained AlexNetCIFAR export + real test
    // batch, fail-loud on any problem. Unset -> the synthetic seeded init
    // below, byte-identical to the hermetic-test behavior.
    if (const char* dir = std::getenv("BT_WEIGHTS_DIR")) {
      load_real_weights(dir);
      return;
    }

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
  }

  // Load the real BN-folded weights ($BT_WEIGHTS_DIR/dense/, PyTorch OIHW /
  // (out,in) row-major) and the first BATCH_SIZE images of the real normalized
  // CIFAR-10 test batch. Any missing file or shape mismatch throws
  // (bt::npy::load) -- never a silent fallback.
  // See docs/instruction-for-ai/04-alexnet-cifar-spec.md §7.
  void load_real_weights(const std::string& dir) {
    const auto f1 = [](const std::string& p, Ndarray1D& a) {
      bt::npy::load(p, "<f4", {static_cast<size_t>(a.d0())}, a.data());
    };
    const auto f2 = [](const std::string& p, Ndarray2D& a) {
      bt::npy::load(p, "<f4", {static_cast<size_t>(a.d0()), static_cast<size_t>(a.d1())}, a.data());
    };
    const auto f4 = [](const std::string& p, Ndarray4D& a) {
      bt::npy::load(p,
                    "<f4",
                    {static_cast<size_t>(a.d0()),
                     static_cast<size_t>(a.d1()),
                     static_cast<size_t>(a.d2()),
                     static_cast<size_t>(a.d3())},
                    a.data());
    };

    const std::string d = dir + "/dense/";
    f4(d + "conv1_w.npy", u_conv1_w);
    f1(d + "conv1_b.npy", u_conv1_b);
    f4(d + "conv2_w.npy", u_conv2_w);
    f1(d + "conv2_b.npy", u_conv2_b);
    f4(d + "conv3_w.npy", u_conv3_w);
    f1(d + "conv3_b.npy", u_conv3_b);
    f4(d + "conv4_w.npy", u_conv4_w);
    f1(d + "conv4_b.npy", u_conv4_b);
    f4(d + "conv5_w.npy", u_conv5_w);
    f1(d + "conv5_b.npy", u_conv5_b);
    f2(d + "fc1_w.npy", u_fc1_w);
    f1(d + "fc1_b.npy", u_fc1_b);
    f2(d + "fc2_w.npy", u_fc2_w);
    f1(d + "fc2_b.npy", u_fc2_b);
    f2(d + "fc3_w.npy", u_fc3_w);
    f1(d + "fc3_b.npy", u_fc3_b);

    // u_input is (BATCH_SIZE, 3, 32, 32); the exported test batch holds 128
    // images -- take its first BATCH_SIZE rows (trailing dims still checked).
    bt::npy::load_prefix(dir + "/test_batch.npy",
                         "<f4",
                         {BATCH_SIZE, 3, 32, 32},
                         u_input.data());
  }

  // Input and intermediate outputs
  Ndarray4D u_input;      // (N=16, 3, 32, 32)
  Ndarray4D u_conv1_out;  // (N=16, 64, 32, 32)
  Ndarray4D u_pool1_out;  // (N=16, 64, 16, 16)
  Ndarray4D u_conv2_out;  // (N=16, 192, 16, 16)
  Ndarray4D u_pool2_out;  // (N=16, 192, 8, 8)
  Ndarray4D u_conv3_out;  // (N=16, 384, 8, 8)
  Ndarray4D u_conv4_out;  // (N=16, 256, 8, 8)
  Ndarray4D u_conv5_out;  // (N=16, 256, 8, 8)
  Ndarray4D u_pool3_out;  // (N=16, 256, 4, 4)

  // Flatten would be (N, 4096), stored or created on-the-fly
  Ndarray2D u_fc1_out;  // (N=16, 4096)
  Ndarray2D u_fc2_out;  // (N=16, 4096)
  Ndarray2D u_fc3_out;  // shape = (N=16, 10) for final classification

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
