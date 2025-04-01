#pragma once

#include <cnpy.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>

#include "load_npy.hpp"

namespace cifar_dense {

struct AppData {
  explicit AppData(const std::string& input_file)
      : input(load_from_npy<3>(input_file, {3, 32, 32})),
        conv1_out({16, 32, 32}),
        pool1_out({16, 16, 16}),
        conv2_out({32, 16, 16}),
        pool2_out({32, 8, 8}),
        conv3_out({64, 8, 8}),
        conv4_out({64, 8, 8}),
        conv5_out({64, 8, 8}),
        pool3_out({64, 4, 4}),
        linear_out({10}),
        conv1_weights(load_from_npy<4>("conv1_w.npy", {16, 3, 3, 3})),
        conv1_bias(load_from_npy<1>("conv1_b.npy", {16})),
        conv2_weights(load_from_npy<4>("conv2_w.npy", {32, 16, 3, 3})),
        conv2_bias(load_from_npy<1>("conv2_b.npy", {32})),
        conv3_weights(load_from_npy<4>("conv3_w.npy", {64, 32, 3, 3})),
        conv3_bias(load_from_npy<1>("conv3_b.npy", {64})),
        conv4_weights(load_from_npy<4>("conv4_w.npy", {64, 64, 3, 3})),
        conv4_bias(load_from_npy<1>("conv4_b.npy", {64})),
        conv5_weights(load_from_npy<4>("conv5_w.npy", {64, 64, 3, 3})),
        conv5_bias(load_from_npy<1>("conv5_b.npy", {64})),
        linear_weights(load_from_npy<2>("linear_w.npy", {10, 1024})),
        linear_bias(load_from_npy<1>("linear_b.npy", {10})) {
    // conv1_out.print_shape("conv1_out");
    // pool1_out.print_shape("pool1_out");
    // conv2_out.print_shape("conv2_out");
    // pool2_out.print_shape("pool2_out");
    // conv3_out.print_shape("conv3_out");
    // conv4_out.print_shape("conv4_out");
    // conv5_out.print_shape("conv5_out");
    // pool3_out.print_shape("pool3_out");
    // linear_out.print_shape("linear_out");

    // // report total memory usage in MB
    // size_t total_memory_usage = 0;
    // total_memory_usage += conv1_out.memory_usage_bytes();
    // total_memory_usage += pool1_out.memory_usage_bytes();
    // total_memory_usage += conv2_out.memory_usage_bytes();
    // total_memory_usage += pool2_out.memory_usage_bytes();
    // total_memory_usage += conv3_out.memory_usage_bytes();
    // total_memory_usage += conv4_out.memory_usage_bytes();
    // total_memory_usage += conv5_out.memory_usage_bytes();
    // total_memory_usage += pool3_out.memory_usage_bytes();
    // total_memory_usage += linear_out.memory_usage_bytes();
    // total_memory_usage += conv1_weights.memory_usage_bytes();
    // total_memory_usage += conv2_weights.memory_usage_bytes();
    // total_memory_usage += conv3_weights.memory_usage_bytes();
    // total_memory_usage += conv4_weights.memory_usage_bytes();
    // total_memory_usage += conv5_weights.memory_usage_bytes();
    // total_memory_usage += linear_weights.memory_usage_bytes();
    // total_memory_usage += conv1_bias.memory_usage_bytes();
    // total_memory_usage += conv2_bias.memory_usage_bytes();
    // total_memory_usage += conv3_bias.memory_usage_bytes();
    // total_memory_usage += conv4_bias.memory_usage_bytes();
    // total_memory_usage += conv5_bias.memory_usage_bytes();
    // total_memory_usage += linear_bias.memory_usage_bytes();

    // std::cout << "Total memory usage: " << total_memory_usage / 1024.0 / 1024.0 << " MB"
    //           << std::endl;
  }

  const NDArray<3> input;
  NDArray<3> conv1_out;
  NDArray<3> pool1_out;
  NDArray<3> conv2_out;
  NDArray<3> pool2_out;
  NDArray<3> conv3_out;
  NDArray<3> conv4_out;
  NDArray<3> conv5_out;
  NDArray<3> pool3_out;
  NDArray<1> linear_out;

  const NDArray<4> conv1_weights;
  const NDArray<1> conv1_bias;
  const NDArray<4> conv2_weights;
  const NDArray<1> conv2_bias;
  const NDArray<4> conv3_weights;
  const NDArray<1> conv3_bias;
  const NDArray<4> conv4_weights;
  const NDArray<1> conv4_bias;
  const NDArray<4> conv5_weights;
  const NDArray<1> conv5_bias;
  const NDArray<2> linear_weights;
  const NDArray<1> linear_bias;
};

[[nodiscard]] inline int arg_max(const float* ptr) {
  const auto max_index = std::distance(ptr, std::ranges::max_element(ptr, ptr + 10));

  return max_index;
}

inline void print_prediction(const int max_index) {
  static const std::unordered_map<int, std::string_view> class_names{{0, "airplanes"},
                                                                     {1, "cars"},
                                                                     {2, "birds"},
                                                                     {3, "cats"},
                                                                     {4, "deer"},
                                                                     {5, "dogs"},
                                                                     {6, "frogs"},
                                                                     {7, "horses"},
                                                                     {8, "ships"},
                                                                     {9, "trucks"}};

  std::cout << "Predicted Image: ";
  std::cout << (class_names.contains(max_index) ? class_names.at(max_index) : "Unknown");
  std::cout << std::endl;
}

// ----------------------------------------------------------------------------
// Batched version (this is better)
// ----------------------------------------------------------------------------

struct AppDataBatch {
  static constexpr size_t BATCH_SIZE = 128;

  // Constructor loads a file containing a batch of 128 images,
  // shaped (128, 3, 32, 32).
  explicit AppDataBatch([[maybe_unused]] const std::string& input_file)
      :  // input(load_from_npy<4>(input_file, {BATCH_SIZE, 3, 32, 32})),

        input({BATCH_SIZE, 3, 32, 32}),

        // After conv1, shape => (128, 16, 32, 32)
        conv1_out({BATCH_SIZE, 16, 32, 32}),
        // Pool1 => (128, 16, 16, 16)
        pool1_out({BATCH_SIZE, 16, 16, 16}),

        conv2_out({BATCH_SIZE, 32, 16, 16}),
        pool2_out({BATCH_SIZE, 32, 8, 8}),

        conv3_out({BATCH_SIZE, 64, 8, 8}),
        conv4_out({BATCH_SIZE, 64, 8, 8}),
        conv5_out({BATCH_SIZE, 64, 8, 8}),
        // Pool3 => (128, 64, 4, 4)
        pool3_out({BATCH_SIZE, 64, 4, 4}),

        // After flatten, shape => (128, 1024), but we can create it on the fly.
        // The final linear output => (128, 10).
        linear_out({BATCH_SIZE, 10}),

        // Weights/bias shapes are unchanged (no batch dimension).
        conv1_weights(load_from_npy<4>("conv1_w.npy", {16, 3, 3, 3})),
        conv1_bias(load_from_npy<1>("conv1_b.npy", {16})),
        conv2_weights(load_from_npy<4>("conv2_w.npy", {32, 16, 3, 3})),
        conv2_bias(load_from_npy<1>("conv2_b.npy", {32})),
        conv3_weights(load_from_npy<4>("conv3_w.npy", {64, 32, 3, 3})),
        conv3_bias(load_from_npy<1>("conv3_b.npy", {64})),
        conv4_weights(load_from_npy<4>("conv4_w.npy", {64, 64, 3, 3})),
        conv4_bias(load_from_npy<1>("conv4_b.npy", {64})),
        conv5_weights(load_from_npy<4>("conv5_w.npy", {64, 64, 3, 3})),
        conv5_bias(load_from_npy<1>("conv5_b.npy", {64})),
        // Linear weights: shape (10, 1024)
        linear_weights(load_from_npy<2>("linear_w.npy", {10, 1024})),
        linear_bias(load_from_npy<1>("linear_b.npy", {10})) {
    // Optional: debug prints or memory usage checks
    // e.g. input.print_shape("Batched input");

    // Let's fill the input with random values
    std::mt19937 gen(114514);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Calculate total elements
    size_t total_elements = 1;
    for (auto dim : input.shape()) {
      total_elements *= dim;
    }

    // Fill with random values
    float* data_ptr = input.raw();
    for (size_t i = 0; i < total_elements; ++i) {
      data_ptr[i] = dis(gen);
    }
  }

  // Input and intermediate outputs
  NDArray<4> input;      // shape = (128, 3, 32, 32)
  NDArray<4> conv1_out;  // (128, 16, 32, 32)
  NDArray<4> pool1_out;  // (128, 16, 16, 16)
  NDArray<4> conv2_out;  // (128, 32, 16, 16)
  NDArray<4> pool2_out;  // (128, 32, 8, 8)
  NDArray<4> conv3_out;  // (128, 64, 8, 8)
  NDArray<4> conv4_out;  // (128, 64, 8, 8)
  NDArray<4> conv5_out;  // (128, 64, 8, 8)
  NDArray<4> pool3_out;  // (128, 64, 4, 4)

  // Flatten would be (128, 1024), stored or created on-the-fly
  NDArray<2> linear_out;  // shape = (128, 10) for final classification

  // Convolution & linear weights/biases (no batch dimension needed)
  const NDArray<4> conv1_weights;
  const NDArray<1> conv1_bias;
  const NDArray<4> conv2_weights;
  const NDArray<1> conv2_bias;
  const NDArray<4> conv3_weights;
  const NDArray<1> conv3_bias;
  const NDArray<4> conv4_weights;
  const NDArray<1> conv4_bias;
  const NDArray<4> conv5_weights;
  const NDArray<1> conv5_bias;
  const NDArray<2> linear_weights;  // (10, 1024)
  const NDArray<1> linear_bias;     // (10)
};

}  // namespace cifar_dense
