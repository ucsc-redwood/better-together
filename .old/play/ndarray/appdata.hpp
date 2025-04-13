#pragma once

#include <cnpy.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <memory_resource>
#include <random>
#include <string>
#include <unordered_map>

#include "load_npy.hpp"

namespace cifar_dense {

// ----------------------------------------------------------------------------
// Batched version (this is better)
// ----------------------------------------------------------------------------

// struct ModelData {
//   NDArray<4> h_conv1_w;
//   NDArray<1> h_conv1_b;
//   NDArray<4> h_conv2_w;
//   NDArray<1> h_conv2_b;
//   NDArray<4> h_conv3_w;
//   NDArray<1> h_conv3_b;
//   NDArray<4> h_conv4_w;
//   NDArray<1> h_conv4_b;
//   NDArray<4> h_conv5_w;
//   NDArray<1> h_conv5_b;
//   NDArray<2> h_linear_w;  // (10, 1024)
//   NDArray<1> h_linear_b;  // (10)

//   explicit ModelData()
//       : h_conv1_w({16, 3, 3, 3}),
//         h_conv1_b({16}),
//         h_conv2_w({32, 16, 3, 3}),
//         h_conv2_b({32}),
//         h_conv3_w({64, 32, 3, 3}),
//         h_conv3_b({64}),
//         h_conv4_w({64, 64, 3, 3}),
//         h_conv4_b({64}),
//         h_conv5_w({64, 64, 3, 3}),
//         h_conv5_b({64}),
//         h_linear_w({10, 1024}),
//         h_linear_b({10}) {
//     spdlog::trace("ModelData::ModelData(), Loading model data from npy files");
//     load_from_npy_to_raw("conv1_w.npy", h_conv1_w);
//     load_from_npy_to_raw("conv1_b.npy", h_conv1_b);
//     load_from_npy_to_raw("conv2_w.npy", h_conv2_w);
//     load_from_npy_to_raw("conv2_b.npy", h_conv2_b);
//     load_from_npy_to_raw("conv3_w.npy", h_conv3_w);
//     load_from_npy_to_raw("conv3_b.npy", h_conv3_b);
//     load_from_npy_to_raw("conv4_w.npy", h_conv4_w);
//     load_from_npy_to_raw("conv4_b.npy", h_conv4_b);
//     load_from_npy_to_raw("conv5_w.npy", h_conv5_w);
//     load_from_npy_to_raw("conv5_b.npy", h_conv5_b);
//     load_from_npy_to_raw("linear_w.npy", h_linear_w);
//     load_from_npy_to_raw("linear_b.npy", h_linear_b);
//   }
// };

struct AppDataBatch {
  static constexpr size_t BATCH_SIZE = 128;

  // Constructor loads a file containing a batch of 128 images, shaped (128, 3, 32, 32).
  explicit AppDataBatch(std::pmr::memory_resource* mr)
      : u_input({BATCH_SIZE, 3, 32, 32}, mr),
        u_conv1_out({BATCH_SIZE, 16, 32, 32}, mr),
        u_pool1_out({BATCH_SIZE, 16, 16, 16}, mr),
        u_conv2_out({BATCH_SIZE, 32, 16, 16}, mr),
        u_pool2_out({BATCH_SIZE, 32, 8, 8}, mr),
        u_conv3_out({BATCH_SIZE, 64, 8, 8}, mr),
        u_conv4_out({BATCH_SIZE, 64, 8, 8}, mr),
        u_conv5_out({BATCH_SIZE, 64, 8, 8}, mr),
        u_pool3_out({BATCH_SIZE, 64, 4, 4}, mr),
        u_linear_out({BATCH_SIZE, 10}, mr),
        u_conv1_w({16, 3, 3, 3}, mr),
        u_conv1_b({16}, mr),
        u_conv2_w({32, 16, 3, 3}, mr),
        u_conv2_b({32}, mr),
        u_conv3_w({64, 32, 3, 3}, mr),
        u_conv3_b({64}, mr),
        u_conv4_w({64, 64, 3, 3}, mr),
        u_conv4_b({64}, mr),
        u_conv5_w({64, 64, 3, 3}, mr),
        u_conv5_b({64}, mr),
        u_linear_w({10, 1024}, mr),
        u_linear_b({10}, mr) {
    spdlog::trace("AppDataBatch::AppDataBatch(), Initializing AppDataBatch");

    // Fill with random values for now
    std::generate_n(u_input.raw(), u_input.size(), [] {
      static std::mt19937 gen(114514);
      static std::uniform_real_distribution<> dis(0.0, 1.0);
      return dis(gen);
    });

    load_from_npy_to_raw("conv1_w.npy", u_conv1_w);
    load_from_npy_to_raw("conv1_b.npy", u_conv1_b);
    load_from_npy_to_raw("conv2_w.npy", u_conv2_w);
    load_from_npy_to_raw("conv2_b.npy", u_conv2_b);
    load_from_npy_to_raw("conv3_w.npy", u_conv3_w);
    load_from_npy_to_raw("conv3_b.npy", u_conv3_b);
    load_from_npy_to_raw("conv4_w.npy", u_conv4_w);
    load_from_npy_to_raw("conv4_b.npy", u_conv4_b);
    load_from_npy_to_raw("conv5_w.npy", u_conv5_w);
    load_from_npy_to_raw("conv5_b.npy", u_conv5_b);
    load_from_npy_to_raw("linear_w.npy", u_linear_w);
    load_from_npy_to_raw("linear_b.npy", u_linear_b);
  }

  // Input and intermediate outputs
  NDArray<4> u_input;      // shape = (128, 3, 32, 32)
  NDArray<4> u_conv1_out;  // (128, 16, 32, 32)
  NDArray<4> u_pool1_out;  // (128, 16, 16, 16)
  NDArray<4> u_conv2_out;  // (128, 32, 16, 16)
  NDArray<4> u_pool2_out;  // (128, 32, 8, 8)
  NDArray<4> u_conv3_out;  // (128, 64, 8, 8)
  NDArray<4> u_conv4_out;  // (128, 64, 8, 8)
  NDArray<4> u_conv5_out;  // (128, 64, 8, 8)
  NDArray<4> u_pool3_out;  // (128, 64, 4, 4)

  // Flatten would be (128, 1024), stored or created on-the-fly
  NDArray<2> u_linear_out;  // shape = (128, 10) for final classification

  // Model parameters
  NDArray<4> u_conv1_w;
  NDArray<1> u_conv1_b;
  NDArray<4> u_conv2_w;
  NDArray<1> u_conv2_b;
  NDArray<4> u_conv3_w;
  NDArray<1> u_conv3_b;
  NDArray<4> u_conv4_w;
  NDArray<1> u_conv4_b;
  NDArray<4> u_conv5_w;
  NDArray<1> u_conv5_b;
  NDArray<2> u_linear_w;  // (10, 1024)
  NDArray<1> u_linear_b;  // (10)

  // static const ModelData& get_model() {
  //   static const ModelData model_data;
  //   return model_data;
  // }
};

// ----------------------------------------------------------------------------
// Helper functions
// ----------------------------------------------------------------------------

inline void print_prediction(const int max_index) {
  static const std::unordered_map<int, std::string> class_names{{0, "airplanes"},
                                                                {1, "cars"},
                                                                {2, "birds"},
                                                                {3, "cats"},
                                                                {4, "deer"},
                                                                {5, "dogs"},
                                                                {6, "frogs"},
                                                                {7, "horses"},
                                                                {8, "ships"},
                                                                {9, "trucks"}};

  std::cout << "\t===>\t";
  std::cout << (class_names.contains(max_index) ? class_names.at(max_index) : "Unknown");
  std::cout << '\n';
}

inline void print_batch_predictions(const AppDataBatch& batch_data, const size_t num_to_print = 1) {
  // Get the output predictions (batch_size x 10 values)
  const auto& predictions = batch_data.u_linear_out;

  std::cout << "Predictions for batch of " << AppDataBatch::BATCH_SIZE << " images:" << std::endl;

  // For each image in the batch
  for (size_t i = 0; i < num_to_print; ++i) {
    // Find the index of the maximum value (predicted class)
    const float* pred_ptr = predictions.raw() + i * 10;  // Point to the 10 values for this image
    const auto max_index = std::distance(pred_ptr, std::max_element(pred_ptr, pred_ptr + 10));

    std::cout << "Image " << i << ": ";

    // also print the raw value
    // "[xxx,xxx,xxx,...,xxx]"
    std::cout << "[";
    for (size_t j = 0; j < 10; ++j) {
      std::cout << pred_ptr[j] << ",";
    }
    std::cout << "]\t";

    print_prediction(max_index);
  }
}

}  // namespace cifar_dense
