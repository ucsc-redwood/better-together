#pragma once

#include <memory_resource>
#include <vector>

// #include "builtin-apps/cifar-sparse/ndarray.h"
#include "ndarray.hpp"

namespace cifar_sparse {

// note this pointer may came from USM vector
struct CSRMatrix {
  const float* values;
  const int* row_ptr;
  const int* col_idx;
  int rows;
  int cols;
  int nnz;
};

// Maximum sizes for static arrays
constexpr int MAX_NNZ_CONV1 = 1728;    // 3*3*3*64
constexpr int MAX_NNZ_CONV2 = 110592;  // 3*3*64*192
constexpr int MAX_NNZ_CONV3 = 663552;  // 3*3*192*384
constexpr int MAX_NNZ_CONV4 = 884736;  // 3*3*384*256
constexpr int MAX_NNZ_CONV5 = 589824;  // 3*3*256*256
constexpr int MAX_NNZ_LINEAR = 40960;  // 256*4*4*10

struct AppData {
  std::pmr::vector<float> u_image_data;  // initial input

  std::pmr::vector<float> u_conv1_values;
  std::pmr::vector<int> u_conv1_row_ptr;
  std::pmr::vector<int> u_conv1_col_idx;

  std::pmr::vector<float> u_conv2_values;
  std::pmr::vector<int> u_conv2_row_ptr;
  std::pmr::vector<int> u_conv2_col_idx;

  std::pmr::vector<float> u_conv3_values;
  std::pmr::vector<int> u_conv3_row_ptr;
  std::pmr::vector<int> u_conv3_col_idx;

  std::pmr::vector<float> u_conv4_values;
  std::pmr::vector<int> u_conv4_row_ptr;
  std::pmr::vector<int> u_conv4_col_idx;

  std::pmr::vector<float> u_conv5_values;
  std::pmr::vector<int> u_conv5_row_ptr;
  std::pmr::vector<int> u_conv5_col_idx;

  std::pmr::vector<float> u_linear_values;
  std::pmr::vector<int> u_linear_row_ptr;
  std::pmr::vector<int> u_linear_col_idx;

  std::pmr::vector<float> u_conv1_output;
  std::pmr::vector<float> u_pool1_output;
  std::pmr::vector<float> u_conv2_output;
  std::pmr::vector<float> u_pool2_output;
  std::pmr::vector<float> u_conv3_output;
  std::pmr::vector<float> u_conv4_output;
  std::pmr::vector<float> u_conv5_output;
  std::pmr::vector<float> u_pool3_output;
  std::pmr::vector<float> u_linear_output;  // final output

  std::pmr::vector<float> u_conv1_bias;
  std::pmr::vector<float> u_conv2_bias;
  std::pmr::vector<float> u_conv3_bias;
  std::pmr::vector<float> u_conv4_bias;
  std::pmr::vector<float> u_conv5_bias;
  std::pmr::vector<float> u_linear_bias;

  CSRMatrix conv1_weights;
  CSRMatrix conv2_weights;
  CSRMatrix conv3_weights;
  CSRMatrix conv4_weights;
  CSRMatrix conv5_weights;
  CSRMatrix linear_weights;

  explicit AppData(std::pmr::memory_resource* mr);
};

namespace v2 {

struct CSRMatrix {
  const int rows;
  const int cols;
  const int nnz;
  std::pmr::vector<float> values;
  std::pmr::vector<int> row_ptr;
  std::pmr::vector<int> col_idx;

  // Basic constructor
  CSRMatrix(const int r,
            const int c,
            std::pmr::memory_resource* mr = std::pmr::get_default_resource())
      : rows(r),
        cols(c),
        nnz(0),
        values(r * c, 0.0f, mr),
        row_ptr(r + 1, 0, mr),
        col_idx(r * c, 0, mr) {}

  // Get raw pointers for compatibility with old code
  [[nodiscard]] const float* values_data() const { return values.data(); }
  [[nodiscard]] const int* row_ptr_data() const { return row_ptr.data(); }
  [[nodiscard]] const int* col_idx_data() const { return col_idx.data(); }
  [[nodiscard]] std::pmr::vector<float>& values_pmr_vec() { return values; }
  [[nodiscard]] std::pmr::vector<int>& row_ptr_pmr_vec() { return row_ptr; }
  [[nodiscard]] std::pmr::vector<int>& col_idx_pmr_vec() { return col_idx; }
  [[nodiscard]] const std::pmr::vector<float>& values_pmr_vec() const { return values; }
  [[nodiscard]] const std::pmr::vector<int>& row_ptr_pmr_vec() const { return row_ptr; }
  [[nodiscard]] const std::pmr::vector<int>& col_idx_pmr_vec() const { return col_idx; }
};

struct AppData {
  static constexpr size_t BATCH_SIZE = 512;

  // conv1: 16 output channels, 3×3×3 kernel = 27 inputs
  // conv2: 32 output channels, 16×3×3 kernel = 144 inputs
  // conv3: 64 output channels, 32×3×3 kernel = 288 inputs
  // conv4: 64 output channels, 64×3×3 kernel = 576 inputs
  // conv5: 64 output channels, 64×3×3 kernel = 576 inputs
  // linear: 10 output channels, 1024 inputs

  explicit AppData(std::pmr::memory_resource* mr)
      : u_input(BATCH_SIZE, 3, 32, 32, mr),
        u_conv1_out(BATCH_SIZE, 16, 32, 32, mr),
        u_pool1_out(BATCH_SIZE, 16, 16, 16, mr),
        u_conv2_out(BATCH_SIZE, 32, 16, 16, mr),
        u_pool2_out(BATCH_SIZE, 32, 8, 8, mr),
        u_conv3_out(BATCH_SIZE, 64, 8, 8, mr),
        u_conv4_out(BATCH_SIZE, 64, 8, 8, mr),
        u_conv5_out(BATCH_SIZE, 64, 8, 8, mr),
        u_pool3_out(BATCH_SIZE, 64, 4, 4, mr),
        u_linear_out(BATCH_SIZE, 10, mr),
        u_conv1_b(16, mr),
        u_conv2_b(32, mr),
        u_conv3_b(64, mr),
        u_conv4_b(64, mr),
        u_conv5_b(64, mr),
        u_linear_b(10, mr),
        // Initialize CSR matrices
        conv1_sparse(16, 27, mr),
        conv2_sparse(32, 144, mr),  // 16*3*3*192
        conv3_sparse(64, 288, mr),
        conv4_sparse(64, 576, mr),
        conv5_sparse(64, 576, mr),
        linear_sparse(10, 1024, mr) {}

  void reset() {
    // u_input.pmr_vec().clear();
    // u_conv1_out.pmr_vec().clear();
    // u_pool1_out.pmr_vec().clear();
    // u_conv2_out.pmr_vec().clear();
    // u_pool2_out.pmr_vec().clear();
    // u_conv3_out.pmr_vec().clear();
    // u_conv4_out.pmr_vec().clear();
    // u_conv5_out.pmr_vec().clear();
    // u_pool3_out.pmr_vec().clear();
    // u_linear_out.pmr_vec().clear();
    // u_conv1_b.pmr_vec().clear();
    // u_conv2_b.pmr_vec().clear();
    // u_conv3_b.pmr_vec().clear();
    // u_conv4_b.pmr_vec().clear();
    // u_conv5_b.pmr_vec().clear();
    // u_linear_b.pmr_vec().clear();
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
  Ndarray1D u_conv1_b;
  Ndarray1D u_conv2_b;
  Ndarray1D u_conv3_b;
  Ndarray1D u_conv4_b;
  Ndarray1D u_conv5_b;
  Ndarray1D u_linear_b;  // (10)

  // Sparse matrices
  CSRMatrix conv1_sparse;   // (16, 27)
  CSRMatrix conv2_sparse;   // (32, 144)
  CSRMatrix conv3_sparse;   // (64, 288)
  CSRMatrix conv4_sparse;   // (64, 576)
  CSRMatrix conv5_sparse;   // (64, 576)
  CSRMatrix linear_sparse;  // (10, 1024)
};

}  // namespace v2

}  // namespace cifar_sparse
