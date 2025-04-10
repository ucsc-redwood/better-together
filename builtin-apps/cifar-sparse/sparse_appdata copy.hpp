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
  CSRMatrix(int r,
            int c,
            int estimated_nnz,
            std::pmr::memory_resource* mr = std::pmr::get_default_resource())
      : rows(r),
        cols(c),
        nnz(0),
        values(estimated_nnz, 0.0f, mr),
        row_ptr(r + 1, 0, mr),
        col_idx(estimated_nnz, 0, mr) {}

  // Get raw pointers for compatibility with old code
  const float* values_data() const { return values.data(); }
  const int* row_ptr_data() const { return row_ptr.data(); }
  const int* col_idx_data() const { return col_idx.data(); }
};

struct AppData {
  static constexpr size_t BATCH_SIZE = 128;

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
        u_linear_b(10, mr),
        // Initialize CSR matrices
        conv1_sparse(16, 27, MAX_NNZ_CONV1, mr),
        conv2_sparse(32, 144, MAX_NNZ_CONV2, mr),
        conv3_sparse(64, 288, MAX_NNZ_CONV3, mr),
        conv4_sparse(64, 576, MAX_NNZ_CONV4, mr),
        conv5_sparse(64, 576, MAX_NNZ_CONV5, mr),
        linear_sparse(10, 1024, MAX_NNZ_LINEAR, mr) {}

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
  Ndarray4D u_conv1_w;
  Ndarray1D u_conv1_b;
  Ndarray4D u_conv2_w;
  Ndarray1D u_conv2_b;
  Ndarray4D u_conv3_w;
  Ndarray1D u_conv3_b;
  Ndarray4D u_conv4_w;
  Ndarray1D u_conv4_b;
  Ndarray4D u_conv5_w;
  Ndarray1D u_conv5_b;
  Ndarray2D u_linear_w;  // (10, 1024)
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
