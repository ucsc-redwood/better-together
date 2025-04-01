#pragma once

#include <cuda_runtime.h>

// #include "../csr.hpp"

namespace cifar_sparse::cuda {

// ----------------------------------------------------------------------------
// Convolution 2D (Sparse)
// ----------------------------------------------------------------------------

// start, end = 0, weight_matrix.rows;
__global__ void conv2d(const float* input_data,
                       int image_input_channels,
                       int input_height,
                       int input_width,
                       //    const CSRMatrix& weight_matrix,
                       const float* weight_matrix_values,
                       const int* weight_matrix_row_ptr,
                       const int* weight_matrix_col_idx,
                       int weight_matrix_rows,
                       int weight_matrix_cols,
                       int weight_matrix_nnz,

                       const float* bias_data,
                       int bias_size,
                       int kernel_size,
                       int stride,
                       int padding,
                       bool relu,
                       float* output_data);

// ----------------------------------------------------------------------------
// Max Pooling 2D (Sparse)
// ----------------------------------------------------------------------------

__global__ void maxpool2d(const float* input_data,
                          int input_channels,
                          int input_height,
                          int input_width,
                          int pool_size,
                          int stride,
                          float* output_data);

// ----------------------------------------------------------------------------
// Linear Layer (Sparse)
// ----------------------------------------------------------------------------

__global__ void linear(const float* input_data,
                       //    const CSRMatrix& weight_matrix,
                       const float* weight_matrix_values,
                       const int* weight_matrix_row_ptr,
                       const int* weight_matrix_col_idx,
                       int weight_matrix_rows,
                       int weight_matrix_cols,
                       int weight_matrix_nnz,

                       const float* bias_data,
                       float* output_data);

}  // namespace cifar_sparse::cuda
