#include <device_launch_parameters.h>

#include <cfloat>

#include "all_kernels.cuh"

namespace cifar_sparse::cuda {

// ----------------------------------------------------------------------------
// Convolution 2D (Sparse)
// ----------------------------------------------------------------------------

__global__ void conv2d(const float* input_data,
                       const int image_input_channels,
                       const int input_height,
                       const int input_width,
                       //    const CSRMatrix& weight_matrix,

                       const float* weight_matrix_values,
                       const int* weight_matrix_row_ptr,
                       const int* weight_matrix_col_idx,
                       const int weight_matrix_rows,
                       const int weight_matrix_cols,
                       const int weight_matrix_nnz,

                       const float* bias_data,
                       const int bias_size,
                       const int kernel_size,
                       const int stride,
                       const int padding,
                       const bool relu,
                       float* output_data) {
  auto thread_idx = threadIdx.x;
  auto i = blockIdx.x * blockDim.x + thread_idx;

  int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
  int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
  // int output_channels = weight_matrix.rows;
  // int spatial_size = kernel_size * kernel_size * image_input_channels;

  // // Zero initialize output
  // int output_size = output_channels * output_height * output_width;
  // for (int i = 0; i < output_size; ++i) {
  //   output_data[i] = 0.0f;
  // }

  //   for (int out_c = start; out_c < end; ++out_c) {
  if (i >= weight_matrix_rows) {
    return;
  }

  auto out_c = i;

  // for (int out_c = 0; out_c < output_channels; ++out_c) {
  int row_start = weight_matrix_row_ptr[out_c];
  int row_end = weight_matrix_row_ptr[out_c + 1];

  for (int oh = 0; oh < output_height; ++oh) {
    for (int ow = 0; ow < output_width; ++ow) {
      float sum = 0.0f;

      for (int nz_idx = row_start; nz_idx < row_end; ++nz_idx) {
        int flat_kernel_idx = weight_matrix_col_idx[nz_idx];
        float weight_value = weight_matrix_values[nz_idx];

        int in_c = flat_kernel_idx / (kernel_size * kernel_size);
        int rem = flat_kernel_idx % (kernel_size * kernel_size);
        int ky = rem / kernel_size;
        int kx = rem % kernel_size;

        int ih = oh * stride + ky - padding;
        int iw = ow * stride + kx - padding;

        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
          int input_idx = (in_c * input_height + ih) * input_width + iw;
          sum += input_data[input_idx] * weight_value;
        }
      }

      if (bias_data && out_c < bias_size) {
        sum += bias_data[out_c];
      }

      if (relu && sum < 0) {
        sum = 0.0f;
      }

      output_data[(out_c * output_height + oh) * output_width + ow] = sum;
    }
  }

  //   }
}

// // start, end = 0, weight_matrix.rows;
// __global__ void conv2d(const float* input_data,
//                        const int image_input_channels,
//                        const int input_height,
//                        const int input_width,
//                        const CSRMatrix& weight_matrix,
//                        const float* bias_data,
//                        const int bias_size,
//                        const int kernel_size,
//                        const int stride,
//                        const int padding,
//                        const bool relu,
//                        float* output_data
//                        //    const int output_height,
//                        //    const int output_width,
// ) {
//   int out_c = blockIdx.x;                          // Output channel index
//   int oh = blockIdx.y * blockDim.y + threadIdx.y;  // Output height index
//   int ow = blockIdx.z * blockDim.z + threadIdx.z;  // Output width index

//   auto output_height = (input_height + 2 * padding - kernel_size) / stride +
//   1; auto output_width = (input_width + 2 * padding - kernel_size) / stride +
//   1;

//   if (out_c >= weight_matrix.rows || oh >= output_height ||
//       ow >= output_width) {
//     return;  // Out-of-bounds check
//   }

//   int row_start = weight_matrix.row_ptr[out_c];
//   int row_end = weight_matrix.row_ptr[out_c + 1];
//   float sum = 0.0f;

//   for (int nz_idx = row_start; nz_idx < row_end; ++nz_idx) {
//     int flat_kernel_idx = weight_matrix.col_idx[nz_idx];
//     float weight_value = weight_matrix.values[nz_idx];

//     int in_c = flat_kernel_idx / (kernel_size * kernel_size);
//     int rem = flat_kernel_idx % (kernel_size * kernel_size);
//     int ky = rem / kernel_size;
//     int kx = rem % kernel_size;

//     int ih = oh * stride + ky - padding;
//     int iw = ow * stride + kx - padding;

//     if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
//       int input_idx = (in_c * input_height + ih) * input_width + iw;
//       sum += input_data[input_idx] * weight_value;
//     }
//   }

//   if (bias_data && out_c < bias_size) {
//     sum += bias_data[out_c];
//   }

//   if (relu && sum < 0) {
//     sum = 0.0f;
//   }

//   int output_idx = (out_c * output_height + oh) * output_width + ow;
//   output_data[output_idx] = sum;
// }

// ----------------------------------------------------------------------------
// Max Pooling 2D (Sparse)
// ----------------------------------------------------------------------------

__global__ void maxpool2d(const float* input_data,
                          int input_channels,
                          int input_height,
                          int input_width,
                          int pool_size,
                          int stride,
                          float* output_data) {
  auto thread_idx = threadIdx.x;
  auto i = blockIdx.x * blockDim.x + thread_idx;

  int output_height = (input_height - pool_size) / stride + 1;
  int output_width = (input_width - pool_size) / stride + 1;
  // int total_iterations = input_channels * output_height * output_width;

  if (i >= input_channels * output_height * output_width) {
    return;
  }

  auto index = i;

  //   for (int index = start; index < end; index++) {
  int c = index / (output_height * output_width);
  int h = (index / output_width) % output_height;
  int w = index % output_width;

  float max_val = -FLT_MAX;
  for (int p = 0; p < pool_size * pool_size; p++) {
    int ph = p / pool_size;
    int pw = p % pool_size;

    int input_h = h * stride + ph;
    int input_w = w * stride + pw;
    if (input_h < input_height && input_w < input_width) {
      int input_index = c * (input_height * input_width) + input_h * input_width + input_w;
      max_val = max(max_val, input_data[input_index]);
    }
  }
  int output_index = c * (output_height * output_width) + h * output_width + w;
  output_data[output_index] = max_val;
  //   }
}

// ----------------------------------------------------------------------------
// Linear Layer (Sparse)
// ----------------------------------------------------------------------------

__global__ void linear(const float* input_data,
                       //    const CSRMatrix& weight_matrix,

                       const float* weight_matrix_values,
                       const int* weight_matrix_row_ptr,
                       const int* weight_matrix_col_idx,
                       const int weight_matrix_rows,
                       const int weight_matrix_cols,
                       const int weight_matrix_nnz,

                       const float* bias_data,
                       float* output_data) {
  auto thread_idx = threadIdx.x;
  auto i = blockIdx.x * blockDim.x + thread_idx;

  if (i >= weight_matrix_rows) {
    return;
  }

  //   for (int i = start; i < end; ++i) {
  //   for (int i = start; i < end; ++i) {
  float sum = 0.0f;

  for (int nz_idx = weight_matrix_row_ptr[i]; nz_idx < weight_matrix_row_ptr[i + 1]; ++nz_idx) {
    int col = weight_matrix_col_idx[nz_idx];
    sum += input_data[col] * weight_matrix_values[nz_idx];
  }

  output_data[i] = sum + bias_data[i];
  //   }
}

// ----------------------------------------------------------------------------
// v2
// ----------------------------------------------------------------------------

namespace v2 {

__global__ void conv2d_cuda_kernel(const float* input_data,
                                   int batch_size,
                                   int in_channels,
                                   int in_height,
                                   int in_width,
                                   const float* weight_vals,
                                   const int* weight_row_ptr,
                                   const int* weight_col_idx,
                                   int out_channels,
                                   const float* bias_data,
                                   int bias_size,
                                   int kernel_size,
                                   int stride,
                                   int padding,
                                   bool relu,
                                   float* output_data,
                                   int out_height,
                                   int out_width) {
  // Compute a linear index across the entire output tensor.
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total_output = batch_size * out_channels * out_height * out_width;
  if (index >= total_output) return;

  // Decode the 1D index into 4D indices: (b, out_c, oh, ow).
  int ow = index % out_width;
  int tmp = index / out_width;
  int oh = tmp % out_height;
  tmp = tmp / out_height;
  int out_c = tmp % out_channels;
  int b = tmp / out_channels;

  float sum = 0.0f;
  // Get the start and end indices in the CSR for this output channel.
  int row_start = weight_row_ptr[out_c];
  int row_end = weight_row_ptr[out_c + 1];
  int kernel_area = kernel_size * kernel_size;

  // Loop over each nonzero weight contributing to this output channel.
  for (int nz = row_start; nz < row_end; nz++) {
    int flat_kernel_idx = weight_col_idx[nz];
    float weight_val = weight_vals[nz];

    // Decode the flat index into an input channel and kernel (ky, kx) position.
    int in_c = flat_kernel_idx / kernel_area;
    int rem = flat_kernel_idx % kernel_area;
    int ky = rem / kernel_size;
    int kx = rem % kernel_size;

    // Compute the corresponding input spatial coordinates.
    int in_y = oh * stride + ky - padding;
    int in_x = ow * stride + kx - padding;

    // If within the input boundaries, accumulate the weighted input.
    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
      int input_idx = ((b * in_channels + in_c) * in_height + in_y) * in_width + in_x;
      sum += input_data[input_idx] * weight_val;
    }
  }

  // Add bias if provided.
  if (bias_data != nullptr && out_c < bias_size) {
    sum += bias_data[out_c];
  }
  // Apply ReLU activation if needed.
  if (relu && sum < 0.0f) {
    sum = 0.0f;
  }

  // Write the computed sum to the output tensor.
  output_data[index] = sum;
}

}  // namespace v2

}  // namespace cifar_sparse::cuda
