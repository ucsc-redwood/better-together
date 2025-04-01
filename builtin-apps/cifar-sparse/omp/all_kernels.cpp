#include "all_kernels.hpp"

#include <omp.h>
#include <spdlog/spdlog.h>

namespace cifar_sparse::omp {

// ----------------------------------------------------------------------------
// Convolution 2D (Sparse)
// ----------------------------------------------------------------------------
void conv2d_omp(const float *input_data,
                [[maybe_unused]] int image_input_channels,
                int input_height,
                int input_width,
                const CSRMatrix &weight_matrix,
                const float *bias_data,
                int bias_size,
                int kernel_size,
                int stride,
                int padding,
                bool relu,
                float *output_data,
                int start,
                int end) {
  int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
  int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

#pragma omp for
  for (int out_c = start; out_c < end; ++out_c) {
    int row_start = weight_matrix.row_ptr[out_c];
    int row_end = weight_matrix.row_ptr[out_c + 1];

    for (int oh = 0; oh < output_height; ++oh) {
      for (int ow = 0; ow < output_width; ++ow) {
        float sum = 0.0f;

        for (int nz_idx = row_start; nz_idx < row_end; ++nz_idx) {
          int flat_kernel_idx = weight_matrix.col_idx[nz_idx];
          float weight_value = weight_matrix.values[nz_idx];

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
  }
}

// ----------------------------------------------------------------------------
// Max Pooling 2D (Dense)
// ----------------------------------------------------------------------------
void maxpool2d_omp(const float *input_data,
                   [[maybe_unused]] int input_channels,
                   int input_height,
                   int input_width,
                   int pool_size,
                   int stride,
                   float *output_data,
                   int start,
                   int end) {
  int output_height = (input_height - pool_size) / stride + 1;
  int output_width = (input_width - pool_size) / stride + 1;

#pragma omp for
  for (int index = start; index < end; index++) {
    int c = index / (output_height * output_width);
    int h = (index / output_width) % output_height;
    int w = index % output_width;

    float max_val = -std::numeric_limits<float>::max();
    for (int p = 0; p < pool_size * pool_size; p++) {
      int ph = p / pool_size;
      int pw = p % pool_size;

      int input_h = h * stride + ph;
      int input_w = w * stride + pw;
      if (input_h < input_height && input_w < input_width) {
        int input_index = c * (input_height * input_width) + input_h * input_width + input_w;
        max_val = std::max(max_val, input_data[input_index]);
      }
    }
    int output_index = c * (output_height * output_width) + h * output_width + w;
    output_data[output_index] = max_val;
  }
}

// ----------------------------------------------------------------------------
// Linear Layer (Dense)
// ----------------------------------------------------------------------------
void linear_omp(const float *input_data,
                const CSRMatrix &weight_matrix,
                const float *bias_data,
                float *output_data,
                int start,
                int end) {
#pragma omp for
  for (int i = start; i < end; ++i) {
    float sum = 0.0f;

    for (int nz_idx = weight_matrix.row_ptr[i]; nz_idx < weight_matrix.row_ptr[i + 1]; ++nz_idx) {
      int col = weight_matrix.col_idx[nz_idx];
      sum += input_data[col] * weight_matrix.values[nz_idx];
    }

    output_data[i] = sum + bias_data[i];
  }
}

}  // namespace cifar_sparse::omp