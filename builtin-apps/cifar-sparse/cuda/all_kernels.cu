#include <cfloat>

#include "all_kernels.cuh"

namespace cifar_sparse::cuda {

void conv2d_csr_batch_kernel(const float* __restrict__ input_data,
                             int batch_size,
                             int in_channels,
                             int in_height,
                             int in_width,
                             const float* __restrict__ weight_vals,
                             const int* __restrict__ weight_row_ptr,
                             const int* __restrict__ weight_col_idx,
                             int out_channels,
                             const float* __restrict__ bias_data,
                             int bias_size,
                             int kernel_size,
                             int stride,
                             int padding,
                             bool relu,
                             float* __restrict__ output_data) {
  // recompute output dims
  int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
  int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * out_channels * out_height * out_width;
  if (idx >= total) return;

  // decode (b, out_c, oh, ow)
  int ow = idx % out_width;
  int tmp = idx / out_width;
  int oh = tmp % out_height;
  tmp /= out_height;
  int out_c = tmp % out_channels;
  int b = tmp / out_channels;

  float sum = 0.0f;
  int row_start = weight_row_ptr[out_c];
  int row_end = weight_row_ptr[out_c + 1];
  int area = kernel_size * kernel_size;

  // loop over nonzeros in this output channel's CSR row
  for (int nz = row_start; nz < row_end; ++nz) {
    int flat_k = weight_col_idx[nz];
    float w = weight_vals[nz];

    int in_c = flat_k / area;
    int rem = flat_k % area;
    int ky = rem / kernel_size;
    int kx = rem % kernel_size;

    int in_y = oh * stride + ky - padding;
    int in_x = ow * stride + kx - padding;

    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
      int in_idx = ((b * in_channels + in_c) * in_height + in_y) * in_width + in_x;
      sum += input_data[in_idx] * w;
    }
  }

  // bias + ReLU
  if (bias_data && out_c < bias_size) sum += bias_data[out_c];
  if (relu && sum < 0.0f) sum = 0.0f;

  int out_idx = ((b * out_channels + out_c) * out_height + oh) * out_width + ow;
  output_data[out_idx] = sum;
}

void maxpool2d_batch_kernel(const float* __restrict__ input_data,
                            float* __restrict__ output_data,
                            int batch_size,
                            int channels,
                            int in_height,
                            int in_width,
                            int out_height,
                            int out_width,
                            int pool_size,
                            int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * channels * out_height * out_width;
  if (idx >= total) return;

  // decode (b, c, oh, ow)
  int ow = idx % out_width;
  int tmp = idx / out_width;
  int oh = tmp % out_height;
  tmp /= out_height;
  int c = tmp % channels;
  int b = tmp / channels;

  int h0 = oh * stride;
  int w0 = ow * stride;
  int h1 = h0 + pool_size < in_height ? h0 + pool_size : in_height;
  int w1 = w0 + pool_size < in_width ? w0 + pool_size : in_width;

  float maxv = -FLT_MAX;
  for (int y = h0; y < h1; ++y) {
    for (int x = w0; x < w1; ++x) {
      int in_idx = ((b * channels + c) * in_height + y) * in_width + x;
      float v = input_data[in_idx];
      if (v > maxv) maxv = v;
    }
  }
  int out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
  output_data[out_idx] = maxv;
}

void linear_csr_batch_kernel(const float* __restrict__ input_data,
                             int batch_size,
                             int input_features,
                             const float* __restrict__ weight_vals,
                             const int* __restrict__ weight_row_ptr,
                             const int* __restrict__ weight_col_idx,
                             const float* __restrict__ bias_data,
                             int out_neurons,
                             float* __restrict__ output_data) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * out_neurons;
  if (idx >= total) return;

  int of = idx % out_neurons;
  int b = idx / out_neurons;

  float sum = 0.0f;
  int start = weight_row_ptr[of];
  int end = weight_row_ptr[of + 1];

  for (int nz = start; nz < end; ++nz) {
    int col = weight_col_idx[nz];
    int in_idx = b * input_features + col;
    sum += input_data[in_idx] * weight_vals[nz];
  }
  sum += bias_data[of];  // assume bias_data != nullptr
  output_data[idx] = sum;
}

}  // namespace cifar_sparse::cuda
