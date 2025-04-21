#pragma once

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
                             float* __restrict__ output_data);

inline void conv2d_csr_batch_cuda(const float* input_data,
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
                                  float* output_data) {
  int outH = (in_height + 2 * padding - kernel_size) / stride + 1;
  int outW = (in_width + 2 * padding - kernel_size) / stride + 1;
  int total = batch_size * out_channels * outH * outW;

  const int TPB = 256;
  int blocks = (total + TPB - 1) / TPB;

  conv2d_csr_batch_kernel<<<blocks, TPB>>>(input_data,
                                           batch_size,
                                           in_channels,
                                           in_height,
                                           in_width,
                                           weight_vals,
                                           weight_row_ptr,
                                           weight_col_idx,
                                           out_channels,
                                           bias_data,
                                           bias_size,
                                           kernel_size,
                                           stride,
                                           padding,
                                           relu,
                                           output_data);
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
                            int stride);

inline void maxpool2d_batch_cuda(const float* input_data,
                                 float* output_data,
                                 int batch_size,
                                 int channels,
                                 int in_height,
                                 int in_width,
                                 int out_height,
                                 int out_width,
                                 int pool_size,
                                 int stride) {
  int total = batch_size * channels * out_height * out_width;
  const int TPB = 256;
  int blocks = (total + TPB - 1) / TPB;

  maxpool2d_batch_kernel<<<blocks, TPB>>>(input_data,
                                          output_data,
                                          batch_size,
                                          channels,
                                          in_height,
                                          in_width,
                                          out_height,
                                          out_width,
                                          pool_size,
                                          stride);
}

void linear_csr_batch_kernel(const float* __restrict__ input_data,
                             int batch_size,
                             int input_features,
                             const float* __restrict__ weight_vals,
                             const int* __restrict__ weight_row_ptr,
                             const int* __restrict__ weight_col_idx,
                             const float* __restrict__ bias_data,
                             int out_neurons,
                             float* __restrict__ output_data);

inline void linear_csr_batch_cuda(const float* input_data,
                                  int batch_size,
                                  int input_features,
                                  const float* weight_vals,
                                  const int* weight_row_ptr,
                                  const int* weight_col_idx,
                                  const float* bias_data,
                                  int out_neurons,
                                  float* output_data) {
  int total = batch_size * out_neurons;
  const int TPB = 256;
  int blocks = (total + TPB - 1) / TPB;

  linear_csr_batch_kernel<<<blocks, TPB>>>(input_data,
                                           batch_size,
                                           input_features,
                                           weight_vals,
                                           weight_row_ptr,
                                           weight_col_idx,
                                           bias_data,
                                           out_neurons,
                                           output_data);
}

}  // namespace cifar_sparse::cuda
