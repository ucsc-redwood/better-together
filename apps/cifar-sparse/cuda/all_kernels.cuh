#pragma once

#include "platform/engine/cuda/helpers.cuh"  // CheckCudaLaunch

namespace cifar_sparse::cuda {

__global__ void conv2d_csr_batch_k3s1p1_kernel(const float* __restrict__ input_data,
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
                                               bool relu,
                                               float* __restrict__ output_data);

__global__ void conv2d_csr_batch_kernel(const float* __restrict__ input_data,
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

  if (kernel_size == 3 && stride == 1 && padding == 1) {
    conv2d_csr_batch_k3s1p1_kernel<<<blocks, TPB>>>(input_data,
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
                                                    relu,
                                                    output_data);
    CheckCudaLaunch("conv2d_csr_batch_k3s1p1_kernel");
    return;
  }

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
  CheckCudaLaunch("conv2d_csr_batch_kernel");
}

__global__ void maxpool2d_batch_kernel(const float* __restrict__ input_data,
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
  CheckCudaLaunch("maxpool2d_batch_kernel");
}

// Dense linear for the FC head (only the convs are pruned) -- same kernel as
// cifar-dense's.
__global__ void linear_batch_bt_kernel(const float* __restrict__ input,
                                       const float* __restrict__ weights,
                                       const float* __restrict__ bias,
                                       float* __restrict__ output,
                                       int N,
                                       int inF,
                                       int outF,
                                       bool relu);

__global__ void linear_batch_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weights,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int N,
                                    int inF,
                                    int outF,
                                    bool relu);

inline void linear_batch_cuda(const float* input,
                              const float* weights,
                              const float* bias,
                              float* output,
                              int N,
                              int inF,
                              int outF,
                              bool relu) {
  if ((inF & 3) == 0 && (inF % 512) == 0) {
    // Batch-tiled path: one weight-row pass serves a 16-image tile (grid.y
    // covers larger batches, e.g. cifar-sparse at N=128).
    const int TPB = 512;
    const int warps_per_block = TPB / 32;
    const dim3 blocks((outF + warps_per_block - 1) / warps_per_block, (N + 15) / 16);
    const size_t shmem = static_cast<size_t>(16) * 512 * sizeof(float);
    linear_batch_bt_kernel<<<blocks, TPB, shmem>>>(
        input, weights, bias, output, N, inF, outF, relu);
    CheckCudaLaunch("linear_batch_bt_kernel");
    return;
  }

  // Fallback: warp-cooperative FC, 4 consecutive outputs per warp from one
  // staged input row (16 KB at inF=4096).
  const int TPB = 512;
  const int outs_per_block = (TPB / 32) * 4;  // warps x kOutsPerWarp
  const dim3 blocks((outF + outs_per_block - 1) / outs_per_block, N);
  const size_t shmem = static_cast<size_t>(inF) * sizeof(float);

  linear_batch_kernel<<<blocks, TPB, shmem>>>(input, weights, bias, output, N, inF, outF, relu);
  CheckCudaLaunch("linear_batch_kernel");
}

}  // namespace cifar_sparse::cuda
