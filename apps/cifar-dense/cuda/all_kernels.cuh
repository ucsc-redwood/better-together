#pragma once

#include "platform/engine/cuda/helpers.cuh"  // CheckCudaLaunch

namespace cifar_dense::cuda {

__global__ void conv2d_batch_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weights,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int N,
                                    int inC,
                                    int inH,
                                    int inW,
                                    int outC,
                                    int kH,
                                    int kW,
                                    int outH,
                                    int outW,
                                    int stride,
                                    int padding,
                                    bool relu);

__global__ void conv2d_batch_k3s1p1_kernel(const float* __restrict__ input,
                                           const float* __restrict__ weights,
                                           const float* __restrict__ bias,
                                           float* __restrict__ output,
                                           int N,
                                           int inC,
                                           int inH,
                                           int inW,
                                           int outC,
                                           int outH,
                                           int outW,
                                           bool relu);

// ---------------------------------------------------------
// 2) Host‐side launcher (Helper to make it easier to call)
// ---------------------------------------------------------
inline void conv2d_batch_cuda(const float* input,
                              const float* weights,
                              const float* bias,
                              float* output,
                              int N,
                              int inC,
                              int inH,
                              int inW,
                              int outC,
                              int kH,
                              int kW,
                              int outH,
                              int outW,
                              int stride,
                              int padding,
                              bool relu) {
  int total = N * outC * outH * outW;
  const int TPB = 256;
  int blocks = (total + TPB - 1) / TPB;

  if (kH == 3 && kW == 3 && stride == 1 && padding == 1) {
    conv2d_batch_k3s1p1_kernel<<<blocks, TPB>>>(
        input, weights, bias, output, N, inC, inH, inW, outC, outH, outW, relu);
    CheckCudaLaunch("conv2d_batch_k3s1p1_kernel");
    return;
  }

  conv2d_batch_kernel<<<blocks, TPB>>>(input,
                                       weights,
                                       bias,
                                       output,
                                       N,
                                       inC,
                                       inH,
                                       inW,
                                       outC,
                                       kH,
                                       kW,
                                       outH,
                                       outW,
                                       stride,
                                       padding,
                                       relu);
  CheckCudaLaunch("conv2d_batch_kernel");
}

__global__ void maxpool2d_batch_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int N,
                                       int C,
                                       int inH,
                                       int inW,
                                       int outH,
                                       int outW,
                                       int pool_size,
                                       int stride);

inline void maxpool2d_batch_cuda(const float* input,
                                 float* output,
                                 int N,
                                 int C,
                                 int inH,
                                 int inW,
                                 int outH,
                                 int outW,
                                 int pool_size,
                                 int stride) {
  int total = N * C * outH * outW;
  const int TPB = 256;
  int blocks = (total + TPB - 1) / TPB;

  maxpool2d_batch_kernel<<<blocks, TPB>>>(
      input, output, N, C, inH, inW, outH, outW, pool_size, stride);
  CheckCudaLaunch("maxpool2d_batch_kernel");
}

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
  // Warp-cooperative FC: 16 warps per block, 4 consecutive outputs per warp
  // (= 64 outputs per block) from ONE staged input row (16 KB at inF=4096).
  const int TPB = 512;
  const int outs_per_block = (TPB / 32) * 4;  // warps x kOutsPerWarp
  const dim3 blocks((outF + outs_per_block - 1) / outs_per_block, N);
  const size_t shmem = static_cast<size_t>(inF) * sizeof(float);

  linear_batch_kernel<<<blocks, TPB, shmem>>>(input, weights, bias, output, N, inF, outF, relu);
  CheckCudaLaunch("linear_batch_kernel");
}

// ---------------------------------------------------------------------------
// Tiled
// ---------------------------------------------------------------------------

constexpr int TILE_W = 16;
constexpr int TILE_H = 16;

__global__ void conv2d_tiled_shared(const float* __restrict__ input,
                                    const float* __restrict__ weights,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int N,
                                    int inC,
                                    int inH,
                                    int inW,
                                    int outC,
                                    int kH,
                                    int kW,
                                    int outH,
                                    int outW,
                                    int stride,
                                    int padding,
                                    bool relu);

inline void conv2d_tiled_cuda(const float* input,
                              const float* weights,
                              const float* bias,
                              float* output,
                              int N,
                              int inC,
                              int inH,
                              int inW,
                              int outC,
                              int kH,
                              int kW,
                              int outH,
                              int outW,
                              int stride,
                              int padding,
                              bool relu) {
  dim3 block(TILE_W, TILE_H);
  dim3 grid((outW + TILE_W - 1) / TILE_W, (outH + TILE_H - 1) / TILE_H, N * outC);

  // compute shared‐mem size
  int tile_in_w = TILE_W * stride + (kW - 1);
  int tile_in_h = TILE_H * stride + (kH - 1);
  size_t shmem_bytes = inC * tile_in_h * tile_in_w * sizeof(float);

  conv2d_tiled_shared<<<grid, block, shmem_bytes>>>(input,
                                                    weights,
                                                    bias,
                                                    output,
                                                    N,
                                                    inC,
                                                    inH,
                                                    inW,
                                                    outC,
                                                    kH,
                                                    kW,
                                                    outH,
                                                    outW,
                                                    stride,
                                                    padding,
                                                    relu);
  CheckCudaLaunch("conv2d_tiled_shared");
}

}  // namespace cifar_dense::cuda
