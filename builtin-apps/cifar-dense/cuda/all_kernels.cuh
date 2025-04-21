#pragma once

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
}

}  // namespace cifar_dense::cuda
