#include <cfloat>

#include "all_kernels.cuh"

namespace cifar_dense::cuda {

// ---------------------------------------------------------
// 1) The CUDA kernel: one thread per output element
// ---------------------------------------------------------
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
                                    bool relu) {
  // flatten thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * outC * outH * outW;
  if (idx >= total) return;

  // decode (n, oc, oh, ow)
  int ow = idx % outW;
  int tmp = idx / outW;
  int oh = tmp % outH;
  tmp /= outH;
  int oc = tmp % outC;
  int n = tmp / outC;

  // bias
  float sum = bias[oc];

  // convolution over inC × kH × kW
  for (int ic = 0; ic < inC; ++ic) {
    for (int kh = 0; kh < kH; ++kh) {
      for (int kw = 0; kw < kW; ++kw) {
        int ih = oh * stride - padding + kh;
        int iw = ow * stride - padding + kw;
        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
          int in_idx = ((n * inC + ic) * inH + ih) * inW + iw;
          int w_idx = ((oc * inC + ic) * kH + kh) * kW + kw;
          sum += input[in_idx] * weights[w_idx];
        }
      }
    }
  }

  if (relu && sum < 0.f) sum = 0.f;

  int out_idx = ((n * outC + oc) * outH + oh) * outW + ow;
  output[out_idx] = sum;
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
                                       int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * outH * outW;
  if (idx >= total) return;

  // decode
  int ow = idx % outW;
  int tmp = idx / outW;
  int oh = tmp % outH;
  tmp /= outH;
  int c = tmp % C;
  int n = tmp / C;

  int h_start = oh * stride;
  int w_start = ow * stride;
  int h_end = min(h_start + pool_size, inH);
  int w_end = min(w_start + pool_size, inW);

  float maxv = -FLT_MAX;
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      int in_idx = ((n * C + c) * inH + h) * inW + w;
      maxv = max(maxv, input[in_idx]);
    }
  }

  int out_idx = ((n * C + c) * outH + oh) * outW + ow;
  output[out_idx] = maxv;
}

// ---------------------------------------------------------------------------
// 2) linear_batch
//    One thread per (n, of)
// ---------------------------------------------------------------------------
__global__ void linear_batch_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weights,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int N,
                                    int inF,
                                    int outF) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * outF;
  if (idx >= total) return;

  int of = idx % outF;
  int n = idx / outF;

  float sum = bias[of];
  const float* in_row = input + n * inF;
  const float* w_row = weights + of * inF;
  for (int i = 0; i < inF; ++i) {
    sum += in_row[i] * w_row[i];
  }

  output[n * outF + of] = sum;
}

}  // namespace cifar_dense::cuda
