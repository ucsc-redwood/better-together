#include <iostream>

#include "kernels.cuh"

namespace cuda {

__global__ void test_access(float* __restrict__ an_array) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= 10) return;
  an_array[tid] = tid;
}

__global__ void conv2d_batch_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weights,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int N,     // batch size
                                    int inC,   // input channels
                                    int inH,   // input height
                                    int inW,   // input width
                                    int outC,  // output channels
                                    int kH,    // kernel height
                                    int kW,    // kernel width
                                    int stride,
                                    int padding,
                                    bool relu,
                                    int outH,
                                    int outW) {
  if (threadIdx.x == 0) {
    printf("1111111111111111111111111111111\n");
  }

  // Each thread will map to one element in the output:
  //   (n, oc, oh, ow) in [0, N) x [0, outC) x [0, outH) x [0, outW)
  //
  // We'll linearize that 4D index into a single index: tid.
  // Then we recover (n, oc, oh, ow) by repeated modulo/div operations.
  //
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * outC * outH * outW;
  if (tid >= total) return;

  // Decompose linear index tid => (n, oc, oh, ow).
  int ow = tid % outW;
  int tmp = tid / outW;
  int oh = tmp % outH;
  tmp = tmp / outH;
  int oc = tmp % outC;
  tmp = tmp / outC;
  int n = tmp;  // batch index

  if (threadIdx.x == 0) {
    printf("2222222222222222222222222222222\n");
  }

  // Validation checks - make sure indices are within bounds
  if (n < 0 || n >= N || oc < 0 || oc >= outC || oh < 0 || oh >= outH || ow < 0 || ow >= outW) {
    return;  // Invalid indices, skip this thread
  }

  if (threadIdx.x == 0) {
    printf("3333333333333333333333333333333\n");
  }

  // Start with the bias for this output channel.
  float sum = bias[oc];

  if (threadIdx.x == 0) {
    printf("4444444444444444444444444444444\n");
  }

  // Compute the convolution sum for:
  //   sum_{ic, kh, kw} [ input(n, ic, ih, iw) * weights(oc, ic, kh, kw) ]
  for (int ic = 0; ic < inC; ic++) {
    for (int kh = 0; kh < kH; kh++) {
      for (int kw2 = 0; kw2 < kW; kw2++) {
        int ih = oh * stride - padding + kh;
        int iw = ow * stride - padding + kw2;

        // bounds check
        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
          // Calculate input and weight indices
          int input_index = (((n * inC + ic) * inH + ih) * inW + iw);
          int weights_index = (((oc * inC + ic) * kH + kh) * kW + kw2);

          // Extra bounds check on indices
          if (input_index >= 0 && input_index < N * inC * inH * inW && weights_index >= 0 &&
              weights_index < outC * inC * kH * kW) {
            sum += input[input_index] * weights[weights_index];
          }
        }
      }
    }
  }

  if (threadIdx.x == 0) {
    printf("5555555555555555555555555555555\n");
  }

  // Optional ReLU
  if (relu && sum < 0.0f) {
    sum = 0.0f;
  }

  // Write to output at index (n, oc, oh, ow).
  // Calculate the flat index directly to avoid potential errors
  int out_index = n;
  out_index = out_index * outC + oc;
  out_index = out_index * outH + oh;
  out_index = out_index * outW + ow;

  if (threadIdx.x == 0) {
    printf("6666666666666666666666666666666\n");
    printf("out_index: %d\n", out_index);
  }

  // Final bounds check before writing
  if (out_index >= 0 && out_index < N * outC * outH * outW) {
    output[out_index] = sum;
  }
}

// ----------------------------------------------------------------------------
// New
// ----------------------------------------------------------------------------

__global__ void k_conv2d_batch_u(const float* __restrict__ u_input,
                                 const float* __restrict__ u_weights,
                                 const float* __restrict__ u_bias,
                                 float* u_output,
                                 const int N,     // in_shape[0]
                                 const int inC,   // in_shape[1]
                                 const int inH,   // in_shape[2]
                                 const int inW,   // in_shape[3]
                                 const int outC,  // w_shape[0]
                                 const int kH,    // w_shape[2]
                                 const int kW,    // w_shape[3]
                                 const int outH,  // out_shape[2]
                                 const int outW,  // out_shape[3]
                                 const int stride,
                                 const int padding,
                                 const bool relu) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid != 0) return;

  for (int n = 0; n < N; n++) {
    for (int oc = 0; oc < outC; oc++) {
      for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
          printf("n: %d, oc: %d, oh: %d, ow: %d\n", n, oc, oh, ow);

          // float sum = u_bias[oc];  // start with bias for this out-channel
          // // Accumulate over in_channels and kernel area
          // for (int ic = 0; ic < inC; ic++) {
          //   for (int kh = 0; kh < kH; kh++) {
          //     for (int kw2 = 0; kw2 < kW; kw2++) {
          //       int ih = oh * stride - padding + kh;
          //       int iw = ow * stride - padding + kw2;
          //       // bounds check
          //       if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
          //         // sum += u_input(n, ic, ih, iw) * u_weights(oc, ic, kh, kw2);
          //         sum += u_input[n * (inC * inH * inW) + ic * (inH * inW) + ih * (inW) + iw];
          //       }
          //     }
          //   }
          // }
          // // Optional ReLU
          // if (relu && sum < 0) sum = 0;
          // u_output[n * (outC * outH * outW) + oc * (outH * outW) + oh * (outW) + ow] = sum;
        }
      }
    }
  }
}

// clang-format off
__global__ void conv2d_kernel(
    const float* __restrict__ input,     // [N, C, H, W]
    const float* __restrict__ weights,   // [K, C, R, S]
    const float* __restrict__ bias,      // [K]
    float* output,                       // [N, K, P, Q]
    int N, int C, int H, int W,          // input shape
    int K, int R, int S,                 // output channels and kernel shape
    int stride, int padding,
    bool apply_relu
) {
    int n = blockIdx.z;   // batch
    int k = blockIdx.y;   // output channel
    int pq = blockIdx.x * blockDim.x + threadIdx.x;
    
    int P = (H + 2 * padding - R) / stride + 1;
    int Q = (W + 2 * padding - S) / stride + 1;

    if (pq >= P * Q) return;
    int p = pq / Q;
    int q = pq % Q;

    float sum = bias[k];
    // float sum = 0.0f;


    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int in_y = p * stride + r - padding;
                int in_x = q * stride + s - padding;

                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    float in_val = input[((n * C + c) * H + in_y) * W + in_x];
                    // float wt = weights[((k * C + c) * R + r) * S + s];
                    float wt = 1.0f;
                    sum += in_val * wt;
                }
            }
        }
    }

    if (apply_relu && sum < 0) sum = 0.0f;
    output[((n * K + k) * P + p) * Q + q] = sum;
}
// clang-format on

}  // namespace cuda