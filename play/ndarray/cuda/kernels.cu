#include "kernels.cuh"

#include <iostream>

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
    return; // Invalid indices, skip this thread
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
          if (input_index >= 0 && input_index < N * inC * inH * inW &&
              weights_index >= 0 && weights_index < outC * inC * kH * kW) {
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

}  // namespace cuda