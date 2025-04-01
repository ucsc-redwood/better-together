#include <cfloat>

#include "kernels.cuh"

namespace cuda {

__global__ void conv2d_kernel(const float* __restrict__ input,    // [N, C, H, W]
                              const float* __restrict__ weights,  // [K, C, R, S]
                              const float* __restrict__ bias,     // [K]
                              float* output,                      // [N, K, P, Q]
                              const int N,
                              const int C,
                              const int H,
                              const int W,  // input shape
                              const int K,
                              const int R,
                              const int S,  // output channels and kernel shape
                              const int stride,
                              const int padding,
                              const bool apply_relu) {
  const int n = blockIdx.z;  // batch
  const int k = blockIdx.y;  // output channel
  const int pq = blockIdx.x * blockDim.x + threadIdx.x;

  const int P = (H + 2 * padding - R) / stride + 1;
  const int Q = (W + 2 * padding - S) / stride + 1;

  if (pq >= P * Q) return;
  const int p = pq / Q;
  const int q = pq % Q;

  float sum = bias[k];
  // float sum = 0.0f;

  for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        const int in_y = p * stride + r - padding;
        const int in_x = q * stride + s - padding;

        if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
          const float in_val = input[((n * C + c) * H + in_y) * W + in_x];
          const float wt = weights[((k * C + c) * R + r) * S + s];
          // float wt = 1.0f;
          sum += in_val * wt;
        }
      }
    }
  }

  if (apply_relu && sum < 0) sum = 0.0f;
  output[((n * K + k) * P + p) * Q + q] = sum;
}

__global__ void maxpool2d_kernel(const float* __restrict__ input,  // [N, C, H, W]
                                 float* output,                    // [N, C, P, Q]
                                 const int N,
                                 const int C,
                                 const int H,
                                 const int W,  // input shape
                                 const int pool_h,
                                 const int pool_w,  // pooling kernel size
                                 const int stride,
                                 const int padding) {
  const int n = blockIdx.z;  // batch
  const int c = blockIdx.y;  // channel
  const int pq = blockIdx.x * blockDim.x + threadIdx.x;

  const int P = (H + 2 * padding - pool_h) / stride + 1;
  const int Q = (W + 2 * padding - pool_w) / stride + 1;

  if (pq >= P * Q) return;

  const int p = pq / Q;
  const int q = pq % Q;

  float max_val = -FLT_MAX;
  bool found_valid_input = false;

  for (int r = 0; r < pool_h; ++r) {
    for (int s = 0; s < pool_w; ++s) {
      const int in_y = p * stride + r - padding;
      const int in_x = q * stride + s - padding;

      if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
        const float val = input[((n * C + c) * H + in_y) * W + in_x];
        max_val = fmaxf(max_val, val);
        found_valid_input = true;
      }
    }
  }

  // If no valid inputs were found in the pooling region, set max_val to 0
  if (!found_valid_input) {
    max_val = 0.0f;
  }

  output[((n * C + c) * P + p) * Q + q] = max_val;
}

__global__ void linear_kernel(const float* __restrict__ input,    // [N, in_features]
                              const float* __restrict__ weights,  // [out_features, in_features]
                              const float* __restrict__ bias,     // [out_features]
                              float* output,                      // [N, out_features]
                              const int N,
                              const int in_features,
                              const int out_features) {
  // Calculate global thread index
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Compute (batch, output_feature) indices from the global index
  const int n = idx / out_features;   // batch index
  const int of = idx % out_features;  // output feature index

  // Check if this thread should compute something
  if (n >= N) return;

  // For this (n, of) pair, compute the dot product and add bias
  float sum = bias[of];

  // Multiply input with weights and accumulate
  for (int inf = 0; inf < in_features; ++inf) {
    sum += input[n * in_features + inf] * weights[of * in_features + inf];
  }

  // Store the result
  output[n * out_features + of] = sum;
}

}  // namespace cuda