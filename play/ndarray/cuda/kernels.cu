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

  for (int r = 0; r < pool_h; ++r) {
    for (int s = 0; s < pool_w; ++s) {
      const int in_y = p * stride + r - padding;
      const int in_x = q * stride + s - padding;

      if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
        const float val = input[((n * C + c) * H + in_y) * W + in_x];
        max_val = fmaxf(max_val, val);
      }
    }
  }

  output[((n * C + c) * P + p) * Q + q] = max_val;
}

}  // namespace cuda