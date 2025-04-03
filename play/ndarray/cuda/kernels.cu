#include <cfloat>

#include "kernels.cuh"

namespace cuda {

// ----------------------------------------------------------------
// Convolution 2d (Simple)
// ----------------------------------------------------------------

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

// ----------------------------------------------------------------
// Convolution 2d (Tiled)
// ----------------------------------------------------------------

// // Define tile dimensions and the tiling factor for channels
// constexpr int TILE_WIDTH = 16;
// constexpr int TILE_HEIGHT = 16;
// constexpr int T_C = 4;  // process T_C channels at a time
// Tiled convolution kernel using shared memory.
__global__ void conv2d_kernel_shared_tiled(const float* __restrict__ input,    // [N, C, H, W]
                                           const float* __restrict__ weights,  // [K, C, R, S]
                                           const float* __restrict__ bias,     // [K]
                                           float* output,                      // [N, K, P, Q]
                                           const int N,
                                           const int C,
                                           const int H,
                                           const int W,
                                           const int K,
                                           const int R,
                                           const int S,
                                           const int stride,
                                           const int padding,
                                           const bool apply_relu) {
  // Calculate output dimensions.
  const int P = (H + 2 * padding - R) / stride + 1;
  const int Q = (W + 2 * padding - S) / stride + 1;

  // Grid organization:
  //   gridDim.x: number of tiles along output width (Q)
  //   gridDim.y: number of tiles along output height (P)
  //   gridDim.z: combined batch and output channel; total blocks = N * K.
  //
  // Decode batch and output channel:
  const int n = blockIdx.z / K;
  const int k = blockIdx.z % K;

  // Tile (block) starting output coordinates.
  const int tile_col = blockIdx.x;
  const int tile_row = blockIdx.y;
  const int out_x0 = tile_col * TILE_WIDTH;
  const int out_y0 = tile_row * TILE_HEIGHT;

  // Thread indices within the tile.
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Global output coordinates (for this thread).
  const int out_x = out_x0 + tx;
  const int out_y = out_y0 + ty;

  // Shared memory tile dimensions:
  // For a TILE_WIDTH x TILE_HEIGHT block, the shared tile must cover
  // the region required by the convolution: a width of (TILE_WIDTH-1)*stride + S,
  // and a height of (TILE_HEIGHT-1)*stride + R.
  const int shared_width = (TILE_WIDTH - 1) * stride + S;
  const int shared_height = (TILE_HEIGHT - 1) * stride + R;

  // Declare dynamic shared memory.
  // Layout: for each channel in the current chunk (of size T_C), an array of
  // [shared_height x shared_width] floats.
  extern __shared__ float shared_input[];

  // Initialize accumulator. Only threads computing valid output positions do work.
  float sum = 0.0f;
  if (out_y < P && out_x < Q) {
    sum = bias[k];
  }

  // Loop over input channels in chunks. Here, since T_C == C for your case, this loop runs once.
  for (int c_base = 0; c_base < C; c_base += T_C) {
    const int curr_T_C = min(T_C, C - c_base);

    // Cooperative loading: each thread loads several elements from global memory
    // into shared memory for each channel in the current chunk.
    for (int c = 0; c < curr_T_C; ++c) {
      // Loop over the shared tile region.
      // Use the thread indices to cooperatively load the entire tile.
      for (int j = ty; j < shared_height; j += blockDim.y) {
        for (int i = tx; i < shared_width; i += blockDim.x) {
          // Compute corresponding input coordinates.
          // The top-left of the shared region corresponds to:
          //    in_y0 = out_y0 * stride - padding
          //    in_x0 = out_x0 * stride - padding
          int in_y = out_y0 * stride + j - padding;
          int in_x = out_x0 * stride + i - padding;
          float value = 0.0f;
          if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
            int input_index = ((n * C + (c_base + c)) * H + in_y) * W + in_x;
            value = input[input_index];
          }
          // Store into shared memory.
          shared_input[c * shared_width * shared_height + j * shared_width + i] = value;
        }
      }
    }
    __syncthreads();

    // Now each thread computes its convolution result using the shared memory tile.
    if (out_y < P && out_x < Q) {
      for (int c = 0; c < curr_T_C; ++c) {
        // For each position in the kernel.
        for (int r = 0; r < R; ++r) {
          for (int s = 0; s < S; ++s) {
            // The corresponding shared memory index is based on the thread's
            // output coordinate multiplied by stride plus the kernel offset.
            int sh_y = ty * stride + r;
            int sh_x = tx * stride + s;
            float in_val =
                shared_input[c * shared_width * shared_height + sh_y * shared_width + sh_x];
            int weight_index = (((k * C + (c_base + c)) * R) + r) * S + s;
            float wt = weights[weight_index];
            sum += in_val * wt;
          }
        }
      }
    }
    __syncthreads();
  }

  // Optionally apply ReLU.
  if (out_y < P && out_x < Q && apply_relu && sum < 0.0f) {
    sum = 0.0f;
  }

  // Write the computed output value to global memory if within output bounds.
  if (out_y < P && out_x < Q) {
    int output_index = ((n * K + k) * P + out_y) * Q + out_x;
    output[output_index] = sum;
  }
}

// ----------------------------------------------------------------
// Max Pooling 2d
// ----------------------------------------------------------------

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

// ----------------------------------------------------------------
// Linear (Fully Connected)
// ----------------------------------------------------------------

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