#include <cfloat>

#include "all_kernels.cuh"

namespace cifar_sparse::cuda {

// 3x3 / stride-1 / pad-1 CSR specialization (every pruned conv in the model).
// The generic kernel drags one FMA chain through the whole CSR row with a
// runtime `/ area` + `% area` per nonzero; with the literal 9 the compiler
// strength-reduces the decode, and two interleaved accumulators break the
// serial dependency (same medicine as the dense k3s1p1 kernel).
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
                                               float* __restrict__ output_data) {
  const int out_height = in_height;
  const int out_width = in_width;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * out_channels * out_height * out_width;
  if (idx >= total) return;

  int ow = idx % out_width;
  int tmp = idx / out_width;
  int oh = tmp % out_height;
  tmp /= out_height;
  int out_c = tmp % out_channels;
  int b = tmp / out_channels;

  const int row_start = weight_row_ptr[out_c];
  const int row_end = weight_row_ptr[out_c + 1];
  const float* in_b = input_data + static_cast<size_t>(b) * in_channels * in_height * in_width;
  const int hw = in_height * in_width;

  auto tap = [&](int nz) -> float {
    const int flat_k = weight_col_idx[nz];
    const int in_c = flat_k / 9;
    const int rem = flat_k - in_c * 9;
    const int ky = rem / 3;
    const int kx = rem - ky * 3;
    const int in_y = oh + ky - 1;
    const int in_x = ow + kx - 1;
    if (in_y < 0 || in_y >= in_height || in_x < 0 || in_x >= in_width) return 0.f;
    return in_b[in_c * hw + in_y * in_width + in_x] * weight_vals[nz];
  };

  float s0 = 0.f, s1 = 0.f;
  int nz = row_start;
  for (; nz + 1 < row_end; nz += 2) {
    s0 += tap(nz);
    s1 += tap(nz + 1);
  }
  if (nz < row_end) s0 += tap(nz);

  float sum = s0 + s1;
  if (bias_data && out_c < bias_size) sum += bias_data[out_c];
  if (relu && sum < 0.0f) sum = 0.0f;

  int out_idx = ((b * out_channels + out_c) * out_height + oh) * out_width + ow;
  output_data[out_idx] = sum;
}

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
                                        float* __restrict__ output_data) {
  // recompute output dims
  int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
  int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * out_channels * out_height * out_width;
  if (idx >= total) return;

  // decode (b, out_c, oh, ow)
  int ow = idx % out_width;
  int tmp = idx / out_width;
  int oh = tmp % out_height;
  tmp /= out_height;
  int out_c = tmp % out_channels;
  int b = tmp / out_channels;

  float sum = 0.0f;
  int row_start = weight_row_ptr[out_c];
  int row_end = weight_row_ptr[out_c + 1];
  int area = kernel_size * kernel_size;

  // loop over nonzeros in this output channel's CSR row
  for (int nz = row_start; nz < row_end; ++nz) {
    int flat_k = weight_col_idx[nz];
    float w = weight_vals[nz];

    int in_c = flat_k / area;
    int rem = flat_k % area;
    int ky = rem / kernel_size;
    int kx = rem % kernel_size;

    int in_y = oh * stride + ky - padding;
    int in_x = ow * stride + kx - padding;

    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
      int in_idx = ((b * in_channels + in_c) * in_height + in_y) * in_width + in_x;
      sum += input_data[in_idx] * w;
    }
  }

  // bias + ReLU
  if (bias_data && out_c < bias_size) sum += bias_data[out_c];
  if (relu && sum < 0.0f) sum = 0.0f;

  int out_idx = ((b * out_channels + out_c) * out_height + oh) * out_width + ow;
  output_data[out_idx] = sum;
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
                                       int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * channels * out_height * out_width;
  if (idx >= total) return;

  // decode (b, c, oh, ow)
  int ow = idx % out_width;
  int tmp = idx / out_width;
  int oh = tmp % out_height;
  tmp /= out_height;
  int c = tmp % channels;
  int b = tmp / channels;

  int h0 = oh * stride;
  int w0 = ow * stride;
  int h1 = h0 + pool_size < in_height ? h0 + pool_size : in_height;
  int w1 = w0 + pool_size < in_width ? w0 + pool_size : in_width;

  float maxv = -FLT_MAX;
  for (int y = h0; y < h1; ++y) {
    for (int x = w0; x < w1; ++x) {
      int in_idx = ((b * channels + c) * in_height + y) * in_width + x;
      float v = input_data[in_idx];
      if (v > maxv) maxv = v;
    }
  }
  int out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
  output_data[out_idx] = maxv;
}

// Batch-tiled FC (N <= kFcMaxBatch, inF % 4 == 0). The warp-per-(n, of) kernel
// above re-streams every 16 KB weight row once PER IMAGE (67 MB x 16 = 1.07 GB
// at fc1) and sits at the DRAM ceiling (~94 GB/s -> 11.4 ms). Here a warp owns
// one output feature and accumulates ALL N images while streaming its weight
// row ONCE; input columns are staged chunk-by-chunk in shared memory and shared
// by the whole block. Weight traffic drops N-fold.
// Launch: TPB = 512 (16 warps = 16 outputs/block); grid.x = ceil(outF/16);
// dynamic shared = N * kFcChunk * 4 bytes.
__global__ void linear_batch_bt_kernel(const float* __restrict__ input,
                                       const float* __restrict__ weights,
                                       const float* __restrict__ bias,
                                       float* __restrict__ output,
                                       int N,
                                       int inF,
                                       int outF,
                                       bool relu) {
  extern __shared__ float sh[];  // [N][kFcChunk]
  constexpr int kChunk = 512;
  constexpr int kMaxBatch = 16;

  const int warps_per_block = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int of = blockIdx.x * warps_per_block + warp;

  // grid.y tiles the batch in groups of kMaxBatch (cifar-sparse runs N=128).
  const int n0 = blockIdx.y * kMaxBatch;
  const int nb = (N - n0) < kMaxBatch ? (N - n0) : kMaxBatch;

  float acc[kMaxBatch];
#pragma unroll
  for (int n = 0; n < kMaxBatch; ++n) acc[n] = 0.f;

  const float4* w4 = reinterpret_cast<const float4*>(
      weights + static_cast<size_t>(of < outF ? of : 0) * inF);
  float4* sh4 = reinterpret_cast<float4*>(sh);

  for (int c = 0; c < inF; c += kChunk) {
    // Cooperative float4 staging of columns [c, c+kChunk) for all N images.
    const int n_f4 = (nb * kChunk) >> 2;
    for (int i = threadIdx.x; i < n_f4; i += blockDim.x) {
      const int flat = i << 2;
      const int n = flat / kChunk;
      const int j = flat - n * kChunk;
      sh4[i] = *reinterpret_cast<const float4*>(&input[(n0 + n) * inF + c + j]);
    }
    __syncthreads();

    if (of < outF) {
      const int c4 = c >> 2;
      const int chunk4 = kChunk >> 2;
      for (int k = lane; k < chunk4; k += 32) {
        const float4 w = w4[c4 + k];
        const float4* col = sh4 + k;
        for (int n = 0; n < nb; ++n) {
          const float4 x = col[n * chunk4];
          acc[n] += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
        }
      }
    }
    __syncthreads();
  }

  if (of >= outF) return;
  for (int n = 0; n < nb; ++n) {
    float v = acc[n];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) {
      v += bias[of];
      if (relu && v < 0.f) v = 0.f;
      output[(n0 + n) * outF + of] = v;
    }
  }
}


// ---------------------------------------------------------------------------
// 3) linear_batch
//    Warp-per-(n, of). The old thread-per-output kernel had each lane of a warp
//    scanning a DIFFERENT 16 KB weight row, so every 128-byte transaction served
//    one 4-byte float (measured 2.6% bandwidth utilization on Orin). Here the 32
//    lanes stride one row together (fully coalesced) and reduce via shuffles; the
//    input row is staged in shared memory once per block.
//    Launch: blockDim.x = multiple of 32; grid = (ceil(outF/warps_per_block), N);
//    dynamic shared memory = inF * sizeof(float).
// ---------------------------------------------------------------------------
__global__ void linear_batch_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weights,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int N,
                                    int inF,
                                    int outF,
                                    bool relu) {
  extern __shared__ float sh_in[];
  const int n = blockIdx.y;

  for (int i = threadIdx.x; i < inF; i += blockDim.x) {
    sh_in[i] = input[n * inF + i];
  }
  __syncthreads();

  const int warps_per_block = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  // Each warp computes kOutsPerWarp consecutive outputs from the SAME staged
  // input row: with one output per warp, 16 K blocks each re-loaded the 16 KB
  // row (268 MB of redundant input traffic vs 67 MB of weights).
  constexpr int kOutsPerWarp = 4;
  const int of0 = (blockIdx.x * warps_per_block + warp) * kOutsPerWarp;
  for (int of = of0; of < of0 + kOutsPerWarp && of < outF; ++of) {

  const float* w_row = weights + static_cast<size_t>(of) * inF;
  float sum = 0.f;
  if ((inF & 3) == 0) {
    // float4 loads + two independent accumulators: 4x fewer load instructions and
    // a split FMA chain (the scalar loop was latency-bound, ~5.7 GB/s).
    const float4* w4 = reinterpret_cast<const float4*>(w_row);
    const float4* i4 = reinterpret_cast<const float4*>(sh_in);
    const int n4 = inF >> 2;
    float acc0 = 0.f, acc1 = 0.f;
    for (int k = lane; k < n4; k += 32) {
      const float4 w = w4[k];
      const float4 x = i4[k];
      acc0 += w.x * x.x + w.y * x.y;
      acc1 += w.z * x.z + w.w * x.w;
    }
    sum = acc0 + acc1;
  } else {
    for (int k = lane; k < inF; k += 32) {
      sum += sh_in[k] * w_row[k];
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, off);
  }

  if (lane == 0) {
    float v = sum + bias[of];
    if (relu && v < 0.f) v = 0.f;
    output[n * outF + of] = v;
  }
  }  // for of
}


}  // namespace cifar_sparse::cuda
