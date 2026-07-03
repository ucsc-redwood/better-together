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

// ---------------------------------------------------------------------------
// 2) maxpool2d_batch
//    One thread per output element
// ---------------------------------------------------------------------------
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


// 3x3 / stride-1 / pad-1 specialization (every conv in AlexNetCIFAR). The generic
// kernel is latency-bound: one accumulator dragging a serial FMA chain through
// inC*9 iterations with per-tap bounds checks. Here the 3x3 taps are fully
// unrolled with edge masks hoisted out (constant per thread), and two ic-slices
// accumulate independently to break the dependency chain.
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
                                           bool relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * outC * outH * outW;
  if (idx >= total) return;

  int ow = idx % outW;
  int tmp = idx / outW;
  int oh = tmp % outH;
  tmp /= outH;
  int oc = tmp % outC;
  int n = tmp / outC;

  // Edge masks, constant per thread (outH==inH, outW==inW for s1p1).
  const bool t = oh > 0, b = oh < inH - 1, l = ow > 0, r = ow < inW - 1;

  const float* in_n = input + static_cast<size_t>(n) * inC * inH * inW;
  const float* w_oc = weights + static_cast<size_t>(oc) * inC * 9;

  float s0 = 0.f, s1 = 0.f;
  const int hw = inH * inW;
  const float* ic_ptr = in_n + (oh - 1) * inW + (ow - 1);

  for (int ic = 0; ic < inC; ++ic) {
    const float* p0 = ic_ptr + ic * hw;            // row oh-1, col ow-1
    const float* p1 = p0 + inW;                    // row oh
    const float* p2 = p1 + inW;                    // row oh+1
    const float* w = w_oc + ic * 9;

    float a0 = 0.f, a1 = 0.f;
    if (t) {
      if (l) a0 += p0[0] * w[0];
      a1 += p0[1] * w[1];
      if (r) a0 += p0[2] * w[2];
    }
    if (l) a1 += p1[0] * w[3];
    a0 += p1[1] * w[4];
    if (r) a1 += p1[2] * w[5];
    if (b) {
      if (l) a0 += p2[0] * w[6];
      a1 += p2[1] * w[7];
      if (r) a0 += p2[2] * w[8];
    }
    s0 += a0;
    s1 += a1;
  }

  float v = bias[oc] + s0 + s1;
  if (relu && v < 0.f) v = 0.f;
  output[idx] = v;
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

  float acc[kMaxBatch];
#pragma unroll
  for (int n = 0; n < kMaxBatch; ++n) acc[n] = 0.f;

  const float4* w4 = reinterpret_cast<const float4*>(
      weights + static_cast<size_t>(of < outF ? of : 0) * inF);
  float4* sh4 = reinterpret_cast<float4*>(sh);

  for (int c = 0; c < inF; c += kChunk) {
    // Cooperative float4 staging of columns [c, c+kChunk) for all N images.
    const int n_f4 = (N * kChunk) >> 2;
    for (int i = threadIdx.x; i < n_f4; i += blockDim.x) {
      const int flat = i << 2;
      const int n = flat / kChunk;
      const int j = flat - n * kChunk;
      sh4[i] = *reinterpret_cast<const float4*>(&input[n * inF + c + j]);
    }
    __syncthreads();

    if (of < outF) {
      const int c4 = c >> 2;
      const int chunk4 = kChunk >> 2;
      for (int k = lane; k < chunk4; k += 32) {
        const float4 w = w4[c4 + k];
        const float4* col = sh4 + k;
        for (int n = 0; n < N; ++n) {
          const float4 x = col[n * chunk4];
          acc[n] += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
        }
      }
    }
    __syncthreads();
  }

  if (of >= outF) return;
  for (int n = 0; n < N; ++n) {
    float v = acc[n];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) {
      v += bias[of];
      if (relu && v < 0.f) v = 0.f;
      output[n * outF + of] = v;
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

// ---------------------------------------------------------------------------
// Tiled
// ---------------------------------------------------------------------------

// kernel: grid.z = N * outC, grid.x = ceil(outW / TILE_W), grid.y = ceil(outH / TILE_H)
// dynamic shared memory size = inC * (TILE_H*stride + kH-1) * (TILE_W*stride + kW-1) *
// sizeof(float)
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
                                    bool relu) {
  // decode which (n, oc) this block is doing
  int bic = blockIdx.z;  // [0 .. N*outC)
  int n = bic / outC;
  int oc = bic % outC;

  // 2D thread coords within tile
  int tx = threadIdx.x;  // [0 .. TILE_W)
  int ty = threadIdx.y;  // [0 .. TILE_H)

  // output pixel coords
  int out_x0 = blockIdx.x * TILE_W + tx;
  int out_y0 = blockIdx.y * TILE_H + ty;

  // compute shared‐mem tile dims
  int tile_in_w = TILE_W * stride + (kW - 1);
  int tile_in_h = TILE_H * stride + (kH - 1);

  // allocate shared memory: flattened [inC][tile_in_h][tile_in_w]
  extern __shared__ float shmem[];
  // pointer to channel‐0 base
  // offset per‐channel = tile_in_h * tile_in_w
  int patch_sz = tile_in_h * tile_in_w;
  // total = inC * patch_sz floats

  // 1) load input patch into shared memory
  for (int ic = 0; ic < inC; ++ic) {
    float* patch = shmem + ic * patch_sz;
    // each thread strides over the patch
    for (int y = ty; y < tile_in_h; y += blockDim.y) {
      for (int x = tx; x < tile_in_w; x += blockDim.x) {
        // global input coords for this element
        int in_y = blockIdx.y * TILE_H * stride - padding + y;
        int in_x = blockIdx.x * TILE_W * stride - padding + x;
        float v = 0.0f;
        if (in_y >= 0 && in_y < inH && in_x >= 0 && in_x < inW) {
          int idx = ((n * inC + ic) * inH + in_y) * inW + in_x;
          v = input[idx];
        }
        patch[y * tile_in_w + x] = v;
      }
    }
  }
  __syncthreads();

  // 2) if this thread’s output pixel is in‐bounds, do convolution
  if (out_x0 < outW && out_y0 < outH) {
    float sum = bias[oc];

    for (int ic = 0; ic < inC; ++ic) {
      const float* patch = shmem + ic * patch_sz;
      const float* wptr = weights + ((oc * inC + ic) * kH) * kW;
      // for each kernel element
      for (int ky = 0; ky < kH; ++ky) {
        for (int kx = 0; kx < kW; ++kx) {
          int sh_y = ty * stride + ky;
          int sh_x = tx * stride + kx;
          float iv = patch[sh_y * tile_in_w + sh_x];
          float wv = wptr[ky * kW + kx];
          sum += iv * wv;
        }
      }
    }
    if (relu && sum < 0.f) sum = 0.f;
    int out_idx = ((n * outC + oc) * outH + out_y0) * outW + out_x0;
    output[out_idx] = sum;
  }
}

}  // namespace cifar_dense::cuda
