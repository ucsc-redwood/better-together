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

// ---------------------------------------------------------------------------
// 3) linear_batch
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

// ---------------------------------------------------------------------------
// Tiled
// ---------------------------------------------------------------------------

// kernel: grid.z = N * outC, grid.x = ceil(outW / TILE_W), grid.y = ceil(outH / TILE_H)
// dynamic shared memory size = inC * (TILE_H*stride + kH-1) * (TILE_W*stride + kW-1) * sizeof(float)
__global__
void conv2d_tiled_shared(const float* __restrict__ input,
                         const float* __restrict__ weights,
                         const float* __restrict__ bias,
                         float* __restrict__ output,
                         int N, int inC, int inH, int inW,
                         int outC, int kH, int kW,
                         int outH, int outW,
                         int stride, int padding,
                         bool relu)
{
    // decode which (n, oc) this block is doing
    int bic = blockIdx.z;                 // [0 .. N*outC)
    int n  = bic / outC;
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
            const float* wptr  = weights + ((oc*inC + ic)*kH)*kW;
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
