#pragma once

#include <omp.h>

#include <algorithm>
#include <limits>

namespace cifar_dense::omp {

// ----------------------------------------------------------------------------
// Convolution 2D (Dense, Batched)
// ----------------------------------------------------------------------------

namespace detail {

// Register-tiled 3x3 / stride-1 / pad-1 convolution. OUTW is the compile-time
// row width (== inW == outW), OCB the output-channel block: each (n, oc-block,
// oh) iteration accumulates OCB full output rows in a fixed-size local tile, so
// the inner FMA loops are branch-free, full-width, and the accumulators stay in
// registers across the whole (ic, kh) reduction instead of round-tripping
// through the output buffer on every term. One input row is loaded once and
// reused by all OCB output channels.
//
// Numerics: for every output element the terms are added in the exact
// (ic, kh, kw) order of the generic path below. The pad=1 columns contribute
// literal 0.0f terms (x + w*0.0f == x), so results are value-identical to the
// generic path; this preserves the OMP-oracle contract.
template <int OUTW, int OCB>
inline void conv3x3s1p1_batch(const float* __restrict__ u_input,
                              const float* __restrict__ u_weights,
                              const float* __restrict__ u_bias,
                              float* __restrict__ u_output,
                              const int N,
                              const int inC,
                              const int inH,   // == outH (square, same-padded)
                              const int outC,  // % OCB == 0
                              const bool relu) {
#pragma omp for collapse(3)
  for (int n = 0; n < N; n++) {
    for (int oc0 = 0; oc0 < outC; oc0 += OCB) {
      for (int oh = 0; oh < inH; oh++) {
        float acc[OCB][OUTW];
        for (int j = 0; j < OCB; j++) {
          for (int t = 0; t < OUTW; t++) acc[j][t] = u_bias[oc0 + j];
        }

        for (int ic = 0; ic < inC; ic++) {
          const float* __restrict__ plane = u_input + ((n * inC + ic) * inH) * OUTW;
          for (int kh = 0; kh < 3; kh++) {
            const int ih = oh - 1 + kh;
            if (ih < 0 || ih >= inH) continue;  // whole input row is padding
            const float* __restrict__ in_row = plane + ih * OUTW;

            // Input row with the pad=1 zeros materialized: row[t + kw] is the
            // input at iw = t - 1 + kw, 0.0f at both edges.
            float row[OUTW + 2];
            row[0] = 0.0f;
            row[OUTW + 1] = 0.0f;
#pragma omp simd
            for (int t = 0; t < OUTW; t++) row[t + 1] = in_row[t];

            float wk[OCB][3];
            for (int j = 0; j < OCB; j++) {
              const float* __restrict__ w =
                  u_weights + (((oc0 + j) * inC + ic) * 3 + kh) * 3;
              wk[j][0] = w[0];
              wk[j][1] = w[1];
              wk[j][2] = w[2];
            }

#pragma omp simd
            for (int t = 0; t < OUTW; t++) {
              const float x0 = row[t];
              const float x1 = row[t + 1];
              const float x2 = row[t + 2];
              for (int j = 0; j < OCB; j++) {
                acc[j][t] += wk[j][0] * x0;  // kw = 0
                acc[j][t] += wk[j][1] * x1;  // kw = 1
                acc[j][t] += wk[j][2] * x2;  // kw = 2
              }
            }
          }
        }

        for (int j = 0; j < OCB; j++) {
          float* __restrict__ out_row = u_output + ((n * outC + oc0 + j) * inH + oh) * OUTW;
          if (relu) {
#pragma omp simd
            for (int t = 0; t < OUTW; t++) out_row[t] = acc[j][t] < 0.0f ? 0.0f : acc[j][t];
          } else {
#pragma omp simd
            for (int t = 0; t < OUTW; t++) out_row[t] = acc[j][t];
          }
        }
      }
    }
  }
}

}  // namespace detail

inline void conv2d_batch_u(const float* __restrict__ u_input,
                           const float* __restrict__ u_weights,
                           const float* __restrict__ u_bias,
                           float* __restrict__ u_output,
                           const int N,        // in_shape[0]
                           const int inC,      // in_shape[1]
                           const int inH,      // in_shape[2]
                           const int inW,      // in_shape[3]
                           const int outC,     // w_shape[0]
                           const int kH,       // w_shape[2]
                           const int kW,       // w_shape[3]
                           const int outH,     // out_shape[2]
                           const int outW,     // out_shape[3]
                           const int stride,   // 1
                           const int padding,  // 0
                           const bool relu) {
  // Shape-specialized fast path: every conv in AlexNetCIFAR is 3x3/s1/p1 on a
  // square feature map of width 8/16/32. All threads in the parallel region see
  // the same dims, so every thread takes the same branch and the worksharing
  // constructs stay consistent. (OCB=2 for width 32: 4x32 accumulators would
  // exceed the vector register file.)
  if (stride == 1 && padding == 1 && kH == 3 && kW == 3 && outH == inH && outW == inW) {
    if (outW == 8 && outC % 4 == 0) {
      detail::conv3x3s1p1_batch<8, 4>(
          u_input, u_weights, u_bias, u_output, N, inC, inH, outC, relu);
      return;
    }
    if (outW == 16 && outC % 4 == 0) {
      detail::conv3x3s1p1_batch<16, 4>(
          u_input, u_weights, u_bias, u_output, N, inC, inH, outC, relu);
      return;
    }
    if (outW == 32 && outC % 2 == 0) {
      detail::conv3x3s1p1_batch<32, 2>(
          u_input, u_weights, u_bias, u_output, N, inC, inH, outC, relu);
      return;
    }
  }

  // Each thread owns one full output row out[n][oc][oh][:]. The accumulation
  // runs (ic, kh, kw) on the outside and vectorizes across output columns (ow),
  // which are independent accumulators. For a fixed ow, the (ic,kh,kw) terms are
  // added in the same order as the scalar version, so the result is numerically
  // equivalent -- the speedup is from hoisting the per-multiply-add padding
  // bounds check out of the innermost loop (it was branching every MAC and
  // blocking auto-vectorization), not from reassociating the sum.
#pragma omp for collapse(3)
  for (int n = 0; n < N; n++) {
    for (int oc = 0; oc < outC; oc++) {
      for (int oh = 0; oh < outH; oh++) {
        float* __restrict__ out_row = u_output + ((n * outC + oc) * outH + oh) * outW;
        for (int ow = 0; ow < outW; ow++) out_row[ow] = u_bias[oc];

        for (int ic = 0; ic < inC; ic++) {
          for (int kh = 0; kh < kH; kh++) {
            const int ih = oh * stride - padding + kh;
            if (ih < 0 || ih >= inH) continue;  // whole input row is padding
            const float* __restrict__ in_row = u_input + ((n * inC + ic) * inH + ih) * inW;

            for (int kw = 0; kw < kW; kw++) {
              const float w = u_weights[((oc * inC + ic) * kH + kh) * kW + kw];
              // ow range for which iw = ow*stride - padding + kw lands in [0,inW)
              const int lo_num = padding - kw;
              const int ow_lo = lo_num <= 0 ? 0 : (lo_num + stride - 1) / stride;
              const int hi_num = inW - 1 + padding - kw;
              int ow_hi = hi_num < 0 ? 0 : hi_num / stride + 1;
              if (ow_hi > outW) ow_hi = outW;

#pragma omp simd
              for (int ow = ow_lo; ow < ow_hi; ow++) {
                out_row[ow] += w * in_row[ow * stride - padding + kw];
              }
            }
          }
        }

        if (relu) {
#pragma omp simd
          for (int ow = 0; ow < outW; ow++) {
            if (out_row[ow] < 0.0f) out_row[ow] = 0.0f;
          }
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Max Pooling 2D (Dense, Batched)
// ----------------------------------------------------------------------------

inline void maxpool2d_batch_u(const float* __restrict__ u_input,
                              float* __restrict__ u_output,
                              const int N,     // in_shape[0]
                              const int C,     // in_shape[1]
                              const int inH,   // in_shape[2]
                              const int inW,   // in_shape[3]
                              const int outH,  // out_shape[2]
                              const int outW,  // out_shape[3]
                              const int pool_size,
                              const int stride) {
  // Parallelize over (N, C, outH, outW)
#pragma omp for collapse(4)
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
          int h_start = oh * stride;
          int w_start = ow * stride;
          int h_end = std::min(h_start + pool_size, inH);
          int w_end = std::min(w_start + pool_size, inW);

          float max_val = -std::numeric_limits<float>::infinity();
          for (int h = h_start; h < h_end; h++) {
            for (int w = w_start; w < w_end; w++) {
              // float val = input(n, c, h, w);
              float val = u_input[n * (C * inH * inW) + c * (inH * inW) + h * (inW) + w];
              if (val > max_val) {
                max_val = val;
              }
            }
          }
          u_output[n * (C * outH * outW) + c * (outH * outW) + oh * (outW) + ow] = max_val;
          // output(n, c, oh, ow) = max_val;
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Linear Layer (Dense, Batched)
// ----------------------------------------------------------------------------
// input:  (N, in_features)
// weights: (out_features, in_features)
// bias:   (out_features)
// output: (N, out_features), optional ReLU
inline void linear_batch_u(const float* __restrict__ u_input,
                           const float* __restrict__ u_weights,
                           const float* __restrict__ u_bias,
                           float* __restrict__ u_output,
                           const int N,             // in_shape[0]
                           const int in_features,   // in_shape[1]
                           const int out_features,  // w_shape[0]
                           const bool relu) {
  // Parallelize over (out_features, N); vectorize the per-output dot product.
  // of-major: a thread's contiguous chunk of the collapse space visits each
  // weight row N consecutive times (L1-resident), so the (out x in) weight
  // matrix -- 64 MB for the 4096-wide FCs -- streams from DRAM once per task
  // instead of once per image. Each (n, of) dot product is unchanged.
#pragma omp for collapse(2)
  for (int of = 0; of < out_features; of++) {
    for (int n = 0; n < N; n++) {
      const float* __restrict__ in_row = u_input + n * in_features;
      const float* __restrict__ w_row = u_weights + of * in_features;
      float sum = u_bias[of];
#pragma omp simd reduction(+ : sum)
      for (int inf = 0; inf < in_features; inf++) {
        sum += in_row[inf] * w_row[inf];
      }
      if (relu) sum = std::max(sum, 0.0f);
      u_output[n * (out_features) + of] = sum;
    }
  }
}

}  // namespace cifar_dense::omp
