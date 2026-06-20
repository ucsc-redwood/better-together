#pragma once

#include <omp.h>

#include <algorithm>
#include <limits>

namespace cifar_dense::omp {

// ----------------------------------------------------------------------------
// Convolution 2D (Dense, Batched)
// ----------------------------------------------------------------------------

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
// output: (N, out_features)
inline void linear_batch_u(const float* __restrict__ u_input,
                           const float* __restrict__ u_weights,
                           const float* __restrict__ u_bias,
                           float* __restrict__ u_output,
                           const int N,            // in_shape[0]
                           const int in_features,  // in_shape[1]
                           const int out_features  // w_shape[0]
) {
  // Parallelize over (N, out_features); vectorize the per-output dot product.
#pragma omp for collapse(2)
  for (int n = 0; n < N; n++) {
    for (int of = 0; of < out_features; of++) {
      const float* __restrict__ in_row = u_input + n * in_features;
      const float* __restrict__ w_row = u_weights + of * in_features;
      float sum = u_bias[of];
#pragma omp simd reduction(+ : sum)
      for (int inf = 0; inf < in_features; inf++) {
        sum += in_row[inf] * w_row[inf];
      }
      u_output[n * (out_features) + of] = sum;
    }
  }
}

}  // namespace cifar_dense::omp
