#pragma once

#include <omp.h>

#include <algorithm>
#include <limits>

namespace cifar_dense::omp {

// ----------------------------------------------------------------------------
// Convolution 2D (Dense, Batched)
// ----------------------------------------------------------------------------

inline void conv2d_batch_u(const float* u_input,
                           const float* u_weights,
                           const float* u_bias,
                           float* u_output,
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
// Parallelize over (N, outC, outH, outW)
#pragma omp for collapse(4)
  for (int n = 0; n < N; n++) {
    for (int oc = 0; oc < outC; oc++) {
      for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
          float sum = u_bias[oc];  // start with bias for this out-channel
          // Accumulate over in_channels and kernel area
          for (int ic = 0; ic < inC; ic++) {
            for (int kh = 0; kh < kH; kh++) {
              for (int kw2 = 0; kw2 < kW; kw2++) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw2;
                // bounds check
                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                  // sum += u_input(n, ic, ih, iw) * u_weights(oc, ic, kh, kw2);
                  sum += u_input[n * (inC * inH * inW) + ic * (inH * inW) + ih * (inW) + iw] *
                         u_weights[oc * (inC * kH * kW) + ic * (kH * kW) + kh * (kW) + kw2];
                }
              }
            }
          }
          // Optional ReLU
          if (relu && sum < 0) sum = 0;
          u_output[n * (outC * outH * outW) + oc * (outH * outW) + oh * (outW) + ow] = sum;
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Max Pooling 2D (Dense, Batched)
// ----------------------------------------------------------------------------

inline void maxpool2d_batch_u(const float* u_input,
                              float* u_output,
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
// Linear Layer (Sparse, Batched)
// ----------------------------------------------------------------------------
// input:  (N, in_features)
// weights: (out_features, in_features)
// bias:   (out_features)
// output: (N, out_features)
inline void linear_batch_u(const float* u_input,
                           const float* u_weights,
                           const float* u_bias,
                           float* u_output,
                           const int N,            // in_shape[0]
                           const int in_features,  // in_shape[1]
                           const int out_features  // w_shape[0]
) {
  // Parallelize over (N, out_features)
#pragma omp for collapse(2)
  for (int n = 0; n < N; n++) {
    for (int of = 0; of < out_features; of++) {
      float sum = u_bias[of];
      for (int inf = 0; inf < in_features; inf++) {
        sum += u_input[n * (in_features) + inf] * u_weights[of * (in_features) + inf];
      }
      u_output[n * (out_features) + of] = sum;
    }
  }
}

}  // namespace cifar_dense::omp
