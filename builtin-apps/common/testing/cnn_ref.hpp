#pragma once
// ----------------------------------------------------------------------------
// Independent reference implementations of the CNN primitives (conv2d, maxpool,
// linear) for differential correctness testing of the cifar kernels.
//
// These are deliberately the simplest possible nested loops, written separately
// from builtin-apps/cifar-*/omp/all_kernels.hpp, and they ACCUMULATE IN DOUBLE.
// A backend kernel (OMP/CUDA/Vulkan, which accumulate in float) is correct iff
// its stage output matches this reference within a float tolerance. Because the
// reference is independent code, a mismatch points at a real kernel bug (wrong
// indexing, padding, stride, missing ReLU/bias, wrong dims) rather than shared
// rounding. Layout is NCHW, weights OIHW, row-major — matching the kernels.
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace bt::testing::cnn {

// Conv2d: out[n,oc,oh,ow] = bias[oc] + sum_{ic,kh,kw} in * w ; optional ReLU.
inline std::vector<float> Conv2dRef(const float* in, const float* w, const float* b,
                                    int N, int inC, int inH, int inW,
                                    int outC, int kH, int kW, int outH, int outW,
                                    int stride, int pad, bool relu) {
  std::vector<float> out(static_cast<std::size_t>(N) * outC * outH * outW);
  for (int n = 0; n < N; ++n) {
    for (int oc = 0; oc < outC; ++oc) {
      for (int oh = 0; oh < outH; ++oh) {
        for (int ow = 0; ow < outW; ++ow) {
          double sum = b[oc];
          for (int ic = 0; ic < inC; ++ic) {
            for (int kh = 0; kh < kH; ++kh) {
              for (int kw = 0; kw < kW; ++kw) {
                const int ih = oh * stride - pad + kh;
                const int iw = ow * stride - pad + kw;
                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                  sum += static_cast<double>(
                             in[((static_cast<std::size_t>(n) * inC + ic) * inH + ih) * inW + iw]) *
                         static_cast<double>(
                             w[((static_cast<std::size_t>(oc) * inC + ic) * kH + kh) * kW + kw]);
                }
              }
            }
          }
          if (relu && sum < 0) sum = 0;
          out[((static_cast<std::size_t>(n) * outC + oc) * outH + oh) * outW + ow] =
              static_cast<float>(sum);
        }
      }
    }
  }
  return out;
}

// MaxPool2d: window pool_size x pool_size, given stride; edge windows clamped.
inline std::vector<float> MaxPool2dRef(const float* in, int N, int C, int inH, int inW,
                                       int outH, int outW, int pool_size, int stride) {
  std::vector<float> out(static_cast<std::size_t>(N) * C * outH * outW);
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int oh = 0; oh < outH; ++oh) {
        for (int ow = 0; ow < outW; ++ow) {
          const int hs = oh * stride, ws = ow * stride;
          const int he = std::min(hs + pool_size, inH), we = std::min(ws + pool_size, inW);
          float maxv = -std::numeric_limits<float>::infinity();
          for (int h = hs; h < he; ++h)
            for (int wi = ws; wi < we; ++wi)
              maxv = std::max(maxv, in[((static_cast<std::size_t>(n) * C + c) * inH + h) * inW + wi]);
          out[((static_cast<std::size_t>(n) * C + c) * outH + oh) * outW + ow] = maxv;
        }
      }
    }
  }
  return out;
}

// Linear: out[n,of] = bias[of] + sum_inf in[n,inf] * w[of,inf]  (no activation).
inline std::vector<float> LinearRef(const float* in, const float* w, const float* b,
                                    int N, int in_features, int out_features) {
  std::vector<float> out(static_cast<std::size_t>(N) * out_features);
  for (int n = 0; n < N; ++n) {
    for (int of = 0; of < out_features; ++of) {
      double sum = b[of];
      for (int inf = 0; inf < in_features; ++inf) {
        sum += static_cast<double>(in[static_cast<std::size_t>(n) * in_features + inf]) *
               static_cast<double>(w[static_cast<std::size_t>(of) * in_features + inf]);
      }
      out[static_cast<std::size_t>(n) * out_features + of] = static_cast<float>(sum);
    }
  }
  return out;
}

}  // namespace bt::testing::cnn
