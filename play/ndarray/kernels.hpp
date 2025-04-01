#pragma once

#include "ndarray.hpp"

namespace omp {

// Function declarations using NDArray buffers

// Convolution kernel
// input: (in_channels, in_height, in_width)
// weights: (out_channels, in_channels, kernel_height, kernel_width)
// bias: (out_channels)
// output: (out_channels, out_height, out_width)
inline void conv2d(const NDArray<3>& input,
                   const NDArray<4>& weights,
                   const NDArray<1>& bias,
                   const int stride,
                   const int padding,
                   const bool relu,
                   NDArray<3>& output) {
  const auto& input_shape = input.shape();
  const auto& weight_shape = weights.shape();
  const auto& output_shape = output.shape();

  int in_channels = static_cast<int>(input_shape[0]);
  int in_height = static_cast<int>(input_shape[1]);
  int in_width = static_cast<int>(input_shape[2]);

  int out_channels = static_cast<int>(weight_shape[0]);
  int kernel_h = static_cast<int>(weight_shape[2]);
  int kernel_w = static_cast<int>(weight_shape[3]);

  int out_height = static_cast<int>(output_shape[1]);
  int out_width = static_cast<int>(output_shape[2]);

// Loop over output channels and spatial dimensions in parallel
#pragma omp for collapse(3)
  for (int oc = 0; oc < out_channels; oc++) {
    for (int oh = 0; oh < out_height; oh++) {
      for (int ow = 0; ow < out_width; ow++) {
        float sum = bias(oc);
        // Loop over input channels and kernel spatial dimensions
        for (int ic = 0; ic < in_channels; ic++) {
          for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
              int ih = oh * stride - padding + kh;
              int iw = ow * stride - padding + kw;
              if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                sum += input(ic, ih, iw) * weights(oc, ic, kh, kw);
              }
            }
          }
        }
        if (relu && sum < 0) sum = 0;
        output(oc, oh, ow) = sum;
      }
    }
  }
}

// Max pooling kernel
// input: (channels, in_height, in_width)
// output: (channels, out_height, out_width)
inline void maxpool2d(const NDArray<3>& input,
                      const int pool_size,
                      const int stride,
                      NDArray<3>& output) {
  const auto& input_shape = input.shape();
  const auto& output_shape = output.shape();

  int channels = static_cast<int>(input_shape[0]);
  int in_height = static_cast<int>(input_shape[1]);
  int in_width = static_cast<int>(input_shape[2]);

  int out_height = static_cast<int>(output_shape[1]);
  int out_width = static_cast<int>(output_shape[2]);

#pragma omp for collapse(3)
  for (int c = 0; c < channels; c++) {
    for (int oh = 0; oh < out_height; oh++) {
      for (int ow = 0; ow < out_width; ow++) {
        int h_start = oh * stride;
        int w_start = ow * stride;
        int h_end = std::min(h_start + pool_size, in_height);
        int w_end = std::min(w_start + pool_size, in_width);
        float max_val = -std::numeric_limits<float>::infinity();
        for (int h = h_start; h < h_end; h++) {
          for (int w = w_start; w < w_end; w++) {
            max_val = std::max(max_val, input(c, h, w));
          }
        }
        output(c, oh, ow) = max_val;
      }
    }
  }
}

// Fully-connected (linear) layer kernel
// input: (input_features)
// weights: (output_features, input_features)
// bias: (output_features)
// output: (output_features)
inline void linear(const NDArray<1>& input,
                   const NDArray<2>& weights,
                   const NDArray<1>& bias,
                   NDArray<1>& output) {
  const auto& weight_shape = weights.shape();
  int output_features = static_cast<int>(weight_shape[0]);
  int input_features = static_cast<int>(weight_shape[1]);

#pragma omp for
  for (int i = 0; i < output_features; i++) {
    float sum = bias(i);
    for (int j = 0; j < input_features; j++) {
      sum += weights(i, j) * input(j);
    }
    output(i) = sum;
  }
}

}  // namespace omp
