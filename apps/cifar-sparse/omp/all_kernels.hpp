#pragma once

#include <omp.h>

#include <algorithm>
#include <limits>

namespace cifar_sparse::omp {

namespace detail {

// Register-blocked FC microkernel -- same 4 images x 2 output features tile as
// cifar-dense's detail::fc_tile4x2 (the FC head is dense math; only the convs
// are pruned). See apps/cifar-dense/omp/all_kernels.hpp for the full design
// rationale: the XNNPACK/MLAS mr x nr GEMM register-tile shape, vectorized
// along k (eight scalar `omp simd` reductions the compiler widens into eight
// vector accumulators) so the row-major weights need no packing pass. The
// reassociation is the same class the previous `omp simd reduction(+:sum)`
// dot product already had; loop order and tiling are fixed and there are no
// atomics, so results are deterministic for any thread count, and the float
// gates (NearEqual vs a double reference) stay the oracle contract.
inline void fc_tile4x2(const float* __restrict__ a_base,  // &input[n0 * inF]
                       const float* __restrict__ w_base,  // &weights[of0 * inF]
                       const float* __restrict__ bias,    // &bias[of0]
                       float* __restrict__ out_base,      // &output[n0 * outF + of0]
                       const int inF,
                       const int outF,
                       const bool relu) {
  const float* __restrict__ a0 = a_base;
  const float* __restrict__ a1 = a_base + inF;
  const float* __restrict__ a2 = a_base + 2 * inF;
  const float* __restrict__ a3 = a_base + 3 * inF;
  const float* __restrict__ w0 = w_base;
  const float* __restrict__ w1 = w_base + inF;
  float s00 = 0.0f, s01 = 0.0f, s10 = 0.0f, s11 = 0.0f;
  float s20 = 0.0f, s21 = 0.0f, s30 = 0.0f, s31 = 0.0f;
#pragma omp simd reduction(+ : s00, s01, s10, s11, s20, s21, s30, s31)
  for (int k = 0; k < inF; ++k) {
    const float x0 = a0[k], x1 = a1[k], x2 = a2[k], x3 = a3[k];
    const float y0 = w0[k], y1 = w1[k];
    s00 += x0 * y0;
    s01 += x0 * y1;
    s10 += x1 * y0;
    s11 += x1 * y1;
    s20 += x2 * y0;
    s21 += x2 * y1;
    s30 += x3 * y0;
    s31 += x3 * y1;
  }
  float r[4][2] = {{s00, s01}, {s10, s11}, {s20, s21}, {s30, s31}};
  for (int m = 0; m < 4; ++m) {
    for (int j = 0; j < 2; ++j) {
      float s = r[m][j] + bias[j];
      if (relu && s < 0.0f) s = 0.0f;
      out_base[m * outF + j] = s;
    }
  }
}

}  // namespace detail

// ----------------------------------------------------------------------------
// Convolution 2D (Sparse, Batched)
// ----------------------------------------------------------------------------

// Batched sparse convolution kernel using raw pointers only.
// Input layout: (batch, in_channels, in_height, in_width)
// Output layout: (batch, out_channels, out_height, out_width)
// The sparse weight matrix is given by its CSR components:
//   - weight_vals: nonzero weight values
//   - weight_row_ptr: row offsets for each output channel (length = out_channels + 1)
//   - weight_col_idx: column indices (flat kernel index) for nonzero values
// kernel parameters: kernel_size, stride, padding, and a flag for ReLU activation.
inline void conv2d_omp_batched(const float* __restrict__ input_data,
                               const int batch_size,
                               const int in_channels,
                               const int in_height,
                               const int in_width,
                               // Sparse weights for this convolution layer:
                               const float* __restrict__ weight_vals,
                               const int* __restrict__ weight_row_ptr,
                               const int* __restrict__ weight_col_idx,
                               const int out_channels,  // equals number of rows in CSR matrix
                               const float* __restrict__ bias_data,  // may be nullptr if unused
                               const int bias_size,                  // usually equals out_channels
                               const int kernel_size,
                               const int stride,
                               const int padding,
                               const bool relu,
                               float* __restrict__ output_data)  // preallocated output array
{
  // Compute spatial output dimensions.
  const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
  const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
  const int kernel_area = kernel_size * kernel_size;
  const int plane = out_height * out_width;

  // One thread owns the output plane for an (image, out_channel). Nonzeros are
  // looped on the OUTSIDE so each one's flat-index decode + in_y bounds check is
  // done once instead of once per output pixel, and the inner sweep vectorizes
  // across output columns (ow). For a fixed output pixel the nonzero terms are
  // still summed in CSR order, then bias, then ReLU -- numerically equivalent to
  // the per-pixel scalar version.
#pragma omp for schedule(static) collapse(2)
  for (int b = 0; b < batch_size; ++b) {
    for (int out_c = 0; out_c < out_channels; ++out_c) {
      const int row_start = weight_row_ptr[out_c];
      const int row_end = weight_row_ptr[out_c + 1];

      float* __restrict__ out_plane = output_data + (b * out_channels + out_c) * plane;
      for (int i = 0; i < plane; ++i) out_plane[i] = 0.0f;

      // Accumulate each nonzero weight across the whole output plane.
      for (int nz = row_start; nz < row_end; ++nz) {
        const int flat_kernel_idx = weight_col_idx[nz];
        const float weight_val = weight_vals[nz];
        const int in_c = flat_kernel_idx / kernel_area;
        const int rem = flat_kernel_idx % kernel_area;
        const int ky = rem / kernel_size;
        const int kx = rem % kernel_size;

        // ow range for which in_x = ow*stride + kx - padding lands in [0,in_width)
        const int lo_num = padding - kx;
        const int ow_lo = lo_num <= 0 ? 0 : (lo_num + stride - 1) / stride;
        const int hi_num = in_width - 1 + padding - kx;
        int ow_hi = hi_num < 0 ? 0 : hi_num / stride + 1;
        if (ow_hi > out_width) ow_hi = out_width;

        for (int oh = 0; oh < out_height; ++oh) {
          const int in_y = oh * stride + ky - padding;
          if (in_y < 0 || in_y >= in_height) continue;  // whole input row is padding
          const float* __restrict__ in_row =
              input_data + ((b * in_channels + in_c) * in_height + in_y) * in_width;
          float* __restrict__ out_row = out_plane + oh * out_width;

#pragma omp simd
          for (int ow = ow_lo; ow < ow_hi; ++ow) {
            out_row[ow] += weight_val * in_row[ow * stride + kx - padding];
          }
        }
      }  // end sparse weight loop

      // Add bias (if provided), then optional ReLU, over the whole plane.
      const float bval = (bias_data && out_c < bias_size) ? bias_data[out_c] : 0.0f;
#pragma omp simd
      for (int i = 0; i < plane; ++i) {
        float v = out_plane[i] + bval;
        if (relu && v < 0.0f) v = 0.0f;
        out_plane[i] = v;
      }
    }  // end out_c loop
  }  // end batch loop
}

// ----------------------------------------------------------------------------
// Max Pooling 2D (Sparse, Batched)
// ----------------------------------------------------------------------------

// A cleaner batched max pooling kernel that processes the full range of outputs.
// Input layout: (batch, channels, in_height, in_width)
// Output layout: (batch, channels, out_height, out_width)
inline void maxpool2d_omp_batched_clean(const float* __restrict__ input_data,
                                        const int batch_size,
                                        const int channels,
                                        const int in_height,
                                        const int in_width,
                                        const int pool_size,
                                        const int stride,
                                        float* __restrict__ output_data) {
  // Calculate output spatial dimensions.
  const int out_height = (in_height - pool_size) / stride + 1;
  const int out_width = (in_width - pool_size) / stride + 1;

// No-padding max pool: window [h_start,h_end) x [w_start,w_end), upper-clamped to
// the input. Same contract as the dense OMP/CUDA kernels and MaxPool2dRef (-inf
// init; the window is always in-bounds, so no per-element bounds branch is needed).
#pragma omp for collapse(3) schedule(static)
  for (int b = 0; b < batch_size; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int oh = 0; oh < out_height; ++oh) {
        const int h_start = oh * stride;
        const int h_end = std::min(h_start + pool_size, in_height);
        for (int ow = 0; ow < out_width; ++ow) {
          const int w_start = ow * stride;
          const int w_end = std::min(w_start + pool_size, in_width);

          float max_val = -std::numeric_limits<float>::infinity();
          for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
              const float val = input_data[((b * channels + c) * in_height + h) * in_width + w];
              if (val > max_val) max_val = val;
            }
          }
          output_data[((b * channels + c) * out_height + oh) * out_width + ow] = max_val;
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Linear Layer (Dense FC head, Batched)
// ----------------------------------------------------------------------------
// The AlexNetCIFAR FC head is dense (only the convs are pruned), so this is the
// same dense linear kernel as cifar-dense's.
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
  // Fast path: register-blocked GEMM (see detail::fc_tile4x2). Unlike
  // cifar-dense (N=16, DRAM-bound -> plain of-pair-major streaming), N=128
  // here makes this stage compute-bound and the 2 MB of activations exceed
  // L2 -- so of-pairs are grouped into NB-row cache blocks: a block's weight
  // panel (NB x 16 KB) is streamed from DRAM once (first image tile) and
  // stays L2-resident for the remaining N/MR - 1 image-tile passes (NB sized
  // for two SMT siblings sharing an L2), and each image tile's 4 input rows
  // are reused from cache across the block's of-pairs instead of re-pulling
  // all 2 MB of activations from L3 per of-pair. Guards are uniform across
  // threads, so the worksharing constructs stay consistent.
  constexpr int MR = 4;   // images per register tile
  constexpr int NR = 2;   // output features per register tile
  constexpr int NB = 16;  // weight rows per cache block (16 x 16 KB = 256 KB)
  if (N % MR == 0 && out_features % NR == 0) {
    const int n_blocks = (out_features + NB - 1) / NB;
#pragma omp for collapse(2) schedule(static)
    for (int ofb = 0; ofb < n_blocks; ofb++) {
      for (int n0 = 0; n0 < N; n0 += MR) {
        const int of_end = std::min((ofb + 1) * NB, out_features);
        for (int of0 = ofb * NB; of0 < of_end; of0 += NR) {
          detail::fc_tile4x2(u_input + n0 * in_features,
                             u_weights + of0 * in_features,
                             u_bias + of0,
                             u_output + n0 * out_features + of0,
                             in_features,
                             out_features,
                             relu);
        }
      }
    }
    return;
  }

  // Generic fallback (any shape): parallelize over (out_features, N);
  // vectorize the per-output dot product. of-major: a thread's contiguous
  // chunk of the collapse space visits each weight row N consecutive times
  // (L1-resident), so the (out x in) weight matrix streams from DRAM once per
  // task instead of once per image.
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

}  // namespace cifar_sparse::omp
