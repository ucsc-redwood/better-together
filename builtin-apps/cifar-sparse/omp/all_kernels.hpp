#pragma once

#include "../sparse_appdata.hpp"

namespace cifar_sparse::omp {

// How many images to process per iteration together
constexpr auto kNumBatches = 16;

// ----------------------------------------------------------------------------
// Convolution 2D (Sparse)
// ----------------------------------------------------------------------------

// Input Image dimensions
constexpr int kInputChannels = 3;
constexpr int kInputHeight = 32;
constexpr int kInputWidth = 32;

// Convolution parameters
constexpr int kKernelSize = 3;
constexpr int kStride = 1;
constexpr int kPadding = 1;

// Pooling parameters
constexpr int kPoolSize = 2;
constexpr int kPoolStride = 2;

constexpr bool kRelu = true;

// Function declarations
void conv2d_omp(const float *input_data,
                int image_input_channels,
                int input_height,
                int input_width,
                const CSRMatrix &weight_matrix,
                const float *bias_data,
                int bias_size,
                int kernel_size,
                int stride,
                int padding,
                bool relu,
                float *output_data,
                int start,
                int end);

void maxpool2d_omp(const float *input_data,
                   int input_channels,
                   int input_height,
                   int input_width,
                   int pool_size,
                   int stride,
                   float *output_data,
                   int start,
                   int end);

void linear_omp(const float *input_data,
                const CSRMatrix &weight_matrix,
                const float *bias_data,
                float *output_data,
                int start,
                int end);

namespace v2 {

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
inline void conv2d_omp_batched(const float *input_data,
                               const int batch_size,
                               const int in_channels,
                               const int in_height,
                               const int in_width,
                               // Sparse weights for this convolution layer:
                               const float *weight_vals,
                               const int *weight_row_ptr,
                               const int *weight_col_idx,
                               const int out_channels,  // equals number of rows in CSR matrix
                               const float *bias_data,  // may be nullptr if no bias is used
                               const int bias_size,     // usually equals out_channels
                               const int kernel_size,
                               const int stride,
                               const int padding,
                               const bool relu,
                               float *output_data)  // preallocated output array
{
  // Compute spatial output dimensions.
  const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
  const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

// Use collapse on the batch and output channel loops for parallelism.
#pragma omp for
  for (int b = 0; b < batch_size; ++b) {
    for (int out_c = 0; out_c < out_channels; ++out_c) {
      // Get the CSR index range for the current output channel.
      int row_start = weight_row_ptr[out_c];
      int row_end = weight_row_ptr[out_c + 1];

      // Iterate over the spatial positions of the output feature map.
      for (int oh = 0; oh < out_height; ++oh) {
        for (int ow = 0; ow < out_width; ++ow) {
          float sum = 0.0f;

          // Loop over the nonzero sparse weights for this output channel.
          for (int nz = row_start; nz < row_end; ++nz) {
            int flat_kernel_idx = weight_col_idx[nz];
            float weight_val = weight_vals[nz];

            // Decode the flat kernel index:
            //   in_channel = flat_idx / (kernel_size * kernel_size)
            //   kernel_y = (flat_idx % (kernel_size * kernel_size)) / kernel_size
            //   kernel_x = (flat_idx % (kernel_size * kernel_size)) % kernel_size
            const int kernel_area = kernel_size * kernel_size;
            int in_c = flat_kernel_idx / kernel_area;
            int rem = flat_kernel_idx % kernel_area;
            int ky = rem / kernel_size;
            int kx = rem % kernel_size;

            // Compute corresponding input spatial coordinates.
            int in_y = oh * stride + ky - padding;
            int in_x = ow * stride + kx - padding;

            // Check for valid coordinates.
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
              // Compute the index in the input (flattened).
              int input_idx = ((b * in_channels + in_c) * in_height + in_y) * in_width + in_x;
              sum += input_data[input_idx] * weight_val;
            }
          }  // end sparse weight loop

          // Add bias if provided.
          if (bias_data && out_c < bias_size) {
            sum += bias_data[out_c];
          }
          // Apply ReLU if needed.
          if (relu && sum < 0.0f) {
            sum = 0.0f;
          }

          // Compute the flattened index for the output array.
          int output_idx = ((b * out_channels + out_c) * out_height + oh) * out_width + ow;
          output_data[output_idx] = sum;
        }  // end ow loop
      }  // end oh loop
    }  // end out_c loop
  }  // end batch loop
}

// ----------------------------------------------------------------------------
// Max Pooling 2D (Sparse, Batched)
// ----------------------------------------------------------------------------

// A cleaner batched max pooling kernel that processes the full range of outputs.
// Input layout: (batch, channels, in_height, in_width)
// Output layout: (batch, channels, out_height, out_width)
inline void maxpool2d_omp_batched_clean(const float *input_data,
                                        const int batch_size,
                                        const int channels,
                                        const int in_height,
                                        const int in_width,
                                        const int pool_size,
                                        const int stride,
                                        float *output_data) {
  // Calculate output spatial dimensions.
  int out_height = (in_height - pool_size) / stride + 1;
  int out_width = (in_width - pool_size) / stride + 1;

// Parallelize over batch, channels and output height dimensions.
// Using collapse helps combine loops into one large iteration space.
#pragma omp parallel for collapse(3) schedule(static)
  for (int b = 0; b < batch_size; ++b) {
    for (int c = 0; c < channels; ++c) {
      for (int oh = 0; oh < out_height; ++oh) {
        for (int ow = 0; ow < out_width; ++ow) {
          float max_val = -std::numeric_limits<float>::max();

          // Iterate over the pooling window.
          for (int p = 0; p < pool_size * pool_size; ++p) {
            int ph = p / pool_size;
            int pw = p % pool_size;
            int in_y = oh * stride + ph;
            int in_x = ow * stride + pw;

            // Check for in-bound coordinates.
            if (in_y < in_height && in_x < in_width) {
              int input_idx = ((b * channels + c) * in_height + in_y) * in_width + in_x;
              max_val = std::max(max_val, input_data[input_idx]);
            }
          }

          int output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
          output_data[output_idx] = max_val;
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Linear Layer (Sparse, Batched)
// ----------------------------------------------------------------------------

// Batched sparse linear (dense) layer kernel.
// Assumptions:
//   - input_data is of shape (batch_size, input_features) (flattened)
//   - weight matrix is in CSR format with dimensions (out_neurons x input_features)
//   - output_data will be of shape (batch_size, out_neurons) (flattened)
inline void linear_omp_batched(
    const float *input_data,
    const int batch_size,
    const int input_features,  // needed for indexing in each sample's input
    const float *weight_vals,
    const int *weight_row_ptr,
    const int *weight_col_idx,
    const float *bias_data,
    float *output_data,
    const int out_neurons) {
// The parallelization is over batch and the output neurons.
#pragma omp for
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < out_neurons; i++) {
      float sum = 0.0f;
      // Loop over the nonzero entries in the i-th row of the weight matrix.
      for (int idx = weight_row_ptr[i]; idx < weight_row_ptr[i + 1]; ++idx) {
        int col = weight_col_idx[idx];
        // Compute the index in the current batch sample's input vector.
        int input_idx = b * input_features + col;
        sum += input_data[input_idx] * weight_vals[idx];
      }
      // Compute the index in the flattened output array:
      int output_idx = b * out_neurons + i;
      output_data[output_idx] = sum + bias_data[i];
    }
  }
}

}  // namespace v2

}  // namespace cifar_sparse::omp
