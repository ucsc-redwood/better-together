#pragma once

#include <cstdint>

namespace cifar_dense::omp {

// Function declarations
void conv2d_omp(const float *input_data,
                const int image_input_channels,
                const int input_height,
                const int input_width,
                const float *weight_data,
                const int weight_output_channels,
                const int weight_input_channels,
                const int weight_height,
                const int weight_width,
                const float *bias_data,
                const int bias_number_of_elements,
                const int kernel_size,
                const int stride,
                const int padding,
                const bool relu,
                float *output_data,
                const int start,
                const int end);

void maxpool2d_omp(const float *input_data,
                   const int input_channels,
                   const int input_height,
                   const int input_width,
                   const int pool_size,
                   const int stride,
                   float *output_data,
                   const int start,
                   const int end);

void linear_omp(const float *input,
                const float *weights,
                const float *bias,
                float *output,
                const uint32_t input_size,
                const uint32_t output_size,
                const int start,
                const int end);

}  // namespace cifar_dense::omp
