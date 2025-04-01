#pragma once

#include <cuda_runtime.h>

namespace cifar_dense::cuda {

__global__ void conv2d(const float* __restrict__ input_data,
                       const float* __restrict__ weight_data,
                       const float* __restrict__ bias_data,
                       float* __restrict__ output_data,
                       int input_height,
                       int input_width,
                       int weight_output_channels,
                       int weight_input_channels,
                       int weight_height,
                       int weight_width,
                       int bias_number_of_elements,
                       int kernel_size,
                       int stride,
                       int padding,
                       int output_height,
                       int output_width,
                       bool relu);

__global__ void maxpool2d(const float* __restrict__ input_data,
                          float* __restrict__ output_data,
                          int input_channels,
                          int input_height,
                          int input_width,
                          int pool_size,
                          int stride,
                          int output_height,
                          int output_width);

__global__ void linear(const float* __restrict__ input_data,
                       const float* __restrict__ weight_data,
                       const float* __restrict__ bias_data,
                       float* __restrict__ output_data,
                       int input_size,
                       int output_size);

}  // namespace cifar_dense::cuda
