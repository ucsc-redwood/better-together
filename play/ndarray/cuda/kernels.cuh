#pragma once

namespace cuda {

__global__ void conv2d_kernel(const float* __restrict__ input,    // [N, C, H, W]
                              const float* __restrict__ weights,  // [K, C, R, S]
                              const float* __restrict__ bias,     // [K]
                              float* output,                      // [N, K, P, Q]
                              int N,
                              int C,
                              int H,
                              int W,  // input shape
                              int K,
                              int R,
                              int S,  // output channels and kernel shape
                              int stride = 1,
                              int padding = 0,
                              bool apply_relu = true);
/**
 * @brief CUDA kernel for 2D max pooling operation
 *
 * This kernel performs max pooling on a 4D input tensor with shape [N, C, H, W].
 * The output tensor has shape [N, C, P, Q] where:
 *   P = (H + 2*padding - pool_h)/stride + 1
 *   Q = (W + 2*padding - pool_w)/stride + 1
 *
 * @param input Input tensor of shape [N, C, H, W]
 * @param output Output tensor of shape [N, C, P, Q]
 * @param N Batch size
 * @param C Number of channels
 * @param H Input height
 * @param W Input width
 * @param pool_h Pooling kernel height
 * @param pool_w Pooling kernel width
 * @param stride Stride for pooling operation (default: 2)
 * @param padding Padding size (default: 2)

 * Example usage:
 * @code
 * // Calculate output dimensions
 * int P = (H + 2 * padding - pool_h) / stride + 1;
 * int Q = (W + 2 * padding - pool_w) / stride + 1;
 * int PQ = P * Q;
 *
 * // Configure kernel launch parameters
 * dim3 blockDim(256);
 * dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, C, N);
 *
 * // Launch kernel
 * maxpool2d_kernel<<<gridDim, blockDim>>>(
 *     d_input, d_output,
 *     N, C, H, W,
 *     pool_h, pool_w,
 *     stride, padding
 * );
 * cudaDeviceSynchronize();
 * @endcode
 *
 */
__global__ void maxpool2d_kernel(const float* __restrict__ input,  // [N, C, H, W]
                                 float* output,                    // [N, C, P, Q]
                                 int N,
                                 int C,
                                 int H,
                                 int W,  // input shape
                                 int pool_h,
                                 int pool_w,  // pooling kernel size
                                 int stride = 2,
                                 int padding = 2);

/**
 * @brief CUDA kernel for linear (fully connected) layer operation
 *
 * This kernel performs a linear transformation on the flattened input tensor.
 * For each batch item and each output feature, it computes the dot product of
 * the input and the corresponding weight row, then adds the bias.
 *
 * @param input Input tensor with shape [N, in_features] 
 * @param weights Weight matrix with shape [out_features, in_features]
 * @param bias Bias vector with shape [out_features]
 * @param output Output tensor with shape [N, out_features]
 * @param N Batch size
 * @param in_features Number of input features
 * @param out_features Number of output features
 */
__global__ void linear_kernel(const float* __restrict__ input,    // [N, in_features]
                              const float* __restrict__ weights,  // [out_features, in_features]
                              const float* __restrict__ bias,     // [out_features]
                              float* output,                      // [N, out_features]
                              int N,
                              int in_features,
                              int out_features);

}  // namespace cuda
