#pragma once

namespace cuda {

__global__ void test_access(float* __restrict__ an_array);

// __global__ void conv2d_batch_kernel(const float* __restrict__ input,
//                                     const float* __restrict__ weights,
//                                     const float* __restrict__ bias,
//                                     float* __restrict__ output,
//                                     int N,     // batch size
//                                     int inC,   // input channels
//                                     int inH,   // input height
//                                     int inW,   // input width
//                                     int outC,  // output channels
//                                     int kH,    // kernel height
//                                     int kW,    // kernel width
//                                     int stride,
//                                     int padding,
//                                     bool relu,
//                                     int outH,
//                                     int outW);

__global__ void k_conv2d_batch_u(const float* __restrict__ u_input,
                                 const float* __restrict__ u_weights,
                                 const float* __restrict__ u_bias,
                                 float* __restrict__ u_output,
                                 const int N,     // in_shape[0]
                                 const int inC,   // in_shape[1]
                                 const int inH,   // in_shape[2]
                                 const int inW,   // in_shape[3]
                                 const int outC,  // w_shape[0]
                                 const int kH,    // w_shape[2]
                                 const int kW,    // w_shape[3]
                                 const int outH,  // out_shape[2]
                                 const int outW,  // out_shape[3]
                                 const int stride,
                                 const int padding,
                                 const bool relu);

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
                              int stride,
                              int padding,
                              bool apply_relu);

}  // namespace cuda
