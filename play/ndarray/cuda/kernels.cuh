#pragma once

namespace cuda {

__global__ void test_access(float* __restrict__ an_array);

__global__ void conv2d_batch_kernel(const float* __restrict__ input,
                                    const float* __restrict__ weights,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    int N,     // batch size
                                    int inC,   // input channels
                                    int inH,   // input height
                                    int inW,   // input width
                                    int outC,  // output channels
                                    int kH,    // kernel height
                                    int kW,    // kernel width
                                    int stride,
                                    int padding,
                                    bool relu,
                                    int outH,
                                    int outW);

}  // namespace cuda
