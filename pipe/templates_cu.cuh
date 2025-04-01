#pragma once

#include "../builtin-apps/common/cuda/helpers.cuh"

__global__ void warmup_kernel() {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Only one thread performs a dummy loop to generate some work.
  if (tid == 0) {
    volatile int dummy = 0;
    for (int i = 0; i < 1000; ++i) {
      dummy += i;
    }
  }
}

inline void warmup() {
  warmup_kernel<<<1, 1>>>();
  CheckCuda(cudaDeviceSynchronize());
}
