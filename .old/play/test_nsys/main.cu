#include <stdio.h>

__global__ void helloFromGPU() { printf("Hello World from GPU!\n"); }

int main() {
  // Launch the kernel
  helloFromGPU<<<1, 10>>>();

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  return 0;
}
