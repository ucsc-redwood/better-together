#include "dispatchers.cuh"
#include "kernels.cuh"

namespace cuda {

// void conv2d_batch_cuda(const NDArray<4>& input,
//                        const NDArray<4>& weights,
//                        const NDArray<1>& bias,
//                        int stride,
//                        int padding,
//                        bool relu,
//                        NDArray<4>& output) {
//   // 1) Extract shapes
//   auto in_shape = input.shape();    // {N, inC, inH, inW}
//   auto w_shape = weights.shape();   // {outC, inC, kH, kW}
//   auto out_shape = output.shape();  // {N, outC, outH, outW}

//   int N = static_cast<int>(in_shape[0]);
//   int inC = static_cast<int>(in_shape[1]);
//   int inH = static_cast<int>(in_shape[2]);
//   int inW = static_cast<int>(in_shape[3]);
//   int outC = static_cast<int>(w_shape[0]);
//   int kH = static_cast<int>(w_shape[2]);
//   int kW = static_cast<int>(w_shape[3]);
//   int outH = static_cast<int>(out_shape[2]);
//   int outW = static_cast<int>(out_shape[3]);

//   // 2) Flatten the input arrays for easier device use
//   NDArray<1> input_flat = input.flatten();
//   NDArray<1> weights_flat = weights.flatten();
//   NDArray<1> bias_flat = bias;  // already 1D
//   NDArray<1> output_flat = output.flatten();

//   // 3) Allocate device memory
//   float* d_input = nullptr;
//   float* d_weights = nullptr;
//   float* d_bias = nullptr;
//   float* d_output = nullptr;

//   size_t input_bytes = input_flat.memory_usage_bytes();      // N * inC * inH * inW *
//   sizeof(float) size_t weights_bytes = weights_flat.memory_usage_bytes();  // outC * inC * kH *
//   kW * sizeof(float) size_t bias_bytes = bias_flat.memory_usage_bytes();        // outC *
//   sizeof(float) size_t output_bytes = output_flat.memory_usage_bytes();  // N * outC * outH *
//   outW * sizeof(float)

//   CheckCuda(cudaMalloc(&d_input, input_bytes));
//   CheckCuda(cudaMalloc(&d_weights, weights_bytes));
//   CheckCuda(cudaMalloc(&d_bias, bias_bytes));
//   CheckCuda(cudaMalloc(&d_output, output_bytes));

//   // 4) Copy host->device
//   CheckCuda(cudaMemcpy(d_input, input_flat.raw(), input_bytes, cudaMemcpyHostToDevice));
//   CheckCuda(cudaMemcpy(d_weights, weights_flat.raw(), weights_bytes, cudaMemcpyHostToDevice));
//   CheckCuda(cudaMemcpy(d_bias, bias_flat.raw(), bias_bytes, cudaMemcpyHostToDevice));
//   // We do not copy output since it's only for write-back

//   // 5) Launch kernel
//   int total = N * outC * outH * outW;
//   // e.g. use 256 threads per block:
//   int blockSize = 256;
//   int gridSize = (total + blockSize - 1) / blockSize;

//   conv2d_batch_kernel<<<gridSize, blockSize>>>(d_input,
//                                                d_weights,
//                                                d_bias,
//                                                d_output,
//                                                N,
//                                                inC,
//                                                inH,
//                                                inW,
//                                                outC,
//                                                kH,
//                                                kW,
//                                                stride,
//                                                padding,
//                                                relu,
//                                                outH,
//                                                outW);
//   CheckCuda(cudaGetLastError());

//   // 6) Copy device->host
//   CheckCuda(cudaMemcpy(output_flat.raw(), d_output, output_bytes, cudaMemcpyDeviceToHost));

//   // 7) Cleanup device allocations
//   CheckCuda(cudaFree(d_input));
//   CheckCuda(cudaFree(d_weights));
//   CheckCuda(cudaFree(d_bias));
//   CheckCuda(cudaFree(d_output));

//   // 8) Copy the flattened output back into the 4D output NDArray
//   //    (This step is optional if your NDArray allows direct constructor usage from a raw
//   pointer.)
//   //    We'll just do a loop for simplicity:
//   float* out_ptr = output.raw();
//   const float* flat_ptr = output_flat.raw();
//   for (size_t i = 0; i < out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]; i++) {
//     out_ptr[i] = flat_ptr[i];
//   }
// }

void conv2d_batch_cuda(const NDArray<4>& input,
                       const NDArray<4>& weights,
                       const NDArray<1>& bias,
                       int stride,
                       int padding,
                       bool relu,
                       NDArray<4>& output) {
  // 1) Extract shapes
  auto in_shape = input.shape();    // {N, inC, inH, inW}
  auto w_shape = weights.shape();   // {outC, inC, kH, kW}
  auto out_shape = output.shape();  // {N, outC, outH, outW}

  int N = static_cast<int>(in_shape[0]);
  int inC = static_cast<int>(in_shape[1]);
  int inH = static_cast<int>(in_shape[2]);
  int inW = static_cast<int>(in_shape[3]);
  int outC = static_cast<int>(w_shape[0]);
  int kH = static_cast<int>(w_shape[2]);
  int kW = static_cast<int>(w_shape[3]);
  int outH = static_cast<int>(out_shape[2]);
  int outW = static_cast<int>(out_shape[3]);

  // Launch kernel with managed memory
  int total = N * outC * outH * outW;
  int blockSize = 256;
  int gridSize = (total + blockSize - 1) / blockSize;
  
  std::cout << "Launching kernel with grid=" << gridSize << ", block=" << blockSize << std::endl;
  std::cout << "Total threads needed: " << total << std::endl;

  conv2d_batch_kernel<<<gridSize, blockSize>>>(input.raw(),
                                               weights.raw(),
                                               bias.raw(),
                                               output.raw(),
                                               N,
                                               inC,
                                               inH,
                                               inW,
                                               outC,
                                               kH,
                                               kW,
                                               stride,
                                               padding,
                                               relu,
                                               outH,
                                               outW);

  // Check for errors and synchronize
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  }
  CheckCuda(cudaDeviceSynchronize());
}

void run_stage_1(cifar_dense::AppDataBatch& appdata) {
  std::cout << '\n';
  appdata.conv1_out.print("conv1_out before: ");
  std::cout << '\n';

  // Print shape information for debugging
  int N = static_cast<int>(appdata.input.shape()[0]);
  int inC = static_cast<int>(appdata.input.shape()[1]);
  int inH = static_cast<int>(appdata.input.shape()[2]);
  int inW = static_cast<int>(appdata.input.shape()[3]);
  int outC = static_cast<int>(appdata.conv1_w.shape()[0]);
  int kH = static_cast<int>(appdata.conv1_w.shape()[2]);
  int kW = static_cast<int>(appdata.conv1_w.shape()[3]);
  int outH = static_cast<int>(appdata.conv1_out.shape()[2]);
  int outW = static_cast<int>(appdata.conv1_out.shape()[3]);
  
  std::cout << "Input shape: " << N << "x" << inC << "x" << inH << "x" << inW << std::endl;
  std::cout << "Weight shape: " << outC << "x" << inC << "x" << kH << "x" << kW << std::endl;
  std::cout << "Output shape: " << N << "x" << outC << "x" << outH << "x" << outW << std::endl;
  
  // Run the complete convolution
  conv2d_batch_cuda(appdata.input, appdata.conv1_w, appdata.conv1_b, 1, 0, true, appdata.conv1_out);

  // Print the results
  std::cout << '\n';
  appdata.conv1_out.print("conv1_out after: ");
  std::cout << '\n';
}

void run_stage_2(cifar_dense::AppDataBatch& appdata) {}

void run_stage_3(cifar_dense::AppDataBatch& appdata) {}

void run_stage_4(cifar_dense::AppDataBatch& appdata) {}

void run_stage_5(cifar_dense::AppDataBatch& appdata) {}

void run_stage_6(cifar_dense::AppDataBatch& appdata) {}

void run_stage_7(cifar_dense::AppDataBatch& appdata) {}

void run_stage_8(cifar_dense::AppDataBatch& appdata) {}

void run_stage_9(cifar_dense::AppDataBatch& appdata) {}

}  // namespace cuda
