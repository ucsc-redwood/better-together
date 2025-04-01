#include "builtin-apps/common/cuda/helpers.cuh"
#include "dispatchers.cuh"
#include "kernels.cuh"

namespace cuda {

#define CudaAttachSingle(ptr) \
  (cudaStreamAttachMemAsync(mgr.get_stream(), ptr, 0, cudaMemAttachSingle))
#define CudaAttachHost(ptr) (cudaStreamAttachMemAsync(mgr.get_stream(), ptr, 0, cudaMemAttachHost))

void run_stage_1(cifar_dense::AppDataBatch& appdata,
                 const cuda::DeviceModelData& d_model_data,
                 cuda::CudaManager& mgr) {
  const auto& in_shape = appdata.input.shape();                      // [128, 3, 32, 32]
  const auto& w_shape = d_model_data.h_model_ref.h_conv1_w.shape();  // [16, 3, 3, 3]
  const auto& out_shape = appdata.conv1_out.shape();                 // [128, 16, 30, 30]

  appdata.input.print_shape("input");
  // appdata.conv1_w.print_shape("conv1_w");
  // appdata.conv1_b.print_shape("conv1_b");
  appdata.conv1_out.print_shape("conv1_out");

  const auto N = in_shape[0];   // batch, 128
  const auto C = in_shape[1];   // in channels, 3
  const auto H = in_shape[2];   // in height, 32
  const auto W = in_shape[3];   // in width, 32
  const auto R = w_shape[2];    // kernel height, 3
  const auto S = w_shape[3];    // kernel width, 3
  const auto K = out_shape[1];  // out channels, 16

  CheckCuda(cudaStreamSynchronize(mgr.get_stream()));

  constexpr auto padding = 0;
  constexpr auto stride = 1;
  int P = (H + 2 * padding - R) / stride + 1;
  int Q = (W + 2 * padding - S) / stride + 1;
  int PQ = P * Q;

  // // Allocate device memory
  // float* d_weights;
  // float* d_bias;

  // auto w_total_size = K * C * R * S * sizeof(float);
  // auto b_total_size = K * sizeof(float);

  // CheckCuda(cudaMalloc(&d_weights, w_total_size));
  // CheckCuda(cudaMalloc(&d_bias, b_total_size));

  // // Copy data to device
  // CheckCuda(cudaMemcpy(d_weights, appdata.conv1_w.raw(), w_total_size, cudaMemcpyHostToDevice));
  // CheckCuda(cudaMemcpy(d_bias, appdata.conv1_b.raw(), b_total_size, cudaMemcpyHostToDevice));

  // Launch kernel
  dim3 blockDim(256);
  dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, K, N);
  conv2d_kernel<<<gridDim, blockDim, 0, mgr.get_stream()>>>(appdata.input.raw(),
                                                            d_model_data.d_conv1_w,
                                                            d_model_data.d_conv1_b,
                                                            appdata.conv1_out.raw(),
                                                            N,
                                                            C,
                                                            H,
                                                            W,
                                                            K,
                                                            R,
                                                            S,
                                                            stride,
                                                            padding,
                                                            true);

  CheckCuda(cudaStreamSynchronize(mgr.get_stream()));

  // Free device memory
  // CheckCuda(cudaFree(d_weights));
  // CheckCuda(cudaFree(d_bias));

  appdata.conv1_out.print("conv1_out");
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
