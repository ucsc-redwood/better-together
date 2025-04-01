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

  const int N = in_shape[0];   // batch, 128
  const int C = in_shape[1];   // in channels, 3
  const int H = in_shape[2];   // in height, 32
  const int W = in_shape[3];   // in width, 32
  const int R = w_shape[2];    // kernel height, 3
  const int S = w_shape[3];    // kernel width, 3
  const int K = out_shape[1];  // out channels, 16

  constexpr int padding = 0;
  constexpr int stride = 1;
  const int P = (H + 2 * padding - R) / stride + 1;
  const int Q = (W + 2 * padding - S) / stride + 1;
  const int PQ = P * Q;

  // Launch kernel
  const dim3 blockDim(256);
  const dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, K, N);
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
}

void run_stage_2(cifar_dense::AppDataBatch& appdata,
                 const cuda::DeviceModelData& d_model_data,
                 cuda::CudaManager& mgr) {
  const auto& in_shape = appdata.conv1_out.shape();  // [128, 16, 30, 30]
  // const auto& out_shape = appdata.pool1_out.shape();  // [128, 16, 15, 15]

  constexpr int padding = 2;
  constexpr int stride = 2;
  constexpr int pool_h = 2;
  constexpr int pool_w = 2;

  const int N = in_shape[0];  // batch, 128
  const int C = in_shape[1];  // in channels, 16
  const int H = in_shape[2];  // in height, 30. why not 32? answer: padding
  const int W = in_shape[3];  // in width, 30

  int P = (H + 2 * padding - pool_h) / stride + 1;
  int Q = (W + 2 * padding - pool_w) / stride + 1;
  int PQ = P * Q;

  dim3 blockDim(256);
  dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, C, N);
  maxpool2d_kernel<<<gridDim, blockDim, 0, mgr.get_stream()>>>(appdata.conv1_out.raw(),
                                                               appdata.pool1_out.raw(),
                                                               N,
                                                               C,
                                                               H,
                                                               W,
                                                               pool_h,
                                                               pool_w,
                                                               stride,
                                                               padding);
  CheckCuda(cudaStreamSynchronize(mgr.get_stream()));
}

void run_stage_3(cifar_dense::AppDataBatch& appdata) {



  
}

void run_stage_4(cifar_dense::AppDataBatch& appdata) {}

void run_stage_5(cifar_dense::AppDataBatch& appdata) {}

void run_stage_6(cifar_dense::AppDataBatch& appdata) {}

void run_stage_7(cifar_dense::AppDataBatch& appdata) {}

void run_stage_8(cifar_dense::AppDataBatch& appdata) {}

void run_stage_9(cifar_dense::AppDataBatch& appdata) {}

}  // namespace cuda
