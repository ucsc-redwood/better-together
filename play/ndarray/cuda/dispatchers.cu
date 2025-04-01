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

void run_stage_3(cifar_dense::AppDataBatch& appdata,
                 const cuda::DeviceModelData& d_model_data,
                 cuda::CudaManager& mgr) {
  const auto& in_shape = appdata.pool1_out.shape();                  // [128, 16, 15, 15]
  const auto& w_shape = d_model_data.h_model_ref.h_conv2_w.shape();  // [20, 16, 3, 3]
  const auto& out_shape = appdata.conv2_out.shape();                 // [128, 20, 13, 13]

  const int N = in_shape[0];   // batch, 128
  const int C = in_shape[1];   // in channels, 16
  const int H = in_shape[2];   // in height, 15
  const int W = in_shape[3];   // in width, 15
  const int R = w_shape[2];    // kernel height, 3
  const int S = w_shape[3];    // kernel width, 3
  const int K = out_shape[1];  // out channels, 20

  constexpr int padding = 0;
  constexpr int stride = 1;
  const int P = (H + 2 * padding - R) / stride + 1;
  const int Q = (W + 2 * padding - S) / stride + 1;
  const int PQ = P * Q;

  // Launch kernel
  const dim3 blockDim(256);
  const dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, K, N);
  conv2d_kernel<<<gridDim, blockDim, 0, mgr.get_stream()>>>(appdata.pool1_out.raw(),
                                                            d_model_data.d_conv2_w,
                                                            d_model_data.d_conv2_b,
                                                            appdata.conv2_out.raw(),
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

void run_stage_4(cifar_dense::AppDataBatch& appdata,
                 const cuda::DeviceModelData& d_model_data,
                 cuda::CudaManager& mgr) {
  const auto& in_shape = appdata.conv2_out.shape();  // [128, 20, 13, 13]

  constexpr int padding = 2;
  constexpr int stride = 2;
  constexpr int pool_h = 2;
  constexpr int pool_w = 2;

  const int N = in_shape[0];  // batch, 128
  const int C = in_shape[1];  // in channels, 20
  const int H = in_shape[2];  // in height, 13
  const int W = in_shape[3];  // in width, 13

  int P = (H + 2 * padding - pool_h) / stride + 1;
  int Q = (W + 2 * padding - pool_w) / stride + 1;
  int PQ = P * Q;

  dim3 blockDim(256);
  dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, C, N);
  maxpool2d_kernel<<<gridDim, blockDim, 0, mgr.get_stream()>>>(appdata.conv2_out.raw(),
                                                               appdata.pool2_out.raw(),
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

void run_stage_5(cifar_dense::AppDataBatch& appdata,
                 const cuda::DeviceModelData& d_model_data,
                 cuda::CudaManager& mgr) {
  const auto& in_shape = appdata.pool2_out.shape();                  // [128, 20, 7, 7]
  const auto& w_shape = d_model_data.h_model_ref.h_conv3_w.shape();  // [20, 20, 3, 3]
  const auto& out_shape = appdata.conv3_out.shape();                 // [128, 20, 5, 5]

  const int N = in_shape[0];   // batch, 128
  const int C = in_shape[1];   // in channels, 20
  const int H = in_shape[2];   // in height, 7
  const int W = in_shape[3];   // in width, 7
  const int R = w_shape[2];    // kernel height, 3
  const int S = w_shape[3];    // kernel width, 3
  const int K = out_shape[1];  // out channels, 20

  constexpr int padding = 0;
  constexpr int stride = 1;
  const int P = (H + 2 * padding - R) / stride + 1;
  const int Q = (W + 2 * padding - S) / stride + 1;
  const int PQ = P * Q;

  // Launch kernel
  const dim3 blockDim(256);
  const dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, K, N);
  conv2d_kernel<<<gridDim, blockDim, 0, mgr.get_stream()>>>(appdata.pool2_out.raw(),
                                                            d_model_data.d_conv3_w,
                                                            d_model_data.d_conv3_b,
                                                            appdata.conv3_out.raw(),
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

void run_stage_6(cifar_dense::AppDataBatch& appdata,
                 const cuda::DeviceModelData& d_model_data,
                 cuda::CudaManager& mgr) {
  const auto& in_shape = appdata.conv3_out.shape();                  // [128, 20, 5, 5]
  const auto& w_shape = d_model_data.h_model_ref.h_conv4_w.shape();  // [50, 20, 3, 3]
  const auto& out_shape = appdata.conv4_out.shape();                 // [128, 50, 3, 3]

  const int N = in_shape[0];   // batch, 128
  const int C = in_shape[1];   // in channels, 20
  const int H = in_shape[2];   // in height, 5
  const int W = in_shape[3];   // in width, 5
  const int R = w_shape[2];    // kernel height, 3
  const int S = w_shape[3];    // kernel width, 3
  const int K = out_shape[1];  // out channels, 50

  constexpr int padding = 0;
  constexpr int stride = 1;
  const int P = (H + 2 * padding - R) / stride + 1;
  const int Q = (W + 2 * padding - S) / stride + 1;
  const int PQ = P * Q;

  // Launch kernel
  const dim3 blockDim(256);
  const dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, K, N);
  conv2d_kernel<<<gridDim, blockDim, 0, mgr.get_stream()>>>(appdata.conv3_out.raw(),
                                                            d_model_data.d_conv4_w,
                                                            d_model_data.d_conv4_b,
                                                            appdata.conv4_out.raw(),
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

void run_stage_7(cifar_dense::AppDataBatch& appdata,
                 const cuda::DeviceModelData& d_model_data,
                 cuda::CudaManager& mgr) {
  const auto& in_shape = appdata.conv4_out.shape();                  // [128, 50, 3, 3]
  const auto& w_shape = d_model_data.h_model_ref.h_conv5_w.shape();  // [64, 50, 3, 3]
  const auto& out_shape = appdata.conv5_out.shape();                 // [128, 64, 1, 1]

  const int N = in_shape[0];   // batch, 128
  const int C = in_shape[1];   // in channels, 50
  const int H = in_shape[2];   // in height, 3
  const int W = in_shape[3];   // in width, 3
  const int R = w_shape[2];    // kernel height, 3
  const int S = w_shape[3];    // kernel width, 3
  const int K = out_shape[1];  // out channels, 64

  constexpr int padding = 0;
  constexpr int stride = 1;
  const int P = (H + 2 * padding - R) / stride + 1;
  const int Q = (W + 2 * padding - S) / stride + 1;
  const int PQ = P * Q;

  // Launch kernel
  const dim3 blockDim(256);
  const dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, K, N);
  conv2d_kernel<<<gridDim, blockDim, 0, mgr.get_stream()>>>(appdata.conv4_out.raw(),
                                                            d_model_data.d_conv5_w,
                                                            d_model_data.d_conv5_b,
                                                            appdata.conv5_out.raw(),
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

void run_stage_8(cifar_dense::AppDataBatch& appdata,
                 const cuda::DeviceModelData& d_model_data,
                 cuda::CudaManager& mgr) {
  const auto& in_shape = appdata.conv5_out.shape();  // [128, 64, 1, 1]

  constexpr int padding = 2;
  constexpr int stride = 2;
  constexpr int pool_h = 2;
  constexpr int pool_w = 2;

  const int N = in_shape[0];  // batch, 128
  const int C = in_shape[1];  // in channels, 64
  const int H = in_shape[2];  // in height, 1
  const int W = in_shape[3];  // in width, 1

  int P = (H + 2 * padding - pool_h) / stride + 1;
  int Q = (W + 2 * padding - pool_w) / stride + 1;
  int PQ = P * Q;

  dim3 blockDim(256);
  dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, C, N);
  maxpool2d_kernel<<<gridDim, blockDim, 0, mgr.get_stream()>>>(appdata.conv5_out.raw(),
                                                               appdata.pool3_out.raw(),
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

void run_stage_9(cifar_dense::AppDataBatch& appdata,
                 const cuda::DeviceModelData& d_model_data,
                 cuda::CudaManager& mgr) {
  const auto& in_shape = appdata.pool3_out.shape();                   // [128, 64, 1, 1]
  const auto& w_shape = d_model_data.h_model_ref.h_linear_w.shape();  // [10, 1024]

  // For the linear layer, we need to flatten the 4D tensor to 2D
  const int N = in_shape[0];  // batch size, 128
  const int C = in_shape[1];  // channels, 64
  const int H = in_shape[2];  // height, 1
  const int W = in_shape[3];  // width, 1

  const int in_features =
      C * H * W;  // 64*1*1 = 64 (or could be 1024 depending on actual dimensions)
  const int out_features = w_shape[0];  // output features, 10

  // Launch kernel for linear layer (matrix multiplication)
  const int block_size = 256;
  const int num_blocks = (N * out_features + block_size - 1) / block_size;

  linear_kernel<<<num_blocks, block_size, 0, mgr.get_stream()>>>(appdata.pool3_out.raw(),
                                                                 d_model_data.d_linear_w,
                                                                 d_model_data.d_linear_b,
                                                                 appdata.linear_out.raw(),
                                                                 N,
                                                                 in_features,
                                                                 out_features);

  CheckCuda(cudaStreamSynchronize(mgr.get_stream()));
}

}  // namespace cuda
