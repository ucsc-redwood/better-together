#pragma once

#include "../appdata.hpp"
#include "builtin-apps/common/cuda/cu_mem_resource.cuh"
#include "builtin-apps/common/cuda/helpers.cuh"
#include "builtin-apps/common/cuda/manager.cuh"
#include "builtin-apps/debug_logger.hpp"
#include "kernels.cuh"
#include "model_data.cuh"

namespace cuda {

constexpr bool debug_layer_outputs = false;

template <typename MemResourceT>
  requires std::is_same_v<MemResourceT, CudaManagedResource> ||
           std::is_same_v<MemResourceT, CudaPinnedResource>
class CudaDispatcher final : public cuda::CudaManager<MemResourceT> {
 public:
  explicit CudaDispatcher() : d_model_data_(cifar_dense::AppDataBatch::get_model()) {}

  void run_stage_1_async(cifar_dense::AppDataBatch& appdata) {
    const auto& in_shape = appdata.input.shape();                       // [128, 3, 32, 32]
    const auto& w_shape = d_model_data_.h_model_ref.h_conv1_w.shape();  // [16, 3, 3, 3]
    const auto& out_shape = appdata.conv1_out.shape();                  // [128, 16, 30, 30]

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
    LOG_KERNEL(LogKernelType::kCUDA, 1, &appdata);

    const dim3 blockDim(256);
    const dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, K, N);
    conv2d_kernel<<<gridDim, blockDim, 0, mgr_.get_stream()>>>(appdata.input.raw(),
                                                               d_model_data_.d_conv1_w,
                                                               d_model_data_.d_conv1_b,
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

    if constexpr (debug_layer_outputs) {
      CheckCuda(cudaStreamSynchronize(mgr_.get_stream()));
      appdata.conv1_out.print("conv1_out");
    }
  }

  void run_stage_2_async(cifar_dense::AppDataBatch& appdata) {
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

    LOG_KERNEL(LogKernelType::kCUDA, 2, &appdata);

    dim3 blockDim(256);
    dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, C, N);
    maxpool2d_kernel<<<gridDim, blockDim, 0, mgr_.get_stream()>>>(appdata.conv1_out.raw(),
                                                                  appdata.pool1_out.raw(),
                                                                  N,
                                                                  C,
                                                                  H,
                                                                  W,
                                                                  pool_h,
                                                                  pool_w,
                                                                  stride,
                                                                  padding);

    if constexpr (debug_layer_outputs) {
      CheckCuda(cudaStreamSynchronize(mgr_.get_stream()));
      appdata.pool1_out.print("pool1_out");
    }
  }

  void run_stage_3_async(cifar_dense::AppDataBatch& appdata) {
    const auto& in_shape = appdata.pool1_out.shape();                   // [128, 16, 15, 15]
    const auto& w_shape = d_model_data_.h_model_ref.h_conv2_w.shape();  // [20, 16, 3, 3]
    const auto& out_shape = appdata.conv2_out.shape();                  // [128, 20, 13, 13]

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

    LOG_KERNEL(LogKernelType::kCUDA, 3, &appdata);

    const dim3 blockDim(256);
    const dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, K, N);
    conv2d_kernel<<<gridDim, blockDim, 0, mgr_.get_stream()>>>(appdata.pool1_out.raw(),
                                                               d_model_data_.d_conv2_w,
                                                               d_model_data_.d_conv2_b,
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

    if constexpr (debug_layer_outputs) {
      CheckCuda(cudaStreamSynchronize(mgr_.get_stream()));
      appdata.conv2_out.print("conv2_out");
    }
  }

  void run_stage_4_async(cifar_dense::AppDataBatch& appdata) {
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

    LOG_KERNEL(LogKernelType::kCUDA, 4, &appdata);

    dim3 blockDim(256);
    dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, C, N);
    maxpool2d_kernel<<<gridDim, blockDim, 0, mgr_.get_stream()>>>(appdata.conv2_out.raw(),
                                                                  appdata.pool2_out.raw(),
                                                                  N,
                                                                  C,
                                                                  H,
                                                                  W,
                                                                  pool_h,
                                                                  pool_w,
                                                                  stride,
                                                                  padding);

    if constexpr (debug_layer_outputs) {
      CheckCuda(cudaStreamSynchronize(mgr_.get_stream()));
      appdata.pool2_out.print("pool2_out");
    }
  }

  void run_stage_5_async(cifar_dense::AppDataBatch& appdata) {
    const auto& in_shape = appdata.pool2_out.shape();                   // [128, 20, 7, 7]
    const auto& w_shape = d_model_data_.h_model_ref.h_conv3_w.shape();  // [20, 20, 3, 3]
    const auto& out_shape = appdata.conv3_out.shape();                  // [128, 20, 5, 5]

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

    LOG_KERNEL(LogKernelType::kCUDA, 5, &appdata);

    // Launch kernel
    const dim3 blockDim(256);
    const dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, K, N);
    conv2d_kernel<<<gridDim, blockDim, 0, mgr_.get_stream()>>>(appdata.pool2_out.raw(),
                                                               d_model_data_.d_conv3_w,
                                                               d_model_data_.d_conv3_b,
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

    if constexpr (debug_layer_outputs) {
      CheckCuda(cudaStreamSynchronize(mgr_.get_stream()));
      appdata.conv3_out.print("conv3_out");
    }
  }

  void run_stage_6_async(cifar_dense::AppDataBatch& appdata) {
    const auto& in_shape = appdata.conv3_out.shape();                   // [128, 20, 5, 5]
    const auto& w_shape = d_model_data_.h_model_ref.h_conv4_w.shape();  // [50, 20, 3, 3]
    const auto& out_shape = appdata.conv4_out.shape();                  // [128, 50, 3, 3]

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

    LOG_KERNEL(LogKernelType::kCUDA, 6, &appdata);

    // Launch kernel
    const dim3 blockDim(256);
    const dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, K, N);
    conv2d_kernel<<<gridDim, blockDim, 0, mgr_.get_stream()>>>(appdata.conv3_out.raw(),
                                                               d_model_data_.d_conv4_w,
                                                               d_model_data_.d_conv4_b,
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

    if constexpr (debug_layer_outputs) {
      CheckCuda(cudaStreamSynchronize(mgr_.get_stream()));
      appdata.conv4_out.print("conv4_out");
    }
  }

  void run_stage_7_async(cifar_dense::AppDataBatch& appdata) {
    const auto& in_shape = appdata.conv4_out.shape();                   // [128, 50, 3, 3]
    const auto& w_shape = d_model_data_.h_model_ref.h_conv5_w.shape();  // [64, 50, 3, 3]
    const auto& out_shape = appdata.conv5_out.shape();                  // [128, 64, 1, 1]

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

    LOG_KERNEL(LogKernelType::kCUDA, 7, &appdata);

    // Launch kernel
    const dim3 blockDim(256);
    const dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, K, N);
    conv2d_kernel<<<gridDim, blockDim, 0, mgr_.get_stream()>>>(appdata.conv4_out.raw(),
                                                               d_model_data_.d_conv5_w,
                                                               d_model_data_.d_conv5_b,
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

    if constexpr (debug_layer_outputs) {
      CheckCuda(cudaStreamSynchronize(mgr_.get_stream()));
      appdata.conv5_out.print("conv5_out");
    }
  }

  void run_stage_8_async(cifar_dense::AppDataBatch& appdata) {
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

    LOG_KERNEL(LogKernelType::kCUDA, 8, &appdata);

    dim3 blockDim(256);
    dim3 gridDim((PQ + blockDim.x - 1) / blockDim.x, C, N);
    maxpool2d_kernel<<<gridDim, blockDim, 0, mgr_.get_stream()>>>(appdata.conv5_out.raw(),
                                                                  appdata.pool3_out.raw(),
                                                                  N,
                                                                  C,
                                                                  H,
                                                                  W,
                                                                  pool_h,
                                                                  pool_w,
                                                                  stride,
                                                                  padding);

    if constexpr (debug_layer_outputs) {
      CheckCuda(cudaStreamSynchronize(mgr_.get_stream()));
      appdata.pool3_out.print("pool3_out");
    }
  }

  void run_stage_9_async(cifar_dense::AppDataBatch& appdata) {
    const auto& in_shape = appdata.pool3_out.shape();                    // [128, 64, 1, 1]
    const auto& w_shape = d_model_data_.h_model_ref.h_linear_w.shape();  // [10, 1024]

    // For the linear layer, we need to flatten the 4D tensor to 2D
    const int N = in_shape[0];  // batch size, 128
    const int C = in_shape[1];  // channels, 64
    const int H = in_shape[2];  // height, 1
    const int W = in_shape[3];  // width, 1

    // 64*1*1 = 64 (or could be 1024 depending on actual dimensions)
    const int in_features = C * H * W;
    const int out_features = w_shape[0];  // output features, 10

    LOG_KERNEL(LogKernelType::kCUDA, 9, &appdata);

    // Launch kernel for linear layer (matrix multiplication)
    const int block_size = 256;
    const int num_blocks = (N * out_features + block_size - 1) / block_size;

    linear_kernel<<<num_blocks, block_size, 0, mgr_.get_stream()>>>(appdata.pool3_out.raw(),
                                                                    d_model_data_.d_linear_w,
                                                                    d_model_data_.d_linear_b,
                                                                    appdata.linear_out.raw(),
                                                                    N,
                                                                    in_features,
                                                                    out_features);

    if constexpr (debug_layer_outputs) {
      CheckCuda(cudaStreamSynchronize(mgr_.get_stream()));
      appdata.linear_out.print("linear_out");
    }
  }

  using StageFn = void (CudaDispatcher::*)(cifar_dense::AppDataBatch&);

  static constexpr std::array<StageFn, 9> stage_functions = {
      &CudaDispatcher::run_stage_1_async,
      &CudaDispatcher::run_stage_2_async,
      &CudaDispatcher::run_stage_3_async,
      &CudaDispatcher::run_stage_4_async,
      &CudaDispatcher::run_stage_5_async,
      &CudaDispatcher::run_stage_6_async,
      &CudaDispatcher::run_stage_7_async,
      &CudaDispatcher::run_stage_8_async,
      &CudaDispatcher::run_stage_9_async,
  };

#define CudaAttachSingle(ptr) \
  (cudaStreamAttachMemAsync(mgr_.get_stream(), ptr, 0, cudaMemAttachSingle))
#define CudaAttachHost(ptr) (cudaStreamAttachMemAsync(mgr_.get_stream(), ptr, 0, cudaMemAttachHost))

  void dispatch_multi_stage(cifar_dense::AppDataBatch& data,
                            const int start_stage,
                            const int end_stage) {
    if (start_stage < 1 || end_stage > 9) throw std::out_of_range("Invalid stage");

    // Only attach memory if using CudaManagedResource
    if constexpr (std::is_same_v<MemResourceT, CudaManagedResource>) {
      CudaAttachSingle(data.input.raw());
      CudaAttachSingle(data.conv1_out.raw());
      CudaAttachSingle(data.pool1_out.raw());
      CudaAttachSingle(data.conv2_out.raw());
      CudaAttachSingle(data.pool2_out.raw());
      CudaAttachSingle(data.conv3_out.raw());
      CudaAttachSingle(data.conv4_out.raw());
      CudaAttachSingle(data.conv5_out.raw());
      CudaAttachSingle(data.pool3_out.raw());
      CudaAttachSingle(data.linear_out.raw());
    }

    for (int stage = start_stage; stage <= end_stage; stage++) {
      (this->*stage_functions[stage - 1])(data);
    }

    CheckCuda(cudaStreamSynchronize(mgr_.get_stream()));

    // Only attach memory if using CudaManagedResource
    if constexpr (std::is_same_v<MemResourceT, CudaManagedResource>) {
      CudaAttachHost(data.input.raw());
      CudaAttachHost(data.conv1_out.raw());
      CudaAttachHost(data.pool1_out.raw());
      CudaAttachHost(data.conv2_out.raw());
      CudaAttachHost(data.pool2_out.raw());
      CudaAttachHost(data.conv3_out.raw());
      CudaAttachHost(data.conv4_out.raw());
      CudaAttachHost(data.conv5_out.raw());
      CudaAttachHost(data.pool3_out.raw());
      CudaAttachHost(data.linear_out.raw());
    }
  }

 private:
  // create a manager
  CudaManager<MemResourceT> mgr_;

  // Device-only memory
  const DeviceModelData d_model_data_;
};

// #define CudaAttachSingle(ptr) \
//   (cudaStreamAttachMemAsync(mgr.get_stream(), ptr, 0, cudaMemAttachSingle))
// #define CudaAttachHost(ptr) (cudaStreamAttachMemAsync(mgr.get_stream(), ptr, 0,
// cudaMemAttachHost))

// // clang-format off
// void run_stage_1_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData&
// d_model_data);  // Conv 1 void run_stage_2_async(cifar_dense::AppDataBatch& appdata, const
// cuda::DeviceModelData& d_model_data);  // MaxPool 1 void
// run_stage_3_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData& d_model_data);
// // Conv 2 void run_stage_4_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData&
// d_model_data);  // MaxPool 2 void run_stage_5_async(cifar_dense::AppDataBatch& appdata, const
// cuda::DeviceModelData& d_model_data);  // Conv 3 void
// run_stage_6_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData& d_model_data);
// // Conv 4 void run_stage_7_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData&
// d_model_data);  // Conv 5 void run_stage_8_async(cifar_dense::AppDataBatch& appdata, const
// cuda::DeviceModelData& d_model_data);  // MaxPool 3 void
// run_stage_9_async(cifar_dense::AppDataBatch& appdata, const cuda::DeviceModelData& d_model_data);
// // Linear

// using DispatchFnBatch = void (*)(cifar_dense::AppDataBatch&, const cuda::DeviceModelData&);

// const DispatchFnBatch dispatch_fns_batch[] = {
//     run_stage_1_async,
//     run_stage_2_async,
//     run_stage_3_async,
//     run_stage_4_async,
//     run_stage_5_async,
//     run_stage_6_async,
//     run_stage_7_async,
//     run_stage_8_async,
//     run_stage_9_async,
// };
// // clang-format on

// inline void dispatch_multi_stage(cifar_dense::AppDataBatch& appdata,
//                                  const cuda::DeviceModelData& d_model_data,
//                                  const int start_stage,
//                                  const int end_stage) {
//   assert(start_stage >= 1 && end_stage <= 9);

//   // // Sync to GPU
//   // CudaAttachSingle(appdata.input.raw());
//   // CudaAttachSingle(appdata.conv1_out.raw());
//   // CudaAttachSingle(appdata.pool1_out.raw());
//   // CudaAttachSingle(appdata.conv2_out.raw());
//   // CudaAttachSingle(appdata.pool2_out.raw());
//   // CudaAttachSingle(appdata.conv3_out.raw());
//   // CudaAttachSingle(appdata.conv4_out.raw());
//   // CudaAttachSingle(appdata.conv5_out.raw());
//   // CudaAttachSingle(appdata.pool3_out.raw());
//   // CudaAttachSingle(appdata.linear_out.raw());

//   for (int stage = start_stage; stage <= end_stage; stage++) {
//     dispatch_fns_batch[stage - 1](appdata, d_model_data);
//   }

//   // CheckCuda(cudaStreamSynchronize(mgr.get_stream()));
//   CheckCuda(cudaDeviceSynchronize());

//   // CudaAttachHost(appdata.input.raw());
//   // CudaAttachHost(appdata.conv1_out.raw());
//   // CudaAttachHost(appdata.pool1_out.raw());
//   // CudaAttachHost(appdata.conv2_out.raw());
//   // CudaAttachHost(appdata.pool2_out.raw());
//   // CudaAttachHost(appdata.conv3_out.raw());
//   // CudaAttachHost(appdata.conv4_out.raw());
//   // CudaAttachHost(appdata.conv5_out.raw());
//   // CudaAttachHost(appdata.pool3_out.raw());
//   // CudaAttachHost(appdata.linear_out.raw());
// }

}  // namespace cuda