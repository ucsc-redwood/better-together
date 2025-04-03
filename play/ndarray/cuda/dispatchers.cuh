#pragma once

#include "../appdata.hpp"
#include "builtin-apps/common/cuda/helpers.cuh"
#include "builtin-apps/common/cuda/manager.cuh"
#include "model_data.cuh"

namespace cuda {

class CudaDispatcher final : public cuda::CudaManager {
 public:
  explicit CudaDispatcher() : d_model_data_(cifar_dense::AppDataBatch::get_model()) {}

  void run_stage_1_async(cifar_dense::AppDataBatch& appdata);  // Conv 1
  void run_stage_2_async(cifar_dense::AppDataBatch& appdata);  // MaxPool 1
  void run_stage_3_async(cifar_dense::AppDataBatch& appdata);  // Conv 2
  void run_stage_4_async(cifar_dense::AppDataBatch& appdata);  // MaxPool 2
  void run_stage_5_async(cifar_dense::AppDataBatch& appdata);  // Conv 3
  void run_stage_6_async(cifar_dense::AppDataBatch& appdata);  // Conv 4
  void run_stage_7_async(cifar_dense::AppDataBatch& appdata);  // Conv 5
  void run_stage_8_async(cifar_dense::AppDataBatch& appdata);  // MaxPool 3
  void run_stage_9_async(cifar_dense::AppDataBatch& appdata);  // Linear

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

#define CudaAttachSingle(ptr) (cudaStreamAttachMemAsync(mgr_.get_stream(), ptr, 0, cudaMemAttachSingle))
#define CudaAttachHost(ptr) (cudaStreamAttachMemAsync(mgr_.get_stream(), ptr, 0, cudaMemAttachHost))

  void dispatch_multi_stage(cifar_dense::AppDataBatch& data,
                            const int start_stage,
                            const int end_stage) {
    if (start_stage < 1 || end_stage > 9) throw std::out_of_range("Invalid stage");

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

    for (int stage = start_stage; stage <= end_stage; stage++) {
      (this->*stage_functions[stage - 1])(data);
    }

    CheckCuda(cudaStreamSynchronize(mgr_.get_stream()));

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

 private:
  // create a manager
  CudaManager mgr_;

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