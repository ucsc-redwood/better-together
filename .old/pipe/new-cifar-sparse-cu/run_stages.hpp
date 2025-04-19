// #pragma once

// #include "builtin-apps/affinity.hpp"
// #include "builtin-apps/app.hpp"
// #include "builtin-apps/cifar-sparse/cuda/dispatchers.cuh"
// #include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
// #include "builtin-apps/common/cuda/manager.cuh"

// template <int N>
// concept AllowedStage = (N >= 1 && N <= 9);

// // ---------------------------------------------------------------------
// // CPU stages
// // ---------------------------------------------------------------------

// namespace omp {

// constexpr std::array<void (*)(cifar_sparse::AppData &), 9> cpu_stages = {
//     cifar_sparse::omp::process_stage_1,
//     cifar_sparse::omp::process_stage_2,
//     cifar_sparse::omp::process_stage_3,
//     cifar_sparse::omp::process_stage_4,
//     cifar_sparse::omp::process_stage_5,
//     cifar_sparse::omp::process_stage_6,
//     cifar_sparse::omp::process_stage_7,
//     cifar_sparse::omp::process_stage_8,
//     cifar_sparse::omp::process_stage_9,
// };

// template <int Start, int End, ProcessorType PT, int NThreads>
//   requires AllowedStage<Start> && AllowedStage<End> && (Start <= End)
// void run_multiple_stages(cifar_sparse::AppData &data, cuda::CudaManager &) {
// #pragma omp parallel num_threads(NThreads)
//   {
//     // Bind to core
//     if constexpr (PT == ProcessorType::kLittleCore) {
//       bind_thread_to_cores(g_lit_cores);
//     } else if constexpr (PT == ProcessorType::kMediumCore) {
//       bind_thread_to_cores(g_med_cores);
//     } else if constexpr (PT == ProcessorType::kBigCore) {
//       bind_thread_to_cores(g_big_cores);
//     }

// #pragma unroll
//     for (int s = Start; s <= End; ++s) {
//       cpu_stages[s - 1](data);
//     }
//   }
// }

// }  // namespace omp

// // ---------------------------------------------------------------------
// // GPU stages
// // ---------------------------------------------------------------------

// namespace cuda {

// constexpr std::array<void (*)(cifar_sparse::AppData &), 9> gpu_stages = {
//     cifar_sparse::cuda::process_stage_1,
//     cifar_sparse::cuda::process_stage_2,
//     cifar_sparse::cuda::process_stage_3,
//     cifar_sparse::cuda::process_stage_4,
//     cifar_sparse::cuda::process_stage_5,
//     cifar_sparse::cuda::process_stage_6,
//     cifar_sparse::cuda::process_stage_7,
//     cifar_sparse::cuda::process_stage_8,
//     cifar_sparse::cuda::process_stage_9,
// };

// #define CudaAttachSingle(ptr) \
//   (cudaStreamAttachMemAsync(mgr.get_stream(), ptr, 0, cudaMemAttachSingle))
// #define CudaAttachHost(ptr) (cudaStreamAttachMemAsync(mgr.get_stream(), ptr, 0, cudaMemAttachHost))

// template <int Start, int End>
//   requires AllowedStage<Start> && AllowedStage<End> && (Start <= End)
// void run_multiple_stages(cifar_sparse::AppData &data, cuda::CudaManager &mgr) {
//   CudaAttachSingle(data.u_conv1_bias.data());
//   CudaAttachSingle(data.u_conv1_values.data());
//   CudaAttachSingle(data.u_conv1_row_ptr.data());
//   CudaAttachSingle(data.u_conv1_col_idx.data());
//   CudaAttachSingle(data.u_conv1_output.data());
//   CudaAttachSingle(data.u_pool1_output.data());

//   CudaAttachSingle(data.u_conv2_bias.data());
//   CudaAttachSingle(data.u_conv2_values.data());
//   CudaAttachSingle(data.u_conv2_row_ptr.data());
//   CudaAttachSingle(data.u_conv2_col_idx.data());
//   CudaAttachSingle(data.u_conv2_output.data());
//   CudaAttachSingle(data.u_pool2_output.data());

//   CudaAttachSingle(data.u_conv3_bias.data());
//   CudaAttachSingle(data.u_conv3_values.data());
//   CudaAttachSingle(data.u_conv3_row_ptr.data());
//   CudaAttachSingle(data.u_conv3_col_idx.data());
//   CudaAttachSingle(data.u_conv3_output.data());

//   CudaAttachSingle(data.u_conv4_bias.data());
//   CudaAttachSingle(data.u_conv4_values.data());
//   CudaAttachSingle(data.u_conv4_row_ptr.data());
//   CudaAttachSingle(data.u_conv4_col_idx.data());
//   CudaAttachSingle(data.u_conv4_output.data());

//   CudaAttachSingle(data.u_conv5_bias.data());
//   CudaAttachSingle(data.u_conv5_values.data());
//   CudaAttachSingle(data.u_conv5_row_ptr.data());
//   CudaAttachSingle(data.u_conv5_col_idx.data());
//   CudaAttachSingle(data.u_conv5_output.data());
//   CudaAttachSingle(data.u_pool3_output.data());

//   CudaAttachSingle(data.u_linear_bias.data());
//   CudaAttachSingle(data.u_linear_values.data());
//   CudaAttachSingle(data.u_linear_row_ptr.data());
//   CudaAttachSingle(data.u_linear_col_idx.data());
//   CudaAttachSingle(data.u_linear_output.data());

//   for (int s = Start; s <= End; ++s) {
//     gpu_stages[s - 1](data);
//   }

//   CheckCuda(cudaStreamSynchronize(mgr.get_stream()));

//   CudaAttachHost(data.u_conv1_bias.data());
//   CudaAttachHost(data.u_conv1_values.data());
//   CudaAttachHost(data.u_conv1_row_ptr.data());
//   CudaAttachHost(data.u_conv1_col_idx.data());
//   CudaAttachHost(data.u_conv1_output.data());
//   CudaAttachHost(data.u_pool1_output.data());

//   CudaAttachHost(data.u_conv2_bias.data());
//   CudaAttachHost(data.u_conv2_values.data());
//   CudaAttachHost(data.u_conv2_row_ptr.data());
//   CudaAttachHost(data.u_conv2_col_idx.data());
//   CudaAttachHost(data.u_conv2_output.data());
//   CudaAttachHost(data.u_pool2_output.data());

//   CudaAttachHost(data.u_conv3_bias.data());
//   CudaAttachHost(data.u_conv3_values.data());
//   CudaAttachHost(data.u_conv3_row_ptr.data());
//   CudaAttachHost(data.u_conv3_col_idx.data());
//   CudaAttachHost(data.u_conv3_output.data());

//   CudaAttachHost(data.u_conv4_bias.data());
//   CudaAttachHost(data.u_conv4_values.data());
//   CudaAttachHost(data.u_conv4_row_ptr.data());
//   CudaAttachHost(data.u_conv4_col_idx.data());
//   CudaAttachHost(data.u_conv4_output.data());

//   CudaAttachHost(data.u_conv5_bias.data());
//   CudaAttachHost(data.u_conv5_values.data());
//   CudaAttachHost(data.u_conv5_row_ptr.data());
//   CudaAttachHost(data.u_conv5_col_idx.data());
//   CudaAttachHost(data.u_conv5_output.data());
//   CudaAttachHost(data.u_pool3_output.data());

//   CudaAttachHost(data.u_linear_bias.data());
//   CudaAttachHost(data.u_linear_values.data());
//   CudaAttachHost(data.u_linear_row_ptr.data());
//   CudaAttachHost(data.u_linear_col_idx.data());
//   CudaAttachHost(data.u_linear_output.data());
// }

// }  // namespace cuda
