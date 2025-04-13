#include "dispatchers.hpp"

#include "builtin-apps/debug_logger.hpp"
#include "kernels.hpp"

namespace omp {

// ----------------------------------------------------------------------------
// Batched version
// ----------------------------------------------------------------------------

void run_stage_1(cifar_dense::AppDataBatch& appdata) {
  const auto& in_shape = appdata.u_input.shape();
  const auto& w_shape = appdata.u_conv1_w.shape();
  const auto& out_shape = appdata.u_conv1_out.shape();

  LOG_KERNEL(LogKernelType::kOMP, 1, &appdata);

  conv2d_batch_u(appdata.u_input.raw(),
                 appdata.u_conv1_w.raw(),
                 appdata.u_conv1_b.raw(),
                 appdata.u_conv1_out.raw(),
                 in_shape[0],
                 in_shape[1],
                 in_shape[2],
                 in_shape[3],
                 w_shape[0],
                 w_shape[2],
                 w_shape[3],
                 out_shape[2],
                 out_shape[3],
                 1,
                 0,
                 true);
}

void run_stage_2(cifar_dense::AppDataBatch& appdata) {
  const auto& in_shape = appdata.u_conv1_out.shape();
  const auto& out_shape = appdata.u_pool1_out.shape();

  LOG_KERNEL(LogKernelType::kOMP, 2, &appdata);

  maxpool2d_batch_u(appdata.u_conv1_out.raw(),
                    appdata.u_pool1_out.raw(),
                    in_shape[0],
                    in_shape[1],
                    in_shape[2],
                    in_shape[3],
                    out_shape[2],
                    out_shape[3],
                    2,
                    2);
}

void run_stage_3(cifar_dense::AppDataBatch& appdata) {
  const auto& in_shape = appdata.u_pool1_out.shape();
  const auto& w_shape = appdata.u_conv2_w.shape();
  const auto& out_shape = appdata.u_conv2_out.shape();

  LOG_KERNEL(LogKernelType::kOMP, 3, &appdata);

  conv2d_batch_u(appdata.u_pool1_out.raw(),
                 appdata.u_conv2_w.raw(),
                 appdata.u_conv2_b.raw(),
                 appdata.u_conv2_out.raw(),
                 in_shape[0],
                 in_shape[1],
                 in_shape[2],
                 in_shape[3],
                 w_shape[0],
                 w_shape[2],
                 w_shape[3],
                 out_shape[2],
                 out_shape[3],
                 1,
                 0,
                 true);
}

void run_stage_4(cifar_dense::AppDataBatch& appdata) {
  const auto& in_shape = appdata.u_conv2_out.shape();
  const auto& out_shape = appdata.u_pool2_out.shape();

  LOG_KERNEL(LogKernelType::kOMP, 4, &appdata);

  maxpool2d_batch_u(appdata.u_conv2_out.raw(),
                    appdata.u_pool2_out.raw(),
                    in_shape[0],
                    in_shape[1],
                    in_shape[2],
                    in_shape[3],
                    out_shape[2],
                    out_shape[3],
                    2,
                    2);
}

void run_stage_5(cifar_dense::AppDataBatch& appdata) {
  const auto& in_shape = appdata.u_pool2_out.shape();
  const auto& w_shape = appdata.u_conv3_w.shape();
  const auto& out_shape = appdata.u_conv3_out.shape();

  LOG_KERNEL(LogKernelType::kOMP, 5, &appdata);

  conv2d_batch_u(appdata.u_pool2_out.raw(),
                 appdata.u_conv3_w.raw(),
                 appdata.u_conv3_b.raw(),
                 appdata.u_conv3_out.raw(),
                 in_shape[0],
                 in_shape[1],
                 in_shape[2],
                 in_shape[3],
                 w_shape[0],
                 w_shape[2],
                 w_shape[3],
                 out_shape[2],
                 out_shape[3],
                 1,
                 0,
                 true);
}

void run_stage_6(cifar_dense::AppDataBatch& appdata) {
  const auto& in_shape = appdata.u_conv3_out.shape();
  const auto& w_shape = appdata.u_conv4_w.shape();
  const auto& out_shape = appdata.u_conv4_out.shape();

  LOG_KERNEL(LogKernelType::kOMP, 6, &appdata);

  conv2d_batch_u(appdata.u_conv3_out.raw(),
                 appdata.u_conv4_w.raw(),
                 appdata.u_conv4_b.raw(),
                 appdata.u_conv4_out.raw(),
                 in_shape[0],
                 in_shape[1],
                 in_shape[2],
                 in_shape[3],
                 w_shape[0],
                 w_shape[2],
                 w_shape[3],
                 out_shape[2],
                 out_shape[3],
                 1,
                 0,
                 true);
}

void run_stage_7(cifar_dense::AppDataBatch& appdata) {
  const auto& in_shape = appdata.u_conv4_out.shape();
  const auto& w_shape = appdata.u_conv5_w.shape();
  const auto& out_shape = appdata.u_conv5_out.shape();

  LOG_KERNEL(LogKernelType::kOMP, 7, &appdata);

  conv2d_batch_u(appdata.u_conv4_out.raw(),
                 appdata.u_conv5_w.raw(),
                 appdata.u_conv5_b.raw(),
                 appdata.u_conv5_out.raw(),
                 in_shape[0],
                 in_shape[1],
                 in_shape[2],
                 in_shape[3],
                 w_shape[0],
                 w_shape[2],
                 w_shape[3],
                 out_shape[2],
                 out_shape[3],
                 1,
                 0,
                 true);
}

void run_stage_8(cifar_dense::AppDataBatch& appdata) {
  const auto& in_shape = appdata.u_conv5_out.shape();
  const auto& out_shape = appdata.u_pool3_out.shape();

  LOG_KERNEL(LogKernelType::kOMP, 8, &appdata);

  maxpool2d_batch_u(appdata.u_conv5_out.raw(),
                    appdata.u_pool3_out.raw(),
                    in_shape[0],
                    in_shape[1],
                    in_shape[2],
                    in_shape[3],
                    out_shape[2],
                    out_shape[3],
                    2,
                    2);
}

inline NDArray<2> flatten_batch(const NDArray<4>& input) {
  // input shape: (N, C, H, W)
  const auto& s = input.shape();
  int N = static_cast<int>(s[0]);
  int C = static_cast<int>(s[1]);
  int H = static_cast<int>(s[2]);
  int W = static_cast<int>(s[3]);
  int new_features = C * H * W;  // e.g. 64*4*4 = 1024

  NDArray<2> output({static_cast<size_t>(N), static_cast<size_t>(new_features)});

#pragma omp for
  for (int n = 0; n < N; n++) {
    int idx = 0;
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          output(n, idx++) = input(n, c, h, w);
        }
      }
    }
  }
  return output;
}

void run_stage_9(cifar_dense::AppDataBatch& appdata) {
  // Instead of using the direct NDArray API, use the _u suffix function
  // First create the flattened representation
  NDArray<2> flattened = flatten_batch(appdata.u_pool3_out);

  // Get the necessary dimensions
  const auto& in_shape = flattened.shape();
  const auto& w_shape = appdata.u_linear_w.shape();

  int N = static_cast<int>(in_shape[0]);            // Batch size
  int in_features = static_cast<int>(in_shape[1]);  // Input features (1024)
  int out_features = static_cast<int>(w_shape[0]);  // Output features (10)

  LOG_KERNEL(LogKernelType::kOMP, 9, &appdata);

  linear_batch_u(flattened.raw(),
                 appdata.u_linear_w.raw(),
                 appdata.u_linear_b.raw(),
                 appdata.u_linear_out.raw(),
                 N,
                 in_features,
                 out_features);
}

}  // namespace omp