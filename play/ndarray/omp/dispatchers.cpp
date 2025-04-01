#include "dispatchers.hpp"

#include "kernels.hpp"

namespace omp {

// void run_stage_1(cifar_dense::AppData& appdata) {
//   // conv2d(appdata.input, appdata.conv1_weights, appdata.conv1_bias, 1, 0, true,
//   // appdata.conv1_out);

//   // inline void conv2d_batch_u(float* u_input,
//   //                            float* u_weights,
//   //                            float* u_bias,
//   //                            float* u_output,
//   //                            const int N,     // in_shape[0]
//   //                            const int inC,   // in_shape[1]
//   //                            const int inH,   // in_shape[2]
//   //                            const int inW,   // in_shape[3]
//   //                            const int outC,  // w_shape[0]
//   //                            const int kH,    // w_shape[2]
//   //                            const int kW,    // w_shape[3]
//   //                            const int outH,  // out_shape[2]
//   //                            const int outW,  // out_shape[3]
//   //                            const int stride,
//   //                            const int padding,
//   //                            const bool relu) ;

//   const auto& in_shape = appdata.input.shape();
//   const auto& w_shape = appdata.conv1_w.shape();
//   const auto& out_shape = appdata.conv1_out.shape();

//   conv2d_batch_u(appdata.input.raw(),
//                  appdata.conv1_w.raw(),
//                  appdata.conv1_b.raw(),
//                  appdata.conv1_out.raw(),
//                  in_shape[0],
//                  in_shape[1],
//                  in_shape[2],
//                  in_shape[3],
//                  w_shape[0],
//                  w_shape[2],
//                  w_shape[3],
//                  out_shape[2],
//                  out_shape[3],
//                  1,
//                  0,
//                  true);
// }

// void run_stage_2(cifar_dense::AppData& appdata) {
//   // maxpool2d(appdata.conv1_out, 2, 2, appdata.pool1_out);

//   // inline void maxpool2d_batch_u(const float* u_input,
//   //                               float* u_output,
//   //                               const int N,     // in_shape[0]
//   //                               const int C,     // in_shape[1]
//   //                               const int inH,   // in_shape[2]
//   //                               const int inW,   // in_shape[3]
//   //                               const int outH,  // out_shape[2]
//   //                               const int outW,  // out_shape[3]
//   //                               const int pool_size,
//   //                               const int stride) {

//   const auto& in_shape = appdata.conv1_out.shape();
//   const auto& out_shape = appdata.pool1_out.shape();

//   maxpool2d_batch_u(appdata.conv1_out.raw(),
//                     appdata.pool1_out.raw(),
//                     in_shape[0],
//                     in_shape[1],
//                     in_shape[2],
//                     in_shape[3],
//                     out_shape[2],
//                     out_shape[3],
//                     2,
//                     2);
// }

// void run_stage_3(cifar_dense::AppData& appdata) {
//   // conv2d(
//   // appdata.pool1_out, appdata.conv2_weights, appdata.conv2_bias, 1, 0, true, appdata.conv2_out);

//   const auto& in_shape = appdata.pool1_out.shape();
//   const auto& w_shape = appdata.conv2_w.shape();
//   const auto& out_shape = appdata.conv2_out.shape();

//   conv2d_batch_u(appdata.pool1_out.raw(),
//                  appdata.conv2_w.raw(),
//                  appdata.conv2_b.raw(),
//                  appdata.conv2_out.raw(),
//                  in_shape[0],
//                  in_shape[1],
//                  in_shape[2],
//                  in_shape[3],
//                  w_shape[0],
//                  w_shape[2],
//                  w_shape[3],
//                  out_shape[2],
//                  out_shape[3],
//                  1,
//                  0,
//                  true);
// }

// void run_stage_4(cifar_dense::AppData& appdata) {
//   // maxpool2d(appdata.conv2_out, 2, 2, appdata.pool2_out);

//   const auto& in_shape = appdata.conv2_out.shape();
//   const auto& out_shape = appdata.pool2_out.shape();

//   maxpool2d_batch_u(appdata.conv2_out.raw(),
//                     appdata.pool2_out.raw(),
//                     in_shape[0],
//                     in_shape[1],
//                     in_shape[2],
//                     in_shape[3],
//                     out_shape[2],
//                     out_shape[3],
//                     2,
//                     2);
// }

// void run_stage_5(cifar_dense::AppData& appdata) {
//   // conv2d(
//   // appdata.pool2_out, appdata.conv3_weights, appdata.conv3_bias, 1, 0, true, appdata.conv3_out);

//   const auto& in_shape = appdata.pool2_out.shape();
//   const auto& w_shape = appdata.conv3_w.shape();
//   const auto& out_shape = appdata.conv3_out.shape();

//   conv2d_batch_u(appdata.pool2_out.raw(),
//                  appdata.conv3_w.raw(),
//                  appdata.conv3_b.raw(),
//                  appdata.conv3_out.raw(),
//                  in_shape[0],
//                  in_shape[1],
//                  in_shape[2],
//                  in_shape[3],
//                  w_shape[0],
//                  w_shape[2],
//                  w_shape[3],
//                  out_shape[2],
//                  out_shape[3],
//                  1,
//                  0,
//                  true);
// }

// void run_stage_6(cifar_dense::AppData& appdata) {
//   // conv2d(
//   // appdata.conv3_out, appdata.conv4_weights, appdata.conv4_bias, 1, 0, true, appdata.conv4_out);

//   const auto& in_shape = appdata.conv3_out.shape();
//   const auto& w_shape = appdata.conv4_w.shape();
//   const auto& out_shape = appdata.conv4_out.shape();

//   conv2d_batch_u(appdata.conv3_out.raw(),
//                  appdata.conv4_w.raw(),
//                  appdata.conv4_b.raw(),
//                  appdata.conv4_out.raw(),
//                  in_shape[0],
//                  in_shape[1],
//                  in_shape[2],
//                  in_shape[3],
//                  w_shape[0],
//                  w_shape[2],
//                  w_shape[3],
//                  out_shape[2],
//                  out_shape[3],
//                  1,
//                  0,
//                  true);
// }

// void run_stage_7(cifar_dense::AppData& appdata) {
//   // conv2d(appdata.conv4_out, appdata.conv5_weights, appdata.conv5_bias, 1, 0, true,
//   // appdata.conv5_out);

//   const auto& in_shape = appdata.conv4_out.shape();
//   const auto& w_shape = appdata.conv5_w.shape();
//   const auto& out_shape = appdata.conv5_out.shape();

//   conv2d_batch_u(appdata.conv4_out.raw(),
//                  appdata.conv5_w.raw(),
//                  appdata.conv5_b.raw(),
//                  appdata.conv5_out.raw(),
//                  in_shape[0],
//                  in_shape[1],
//                  in_shape[2],
//                  in_shape[3],
//                  w_shape[0],
//                  w_shape[2],
//                  w_shape[3],
//                  out_shape[2],
//                  out_shape[3],
//                  1,
//                  0,
//                  true);
// }

// void run_stage_8(cifar_dense::AppData& appdata) {
//   // maxpool2d(appdata.conv5_out, 2, 2, appdata.pool3_out);

//   const auto& in_shape = appdata.conv5_out.shape();
//   const auto& out_shape = appdata.pool3_out.shape();

//   maxpool2d_batch_u(appdata.conv5_out.raw(),
//                     appdata.pool3_out.raw(),
//                     in_shape[0],
//                     in_shape[1],
//                     in_shape[2],
//                     in_shape[3],
//                     out_shape[2],
//                     out_shape[3],
//                     2,
//                     2);
// }

// void run_stage_9(cifar_dense::AppData& appdata) {
//   // linear(appdata.pool3_out.flatten(), appdata.linear_weights, appdata.linear_bias,
//   // appdata.linear_out);

//   // Use an implementation using the raw pointers following the pattern of other functions
//   // First, flatten the pool3_out tensor
//   NDArray<1> flattened = appdata.pool3_out.flatten();

//   // Get the necessary dimensions for the linear operation
//   const auto& in_shape = flattened.shape();
//   const auto& w_shape = appdata.linear_w.shape();

//   int in_features = static_cast<int>(in_shape[0]);  // Should be 1024 (64*4*4)
//   int out_features = static_cast<int>(w_shape[0]);  // Should be 10

//   // Call linear_batch_u with a batch size of 1
//   linear_batch_u(flattened.raw(),
//                  appdata.linear_w.raw(),
//                  appdata.linear_b.raw(),
//                  appdata.linear_out.raw(),
//                  1,              // N = 1 for single sample
//                  in_features,    // Input features
//                  out_features);  // Output features
// }

// ----------------------------------------------------------------------------
// Batched version
// ----------------------------------------------------------------------------

void run_stage_1(cifar_dense::AppDataBatch& appdata) {
  // Use the "u" suffix functions for consistency
  const auto& in_shape = appdata.input.shape();
  const auto& w_shape = appdata.conv1_w.shape();
  const auto& out_shape = appdata.conv1_out.shape();

  conv2d_batch_u(appdata.input.raw(),
                 appdata.conv1_w.raw(),
                 appdata.conv1_b.raw(),
                 appdata.conv1_out.raw(),
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
  const auto& in_shape = appdata.conv1_out.shape();
  const auto& out_shape = appdata.pool1_out.shape();

  maxpool2d_batch_u(appdata.conv1_out.raw(),
                    appdata.pool1_out.raw(),
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
  const auto& in_shape = appdata.pool1_out.shape();
  const auto& w_shape = appdata.conv2_w.shape();
  const auto& out_shape = appdata.conv2_out.shape();

  conv2d_batch_u(appdata.pool1_out.raw(),
                 appdata.conv2_w.raw(),
                 appdata.conv2_b.raw(),
                 appdata.conv2_out.raw(),
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
  const auto& in_shape = appdata.conv2_out.shape();
  const auto& out_shape = appdata.pool2_out.shape();

  maxpool2d_batch_u(appdata.conv2_out.raw(),
                    appdata.pool2_out.raw(),
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
  const auto& in_shape = appdata.pool2_out.shape();
  const auto& w_shape = appdata.conv3_w.shape();
  const auto& out_shape = appdata.conv3_out.shape();

  conv2d_batch_u(appdata.pool2_out.raw(),
                 appdata.conv3_w.raw(),
                 appdata.conv3_b.raw(),
                 appdata.conv3_out.raw(),
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
  const auto& in_shape = appdata.conv3_out.shape();
  const auto& w_shape = appdata.conv4_w.shape();
  const auto& out_shape = appdata.conv4_out.shape();

  conv2d_batch_u(appdata.conv3_out.raw(),
                 appdata.conv4_w.raw(),
                 appdata.conv4_b.raw(),
                 appdata.conv4_out.raw(),
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
  const auto& in_shape = appdata.conv4_out.shape();
  const auto& w_shape = appdata.conv5_w.shape();
  const auto& out_shape = appdata.conv5_out.shape();

  conv2d_batch_u(appdata.conv4_out.raw(),
                 appdata.conv5_w.raw(),
                 appdata.conv5_b.raw(),
                 appdata.conv5_out.raw(),
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
  const auto& in_shape = appdata.conv5_out.shape();
  const auto& out_shape = appdata.pool3_out.shape();

  maxpool2d_batch_u(appdata.conv5_out.raw(),
                    appdata.pool3_out.raw(),
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
  NDArray<2> flattened = flatten_batch(appdata.pool3_out);

  // Get the necessary dimensions
  const auto& in_shape = flattened.shape();
  const auto& w_shape = appdata.linear_w.shape();

  int N = static_cast<int>(in_shape[0]);            // Batch size
  int in_features = static_cast<int>(in_shape[1]);  // Input features (1024)
  int out_features = static_cast<int>(w_shape[0]);  // Output features (10)

  linear_batch_u(flattened.raw(),
                 appdata.linear_w.raw(),
                 appdata.linear_b.raw(),
                 appdata.linear_out.raw(),
                 N,
                 in_features,
                 out_features);
}

}  // namespace omp