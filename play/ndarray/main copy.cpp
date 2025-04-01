#include <string>

#include "ndarray.hpp"

// Function declarations using NDArray buffers

void conv2d_omp(const NDArray<3>& input,
                const NDArray<4>& weights,
                const NDArray<1>& bias,
                const int stride,
                const int padding,
                const bool relu,
                NDArray<3>& output,
                const int start,
                const int end) {}

void maxpool2d_omp(const NDArray<3>& input,
                   const int pool_size,
                   const int stride,
                   NDArray<3>& output,
                   const int start,
                   const int end) {}

void linear_omp(const NDArray<1>& input,
                const NDArray<2>& weights,
                const NDArray<1>& bias,
                NDArray<1>& output,
                const int start,
                const int end) {}

int main() {
  // Input: 3×32×32 CIFAR-10 image (channels, height, width)
  NDArray<3> input({3, 32, 32});
  input.print_shape("Input");

  // Conv1: 3 → 16 channels, spatial size remains 32×32
  NDArray<3> conv1_out({16, 32, 32});
  conv1_out.print_shape("Conv1 Output");

  // Pool1: 2×2 window with stride 2 → reduces spatial dimensions to 16×16
  NDArray<3> pool1_out({16, 16, 16});
  pool1_out.print_shape("Pool1 Output");

  // Conv2: 16 → 32 channels, spatial size remains 16×16
  NDArray<3> conv2_out({32, 16, 16});
  conv2_out.print_shape("Conv2 Output");

  // Pool2: 2×2 window with stride 2 → reduces spatial dimensions to 8×8
  NDArray<3> pool2_out({32, 8, 8});
  pool2_out.print_shape("Pool2 Output");

  // Conv3: 32 → 64 channels, spatial size remains 8×8
  NDArray<3> conv3_out({64, 8, 8});
  conv3_out.print_shape("Conv3 Output");

  // Conv4: 64 → 64 channels, spatial size remains 8×8
  NDArray<3> conv4_out({64, 8, 8});
  conv4_out.print_shape("Conv4 Output");

  // Conv5: 64 → 64 channels, spatial size remains 8×8
  NDArray<3> conv5_out({64, 8, 8});
  conv5_out.print_shape("Conv5 Output");

  // Pool3: 2×2 window with stride 2 → reduces spatial dimensions to 4×4
  NDArray<3> pool3_out({64, 4, 4});
  pool3_out.print_shape("Pool3 Output");

  // Linear layer: flatten the final tensor (64 channels × 4×4 = 1024 features)
  // and produce 10 outputs.
  NDArray<1> linear_out({10});
  linear_out.print_shape("Linear Output");

  // Weight and bias buffers for each layer

  // Conv1: 3 -> 16 channels, 3x3 kernel
  NDArray<4> conv1_weights({16, 3, 3, 3});
  conv1_weights.print_shape("Conv1 Weights");
  NDArray<1> conv1_bias({16});
  conv1_bias.print_shape("Conv1 Bias");

  // Conv2: 16 -> 32 channels, 3x3 kernel
  NDArray<4> conv2_weights({32, 16, 3, 3});
  conv2_weights.print_shape("Conv2 Weights");
  NDArray<1> conv2_bias({32});
  conv2_bias.print_shape("Conv2 Bias");

  // Conv3: 32 -> 64 channels, 3x3 kernel
  NDArray<4> conv3_weights({64, 32, 3, 3});
  conv3_weights.print_shape("Conv3 Weights");
  NDArray<1> conv3_bias({64});
  conv3_bias.print_shape("Conv3 Bias");

  // Conv4: 64 -> 64 channels, 3x3 kernel
  NDArray<4> conv4_weights({64, 64, 3, 3});
  conv4_weights.print_shape("Conv4 Weights");
  NDArray<1> conv4_bias({64});
  conv4_bias.print_shape("Conv4 Bias");

  // Conv5: 64 -> 64 channels, 3x3 kernel
  NDArray<4> conv5_weights({64, 64, 3, 3});
  conv5_weights.print_shape("Conv5 Weights");
  NDArray<1> conv5_bias({64});
  conv5_bias.print_shape("Conv5 Bias");

  // Linear layer: 1024 flattened features to 10 outputs
  NDArray<2> linear_weights({10, 1024});
  linear_weights.print_shape("Linear Weights");
  NDArray<1> linear_bias({10});
  linear_bias.print_shape("Linear Bias");

  // For demonstration, we use a dummy parallel region from start=0 to end=1.
  int start = 0, end = 1;

  conv2d_omp(input, conv1_weights, conv1_bias, 1, 1, true, conv1_out, start, end);

  maxpool2d_omp(conv1_out, 2, 2, pool1_out, start, end);

  conv2d_omp(pool1_out, conv2_weights, conv2_bias, 1, 1, true, conv2_out, start, end);

  maxpool2d_omp(conv2_out, 2, 2, pool2_out, start, end);

  conv2d_omp(pool2_out, conv3_weights, conv3_bias, 1, 1, true, conv3_out, start, end);

  conv2d_omp(conv3_out, conv4_weights, conv4_bias, 1, 1, true, conv4_out, start, end);

  conv2d_omp(conv4_out, conv5_weights, conv5_bias, 1, 1, true, conv5_out, start, end);

  maxpool2d_omp(conv5_out, 2, 2, pool3_out, start, end);

  // ---------------------------
  // Flatten operation
  // ---------------------------
  // pool3_out is of shape (64, 4, 4) = 1024 features.
  NDArray<1> flattened({64 * 4 * 4});
  {
    size_t idx = 0;
    for (size_t c = 0; c < 64; ++c) {
      for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
          flattened(idx) = pool3_out(c, i, j);
          ++idx;
        }
      }
    }
  }

  // ---------------------------
  // 9. Linear (fully-connected) layer: flattened -> linear_out
  linear_omp(flattened, linear_weights, linear_bias, linear_out, start, end);

  // Final output shape and (optionally) result inspection.
  linear_out.print_shape("Final Output");

  return 0;
}
