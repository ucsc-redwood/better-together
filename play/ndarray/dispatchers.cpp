#include "dispatchers.hpp"

#include "kernels.hpp"

namespace omp {

void run_stage_1(cifar_dense::AppData& appdata) {
  conv2d(appdata.input, appdata.conv1_weights, appdata.conv1_bias, 1, 0, true, appdata.conv1_out);
}

void run_stage_2(cifar_dense::AppData& appdata) {
  maxpool2d(appdata.conv1_out, 2, 2, appdata.pool1_out);
}

void run_stage_3(cifar_dense::AppData& appdata) {
  conv2d(
      appdata.pool1_out, appdata.conv2_weights, appdata.conv2_bias, 1, 0, true, appdata.conv2_out);
}

void run_stage_4(cifar_dense::AppData& appdata) {
  maxpool2d(appdata.conv2_out, 2, 2, appdata.pool2_out);
}

void run_stage_5(cifar_dense::AppData& appdata) {
  conv2d(
      appdata.pool2_out, appdata.conv3_weights, appdata.conv3_bias, 1, 0, true, appdata.conv3_out);
}

void run_stage_6(cifar_dense::AppData& appdata) {
  conv2d(
      appdata.conv3_out, appdata.conv4_weights, appdata.conv4_bias, 1, 0, true, appdata.conv4_out);
}

void run_stage_7(cifar_dense::AppData& appdata) {
  conv2d(
      appdata.conv4_out, appdata.conv5_weights, appdata.conv5_bias, 1, 0, true, appdata.conv5_out);
}

void run_stage_8(cifar_dense::AppData& appdata) {
  maxpool2d(appdata.conv5_out, 2, 2, appdata.pool3_out);
}

void run_stage_9(cifar_dense::AppData& appdata) {
  linear(
      appdata.pool3_out.flatten(), appdata.linear_weights, appdata.linear_bias, appdata.linear_out);
}

}  // namespace omp