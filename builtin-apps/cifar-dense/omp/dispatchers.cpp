#include "dispatchers.hpp"

#include <omp.h>
#include <spdlog/spdlog.h>

#include "../../debug_logger.hpp"
#include "all_kernels.hpp"

namespace cifar_dense::omp {

// ----------------------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------------------

void process_stage_1(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv1OutChannels * kConv1OutHeight * kConv1OutWidth;

  const int start = 0;
  const int end = total_iterations;

  LOG_KERNEL(LogKernelType::kOMP, 1, &app_data);

  conv2d_omp(app_data.u_image.data(),
             kInputChannels,  // image_input_channels
             kInputHeight,
             kInputWidth,
             app_data.u_conv1_weights.data(),
             kConv1OutChannels,
             kInputChannels,
             kKernelSize,  // weight_height
             kKernelSize,  // weight_width
             app_data.u_conv1_bias.data(),
             kConv1BiasSize,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv1_out.data(),
             start,
             end);
}

// ----------------------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------------------

void process_stage_2(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv1OutChannels * kPool1OutHeight * kPool1OutWidth;

  const int start = 0;
  const int end = total_iterations;

  LOG_KERNEL(LogKernelType::kOMP, 2, &app_data);

  maxpool2d_omp(app_data.u_conv1_out.data(),
                kConv1OutChannels,
                kConv1OutHeight,
                kConv1OutWidth,
                kPoolSize,
                kPoolStride,
                app_data.u_pool1_out.data(),
                start,
                end);
}

// ----------------------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------------------

void process_stage_3(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv2OutChannels * kConv2OutHeight * kConv2OutWidth;

  const int start = 0;
  const int end = total_iterations;

  LOG_KERNEL(LogKernelType::kOMP, 3, &app_data);

  conv2d_omp(app_data.u_pool1_out.data(),
             kConv1OutChannels,
             kPool1OutHeight,
             kPool1OutWidth,
             app_data.u_conv2_weights.data(),
             kConv2OutChannels,
             kConv1OutChannels,
             kKernelSize,  // weight_height
             kKernelSize,  // weight_width
             app_data.u_conv2_bias.data(),
             kConv2BiasSize,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv2_out.data(),
             start,
             end);
}

// ----------------------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------------------

void process_stage_4(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv2OutChannels * kPool2OutHeight * kPool2OutWidth;

  const int start = 0;
  const int end = total_iterations;

  LOG_KERNEL(LogKernelType::kOMP, 4, &app_data);

  maxpool2d_omp(app_data.u_conv2_out.data(),
                kConv2OutChannels,
                kConv2OutHeight,
                kConv2OutWidth,
                kPoolSize,
                kPoolStride,
                app_data.u_pool2_out.data(),
                start,
                end);
}

// ----------------------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------------------

void process_stage_5(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv3OutChannels * kConv3OutHeight * kConv3OutWidth;

  const int start = 0;
  const int end = total_iterations;

  LOG_KERNEL(LogKernelType::kOMP, 5, &app_data);

  conv2d_omp(app_data.u_pool2_out.data(),
             kConv2OutChannels,
             kPool2OutHeight,
             kPool2OutWidth,
             app_data.u_conv3_weights.data(),
             kConv3OutChannels,
             kConv2OutChannels,
             kKernelSize,
             kKernelSize,
             app_data.u_conv3_bias.data(),
             kConv3BiasSize,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv3_out.data(),
             start,
             end);
}

// ----------------------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------------------

void process_stage_6(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv4OutChannels * kConv4OutHeight * kConv4OutWidth;

  const int start = 0;
  const int end = total_iterations;

  LOG_KERNEL(LogKernelType::kOMP, 6, &app_data);

  conv2d_omp(app_data.u_conv3_out.data(),
             kConv3OutChannels,
             kConv3OutHeight,
             kConv3OutWidth,
             app_data.u_conv4_weights.data(),
             kConv4OutChannels,
             kConv3OutChannels,
             kKernelSize,
             kKernelSize,
             app_data.u_conv4_bias.data(),
             kConv4BiasSize,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv4_out.data(),
             start,
             end);
}

// ----------------------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------------------

void process_stage_7(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv5OutChannels * kConv5OutHeight * kConv5OutWidth;

  const int start = 0;
  const int end = total_iterations;

  LOG_KERNEL(LogKernelType::kOMP, 7, &app_data);

  conv2d_omp(app_data.u_conv4_out.data(),
             kConv4OutChannels,
             kConv4OutHeight,
             kConv4OutWidth,
             app_data.u_conv5_weights.data(),
             kConv5OutChannels,
             kConv4OutChannels,
             kKernelSize,
             kKernelSize,
             app_data.u_conv5_bias.data(),
             kConv5BiasSize,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv5_out.data(),
             start,
             end);
}

// ----------------------------------------------------------------------------
// Stage 8
// ----------------------------------------------------------------------------

void process_stage_8(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv5OutChannels * kPool3OutHeight * kPool3OutWidth;

  const int start = 0;
  const int end = total_iterations;

  LOG_KERNEL(LogKernelType::kOMP, 8, &app_data);

  maxpool2d_omp(app_data.u_conv5_out.data(),
                kConv5OutChannels,
                kConv5OutHeight,
                kConv5OutWidth,
                kPoolSize,
                kPoolStride,
                app_data.u_pool3_out.data(),
                start,
                end);
}

// ----------------------------------------------------------------------------
// Stage 9
// ----------------------------------------------------------------------------

void process_stage_9(cifar_dense::AppData &app_data) {
  const int total_iterations = kLinearOutFeatures;

  const int start = 0;
  const int end = total_iterations;

  LOG_KERNEL(LogKernelType::kOMP, 9, &app_data);

  linear_omp(app_data.u_pool3_out.data(),
             app_data.u_linear_weights.data(),
             app_data.u_linear_bias.data(),
             app_data.u_linear_out.data(),
             kLinearInFeatures,
             kLinearOutFeatures,
             start,
             end);
}

}  // namespace cifar_dense::omp
