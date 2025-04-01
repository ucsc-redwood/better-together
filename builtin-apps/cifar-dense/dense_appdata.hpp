#pragma once

#include <memory_resource>
#include <vector>

namespace cifar_dense {

//--------------------------------------
// Model Constants
//--------------------------------------

// clang-format off

// Input Image dimensions
constexpr int kInputChannels = 3;
constexpr int kInputHeight = 32;
constexpr int kInputWidth = 32;

// Convolution parameters
constexpr int kKernelSize = 3;
constexpr int kStride = 1;
constexpr int kPadding = 1;

// Pooling parameters
constexpr int kPoolSize = 2;
constexpr int kPoolStride = 2;

constexpr bool kRelu = true;

// Channels after each convolution layer
constexpr int kConv1OutChannels = 64;
constexpr int kConv2OutChannels = 192;
constexpr int kConv3OutChannels = 384;
constexpr int kConv4OutChannels = 256;
constexpr int kConv5OutChannels = 256;

//--------------------------------------
// Helper function to compute dimensions
//--------------------------------------
constexpr int CalcConvDim(const int input_size) {
  return (input_size + 2 * kPadding - kKernelSize) / kStride + 1;
}

constexpr int CalcPoolDim(const int input_size) {
  return (input_size - kPoolSize) / kPoolStride + 1;
}

//--------------------------------------
// Compute intermediate dimensions
//--------------------------------------

// After Conv1
constexpr int kConv1OutHeight = CalcConvDim(kInputHeight); // 32
constexpr int kConv1OutWidth  = CalcConvDim(kInputWidth);  // 32

// After Pool1
constexpr int kPool1OutHeight = CalcPoolDim(kConv1OutHeight); // 16
constexpr int kPool1OutWidth  = CalcPoolDim(kConv1OutWidth);  // 16

// After Conv2
constexpr int kConv2OutHeight = kPool1OutHeight; // 16
constexpr int kConv2OutWidth  = kPool1OutWidth;  // 16

// After Pool2
constexpr int kPool2OutHeight = CalcPoolDim(kConv2OutHeight); // 8
constexpr int kPool2OutWidth  = CalcPoolDim(kConv2OutWidth);  // 8

// After Conv3
constexpr int kConv3OutHeight = kPool2OutHeight; // 8
constexpr int kConv3OutWidth  = kPool2OutWidth;  // 8

// After Conv4
constexpr int kConv4OutHeight = kConv3OutHeight; // 8
constexpr int kConv4OutWidth  = kConv3OutWidth;  // 8

// After Conv5
constexpr int kConv5OutHeight = kConv4OutHeight; // 8
constexpr int kConv5OutWidth  = kConv4OutWidth;  // 8

// After Pool3
constexpr int kPool3OutHeight = CalcPoolDim(kConv5OutHeight); // 4
constexpr int kPool3OutWidth  = CalcPoolDim(kConv5OutWidth);  // 4

//--------------------------------------
// Fully connected layer
//--------------------------------------
constexpr int kLinearInFeatures = kConv5OutChannels * kPool3OutHeight * kPool3OutWidth; // 4096
constexpr int kLinearOutFeatures = 10;

//--------------------------------------
// Weight and bias sizes
//--------------------------------------
constexpr int kConv1WeightSize = kConv1OutChannels * kInputChannels * kKernelSize * kKernelSize; // 1728
constexpr int kConv1BiasSize = kConv1OutChannels; // 64

constexpr int kConv2WeightSize = kConv2OutChannels * kConv1OutChannels * kKernelSize * kKernelSize; // 110592
constexpr int kConv2BiasSize = kConv2OutChannels; // 192

constexpr int kConv3WeightSize = kConv3OutChannels * kConv2OutChannels * kKernelSize * kKernelSize; // 663552
constexpr int kConv3BiasSize = kConv3OutChannels; // 384

constexpr int kConv4WeightSize = kConv4OutChannels * kConv3OutChannels * kKernelSize * kKernelSize; // 884736
constexpr int kConv4BiasSize = kConv4OutChannels; // 256

constexpr int kConv5WeightSize = kConv5OutChannels * kConv4OutChannels * kKernelSize * kKernelSize; // 589824
constexpr int kConv5BiasSize = kConv5OutChannels; // 256

constexpr int kLinearWeightSize = kLinearOutFeatures * kLinearInFeatures; // 40960
constexpr int kLinearBiasSize = kLinearOutFeatures; // 10


//--------------------------------------
// Intermediate buffer sizes
//--------------------------------------
constexpr int kImageSize    = kInputChannels * kInputHeight * kInputWidth; // 3072
constexpr int kConv1OutSize = kConv1OutChannels * kConv1OutHeight * kConv1OutWidth; // 64*32*32=65536
constexpr int kPool1OutSize = kConv1OutChannels * kPool1OutHeight * kPool1OutWidth;  // 64*16*16=16384
constexpr int kConv2OutSize = kConv2OutChannels * kConv2OutHeight * kConv2OutWidth;  //192*16*16=49152
constexpr int kPool2OutSize = kConv2OutChannels * kPool2OutHeight * kPool2OutWidth;  //192*8*8=12288
constexpr int kConv3OutSize = kConv3OutChannels * kConv3OutHeight * kConv3OutWidth;  //384*8*8=24576
constexpr int kConv4OutSize = kConv4OutChannels * kConv4OutHeight * kConv4OutWidth;  //256*8*8=16384
constexpr int kConv5OutSize = kConv5OutChannels * kConv5OutHeight * kConv5OutWidth;  //256*8*8=16384
constexpr int kPool3OutSize = kConv5OutChannels * kPool3OutHeight * kPool3OutWidth;   //4096
constexpr int kLinearOutSize = kLinearOutFeatures; // 10

// clang-format on

// ----------------------------------------------------------------------------
// Application Data
// ----------------------------------------------------------------------------

struct AppData {
  explicit AppData(std::pmr::memory_resource* mr);

  // Input
  std::pmr::vector<float> u_image;

  // Conv1
  std::pmr::vector<float> u_conv1_weights;
  std::pmr::vector<float> u_conv1_bias;
  std::pmr::vector<float> u_conv1_out;

  // Pool1
  std::pmr::vector<float> u_pool1_out;

  // Conv2
  std::pmr::vector<float> u_conv2_weights;
  std::pmr::vector<float> u_conv2_bias;
  std::pmr::vector<float> u_conv2_out;

  // Pool2
  std::pmr::vector<float> u_pool2_out;

  // Conv3
  std::pmr::vector<float> u_conv3_weights;
  std::pmr::vector<float> u_conv3_bias;
  std::pmr::vector<float> u_conv3_out;

  // Conv4
  std::pmr::vector<float> u_conv4_weights;
  std::pmr::vector<float> u_conv4_bias;
  std::pmr::vector<float> u_conv4_out;

  // Conv5
  std::pmr::vector<float> u_conv5_weights;
  std::pmr::vector<float> u_conv5_bias;
  std::pmr::vector<float> u_conv5_out;

  // Pool3 (also used as flattened)
  std::pmr::vector<float> u_pool3_out;

  // Linear
  std::pmr::vector<float> u_linear_weights;
  std::pmr::vector<float> u_linear_bias;
  std::pmr::vector<float> u_linear_out;
};

}  // namespace cifar_dense
