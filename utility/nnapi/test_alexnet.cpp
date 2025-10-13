#include <android/NeuralNetworks.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>

void check(int status, const char* msg) {
  if (status != ANEURALNETWORKS_NO_ERROR) {
    std::cerr << msg << " failed: " << status << std::endl;
    exit(1);
  }
}

template <typename T>
[[nodiscard]]
constexpr auto getBytes(std::vector<T>& data) {
  return data.size() * sizeof(T);
}

// Simple stage setup functions
uint32_t setup_stage1_conv2d(ANeuralNetworksModel* model, uint32_t inputIndex, 
                             const uint8_t* filterData, size_t filterSize,
                             const int32_t* biasData, size_t biasSize) {
  // First Conv2D: [128, 32, 32, 3] -> [128, 32, 32, 16]
  constexpr uint32_t filterDims[] = {16, 3, 3, 3};
  constexpr uint32_t biasDims[] = {16};
  constexpr uint32_t outputDims[] = {128, 32, 32, 16};
  
  const ANeuralNetworksOperandType filterType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(filterDims),
                                              .dimensions = filterDims,
                                              .scale = 0.01f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType biasType{.type = ANEURALNETWORKS_TENSOR_INT32,
                                            .dimensionCount = std::size(biasDims),
                                            .dimensions = biasDims,
                                            .scale = 0.001f,
                                            .zeroPoint = 0};
  const ANeuralNetworksOperandType outputType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(outputDims),
                                              .dimensions = outputDims,
                                              .scale = 0.1f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType paramType{.type = ANEURALNETWORKS_INT32,
                                             .dimensionCount = 0,
                                             .dimensions = nullptr,
                                             .scale = 0.0f,
                                             .zeroPoint = 0};

  uint32_t filterIndex = 1, biasIndex = 2, paddingIndex = 3, strideIndex = 4, fuseIndex = 5, outputIndex = 6;
  
  check(ANeuralNetworksModel_addOperand(model, &filterType), "Add filter");
  check(ANeuralNetworksModel_addOperand(model, &biasType), "Add bias");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add padding");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add stride");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add fuse");
  check(ANeuralNetworksModel_addOperand(model, &outputType), "Add output");

  check(ANeuralNetworksModel_setOperandValue(model, filterIndex, filterData, filterSize), "Set filter");
  check(ANeuralNetworksModel_setOperandValue(model, biasIndex, biasData, biasSize), "Set bias");
  
  int32_t padding = ANEURALNETWORKS_PADDING_SAME;
  int32_t stride = 1;
  int32_t fuse_none = ANEURALNETWORKS_FUSED_NONE;
  
  check(ANeuralNetworksModel_setOperandValue(model, paddingIndex, &padding, sizeof(padding)), "Set padding");
  check(ANeuralNetworksModel_setOperandValue(model, strideIndex, &stride, sizeof(stride)), "Set stride");
  check(ANeuralNetworksModel_setOperandValue(model, fuseIndex, &fuse_none, sizeof(fuse_none)), "Set fuse");

  uint32_t inputs[] = {inputIndex, filterIndex, biasIndex, paddingIndex, strideIndex, strideIndex, fuseIndex};
  uint32_t outputs[] = {outputIndex};
  check(ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, 7, inputs, 1, outputs), "Add conv2d");
  
  return outputIndex;
}

uint32_t setup_stage2_maxpool(ANeuralNetworksModel* model, uint32_t inputIndex) {
  // First MaxPool: [128, 32, 32, 16] -> [128, 16, 16, 16]
  constexpr uint32_t outputDims[] = {128, 16, 16, 16};
  const ANeuralNetworksOperandType outputType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(outputDims),
                                              .dimensions = outputDims,
                                              .scale = 0.1f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType paramType{.type = ANEURALNETWORKS_INT32,
                                             .dimensionCount = 0,
                                             .dimensions = nullptr,
                                             .scale = 0.0f,
                                             .zeroPoint = 0};

  uint32_t paddingIndex = 7, strideWidthIndex = 8, strideHeightIndex = 9;
  uint32_t filterWidthIndex = 10, filterHeightIndex = 11, fuseIndex = 12, outputIndex = 13;
  
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool padding");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool stride width");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool stride height");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool filter width");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool filter height");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool fuse");
  check(ANeuralNetworksModel_addOperand(model, &outputType), "Add pool output");

  int32_t padding = ANEURALNETWORKS_PADDING_VALID;
  int32_t stride = 2;
  int32_t filterSize = 2;
  int32_t fuse = ANEURALNETWORKS_FUSED_NONE;

  check(ANeuralNetworksModel_setOperandValue(model, paddingIndex, &padding, sizeof(padding)), "Set pool padding");
  check(ANeuralNetworksModel_setOperandValue(model, strideWidthIndex, &stride, sizeof(stride)), "Set pool stride width");
  check(ANeuralNetworksModel_setOperandValue(model, strideHeightIndex, &stride, sizeof(stride)), "Set pool stride height");
  check(ANeuralNetworksModel_setOperandValue(model, filterWidthIndex, &filterSize, sizeof(filterSize)), "Set pool filter width");
  check(ANeuralNetworksModel_setOperandValue(model, filterHeightIndex, &filterSize, sizeof(filterSize)), "Set pool filter height");
  check(ANeuralNetworksModel_setOperandValue(model, fuseIndex, &fuse, sizeof(fuse)), "Set pool fuse");

  uint32_t inputs[] = {inputIndex, paddingIndex, strideWidthIndex, strideHeightIndex, 
                       filterWidthIndex, filterHeightIndex, fuseIndex};
  uint32_t outputs[] = {outputIndex};
  check(ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MAX_POOL_2D, 7, inputs, 1, outputs), "Add maxpool");
  
  return outputIndex;
}

uint32_t setup_stage3_conv2d(ANeuralNetworksModel* model, uint32_t inputIndex, 
                             const uint8_t* filterData, size_t filterSize,
                             const int32_t* biasData, size_t biasSize) {
  // Second Conv2D: [128, 16, 16, 16] -> [128, 16, 16, 32]
  constexpr uint32_t filterDims[] = {32, 3, 3, 16};
  constexpr uint32_t biasDims[] = {32};
  constexpr uint32_t outputDims[] = {128, 16, 16, 32};
  
  const ANeuralNetworksOperandType filterType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(filterDims),
                                              .dimensions = filterDims,
                                              .scale = 0.01f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType biasType{.type = ANEURALNETWORKS_TENSOR_INT32,
                                            .dimensionCount = std::size(biasDims),
                                            .dimensions = biasDims,
                                            .scale = 0.001f,
                                            .zeroPoint = 0};
  const ANeuralNetworksOperandType outputType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(outputDims),
                                              .dimensions = outputDims,
                                              .scale = 0.1f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType paramType{.type = ANEURALNETWORKS_INT32,
                                             .dimensionCount = 0,
                                             .dimensions = nullptr,
                                             .scale = 0.0f,
                                             .zeroPoint = 0};

  uint32_t filterIndex = 14, biasIndex = 15, paddingIndex = 16, strideIndex = 17, fuseIndex = 18, outputIndex = 19;
  
  check(ANeuralNetworksModel_addOperand(model, &filterType), "Add filter");
  check(ANeuralNetworksModel_addOperand(model, &biasType), "Add bias");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add padding");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add stride");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add fuse");
  check(ANeuralNetworksModel_addOperand(model, &outputType), "Add output");

  check(ANeuralNetworksModel_setOperandValue(model, filterIndex, filterData, filterSize), "Set filter");
  check(ANeuralNetworksModel_setOperandValue(model, biasIndex, biasData, biasSize), "Set bias");
  
  int32_t padding = ANEURALNETWORKS_PADDING_SAME;
  int32_t stride = 1;
  int32_t fuse_none = ANEURALNETWORKS_FUSED_NONE;
  
  check(ANeuralNetworksModel_setOperandValue(model, paddingIndex, &padding, sizeof(padding)), "Set padding");
  check(ANeuralNetworksModel_setOperandValue(model, strideIndex, &stride, sizeof(stride)), "Set stride");
  check(ANeuralNetworksModel_setOperandValue(model, fuseIndex, &fuse_none, sizeof(fuse_none)), "Set fuse");

  uint32_t inputs[] = {inputIndex, filterIndex, biasIndex, paddingIndex, strideIndex, strideIndex, fuseIndex};
  uint32_t outputs[] = {outputIndex};
  check(ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, 7, inputs, 1, outputs), "Add conv2d");
  
  return outputIndex;
}

uint32_t setup_stage4_maxpool(ANeuralNetworksModel* model, uint32_t inputIndex) {
  // Second MaxPool: [128, 16, 16, 32] -> [128, 8, 8, 32]
  constexpr uint32_t outputDims[] = {128, 8, 8, 32};
  const ANeuralNetworksOperandType outputType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(outputDims),
                                              .dimensions = outputDims,
                                              .scale = 0.1f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType paramType{.type = ANEURALNETWORKS_INT32,
                                             .dimensionCount = 0,
                                             .dimensions = nullptr,
                                             .scale = 0.0f,
                                             .zeroPoint = 0};

  uint32_t paddingIndex = 20, strideWidthIndex = 21, strideHeightIndex = 22;
  uint32_t filterWidthIndex = 23, filterHeightIndex = 24, fuseIndex = 25, outputIndex = 26;
  
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool padding");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool stride width");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool stride height");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool filter width");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool filter height");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool fuse");
  check(ANeuralNetworksModel_addOperand(model, &outputType), "Add pool output");

  int32_t padding = ANEURALNETWORKS_PADDING_VALID;
  int32_t stride = 2;
  int32_t filterSize = 2;
  int32_t fuse = ANEURALNETWORKS_FUSED_NONE;

  check(ANeuralNetworksModel_setOperandValue(model, paddingIndex, &padding, sizeof(padding)), "Set pool padding");
  check(ANeuralNetworksModel_setOperandValue(model, strideWidthIndex, &stride, sizeof(stride)), "Set pool stride width");
  check(ANeuralNetworksModel_setOperandValue(model, strideHeightIndex, &stride, sizeof(stride)), "Set pool stride height");
  check(ANeuralNetworksModel_setOperandValue(model, filterWidthIndex, &filterSize, sizeof(filterSize)), "Set pool filter width");
  check(ANeuralNetworksModel_setOperandValue(model, filterHeightIndex, &filterSize, sizeof(filterSize)), "Set pool filter height");
  check(ANeuralNetworksModel_setOperandValue(model, fuseIndex, &fuse, sizeof(fuse)), "Set pool fuse");

  uint32_t inputs[] = {inputIndex, paddingIndex, strideWidthIndex, strideHeightIndex, 
                       filterWidthIndex, filterHeightIndex, fuseIndex};
  uint32_t outputs[] = {outputIndex};
  check(ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MAX_POOL_2D, 7, inputs, 1, outputs), "Add maxpool");
  
  return outputIndex;
}

uint32_t setup_stage5_conv2d(ANeuralNetworksModel* model, uint32_t inputIndex, 
                             const uint8_t* filterData, size_t filterSize,
                             const int32_t* biasData, size_t biasSize) {
  // Third Conv2D: [128, 8, 8, 32] -> [128, 8, 8, 64]
  constexpr uint32_t filterDims[] = {64, 3, 3, 32};
  constexpr uint32_t biasDims[] = {64};
  constexpr uint32_t outputDims[] = {128, 8, 8, 64};
  
  const ANeuralNetworksOperandType filterType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(filterDims),
                                              .dimensions = filterDims,
                                              .scale = 0.01f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType biasType{.type = ANEURALNETWORKS_TENSOR_INT32,
                                            .dimensionCount = std::size(biasDims),
                                            .dimensions = biasDims,
                                            .scale = 0.001f,
                                            .zeroPoint = 0};
  const ANeuralNetworksOperandType outputType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(outputDims),
                                              .dimensions = outputDims,
                                              .scale = 0.1f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType paramType{.type = ANEURALNETWORKS_INT32,
                                             .dimensionCount = 0,
                                             .dimensions = nullptr,
                                             .scale = 0.0f,
                                             .zeroPoint = 0};

  uint32_t filterIndex = 27, biasIndex = 28, paddingIndex = 29, strideIndex = 30, fuseIndex = 31, outputIndex = 32;
  
  check(ANeuralNetworksModel_addOperand(model, &filterType), "Add filter");
  check(ANeuralNetworksModel_addOperand(model, &biasType), "Add bias");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add padding");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add stride");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add fuse");
  check(ANeuralNetworksModel_addOperand(model, &outputType), "Add output");

  check(ANeuralNetworksModel_setOperandValue(model, filterIndex, filterData, filterSize), "Set filter");
  check(ANeuralNetworksModel_setOperandValue(model, biasIndex, biasData, biasSize), "Set bias");
  
  int32_t padding = ANEURALNETWORKS_PADDING_SAME;
  int32_t stride = 1;
  int32_t fuse_none = ANEURALNETWORKS_FUSED_NONE;
  
  check(ANeuralNetworksModel_setOperandValue(model, paddingIndex, &padding, sizeof(padding)), "Set padding");
  check(ANeuralNetworksModel_setOperandValue(model, strideIndex, &stride, sizeof(stride)), "Set stride");
  check(ANeuralNetworksModel_setOperandValue(model, fuseIndex, &fuse_none, sizeof(fuse_none)), "Set fuse");

  uint32_t inputs[] = {inputIndex, filterIndex, biasIndex, paddingIndex, strideIndex, strideIndex, fuseIndex};
  uint32_t outputs[] = {outputIndex};
  check(ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, 7, inputs, 1, outputs), "Add conv2d");
  
  return outputIndex;
}

uint32_t setup_stage6_conv2d(ANeuralNetworksModel* model, uint32_t inputIndex, 
                             const uint8_t* filterData, size_t filterSize,
                             const int32_t* biasData, size_t biasSize) {
  // Fourth Conv2D: [128, 8, 8, 64] -> [128, 8, 8, 64]
  constexpr uint32_t filterDims[] = {64, 3, 3, 64};
  constexpr uint32_t biasDims[] = {64};
  constexpr uint32_t outputDims[] = {128, 8, 8, 64};
  
  const ANeuralNetworksOperandType filterType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(filterDims),
                                              .dimensions = filterDims,
                                              .scale = 0.01f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType biasType{.type = ANEURALNETWORKS_TENSOR_INT32,
                                            .dimensionCount = std::size(biasDims),
                                            .dimensions = biasDims,
                                            .scale = 0.001f,
                                            .zeroPoint = 0};
  const ANeuralNetworksOperandType outputType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(outputDims),
                                              .dimensions = outputDims,
                                              .scale = 0.1f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType paramType{.type = ANEURALNETWORKS_INT32,
                                             .dimensionCount = 0,
                                             .dimensions = nullptr,
                                             .scale = 0.0f,
                                             .zeroPoint = 0};

  uint32_t filterIndex = 33, biasIndex = 34, paddingIndex = 35, strideIndex = 36, fuseIndex = 37, outputIndex = 38;
  
  check(ANeuralNetworksModel_addOperand(model, &filterType), "Add filter");
  check(ANeuralNetworksModel_addOperand(model, &biasType), "Add bias");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add padding");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add stride");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add fuse");
  check(ANeuralNetworksModel_addOperand(model, &outputType), "Add output");

  check(ANeuralNetworksModel_setOperandValue(model, filterIndex, filterData, filterSize), "Set filter");
  check(ANeuralNetworksModel_setOperandValue(model, biasIndex, biasData, biasSize), "Set bias");
  
  int32_t padding = ANEURALNETWORKS_PADDING_SAME;
  int32_t stride = 1;
  int32_t fuse_none = ANEURALNETWORKS_FUSED_NONE;
  
  check(ANeuralNetworksModel_setOperandValue(model, paddingIndex, &padding, sizeof(padding)), "Set padding");
  check(ANeuralNetworksModel_setOperandValue(model, strideIndex, &stride, sizeof(stride)), "Set stride");
  check(ANeuralNetworksModel_setOperandValue(model, fuseIndex, &fuse_none, sizeof(fuse_none)), "Set fuse");

  uint32_t inputs[] = {inputIndex, filterIndex, biasIndex, paddingIndex, strideIndex, strideIndex, fuseIndex};
  uint32_t outputs[] = {outputIndex};
  check(ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, 7, inputs, 1, outputs), "Add conv2d");
  
  return outputIndex;
}

uint32_t setup_stage7_conv2d(ANeuralNetworksModel* model, uint32_t inputIndex, 
                             const uint8_t* filterData, size_t filterSize,
                             const int32_t* biasData, size_t biasSize) {
  // Fifth Conv2D: [128, 8, 8, 64] -> [128, 8, 8, 64]
  constexpr uint32_t filterDims[] = {64, 3, 3, 64};
  constexpr uint32_t biasDims[] = {64};
  constexpr uint32_t outputDims[] = {128, 8, 8, 64};
  
  const ANeuralNetworksOperandType filterType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(filterDims),
                                              .dimensions = filterDims,
                                              .scale = 0.01f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType biasType{.type = ANEURALNETWORKS_TENSOR_INT32,
                                            .dimensionCount = std::size(biasDims),
                                            .dimensions = biasDims,
                                            .scale = 0.001f,
                                            .zeroPoint = 0};
  const ANeuralNetworksOperandType outputType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(outputDims),
                                              .dimensions = outputDims,
                                              .scale = 0.1f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType paramType{.type = ANEURALNETWORKS_INT32,
                                             .dimensionCount = 0,
                                             .dimensions = nullptr,
                                             .scale = 0.0f,
                                             .zeroPoint = 0};

  uint32_t filterIndex = 39, biasIndex = 40, paddingIndex = 41, strideIndex = 42, fuseIndex = 43, outputIndex = 44;
  
  check(ANeuralNetworksModel_addOperand(model, &filterType), "Add filter");
  check(ANeuralNetworksModel_addOperand(model, &biasType), "Add bias");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add padding");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add stride");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add fuse");
  check(ANeuralNetworksModel_addOperand(model, &outputType), "Add output");

  check(ANeuralNetworksModel_setOperandValue(model, filterIndex, filterData, filterSize), "Set filter");
  check(ANeuralNetworksModel_setOperandValue(model, biasIndex, biasData, biasSize), "Set bias");
  
  int32_t padding = ANEURALNETWORKS_PADDING_SAME;
  int32_t stride = 1;
  int32_t fuse_none = ANEURALNETWORKS_FUSED_NONE;
  
  check(ANeuralNetworksModel_setOperandValue(model, paddingIndex, &padding, sizeof(padding)), "Set padding");
  check(ANeuralNetworksModel_setOperandValue(model, strideIndex, &stride, sizeof(stride)), "Set stride");
  check(ANeuralNetworksModel_setOperandValue(model, fuseIndex, &fuse_none, sizeof(fuse_none)), "Set fuse");

  uint32_t inputs[] = {inputIndex, filterIndex, biasIndex, paddingIndex, strideIndex, strideIndex, fuseIndex};
  uint32_t outputs[] = {outputIndex};
  check(ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, 7, inputs, 1, outputs), "Add conv2d");
  
  return outputIndex;
}

uint32_t setup_stage8_maxpool(ANeuralNetworksModel* model, uint32_t inputIndex) {
  // Third MaxPool: [128, 8, 8, 64] -> [128, 4, 4, 64]
  constexpr uint32_t outputDims[] = {128, 4, 4, 64};
  const ANeuralNetworksOperandType outputType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(outputDims),
                                              .dimensions = outputDims,
                                              .scale = 0.1f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType paramType{.type = ANEURALNETWORKS_INT32,
                                             .dimensionCount = 0,
                                             .dimensions = nullptr,
                                             .scale = 0.0f,
                                             .zeroPoint = 0};

  uint32_t paddingIndex = 45, strideWidthIndex = 46, strideHeightIndex = 47;
  uint32_t filterWidthIndex = 48, filterHeightIndex = 49, fuseIndex = 50, outputIndex = 51;
  
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool padding");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool stride width");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool stride height");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool filter width");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool filter height");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add pool fuse");
  check(ANeuralNetworksModel_addOperand(model, &outputType), "Add pool output");

  int32_t padding = ANEURALNETWORKS_PADDING_VALID;
  int32_t stride = 2;
  int32_t filterSize = 2;
  int32_t fuse = ANEURALNETWORKS_FUSED_NONE;

  check(ANeuralNetworksModel_setOperandValue(model, paddingIndex, &padding, sizeof(padding)), "Set pool padding");
  check(ANeuralNetworksModel_setOperandValue(model, strideWidthIndex, &stride, sizeof(stride)), "Set pool stride width");
  check(ANeuralNetworksModel_setOperandValue(model, strideHeightIndex, &stride, sizeof(stride)), "Set pool stride height");
  check(ANeuralNetworksModel_setOperandValue(model, filterWidthIndex, &filterSize, sizeof(filterSize)), "Set pool filter width");
  check(ANeuralNetworksModel_setOperandValue(model, filterHeightIndex, &filterSize, sizeof(filterSize)), "Set pool filter height");
  check(ANeuralNetworksModel_setOperandValue(model, fuseIndex, &fuse, sizeof(fuse)), "Set pool fuse");

  uint32_t inputs[] = {inputIndex, paddingIndex, strideWidthIndex, strideHeightIndex, 
                       filterWidthIndex, filterHeightIndex, fuseIndex};
  uint32_t outputs[] = {outputIndex};
  check(ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MAX_POOL_2D, 7, inputs, 1, outputs), "Add maxpool");
  
  return outputIndex;
}

uint32_t setup_stage9_fc(ANeuralNetworksModel* model, uint32_t inputIndex, 
                         const uint8_t* weightData, size_t weightSize,
                         const int32_t* biasData, size_t biasSize) {
  // FC Layer: [128, 4, 4, 64] -> [128, 10]
  constexpr uint32_t weightDims[] = {10, 1024};
  constexpr uint32_t biasDims[] = {10};
  constexpr uint32_t outputDims[] = {128, 10};
  
  const ANeuralNetworksOperandType weightType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(weightDims),
                                              .dimensions = weightDims,
                                              .scale = 0.01f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType biasType{.type = ANEURALNETWORKS_TENSOR_INT32,
                                            .dimensionCount = std::size(biasDims),
                                            .dimensions = biasDims,
                                            .scale = 0.001f,
                                            .zeroPoint = 0};
  const ANeuralNetworksOperandType outputType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(outputDims),
                                              .dimensions = outputDims,
                                              .scale = 0.1f,
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType paramType{.type = ANEURALNETWORKS_INT32,
                                             .dimensionCount = 0,
                                             .dimensions = nullptr,
                                             .scale = 0.0f,
                                             .zeroPoint = 0};

  uint32_t weightIndex = 52, biasIndex = 53, fuseIndex = 54, outputIndex = 55;
  
  check(ANeuralNetworksModel_addOperand(model, &weightType), "Add fc weight");
  check(ANeuralNetworksModel_addOperand(model, &biasType), "Add fc bias");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add fc fuse");
  check(ANeuralNetworksModel_addOperand(model, &outputType), "Add fc output");

  check(ANeuralNetworksModel_setOperandValue(model, weightIndex, weightData, weightSize), "Set fc weight");
  check(ANeuralNetworksModel_setOperandValue(model, biasIndex, biasData, biasSize), "Set fc bias");
  
  int32_t fuse_none = ANEURALNETWORKS_FUSED_NONE;
  check(ANeuralNetworksModel_setOperandValue(model, fuseIndex, &fuse_none, sizeof(fuse_none)), "Set fc fuse");

  uint32_t inputs[] = {inputIndex, weightIndex, biasIndex, fuseIndex};
  uint32_t outputs[] = {outputIndex};
  check(ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED, 4, inputs, 1, outputs), "Add fc");
  
  return outputIndex;
}

int main() {
  // Input dimensions: (128, 32, 32, 3) - batch, height, width, channels (NHWC format)
  constexpr uint32_t inputDims[] = {128, 32, 32, 3};
  std::vector<uint8_t> inputData(128 * 32 * 32 * 3);

  // Fill with some test data
  for (size_t i = 0; i < inputData.size(); ++i) {
    inputData[i] = static_cast<uint8_t>(i % 256);
  }

  // Prepare all filter and bias data
  // First Conv2D: 3->16 channels
  std::vector<uint8_t> filterData(16 * 3 * 3 * 3);
  for (size_t i = 0; i < filterData.size(); ++i) {
    filterData[i] = static_cast<uint8_t>((i % 128) + 64);
  }
  std::vector<int32_t> biasData(16, 0);

  // Second Conv2D: 16->32 channels
  std::vector<uint8_t> filter2Data(32 * 3 * 3 * 16);
  for (size_t i = 0; i < filter2Data.size(); ++i) {
    filter2Data[i] = static_cast<uint8_t>((i % 96) + 32);
  }
  std::vector<int32_t> bias2Data(32, 0);

  // Third Conv2D: 32->64 channels
  std::vector<uint8_t> filter3Data(64 * 3 * 3 * 32);
  for (size_t i = 0; i < filter3Data.size(); ++i) {
    filter3Data[i] = static_cast<uint8_t>((i % 112) + 16);
  }
  std::vector<int32_t> bias3Data(64, 0);

  // Fourth Conv2D: 64->64 channels
  std::vector<uint8_t> filter4Data(64 * 3 * 3 * 64);
  for (size_t i = 0; i < filter4Data.size(); ++i) {
    filter4Data[i] = static_cast<uint8_t>((i % 128) + 48);
  }
  std::vector<int32_t> bias4Data(64, 0);

  // Fifth Conv2D: 64->64 channels
  std::vector<uint8_t> filter5Data(64 * 3 * 3 * 64);
  for (size_t i = 0; i < filter5Data.size(); ++i) {
    filter5Data[i] = static_cast<uint8_t>((i % 144) + 80);
  }
  std::vector<int32_t> bias5Data(64, 0);

  // FC Layer: 1024->10 features
  std::vector<uint8_t> fcWeightData(10 * 1024);
  for (size_t i = 0; i < fcWeightData.size(); ++i) {
    fcWeightData[i] = static_cast<uint8_t>((i % 160) + 96);
  }
  std::vector<int32_t> fcBiasData(10, 0);

  // Output data buffers
  std::vector<uint8_t> fcOutputData(128 * 10, 0);


  // Build model
  ANeuralNetworksModel* model = nullptr;
  check(ANeuralNetworksModel_create(&model), "Model create");

  // Add input operand
  const ANeuralNetworksOperandType inputType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                             .dimensionCount = std::size(inputDims),
                                             .dimensions = inputDims,
                                             .scale = 0.1f,
                                             .zeroPoint = 0};
  uint32_t inputIndex = 0;
  check(ANeuralNetworksModel_addOperand(model, &inputType), "Add input");

  // Setup all 9 stages
  uint32_t stage1_output = setup_stage1_conv2d(model, inputIndex, filterData.data(), filterData.size() * sizeof(uint8_t), biasData.data(), biasData.size() * sizeof(int32_t));
  uint32_t stage2_output = setup_stage2_maxpool(model, stage1_output);
  uint32_t stage3_output = setup_stage3_conv2d(model, stage2_output, filter2Data.data(), filter2Data.size() * sizeof(uint8_t), bias2Data.data(), bias2Data.size() * sizeof(int32_t));
  uint32_t stage4_output = setup_stage4_maxpool(model, stage3_output);
  uint32_t stage5_output = setup_stage5_conv2d(model, stage4_output, filter3Data.data(), filter3Data.size() * sizeof(uint8_t), bias3Data.data(), bias3Data.size() * sizeof(int32_t));
  uint32_t stage6_output = setup_stage6_conv2d(model, stage5_output, filter4Data.data(), filter4Data.size() * sizeof(uint8_t), bias4Data.data(), bias4Data.size() * sizeof(int32_t));
  uint32_t stage7_output = setup_stage7_conv2d(model, stage6_output, filter5Data.data(), filter5Data.size() * sizeof(uint8_t), bias5Data.data(), bias5Data.size() * sizeof(int32_t));
  uint32_t stage8_output = setup_stage8_maxpool(model, stage7_output);
  uint32_t stage9_output = setup_stage9_fc(model, stage8_output, fcWeightData.data(), fcWeightData.size() * sizeof(uint8_t), fcBiasData.data(), fcBiasData.size() * sizeof(int32_t));

  // Set inputs and outputs
  uint32_t modelInputs[] = {inputIndex};
  uint32_t modelOutputs[] = {stage9_output};
  check(ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, modelInputs, 1, modelOutputs), "Identify IO");

  // Finalize model
  check(ANeuralNetworksModel_finish(model), "Finish model");

  // Compile
  ANeuralNetworksCompilation* compilation = nullptr;
  check(ANeuralNetworksCompilation_create(model, &compilation), "Create compilation");
  check(ANeuralNetworksCompilation_finish(compilation), "Finish compilation");

  // Execution
  ANeuralNetworksExecution* execution = nullptr;
  check(ANeuralNetworksExecution_create(compilation, &execution), "Create execution");

  check(ANeuralNetworksExecution_setInput(
            execution, 0, nullptr, inputData.data(), getBytes(inputData)),
        "Set input");
  check(ANeuralNetworksExecution_setOutput(
            execution, 0, nullptr, fcOutputData.data(), getBytes(fcOutputData)),
        "Set output");

  auto start = std::chrono::high_resolution_clock::now();

  check(ANeuralNetworksExecution_compute(execution), "Compute");

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  // duration cast to ms
  std::cout << "Time taken: " << duration.count() * 1000 << " ms" << std::endl;

  // Print result
  std::cout << "FC Layer Output (quantized) - first 10 values:\n";
  for (int i = 0; i < 10 && i < static_cast<int>(fcOutputData.size()); i++) {
    std::cout << static_cast<int>(fcOutputData[i]) << " ";
  }
  std::cout << std::endl;
  std::cout << "Total output size: " << fcOutputData.size() << " elements" << std::endl;
  std::cout << "AlexNet-like model with 9 stages:" << std::endl;
  std::cout << "Stage 1: Conv2D [128, 32, 32, 3] -> [128, 32, 32, 16]" << std::endl;
  std::cout << "Stage 2: MaxPool [128, 32, 32, 16] -> [128, 16, 16, 16]" << std::endl;
  std::cout << "Stage 3: Conv2D [128, 16, 16, 16] -> [128, 16, 16, 32]" << std::endl;
  std::cout << "Stage 4: MaxPool [128, 16, 16, 32] -> [128, 8, 8, 32]" << std::endl;
  std::cout << "Stage 5: Conv2D [128, 8, 8, 32] -> [128, 8, 8, 64]" << std::endl;
  std::cout << "Stage 6: Conv2D [128, 8, 8, 64] -> [128, 8, 8, 64]" << std::endl;
  std::cout << "Stage 7: Conv2D [128, 8, 8, 64] -> [128, 8, 8, 64]" << std::endl;
  std::cout << "Stage 8: MaxPool [128, 8, 8, 64] -> [128, 4, 4, 64]" << std::endl;
  std::cout << "Stage 9: FC [128, 4, 4, 64] -> [128, 10]" << std::endl;

  // Clean up
  ANeuralNetworksExecution_free(execution);
  ANeuralNetworksCompilation_free(compilation);
  ANeuralNetworksModel_free(model);
  return 0;
}
