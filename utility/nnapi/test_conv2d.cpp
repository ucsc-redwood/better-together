#include <android/NeuralNetworks.h>

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

int main() {
  constexpr uint32_t inputDims[] = {128, 32, 32, 3};
  std::vector<uint8_t> inputData(128 * 32 * 32 * 3);

  // Fill with some test data
  for (size_t i = 0; i < inputData.size(); ++i) {
    inputData[i] = static_cast<uint8_t>(i % 256);
  }

  constexpr uint32_t filterDims[] = {16, 3, 3, 3};
  std::vector<uint8_t> filterData(16 * 3 * 3 * 3);
  // Fill with some test filter data
  for (size_t i = 0; i < filterData.size(); ++i) {
    filterData[i] = static_cast<uint8_t>((i % 128) + 64);
  }

  constexpr uint32_t biasDims[] = {16};
  std::vector<int32_t> biasData(16, 0);

  constexpr uint32_t outputDims[] = {128, 32, 32, 16};
  std::vector<uint8_t> outputData(128 * 32 * 32 * 16, 0);

  // Operand types
  const ANeuralNetworksOperandType tensorType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(inputDims),
                                              .dimensions = inputDims,
                                              .scale = 0.1f,
                                              .zeroPoint = 0};
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

  const ANeuralNetworksOperandType paramType{.type = ANEURALNETWORKS_INT32,
                                             .dimensionCount = 0,
                                             .dimensions = nullptr,
                                             .scale = 0.0f,
                                             .zeroPoint = 0};

  const ANeuralNetworksOperandType outputType{.type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                              .dimensionCount = std::size(outputDims),
                                              .dimensions = outputDims,
                                              .scale = 0.1f,
                                              .zeroPoint = 0};

  // Build model
  ANeuralNetworksModel* model = nullptr;
  check(ANeuralNetworksModel_create(&model), "Model create");

  // Add operands
  uint32_t inputIndex = 0, filterIndex = 1, biasIndex = 2;
  uint32_t paddingIndex = 3, strideIndex = 4, fuseIndex = 5;
  uint32_t outputIndex = 6;

  check(ANeuralNetworksModel_addOperand(model, &tensorType), "Add input");
  check(ANeuralNetworksModel_addOperand(model, &filterType), "Add filter");
  check(ANeuralNetworksModel_addOperand(model, &biasType), "Add bias");

  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add padding");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add stride");
  check(ANeuralNetworksModel_addOperand(model, &paramType), "Add fuse");

  check(ANeuralNetworksModel_addOperand(model, &outputType), "Add output");

  // Set constants: filter, bias, padding, stride, activation
  check(ANeuralNetworksModel_setOperandValue(
            model, filterIndex, filterData.data(), getBytes(filterData)),
        "Set filter");
  check(ANeuralNetworksModel_setOperandValue(model, biasIndex, biasData.data(), getBytes(biasData)),
        "Set bias");

  int32_t padding = ANEURALNETWORKS_PADDING_SAME;
  int32_t stride = 1;
  int32_t fuse_none = ANEURALNETWORKS_FUSED_NONE;

  check(ANeuralNetworksModel_setOperandValue(model, paddingIndex, &padding, sizeof(padding)),
        "Set padding");
  check(ANeuralNetworksModel_setOperandValue(model, strideIndex, &stride, sizeof(stride)),
        "Set stride");
  check(ANeuralNetworksModel_setOperandValue(model, fuseIndex, &fuse_none, sizeof(fuse_none)),
        "Set fuse");

  // Add CONV_2D operation
  uint32_t inputs[] = {
      inputIndex, filterIndex, biasIndex, paddingIndex, strideIndex, strideIndex, fuseIndex};
  uint32_t outputs[] = {outputIndex};
  check(ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, 7, inputs, 1, outputs),
        "Add conv2d");

  // Set inputs and outputs
  check(ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, &inputIndex, 1, outputs),
        "Identify IO");

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
            execution, 0, nullptr, outputData.data(), getBytes(outputData)),
        "Set output");

  auto start = std::chrono::high_resolution_clock::now();

  check(ANeuralNetworksExecution_compute(execution), "Compute");

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  // duration cast to ms
  std::cout << "Time taken: " << duration.count() * 1000 << " ms" << std::endl;

  // Print result
  std::cout << "Output (quantized) - first 16 values:\n";
  for (int i = 0; i < 16 && i < static_cast<int>(outputData.size()); i++) {
    std::cout << static_cast<int>(outputData[i]) << " ";
  }
  std::cout << std::endl;
  std::cout << "Total output size: " << outputData.size() << " elements" << std::endl;
  std::cout << "Input shape: [128, 32, 32, 3] -> Output shape: [128, 32, 32, 16]" << std::endl;

  // Clean up
  ANeuralNetworksExecution_free(execution);
  ANeuralNetworksCompilation_free(compilation);
  ANeuralNetworksModel_free(model);
  return 0;
}
