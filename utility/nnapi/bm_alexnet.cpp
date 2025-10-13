#include <android/NeuralNetworks.h>
#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>

#include "builtin-apps/app.hpp"

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

constexpr size_t kBatchSize = 128;

static void BM_AlexNet_uint8(benchmark::State& state) {
  constexpr uint32_t inputDims[] = {kBatchSize, 32, 32, 3};
  std::vector<uint8_t> inputData(kBatchSize * 32 * 32 * 3);

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

  constexpr uint32_t outputDims[] = {kBatchSize, 32, 32, 16};
  std::vector<uint8_t> outputData(kBatchSize * 32 * 32 * 16, 0);

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

  // Benchmark
  for (auto _ : state) {
    // Execution
    ANeuralNetworksExecution* execution = nullptr;
    check(ANeuralNetworksExecution_create(compilation, &execution), "Create execution");

    check(ANeuralNetworksExecution_setInput(
              execution, 0, nullptr, inputData.data(), getBytes(inputData)),
          "Set input");
    check(ANeuralNetworksExecution_setOutput(
              execution, 0, nullptr, outputData.data(), getBytes(outputData)),
          "Set output");

    check(ANeuralNetworksExecution_compute(execution), "Compute");

    ANeuralNetworksExecution_free(execution);
  }

  // Clean up

  ANeuralNetworksCompilation_free(compilation);
  ANeuralNetworksModel_free(model);
}

// ... existing code ...

static void BM_AlexNet_float32(benchmark::State& state) {
  constexpr uint32_t inputDims[] = {kBatchSize, 32, 32, 3};
  std::vector<float> inputData(kBatchSize * 32 * 32 * 3);

  // Fill with some test data
  for (size_t i = 0; i < inputData.size(); ++i) {
    inputData[i] = static_cast<float>(i % 256) * 0.01f;  // Scale down for reasonable values
  }

  constexpr uint32_t filterDims[] = {16, 3, 3, 3};
  std::vector<float> filterData(16 * 3 * 3 * 3);
  // Fill with some test filter data
  for (size_t i = 0; i < filterData.size(); ++i) {
    filterData[i] =
        static_cast<float>((i % 128) + 64) * 0.001f;  // Scale down for reasonable values
  }

  constexpr uint32_t biasDims[] = {16};
  std::vector<float> biasData(16, 0.0f);

  constexpr uint32_t outputDims[] = {kBatchSize, 32, 32, 16};
  std::vector<float> outputData(kBatchSize * 32 * 32 * 16, 0.0f);

  // Operand types for FLOAT32
  const ANeuralNetworksOperandType tensorType{.type = ANEURALNETWORKS_TENSOR_FLOAT32,
                                              .dimensionCount = std::size(inputDims),
                                              .dimensions = inputDims,
                                              .scale = 0.0f,  // Not used for FLOAT32
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType filterType{.type = ANEURALNETWORKS_TENSOR_FLOAT32,
                                              .dimensionCount = std::size(filterDims),
                                              .dimensions = filterDims,
                                              .scale = 0.0f,  // Not used for FLOAT32
                                              .zeroPoint = 0};

  const ANeuralNetworksOperandType biasType{.type = ANEURALNETWORKS_TENSOR_FLOAT32,
                                            .dimensionCount = std::size(biasDims),
                                            .dimensions = biasDims,
                                            .scale = 0.0f,  // Not used for FLOAT32
                                            .zeroPoint = 0};

  const ANeuralNetworksOperandType paramType{.type = ANEURALNETWORKS_INT32,
                                             .dimensionCount = 0,
                                             .dimensions = nullptr,
                                             .scale = 0.0f,
                                             .zeroPoint = 0};

  const ANeuralNetworksOperandType outputType{.type = ANEURALNETWORKS_TENSOR_FLOAT32,
                                              .dimensionCount = std::size(outputDims),
                                              .dimensions = outputDims,
                                              .scale = 0.0f,  // Not used for FLOAT32
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

  // Benchmark
  for (auto _ : state) {
    // Execution
    ANeuralNetworksExecution* execution = nullptr;
    check(ANeuralNetworksExecution_create(compilation, &execution), "Create execution");

    check(ANeuralNetworksExecution_setInput(
              execution, 0, nullptr, inputData.data(), getBytes(inputData)),
          "Set input");
    check(ANeuralNetworksExecution_setOutput(
              execution, 0, nullptr, outputData.data(), getBytes(outputData)),
          "Set output");

    check(ANeuralNetworksExecution_compute(execution), "Compute");

    ANeuralNetworksExecution_free(execution);
  }

  // Clean up
  ANeuralNetworksCompilation_free(compilation);
  ANeuralNetworksModel_free(model);
}

static void BM_AlexNet_float16(benchmark::State& state) {
  constexpr uint32_t inputDims[] = {kBatchSize, 32, 32, 3};
  std::vector<uint16_t> inputData(kBatchSize * 32 * 32 * 3);  // FLOAT16 stored as uint16_t

  // Fill with some test data (converted to FLOAT16)
  for (size_t i = 0; i < inputData.size(); ++i) {
    float val = static_cast<float>(i % 256) * 0.01f;
    // Simple float to half conversion (this is a simplified version)
    uint16_t half = static_cast<uint16_t>(val * 65536.0f);  // Scale for reasonable FLOAT16 range
    inputData[i] = half;
  }

  constexpr uint32_t filterDims[] = {16, 3, 3, 3};
  std::vector<uint16_t> filterData(16 * 3 * 3 * 3);
  // Fill with some test filter data
  for (size_t i = 0; i < filterData.size(); ++i) {
    float val = static_cast<float>((i % 128) + 64) * 0.001f;
    uint16_t half = static_cast<uint16_t>(val * 65536.0f);
    filterData[i] = half;
  }

  constexpr uint32_t biasDims[] = {16};
  std::vector<uint16_t> biasData(16, 0);

  constexpr uint32_t outputDims[] = {kBatchSize, 32, 32, 16};
  std::vector<uint16_t> outputData(kBatchSize * 32 * 32 * 16, 0);

  // Operand types for FLOAT16
  const ANeuralNetworksOperandType tensorType{.type = ANEURALNETWORKS_TENSOR_FLOAT16,
                                              .dimensionCount = std::size(inputDims),
                                              .dimensions = inputDims,
                                              .scale = 0.0f,  // Not used for FLOAT16
                                              .zeroPoint = 0};
  const ANeuralNetworksOperandType filterType{.type = ANEURALNETWORKS_TENSOR_FLOAT16,
                                              .dimensionCount = std::size(filterDims),
                                              .dimensions = filterDims,
                                              .scale = 0.0f,  // Not used for FLOAT16
                                              .zeroPoint = 0};

  const ANeuralNetworksOperandType biasType{.type = ANEURALNETWORKS_TENSOR_FLOAT16,
                                            .dimensionCount = std::size(biasDims),
                                            .dimensions = biasDims,
                                            .scale = 0.0f,  // Not used for FLOAT16
                                            .zeroPoint = 0};

  const ANeuralNetworksOperandType paramType{.type = ANEURALNETWORKS_INT32,
                                             .dimensionCount = 0,
                                             .dimensions = nullptr,
                                             .scale = 0.0f,
                                             .zeroPoint = 0};

  const ANeuralNetworksOperandType outputType{.type = ANEURALNETWORKS_TENSOR_FLOAT16,
                                              .dimensionCount = std::size(outputDims),
                                              .dimensions = outputDims,
                                              .scale = 0.0f,  // Not used for FLOAT16
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

  // Benchmark
  for (auto _ : state) {
    // Execution
    ANeuralNetworksExecution* execution = nullptr;
    check(ANeuralNetworksExecution_create(compilation, &execution), "Create execution");

    check(ANeuralNetworksExecution_setInput(
              execution, 0, nullptr, inputData.data(), getBytes(inputData)),
          "Set input");
    check(ANeuralNetworksExecution_setOutput(
              execution, 0, nullptr, outputData.data(), getBytes(outputData)),
          "Set output");

    check(ANeuralNetworksExecution_compute(execution), "Compute");

    ANeuralNetworksExecution_free(execution);
  }

  // Clean up
  ANeuralNetworksCompilation_free(compilation);
  ANeuralNetworksModel_free(model);
}

BENCHMARK(BM_AlexNet_float32)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AlexNet_float16)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AlexNet_uint8)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup
  return 0;
}
