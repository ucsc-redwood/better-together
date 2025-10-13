#include <android/NeuralNetworks.h>

#include <iostream>

const char* deviceTypeToString(int32_t type) {
  switch (type) {
    case ANEURALNETWORKS_DEVICE_CPU:
      return "CPU";
    case ANEURALNETWORKS_DEVICE_GPU:
      return "GPU";
    case ANEURALNETWORKS_DEVICE_ACCELERATOR:
      return "ACCELERATOR";
    default:
      return "UNKNOWN";
  }
}

const char* dataTypeToString(int32_t type) {
  switch (type) {
    case ANEURALNETWORKS_TENSOR_FLOAT16:
      return "FLOAT16";
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      return "FLOAT32";
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      return "QUANT8_ASYMM";
    case ANEURALNETWORKS_TENSOR_QUANT8_SYMM:
      return "QUANT8_SYMM";
    case ANEURALNETWORKS_TENSOR_INT32:
      return "INT32";
    case ANEURALNETWORKS_TENSOR_BOOL8:
      return "BOOL8";
    case ANEURALNETWORKS_TENSOR_QUANT16_SYMM:
      return "QUANT16_SYMM";
    case ANEURALNETWORKS_TENSOR_QUANT16_ASYMM:
      return "QUANT16_ASYMM";
    default:
      return "UNKNOWN";
  }
}

void checkSupportedTypes(ANeuralNetworksDevice* device, const char* deviceName) {
  std::cout << "  Supported data types for " << deviceName << ":\n";

  // Test each data type by trying to create a simple model
  int32_t dataTypes[] = {ANEURALNETWORKS_TENSOR_FLOAT16,
                         ANEURALNETWORKS_TENSOR_FLOAT32,
                         ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                         ANEURALNETWORKS_TENSOR_QUANT8_SYMM,
                         ANEURALNETWORKS_TENSOR_INT32,
                         ANEURALNETWORKS_TENSOR_BOOL8,
                         ANEURALNETWORKS_TENSOR_QUANT16_SYMM,
                         ANEURALNETWORKS_TENSOR_QUANT16_ASYMM};

  for (auto dataType : dataTypes) {
    ANeuralNetworksModel* model = nullptr;
    int status = ANeuralNetworksModel_create(&model);
    if (status != ANEURALNETWORKS_NO_ERROR) {
      continue;
    }

    // Try to add an operand with this data type
    uint32_t dims[] = {1, 1, 1, 1};  // Simple 1x1x1x1 tensor
    ANeuralNetworksOperandType operandType = {
        .type = dataType, .dimensionCount = 4, .dimensions = dims, .scale = 0.0f, .zeroPoint = 0};

    status = ANeuralNetworksModel_addOperand(model, &operandType);
    if (status == ANEURALNETWORKS_NO_ERROR) {
      std::cout << "    + " << dataTypeToString(dataType) << "\n";
    } else {
      std::cout << "    - " << dataTypeToString(dataType) << " (error: " << status << ")\n";
    }

    ANeuralNetworksModel_free(model);
  }
}

int main() {
  uint32_t numDevices = 0;
  int status = ANeuralNetworks_getDeviceCount(&numDevices);
  if (status != ANEURALNETWORKS_NO_ERROR) {
    std::cerr << "Failed to get NNAPI device count, error code: " << status << std::endl;
    return 1;
  }

  std::cout << "Number of NNAPI devices: " << numDevices << "\n";

  for (uint32_t i = 0; i < numDevices; ++i) {
    ANeuralNetworksDevice* device = nullptr;
    status = ANeuralNetworks_getDevice(i, &device);
    if (status != ANEURALNETWORKS_NO_ERROR) {
      std::cerr << "Failed to get device " << i << ", error code: " << status << std::endl;
      continue;
    }

    const char* name = nullptr;
    ANeuralNetworksDevice_getName(device, &name);

    int32_t type = -1;
    ANeuralNetworksDevice_getType(device, &type);

    std::cout << "Device " << i << ": " << (name ? name : "Unnamed") << " ("
              << deviceTypeToString(type) << ")\n";

    // Check supported data types for this device
    checkSupportedTypes(device, name ? name : "Unnamed");
    std::cout << "\n";
  }

  return 0;
}
