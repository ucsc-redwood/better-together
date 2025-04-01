#include "dense_appdata.hpp"

#include <algorithm>
#include <fstream>

#include "../resources_path.hpp"

namespace cifar_dense {
void readDataFromFile(const std::string_view filename, float *data, const int size) {
  const auto base_path = helpers::get_resource_base_path();
  const auto full_path = base_path / filename;

  std::ifstream file(full_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open file '" + full_path.string() + "'");
  }

  // Read entire file at once for better performance
  std::vector<float> buffer(size);
  if (!file.read(reinterpret_cast<char *>(buffer.data()), size * sizeof(float))) {
    throw std::runtime_error("Failed to read " + std::to_string(size * sizeof(float)) +
                             " bytes from '" + full_path.string() + "'");
  }

  std::ranges::copy(buffer, data);
}

AppData::AppData(std::pmr::memory_resource *mr)
    : u_image(kImageSize, mr),
      u_conv1_weights(kConv1WeightSize, mr),
      u_conv1_bias(kConv1BiasSize, mr),
      u_conv1_out(kConv1OutSize, mr),
      u_pool1_out(kPool1OutSize, mr),
      u_conv2_weights(kConv2WeightSize, mr),
      u_conv2_bias(kConv2BiasSize, mr),
      u_conv2_out(kConv2OutSize, mr),
      u_pool2_out(kPool2OutSize, mr),
      u_conv3_weights(kConv3WeightSize, mr),
      u_conv3_bias(kConv3BiasSize, mr),
      u_conv3_out(kConv3OutSize, mr),
      u_conv4_weights(kConv4WeightSize, mr),
      u_conv4_bias(kConv4BiasSize, mr),
      u_conv4_out(kConv4OutSize, mr),
      u_conv5_weights(kConv5WeightSize, mr),
      u_conv5_bias(kConv5BiasSize, mr),
      u_conv5_out(kConv5OutSize, mr),
      u_pool3_out(kPool3OutSize, mr),
      u_linear_weights(kLinearWeightSize, mr),
      u_linear_bias(kLinearBiasSize, mr),
      u_linear_out(kLinearOutSize, mr) {
  // Load input image
  readDataFromFile("images/flattened_deer_deer_35.txt", u_image.data(), kImageSize);

  // Load conv1 parameters
  readDataFromFile("dense/features_0_weight.txt", u_conv1_weights.data(), kConv1WeightSize);
  readDataFromFile("dense/features_0_bias.txt", u_conv1_bias.data(), kConv1BiasSize);

  // Load conv2 parameters
  readDataFromFile("dense/features_3_weight.txt", u_conv2_weights.data(), kConv2WeightSize);
  readDataFromFile("dense/features_3_bias.txt", u_conv2_bias.data(), kConv2BiasSize);

  // Load conv3 parameters
  readDataFromFile("dense/features_6_weight.txt", u_conv3_weights.data(), kConv3WeightSize);
  readDataFromFile("dense/features_6_bias.txt", u_conv3_bias.data(), kConv3BiasSize);

  // Load conv4 parameters
  readDataFromFile("dense/features_8_weight.txt", u_conv4_weights.data(), kConv4WeightSize);
  readDataFromFile("dense/features_8_bias.txt", u_conv4_bias.data(), kConv4BiasSize);

  // Load conv5 parameters
  readDataFromFile("dense/features_10_weight.txt", u_conv5_weights.data(), kConv5WeightSize);
  readDataFromFile("dense/features_10_bias.txt", u_conv5_bias.data(), kConv5BiasSize);

  // Load linear parameters
  readDataFromFile("dense/classifier_weight.txt", u_linear_weights.data(), kLinearWeightSize);
  readDataFromFile("dense/classifier_bias.txt", u_linear_bias.data(), kLinearBiasSize);
}

}  // namespace cifar_dense