#include "sparse_appdata.hpp"

#include <fstream>

#include "../resources_path.hpp"

namespace cifar_sparse {

void readDataFromFile(const char* filename, float* data, int maxSize) {
  const auto base_path = helpers::get_resource_base_path();
  std::ifstream file(base_path / filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open the file - '" + std::string(filename) + "'");
  }

  // Zero initialize the entire array
  for (int i = 0; i < maxSize; ++i) {
    data[i] = 0.0f;
  }

  // Read available values
  float value;
  int count = 0;
  while (file >> value && count < maxSize) {
    data[count++] = value;
  }

  file.close();
}

void readCSRFromFiles(const char* values_file,
                      const char* row_ptr_file,
                      const char* col_idx_file,
                      float* values,
                      int* row_ptr,
                      int* col_idx,
                      int nnz,
                      int rows) {
  readDataFromFile(values_file, values, nnz);

  std::ifstream row_file(row_ptr_file);
  for (int i = 0; i <= rows; ++i) {
    row_file >> row_ptr[i];
  }
  row_file.close();

  std::ifstream col_file(col_idx_file);
  for (int i = 0; i < nnz; ++i) {
    col_file >> col_idx[i];
  }
  col_file.close();
}

AppData::AppData(std::pmr::memory_resource* mr)
    : u_image_data(3 * 32 * 32, mr),

      // Conv1 arrays
      u_conv1_values(MAX_NNZ_CONV1, mr),
      u_conv1_row_ptr(65, mr),  // 64 + 1
      u_conv1_col_idx(MAX_NNZ_CONV1, mr),

      // Conv2 arrays
      u_conv2_values(MAX_NNZ_CONV2, mr),
      u_conv2_row_ptr(193, mr),  // 192 + 1
      u_conv2_col_idx(MAX_NNZ_CONV2, mr),

      // Conv3 arrays
      u_conv3_values(MAX_NNZ_CONV3, mr),
      u_conv3_row_ptr(385, mr),  // 384 + 1
      u_conv3_col_idx(MAX_NNZ_CONV3, mr),

      // Conv4 arrays
      u_conv4_values(MAX_NNZ_CONV4, mr),
      u_conv4_row_ptr(257, mr),  // 256 + 1
      u_conv4_col_idx(MAX_NNZ_CONV4, mr),

      // Conv5 arrays
      u_conv5_values(MAX_NNZ_CONV5, mr),
      u_conv5_row_ptr(257, mr),  // 256 + 1
      u_conv5_col_idx(MAX_NNZ_CONV5, mr),

      // Linear arrays
      u_linear_values(MAX_NNZ_LINEAR, mr),
      u_linear_row_ptr(11, mr),  // 10 + 1
      u_linear_col_idx(MAX_NNZ_LINEAR, mr),

      // Intermediate results
      u_conv1_output(64 * 32 * 32, mr),
      u_pool1_output(64 * 16 * 16, mr),
      u_conv2_output(192 * 16 * 16, mr),
      u_pool2_output(192 * 8 * 8, mr),
      u_conv3_output(384 * 8 * 8, mr),
      u_conv4_output(256 * 8 * 8, mr),
      u_conv5_output(256 * 8 * 8, mr),
      u_pool3_output(256 * 4 * 4, mr),
      u_linear_output(10, mr),

      // Biases
      u_conv1_bias(64, mr),
      u_conv2_bias(192, mr),
      u_conv3_bias(384, mr),
      u_conv4_bias(256, mr),
      u_conv5_bias(256, mr),
      u_linear_bias(10, mr) {
  // Load input image data
  readDataFromFile("images/flattened_dog_dog_13.txt", u_image_data.data(), 3072);

  // Load CSR data for all layers
  readCSRFromFiles("sparse/conv1_values.txt",
                   "sparse/conv1_row_ptr.txt",
                   "sparse/conv1_col_idx.txt",
                   u_conv1_values.data(),
                   u_conv1_row_ptr.data(),
                   u_conv1_col_idx.data(),
                   MAX_NNZ_CONV1,
                   64);

  readCSRFromFiles("sparse/conv2_values.txt",
                   "sparse/conv2_row_ptr.txt",
                   "sparse/conv2_col_idx.txt",
                   u_conv2_values.data(),
                   u_conv2_row_ptr.data(),
                   u_conv2_col_idx.data(),
                   MAX_NNZ_CONV2,
                   192);

  readCSRFromFiles("sparse/conv3_values.txt",
                   "sparse/conv3_row_ptr.txt",
                   "sparse/conv3_col_idx.txt",
                   u_conv3_values.data(),
                   u_conv3_row_ptr.data(),
                   u_conv3_col_idx.data(),
                   MAX_NNZ_CONV3,
                   384);

  readCSRFromFiles("sparse/conv4_values.txt",
                   "sparse/conv4_row_ptr.txt",
                   "sparse/conv4_col_idx.txt",
                   u_conv4_values.data(),
                   u_conv4_row_ptr.data(),
                   u_conv4_col_idx.data(),
                   MAX_NNZ_CONV4,
                   256);

  readCSRFromFiles("sparse/conv5_values.txt",
                   "sparse/conv5_row_ptr.txt",
                   "sparse/conv5_col_idx.txt",
                   u_conv5_values.data(),
                   u_conv5_row_ptr.data(),
                   u_conv5_col_idx.data(),
                   MAX_NNZ_CONV5,
                   256);

  readCSRFromFiles("sparse/linear_values.txt",
                   "sparse/linear_row_ptr.txt",
                   "sparse/linear_col_idx.txt",
                   u_linear_values.data(),
                   u_linear_row_ptr.data(),
                   u_linear_col_idx.data(),
                   MAX_NNZ_LINEAR,
                   10);

  // Load biases
  readDataFromFile("sparse/conv1_bias.txt", u_conv1_bias.data(), 64);
  readDataFromFile("sparse/conv2_bias.txt", u_conv2_bias.data(), 192);
  readDataFromFile("sparse/conv3_bias.txt", u_conv3_bias.data(), 384);
  readDataFromFile("sparse/conv4_bias.txt", u_conv4_bias.data(), 256);
  readDataFromFile("sparse/conv5_bias.txt", u_conv5_bias.data(), 256);
  readDataFromFile("sparse/linear_bias.txt", u_linear_bias.data(), 10);

  // Create CSR matrices
  conv1_weights = {
      u_conv1_values.data(), u_conv1_row_ptr.data(), u_conv1_col_idx.data(), 64, 27, MAX_NNZ_CONV1};
  conv2_weights = {u_conv2_values.data(),
                   u_conv2_row_ptr.data(),
                   u_conv2_col_idx.data(),
                   192,
                   576,
                   MAX_NNZ_CONV2};
  conv3_weights = {u_conv3_values.data(),
                   u_conv3_row_ptr.data(),
                   u_conv3_col_idx.data(),
                   384,
                   1728,
                   MAX_NNZ_CONV3};
  conv4_weights = {u_conv4_values.data(),
                   u_conv4_row_ptr.data(),
                   u_conv4_col_idx.data(),
                   256,
                   3456,
                   MAX_NNZ_CONV4};
  conv5_weights = {u_conv5_values.data(),
                   u_conv5_row_ptr.data(),
                   u_conv5_col_idx.data(),
                   256,
                   2304,
                   MAX_NNZ_CONV5};
  linear_weights = {u_linear_values.data(),
                    u_linear_row_ptr.data(),
                    u_linear_col_idx.data(),
                    10,
                    4096,
                    MAX_NNZ_LINEAR};
}

}  // namespace cifar_sparse