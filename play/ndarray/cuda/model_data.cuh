#pragma once

#include "../appdata.hpp"
#include "../ndarray.hpp"
#include "builtin-apps/common/cuda/helpers.cuh"

namespace cuda {

struct DeviceModelData {
  explicit DeviceModelData(const cifar_dense::ModelData& h_model_data)
      : h_model_ref(h_model_data) {
    // Allocate device memory

    // const auto& conv1_w_shape = h_model_data.h_conv1_w.shape();
    // const auto& conv1_b_shape = h_model_data.h_conv1_b.shape();
    // const auto& conv2_w_shape = h_model_data.h_conv2_w.shape();
    // const auto& conv2_b_shape = h_model_data.h_conv2_b.shape();
    // const auto& conv3_w_shape = h_model_data.h_conv3_w.shape();
    // const auto& conv3_b_shape = h_model_data.h_conv3_b.shape();
    // const auto& conv4_w_shape = h_model_data.h_conv4_w.shape();
    // const auto& conv4_b_shape = h_model_data.h_conv4_b.shape();
    // const auto& conv5_w_shape = h_model_data.h_conv5_w.shape();
    // const auto& conv5_b_shape = h_model_data.h_conv5_b.shape();
    // const auto& linear_w_shape = h_model_data.h_linear_w.shape();
    // const auto& linear_b_shape = h_model_data.h_linear_b.shape();

    CheckCuda(cudaMalloc(&d_conv1_w, h_model_data.h_conv1_w.memory_usage_bytes()));
    CheckCuda(cudaMalloc(&d_conv1_b, h_model_data.h_conv1_b.memory_usage_bytes()));
    CheckCuda(cudaMalloc(&d_conv2_w, h_model_data.h_conv2_w.memory_usage_bytes()));
    CheckCuda(cudaMalloc(&d_conv2_b, h_model_data.h_conv2_b.memory_usage_bytes()));
    CheckCuda(cudaMalloc(&d_conv3_w, h_model_data.h_conv3_w.memory_usage_bytes()));
    CheckCuda(cudaMalloc(&d_conv3_b, h_model_data.h_conv3_b.memory_usage_bytes()));
    CheckCuda(cudaMalloc(&d_conv4_w, h_model_data.h_conv4_w.memory_usage_bytes()));
    CheckCuda(cudaMalloc(&d_conv4_b, h_model_data.h_conv4_b.memory_usage_bytes()));
    CheckCuda(cudaMalloc(&d_conv5_w, h_model_data.h_conv5_w.memory_usage_bytes()));
    CheckCuda(cudaMalloc(&d_conv5_b, h_model_data.h_conv5_b.memory_usage_bytes()));
    CheckCuda(cudaMalloc(&d_linear_w, h_model_data.h_linear_w.memory_usage_bytes()));
    CheckCuda(cudaMalloc(&d_linear_b, h_model_data.h_linear_b.memory_usage_bytes()));

    CheckCuda(cudaMemcpy(d_conv1_w, h_model_data.h_conv1_w.raw(), h_model_data.h_conv1_w.memory_usage_bytes(), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_conv1_b, h_model_data.h_conv1_b.raw(), h_model_data.h_conv1_b.memory_usage_bytes(), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_conv2_w, h_model_data.h_conv2_w.raw(), h_model_data.h_conv2_w.memory_usage_bytes(), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_conv2_b, h_model_data.h_conv2_b.raw(), h_model_data.h_conv2_b.memory_usage_bytes(), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_conv3_w, h_model_data.h_conv3_w.raw(), h_model_data.h_conv3_w.memory_usage_bytes(), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_conv3_b, h_model_data.h_conv3_b.raw(), h_model_data.h_conv3_b.memory_usage_bytes(), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_conv4_w, h_model_data.h_conv4_w.raw(), h_model_data.h_conv4_w.memory_usage_bytes(), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_conv4_b, h_model_data.h_conv4_b.raw(), h_model_data.h_conv4_b.memory_usage_bytes(), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_conv5_w, h_model_data.h_conv5_w.raw(), h_model_data.h_conv5_w.memory_usage_bytes(), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_conv5_b, h_model_data.h_conv5_b.raw(), h_model_data.h_conv5_b.memory_usage_bytes(), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_linear_w, h_model_data.h_linear_w.raw(), h_model_data.h_linear_w.memory_usage_bytes(), cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(d_linear_b, h_model_data.h_linear_b.raw(), h_model_data.h_linear_b.memory_usage_bytes(), cudaMemcpyHostToDevice));

    CheckCuda(cudaDeviceSynchronize());
  }

  ~DeviceModelData() {
    CheckCuda(cudaFree(d_conv1_w));
    CheckCuda(cudaFree(d_conv1_b));
    CheckCuda(cudaFree(d_conv2_w));
    CheckCuda(cudaFree(d_conv2_b));
    CheckCuda(cudaFree(d_conv3_w));
    CheckCuda(cudaFree(d_conv3_b));
    CheckCuda(cudaFree(d_conv4_w));
    CheckCuda(cudaFree(d_conv4_b));
    CheckCuda(cudaFree(d_conv5_w));
    CheckCuda(cudaFree(d_conv5_b));
    CheckCuda(cudaFree(d_linear_w));
    CheckCuda(cudaFree(d_linear_b));
    CheckCuda(cudaDeviceSynchronize());
  }

//   int w_shape_2() const { return conv1_w_shape[2]; }
  const cifar_dense::ModelData& h_model_ref;

  float* d_conv1_w;
  float* d_conv1_b;
  float* d_conv2_w;
  float* d_conv2_b;
  float* d_conv3_w;
  float* d_conv3_b;
  float* d_conv4_w;
  float* d_conv4_b;
  float* d_conv5_w;
  float* d_conv5_b;
  float* d_linear_w;
  float* d_linear_b;
};

}  // namespace cuda
