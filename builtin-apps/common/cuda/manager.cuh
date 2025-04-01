#pragma once

#include "cu_mem_resource.cuh"
#include "helpers.cuh"

namespace cuda {

// ----------------------------------------------------------------------------
// CudaManager (contains a stream and a memory resource)
// ----------------------------------------------------------------------------

class CudaManager {
 public:
  CudaManager() { CheckCuda(cudaStreamCreate(&stream_)); }

  ~CudaManager() { CheckCuda(cudaStreamDestroy(stream_)); }

  [[nodiscard]] cudaStream_t get_stream() const { return stream_; }

  [[nodiscard]] CudaManagedResource &get_mr() { return mr_; }

 protected:
  cudaStream_t stream_;
  CudaManagedResource mr_;
};

}  // namespace cuda
