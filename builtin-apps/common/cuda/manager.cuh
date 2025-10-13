#pragma once

#include "cu_mem_resource.cuh"
#include "helpers.cuh"

namespace cuda {

// ----------------------------------------------------------------------------
// CudaManager (contains a stream and a memory resource)
// ----------------------------------------------------------------------------

template <typename MemResourceT>
  requires std::is_same_v<MemResourceT, CudaManagedResource> ||
           std::is_same_v<MemResourceT, CudaPinnedResource>
class CudaManager {
 public:
  CudaManager() { CheckCuda(cudaStreamCreate(&stream_)); }

  ~CudaManager() { CheckCuda(cudaStreamDestroy(stream_)); }

  [[nodiscard]] cudaStream_t get_stream() const { return stream_; }

  [[nodiscard]] MemResourceT &get_mr() { return mr_; }

 protected:
  cudaStream_t stream_;
  MemResourceT mr_;
};

}  // namespace cuda
