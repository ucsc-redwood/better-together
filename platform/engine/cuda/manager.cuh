#pragma once

#include "cu_mem_resource.cuh"
#include "helpers.cuh"

namespace cuda {

// ----------------------------------------------------------------------------
// CudaManager (owns a memory resource)
// ----------------------------------------------------------------------------
// The dispatchers run on the default stream (see cu_mem_resource.cu §1 note), so
// the previously-held cudaStream_ was created/destroyed but never used; dropped.

template <typename MemResourceT>
  requires std::is_same_v<MemResourceT, CudaManagedResource> ||
           std::is_same_v<MemResourceT, CudaPinnedResource>
class CudaManager {
 public:
  [[nodiscard]] MemResourceT &get_mr() { return mr_; }

 protected:
  MemResourceT mr_;
};

}  // namespace cuda
