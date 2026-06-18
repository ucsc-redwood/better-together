#pragma once

#include <memory_resource>
#include <string>

namespace cuda {

std::string format_bytes(std::size_t bytes);

// ----------------------------------------------------------------------------
// CudaPinnedResource -- zero-copy mapped pinned host memory (cudaHostAllocMapped +
// cudaHostGetDevicePointer). On the Jetson UMA it is physically shared/coherent and
// stays host-accessible CONCURRENTLY with GPU kernels, which is what the §1 visibility
// requirement needs. (We deliberately do NOT offer a cudaMallocManaged resource: with
// cudaMemAttachHost it is the §1 partial-visibility defect on Tegra, concurrentManaged
// Access=0, and switching it to global attach faults the concurrent CPU thread. See
// docs/reports-for-human/bugs-found.md §1 and the git history for the removed class.)
// ----------------------------------------------------------------------------

class CudaPinnedResource final : public std::pmr::memory_resource {
 protected:
  void *do_allocate(std::size_t bytes, std::size_t alignment) override;
  void do_deallocate(void *p, std::size_t bytes, std::size_t alignment) override;
  bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override;
};

}  // namespace cuda
