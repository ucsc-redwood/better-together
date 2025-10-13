#pragma once

#include <memory_resource>
#include <string>

namespace cuda {

std::string format_bytes(std::size_t bytes);

// ----------------------------------------------------------------------------
// CudaManagedResource
// ----------------------------------------------------------------------------

class CudaManagedResource final : public std::pmr::memory_resource {
 protected:
  void *do_allocate(std::size_t bytes, std::size_t alignment) override;
  void do_deallocate(void *p, std::size_t bytes, std::size_t alignment) override;
  bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override;
};

// ----------------------------------------------------------------------------
// CudaPinnedResource
// ----------------------------------------------------------------------------

class CudaPinnedResource final : public std::pmr::memory_resource {
 protected:
  void *do_allocate(std::size_t bytes, std::size_t alignment) override;
  void do_deallocate(void *p, std::size_t bytes, std::size_t alignment) override;
  bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override;
};

}  // namespace cuda
