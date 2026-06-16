#include <spdlog/spdlog.h>

#include "cu_mem_resource.cuh"
#include "helpers.cuh"

namespace cuda {

std::string format_bytes(std::size_t bytes) {
  constexpr std::size_t KB = 1024;
  constexpr std::size_t MB = KB * 1024;
  constexpr std::size_t GB = MB * 1024;

  if (bytes >= GB) {
    return fmt::format("{:.2f} GB", static_cast<double>(bytes) / GB);
  } else if (bytes >= MB) {
    return fmt::format("{:.2f} MB", static_cast<double>(bytes) / MB);
  } else if (bytes >= KB) {
    return fmt::format("{:.2f} KB", static_cast<double>(bytes) / KB);
  }
  return fmt::format("{} bytes", bytes);
}

// ----------------------------------------------------------------------------
// CudaManagedResource
// ----------------------------------------------------------------------------

void *CudaManagedResource::do_allocate(std::size_t bytes, std::size_t /*alignment*/) {
  void *ptr = nullptr;
  // NOTE: cudaMemAttachHost is deliberate — it lets a CPU pipeline thread access
  // these buffers concurrently with GPU kernels running on other items (the
  // CPU+GPU hybrid that is the point of the paper) on Tegra/Jetson, where
  // concurrentManagedAccess=0 forbids host access to *global*-attached managed
  // memory while any kernel runs. BUT the other half is missing: the GPU side
  // must cudaStreamAttachMemAsync(stream, ptr, 0, cudaMemAttachSingle) before a
  // kernel uses a buffer and launch that kernel on `stream` — this code never
  // does, and kernels run on the default stream, so GPU access is racy and the
  // host reads partial output (see docs/BUGS-FOUND.md §1). Do NOT "fix" this by
  // switching to global attach: that corrects sequential GPU output but faults
  // the concurrent CPU thread. The real fix is stream-attach + per-stream launch.
  cudaError_t err = cudaMallocManaged(&ptr, bytes, cudaMemAttachHost);
  if (err != cudaSuccess) {
    throw std::bad_alloc();
  }

  spdlog::trace(
      "CudaManagedResource::do_allocate: {}, {}", static_cast<void *>(ptr), format_bytes(bytes));

  return ptr;
}

void CudaManagedResource::do_deallocate(void *p, std::size_t /*bytes*/, std::size_t /*alignment*/) {
  spdlog::trace("CudaManagedResource::do_deallocate: {}", static_cast<void *>(p));
  CheckCuda(cudaFree(p));
}

bool CudaManagedResource::do_is_equal(const std::pmr::memory_resource &other) const noexcept {
  return this == &other;
}

// ----------------------------------------------------------------------------
// CudaPinnedResource
// ----------------------------------------------------------------------------

void *CudaPinnedResource::do_allocate(std::size_t bytes, std::size_t /*alignment*/) {
  void *h_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&h_ptr, bytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    throw std::bad_alloc();
  }

  void *d_ptr = nullptr;
  err = cudaHostGetDevicePointer(&d_ptr, h_ptr, 0);
  if (err != cudaSuccess) {
    throw std::bad_alloc();
  }

  spdlog::trace(
      "CudaPinnedResource::do_allocate: {}, {}", static_cast<void *>(d_ptr), format_bytes(bytes));

  return d_ptr;
}

void CudaPinnedResource::do_deallocate(void *p, std::size_t /*bytes*/, std::size_t /*alignment*/) {
  spdlog::trace("CudaPinnedResource::do_deallocate: {}", static_cast<void *>(p));
  CheckCuda(cudaFreeHost(p));
}

bool CudaPinnedResource::do_is_equal(const std::pmr::memory_resource &other) const noexcept {
  return this == &other;
}

}  // namespace cuda