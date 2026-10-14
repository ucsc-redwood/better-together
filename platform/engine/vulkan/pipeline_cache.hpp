#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "vk.hpp"

namespace kiss_vk {

// How the Engine's pipeline cache started (design D7).
enum class PipelineCacheLoad {
  kLoaded,    // seeded from the file: pipelines may come warm
  kMissing,   // no file: cold compile
  kRejected,  // file unreadable, failed the header check, or refused by the driver: cold
};

// <dir>/bt-vk-pipeline-cache-<vendorID>-<deviceID>-<pipelineCacheUUID>.bin (hex), where
// <dir> is $BT_VK_PIPELINE_CACHE_DIR, default the current directory.
[[nodiscard]] std::string pipeline_cache_path(const vk::PhysicalDeviceProperties& props);

// Empty if `data` starts with a VkPipelineCacheHeaderVersionOne for this device and driver
// (header length and version, vendorID, deviceID, pipelineCacheUUID); otherwise the reason
// it does not. Not every mobile driver can be trusted to reject a foreign blob itself.
[[nodiscard]] std::string pipeline_cache_header_mismatch(std::span<const uint8_t> data,
                                                         const vk::PhysicalDeviceProperties& props);

// A cache file larger than this is rejected without being read. Real caches are far smaller
// (at most 63 KB on any fleet device in Phase C), so this only stops a damaged or hostile
// file from costing the run a huge allocation (design D7).
inline constexpr uint64_t kMaxPipelineCacheBytes = 64ull << 20;

// The contents of a cache file, or why there are none: kMissing (no file) or kRejected (not a
// regular file -- a FIFO is never waited on --, unreadable, short, a foreign or corrupt
// header, over kMaxPipelineCacheBytes, or not allocatable). The header is checked before
// the payload is allocated. kLoaded data has passed the header check.
struct PipelineCacheFile {
  PipelineCacheLoad load = PipelineCacheLoad::kMissing;
  std::vector<uint8_t> data;
  std::string why;
};
[[nodiscard]] PipelineCacheFile read_pipeline_cache_file(const std::string& path,
                                                         const vk::PhysicalDeviceProperties& props);

struct LoadedPipelineCache {
  vk::PipelineCache cache;
  PipelineCacheLoad load;
};

// A pipeline cache seeded from `path` when the file passes the header check, else empty.
[[nodiscard]] LoadedPipelineCache load_pipeline_cache(vk::Device device,
                                                      const vk::PhysicalDeviceProperties& props,
                                                      const std::string& path);

// Writes `data` to `<path>.tmp.<pid>` (every write and the close checked), then renames it
// over `path`. Returns the empty string on success; otherwise why, with the temporary file
// removed and any previous `path` left untouched.
[[nodiscard]] std::string write_file_atomically(const std::string& path,
                                                std::span<const uint8_t> data);

// Writes the cache to `path` with write_file_atomically. A failure (e.g. a read-only
// directory or a full disk) is logged at debug level and otherwise ignored.
void save_pipeline_cache(vk::Device device,
                         vk::PipelineCache cache,
                         const std::string& path) noexcept;

}  // namespace kiss_vk
