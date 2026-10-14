#pragma once

#include <cstdint>
#include <span>
#include <string>

#include "vk.hpp"

namespace kiss_vk {

// How the Engine's pipeline cache started (design D7).
enum class PipelineCacheLoad {
  kLoaded,    // seeded from the file: pipelines may come warm
  kMissing,   // no readable file: cold compile
  kRejected,  // file failed the header check (or the driver refused it): cold compile
};

// <dir>/bt-vk-pipeline-cache-<vendorID>-<deviceID>-<pipelineCacheUUID>.bin (hex), where
// <dir> is $BT_VK_PIPELINE_CACHE_DIR, default the current directory.
[[nodiscard]] std::string pipeline_cache_path(const vk::PhysicalDeviceProperties& props);

// Empty if `data` starts with a VkPipelineCacheHeaderVersionOne for this device and driver
// (header length and version, vendorID, deviceID, pipelineCacheUUID); otherwise the reason
// it does not. Not every mobile driver can be trusted to reject a foreign blob itself.
[[nodiscard]] std::string pipeline_cache_header_mismatch(std::span<const uint8_t> data,
                                                         const vk::PhysicalDeviceProperties& props);

struct LoadedPipelineCache {
  vk::PipelineCache cache;
  PipelineCacheLoad load;
};

// A pipeline cache seeded from `path` when the file passes the header check, else empty.
[[nodiscard]] LoadedPipelineCache load_pipeline_cache(vk::Device device,
                                                      const vk::PhysicalDeviceProperties& props,
                                                      const std::string& path);

// Writes the cache to `path` via a temporary file and a rename. An I/O error (e.g. a
// read-only directory) is logged at debug level and otherwise ignored.
void save_pipeline_cache(vk::Device device,
                         vk::PipelineCache cache,
                         const std::string& path) noexcept;

}  // namespace kiss_vk
