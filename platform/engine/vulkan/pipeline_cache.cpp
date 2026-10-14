#include "pipeline_cache.hpp"

#include <spdlog/spdlog.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace kiss_vk {

std::string pipeline_cache_path(const vk::PhysicalDeviceProperties& props) {
  const char* dir = std::getenv("BT_VK_PIPELINE_CACHE_DIR");
  std::string uuid;
  for (const uint8_t b : props.pipelineCacheUUID) uuid += fmt::format("{:02x}", b);
  return fmt::format("{}/bt-vk-pipeline-cache-{:04x}-{:04x}-{}.bin",
                     dir && *dir ? dir : ".",
                     props.vendorID,
                     props.deviceID,
                     uuid);
}

std::string pipeline_cache_header_mismatch(const std::span<const uint8_t> data,
                                           const vk::PhysicalDeviceProperties& props) {
  // The header's fields are little-endian, as is every fleet target.
  VkPipelineCacheHeaderVersionOne header;
  static_assert(sizeof(header) == 32);
  if (data.size() < sizeof(header)) {
    return fmt::format("{} bytes is shorter than the 32-byte header", data.size());
  }
  std::memcpy(&header, data.data(), sizeof(header));
  if (header.headerSize != sizeof(header)) {
    return fmt::format("headerSize {}, expected 32", header.headerSize);
  }
  if (header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE) {
    return fmt::format("headerVersion {}, expected 1", static_cast<uint32_t>(header.headerVersion));
  }
  if (header.vendorID != props.vendorID) {
    return fmt::format("vendorID {:#06x}, device has {:#06x}", header.vendorID, props.vendorID);
  }
  if (header.deviceID != props.deviceID) {
    return fmt::format("deviceID {:#06x}, device has {:#06x}", header.deviceID, props.deviceID);
  }
  if (std::memcmp(header.pipelineCacheUUID, props.pipelineCacheUUID.data(), VK_UUID_SIZE) != 0) {
    return "pipelineCacheUUID differs from the driver's";
  }
  return {};
}

LoadedPipelineCache load_pipeline_cache(const vk::Device device,
                                        const vk::PhysicalDeviceProperties& props,
                                        const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    spdlog::debug("Pipeline cache: no readable {}, compiling cold", path);
    return {device.createPipelineCache(vk::PipelineCacheCreateInfo{}), PipelineCacheLoad::kMissing};
  }
  const std::vector<uint8_t> data{std::istreambuf_iterator<char>(file),
                                  std::istreambuf_iterator<char>()};

  if (const std::string why = pipeline_cache_header_mismatch(data, props); !why.empty()) {
    spdlog::debug("Pipeline cache: ignoring {} ({}), compiling cold", path, why);
    return {device.createPipelineCache(vk::PipelineCacheCreateInfo{}),
            PipelineCacheLoad::kRejected};
  }

  try {
    const vk::PipelineCache cache = device.createPipelineCache(vk::PipelineCacheCreateInfo{
        .initialDataSize = data.size(),
        .pInitialData = data.data(),
    });
    spdlog::debug("Pipeline cache: loaded {} bytes from {}", data.size(), path);
    return {cache, PipelineCacheLoad::kLoaded};
  } catch (const vk::SystemError& e) {
    spdlog::debug("Pipeline cache: driver refused {} ({}), compiling cold", path, e.what());
    return {device.createPipelineCache(vk::PipelineCacheCreateInfo{}),
            PipelineCacheLoad::kRejected};
  }
}

void save_pipeline_cache(const vk::Device device,
                         const vk::PipelineCache cache,
                         const std::string& path) noexcept {
  const std::string tmp = path + ".tmp." + std::to_string(getpid());
  try {
    const std::vector<uint8_t> data = device.getPipelineCacheData(cache);
    {
      std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
      file.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size()));
      if (!file.good()) throw std::runtime_error("cannot write " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
      throw std::runtime_error("cannot rename " + tmp + ": " + std::strerror(errno));
    }
    spdlog::debug("Pipeline cache: saved {} bytes to {}", data.size(), path);
  } catch (const std::exception& e) {
    std::remove(tmp.c_str());
    spdlog::debug("Pipeline cache: not saved ({}), continuing", e.what());
  }
}

}  // namespace kiss_vk
