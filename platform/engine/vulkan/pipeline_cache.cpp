#include "pipeline_cache.hpp"

#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
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

namespace {

std::string errno_text(const std::string& what) { return what + ": " + std::strerror(errno); }

enum class ReadResult { kOk, kMissing, kFailed };

// The whole regular file at `path`. kMissing if it does not exist; any other failure (not
// a regular file, open/read error, a read that ends early) is kFailed with the reason.
ReadResult read_regular_file(const std::string& path,
                             std::vector<uint8_t>& data,
                             std::string& why) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    if (errno == ENOENT) return ReadResult::kMissing;
    why = errno_text("cannot open");
    return ReadResult::kFailed;
  }
  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    why = errno_text("cannot stat");
  } else if (!S_ISREG(st.st_mode)) {
    why = "not a regular file";
  } else {
    data.resize(static_cast<size_t>(st.st_size));
    size_t got = 0;
    while (why.empty() && got < data.size()) {
      const ssize_t n = ::read(fd, data.data() + got, data.size() - got);
      if (n < 0 && errno == EINTR) continue;
      if (n < 0) {
        why = errno_text(fmt::format("read failed after {} of {} bytes", got, data.size()));
      } else if (n == 0) {
        why = fmt::format("short read: {} of {} bytes", got, data.size());
      } else {
        got += static_cast<size_t>(n);
      }
    }
  }
  ::close(fd);
  return why.empty() ? ReadResult::kOk : ReadResult::kFailed;
}

}  // namespace

std::string write_file_atomically(const std::string& path, const std::span<const uint8_t> data) {
  const std::string tmp = path + ".tmp." + std::to_string(getpid());
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
  if (fd < 0) return errno_text("cannot create " + tmp);

  std::string why;
  size_t done = 0;
  while (why.empty() && done < data.size()) {
    const ssize_t n = ::write(fd, data.data() + done, data.size() - done);
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) {
      why = errno_text(fmt::format("write failed after {} of {} bytes", done, data.size()));
    } else if (n == 0) {
      why = fmt::format("write failed after {} of {} bytes: wrote nothing", done, data.size());
    } else {
      done += static_cast<size_t>(n);
    }
  }
  if (::close(fd) != 0 && why.empty()) why = errno_text("close failed");
  if (why.empty() && ::rename(tmp.c_str(), path.c_str()) != 0) {
    why = errno_text("cannot rename to " + path);
  }
  if (!why.empty()) ::unlink(tmp.c_str());
  return why;
}

LoadedPipelineCache load_pipeline_cache(const vk::Device device,
                                        const vk::PhysicalDeviceProperties& props,
                                        const std::string& path) {
  const auto cold = [&](const PipelineCacheLoad load) {
    return LoadedPipelineCache{device.createPipelineCache(vk::PipelineCacheCreateInfo{}), load};
  };

  std::vector<uint8_t> data;
  std::string why;
  switch (read_regular_file(path, data, why)) {
    case ReadResult::kMissing:
      spdlog::debug("Pipeline cache: no file {}, compiling cold", path);
      return cold(PipelineCacheLoad::kMissing);
    case ReadResult::kFailed:
      spdlog::debug("Pipeline cache: ignoring {} ({}), compiling cold", path, why);
      return cold(PipelineCacheLoad::kRejected);
    case ReadResult::kOk:
      break;
  }

  if (why = pipeline_cache_header_mismatch(data, props); !why.empty()) {
    spdlog::debug("Pipeline cache: ignoring {} ({}), compiling cold", path, why);
    return cold(PipelineCacheLoad::kRejected);
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
    return cold(PipelineCacheLoad::kRejected);
  }
}

void save_pipeline_cache(const vk::Device device,
                         const vk::PipelineCache cache,
                         const std::string& path) noexcept {
  try {
    const std::vector<uint8_t> data = device.getPipelineCacheData(cache);
    if (const std::string why = write_file_atomically(path, data); !why.empty()) {
      spdlog::debug("Pipeline cache: not saved to {} ({}), previous file kept", path, why);
      return;
    }
    spdlog::debug("Pipeline cache: saved {} bytes to {}", data.size(), path);
  } catch (const std::exception& e) {
    spdlog::debug("Pipeline cache: not saved to {} ({}), previous file kept", path, e.what());
  }
}

}  // namespace kiss_vk
