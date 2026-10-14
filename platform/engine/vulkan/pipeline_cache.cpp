#include "pipeline_cache.hpp"

#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <new>
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

// Closes the descriptor on every path out of read_pipeline_cache_file, exceptions included.
struct FdCloser {
  int fd;
  ~FdCloser() { ::close(fd); }
};

// Reads exactly data.size() bytes into `data`; empty on success, else why.
std::string read_exactly(const int fd, const std::span<uint8_t> data) {
  size_t got = 0;
  while (got < data.size()) {
    const ssize_t n = ::read(fd, data.data() + got, data.size() - got);
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) return errno_text(fmt::format("read failed after {} of {} bytes", got, data.size()));
    if (n == 0) return fmt::format("short read: {} of {} bytes", got, data.size());
    got += static_cast<size_t>(n);
  }
  return {};
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

PipelineCacheFile read_pipeline_cache_file(const std::string& path,
                                           const vk::PhysicalDeviceProperties& props) {
  const auto rejected = [](std::string why) {
    return PipelineCacheFile{PipelineCacheLoad::kRejected, {}, std::move(why)};
  };

  // O_NONBLOCK: opening a FIFO (or another special file) at the path must not wait for a
  // writer; the type is checked on the opened descriptor below. No effect on regular files.
  const int fd = ::open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) {
    if (errno == ENOENT) return {PipelineCacheLoad::kMissing, {}, "no file"};
    return rejected(errno_text("cannot open"));
  }
  const FdCloser closer{fd};

  struct stat st{};
  if (::fstat(fd, &st) != 0) return rejected(errno_text("cannot stat"));
  if (!S_ISREG(st.st_mode)) return rejected("not a regular file");
  const auto size = static_cast<uint64_t>(st.st_size);

  // The 32-byte header first, so a foreign or corrupt file is rejected before the payload
  // is allocated.
  std::array<uint8_t, sizeof(VkPipelineCacheHeaderVersionOne)> header{};
  if (size < header.size()) {
    return rejected(fmt::format("{} bytes is shorter than the 32-byte header", size));
  }
  if (std::string why = read_exactly(fd, header); !why.empty()) return rejected(std::move(why));
  if (std::string why = pipeline_cache_header_mismatch(header, props); !why.empty()) {
    return rejected(std::move(why));
  }

  if (size > kMaxPipelineCacheBytes) {
    return rejected(
        fmt::format("{} bytes exceeds the {}-byte limit", size, kMaxPipelineCacheBytes));
  }
  std::vector<uint8_t> data;
  try {
    data.resize(static_cast<size_t>(size));
  } catch (const std::bad_alloc&) {
    return rejected(fmt::format("cannot allocate {} bytes", size));
  }
  std::copy(header.begin(), header.end(), data.begin());
  if (std::string why = read_exactly(fd, std::span(data).subspan(header.size())); !why.empty()) {
    return rejected(std::move(why));
  }
  return {PipelineCacheLoad::kLoaded, std::move(data), {}};
}

LoadedPipelineCache load_pipeline_cache(const vk::Device device,
                                        const vk::PhysicalDeviceProperties& props,
                                        const std::string& path) {
  const auto cold = [&](const PipelineCacheLoad load) {
    return LoadedPipelineCache{device.createPipelineCache(vk::PipelineCacheCreateInfo{}), load};
  };

  PipelineCacheFile file;
  try {
    file = read_pipeline_cache_file(path, props);
  } catch (const std::exception& e) {  // e.g. out of memory: never fail the Engine for a cache
    file = {PipelineCacheLoad::kRejected, {}, e.what()};
  }
  if (file.load == PipelineCacheLoad::kMissing) {
    spdlog::debug("Pipeline cache: no file {}, compiling cold", path);
    return cold(PipelineCacheLoad::kMissing);
  }
  if (file.load == PipelineCacheLoad::kRejected) {
    spdlog::debug("Pipeline cache: ignoring {} ({}), compiling cold", path, file.why);
    return cold(PipelineCacheLoad::kRejected);
  }
  const std::vector<uint8_t>& data = file.data;

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
