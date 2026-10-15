#include <fcntl.h>
#include <gtest/gtest.h>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/spdlog.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <thread>

#include "engine.hpp"

TEST(VulkanEngineTest, EngineInitialization) {
  // Test that engine can be initialized
  EXPECT_NO_THROW({ kiss_vk::Engine engine; });
}

TEST(VulkanEngineTest, SequenceCreation) {
  kiss_vk::Engine engine;

  // Test that sequence can be created
  EXPECT_NO_THROW({ auto seq = engine.make_seq(); });
}

// Device-init spec, "Fail loudly on a missing required feature": the check names exactly
// the feature the device lacks. Starts from the real iGPU's supported set, so it skips
// (like the differential suites) on a box without one.
TEST(VulkanDeviceFeatures, MissingShaderInt8IsReported) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  kiss_vk::Engine engine;

  vk::PhysicalDeviceVulkan13Features supported_13{};
  vk::PhysicalDeviceVulkan12Features supported{.pNext = &supported_13};
  vk::PhysicalDeviceFeatures2 features2{.pNext = &supported};
  engine.get_physical_device().getFeatures2(&features2);
  EXPECT_TRUE(kiss_vk::missing_required_features(supported, supported_13).empty());

  supported.shaderInt8 = VK_FALSE;
  EXPECT_EQ(kiss_vk::missing_required_features(supported, supported_13),
            std::vector<std::string>{"shaderInt8"});

  // maintenance4: required for LocalSizeId (VUID-RuntimeSpirv-LocalSizeId-06434).
  supported.shaderInt8 = VK_TRUE;
  supported_13.maintenance4 = VK_FALSE;
  EXPECT_EQ(kiss_vk::missing_required_features(supported, supported_13),
            std::vector<std::string>{"maintenance4"});

  // synchronization2: the Sequence's barriers and timestamps (design A1).
  supported_13.maintenance4 = VK_TRUE;
  supported_13.synchronization2 = VK_FALSE;
  EXPECT_EQ(kiss_vk::missing_required_features(supported, supported_13),
            std::vector<std::string>{"synchronization2"});
}

// Pipeline-config (design D5): the SPIR-V scan reads spec ids and workgroup size from
// committed *_spv.h. Hardware-free.
TEST(SpirvReflection, CommittedShaders) {
  const auto radix = kiss_vk::reflect_spirv(kiss_vk::load_shader_spirv("multi_radixsort"));
  EXPECT_EQ(radix.spec_ids, std::set<uint32_t>{kiss_vk::kSubgroupSizeSpecId});
  EXPECT_EQ(radix.local_size, (std::array<uint32_t, 3>{256, 1, 1}));
  for (const auto& id : radix.local_size_spec_id) EXPECT_FALSE(id.has_value());

  const auto edge = kiss_vk::reflect_spirv(kiss_vk::load_shader_spirv("tree_edge_count"));
  EXPECT_TRUE(edge.spec_ids.empty());
  EXPECT_EQ(edge.local_size, (std::array<uint32_t, 3>{512, 1, 1}));

  EXPECT_THROW((void)kiss_vk::reflect_spirv(std::vector<uint32_t>{1, 2, 3, 4, 5}),
               std::runtime_error);
}

// SPIR-V 1.6 `local_size_*_id`: OpExecutionModeId LocalSizeId over spec constants.
TEST(SpirvReflection, LocalSizeId) {
  const auto r = kiss_vk::reflect_spirv(kiss_vk::load_shader_spirv("engine_test_workgroup_size"));
  EXPECT_EQ(r.spec_ids, (std::set<uint32_t>{100, 101, 102}));
  EXPECT_EQ(r.local_size, (std::array<uint32_t, 3>{1, 1, 1}));  // the spec constants' defaults
  EXPECT_EQ(r.local_size_spec_id[0], 100u);
  EXPECT_EQ(r.local_size_spec_id[1], 101u);
  EXPECT_EQ(r.local_size_spec_id[2], 102u);
}

// Pipeline-config spec, "Declared workgroup size is applied to the shader".
namespace {
std::string build_error(const std::shared_ptr<kiss_vk::Algorithm>& algo) {
  try {
    (void)algo->build();
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}
std::string dims(const auto& s) { return fmt::format("({}, {}, {})", s[0], s[1], s[2]); }
}  // namespace

TEST(VulkanWorkGroupSize, DeclaredSizeReachesShader) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  kiss_vk::Engine engine;
  auto seq = engine.make_seq();
  std::pmr::vector<uint32_t> out(4, 0, engine.get_mr());

  auto algo = engine.make_algo("engine_test_workgroup_size")
                  ->work_group_size(64, 2, 1)
                  ->num_buffers(1)
                  ->build();

  seq->cmd_begin();
  algo->record_bind(*seq, {engine.get_buffer_info(out)});
  algo->record_dispatch(*seq, {1, 1, 1});
  seq->cmd_end();
  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();

  EXPECT_EQ(out[0], 128u);  // invocations in the one workgroup
  EXPECT_EQ(out[1], 64u);
  EXPECT_EQ(out[2], 2u);
  EXPECT_EQ(out[3], 1u);
}

// Sets (or, with nullptr, unsets) an environment variable for one scope.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) old_ = old;
    if (value) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }
  ~ScopedEnv() {
    if (old_) {
      setenv(name_, old_->c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;
  ScopedEnv(ScopedEnv&&) = delete;
  ScopedEnv& operator=(ScopedEnv&&) = delete;

 private:
  const char* name_;
  std::optional<std::string> old_;
};

// Resource-binding spec, "Bindings are recorded per dispatch" and "Portable binding
// fallback", on both binding paths of the same device: push descriptors (where the device
// has VK_KHR_push_descriptor) and the transient-pool fallback (BT_VK_FORCE_DESCRIPTOR_FALLBACK=1,
// even where it has). One kernel is recorded many times into one command buffer, each
// time bound to a different buffer. The shader atomically adds its 128 invocations to
// g_out[0], so each buffer must end at its own start value plus 128 per dispatch bound to
// it: a dispatch that saw another dispatch's binding would read and write the wrong
// buffer. 75 binds also exceed one transient pool.
enum class BindingPath : std::uint8_t { kPushDescriptors, kFallback };

class VulkanBinding : public ::testing::TestWithParam<BindingPath> {};

TEST_P(VulkanBinding, SameKernelManyBuffersInOneCommandBuffer) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  const bool fallback = GetParam() == BindingPath::kFallback;
  // Explicit either way, so an outer BT_VK_FORCE_DESCRIPTOR_FALLBACK=1 still tests push.
  const ScopedEnv force("BT_VK_FORCE_DESCRIPTOR_FALLBACK", fallback ? "1" : nullptr);
  kiss_vk::Engine engine;
  if (fallback) {
    ASSERT_EQ(engine.push_descriptor_fn(), nullptr);
  } else if (engine.push_descriptor_fn() == nullptr) {
    GTEST_SKIP() << "VK_KHR_push_descriptor not supported";
  }

  auto seq = engine.make_seq();
  auto algo = engine.make_algo("engine_test_workgroup_size")
                  ->work_group_size(64, 2, 1)
                  ->num_buffers(1)
                  ->build();

  constexpr int kBuffers = 3;
  constexpr int kRounds = 25;
  std::vector<std::pmr::vector<uint32_t>> outs;
  outs.reserve(kBuffers);
  for (int b = 0; b < kBuffers; ++b) outs.emplace_back(4, 0, engine.get_mr());

  for (int submit = 0; submit < 2; ++submit) {  // the second cmd_begin() reuses the pools
    for (int b = 0; b < kBuffers; ++b) {
      std::ranges::fill(outs[b], 0u);
      outs[b][0] = 1000u * (b + 1);
    }
    seq->cmd_begin();
    for (int i = 0; i < kRounds * kBuffers; ++i) {
      algo->record_bind(*seq, {engine.get_buffer_info(outs[i % kBuffers])});
      algo->record_dispatch(*seq, {1, 1, 1});
    }
    seq->cmd_end();
    seq->submit();
    seq->wait_for_fence();
    seq->reset_fence();

    for (int b = 0; b < kBuffers; ++b) {
      EXPECT_EQ(outs[b][0], 1000u * (b + 1) + 128u * kRounds) << "buffer " << b;
      EXPECT_EQ(outs[b][1], 64u) << "buffer " << b;
      EXPECT_EQ(outs[b][2], 2u) << "buffer " << b;
      EXPECT_EQ(outs[b][3], 1u) << "buffer " << b;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(BothPaths,
                         VulkanBinding,
                         ::testing::Values(BindingPath::kPushDescriptors, BindingPath::kFallback),
                         [](const auto& info) {
                           return info.param == BindingPath::kFallback ? "Fallback"
                                                                       : "PushDescriptors";
                         });

// Pipeline-config spec, "Compiled pipelines persist across runs" (design D7). Each test
// gets its own cache directory under the working directory (writable on every target),
// removed afterwards.
class PipelineCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
    dir_ = std::filesystem::current_path() /
           ("bt-vk-cache-test-" + std::to_string(getpid()) + "-" +
            ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::create_directory(dir_);
    env_.emplace("BT_VK_PIPELINE_CACHE_DIR", dir_.c_str());
  }
  void TearDown() override {
    env_.reset();
    if (dir_.empty()) return;
    std::filesystem::permissions(dir_, std::filesystem::perms::owner_all);
    std::filesystem::remove_all(dir_);
  }

  // One engine lifetime: build a pipeline from the persisted cache, run it, check the
  // result, save the cache on destruction. Returns how the cache started.
  static kiss_vk::PipelineCacheLoad run_once(std::string* path = nullptr) {
    kiss_vk::Engine engine;
    auto seq = engine.make_seq();
    std::pmr::vector<uint32_t> out(4, 0, engine.get_mr());
    auto algo = engine.make_algo("engine_test_workgroup_size")
                    ->work_group_size(64, 2, 1)
                    ->num_buffers(1)
                    ->build();
    seq->cmd_begin();
    algo->record_bind(*seq, {engine.get_buffer_info(out)});
    algo->record_dispatch(*seq, {1, 1, 1});
    seq->cmd_end();
    seq->submit();
    seq->wait_for_fence();
    seq->reset_fence();
    EXPECT_EQ(out, (std::pmr::vector<uint32_t>{128, 64, 2, 1}));
    if (path) *path = engine.pipeline_cache_path();
    return engine.pipeline_cache_load();
  }

  static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
  }
  static void write_file(const std::string& path, const std::vector<char>& data) {
    std::ofstream(path, std::ios::binary | std::ios::trunc)
        .write(data.data(), static_cast<std::streamsize>(data.size()));
  }

  std::filesystem::path dir_;
  std::optional<ScopedEnv> env_;
};

// "Warm start": the second engine loads what the first one saved.
TEST_F(PipelineCacheTest, SecondRunLoadsTheSavedCache) {
  std::string path;
  EXPECT_EQ(run_once(&path), kiss_vk::PipelineCacheLoad::kMissing);
  EXPECT_EQ(path.rfind(dir_.string() + "/bt-vk-pipeline-cache-", 0), 0u) << path;
  ASSERT_TRUE(std::filesystem::exists(path));
  EXPECT_EQ(run_once(), kiss_vk::PipelineCacheLoad::kLoaded);
}

// "Stale or foreign cache": a corrupt file is ignored -- garbage, and a valid file cut
// short inside its header.
TEST_F(PipelineCacheTest, CorruptFileIsIgnored) {
  std::string path;
  run_once(&path);
  const auto valid = read_file(path);
  ASSERT_GE(valid.size(), 32u);

  write_file(path, std::vector<char>(4096, '\xab'));
  EXPECT_EQ(run_once(), kiss_vk::PipelineCacheLoad::kRejected);

  write_file(path, std::vector<char>(valid.begin(), valid.begin() + 20));
  EXPECT_EQ(run_once(), kiss_vk::PipelineCacheLoad::kRejected);
}

// "Stale or foreign cache": a file from another device (vendorID / deviceID) or another
// driver build (pipelineCacheUUID) is ignored, as is a wrong header length or version.
TEST_F(PipelineCacheTest, ForeignDeviceOrDriverIsIgnored) {
  std::string path;
  run_once(&path);
  const auto valid = read_file(path);
  ASSERT_GE(valid.size(), 32u);

  // VkPipelineCacheHeaderVersionOne: headerSize, headerVersion, vendorID, deviceID at
  // byte offsets 0, 4, 8, 12; pipelineCacheUUID at 16..31.
  for (const size_t offset : {0u, 4u, 8u, 12u, 16u, 31u}) {
    auto foreign = valid;
    foreign[offset] = static_cast<char>(foreign[offset] ^ 0x5a);
    write_file(path, foreign);
    EXPECT_EQ(run_once(), kiss_vk::PipelineCacheLoad::kRejected) << "byte " << offset;
  }

  write_file(path, valid);  // control: the untouched file loads
  EXPECT_EQ(run_once(), kiss_vk::PipelineCacheLoad::kLoaded);
}

// A cache path that cannot be read (here: a directory) is ignored like a corrupt file, not
// thrown out of Engine construction; the save then fails without touching it.
TEST_F(PipelineCacheTest, UnreadableCacheFileIsIgnored) {
  std::string path;
  run_once(&path);
  std::filesystem::remove(path);
  std::filesystem::create_directory(path);

  EXPECT_EQ(run_once(), kiss_vk::PipelineCacheLoad::kRejected);
  EXPECT_TRUE(std::filesystem::is_directory(path));
  // No temporary file left behind: the directory holds only the unreadable "cache".
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(dir_),
                          std::filesystem::directory_iterator{}),
            1);
}

// A FIFO at the cache path must not block Engine construction (opening it for reading
// would wait for a writer). Bounded: if construction is still blocked after 5 s, the test
// opens the FIFO's write end to release it, then fails, so a regression cannot hang the
// suite.
TEST_F(PipelineCacheTest, FifoAtCachePathIsIgnoredWithoutBlocking) {
  std::string path;
  run_once(&path);
  std::filesystem::remove(path);
  if (mkfifo(path.c_str(), 0600) != 0) {
    // Android's SELinux policy denies the shell domain `fifo_file create` in /data/local/tmp,
    // so on the phones no FIFO can exist at the cache path; any other error is a failure.
    if (errno == EACCES || errno == EPERM) {
      GTEST_SKIP() << "the OS forbids creating a FIFO here: " << std::strerror(errno);
    }
    FAIL() << "mkfifo: " << std::strerror(errno);
  }

  std::promise<kiss_vk::PipelineCacheLoad> result;
  auto done = result.get_future();
  std::thread worker([&] { result.set_value(run_once()); });
  const bool finished = done.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
  if (!finished) {
    for (int i = 0;
         i < 100 && done.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready;
         ++i) {
      if (const int w = open(path.c_str(), O_WRONLY | O_NONBLOCK); w >= 0) close(w);
    }
  }
  worker.join();
  EXPECT_TRUE(finished) << "Engine construction blocked on the FIFO for 5 s";
  EXPECT_EQ(done.get(), kiss_vk::PipelineCacheLoad::kRejected);
}

// A cache file over kMaxPipelineCacheBytes (here a valid file extended sparsely past it) is
// ignored and compiles cold.
TEST_F(PipelineCacheTest, OversizedCacheFileIsIgnored) {
  std::string path;
  run_once(&path);
  std::filesystem::resize_file(path, kiss_vk::kMaxPipelineCacheBytes + 1);
  EXPECT_EQ(run_once(), kiss_vk::PipelineCacheLoad::kRejected);
}

// "Read-only location": the run completes and nothing is written.
TEST_F(PipelineCacheTest, ReadOnlyDirectoryRunsWithoutPersisting) {
  std::filesystem::permissions(
      dir_, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
  ASSERT_NE(access(dir_.c_str(), W_OK), 0) << dir_ << " is still writable (running as root?)";

  EXPECT_EQ(run_once(), kiss_vk::PipelineCacheLoad::kMissing);
  EXPECT_EQ(run_once(), kiss_vk::PipelineCacheLoad::kMissing);  // still nothing persisted
  EXPECT_TRUE(std::filesystem::is_empty(dir_));
}

// A save that cannot be written completely is reported as failed and leaves the previous
// file intact: the temporary file is never renamed over it. Hardware-free. The write is cut
// off by lowering this process's soft RLIMIT_FSIZE to 32 bytes (SIGXFSZ ignored), only
// around the one call, so the test's own output is unaffected.
TEST(PipelineCacheFile, FailedWriteKeepsThePreviousFile) {
  const auto dir =
      std::filesystem::current_path() / ("bt-vk-cache-test-" + std::to_string(getpid()) + "-write");
  std::filesystem::create_directory(dir);
  const std::string path = (dir / "cache.bin").string();
  const std::vector<uint8_t> previous(1056, 0x11);
  const std::vector<uint8_t> next(1056, 0x22);
  const auto contents = [&] {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>{std::istreambuf_iterator<char>(f),
                                std::istreambuf_iterator<char>()};
  };
  ASSERT_EQ(kiss_vk::write_file_atomically(path, previous), "");

  rlimit old_limit{};
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &old_limit), 0);
  rlimit small = old_limit;
  small.rlim_cur = 32;
  const auto old_handler = std::signal(SIGXFSZ, SIG_IGN);
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &small), 0);
  const std::string why = kiss_vk::write_file_atomically(path, next);
  setrlimit(RLIMIT_FSIZE, &old_limit);
  std::signal(SIGXFSZ, old_handler);

  EXPECT_NE(why.find("write failed after 32 of 1056 bytes"), std::string::npos) << why;
  EXPECT_EQ(contents(), previous);
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(dir),
                          std::filesystem::directory_iterator{}),
            1);  // no temporary file left

  // Control: without the limit the same call replaces the file.
  EXPECT_EQ(kiss_vk::write_file_atomically(path, next), "");
  EXPECT_EQ(contents(), next);
  std::filesystem::remove_all(dir);
}

// Hardware-free: a file whose header matches `props`, extended sparsely to `size` bytes.
namespace {
vk::PhysicalDeviceProperties fake_cache_device() {
  vk::PhysicalDeviceProperties props{};
  props.vendorID = 0x1234;
  props.deviceID = 0x5678;
  for (uint8_t i = 0; i < VK_UUID_SIZE; ++i) props.pipelineCacheUUID[i] = i;
  return props;
}

void write_sparse_cache(const std::string& path,
                        const vk::PhysicalDeviceProperties& props,
                        const uint64_t size) {
  VkPipelineCacheHeaderVersionOne header{.headerSize = sizeof(header),
                                         .headerVersion = VK_PIPELINE_CACHE_HEADER_VERSION_ONE,
                                         .vendorID = props.vendorID,
                                         .deviceID = props.deviceID};
  std::memcpy(header.pipelineCacheUUID, props.pipelineCacheUUID.data(), VK_UUID_SIZE);
  std::ofstream(path, std::ios::binary | std::ios::trunc)
      .write(reinterpret_cast<const char*>(&header), sizeof(header));
  std::filesystem::resize_file(path, size);
}

// This process's address-space size (VmSize, what RLIMIT_AS is checked against), in bytes.
uint64_t vm_size_bytes() {
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);) {
    if (line.rfind("VmSize:", 0) == 0) return std::stoull(line.substr(7)) * 1024;
  }
  return 0;
}

// Child mode of this binary (see main): read the cache file at `path` with the soft
// RLIMIT_AS 16 MiB above current use. RLIMIT_AS, not RLIMIT_DATA: bionic's allocator maps
// memory that RLIMIT_DATA does not cap. Exit 0 iff the read was rejected as unallocatable.
int cache_alloc_child(const char* path) {
  rlimit limit{};
  if (getrlimit(RLIMIT_AS, &limit) != 0) return 2;
  limit.rlim_cur = vm_size_bytes() + (16u << 20);
  if (setrlimit(RLIMIT_AS, &limit) != 0) return 3;
  const auto file = kiss_vk::read_pipeline_cache_file(path, fake_cache_device());
  std::printf("child: load %d, why \"%s\"\n", static_cast<int>(file.load), file.why.c_str());
  return file.load == kiss_vk::PipelineCacheLoad::kRejected &&
                 file.why.find("cannot allocate") != std::string::npos
             ? 0
             : 1;
}

// Runs cache_alloc_child(path) in a fresh copy of this binary and returns its exit status.
// A fresh process: memory this one already holds (a retained heap top, allocator caches)
// could otherwise satisfy the allocation without new address space, making the limit moot.
int run_cache_alloc_child(const std::string& path) {
  const pid_t pid = fork();
  if (pid == 0) {
    execl("/proc/self/exe", "test-kiss-vk", "--cache-alloc-child", path.c_str(), nullptr);
    _exit(127);
  }
  int status = 0;
  if (pid < 0 || waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) return -1;
  return WEXITSTATUS(status);
}
}  // namespace

// Pipeline-config spec, "Stale or foreign cache" for oversized data. The header is read and
// the size bounded before the payload is allocated, so a 2 GiB file is rejected without an
// allocation, even with the address space limited to 16 MiB above current use. A 48 MiB
// file (under the bound) whose allocation fails is rejected too, never thrown: checked in a
// fresh child process (run_cache_alloc_child).
TEST(PipelineCacheFile, OversizedOrUnallocatableDataIsRejected) {
  const auto dir =
      std::filesystem::current_path() / ("bt-vk-cache-test-" + std::to_string(getpid()) + "-size");
  std::filesystem::create_directory(dir);
  const std::string path = (dir / "cache.bin").string();
  const auto props = fake_cache_device();

  write_sparse_cache(path, props, uint64_t{2} << 30);
  kiss_vk::PipelineCacheFile file;
  {
    struct RestoreLimit {
      RestoreLimit() = default;
      ~RestoreLimit() { setrlimit(RLIMIT_AS, &old); }
      RestoreLimit(const RestoreLimit&) = delete;
      RestoreLimit& operator=(const RestoreLimit&) = delete;
      RestoreLimit(RestoreLimit&&) = delete;
      RestoreLimit& operator=(RestoreLimit&&) = delete;
      rlimit old{};
    } restore;
    ASSERT_EQ(getrlimit(RLIMIT_AS, &restore.old), 0);
    rlimit tight = restore.old;
    tight.rlim_cur = vm_size_bytes() + (16u << 20);
    ASSERT_EQ(setrlimit(RLIMIT_AS, &tight), 0);
    file = kiss_vk::read_pipeline_cache_file(path, props);
  }
  EXPECT_EQ(file.load, kiss_vk::PipelineCacheLoad::kRejected);
  EXPECT_NE(file.why.find("2147483648 bytes exceeds the 67108864-byte limit"), std::string::npos)
      << file.why;

  write_sparse_cache(path, props, 48u << 20);
  EXPECT_EQ(run_cache_alloc_child(path), 0) << "the child's read was not rejected as unallocatable";

  // Control: without the limit the 48 MiB file reads in full.
  file = kiss_vk::read_pipeline_cache_file(path, props);
  EXPECT_EQ(file.load, kiss_vk::PipelineCacheLoad::kLoaded) << file.why;
  EXPECT_EQ(file.data.size(), 48u << 20);
  std::filesystem::remove_all(dir);
}

TEST(VulkanWorkGroupSize, MismatchWithLiteralLocalSizeThrows) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  kiss_vk::Engine engine;
  // tree_morton.comp: layout(local_size_x = 256) in;
  const auto err =
      build_error(engine.make_algo("tree_morton")->work_group_size(128, 1, 1)->num_buffers(2));
  EXPECT_NE(err.find("declared workgroup size (128, 1, 1)"), std::string::npos) << err;
  EXPECT_NE(err.find("LocalSize is (256, 1, 1)"), std::string::npos) << err;
}

TEST(VulkanWorkGroupSize, ExceedsDeviceLimitsThrows) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  kiss_vk::Engine engine;
  const auto& limits = engine.compute_limits();
  const auto& max_wg = limits.max_work_group_size;

  const std::array<uint32_t, 3> too_wide{max_wg[0] + 1, 1, 1};
  auto err = build_error(engine.make_algo("engine_test_workgroup_size")
                             ->work_group_size(too_wide[0], too_wide[1], too_wide[2])
                             ->num_buffers(1));
  EXPECT_NE(err.find("workgroup size " + dims(too_wide)), std::string::npos) << err;
  EXPECT_NE(err.find("maxComputeWorkGroupSize " + dims(max_wg)), std::string::npos) << err;

  // Every dimension at its own limit: within maxComputeWorkGroupSize, but far over the
  // invocation limit on every fleet GPU.
  ASSERT_GT(uint64_t{max_wg[0]} * max_wg[1] * max_wg[2], limits.max_work_group_invocations);
  err = build_error(engine.make_algo("engine_test_workgroup_size")
                        ->work_group_size(max_wg[0], max_wg[1], max_wg[2])
                        ->num_buffers(1));
  EXPECT_NE(err.find("workgroup size " + dims(max_wg)), std::string::npos) << err;
  EXPECT_NE(
      err.find(fmt::format("maxComputeWorkGroupInvocations {}", limits.max_work_group_invocations)),
      std::string::npos)
      << err;
}

// A shader that declares constant_id = 0 (subgroup size) must be built
// subgroup_size_dependent(), and only such a shader may be: never run on the default.
TEST(VulkanSubgroupSize, DependentPipelinesMatchTheShader) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  kiss_vk::Engine engine;
  auto err = build_error(engine.make_algo("multi_radixsort")
                             ->work_group_size(256, 1, 1)
                             ->num_buffers(3)
                             ->push_constant_size(16));
  EXPECT_NE(err.find("is not subgroup_size_dependent()"), std::string::npos) << err;

  err = build_error(engine.make_algo("tree_morton")
                        ->work_group_size(256, 1, 1)
                        ->subgroup_size_dependent()
                        ->num_buffers(2));
  EXPECT_NE(err.find("has no constant_id = 0"), std::string::npos) << err;

  EXPECT_NO_THROW((void)engine.make_algo("multi_radixsort")
                      ->work_group_size(256, 1, 1)
                      ->subgroup_size_dependent()
                      ->num_buffers(3)
                      ->push_constant_size(16)
                      ->build());
}

// Design A3: each binding's access comes from the shader's readonly/writeonly qualifiers;
// a plain `buffer` is read and written. Hardware-free.
TEST(SpirvReflection, BufferAccess) {
  using kiss_vk::BufferAccess;
  using Map = std::map<uint32_t, BufferAccess>;
  EXPECT_EQ(kiss_vk::reflect_spirv(kiss_vk::load_shader_spirv("tree_scan_local")).binding_access,
            (Map{{0, BufferAccess::kRead}, {1, BufferAccess::kWrite}, {2, BufferAccess::kWrite}}));
  EXPECT_EQ(
      kiss_vk::reflect_spirv(kiss_vk::load_shader_spirv("new_cifar_dense_linear")).binding_access,
      (Map{{0, BufferAccess::kRead},
           {1, BufferAccess::kRead},
           {2, BufferAccess::kRead},
           {3, BufferAccess::kWrite}}));
  EXPECT_EQ(kiss_vk::reflect_spirv(kiss_vk::load_shader_spirv("multi_radixsort")).binding_access,
            (Map{{0, BufferAccess::kReadWrite},
                 {1, BufferAccess::kReadWrite},
                 {2, BufferAccess::kReadWrite}}));
  EXPECT_EQ(kiss_vk::reflect_spirv(kiss_vk::load_shader_spirv("tree_scan_add")).binding_access,
            (Map{{0, BufferAccess::kReadWrite}, {1, BufferAccess::kRead}}));
}

// Execution-sync spec, "Dependencies only where stages depend" and "Inter-stage dependencies
// cover RAW and WAW": consecutive dispatches on different buffers get no barrier; one that
// touches a buffer written since the last barrier gets one. BT_VK_BARRIER_ALL=1 restores a
// barrier before every dispatch but the first. The shader (binding 0 read and written) adds
// its 128 invocations to g_out[0].
TEST(VulkanStageDependency, OnlyWhereStagesDepend) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  for (const bool barrier_all : {false, true}) {
    SCOPED_TRACE(barrier_all ? "BT_VK_BARRIER_ALL=1" : "tracked");
    const ScopedEnv env("BT_VK_BARRIER_ALL", barrier_all ? "1" : nullptr);
    kiss_vk::Engine engine;
    auto seq = engine.make_seq();
    auto algo = engine.make_algo("engine_test_workgroup_size")
                    ->work_group_size(64, 2, 1)
                    ->num_buffers(1)
                    ->build();
    std::pmr::vector<uint32_t> a(4, 0, engine.get_mr());
    std::pmr::vector<uint32_t> b(4, 0, engine.get_mr());

    const auto record = [&](std::pmr::vector<uint32_t>& buf) {
      algo->record_bind(*seq, {engine.get_buffer_info(buf)});
      algo->record_dispatch(*seq, {1, 1, 1});
    };
    seq->cmd_begin();
    record(a);  // first dispatch: nothing to order
    record(b);  // independent of a
    EXPECT_EQ(seq->barrier_count(), barrier_all ? 1u : 0u);
    record(a);  // a was written since the last barrier
    EXPECT_EQ(seq->barrier_count(), barrier_all ? 2u : 1u);
    seq->cmd_end();
    seq->submit();
    seq->wait_for_fence();
    seq->reset_fence();

    EXPECT_EQ(seq->dispatch_count(), 3u);
    EXPECT_EQ(a[0], 256u);
    EXPECT_EQ(b[0], 128u);
  }
}

// Design A4: the Sequence's two COMPUTE_SHADER timestamps bracket the command buffer's
// compute work, so the GPU time is positive and no longer than the host's submit-to-wait.
TEST(VulkanTimestamps, GpuTimeIsPositiveAndWithinWallTime) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  kiss_vk::Engine engine;
  auto seq = engine.make_seq();
  if (!seq->gpu_timestamps_supported()) GTEST_SKIP() << "no timestamps on the compute queue";
  auto algo = engine.make_algo("engine_test_workgroup_size")
                  ->work_group_size(64, 2, 1)
                  ->num_buffers(1)
                  ->build();
  std::pmr::vector<uint32_t> out(4, 0, engine.get_mr());

  for (int rep = 0; rep < 3; ++rep) {
    seq->cmd_begin();
    for (int i = 0; i < 16; ++i) {
      algo->record_bind(*seq, {engine.get_buffer_info(out)});
      algo->record_dispatch(*seq, {256, 1, 1});
    }
    seq->cmd_end();
    const auto t0 = std::chrono::steady_clock::now();
    seq->submit();
    seq->wait_for_fence();
    const double wall_ns =
        std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
    seq->reset_fence();

    const double gpu_ns = seq->get_last_gpu_time_ns();
    EXPECT_GT(gpu_ns, 0.0) << "rep " << rep;
    EXPECT_LE(gpu_ns, wall_ns) << "rep " << rep;
  }
  EXPECT_EQ(out[0], 3u * 16u * 256u * 128u);
}

// Memory-placement spec (design B2): every buffer class allocates host-mapped memory whose
// pointer resolves to its own VkBuffer, the host's writes reach the GPU and the GPU's writes
// reach the host -- also with the weights class forced into write-combined memory
// ("Weights written through a write-combined mapping").
TEST(VulkanBufferClass, EachClassAllocatesAndResolves) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  for (const char* weights : {static_cast<const char*>(nullptr), "uncached", "device_local"}) {
    SCOPED_TRACE(weights ? weights : "weights placement unset");
    const ScopedEnv env("BT_VK_WEIGHTS_PLACEMENT", weights);
    kiss_vk::Engine engine;
    auto seq = engine.make_seq();
    auto algo = engine.make_algo("engine_test_workgroup_size")
                    ->work_group_size(64, 2, 1)
                    ->num_buffers(1)
                    ->build();

    std::set<VkBuffer> buffers;
    std::vector<std::pmr::vector<uint32_t>> alive;  // all three at once: distinct handles
    alive.reserve(3);
    for (const auto c :
         {bt::BufferClass::kBoundary, bt::BufferClass::kWeights, bt::BufferClass::kGpuLocal}) {
      auto& v = alive.emplace_back(4, 0, engine.get_mr(c));
      v[0] = 1000;  // written through the class's mapping
      const vk::Buffer b = engine.get_buffer(v.data());
      ASSERT_TRUE(b);
      EXPECT_EQ(engine.get_mr(c)->find_buffer(v.data()), b);
      buffers.insert(b);

      seq->cmd_begin();
      algo->record_bind(*seq, {engine.get_buffer_info(v)});
      algo->record_dispatch(*seq, {1, 1, 1});
      seq->cmd_end();
      seq->submit();
      seq->wait_for_fence();
      seq->reset_fence();
      EXPECT_EQ(v[0], 1128u) << "class " << static_cast<int>(c);
      EXPECT_EQ(v[1], 64u) << "class " << static_cast<int>(c);
    }
    EXPECT_EQ(buffers.size(), 3u);
  }
}

// Pipeline-config spec, "Full subgroups available": a subgroup-dependent pipeline whose X
// size is a multiple of the subgroup size runs with that size, and with computeFullSubgroups
// every subgroup is launched full.
TEST(VulkanSubgroupSize, FullSubgroupsDispatch) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  kiss_vk::Engine engine;
  const auto& limits = engine.compute_limits();
  auto seq = engine.make_seq();
  std::pmr::vector<uint32_t> out(3, 0, engine.get_mr());
  out[2] = ~0u;

  auto algo = engine.make_algo("engine_test_subgroup")
                  ->work_group_size(4 * limits.subgroup_size, 1, 1)
                  ->subgroup_size_dependent()
                  ->num_buffers(1)
                  ->build();
  seq->cmd_begin();
  algo->record_bind(*seq, {engine.get_buffer_info(out)});
  algo->record_dispatch(*seq, {8, 1, 1});
  seq->cmd_end();
  seq->submit();
  seq->wait_for_fence();
  seq->reset_fence();

  EXPECT_EQ(out[1], limits.subgroup_size);  // specialized
  if (limits.subgroup_size_pinned) EXPECT_EQ(out[0], limits.subgroup_size);
  if (limits.full_subgroups) EXPECT_EQ(out[2], out[0]) << "a subgroup was launched partial";
}

// Pipeline-config spec, "Full subgroups unavailable and size mismatched": with
// computeFullSubgroups forced off, a subgroup-dependent pipeline whose X size is not a
// multiple of the subgroup size fails to build, naming both sizes.
TEST(VulkanSubgroupSize, NonMultipleWorkgroupFailsWithoutFullSubgroups) {
  if (!kiss_vk::has_integrated_gpu()) GTEST_SKIP() << "no integrated GPU";
  kiss_vk::Engine engine;
  kiss_vk::ComputeLimits limits = engine.compute_limits();
  limits.full_subgroups = false;
  const uint32_t x = limits.subgroup_size + limits.subgroup_size / 2;
  const auto make = [&](const uint32_t wg_x) {
    return std::make_shared<kiss_vk::Algorithm>(engine.get_mr(),
                                                "engine_test_subgroup",
                                                limits,
                                                engine.push_descriptor_fn(),
                                                vk::PipelineCache{})
        ->work_group_size(wg_x, 1, 1)
        ->subgroup_size_dependent()
        ->num_buffers(1);
  };

  const auto err = build_error(make(x));
  EXPECT_NE(err.find(fmt::format("workgroup X size {}", x)), std::string::npos) << err;
  EXPECT_NE(err.find(fmt::format("subgroup size {}", limits.subgroup_size)), std::string::npos)
      << err;
  EXPECT_EQ(build_error(make(2 * limits.subgroup_size)), "");  // a multiple still builds
}

// Pipeline-config spec, "Registry subgroup value is a cross-check only": a registry
// subgroup size that disagrees with the device-derived one warns, naming both values.
// Hardware-free: minipc's devices/minipc.json declares 64.
TEST(RegistrySubgroupCrossCheck, MismatchWarns) {
  auto sink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(8);
  const auto logger = spdlog::default_logger();
  const auto old_level = logger->level();
  logger->sinks().push_back(sink);
  logger->set_level(spdlog::level::warn);
  const std::string old_device = g_device_id;
  g_device_id = "minipc";

  const bool mismatch_ok = check_vulkan_subgroup_size(16);
  const auto mismatch_logs = sink->last_formatted();
  const bool match_ok = check_vulkan_subgroup_size(64);
  const auto all_logs = sink->last_formatted();

  g_device_id = old_device;
  logger->set_level(old_level);
  logger->sinks().pop_back();

  EXPECT_FALSE(mismatch_ok);
  ASSERT_EQ(mismatch_logs.size(), 1u);
  EXPECT_NE(mismatch_logs[0].find("[warning]"), std::string::npos) << mismatch_logs[0];
  EXPECT_NE(mismatch_logs[0].find("gpu.subgroup_size 64"), std::string::npos) << mismatch_logs[0];
  EXPECT_NE(mismatch_logs[0].find("engine uses 16"), std::string::npos) << mismatch_logs[0];
  EXPECT_TRUE(match_ok);
  EXPECT_EQ(all_logs.size(), 1u);  // the match logged nothing
}

int main(int argc, char** argv) {
  if (argc == 3 && std::string_view(argv[1]) == "--cache-alloc-child") {
    return cache_alloc_child(argv[2]);
  }

  // Initialize Google Test
  ::testing::InitGoogleTest(&argc, argv);

  // Set logging level to off (SPDLOG_LEVEL=debug overrides, e.g. to see the enabled layers)
  spdlog::set_level(spdlog::level::off);
  spdlog::cfg::load_env_levels();

  // Run the tests
  return RUN_ALL_TESTS();
}