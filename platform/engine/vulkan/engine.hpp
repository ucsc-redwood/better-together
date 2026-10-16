#pragma once

#include <spdlog/spdlog.h>

#include <charconv>
#include <cstdlib>
#include <mutex>
#include <string_view>

#include "algorithm.hpp"
#include "base_engine.hpp"
#include "env.hpp"
#include "pipeline_cache.hpp"
#include "platform/registry/device_registry.hpp"
#include "platform/util/buffer_placement.hpp"
#include "sequence.hpp"
#include "vma_pmr.hpp"

namespace kiss_vk {

class Engine final : public BaseEngine {
 public:
  // The Khronos validation layer is opt-in: requested only when BT_VK_VALIDATION=1.
  explicit Engine()
      : BaseEngine(validation_requested()),
        mr_ptr_(std::make_unique<VulkanMemoryResource>(
            device_, boundary_class_placement(get_physical_device()))),
        weights_mr_ptr_(std::make_unique<VulkanMemoryResource>(
            device_, weights_class_placement(get_physical_device()))),
        // No measurement has promoted the GPU-local class on any device (design B1/B2).
        gpu_local_mr_ptr_(
            std::make_unique<VulkanMemoryResource>(device_, MemoryPlacement{.name = "gpu_local"})) {
    spdlog::trace("Engine::Engine()");
    // Test hook (design D5): BT_VK_SUBGROUP_SIZE=<n> pins another size the device
    // supports, e.g. its minSubgroupSize; an unsupported value throws.
    if (const auto v = env::get(env::kSubgroupSize)) {
      uint32_t size = 0;
      const std::string_view s = *v;
      if (std::from_chars(s.data(), s.data() + s.size(), size).ptr != s.data() + s.size()) {
        throw std::runtime_error("BT_VK_SUBGROUP_SIZE is not a number: " + std::string(s));
      }
      override_subgroup_size(size);
    }
    // The registry's subgroup size is a cross-check only (design D5).
    check_vulkan_subgroup_size(subgroup_size());
    // Phase timing (BT_VK_PHASES=1) relies on the calibration: verify it once (design D1).
    if (env::flag(env::kPhases)) check_calibration();

    // One pipeline cache for every Algorithm, persisted across runs (design D7). Created
    // last: nothing after it can throw, and ~Engine is what saves and destroys it.
    const auto props = get_physical_device().getProperties();
    pipeline_cache_path_ = kiss_vk::pipeline_cache_path(props);
    const auto loaded = load_pipeline_cache(device_, props, pipeline_cache_path_);
    pipeline_cache_ = loaded.cache;
    pipeline_cache_load_ = loaded.load;
  }

  // Every Algorithm built from this Engine is gone by now (owners declare it first).
  ~Engine() {
    spdlog::trace("Engine::~Engine()");
    save_pipeline_cache(device_, pipeline_cache_, pipeline_cache_path_);
    device_.destroyPipelineCache(pipeline_cache_);
  }

  [[nodiscard]] PipelineCacheLoad pipeline_cache_load() const { return pipeline_cache_load_; }
  [[nodiscard]] const std::string& pipeline_cache_path() const { return pipeline_cache_path_; }

  // The boundary (default) class: host-visible, host-cached where available.
  [[nodiscard]] VulkanMemoryResource* get_mr() const { return mr_ptr_.get(); }
  // One resource per buffer class (design B2); every class is host-visible and mapped.
  [[nodiscard]] VulkanMemoryResource* get_mr(const bt::BufferClass c) const {
    switch (c) {
      case bt::BufferClass::kWeights:
        return weights_mr_ptr_.get();
      case bt::BufferClass::kGpuLocal:
        return gpu_local_mr_ptr_.get();
      case bt::BufferClass::kBoundary:
        break;
    }
    return mr_ptr_.get();
  }

  [[nodiscard]] std::shared_ptr<Algorithm> make_algo(const std::string& shader_name) const {
    return std::make_shared<Algorithm>(
        mr_ptr_.get(), shader_name, compute_limits(), push_descriptor_fn(), pipeline_cache_);
  }

  [[nodiscard]] std::shared_ptr<Sequence> make_seq() {
    const auto props = this->get_physical_device().getProperties();
    const float timestamp_period_ns = props.limits.timestampPeriod;

    // timestampValidBits of the compute queue family (0 => unsupported).
    uint32_t timestamp_valid_bits = 0;
    const auto queue_families = this->get_physical_device().getQueueFamilyProperties();
    const uint32_t qfi = this->get_compute_queue_family_index();
    if (qfi < queue_families.size()) {
      timestamp_valid_bits = queue_families[qfi].timestampValidBits;
    }

    return std::make_shared<Sequence>(
        this->get_device(),
        this->get_compute_queue(),
        this->get_compute_queue_family_index(),
        timestamp_period_ns,
        timestamp_valid_bits,
        resources(),
        &submit_mutex_,
        push_descriptor_fn(),
        [this](uint64_t& device_ticks, uint64_t& host_ns, uint64_t& deviation_ns) {
          return calibrate(device_ticks, host_ns, deviation_ns);
        });
  }

  // To get a 'vk::Buffer' from raw pointer of the 'std::pmr::vector', from whichever of the
  // engine's resources allocated it.
  [[nodiscard]] vk::Buffer get_buffer(void* ptr) const {
    for (const VulkanMemoryResource* mr : resources()) {
      if (const vk::Buffer b = mr->find_buffer(ptr)) return b;
    }
    throw std::runtime_error("Unknown pointer in Engine::get_buffer");
  }

  template <typename T>
  [[nodiscard]] vk::DescriptorBufferInfo get_buffer_info(std::pmr::vector<T>& vec) const {
    // auto vk_buffer = get_buffer(vec.data());
    // spdlog::trace("get_buffer_info: vec.data() = {}, vk_buffer = {}, vec.size() = {}",
    //               static_cast<void*>(vec.data()),
    //               static_cast<void*>(vk_buffer),
    //               vec.size());

    return vk::DescriptorBufferInfo{
        .buffer = get_buffer(vec.data()),
        .offset = 0,
        .range = vec.size() * sizeof(T),
    };
  }

  template <typename T>
  [[nodiscard]] vk::DescriptorBufferInfo get_buffer_info(const std::pmr::vector<T>& vec) const {
    return vk::DescriptorBufferInfo{
        .buffer = get_buffer(const_cast<void*>(static_cast<const void*>(vec.data()))),
        .offset = 0,
        .range = vec.size() * sizeof(T),
    };
  }

 private:
  [[nodiscard]] static bool validation_requested() { return env::flag(env::kValidation); }

  [[nodiscard]] std::vector<VulkanMemoryResource*> resources() const {
    return {mr_ptr_.get(), weights_mr_ptr_.get(), gpu_local_mr_ptr_.get()};
  }

  std::unique_ptr<VulkanMemoryResource> mr_ptr_;  // boundary
  std::unique_ptr<VulkanMemoryResource> weights_mr_ptr_;
  std::unique_ptr<VulkanMemoryResource> gpu_local_mr_ptr_;
  std::mutex submit_mutex_;  // guards vkQueueSubmit on the one compute queue (design C2)
  std::string pipeline_cache_path_;
  vk::PipelineCache pipeline_cache_;
  PipelineCacheLoad pipeline_cache_load_ = PipelineCacheLoad::kMissing;
};

}  // namespace kiss_vk