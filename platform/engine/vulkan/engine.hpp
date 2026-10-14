#pragma once

#include <spdlog/spdlog.h>

#include <charconv>
#include <cstdlib>
#include <string_view>

#include "algorithm.hpp"
#include "base_engine.hpp"
#include "platform/registry/device_registry.hpp"
#include "sequence.hpp"
#include "vma_pmr.hpp"

namespace kiss_vk {

class Engine final : public BaseEngine {
 public:
  // The Khronos validation layer is opt-in: requested only when BT_VK_VALIDATION=1.
  explicit Engine()
      : BaseEngine(validation_requested()),
        mr_ptr_(std::make_unique<VulkanMemoryResource>(device_)) {
    spdlog::trace("Engine::Engine()");
    // Test hook (design D5): BT_VK_SUBGROUP_SIZE=<n> pins another size the device
    // supports, e.g. its minSubgroupSize; an unsupported value throws.
    if (const char* v = std::getenv("BT_VK_SUBGROUP_SIZE")) {
      uint32_t size = 0;
      const std::string_view s(v);
      if (std::from_chars(s.data(), s.data() + s.size(), size).ptr != s.data() + s.size()) {
        throw std::runtime_error("BT_VK_SUBGROUP_SIZE is not a number: " + std::string(s));
      }
      override_subgroup_size(size);
    }
    // The registry's subgroup size is a cross-check only (design D5).
    check_vulkan_subgroup_size(subgroup_size());
  }

  ~Engine() { spdlog::trace("Engine::~Engine()"); }

  [[nodiscard]] VulkanMemoryResource* get_mr() const { return mr_ptr_.get(); }

  [[nodiscard]] std::shared_ptr<Algorithm> make_algo(const std::string& shader_name) const {
    return std::make_shared<Algorithm>(
        mr_ptr_.get(), shader_name, compute_limits(), push_descriptor_fn());
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

    return std::make_shared<Sequence>(this->get_device(),
                                      this->get_compute_queue(),
                                      this->get_compute_queue_family_index(),
                                      timestamp_period_ns,
                                      timestamp_valid_bits,
                                      mr_ptr_.get());
  }

  // To get a 'vk::Buffer' from raw pointer of the 'std::pmr::vector'
  [[nodiscard]] vk::Buffer get_buffer(void* ptr) const {
    return mr_ptr_->get_buffer_from_pointer(ptr);
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
  [[nodiscard]] static bool validation_requested() {
    const char* v = std::getenv("BT_VK_VALIDATION");
    return v != nullptr && std::string_view(v) == "1";
  }

  std::unique_ptr<VulkanMemoryResource> mr_ptr_;
};

}  // namespace kiss_vk