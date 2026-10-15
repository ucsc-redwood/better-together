#include "vma_pmr.hpp"

#include <cstdlib>
#include <string>
#include <string_view>

namespace kiss_vk {

void CHECK_VK_RESULT(VkResult result, const char* msg) {
  if (result != VK_SUCCESS) {
    spdlog::error("Vulkan error: {} - {}", vk::to_string(static_cast<vk::Result>(result)), msg);
    throw std::runtime_error("Vulkan Error");
  }
}

MemoryPlacement weights_class_placement(const vk::PhysicalDevice physical_device) {
  const auto props = physical_device.getProperties();
  // Devices where moving the weights off the default placement measured faster beyond noise
  // (memory-placement spec "Placement is decided by measurement"; tasks 2.3).
  constexpr struct {
    uint32_t vendor_id, device_id;
    const char* placement;
  } kMeasuredGo[] = {
      {0x13B5, 0xa8620000, "uncached"},  // Pixel 7a, Mali-G710: fc1/fc2 -31..-35 %
  };

  std::string_view arm = "default";
  const char* reason = "no measured gain on this device";
  for (const auto& d : kMeasuredGo) {
    if (props.vendorID == d.vendor_id && props.deviceID == d.device_id) {
      arm = d.placement;
      reason = "measured gain on this device";
    }
  }
  if (const char* env = std::getenv("BT_VK_WEIGHTS_PLACEMENT")) {
    arm = env;
    reason = "BT_VK_WEIGHTS_PLACEMENT";
  }
  const MemoryPlacement fallback{.name = "weights"};
  if (arm != "default" && arm != "device_local" && arm != "uncached") {
    throw std::runtime_error(
        "BT_VK_WEIGHTS_PLACEMENT must be default, device_local or uncached, got " +
        std::string(arm));
  }
  if (arm == "default") {
    spdlog::debug("Buffer class weights: default placement ({})", reason);
    return fallback;
  }

  const auto mem = physical_device.getMemoryProperties();
  const auto types_with = [&](const vk::MemoryPropertyFlags required,
                              const vk::MemoryPropertyFlags excluded) {
    uint32_t bits = 0;
    for (uint32_t t = 0; t < mem.memoryTypeCount; ++t) {
      const auto f = mem.memoryTypes[t].propertyFlags;
      if ((f & required) == required && !(f & excluded)) bits |= 1u << t;
    }
    return bits;
  };
  using F = vk::MemoryPropertyFlagBits;
  const vk::MemoryPropertyFlags amd = F::eDeviceCoherentAMD | F::eDeviceUncachedAMD;

  MemoryPlacement p{.name = "weights", .preferred = 0};
  if (arm == "device_local") {
    p.required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    p.memory_type_bits = types_with(F::eHostVisible | F::eDeviceLocal, F::eHostCached | amd);
  } else {
    p.memory_type_bits = types_with(F::eHostVisible, F::eHostCached | F::eDeviceLocal | amd);
    if (p.memory_type_bits == 0) {
      p.memory_type_bits = types_with(F::eHostVisible, F::eHostCached | amd);
    }
  }
  if (p.memory_type_bits == 0) {
    spdlog::warn("Buffer class weights: no {} memory type on this device, keeping the default",
                 arm);
    return fallback;
  }
  spdlog::debug("Buffer class weights: {} placement ({}), memory type bits {:#x}",
                arm,
                reason,
                p.memory_type_bits);
  return p;
}

VulkanMemoryResource::VulkanMemoryResource(const vk::Device device, const MemoryPlacement placement)
    : device_(device), placement_(placement) {
  spdlog::debug(
      "VulkanMemoryResource [{}] created: required {}, preferred {}, memory type bits {:#x}",
      placement_.name,
      vk::to_string(vk::MemoryPropertyFlags(placement_.required)),
      vk::to_string(vk::MemoryPropertyFlags(placement_.preferred)),
      placement_.memory_type_bits);
}

VulkanMemoryResource::~VulkanMemoryResource() {
  spdlog::debug("VulkanMemoryResource [{}] destroyed", placement_.name);
}

vk::Buffer VulkanMemoryResource::find_buffer(const void* p) const {
  std::lock_guard lock(mutex_);
  const auto it = allocations_.find(p);
  return it == allocations_.end() ? vk::Buffer{} : vk::Buffer{it->second.buffer};
}

VmaAllocation VulkanMemoryResource::find_allocation(VkBuffer buffer) const {
  std::lock_guard lock(mutex_);
  const auto it = allocation_of_buffer_.find(buffer);
  return it == allocation_of_buffer_.end() ? nullptr : it->second;
}

void* VulkanMemoryResource::do_allocate(std::size_t bytes, [[maybe_unused]] std::size_t alignment) {
  spdlog::trace("VulkanMemoryResource::do_allocate({}, {})", bytes, alignment);

  const vk::BufferCreateInfo buffer_create_info{
      .size = bytes,
      .usage = vk::BufferUsageFlagBits::eStorageBuffer,
  };

  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  // Prefer a HOST_VISIBLE + HOST_CACHED heap. Previously we FORCED HOST_COHERENT
  // to avoid manual cache maintenance (BUGS-FOUND.md §7: kiss-vk never invalidated
  // after GPU work, so non-coherent Mali read stale data). But on Mali (Pixel 7a)
  // the coherent heap is uncached/write-combined, and the differential test's CPU
  // readback of full output tensors from it is ~90x slower than on a cached heap
  // (Pixel cifar-dense-vk: 233s vs 2.5s on a coherent-friendly GPU). HOST_CACHED
  // restores fast CPU reads; correctness on the resulting NON-coherent memory is
  // handled explicitly by flushing a submit's buffers before it and invalidating them
  // after its wait (Sequence::submit / wait_for_fence) -- the maintenance §7 lacked. VMA
  // falls back to a coherent type if no cached host-visible type exists, and
  // flush/invalidate are no-ops on coherent memory, so other GPUs are unaffected.
  // The default MemoryPlacement is exactly that; the other placements (design B1/B2) pick
  // other host-visible types.
  allocCreateInfo.requiredFlags = placement_.required;
  allocCreateInfo.preferredFlags = placement_.preferred;
  allocCreateInfo.memoryTypeBits = placement_.memory_type_bits;

  VkBuffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo allocInfo{};

  VkResult res = vmaCreateBuffer(g_vma_allocator,
                                 reinterpret_cast<const VkBufferCreateInfo*>(&buffer_create_info),
                                 &allocCreateInfo,
                                 &buffer,
                                 &allocation,
                                 &allocInfo);

  CHECK_VK_RESULT(res, "Failed to create VMA buffer");

  // The memory should already be mapped due to the
  // VMA_ALLOCATION_CREATE_MAPPED_BIT flag.
  void* mappedPtr = allocInfo.pMappedData;
  if (!mappedPtr) {
    // If it's not mapped for some reason, we can attempt to map it.
    // According to the flags, it should be mapped already, but just in case:
    res = vmaMapMemory(g_vma_allocator, allocation, &mappedPtr);
    CHECK_VK_RESULT(res, "Failed to map VMA buffer memory");
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[mappedPtr] = VulkanAllocationRecord{buffer, allocation, allocInfo};
    allocation_of_buffer_[buffer] = allocation;
  }
  spdlog::debug("VulkanMemoryResource [{}]: {} bytes in memoryTypeIndex {}",
                placement_.name,
                bytes,
                allocInfo.memoryType);

  return mappedPtr;
}

// Deallocate the buffer by looking up our record.
void VulkanMemoryResource::do_deallocate(void* p,
                                         [[maybe_unused]] std::size_t bytes,
                                         [[maybe_unused]] std::size_t alignment) {
  spdlog::trace("VulkanMemoryResource::do_deallocate({}, {}, {})", p, bytes, alignment);

  VulkanAllocationRecord record;
  {
    std::lock_guard lock(mutex_);
    const auto it = allocations_.find(p);
    if (it == allocations_.end()) {
      spdlog::error(
          "Attempted to deallocate unknown pointer from "
          "VulkanMemoryResource");
      throw std::runtime_error("Unknown pointer in VulkanMemoryResource::do_deallocate");
    }
    record = it->second;
    allocations_.erase(it);
    allocation_of_buffer_.erase(record.buffer);
  }

  // Destroy the buffer and free the allocation
  vmaDestroyBuffer(g_vma_allocator, record.buffer, record.allocation);
}

bool VulkanMemoryResource::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
  return dynamic_cast<const VulkanMemoryResource*>(&other) != nullptr;
}

}  // namespace kiss_vk