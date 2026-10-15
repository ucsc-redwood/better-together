#include "sequence.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string_view>
#include <vulkan/vulkan.hpp>

namespace kiss_vk {

Sequence::Sequence(const vk::Device device_ref,
                   const vk::Queue compute_queue_ref,
                   const uint32_t compute_queue_index,
                   const float timestamp_period_ns,
                   const uint32_t timestamp_valid_bits,
                   std::vector<VulkanMemoryResource*> resources,
                   std::mutex* submit_mutex)
    : device_ref_(device_ref),
      compute_queue_ref_(compute_queue_ref),
      compute_queue_index_(compute_queue_index),
      timestamp_period_ns_(timestamp_period_ns),
      timestamp_valid_bits_(timestamp_valid_bits),
      resources_(std::move(resources)),
      submit_mutex_(submit_mutex) {
  spdlog::trace("Sequence constructor");

  const char* barrier_all = std::getenv("BT_VK_BARRIER_ALL");
  barrier_all_ = barrier_all != nullptr && std::string_view(barrier_all) == "1";

  create_sync_objects();
  create_command_pool();
  create_command_buffer();
  create_query_pool();
}

Sequence::~Sequence() {
  // Destroy every child object this Sequence created, before BaseEngine destroys the
  // device. Previously only the query pool was freed, so each Sequence leaked a
  // VkCommandPool + VkFence and left them live at vkDestroyDevice (review #5).
  if (timestamp_valid_bits_ != 0 && query_pool_) {
    device_ref_.destroyQueryPool(query_pool_);
  }
  if (command_pool_) {
    device_ref_.destroyCommandPool(command_pool_);  // also frees the command buffer handle_
  }
  if (fence_) {
    device_ref_.destroyFence(fence_);
  }
  for (const auto pool : descriptor_pools_) {
    device_ref_.destroyDescriptorPool(pool);  // also frees its sets
  }
}

void Sequence::create_query_pool() {
  if (timestamp_valid_bits_ == 0) {
    spdlog::trace("Sequence::create_query_pool() skipped (timestamps unsupported)");
    return;
  }

  const vk::QueryPoolCreateInfo create_info{
      .queryType = vk::QueryType::eTimestamp,
      .queryCount = 2,
  };

  query_pool_ = device_ref_.createQueryPool(create_info);
}

void Sequence::create_command_pool() {
  spdlog::trace("Sequence::create_command_pool()");

  const vk::CommandPoolCreateInfo create_info{
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = compute_queue_index_,
  };

  command_pool_ = device_ref_.createCommandPool(create_info);
}

void Sequence::create_sync_objects() {
  spdlog::trace("Sequence::create_sync_objects()");

  constexpr vk::FenceCreateInfo create_info{};
  fence_ = device_ref_.createFence(create_info);
}

void Sequence::create_command_buffer() {
  spdlog::trace("Sequence::create_command_buffer()");

  const vk::CommandBufferAllocateInfo allocate_info{
      .commandPool = command_pool_,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1,
  };

  handle_ = device_ref_.allocateCommandBuffers(allocate_info).front();
}

vk::DescriptorPool Sequence::create_descriptor_pool() const {
  // Sized so a whole chunk fits one pool (the longest, tree stages 1-7, binds 20 sets of
  // <= 13 buffers); a bigger chunk just opens another pool.
  constexpr uint32_t kSetsPerPool = 64;
  const vk::DescriptorPoolSize pool_size{
      .type = vk::DescriptorType::eStorageBuffer,
      .descriptorCount = kSetsPerPool * kMaxBindingBuffers,
  };
  return device_ref_.createDescriptorPool(vk::DescriptorPoolCreateInfo{
      .maxSets = kSetsPerPool,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
  });
}

vk::DescriptorSet Sequence::allocate_descriptor_set(const vk::DescriptorSetLayout layout) {
  for (;; ++current_pool_) {
    const bool fresh = current_pool_ == descriptor_pools_.size();
    if (fresh) descriptor_pools_.push_back(create_descriptor_pool());

    const vk::DescriptorSetAllocateInfo allocate_info{
        .descriptorPool = descriptor_pools_[current_pool_],
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };
    vk::DescriptorSet set;
    const vk::Result result = device_ref_.allocateDescriptorSets(&allocate_info, &set);
    if (result == vk::Result::eSuccess) return set;
    // A full pool moves on to the next one; anything else (or a fresh pool that cannot
    // hold one set) is a real error.
    if (fresh || (result != vk::Result::eErrorOutOfPoolMemory &&
                  result != vk::Result::eErrorFragmentedPool)) {
      throw std::runtime_error("vkAllocateDescriptorSets failed: " + vk::to_string(result));
    }
  }
}

void Sequence::cmd_begin() {
  spdlog::trace("Sequence::cmd_begin()");

  // The sets of the previous (completed) submit are no longer in use.
  for (const auto pool : descriptor_pools_) device_ref_.resetDescriptorPool(pool);
  current_pool_ = 0;

  // Start a fresh touched-buffer set for this chunk; the stages recorded between here
  // and submit() bind exactly the buffers this task uses, so submit/wait flush and
  // invalidate only those (not every pooled allocation).
  touched_.clear();

  constexpr vk::CommandBufferBeginInfo begin_info{
      .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
  };

  handle_.begin(begin_info);
  written_since_barrier_.clear();
  read_since_barrier_.clear();
  dispatches_ = 0;
  barriers_ = 0;

  // Start timestamp at COMPUTE_SHADER (design A4): written once every earlier command has
  // finished its compute-shader stage, i.e. when this command buffer's compute work can
  // start. Cheap and harmless for the production path, which just ignores the result.
  if (timestamp_valid_bits_ != 0) {
    handle_.resetQueryPool(query_pool_, 0, 2);
    handle_.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, query_pool_, 0);
  }
}

void Sequence::cmd_end() const {
  spdlog::trace("Sequence::cmd_end()");
  spdlog::debug("Sequence: command buffer with {} dispatches, {} inter-stage barriers ({})",
                dispatches_,
                barriers_,
                barrier_all_ ? "BT_VK_BARRIER_ALL=1" : "tracked");

  // Make every compute-shader write in this command buffer available to the host
  // domain. A fence wait alone only guarantees completion, not host visibility
  // (Vulkan spec, Synchronization > Fences: "Signaling a fence and waiting on the host
  // does not guarantee that the results of memory accesses will be visible to the
  // host"): without this barrier, reading results after wait_for_fence() relied on
  // driver leniency.
  // invalidate_touched() after the fence still handles the CPU-cache side on
  // HOST_CACHED memory. Recorded before the closing timestamp so GPU time includes it.
  const vk::MemoryBarrier2 host_barrier{
      .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
      .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
      .dstStageMask = vk::PipelineStageFlagBits2::eHost,
      .dstAccessMask = vk::AccessFlagBits2::eHostRead,
  };
  handle_.pipelineBarrier2(vk::DependencyInfo{
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &host_barrier,
  });

  // End timestamp: written once every dispatch has completed COMPUTE_SHADER (design A4).
  if (timestamp_valid_bits_ != 0) {
    handle_.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, query_pool_, 1);
  }

  handle_.end();
}

void Sequence::cmd_stage_dependency() {
  // Global, not per-buffer: buffer barriers buy nothing on these drivers (design A3).
  // Required for cross-stage visibility (esp. on non-coherent Mali) when stages share one
  // command buffer.
  const vk::MemoryBarrier2 barrier{
      .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
      .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
      .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
      .dstAccessMask =
          vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
  };
  handle_.pipelineBarrier2(vk::DependencyInfo{
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &barrier,
  });
  written_since_barrier_.clear();
  read_since_barrier_.clear();
  ++barriers_;
}

void Sequence::track_access(const std::span<const vk::DescriptorBufferInfo> buffers,
                            const std::span<const BufferAccess> access) {
  const auto contains = [](const std::vector<VkBuffer>& set, VkBuffer b) {
    return std::ranges::find(set, b) != set.end();
  };
  const auto reads = [&](size_t i) { return (static_cast<int>(access[i]) & 1) != 0; };
  const auto writes = [&](size_t i) { return (static_cast<int>(access[i]) & 2) != 0; };

  bool needed = barrier_all_ && dispatches_ != 0;
  for (size_t i = 0; i < buffers.size() && !needed; ++i) {
    VkBuffer b = buffers[i].buffer;
    needed = contains(written_since_barrier_, b) || (writes(i) && contains(read_since_barrier_, b));
  }
  if (needed) cmd_stage_dependency();

  for (size_t i = 0; i < buffers.size(); ++i) {
    VkBuffer b = buffers[i].buffer;
    if (writes(i) && !contains(written_since_barrier_, b)) written_since_barrier_.push_back(b);
    if (reads(i) && !contains(read_since_barrier_, b)) read_since_barrier_.push_back(b);
    if (!contains(touched_, b)) touched_.push_back(b);
  }
  ++dispatches_;
}

double Sequence::get_last_gpu_time_ns() const {
  if (timestamp_valid_bits_ == 0) {
    return 0.0;
  }

  std::array<uint64_t, 2> timestamps{};
  const vk::Result result = device_ref_.getQueryPoolResults(
      query_pool_,
      0,
      2,
      sizeof(timestamps),
      timestamps.data(),
      sizeof(uint64_t),
      vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

  if (result != vk::Result::eSuccess) {
    spdlog::warn("getQueryPoolResults returned {}", vk::to_string(result));
    return 0.0;
  }

  // Mask off invalid high bits before subtracting.
  const uint64_t mask =
      (timestamp_valid_bits_ >= 64) ? ~uint64_t{0} : ((uint64_t{1} << timestamp_valid_bits_) - 1);
  const uint64_t t0 = timestamps[0] & mask;
  const uint64_t t1 = timestamps[1] & mask;

  return static_cast<double>(t1 - t0) * static_cast<double>(timestamp_period_ns_);
}

void Sequence::submit() const {
  spdlog::trace("Sequence::submit()");

  // Make any pending host writes (inputs/weights) visible to the GPU before it
  // runs. No-op on coherent memory; required on HOST_CACHED (see do_allocate).
  // Scoped to the buffers this chunk bound -- not the whole pool.
  for (VmaAllocation a : touched_allocations()) {
    vmaFlushAllocation(g_vma_allocator, a, 0, VK_WHOLE_SIZE);
  }

  const vk::SubmitInfo submit_info{
      .commandBufferCount = 1,
      .pCommandBuffers = &handle_,
  };

  std::unique_lock<std::mutex> lock;
  if (submit_mutex_) lock = std::unique_lock(*submit_mutex_);
  compute_queue_ref_.submit(submit_info, fence_);
}

void Sequence::wait_for_fence() const {
  spdlog::trace("Sequence::wait_for_fence()");

  if (vk::Result result = device_ref_.waitForFences(1, &fence_, true, UINT64_MAX);
      result != vk::Result::eSuccess) {
    spdlog::error("waitForFences failed with error: {}", vk::to_string(result));
    throw std::runtime_error("Failed to sync sequence");
  }

  // GPU work is done: invalidate host caches so subsequent CPU reads see the
  // freshly written results. No-op on coherent memory; required on HOST_CACHED
  // (this is the cache maintenance BUGS-FOUND.md §7 originally lacked). Scoped to the
  // buffers this chunk bound -- not the whole pool.
  for (VmaAllocation a : touched_allocations()) {
    vmaInvalidateAllocation(g_vma_allocator, a, 0, VK_WHOLE_SIZE);
  }
}

std::vector<VmaAllocation> Sequence::touched_allocations() const {
  std::vector<VmaAllocation> allocations;
  allocations.reserve(touched_.size());
  for (VkBuffer b : touched_) {
    for (const VulkanMemoryResource* mr : resources_) {
      if (VmaAllocation a = mr->find_allocation(b)) {
        allocations.push_back(a);
        break;
      }
    }
  }
  return allocations;
}

void Sequence::reset_fence() const {
  spdlog::trace("Sequence::reset_fence()");

  if (device_ref_.resetFences(1, &fence_) != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to reset fence");
  }
}

}  // namespace kiss_vk
