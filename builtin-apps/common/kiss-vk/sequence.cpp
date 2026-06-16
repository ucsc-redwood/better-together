#include "sequence.hpp"

#include <vulkan/vulkan.hpp>

#include <array>

namespace kiss_vk {

Sequence::Sequence(const vk::Device device_ref,
                   const vk::Queue compute_queue_ref,
                   const uint32_t compute_queue_index,
                   const float timestamp_period_ns,
                   const uint32_t timestamp_valid_bits)
    : device_ref_(device_ref),
      compute_queue_ref_(compute_queue_ref),
      compute_queue_index_(compute_queue_index),
      timestamp_period_ns_(timestamp_period_ns),
      timestamp_valid_bits_(timestamp_valid_bits) {
  spdlog::trace("Sequence constructor");

  create_sync_objects();
  create_command_pool();
  create_command_buffer();
  create_query_pool();
}

Sequence::~Sequence() {
  if (timestamp_valid_bits_ != 0 && query_pool_) {
    device_ref_.destroyQueryPool(query_pool_);
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

void Sequence::cmd_begin() const {
  spdlog::trace("Sequence::cmd_begin()");

  constexpr vk::CommandBufferBeginInfo begin_info{
      .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
  };

  handle_.begin(begin_info);

  // Record a top-of-pipe timestamp so we can measure GPU-side elapsed time of
  // this command buffer. Cheap and harmless for the production path, which just
  // ignores the result.
  if (timestamp_valid_bits_ != 0) {
    handle_.resetQueryPool(query_pool_, 0, 2);
    handle_.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, query_pool_, 0);
  }
}

void Sequence::cmd_end() const {
  spdlog::trace("Sequence::cmd_end()");

  if (timestamp_valid_bits_ != 0) {
    handle_.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, query_pool_, 1);
  }

  handle_.end();
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
  const uint64_t mask = (timestamp_valid_bits_ >= 64)
                            ? ~uint64_t{0}
                            : ((uint64_t{1} << timestamp_valid_bits_) - 1);
  const uint64_t t0 = timestamps[0] & mask;
  const uint64_t t1 = timestamps[1] & mask;

  return static_cast<double>(t1 - t0) * static_cast<double>(timestamp_period_ns_);
}

// void Sequence::insert_compute_memory_barrier() const {
//   // For compute passes that read->write a buffer, we often do:
//   //   srcAccess = SHADER_WRITE
//   //   dstAccess = SHADER_READ
//   //   pipeline stages = COMPUTE -> COMPUTE

//   vk::MemoryBarrier memory_barrier{.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
//                                    .dstAccessMask = vk::AccessFlagBits::eShaderRead};

//   handle_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,  // source stage
//                           vk::PipelineStageFlagBits::eComputeShader,  // destination stage
//                           vk::DependencyFlags{},                      // flags
//                           1,
//                           &memory_barrier,  // memory barrier(s)
//                           0,
//                           nullptr,  // buffer barriers
//                           0,
//                           nullptr  // image barriers
//   );
// }

void Sequence::launch_kernel_async() const {
  spdlog::trace("Sequence::launch_kernel_async()");

  const vk::SubmitInfo submit_info{
      .commandBufferCount = 1,
      .pCommandBuffers = &handle_,
  };

  compute_queue_ref_.submit(submit_info, fence_);
}

void Sequence::sync() const {
  spdlog::trace("Sequence::sync()");

  if (device_ref_.waitForFences(1, &fence_, true, UINT64_MAX) != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to sync sequence");
  }

  if (device_ref_.resetFences(1, &fence_) != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to reset sequence");
  }
}
// ----------------------------------------------------------------------------
// new names
// ----------------------------------------------------------------------------

void Sequence::submit() const {
  spdlog::trace("Sequence::launch_kernel_async()");

  const vk::SubmitInfo submit_info{
      .commandBufferCount = 1,
      .pCommandBuffers = &handle_,
  };

  compute_queue_ref_.submit(submit_info, fence_);
}

void Sequence::wait_for_fence() const {
  spdlog::trace("Sequence::wait_for_fence()");

  // if (device_ref_.waitForFences(1, &fence_, true, UINT64_MAX) != vk::Result::eSuccess) {
  //   throw std::runtime_error("Failed to wait for fence");
  // }

  if (vk::Result result = device_ref_.waitForFences(1, &fence_, true, UINT64_MAX);
      result != vk::Result::eSuccess) {
    spdlog::error("waitForFences failed with error: {}", vk::to_string(result));
    throw std::runtime_error("Failed to sync sequence");
  }
}

void Sequence::reset_fence() const {
  spdlog::trace("Sequence::reset_fence()");

  if (device_ref_.resetFences(1, &fence_) != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to reset fence");
  }
}

// void Sequence::record_commands(const Algorithm* algo,
//                                const std::array<uint32_t, 3> grid_size) const {
//   spdlog::trace("Sequence::record_commands()");

//   cmd_begin();

//   algo->record_bind_core(handle_);
//   if (algo->has_push_constants()) {
//     algo->record_bind_push(handle_);
//   }

//   algo->record_dispatch(handle_, grid_size);

//   cmd_end();
// }

}  // namespace kiss_vk
