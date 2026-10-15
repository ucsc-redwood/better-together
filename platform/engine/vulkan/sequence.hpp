#pragma once

#include "algorithm.hpp"

namespace kiss_vk {

class Sequence {
 public:
  explicit Sequence(vk::Device device_ref,
                    vk::Queue compute_queue_ref,
                    uint32_t compute_queue_index,
                    float timestamp_period_ns = 0.0f,
                    uint32_t timestamp_valid_bits = 0,
                    VulkanMemoryResource* mr = nullptr);

  ~Sequence();

  // Owns a VkCommandPool/VkFence/VkQueryPool/VkDescriptorPools freed in ~Sequence; only ever held
  // behind a shared_ptr (Engine::make_seq). Copying would double-free those handles and there is no
  // move, so make every copy/move ill-formed.
  Sequence(const Sequence&) = delete;
  Sequence& operator=(const Sequence&) = delete;
  Sequence(Sequence&&) = delete;
  Sequence& operator=(Sequence&&) = delete;

  // Also resets the transient descriptor pools, so the previous submit must have been
  // waited on (as vkBeginCommandBuffer already requires of the command buffer).
  void cmd_begin();
  void cmd_end() const;

  // One descriptor set of `layout` for the command buffer being recorded, freed by the
  // next cmd_begin() (Algorithm::record_bind, design D6). Grows by whole pools, so no
  // caller plans a set count.
  [[nodiscard]] vk::DescriptorSet allocate_descriptor_set(vk::DescriptorSetLayout layout);

  // Records a global COMPUTE->COMPUTE, shaderWrite->shaderRead memory barrier into the
  // open command buffer. Use it between stages when several stages are recorded into a
  // single command buffer (one submit per chunk) so each stage's shader writes are
  // available/visible to the next stage's shader reads. Must be called between
  // cmd_begin() and cmd_end().
  void cmd_memory_barrier() const;

  void submit() const;
  void wait_for_fence() const;
  void reset_fence() const;

  [[nodiscard]] vk::CommandBuffer get_handle() const { return handle_; }

  // GPU-side elapsed time of the last submitted command buffer, in nanoseconds.
  // Valid only after wait_for_fence(). Returns 0.0 if device timestamps are
  // unsupported (timestamp_valid_bits == 0), in which case callers should fall
  // back to wall-clock timing.
  [[nodiscard]] double get_last_gpu_time_ns() const;
  [[nodiscard]] bool gpu_timestamps_supported() const { return timestamp_valid_bits_ != 0; }

 private:
  void create_sync_objects();
  void create_command_pool();
  void create_command_buffer();
  void create_query_pool();
  [[nodiscard]] vk::DescriptorPool create_descriptor_pool() const;

  vk::Device device_ref_;
  vk::Queue compute_queue_ref_;

  uint32_t compute_queue_index_;

  vk::CommandBuffer handle_;
  vk::CommandPool command_pool_;
  vk::Fence fence_;

  // Transient descriptor pools: sets are allocated from descriptor_pools_[current_pool_]
  // and all reset together in cmd_begin().
  std::vector<vk::DescriptorPool> descriptor_pools_;
  size_t current_pool_ = 0;

  // Device-side timestamp query (2 entries: top-of-pipe, bottom-of-pipe).
  vk::QueryPool query_pool_;
  float timestamp_period_ns_;      // nanoseconds per timestamp tick
  uint32_t timestamp_valid_bits_;  // 0 => timestamps unsupported on this queue

  // Optional: the engine's memory resource, for host<->device cache maintenance
  // on non-coherent memory (flush before submit, invalidate after fence). Null =>
  // no cache maintenance (memory is coherent or caller manages it).
  VulkanMemoryResource* mr_;
};

}  // namespace kiss_vk
