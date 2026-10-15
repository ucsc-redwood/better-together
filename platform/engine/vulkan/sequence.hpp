#pragma once

#include <span>
#include <vector>

#include "algorithm.hpp"

namespace kiss_vk {

class Sequence {
 public:
  explicit Sequence(vk::Device device_ref,
                    vk::Queue compute_queue_ref,
                    uint32_t compute_queue_index,
                    float timestamp_period_ns = 0.0f,
                    uint32_t timestamp_valid_bits = 0,
                    std::vector<VulkanMemoryResource*> resources = {});

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

  // Records the inter-stage dependency (design A2): a global synchronization2 memory
  // barrier, COMPUTE_SHADER/SHADER_STORAGE_WRITE -> COMPUTE_SHADER/SHADER_STORAGE_READ|WRITE,
  // which orders read-after-write and write-after-write (and, as an execution dependency,
  // write-after-read). Must be called between cmd_begin() and cmd_end().
  void cmd_stage_dependency();

  // Called by Algorithm::record_bind before each dispatch with the buffers it binds and
  // how the shader accesses them. Records cmd_stage_dependency() only when the dispatch
  // reads or writes a buffer written since the last dependency, or writes one read since
  // it (design A3). BT_VK_BARRIER_ALL=1 records one before every dispatch but the first,
  // as before conditional barriers.
  void track_access(std::span<const vk::DescriptorBufferInfo> buffers,
                    std::span<const BufferAccess> access);

  // The buffers bound since the last cmd_begin(): what submit() flushes and
  // wait_for_fence() invalidates.
  [[nodiscard]] const std::vector<VkBuffer>& touched_buffers() const { return touched_; }

  // Dispatches and inter-stage dependencies in the command buffer recorded last.
  [[nodiscard]] uint32_t dispatch_count() const { return dispatches_; }
  [[nodiscard]] uint32_t barrier_count() const { return barriers_; }

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

  // Dependency tracking for the command buffer being recorded (track_access).
  bool barrier_all_ = false;
  std::vector<VkBuffer> written_since_barrier_;
  std::vector<VkBuffer> read_since_barrier_;
  uint32_t dispatches_ = 0;
  uint32_t barriers_ = 0;

  // Device-side timestamp query (2 entries: start and end of the command buffer's compute
  // work, both written at COMPUTE_SHADER).
  vk::QueryPool query_pool_;
  float timestamp_period_ns_;      // nanoseconds per timestamp tick
  uint32_t timestamp_valid_bits_;  // 0 => timestamps unsupported on this queue

  // Host<->device cache maintenance on non-coherent memory: the buffers bound since
  // cmd_begin() (track_access) are flushed before submit() and invalidated after
  // wait_for_fence(), each through whichever of the engine's resources owns it. Kept per
  // Sequence, so concurrently recording chunks never share it (runtime/gpu-chunks spec).
  // No resources => no cache maintenance (memory is coherent or caller manages it).
  std::vector<VulkanMemoryResource*> resources_;
  std::vector<VkBuffer> touched_;
  [[nodiscard]] std::vector<VmaAllocation> touched_allocations() const;
};

}  // namespace kiss_vk
