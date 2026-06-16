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

  void cmd_begin() const;
  void cmd_end() const;

  // void insert_compute_memory_barrier() const;

  // void record_commands(const Algorithm* algo, std::array<uint32_t, 3> grid_size) const;
  [[deprecated("use submit() instead")]] void launch_kernel_async() const;
  [[deprecated("use wait_for_fence() instead")]] void sync() const;

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

 protected:
  void destroy();

 private:
  void create_sync_objects();
  void create_command_pool();
  void create_command_buffer();
  void create_query_pool();

  vk::Device device_ref_;
  vk::Queue compute_queue_ref_;

  uint32_t compute_queue_index_;

  vk::CommandBuffer handle_;
  vk::CommandPool command_pool_;
  vk::Fence fence_;

  // Device-side timestamp query (2 entries: top-of-pipe, bottom-of-pipe).
  vk::QueryPool query_pool_;
  float timestamp_period_ns_;     // nanoseconds per timestamp tick
  uint32_t timestamp_valid_bits_;  // 0 => timestamps unsupported on this queue

  // Optional: the engine's memory resource, for host<->device cache maintenance
  // on non-coherent memory (flush before submit, invalidate after fence). Null =>
  // no cache maintenance (memory is coherent or caller manages it).
  VulkanMemoryResource* mr_;
};

}  // namespace kiss_vk
