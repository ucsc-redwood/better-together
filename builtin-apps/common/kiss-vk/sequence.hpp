#pragma once

#include "algorithm.hpp"

namespace kiss_vk {

class Sequence {
 public:
  explicit Sequence(vk::Device device_ref,
                    vk::Queue compute_queue_ref,
                    uint32_t compute_queue_index);

  ~Sequence() = default;

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

 protected:
  void destroy();

 private:
  void create_sync_objects();
  void create_command_pool();
  void create_command_buffer();

  vk::Device device_ref_;
  vk::Queue compute_queue_ref_;

  uint32_t compute_queue_index_;

  vk::CommandBuffer handle_;
  vk::CommandPool command_pool_;
  vk::Fence fence_;
};

}  // namespace kiss_vk
