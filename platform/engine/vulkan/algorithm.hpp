#pragma once

#include "vk.hpp"
#include "vma_pmr.hpp"

namespace kiss_vk {

/**
 * @brief Calculate the ceiling of a division
 * @param a Dividend
 * @param b Divisor
 * @return Ceiling of a divided by b
 *
 * Example usage:
 * ```cpp
 * constexpr auto grid_size = div_ceil(n, 256);
 * ```
 */
constexpr auto div_ceil(size_t a, size_t b) { return (a + b - 1) / b; }

class Algorithm final : public std::enable_shared_from_this<Algorithm> {
 public:
  explicit Algorithm(VulkanMemoryResource* mr_ptr, std::string shader_name);

  ~Algorithm() = default;

  // Builder pattern
  [[nodiscard]] std::shared_ptr<Algorithm> work_group_size(uint32_t x, uint32_t y, uint32_t z);
  [[nodiscard]] std::shared_ptr<Algorithm> num_buffers(size_t n);
  [[nodiscard]] std::shared_ptr<Algorithm> num_sets(size_t count);
  [[nodiscard]] std::shared_ptr<Algorithm> push_constant_size(size_t size_in_bytes);
  template <typename T>
  [[nodiscard]] std::shared_ptr<Algorithm> push_constant() {
    return push_constant_size(sizeof(T));
  }
  [[nodiscard]] std::shared_ptr<Algorithm> build();

  // Pass actual data
  void update_push_constant(const void* data_ptr, size_t size_in_bytes);

  /**
   * @brief Update push constant data with a templated type
   * @tparam T Type of push constant data
   * @param data Push constant data to update
   *
   * Example usage:
   * ```cpp
   * struct PushConstants {
   *   uint32_t n;
   * } pc;
   *
   * algo->update_push_constant(pc);
   * ```
   */
  template <typename T>
  void update_push_constant(const T& data) {
    update_push_constant(&data, sizeof(T));
  }

  /**
   * @brief Update buffer bindings for the compute shader
   * @param buffer_infos List of buffer descriptors to bind
   *
   * Example usage:
   * ```cpp
   * algo->update_buffer({
   *   engine.get_buffer_info(input_a),
   *   engine.get_buffer_info(input_b),
   *   engine.get_buffer_info(output)
   * });
   * ```
   */
  // void update_buffer(std::initializer_list<vk::DescriptorBufferInfo> buffer_infos);

  void update_descriptor_set(uint32_t set_index,
                             const std::vector<vk::DescriptorBufferInfo>& buffer_infos) const;

  // Used by 'Sequence' Class
  // void record_bind_core(const vk::CommandBuffer& cmd_buf) const;

  void record_bind_core(const vk::CommandBuffer& cmd_buf, uint32_t set_index) const;

  void record_bind_push(const vk::CommandBuffer& cmd_buf) const;

  // basically CUDA's <<< grid_size >>>
  void record_dispatch(const vk::CommandBuffer& cmd_buf, std::array<uint32_t, 3> grid_size) const;

  [[nodiscard]] bool has_push_constants() const { return internal_.push_constant_size > 0; }

 private:
  void load_compiled_shader();

  void create_shader_module();
  void create_descriptor_set_layout();
  void create_descriptor_pool();
  void allocate_descriptor_sets();

  void create_pipeline();

  // References
  vk::Device device_ref_;
  // VulkanMemoryResource* mr_ptr_;

  // Core vulkan handles
  vk::ShaderModule shader_module_ = nullptr;
  vk::Pipeline pipeline_ = nullptr;
  vk::PipelineCache pipeline_cache_ = nullptr;
  vk::PipelineLayout pipeline_layout_ = nullptr;
  vk::DescriptorSetLayout descriptor_set_layout_ = nullptr;
  vk::DescriptorPool descriptor_pool_ = nullptr;

  // vk::DescriptorSet descriptor_set_ = nullptr;

  // Instead of a single descriptor_set_, store a vector.
  std::vector<vk::DescriptorSet> descriptor_sets_;
  uint32_t descriptor_set_count_ = 1;  // default to 1

  std::string shader_name_;

  // Payloads (buffers and push constants)
  std::array<std::byte, 128> push_constants_buffer_;
  std::vector<vk::DescriptorBufferInfo> buffer_infos_;

  struct {
    std::vector<uint32_t> spirv_binary;
    size_t num_buffers = 0;
    size_t push_constant_size = 0;
    uint32_t work_group_size[3] = {0, 0, 0};
  } internal_;
};

}  // namespace kiss_vk
