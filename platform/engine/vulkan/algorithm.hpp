#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include "base_engine.hpp"
#include "vk.hpp"
#include "vma_pmr.hpp"

namespace kiss_vk {

class Sequence;

// Most storage buffers one Algorithm binds (num_buffers); record_bind keeps its writes on
// the stack. Well under the guaranteed maxPushDescriptors (32).
inline constexpr uint32_t kMaxBindingBuffers = 16;

// Spec constant id carrying the subgroup size into subgroup-dependent shaders:
// `layout(constant_id = 0) const uint SUBGROUP_SIZE = ...;` (design D5).
inline constexpr uint32_t kSubgroupSizeSpecId = 0;

// How a shader accesses one storage-buffer binding: from its GLSL `readonly` (NonWritable)
// or `writeonly` (NonReadable) qualifier; a plain `buffer` is read and written. kNone: the
// module declares no variable for the binding (an unused binding the optimizer removed).
enum class BufferAccess : std::uint8_t { kNone = 0, kRead = 1, kWrite = 2, kReadWrite = 3 };

// What Algorithm needs from a compute shader's SPIR-V, found by a scan of the module's
// OpDecorate SpecId, OpExecutionMode LocalSize and OpExecutionModeId LocalSizeId, and the
// NonWritable/NonReadable decorations of its descriptor-set-0 buffers.
struct SpirvReflection {
  std::set<uint32_t> spec_ids{};  // every SpecId the module declares
  // Workgroup size: the LocalSize literals, or the LocalSizeId constants' (default) values.
  std::array<uint32_t, 3> local_size{};
  // The SpecId behind each LocalSizeId operand (`local_size_{x,y,z}_id`), if any.
  std::array<std::optional<uint32_t>, 3> local_size_spec_id;
  // Binding -> access, for every buffer variable in descriptor set 0 (design A3).
  std::map<uint32_t, BufferAccess> binding_access{};
};

[[nodiscard]] SpirvReflection reflect_spirv(std::span<const uint32_t> words);

// The baked SPIR-V of a shader in shaders/all_shaders.hpp; throws on an unknown name.
[[nodiscard]] std::vector<uint32_t> load_shader_spirv(const std::string& shader_name);

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
  // push_descriptor_fn: BaseEngine::push_descriptor_fn(), null for the transient-pool path.
  // pipeline_cache: the Engine's (design D7), not owned.
  explicit Algorithm(VulkanMemoryResource* mr_ptr,
                     std::string shader_name,
                     const ComputeLimits& limits,
                     PFN_vkCmdPushDescriptorSetKHR push_descriptor_fn,
                     vk::PipelineCache pipeline_cache);

  // Destroys every Vulkan object this Algorithm created (VUID-vkDestroyDevice-device-05137).
  // Owners must destroy it before the Engine; dispatchers declare `engine` first, so their
  // cached algorithms go first.
  ~Algorithm();

  // Holds raw vk::Pipeline/ShaderModule/DescriptorSetLayout handles and is only ever used
  // through a shared_ptr (Engine::make_algo + enable_shared_from_this). A copy would
  // alias those handles and break the shared_from_this identity, so forbid copy/move.
  Algorithm(const Algorithm&) = delete;
  Algorithm& operator=(const Algorithm&) = delete;
  Algorithm(Algorithm&&) = delete;
  Algorithm& operator=(Algorithm&&) = delete;

  // Builder pattern
  // The workgroup size the shader runs with: fed to its `local_size_{x,y,z}_id` spec
  // constants where it declares them; elsewhere it must equal the SPIR-V's LocalSize.
  // build() throws on a mismatch or when it exceeds the device's compute limits.
  [[nodiscard]] std::shared_ptr<Algorithm> work_group_size(uint32_t x, uint32_t y, uint32_t z);
  // The shader depends on the subgroup size (spec constant kSubgroupSizeSpecId): it is
  // specialized with the device's size, pinned when the device supports that (design D5).
  [[nodiscard]] std::shared_ptr<Algorithm> subgroup_size_dependent();
  // Specialize the shader's `constant_id = id` (a 32-bit int, uint or bool) with `value`; build()
  // throws if the shader declares no such constant. Not for the subgroup size or the workgroup
  // size.
  [[nodiscard]] std::shared_ptr<Algorithm> spec_constant(uint32_t id, uint32_t value);
  [[nodiscard]] std::shared_ptr<Algorithm> num_buffers(size_t n);
  [[nodiscard]] std::shared_ptr<Algorithm> push_constant_size(size_t size_in_bytes);
  template <typename T>
  [[nodiscard]] std::shared_ptr<Algorithm> push_constant() {
    return push_constant_size(sizeof(T));
  }
  [[nodiscard]] std::shared_ptr<Algorithm> build();

  /**
   * @brief Bind the pipeline and the stage's buffers into `seq`'s open command buffer
   * @param buffers One buffer per binding 0..num_buffers-1
   *
   * The bindings are captured at record time (push descriptors, or a set from the
   * Sequence's transient pool; design D6), so the same Algorithm may be recorded several
   * times into one command buffer with different buffers.
   *
   * Example usage:
   * ```cpp
   * algo->record_bind(*seq, {
   *   engine.get_buffer_info(input),
   *   engine.get_buffer_info(output)
   * });
   * ```
   */
  void record_bind(Sequence& seq, std::initializer_list<vk::DescriptorBufferInfo> buffers) const;

  /**
   * @brief Record this stage's push constants into `seq`'s open command buffer
   *
   * The values travel with the call (nothing is stored in the Algorithm), so several
   * threads may record the same Algorithm into their own Sequences (design C2).
   *
   * Example usage:
   * ```cpp
   * algo->record_push_constants(seq, PushConstants{.n = n});
   * ```
   */
  template <typename T>
  void record_push_constants(Sequence& seq, const T& data) const {
    record_push_constants(seq, &data, sizeof(T));
  }
  void record_push_constants(Sequence& seq, const void* data, size_t size_in_bytes) const;

  // basically CUDA's <<< grid_size >>>
  void record_dispatch(Sequence& seq, std::array<uint32_t, 3> grid_size) const;

  [[nodiscard]] bool has_push_constants() const { return internal_.push_constant_size > 0; }

 private:
  void create_shader_module();
  void create_descriptor_set_layout();

  void create_pipeline();

  // References
  vk::Device device_ref_;
  // VulkanMemoryResource* mr_ptr_;

  // Core vulkan handles
  vk::ShaderModule shader_module_ = nullptr;
  vk::Pipeline pipeline_ = nullptr;
  vk::PipelineCache pipeline_cache_;  // the Engine's
  vk::PipelineLayout pipeline_layout_ = nullptr;
  vk::DescriptorSetLayout descriptor_set_layout_ = nullptr;
  PFN_vkCmdPushDescriptorSetKHR push_descriptor_fn_ = nullptr;

  std::string shader_name_;
  ComputeLimits limits_;
  SpirvReflection reflection_;
  bool subgroup_size_dependent_ = false;
  std::vector<std::pair<uint32_t, uint32_t>> spec_constants_;  // (constant_id, value)

  struct {
    std::vector<uint32_t> spirv_binary;
    size_t num_buffers = 0;
    size_t push_constant_size = 0;
    uint32_t work_group_size[3] = {0, 0, 0};
  } internal_;
};

}  // namespace kiss_vk
