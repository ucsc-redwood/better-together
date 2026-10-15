#include "algorithm.hpp"

#include <unordered_map>

#include "sequence.hpp"
#include "shaders/all_shaders.hpp"
#include "spdlog/spdlog.h"

namespace kiss_vk {

// ----------------------------------------------------------------------------
// SPIR-V scan (design D5). Only what glslc emits for our compute shaders: SpecId
// decorations, LocalSize literals, and (SPIR-V 1.6) LocalSizeId over 32-bit constants.
// ----------------------------------------------------------------------------

SpirvReflection reflect_spirv(const std::span<const uint32_t> words) {
  constexpr uint32_t kMagic = 0x07230203;
  constexpr uint32_t kHeaderWords = 5;
  constexpr uint32_t kOpExecutionMode = 16, kOpConstant = 43, kOpSpecConstant = 50,
                     kOpDecorate = 71, kOpExecutionModeId = 331;
  constexpr uint32_t kDecorationSpecId = 1, kModeLocalSize = 17, kModeLocalSizeId = 38;

  if (words.size() < kHeaderWords || words[0] != kMagic) {
    throw std::runtime_error("reflect_spirv: not a SPIR-V module");
  }

  SpirvReflection r;
  std::unordered_map<uint32_t, uint32_t> spec_id_of;  // result id -> SpecId
  std::unordered_map<uint32_t, uint32_t> value_of;    // 32-bit (spec) constant -> (default) value
  std::optional<std::array<uint32_t, 3>> local_size_ids;

  for (size_t i = kHeaderWords; i < words.size();) {
    const uint32_t count = words[i] >> 16;
    const uint32_t opcode = words[i] & 0xFFFF;
    if (count == 0 || i + count > words.size()) {
      throw std::runtime_error("reflect_spirv: malformed instruction at word " + std::to_string(i));
    }
    const uint32_t* in = &words[i];
    if (opcode == kOpDecorate && count >= 4 && in[2] == kDecorationSpecId) {
      spec_id_of[in[1]] = in[3];
      r.spec_ids.insert(in[3]);
    } else if (opcode == kOpExecutionMode && count >= 6 && in[2] == kModeLocalSize) {
      r.local_size = {in[3], in[4], in[5]};
      // The branches assign different variables; CI's clang-tidy (pc build, no Vulkan
      // headers) can't resolve them and sees identical bodies.
      // NOLINTNEXTLINE(bugprone-branch-clone)
    } else if (opcode == kOpExecutionModeId && count >= 6 && in[2] == kModeLocalSizeId) {
      local_size_ids = {in[3], in[4], in[5]};
    } else if ((opcode == kOpConstant || opcode == kOpSpecConstant) && count == 4) {
      value_of[in[2]] = in[3];
    }
    i += count;
  }

  if (local_size_ids) {
    for (size_t d = 0; d < 3; ++d) {
      const uint32_t id = (*local_size_ids)[d];
      if (!value_of.contains(id)) {
        throw std::runtime_error("reflect_spirv: LocalSizeId operand is not a 32-bit constant");
      }
      r.local_size[d] = value_of.at(id);
      if (spec_id_of.contains(id)) r.local_size_spec_id[d] = spec_id_of.at(id);
    }
  }
  return r;
}

std::vector<uint32_t> load_shader_spirv(const std::string& shader_name) {
  if (!shaders::all_shaders.contains(shader_name)) {
    throw std::runtime_error("Shader " + shader_name + " not found");
  }

  const auto [shader_binary, shader_binary_size] = shaders::all_shaders.at(shader_name);

  std::vector<uint32_t> spirv(shader_binary_size / sizeof(uint32_t));
  std::memcpy(spirv.data(), shader_binary, shader_binary_size);
  return spirv;
}

// device_ref_, shader_name_ and limits_ are initialized below; CI's clang-tidy (no Vulkan
// headers) misreports them.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
Algorithm::Algorithm(VulkanMemoryResource* mr_ptr,
                     std::string shader_name,
                     const ComputeLimits& limits,
                     const PFN_vkCmdPushDescriptorSetKHR push_descriptor_fn,
                     const vk::PipelineCache pipeline_cache)
    : device_ref_(mr_ptr->get_device()),
      pipeline_cache_(pipeline_cache),
      push_descriptor_fn_(push_descriptor_fn),
      shader_name_(std::move(shader_name)),
      limits_(limits) {
  internal_.spirv_binary = load_shader_spirv(shader_name_);
  reflection_ = reflect_spirv(internal_.spirv_binary);
  create_shader_module();
}

// Not trivial: it destroys six Vulkan handles. CI's clang-tidy runs on the pc build, which
// lacks the Vulkan headers, so the body doesn't parse there and looks empty.
// NOLINTNEXTLINE(modernize-use-equals-default)
Algorithm::~Algorithm() {
  // Reverse creation order; every handle may be null on a partially built Algorithm.
  // Callers have already waited on the fences of any command buffer that used this
  // pipeline. Fallback-path sets live in the Sequence's transient pools and the pipeline
  // cache is the Engine's, so neither is destroyed here.
  if (pipeline_) device_ref_.destroyPipeline(pipeline_);
  if (pipeline_layout_) device_ref_.destroyPipelineLayout(pipeline_layout_);
  if (descriptor_set_layout_) device_ref_.destroyDescriptorSetLayout(descriptor_set_layout_);
  if (shader_module_) device_ref_.destroyShaderModule(shader_module_);
}

std::shared_ptr<Algorithm> Algorithm::work_group_size(const uint32_t x,
                                                      const uint32_t y,
                                                      const uint32_t z) {
  internal_.work_group_size[0] = x;
  internal_.work_group_size[1] = y;
  internal_.work_group_size[2] = z;
  return shared_from_this();
}

std::shared_ptr<Algorithm> Algorithm::subgroup_size_dependent() {
  subgroup_size_dependent_ = true;
  return shared_from_this();
}

std::shared_ptr<Algorithm> Algorithm::num_buffers(const size_t n) {
  internal_.num_buffers = n;
  create_descriptor_set_layout();
  return shared_from_this();
}

std::shared_ptr<Algorithm> Algorithm::push_constant_size(const size_t size_in_bytes) {
  internal_.push_constant_size = size_in_bytes;
  return shared_from_this();
}

void Algorithm::update_push_constant(const void* data_ptr, const size_t size_in_bytes) {
  if (!has_push_constants()) {
    throw std::runtime_error("Algorithm has no push constants allocated");
  }

  if (size_in_bytes != internal_.push_constant_size) {
    throw std::runtime_error("Push constant size mismatch");
  }

  std::memcpy(push_constants_buffer_.data(), data_ptr, size_in_bytes);
}

std::shared_ptr<Algorithm> Algorithm::build() {
  create_pipeline();
  return shared_from_this();
}

// ----------------------------------------------------------------------------
// record_bind / record_bind_push
// ----------------------------------------------------------------------------

void Algorithm::record_bind(Sequence& seq,
                            const std::initializer_list<vk::DescriptorBufferInfo> buffers) const {
  if (buffers.size() != internal_.num_buffers) {
    throw std::runtime_error("Pipeline [" + shader_name_ + "]: record_bind got " +
                             std::to_string(buffers.size()) + " buffers, the layout has " +
                             std::to_string(internal_.num_buffers));
  }

  // Push descriptors ignore dstSet. The fallback takes a set from the Sequence's transient
  // pool, written now and freed by its next cmd_begin().
  const vk::DescriptorSet set =
      push_descriptor_fn_ ? nullptr : seq.allocate_descriptor_set(descriptor_set_layout_);

  std::array<vk::WriteDescriptorSet, kMaxBindingBuffers> writes;
  for (uint32_t binding = 0; binding < buffers.size(); ++binding) {
    writes[binding] = vk::WriteDescriptorSet{
        .dstSet = set,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pBufferInfo = buffers.begin() + binding,
    };
  }
  const auto num_writes = static_cast<uint32_t>(buffers.size());

  const vk::CommandBuffer cmd = seq.get_handle();
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
  if (push_descriptor_fn_) {
    push_descriptor_fn_(cmd,
                        VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipeline_layout_,
                        0,
                        num_writes,
                        reinterpret_cast<const VkWriteDescriptorSet*>(writes.data()));
  } else {
    device_ref_.updateDescriptorSets(num_writes, writes.data(), 0, nullptr);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout_, 0, set, {});
  }
}

void Algorithm::record_bind_push(const vk::CommandBuffer& cmd_buf) const {
  spdlog::trace("Algorithm::record_bind_push(), push_constant_size: {}",
                internal_.push_constant_size);

  if (!has_push_constants()) {
    throw std::runtime_error("Push constants not allocated");
  }

  cmd_buf.pushConstants(pipeline_layout_,
                        vk::ShaderStageFlagBits::eCompute,
                        0,
                        static_cast<uint32_t>(internal_.push_constant_size),
                        push_constants_buffer_.data());
}

void Algorithm::record_dispatch(const vk::CommandBuffer& cmd_buf,
                                const std::array<uint32_t, 3> grid_size) const {
  spdlog::trace(
      "Algorithm::record_dispatch(), grid_size: ({}, {}, {}), work_group_size: ({}, {}, {})",
      grid_size[0],
      grid_size[1],
      grid_size[2],
      internal_.work_group_size[0],
      internal_.work_group_size[1],
      internal_.work_group_size[2]);

  cmd_buf.dispatch(grid_size[0], grid_size[1], grid_size[2]);
}

// -------------------------------------------------------------------------------------------------
// Shader Related
//   create_shader_module();
// -------------------------------------------------------------------------------------------------

void Algorithm::create_shader_module() {
  if (shader_name_.empty()) {
    throw std::runtime_error("Shader name is empty");
  }

  if (internal_.spirv_binary.empty()) {
    throw std::runtime_error("SPIRV binary is empty");
  }

  const vk::ShaderModuleCreateInfo create_info{
      .codeSize = internal_.spirv_binary.size() * sizeof(uint32_t),
      .pCode = internal_.spirv_binary.data(),
  };

  shader_module_ = device_ref_.createShaderModule(create_info);

  spdlog::trace("Shader module [{}] created successfully", shader_name_);
}

// ----------------------------------------------------------------------------
// Descriptor Related
//   create_descriptor_set_layout();
// ----------------------------------------------------------------------------

void Algorithm::create_descriptor_set_layout() {
  spdlog::trace("Algorithm::create_descriptor_set_layout() num_buffers: {}", internal_.num_buffers);

  if (internal_.num_buffers == 0) {
    throw std::runtime_error("Number of buffers is 0");
  }
  if (internal_.num_buffers > kMaxBindingBuffers) {
    throw std::runtime_error("Pipeline [" + shader_name_ +
                             "]: " + std::to_string(internal_.num_buffers) + " buffers exceeds " +
                             std::to_string(kMaxBindingBuffers));
  }

  std::vector<vk::DescriptorSetLayoutBinding> bindings;
  bindings.reserve(internal_.num_buffers);

  for (uint32_t i = 0; i < internal_.num_buffers; ++i) {
    bindings.emplace_back(vk::DescriptorSetLayoutBinding{
        .binding = i,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
    });
  }

  // Push descriptors need a PUSH_DESCRIPTOR layout; the fallback allocates ordinary sets.
  const vk::DescriptorSetLayoutCreateInfo create_info{
      .flags = push_descriptor_fn_ ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                                   : vk::DescriptorSetLayoutCreateFlags{},
      .bindingCount = static_cast<uint32_t>(bindings.size()),
      .pBindings = bindings.data(),
  };

  descriptor_set_layout_ = device_ref_.createDescriptorSetLayout(create_info);
}

// ----------------------------------------------------------------------------
// Pipeline Related
//   create_pipeline();
// ----------------------------------------------------------------------------

void Algorithm::create_pipeline() {
  spdlog::trace("Algorithm::create_pipeline()");

  if (descriptor_set_layout_ == nullptr) {
    throw std::runtime_error("Descriptor set layout is not initialized");
  }

  if (internal_.num_buffers == 0) {
    throw std::runtime_error("Number of buffers is 0");
  }

  if (internal_.work_group_size[0] == 0 || internal_.work_group_size[1] == 0 ||
      internal_.work_group_size[2] == 0) {
    throw std::runtime_error("Work group size is not set");
  }

  assert(internal_.push_constant_size <= push_constants_buffer_.size());

  // Specialization constants (design D5).
  std::vector<vk::SpecializationMapEntry> spec_entries;
  std::vector<uint32_t> spec_values;
  auto specialize = [&](const uint32_t spec_id, const uint32_t value) {
    spec_entries.push_back(vk::SpecializationMapEntry{
        .constantID = spec_id,
        .offset = static_cast<uint32_t>(spec_values.size() * sizeof(uint32_t)),
        .size = sizeof(uint32_t),
    });
    spec_values.push_back(value);
  };

  const auto& wg = internal_.work_group_size;
  const auto wg_str = [](const auto& s) { return fmt::format("({}, {}, {})", s[0], s[1], s[2]); };

  // The declared workgroup size must fit the device's compute limits.
  const auto& max_wg = limits_.max_work_group_size;
  if (wg[0] > max_wg[0] || wg[1] > max_wg[1] || wg[2] > max_wg[2]) {
    throw std::runtime_error("Pipeline [" + shader_name_ + "]: workgroup size " + wg_str(wg) +
                             " exceeds maxComputeWorkGroupSize " + wg_str(max_wg));
  }
  const uint64_t invocations = uint64_t{wg[0]} * wg[1] * wg[2];
  if (invocations > limits_.max_work_group_invocations) {
    throw std::runtime_error(
        fmt::format("Pipeline [{}]: workgroup size {} = {} invocations exceeds "
                    "maxComputeWorkGroupInvocations {}",
                    shader_name_,
                    wg_str(wg),
                    invocations,
                    limits_.max_work_group_invocations));
  }

  // Each dimension is either specialized (local_size_*_id) or must match the literal.
  for (size_t d = 0; d < 3; ++d) {
    if (const auto spec_id = reflection_.local_size_spec_id[d]) {
      specialize(*spec_id, wg[d]);
    } else if (wg[d] != reflection_.local_size[d]) {
      throw std::runtime_error("Pipeline [" + shader_name_ + "]: declared workgroup size " +
                               wg_str(wg) + " but the SPIR-V LocalSize is " +
                               wg_str(reflection_.local_size));
    }
  }

  // Subgroup size: from the device, never from the shader's default.
  const bool declares_subgroup_size = reflection_.spec_ids.contains(kSubgroupSizeSpecId);
  if (declares_subgroup_size != subgroup_size_dependent_) {
    throw std::runtime_error(
        "Pipeline [" + shader_name_ + "]: " +
        (subgroup_size_dependent_
             ? "subgroup_size_dependent() but the shader has no constant_id = 0"
             : "the shader declares constant_id = 0 (subgroup size) but the Algorithm is not "
               "subgroup_size_dependent()"));
  }
  const bool pin = subgroup_size_dependent_ && limits_.subgroup_size_pinned;
  if (subgroup_size_dependent_) specialize(kSubgroupSizeSpecId, limits_.subgroup_size);
  if (pin) {
    // VUID-VkPipelineShaderStageCreateInfo-pNext-02756
    const uint64_t max_pinned =
        uint64_t{limits_.subgroup_size} * limits_.max_compute_workgroup_subgroups;
    if (invocations > max_pinned) {
      throw std::runtime_error(fmt::format(
          "Pipeline [{}]: workgroup size {} = {} invocations exceeds requiredSubgroupSize {} x "
          "maxComputeWorkgroupSubgroups {}",
          shader_name_,
          wg_str(wg),
          invocations,
          limits_.subgroup_size,
          limits_.max_compute_workgroup_subgroups));
    }
  }

  const vk::SpecializationInfo spec_info{
      .mapEntryCount = static_cast<uint32_t>(spec_entries.size()),
      .pMapEntries = spec_entries.data(),
      .dataSize = spec_values.size() * sizeof(uint32_t),
      .pData = spec_values.data(),
  };
  const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo required_subgroup_size{
      .requiredSubgroupSize = limits_.subgroup_size,
  };

  // Push Constants
  std::vector<vk::PushConstantRange> push_constant_ranges;

  if (has_push_constants()) {
    push_constant_ranges.emplace_back(vk::PushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = static_cast<uint32_t>(internal_.push_constant_size),
    });
  }

  // Pipeline Layout
  const vk::PipelineLayoutCreateInfo pipeline_layout_create_info{
      .setLayoutCount = 1,
      .pSetLayouts = &descriptor_set_layout_,
      .pushConstantRangeCount = static_cast<uint32_t>(push_constant_ranges.size()),
      .pPushConstantRanges = push_constant_ranges.empty() ? nullptr : push_constant_ranges.data()};

  pipeline_layout_ = device_ref_.createPipelineLayout(pipeline_layout_create_info);

  // Pipeline
  const vk::PipelineShaderStageCreateInfo shader_stage_create_info{
      .pNext = pin ? &required_subgroup_size : nullptr,
      .stage = vk::ShaderStageFlagBits::eCompute,
      .module = shader_module_,
      .pName = "main",
      .pSpecializationInfo = spec_entries.empty() ? nullptr : &spec_info,
  };

  // Did the Engine's (persisted) cache supply this pipeline? Debug log only.
  vk::PipelineCreationFeedback feedback{};
  const vk::PipelineCreationFeedbackCreateInfo feedback_info{
      .pPipelineCreationFeedback = &feedback,
  };

  const vk::ComputePipelineCreateInfo pipeline_create_info{
      .pNext = &feedback_info,
      .stage = shader_stage_create_info,
      .layout = pipeline_layout_,
      .basePipelineHandle = nullptr,
  };

  pipeline_ = device_ref_.createComputePipeline(pipeline_cache_, pipeline_create_info).value;

  const char* cache_result =
      !(feedback.flags & vk::PipelineCreationFeedbackFlagBits::eValid) ? "not reported"
      : (feedback.flags & vk::PipelineCreationFeedbackFlagBits::eApplicationPipelineCacheHit)
          ? "hit"
          : "miss";
  // The two debug lines differ; CI's clang-tidy (no Vulkan headers) can't resolve them.
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (subgroup_size_dependent_) {
    spdlog::debug("Pipeline [{}] created successfully (subgroup size {}, {}; pipeline cache {})",
                  shader_name_,
                  limits_.subgroup_size,
                  pin ? "pinned" : "not pinned",
                  cache_result);
  } else {
    spdlog::debug(
        "Pipeline [{}] created successfully (pipeline cache {})", shader_name_, cache_result);
  }
}

}  // namespace kiss_vk