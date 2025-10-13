#include "algorithm.hpp"

#include "shaders/all_shaders.hpp"
#include "spdlog/spdlog.h"

namespace kiss_vk {
Algorithm::Algorithm(VulkanMemoryResource* mr_ptr, std::string shader_name)
    : device_ref_(mr_ptr->get_device()), shader_name_(std::move(shader_name)) {
  load_compiled_shader();
  create_shader_module();
}

std::shared_ptr<Algorithm> Algorithm::work_group_size(const uint32_t x,
                                                      const uint32_t y,
                                                      const uint32_t z) {
  internal_.work_group_size[0] = x;
  internal_.work_group_size[1] = y;
  internal_.work_group_size[2] = z;
  return shared_from_this();
}

std::shared_ptr<Algorithm> Algorithm::num_buffers(const size_t n) {
  internal_.num_buffers = n;
  create_descriptor_set_layout();
  create_descriptor_pool();
  allocate_descriptor_sets();
  return shared_from_this();
}

std::shared_ptr<Algorithm> Algorithm::num_sets(const size_t count) {
  if (count == 0) {
    throw std::runtime_error("Number of sets cannot be 0");
  }

  descriptor_set_count_ = static_cast<uint32_t>(count);
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

// ----------------------------------------------------------------------------
// New multi-set version
// ----------------------------------------------------------------------------

void Algorithm::update_descriptor_set(
    uint32_t set_index, const std::vector<vk::DescriptorBufferInfo>& buffer_infos) const {
  spdlog::trace("Algorithm::update_descriptor_set() set_index: {}", set_index);

  if (set_index >= descriptor_sets_.size()) {
    throw std::runtime_error("set_index out of range");
  }
  if (buffer_infos.size() != internal_.num_buffers) {
    throw std::runtime_error("Unexpected number of buffers for this layout");
  }

  const vk::DescriptorSet dst_set = descriptor_sets_[set_index];

  // Build the writes
  std::vector<vk::WriteDescriptorSet> writes;
  writes.reserve(buffer_infos.size());
  for (uint32_t binding = 0; binding < static_cast<uint32_t>(buffer_infos.size()); ++binding) {
    writes.push_back(vk::WriteDescriptorSet{.dstSet = dst_set,
                                            .dstBinding = binding,
                                            .dstArrayElement = 0,
                                            .descriptorCount = 1,
                                            .descriptorType = vk::DescriptorType::eStorageBuffer,
                                            .pImageInfo = nullptr,
                                            .pBufferInfo = &buffer_infos[binding],
                                            .pTexelBufferView = nullptr});
  }

  // Issue the update
  device_ref_.updateDescriptorSets(writes, {});
}

std::shared_ptr<Algorithm> Algorithm::build() {
  create_pipeline();
  return shared_from_this();
}

// ----------------------------------------------------------------------------
// record_bind_core / record_bind_push
// ----------------------------------------------------------------------------

void Algorithm::record_bind_core(const vk::CommandBuffer& cmd_buf, uint32_t set_index) const {
  if (set_index >= descriptor_sets_.size()) {
    throw std::runtime_error("Invalid descriptor set index");
  }
  cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
  cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             pipeline_layout_,
                             0,  // firstSet
                             descriptor_sets_[set_index],
                             {});
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
//   load_compiled_shader();
//   create_shader_module();
// -------------------------------------------------------------------------------------------------

void Algorithm::load_compiled_shader() {
  if (!shaders::all_shaders.contains(shader_name_)) {
    throw std::runtime_error("Shader " + shader_name_ + " not found");
  }

  const auto [shader_binary, shader_binary_size] = shaders::all_shaders.at(shader_name_);

  internal_.spirv_binary.resize(shader_binary_size / sizeof(uint32_t));
  std::memcpy(internal_.spirv_binary.data(), shader_binary, shader_binary_size);
}

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
//   create_descriptor_pool();
//   allocate_descriptor_sets();
// ----------------------------------------------------------------------------

void Algorithm::create_descriptor_set_layout() {
  spdlog::trace("Algorithm::create_descriptor_set_layout() num_buffers: {}", internal_.num_buffers);

  if (internal_.num_buffers == 0) {
    throw std::runtime_error("Number of buffers is 0");
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

  const vk::DescriptorSetLayoutCreateInfo create_info{
      .bindingCount = static_cast<uint32_t>(bindings.size()),
      .pBindings = bindings.data(),
  };

  descriptor_set_layout_ = device_ref_.createDescriptorSetLayout(create_info);
}

void Algorithm::create_descriptor_pool() {
  const uint32_t total_descriptors = descriptor_set_count_ * internal_.num_buffers;

  const std::vector<vk::DescriptorPoolSize> pool_sizes{
      {
          .type = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = total_descriptors,
      },
      // ^ we need enough descriptors for each set
  };

  const vk::DescriptorPoolCreateInfo create_info{
      .maxSets = descriptor_set_count_,
      .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
      .pPoolSizes = pool_sizes.data(),
  };

  descriptor_pool_ = device_ref_.createDescriptorPool(create_info);
}

void Algorithm::allocate_descriptor_sets() {
  if (descriptor_pool_ == nullptr || descriptor_set_layout_ == nullptr) {
    throw std::runtime_error("Descriptor pool or set layout is not initialized");
  }

  std::vector<vk::DescriptorSetLayout> layouts(descriptor_set_count_, descriptor_set_layout_);

  const vk::DescriptorSetAllocateInfo allocate_info{
      .descriptorPool = descriptor_pool_,
      .descriptorSetCount = descriptor_set_count_,
      .pSetLayouts = layouts.data(),
  };

  descriptor_sets_ = device_ref_.allocateDescriptorSets(allocate_info);

  // // old single-set version
  // descriptor_set_ = device_ref_.allocateDescriptorSets(allocate_info).front();
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

  pipeline_cache_ = device_ref_.createPipelineCache(vk::PipelineCacheCreateInfo{});

  // Pipeline
  const vk::PipelineShaderStageCreateInfo shader_stage_create_info{
      .stage = vk::ShaderStageFlagBits::eCompute,
      .module = shader_module_,
      .pName = "main",
  };

  const vk::ComputePipelineCreateInfo pipeline_create_info{
      .stage = shader_stage_create_info,
      .layout = pipeline_layout_,
      .basePipelineHandle = nullptr,
  };

  pipeline_ = device_ref_.createComputePipeline(pipeline_cache_, pipeline_create_info).value;

  spdlog::debug("Pipeline [{}] created successfully", shader_name_);
}

}  // namespace kiss_vk