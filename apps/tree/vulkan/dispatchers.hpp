#pragma once

#include "platform/engine/vulkan/engine.hpp"
#include "vk_appdata.hpp"

namespace tree::vulkan {

class VulkanDispatcher final {
 public:
  explicit VulkanDispatcher();

  // disallow copy/move constructor
  VulkanDispatcher(const VulkanDispatcher&) = delete;
  VulkanDispatcher& operator=(const VulkanDispatcher&) = delete;
  VulkanDispatcher(VulkanDispatcher&&) = delete;
  VulkanDispatcher& operator=(VulkanDispatcher&&) = delete;

  kiss_vk::VulkanMemoryResource::memory_resource* get_mr() { return engine.get_mr(); }

  // Sequence handle, used by benchmarks to read device-side GPU time per stage.
  kiss_vk::Sequence* get_seq() { return seq.get(); }

  // Per-stage single-submit path (cmd_begin -> record -> submit -> wait_for_fence).
  // Kept for the per-stage device-time benchmarks: bm_main.cpp calls these directly and
  // reads get_seq()->get_last_gpu_time_ns() for that one stage. Thin wrappers around
  // dispatch_multi_stage(k, k) so the [stage, stage] path stays byte-for-byte the same.
  void run_stage_1(VkAppData_Safe& appdata);
  void run_stage_2(VkAppData_Safe& appdata);
  void run_stage_3(VkAppData_Safe& appdata);
  void run_stage_4(VkAppData_Safe& appdata);
  void run_stage_5(VkAppData_Safe& appdata);
  void run_stage_6(VkAppData_Safe& appdata);
  void run_stage_7(VkAppData_Safe& appdata);

  void dispatch_stage(VkAppData_Safe& data, const int stage) {
    dispatch_multi_stage(data, stage, stage);
  }

  // Record stages [start_stage, end_stage] into ONE command buffer with a
  // shaderWrite->shaderRead barrier between consecutive stages, then run it with ONE
  // submit + ONE fence wait. Collapses the per-stage CPU<->GPU round-trips of a chunk;
  // host cache flush/invalidate happen once per chunk (inside submit/wait_for_fence).
  // Safe across tree's data-dependent stages: VkAppData_Safe's counts (n_unique,
  // n_brt_nodes) are const, fixed at construction -- they don't need a GPU read-back
  // between stages, so a stage can be recorded before the prior stage executes.
  void dispatch_multi_stage(VkAppData_Safe& data, const int start_stage, const int end_stage) {
    if (start_stage < 1 || end_stage > 7 || start_stage > end_stage)
      throw std::out_of_range("Invalid stage");

    seq->cmd_begin();
    for (int stage = start_stage; stage <= end_stage; ++stage) {
      (this->*record_functions[stage - 1])(data, seq->get_handle());
      if (stage != end_stage) seq->cmd_memory_barrier();
    }
    seq->cmd_end();

    seq->reset_fence();
    seq->submit();
    seq->wait_for_fence();
  }

 private:
  // Record-only stage bodies: descriptor-set update(s) + bind + dispatch(es) into `cmd`
  // (keeping each stage's own intra-stage barriers), no cmd_begin/submit/wait. The scan
  // algos are shared by stage 3 and stage 6, so each binds a distinct scan descriptor-set
  // index (stage 3 -> 0, stage 6 -> 1) to avoid clobbering when both are in one buffer.
  void record_stage_1(VkAppData_Safe& appdata, vk::CommandBuffer cmd);
  void record_stage_2(VkAppData_Safe& appdata, vk::CommandBuffer cmd);
  void record_stage_3(VkAppData_Safe& appdata, vk::CommandBuffer cmd);
  void record_stage_4(VkAppData_Safe& appdata, vk::CommandBuffer cmd);
  void record_stage_5(VkAppData_Safe& appdata, vk::CommandBuffer cmd);
  void record_stage_6(VkAppData_Safe& appdata, vk::CommandBuffer cmd);
  void record_stage_7(VkAppData_Safe& appdata, vk::CommandBuffer cmd);

  using RecordFn = void (VulkanDispatcher::*)(VkAppData_Safe&, vk::CommandBuffer);

  static constexpr std::array<RecordFn, 7> record_functions = {
      &VulkanDispatcher::record_stage_1,
      &VulkanDispatcher::record_stage_2,
      &VulkanDispatcher::record_stage_3,
      &VulkanDispatcher::record_stage_4,
      &VulkanDispatcher::record_stage_5,
      &VulkanDispatcher::record_stage_6,
      &VulkanDispatcher::record_stage_7,
  };

  // Record a device-wide INCLUSIVE prefix scan of `src` (n uints) into `dst`,
  // using `block_sums` as scratch, into the already-open command buffer `cmd`.
  // Shared by stage 3 (flag scan) and stage 6 (edge-count scan).
  void record_device_scan(vk::CommandBuffer cmd,
                          vk::DescriptorBufferInfo src,
                          vk::DescriptorBufferInfo dst,
                          vk::DescriptorBufferInfo block_sums,
                          uint32_t n,
                          uint32_t descriptor_set);

  kiss_vk::Engine engine;
  std::shared_ptr<kiss_vk::Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<kiss_vk::Algorithm>> cached_algorithms;

  // --------------------------------------------------------------------------
  // Temporary storages
  // --------------------------------------------------------------------------

  // (n + 255) / 256;
  // std::pmr::vector<uint32_t> tmp_u_sums;
  // std::pmr::vector<uint32_t> tmp_u_prefix_sums;
};

}  // namespace tree::vulkan
