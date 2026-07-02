#pragma once

#include "apps/cifar-dense/appdata.hpp"
#include "platform/engine/vulkan/engine.hpp"

namespace cifar_dense::vulkan {

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
  void run_stage_1(AppData& appdata);
  void run_stage_2(AppData& appdata);
  void run_stage_3(AppData& appdata);
  void run_stage_4(AppData& appdata);
  void run_stage_5(AppData& appdata);
  void run_stage_6(AppData& appdata);
  void run_stage_7(AppData& appdata);
  void run_stage_8(AppData& appdata);
  void run_stage_9(AppData& appdata);
  void run_stage_10(AppData& appdata);
  void run_stage_11(AppData& appdata);

  void dispatch_stage(AppData& data, const int stage) { dispatch_multi_stage(data, stage, stage); }

  // Record stages [start_stage, end_stage] into ONE command buffer with a
  // shaderWrite->shaderRead barrier between consecutive stages, then run it with ONE
  // submit + ONE fence wait. Collapses the per-stage CPU<->GPU round-trips of a chunk;
  // host cache flush/invalidate happen once per chunk (inside submit/wait_for_fence).
  void dispatch_multi_stage(AppData& data, const int start_stage, const int end_stage) {
    if (start_stage < 1 || end_stage > 11 || start_stage > end_stage)
      throw std::out_of_range("Invalid stage");

    seq->cmd_begin();
    for (int stage = start_stage; stage <= end_stage; ++stage) {
      (this->*record_functions[stage - 1])(data, seq->get_handle());
      if (stage != end_stage) seq->cmd_memory_barrier();
    }
    seq->cmd_end();

    seq->submit();
    seq->wait_for_fence();
    seq->reset_fence();
  }

 private:
  // Record-only stage bodies: descriptor-set update + push + bind + dispatch into `cmd`,
  // no cmd_begin/submit/wait. Each binds its own fixed descriptor-set index (occurrence
  // order within its shared algo) so several stages can share one command buffer without
  // clobbering each other's bindings.
  void record_stage_1(AppData& appdata, vk::CommandBuffer cmd);
  void record_stage_2(AppData& appdata, vk::CommandBuffer cmd);
  void record_stage_3(AppData& appdata, vk::CommandBuffer cmd);
  void record_stage_4(AppData& appdata, vk::CommandBuffer cmd);
  void record_stage_5(AppData& appdata, vk::CommandBuffer cmd);
  void record_stage_6(AppData& appdata, vk::CommandBuffer cmd);
  void record_stage_7(AppData& appdata, vk::CommandBuffer cmd);
  void record_stage_8(AppData& appdata, vk::CommandBuffer cmd);
  void record_stage_9(AppData& appdata, vk::CommandBuffer cmd);
  void record_stage_10(AppData& appdata, vk::CommandBuffer cmd);
  void record_stage_11(AppData& appdata, vk::CommandBuffer cmd);

  using RecordFn = void (VulkanDispatcher::*)(AppData&, vk::CommandBuffer);

  static constexpr std::array<RecordFn, 11> record_functions = {
      &VulkanDispatcher::record_stage_1,
      &VulkanDispatcher::record_stage_2,
      &VulkanDispatcher::record_stage_3,
      &VulkanDispatcher::record_stage_4,
      &VulkanDispatcher::record_stage_5,
      &VulkanDispatcher::record_stage_6,
      &VulkanDispatcher::record_stage_7,
      &VulkanDispatcher::record_stage_8,
      &VulkanDispatcher::record_stage_9,
      &VulkanDispatcher::record_stage_10,
      &VulkanDispatcher::record_stage_11,
  };

  kiss_vk::Engine engine;
  std::shared_ptr<kiss_vk::Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<kiss_vk::Algorithm>> cached_algorithms;
};

}  // namespace cifar_dense::vulkan
