#pragma once

#include "apps/cifar-sparse/appdata.hpp"
#include "platform/engine/vulkan/engine.hpp"

namespace cifar_sparse::vulkan {

class VulkanDispatcher final {
 public:
  explicit VulkanDispatcher();

  // disallow copy/move constructor
  VulkanDispatcher(const VulkanDispatcher&) = delete;
  VulkanDispatcher& operator=(const VulkanDispatcher&) = delete;
  VulkanDispatcher(VulkanDispatcher&&) = delete;
  VulkanDispatcher& operator=(VulkanDispatcher&&) = delete;

  kiss_vk::VulkanMemoryResource::memory_resource* get_mr() { return engine.get_mr(); }
  // The memory resource of a buffer class (runtime/placement.hpp, design B2).
  kiss_vk::VulkanMemoryResource::memory_resource* get_mr(const bt::BufferClass c) {
    return engine.get_mr(c);
  }

  // Sequence handle, used by benchmarks to read device-side GPU time per stage.
  kiss_vk::Sequence* get_seq() { return default_seq.get(); }

  // A Sequence of this dispatcher's engine: one per GPU chunk of a schedule, so chunks on
  // different worker threads never share a command buffer (design C2).
  [[nodiscard]] std::shared_ptr<kiss_vk::Sequence> make_seq() { return engine.make_seq(); }

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

  // Record stages [start_stage, end_stage] into ONE command buffer (the Sequence orders
  // dependent dispatches, design A3), then run it with ONE submit + ONE fence wait. Collapses the
  // per-stage CPU<->GPU round-trips of a chunk; host cache flush/invalidate happen once per chunk
  // (inside submit/wait_for_fence).
  void dispatch_multi_stage(AppData& data, const int start_stage, const int end_stage) {
    dispatch_multi_stage(data, start_stage, end_stage, *default_seq);
  }
  void dispatch_multi_stage(AppData& data,
                            const int start_stage,
                            const int end_stage,
                            kiss_vk::Sequence& seq) {
    if (start_stage < 1 || end_stage > 11 || start_stage > end_stage)
      throw std::out_of_range("Invalid stage");

    seq.cmd_begin();
    for (int stage = start_stage; stage <= end_stage; ++stage) {
      (this->*record_functions[stage - 1])(data, seq);
    }
    seq.cmd_end();

    seq.submit();
    seq.wait_for_fence();
    seq.reset_fence();
  }

 private:
  // Record-only stage bodies: bind (buffers captured per dispatch) + push + dispatch into
  // `cmd` (= seq's command buffer), no cmd_begin/submit/wait, so several stages can share
  // one command buffer.
  void record_stage_1(AppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_2(AppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_3(AppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_4(AppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_5(AppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_6(AppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_7(AppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_8(AppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_9(AppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_10(AppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_11(AppData& appdata, kiss_vk::Sequence& seq);

  using RecordFn = void (VulkanDispatcher::*)(AppData&, kiss_vk::Sequence&);

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
  std::shared_ptr<kiss_vk::Sequence> default_seq;  // dispatch_multi_stage without a Sequence
  std::unordered_map<std::string, std::shared_ptr<kiss_vk::Algorithm>> cached_algorithms;
};

}  // namespace cifar_sparse::vulkan
