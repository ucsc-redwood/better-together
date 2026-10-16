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
  // The memory resource of a buffer class (runtime/placement.hpp, design B2).
  kiss_vk::VulkanMemoryResource::memory_resource* get_mr(const bt::BufferClass c) {
    return engine.get_mr(c);
  }

  // Sequence handle, used by benchmarks to read device-side GPU time per stage.
  kiss_vk::Sequence* get_seq() { return default_seq.get(); }

  // A Sequence of this dispatcher's engine: one per GPU chunk of a schedule, so chunks on
  // different worker threads never share a command buffer (design C2).
  [[nodiscard]] std::shared_ptr<kiss_vk::Sequence> make_seq() { return engine.make_seq(); }

  // Per-stage single-submit path (cmd_begin -> record -> submit -> wait).
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

  // Record stages [start_stage, end_stage] into ONE command buffer (the Sequence orders
  // dependent dispatches, design A3), then run it with ONE submit + ONE fence wait. Collapses the
  // per-stage CPU<->GPU round-trips of a chunk; host cache flush/invalidate happen once per chunk
  // (inside submit/wait). Safe across tree's data-dependent stages: VkAppData_Safe's
  // counts (n_unique, n_brt_nodes) are const, fixed at construction -- they don't need a GPU
  // read-back between stages, so a stage can be recorded before the prior stage executes.
  void dispatch_multi_stage(VkAppData_Safe& data, const int start_stage, const int end_stage) {
    dispatch_multi_stage(data, start_stage, end_stage, *default_seq);
  }
  void dispatch_multi_stage(VkAppData_Safe& data,
                            const int start_stage,
                            const int end_stage,
                            kiss_vk::Sequence& seq) {
    complete_multi_stage(
        data, start_stage, end_stage, seq, submit_multi_stage(data, start_stage, end_stage, seq));
  }

  // The two halves of dispatch_multi_stage, so a GPU chunk can keep several items in flight
  // (design D3): submit records (or reuses) the chunk's command buffer and submits it,
  // returning its ticket; complete waits for that ticket and makes the results host-visible.
  uint64_t submit_multi_stage(VkAppData_Safe& data,
                              const int start_stage,
                              const int end_stage,
                              kiss_vk::Sequence& seq) {
    if (start_stage < 1 || end_stage > 7 || start_stage > end_stage)
      throw std::out_of_range("Invalid stage");

    seq.cmd_begin();
    for (int stage = start_stage; stage <= end_stage; ++stage) {
      (this->*record_functions[stage - 1])(data, seq);
    }
    seq.cmd_end();
    return seq.submit();
  }
  void complete_multi_stage(VkAppData_Safe& /*data*/,
                            int /*start_stage*/,
                            int /*end_stage*/,
                            kiss_vk::Sequence& seq,
                            const uint64_t ticket) {
    seq.wait(ticket);
  }

 private:
  // Record-only stage bodies: bind (buffers captured per dispatch) + dispatch(es) into
  // `cmd` (= seq's command buffer), keeping each stage's own intra-stage barriers, no
  // cmd_begin/submit/wait. The scan algos are shared by stage 3 and stage 6, which may
  // share one command buffer.
  void record_stage_1(VkAppData_Safe& appdata, kiss_vk::Sequence& seq);
  void record_stage_2(VkAppData_Safe& appdata, kiss_vk::Sequence& seq);
  void record_stage_3(VkAppData_Safe& appdata, kiss_vk::Sequence& seq);
  void record_stage_4(VkAppData_Safe& appdata, kiss_vk::Sequence& seq);
  void record_stage_5(VkAppData_Safe& appdata, kiss_vk::Sequence& seq);
  void record_stage_6(VkAppData_Safe& appdata, kiss_vk::Sequence& seq);
  void record_stage_7(VkAppData_Safe& appdata, kiss_vk::Sequence& seq);

  using RecordFn = void (VulkanDispatcher::*)(VkAppData_Safe&, kiss_vk::Sequence&);

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
  // Shared by stage 3 (flag scan) and stage 6 (edge-count scan), and by both
  // the VkAppData_Safe and VkAppData (chained) record bodies below -- this
  // helper only takes buffer infos, not an AppData type, so it needs no
  // chained-specific overload.
  void record_device_scan(kiss_vk::Sequence& seq,
                          vk::DescriptorBufferInfo src,
                          vk::DescriptorBufferInfo dst,
                          vk::DescriptorBufferInfo block_sums,
                          uint32_t n);

 public:
  // --------------------------------------------------------------------------
  // Genuinely-chained overloads (tree::vulkan::VkAppData, no golden/_out split)
  // -- added alongside the VkAppData_Safe overloads above, which stay
  // unmodified. See apps/tree/tree_diff_oracle.hpp's Phase 3 note and this
  // feature's research.md for why dispatch_multi_stage's chained overload
  // needs internal host-readback synchronization that the VkAppData_Safe
  // overload never did (its counts are const, known from the golden).
  // --------------------------------------------------------------------------

  void run_stage_1(VkAppData& appdata);
  void run_stage_2(VkAppData& appdata);
  void run_stage_3(VkAppData& appdata);
  void run_stage_4(VkAppData& appdata);
  void run_stage_5(VkAppData& appdata);
  void run_stage_6(VkAppData& appdata);
  void run_stage_7(VkAppData& appdata);

  void dispatch_stage(VkAppData& data, const int stage) {
    dispatch_multi_stage(data, stage, stage);
  }

  void dispatch_multi_stage(VkAppData& data, const int start_stage, const int end_stage) {
    dispatch_multi_stage(data, start_stage, end_stage, *default_seq);
  }
  void dispatch_multi_stage(VkAppData& data,
                            const int start_stage,
                            const int end_stage,
                            kiss_vk::Sequence& seq) {
    complete_multi_stage(
        data, start_stage, end_stage, seq, submit_multi_stage(data, start_stage, end_stage, seq));
  }
  // See the VkAppData_Safe overloads. A range crossing the stage 3|4 boundary submits and
  // waits for its stages 1-3 inside submit (stage 4's dispatch size needs n_unique), so
  // only its second batch is in flight; complete reads back the counts the range produced.
  uint64_t submit_multi_stage(VkAppData& data,
                              int start_stage,
                              int end_stage,
                              kiss_vk::Sequence& seq);
  void complete_multi_stage(
      VkAppData& data, int start_stage, int end_stage, kiss_vk::Sequence& seq, uint64_t ticket);

 private:
  void record_stage_1(VkAppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_2(VkAppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_3(VkAppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_4(VkAppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_5(VkAppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_6(VkAppData& appdata, kiss_vk::Sequence& seq);
  void record_stage_7(VkAppData& appdata, kiss_vk::Sequence& seq);

  using RecordFnChained = void (VulkanDispatcher::*)(VkAppData&, kiss_vk::Sequence&);

  static constexpr std::array<RecordFnChained, 7> record_functions_chained = {
      &VulkanDispatcher::record_stage_1,
      &VulkanDispatcher::record_stage_2,
      &VulkanDispatcher::record_stage_3,
      &VulkanDispatcher::record_stage_4,
      &VulkanDispatcher::record_stage_5,
      &VulkanDispatcher::record_stage_6,
      &VulkanDispatcher::record_stage_7,
  };

  kiss_vk::Engine engine;
  std::shared_ptr<kiss_vk::Sequence> default_seq;  // dispatch_multi_stage without a Sequence
  std::unordered_map<std::string, std::shared_ptr<kiss_vk::Algorithm>> cached_algorithms;

  // --------------------------------------------------------------------------
  // Temporary storages
  // --------------------------------------------------------------------------

  // (n + 255) / 256;
  // std::pmr::vector<uint32_t> tmp_u_sums;
  // std::pmr::vector<uint32_t> tmp_u_prefix_sums;
};

}  // namespace tree::vulkan
