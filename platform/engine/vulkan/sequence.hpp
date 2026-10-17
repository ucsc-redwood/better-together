#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <span>
#include <vector>

#include "algorithm.hpp"

namespace kiss_vk {

// One (device tick, CLOCK_MONOTONIC ns) pair sampled together by
// VK_KHR/EXT_calibrated_timestamps; false when the device cannot (design D1).
using CalibrateFn =
    std::function<bool(uint64_t& device_ticks, uint64_t& host_ns, uint64_t& deviation_ns)>;

// Where the time of one submission went, all in ns (design D1; submission spec "Per-chunk
// overhead is measured by phase"). The host phases are contiguous, so record + between +
// flush + submit + wait + invalidate == wall. With calibrated timestamps submit + wait split
// further: start_latency (submit call began -> GPU start; the GPU may start before the call
// returns) + gpu + wake (GPU end -> wait returned) == submit + wait.
struct PhaseTimes {
  double record_ns = 0;      // cmd_begin() .. end of cmd_end(): capture (+ recording if any)
  double between_ns = 0;     // end of cmd_end() .. submit()
  double flush_ns = 0;       // host-cache flush of the touched buffers
  double submit_ns = 0;      // vkQueueSubmit2, lock included
  double wait_ns = 0;        // submit returned .. semaphore wait returned
  double invalidate_ns = 0;  // host-cache invalidate + timestamp read-back
  double wall_ns = 0;        // cmd_begin() .. invalidate done
  double gpu_ns = 0;         // GPU start .. end timestamp (0 without timestamps)
  bool calibrated = false;   // start_latency/wake are valid (host and GPU domains correlated)
  double start_latency_ns = 0;
  double wake_ns = 0;
  double calibration_deviation_ns = 0;  // the driver's bound on the calibration pair's error
  bool recorded = false;  // this submission recorded its command buffer (no cached reuse)
  // What the flush and the invalidate walked (whole allocations of the bound buffers), and
  // how the chunk's shaders use them. A buffer no shader writes needs no invalidate (the
  // host copy is still current); a buffer no shader reads needs no flush for the GPU's sake.
  uint64_t touched_bytes = 0;
  uint64_t gpu_written_bytes = 0;     // bound read-write or write-only by some dispatch
  uint64_t gpu_read_only_bytes = 0;   // bound only for reading
  uint64_t gpu_write_only_bytes = 0;  // bound only for writing
};

class Sequence {
 public:
  // push_descriptor_fn: BaseEngine::push_descriptor_fn(); null selects the transient
  // descriptor-pool path, which re-records every submission (design D2).
  explicit Sequence(vk::Device device_ref,
                    vk::Queue compute_queue_ref,
                    uint32_t compute_queue_index,
                    float timestamp_period_ns = 0.0f,
                    uint32_t timestamp_valid_bits = 0,
                    std::vector<VulkanMemoryResource*> resources = {},
                    std::mutex* submit_mutex = nullptr,
                    PFN_vkCmdPushDescriptorSetKHR push_descriptor_fn = nullptr,
                    CalibrateFn calibrate = {});

  // Waits for any submission still in flight, then frees what this Sequence created.
  ~Sequence();

  // Owns command buffers, a timeline semaphore, a query pool and descriptor pools, freed in
  // ~Sequence; only ever held behind a shared_ptr (Engine::make_seq). Copying would
  // double-free those handles and there is no move, so make every copy/move ill-formed.
  Sequence(const Sequence&) = delete;
  Sequence& operator=(const Sequence&) = delete;
  Sequence(Sequence&&) = delete;
  Sequence& operator=(Sequence&&) = delete;

  // ---- Recording -------------------------------------------------------------------
  // Between cmd_begin() and cmd_end() the Algorithms describe one submission's work
  // (bind/push_constants/dispatch). cmd_end() then reuses the command buffer already
  // recorded for exactly that work, or records one (design D2): for a fixed chunk and
  // pooled item the buffers and push constants repeat, so the steady state records nothing.
  void cmd_begin();
  void cmd_end();

  // Called by Algorithm::record_bind: bind `pipeline` and these buffers for the next
  // dispatch. Orders it after the earlier work it depends on (design A3): an inter-stage
  // dependency is added when it reads or writes a buffer written since the last one, or
  // writes one read since it. BT_VK_BARRIER_ALL=1 adds one before every dispatch but the
  // first, as before conditional barriers.
  void bind(vk::Pipeline pipeline,
            vk::PipelineLayout layout,
            vk::DescriptorSetLayout set_layout,
            std::span<const vk::DescriptorBufferInfo> buffers,
            std::span<const BufferAccess> access);
  void push_constants(vk::PipelineLayout layout, const void* data, size_t size);
  void dispatch(std::array<uint32_t, 3> grid);

  // An unconditional inter-stage dependency (design A2): COMPUTE_SHADER/SHADER_STORAGE_WRITE
  // -> COMPUTE_SHADER/SHADER_STORAGE_READ|WRITE, a global synchronization2 memory barrier.
  void cmd_stage_dependency();

  // The buffers bound since the last cmd_begin(): what submit() flushes and the wait
  // invalidates.
  [[nodiscard]] const std::vector<VkBuffer>& touched_buffers() const { return touched_; }

  // Dispatches and inter-stage dependencies of the submission described last.
  [[nodiscard]] uint32_t dispatch_count() const { return dispatches_; }
  [[nodiscard]] uint32_t barrier_count() const { return barriers_; }
  // Command buffers recorded so far (each cached reuse leaves it unchanged), and whether
  // this Sequence reuses them at all (push descriptors; the transient-pool path re-records
  // every submission).
  [[nodiscard]] uint64_t recording_count() const { return recordings_; }
  [[nodiscard]] bool reuses_recordings() const { return push_descriptor_fn_ != nullptr; }

  // ---- Submission ------------------------------------------------------------------
  // Submits the work described since cmd_begin() and returns its ticket, the value the
  // Sequence's timeline semaphore reaches when it completes (design D3). Several may be in
  // flight; wait() on each, in any order.
  uint64_t submit();
  // Blocks until `ticket` has completed, then makes its writes visible to the host.
  void wait(uint64_t ticket);
  // wait() on the most recent submit().
  void wait();

  // GPU-side elapsed time of the submission waited on last, in nanoseconds. Returns 0.0 if
  // device timestamps are unsupported (timestamp_valid_bits == 0), in which case callers
  // should fall back to wall-clock timing.
  [[nodiscard]] double get_last_gpu_time_ns() const { return last_gpu_ns_; }
  [[nodiscard]] bool gpu_timestamps_supported() const { return timestamp_valid_bits_ != 0; }

  // Phase timing of the submission waited on last; filled only with BT_VK_PHASES=1.
  [[nodiscard]] bool phases_enabled() const { return phases_enabled_; }
  [[nodiscard]] const PhaseTimes& last_phases() const { return last_phases_; }

 private:
  struct Op {
    enum class Kind : std::uint8_t { kBind, kPush, kDispatch, kBarrier } kind;
    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
    vk::DescriptorSetLayout set_layout;
    uint32_t first = 0;  // into buffers_ (kBind) or bytes_ (kPush)
    uint32_t count = 0;
    std::array<uint32_t, 3> grid{};
    bool operator==(const Op&) const = default;
  };
  // The work of one submission: what cmd_end() matches recorded command buffers by.
  struct Recipe {
    std::vector<Op> ops;
    std::vector<vk::DescriptorBufferInfo> buffers;
    std::vector<std::byte> bytes;
    bool operator==(const Recipe&) const = default;
  };
  // A command buffer and what it was recorded for.
  struct Slot {
    vk::CommandBuffer cmd;
    Recipe recipe;
    size_t hash = 0;
    uint64_t generation = 0;  // VulkanMemoryResource::generation() when recorded
    bool valid = false;
    bool in_flight = false;
    uint64_t last_use = 0;
    std::vector<vk::DescriptorPool> pools;  // transient-pool path only
    size_t current_pool = 0;
  };
  struct Pending {
    uint64_t ticket = 0;
    size_t slot = 0;
    std::vector<VmaAllocation> touched;
    uint64_t t_begin = 0, t_cmd_end = 0, t_submit = 0, t_flushed = 0, t_submitted = 0;
    bool recorded = false;
    uint64_t touched_bytes = 0, gpu_written_bytes = 0, gpu_read_only_bytes = 0,
             gpu_write_only_bytes = 0;  // PhaseTimes' byte counts, filled with BT_VK_PHASES=1
  };

  [[nodiscard]] size_t acquire_slot();
  void record(Slot& slot, uint32_t slot_index);
  [[nodiscard]] vk::DescriptorSet allocate_descriptor_set(Slot& slot,
                                                          vk::DescriptorSetLayout layout);
  [[nodiscard]] vk::DescriptorPool create_descriptor_pool() const;
  [[nodiscard]] std::vector<VmaAllocation> touched_allocations() const;
  [[nodiscard]] uint64_t mask(uint64_t ticks) const;

  vk::Device device_ref_;
  vk::Queue compute_queue_ref_;
  uint32_t compute_queue_index_;
  PFN_vkCmdPushDescriptorSetKHR push_descriptor_fn_;
  CalibrateFn calibrate_;

  vk::CommandPool command_pool_;
  vk::Semaphore timeline_;
  uint64_t next_ticket_ = 0;

  // Up to kCachedSlots recorded command buffers are kept (one per chunk x pooled item in
  // steady state: the runtime pools hold <= 16 items); past that the least recently used
  // slot not in flight is re-recorded. A slot in flight is never re-recorded.
  static constexpr uint32_t kCachedSlots = 32;
  static constexpr uint32_t kMaxSlots = 64;
  std::vector<Slot> slots_;
  uint64_t use_clock_ = 0;
  uint64_t recordings_ = 0;

  // The submission being described (cmd_begin() .. submit()).
  Recipe recipe_;
  size_t recipe_slot_ = 0;
  bool recipe_recorded_ = false;
  uint64_t t_begin_ = 0, t_cmd_end_ = 0;
  std::deque<Pending> pending_;

  // Dependency tracking for the submission being described (bind()).
  bool barrier_all_ = false;
  std::vector<VkBuffer> written_since_barrier_;
  std::vector<VkBuffer> read_since_barrier_;
  uint32_t dispatches_ = 0;
  uint32_t barriers_ = 0;

  // Device-side timestamps: 2 queries per slot (start and end of the command buffer's
  // compute work, both written at COMPUTE_SHADER).
  vk::QueryPool query_pool_;
  float timestamp_period_ns_;      // nanoseconds per timestamp tick
  uint32_t timestamp_valid_bits_;  // 0 => timestamps unsupported on this queue
  double last_gpu_ns_ = 0;

  bool phases_enabled_ = false;
  PhaseTimes last_phases_;

  // Host<->device cache maintenance on non-coherent memory: the buffers bound since
  // cmd_begin() are flushed before submit() and invalidated after its wait, each through
  // whichever of the engine's resources owns it. Kept per Sequence (and per pending
  // submission), so concurrently recording chunks never share it (runtime/gpu-chunks spec).
  // No resources => no cache maintenance (memory is coherent or caller manages it).
  std::vector<VulkanMemoryResource*> resources_;
  std::vector<VkBuffer> touched_;
  std::vector<uint8_t> touched_access_;  // per touched_ buffer: OR of the BufferAccess bits

  // The Engine's: the queue is externally synchronized, and every Sequence of one Engine
  // submits to the same queue, possibly from different threads (design C2).
  std::mutex* submit_mutex_;
};

}  // namespace kiss_vk
