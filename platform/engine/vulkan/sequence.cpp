#include "sequence.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <string_view>
#include <vulkan/vulkan.hpp>

namespace kiss_vk {

namespace {
uint64_t now_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}

bool env_is_1(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && std::string_view(v) == "1";
}

void hash_combine(size_t& h, const size_t v) {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
}

template <typename Handle>
size_t handle_hash(const Handle handle) {
  return std::hash<uint64_t>{}(
      reinterpret_cast<uint64_t>(static_cast<typename Handle::CType>(handle)));
}
}  // namespace

Sequence::Sequence(const vk::Device device_ref,
                   const vk::Queue compute_queue_ref,
                   const uint32_t compute_queue_index,
                   const float timestamp_period_ns,
                   const uint32_t timestamp_valid_bits,
                   std::vector<VulkanMemoryResource*> resources,
                   std::mutex* submit_mutex,
                   const PFN_vkCmdPushDescriptorSetKHR push_descriptor_fn,
                   CalibrateFn calibrate)
    : device_ref_(device_ref),
      compute_queue_ref_(compute_queue_ref),
      compute_queue_index_(compute_queue_index),
      push_descriptor_fn_(push_descriptor_fn),
      calibrate_(std::move(calibrate)),
      timestamp_period_ns_(timestamp_period_ns),
      timestamp_valid_bits_(timestamp_valid_bits),
      resources_(std::move(resources)),
      submit_mutex_(submit_mutex) {
  spdlog::trace("Sequence constructor");

  barrier_all_ = env_is_1("BT_VK_BARRIER_ALL");
  phases_enabled_ = env_is_1("BT_VK_PHASES");

  command_pool_ = device_ref_.createCommandPool(vk::CommandPoolCreateInfo{
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = compute_queue_index_,
  });

  // The ticket of a submission is the value this semaphore reaches when it completes.
  const vk::SemaphoreTypeCreateInfo timeline_info{
      .semaphoreType = vk::SemaphoreType::eTimeline,
      .initialValue = 0,
  };
  timeline_ = device_ref_.createSemaphore(vk::SemaphoreCreateInfo{.pNext = &timeline_info});

  if (timestamp_valid_bits_ != 0) {
    query_pool_ = device_ref_.createQueryPool(vk::QueryPoolCreateInfo{
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = 2 * kMaxSlots,
    });
  }
}

Sequence::~Sequence() {
  // A command buffer still executing must not be freed: wait for every submission first.
  // Then destroy every child object this Sequence created, before BaseEngine destroys the
  // device.
  if (next_ticket_ != 0) {
    try {
      const vk::SemaphoreWaitInfo wait_info{
          .semaphoreCount = 1,
          .pSemaphores = &timeline_,
          .pValues = &next_ticket_,
      };
      (void)device_ref_.waitSemaphores(wait_info, UINT64_MAX);
    } catch (const std::exception& e) {
      spdlog::warn("Sequence teardown: waiting for in-flight submissions failed: {}", e.what());
    }
  }
  if (query_pool_) device_ref_.destroyQueryPool(query_pool_);
  for (const auto& slot : slots_) {
    for (const auto pool : slot.pools) device_ref_.destroyDescriptorPool(pool);  // + its sets
  }
  if (command_pool_) device_ref_.destroyCommandPool(command_pool_);  // + its command buffers
  if (timeline_) device_ref_.destroySemaphore(timeline_);
}

// ----------------------------------------------------------------------------
// Describing a submission
// ----------------------------------------------------------------------------

void Sequence::cmd_begin() {
  spdlog::trace("Sequence::cmd_begin()");
  t_begin_ = phases_enabled_ ? now_ns() : 0;
  recipe_.ops.clear();
  recipe_.buffers.clear();
  recipe_.bytes.clear();
  // A fresh touched-buffer set: the stages described until submit() bind exactly the
  // buffers this task uses, so submit/wait flush and invalidate only those.
  touched_.clear();
  written_since_barrier_.clear();
  read_since_barrier_.clear();
  dispatches_ = 0;
  barriers_ = 0;
}

void Sequence::bind(const vk::Pipeline pipeline,
                    const vk::PipelineLayout layout,
                    const vk::DescriptorSetLayout set_layout,
                    const std::span<const vk::DescriptorBufferInfo> buffers,
                    const std::span<const BufferAccess> access) {
  const auto contains = [](const std::vector<VkBuffer>& set, VkBuffer b) {
    return std::ranges::find(set, b) != set.end();
  };
  const auto reads = [&](size_t i) { return (static_cast<int>(access[i]) & 1) != 0; };
  const auto writes = [&](size_t i) { return (static_cast<int>(access[i]) & 2) != 0; };

  bool needed = barrier_all_ && dispatches_ != 0;
  for (size_t i = 0; i < buffers.size() && !needed; ++i) {
    VkBuffer b = buffers[i].buffer;
    needed = contains(written_since_barrier_, b) || (writes(i) && contains(read_since_barrier_, b));
  }
  if (needed) cmd_stage_dependency();

  for (size_t i = 0; i < buffers.size(); ++i) {
    VkBuffer b = buffers[i].buffer;
    if (writes(i) && !contains(written_since_barrier_, b)) written_since_barrier_.push_back(b);
    if (reads(i) && !contains(read_since_barrier_, b)) read_since_barrier_.push_back(b);
    if (!contains(touched_, b)) touched_.push_back(b);
  }
  ++dispatches_;

  recipe_.ops.push_back(Op{
      .kind = Op::Kind::kBind,
      .pipeline = pipeline,
      .layout = layout,
      .set_layout = set_layout,
      .first = static_cast<uint32_t>(recipe_.buffers.size()),
      .count = static_cast<uint32_t>(buffers.size()),
  });
  recipe_.buffers.insert(recipe_.buffers.end(), buffers.begin(), buffers.end());
}

void Sequence::push_constants(const vk::PipelineLayout layout,
                              const void* data,
                              const size_t size) {
  recipe_.ops.push_back(Op{
      .kind = Op::Kind::kPush,
      .layout = layout,
      .first = static_cast<uint32_t>(recipe_.bytes.size()),
      .count = static_cast<uint32_t>(size),
  });
  const auto* bytes = static_cast<const std::byte*>(data);
  recipe_.bytes.insert(recipe_.bytes.end(), bytes, bytes + size);
}

void Sequence::dispatch(const std::array<uint32_t, 3> grid) {
  recipe_.ops.push_back(Op{.kind = Op::Kind::kDispatch, .grid = grid});
}

void Sequence::cmd_stage_dependency() {
  recipe_.ops.push_back(Op{.kind = Op::Kind::kBarrier});
  written_since_barrier_.clear();
  read_since_barrier_.clear();
  ++barriers_;
}

void Sequence::cmd_end() {
  spdlog::trace("Sequence::cmd_end()");
  spdlog::debug("Sequence: command buffer with {} dispatches, {} inter-stage barriers ({})",
                dispatches_,
                barriers_,
                barrier_all_ ? "BT_VK_BARRIER_ALL=1" : "tracked");

  size_t hash = recipe_.ops.size();
  for (const Op& op : recipe_.ops) {
    hash_combine(hash, static_cast<size_t>(op.kind));
    hash_combine(hash, handle_hash(op.pipeline));
    hash_combine(hash, op.grid[0] ^ (size_t{op.grid[1]} << 21) ^ (size_t{op.grid[2]} << 42));
  }
  for (const auto& b : recipe_.buffers) {
    hash_combine(hash, handle_hash(b.buffer));
    hash_combine(hash, b.offset ^ (b.range << 1));
  }
  for (const std::byte v : recipe_.bytes) hash_combine(hash, static_cast<size_t>(v));

  // Reuse the command buffer recorded for exactly this work (design D2). Only with push
  // descriptors: the transient-pool path binds sets that the next recording of the same
  // slot resets, so it re-records every submission. A buffer freed since recording may
  // have handed its handle to a new buffer, so any deallocation invalidates the cache.
  const uint64_t generation = VulkanMemoryResource::generation();
  recipe_recorded_ = false;
  if (push_descriptor_fn_ != nullptr) {
    for (size_t i = 0; i < slots_.size(); ++i) {
      Slot& s = slots_[i];
      if (s.valid && !s.in_flight && s.hash == hash && s.generation == generation &&
          s.recipe == recipe_) {
        recipe_slot_ = i;
        s.last_use = ++use_clock_;
        t_cmd_end_ = phases_enabled_ ? now_ns() : 0;
        return;
      }
    }
  }

  recipe_slot_ = acquire_slot();
  Slot& slot = slots_[recipe_slot_];
  slot.recipe = recipe_;
  slot.hash = hash;
  slot.generation = generation;
  slot.last_use = ++use_clock_;
  record(slot, static_cast<uint32_t>(recipe_slot_));
  slot.valid = true;
  recipe_recorded_ = true;
  t_cmd_end_ = phases_enabled_ ? now_ns() : 0;
}

size_t Sequence::acquire_slot() {
  // A new slot while the cache has room; then the least recently used slot not in flight;
  // a new slot again only when every slot is in flight.
  size_t best = slots_.size();
  if (slots_.size() >= kCachedSlots) {
    for (size_t i = 0; i < slots_.size(); ++i) {
      if (slots_[i].in_flight) continue;
      if (best == slots_.size() || slots_[i].last_use < slots_[best].last_use) best = i;
    }
  }
  if (best != slots_.size()) return best;
  if (slots_.size() == kMaxSlots) {
    throw std::runtime_error("Sequence: more than " + std::to_string(kMaxSlots) +
                             " submissions in flight");
  }
  Slot slot;
  slot.cmd = device_ref_
                 .allocateCommandBuffers(vk::CommandBufferAllocateInfo{
                     .commandPool = command_pool_,
                     .level = vk::CommandBufferLevel::ePrimary,
                     .commandBufferCount = 1,
                 })
                 .front();
  slots_.push_back(std::move(slot));
  return slots_.size() - 1;
}

void Sequence::record(Slot& slot, const uint32_t slot_index) {
  ++recordings_;
  // The previous recording's sets (transient-pool path) are no longer referenced: the slot
  // is not in flight.
  for (const auto pool : slot.pools) device_ref_.resetDescriptorPool(pool);
  slot.current_pool = 0;

  const vk::CommandBuffer cmd = slot.cmd;
  cmd.begin(vk::CommandBufferBeginInfo{});  // resubmittable: no ONE_TIME_SUBMIT

  // Start timestamp at COMPUTE_SHADER (design A4): written once every earlier command has
  // finished its compute-shader stage, i.e. when this command buffer's compute work can
  // start. The reset is recorded too, so the command buffer can be resubmitted as is.
  const uint32_t q0 = 2 * slot_index;
  if (timestamp_valid_bits_ != 0) {
    cmd.resetQueryPool(query_pool_, q0, 2);
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, query_pool_, q0);
  }

  for (const Op& op : slot.recipe.ops) {
    switch (op.kind) {
      case Op::Kind::kBind: {
        std::array<vk::WriteDescriptorSet, kMaxBindingBuffers> writes;
        // Push descriptors ignore dstSet. The fallback takes a set from the slot's pools.
        const vk::DescriptorSet set =
            push_descriptor_fn_ ? nullptr : allocate_descriptor_set(slot, op.set_layout);
        for (uint32_t b = 0; b < op.count; ++b) {
          writes[b] = vk::WriteDescriptorSet{
              .dstSet = set,
              .dstBinding = b,
              .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eStorageBuffer,
              .pBufferInfo = &slot.recipe.buffers[op.first + b],
          };
        }
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, op.pipeline);
        if (push_descriptor_fn_) {
          push_descriptor_fn_(cmd,
                              VK_PIPELINE_BIND_POINT_COMPUTE,
                              op.layout,
                              0,
                              op.count,
                              reinterpret_cast<const VkWriteDescriptorSet*>(writes.data()));
        } else {
          device_ref_.updateDescriptorSets(op.count, writes.data(), 0, nullptr);
          cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, op.layout, 0, set, {});
        }
        break;
      }
      case Op::Kind::kPush:
        cmd.pushConstants(op.layout,
                          vk::ShaderStageFlagBits::eCompute,
                          0,
                          op.count,
                          slot.recipe.bytes.data() + op.first);
        break;
      case Op::Kind::kDispatch:
        cmd.dispatch(op.grid[0], op.grid[1], op.grid[2]);
        break;
      case Op::Kind::kBarrier: {
        // Global, not per-buffer: buffer barriers buy nothing on these drivers (design A3).
        const vk::MemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask =
                vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
        };
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
        break;
      }
    }
  }

  // Make every compute-shader write in this command buffer available to the host domain.
  // A completion wait alone only guarantees completion, not host visibility (Vulkan spec,
  // Synchronization: "Signaling a fence and waiting on the host does not guarantee that the
  // results of memory accesses will be visible to the host"). The invalidate after the wait
  // still handles the CPU-cache side on HOST_CACHED memory. Recorded before the closing
  // timestamp so GPU time includes it.
  const vk::MemoryBarrier2 host_barrier{
      .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
      .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
      .dstStageMask = vk::PipelineStageFlagBits2::eHost,
      .dstAccessMask = vk::AccessFlagBits2::eHostRead,
  };
  cmd.pipelineBarrier2(
      vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &host_barrier});

  // End timestamp: written once every dispatch has completed COMPUTE_SHADER (design A4).
  if (timestamp_valid_bits_ != 0) {
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, query_pool_, q0 + 1);
  }
  cmd.end();
}

vk::DescriptorPool Sequence::create_descriptor_pool() const {
  // Sized so a whole chunk fits one pool (the longest, tree stages 1-7, binds 20 sets of
  // <= 13 buffers); a bigger chunk just opens another pool.
  constexpr uint32_t kSetsPerPool = 64;
  const vk::DescriptorPoolSize pool_size{
      .type = vk::DescriptorType::eStorageBuffer,
      .descriptorCount = kSetsPerPool * kMaxBindingBuffers,
  };
  return device_ref_.createDescriptorPool(vk::DescriptorPoolCreateInfo{
      .maxSets = kSetsPerPool,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
  });
}

vk::DescriptorSet Sequence::allocate_descriptor_set(Slot& slot,
                                                    const vk::DescriptorSetLayout layout) {
  for (;; ++slot.current_pool) {
    const bool fresh = slot.current_pool == slot.pools.size();
    if (fresh) slot.pools.push_back(create_descriptor_pool());

    const vk::DescriptorSetAllocateInfo allocate_info{
        .descriptorPool = slot.pools[slot.current_pool],
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };
    vk::DescriptorSet set;
    const vk::Result result = device_ref_.allocateDescriptorSets(&allocate_info, &set);
    if (result == vk::Result::eSuccess) return set;
    // A full pool moves on to the next one; anything else (or a fresh pool that cannot
    // hold one set) is a real error.
    if (fresh || (result != vk::Result::eErrorOutOfPoolMemory &&
                  result != vk::Result::eErrorFragmentedPool)) {
      throw std::runtime_error("vkAllocateDescriptorSets failed: " + vk::to_string(result));
    }
  }
}

// ----------------------------------------------------------------------------
// Submitting and waiting
// ----------------------------------------------------------------------------

uint64_t Sequence::submit() {
  spdlog::trace("Sequence::submit()");
  Pending p{
      .ticket = ++next_ticket_,
      .slot = recipe_slot_,
      .touched = touched_allocations(),
      .t_begin = t_begin_,
      .t_cmd_end = t_cmd_end_,
      .t_submit = phases_enabled_ ? now_ns() : 0,
      .recorded = recipe_recorded_,
  };

  // Make any pending host writes (inputs/weights) visible to the GPU before it runs.
  // No-op on coherent memory; required on HOST_CACHED (see do_allocate). Scoped to the
  // buffers this chunk bound -- not the whole pool.
  for (VmaAllocation a : p.touched) vmaFlushAllocation(g_vma_allocator, a, 0, VK_WHOLE_SIZE);
  p.t_flushed = phases_enabled_ ? now_ns() : 0;

  const vk::CommandBufferSubmitInfo cmd_info{.commandBuffer = slots_[p.slot].cmd};
  const vk::SemaphoreSubmitInfo signal_info{
      .semaphore = timeline_,
      .value = p.ticket,
      .stageMask = vk::PipelineStageFlagBits2::eAllCommands,
  };
  const vk::SubmitInfo2 submit_info{
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &cmd_info,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &signal_info,
  };
  {
    std::unique_lock<std::mutex> lock;
    if (submit_mutex_) lock = std::unique_lock(*submit_mutex_);
    compute_queue_ref_.submit2(submit_info);
  }
  p.t_submitted = phases_enabled_ ? now_ns() : 0;
  slots_[p.slot].in_flight = true;
  pending_.push_back(std::move(p));
  return pending_.back().ticket;
}

void Sequence::wait() {
  if (pending_.empty()) throw std::runtime_error("Sequence::wait(): nothing submitted");
  wait(pending_.back().ticket);
}

void Sequence::wait(const uint64_t ticket) {
  spdlog::trace("Sequence::wait({})", ticket);
  const auto it = std::ranges::find(pending_, ticket, &Pending::ticket);
  if (it == pending_.end()) throw std::runtime_error("Sequence::wait(): unknown ticket");

  const vk::SemaphoreWaitInfo wait_info{
      .semaphoreCount = 1,
      .pSemaphores = &timeline_,
      .pValues = &ticket,
  };
  if (const vk::Result result = device_ref_.waitSemaphores(wait_info, UINT64_MAX);
      result != vk::Result::eSuccess) {
    spdlog::error("vkWaitSemaphores failed with error: {}", vk::to_string(result));
    throw std::runtime_error("Failed to sync sequence");
  }
  const uint64_t t_waited = phases_enabled_ ? now_ns() : 0;
  uint64_t cal_device = 0;
  uint64_t cal_host = 0;
  uint64_t cal_deviation = 0;
  const bool calibrated =
      phases_enabled_ && calibrate_ && calibrate_(cal_device, cal_host, cal_deviation);

  // GPU work is done: invalidate host caches so subsequent CPU reads see the freshly
  // written results. No-op on coherent memory; required on HOST_CACHED (this is the cache
  // maintenance BUGS-FOUND.md §7 originally lacked). Scoped to the buffers this chunk bound.
  for (VmaAllocation a : it->touched) {
    vmaInvalidateAllocation(g_vma_allocator, a, 0, VK_WHOLE_SIZE);
  }

  std::array<uint64_t, 2> ts{};
  if (timestamp_valid_bits_ != 0) {
    const vk::Result result = device_ref_.getQueryPoolResults(
        query_pool_,
        static_cast<uint32_t>(2 * it->slot),
        2,
        sizeof(ts),
        ts.data(),
        sizeof(uint64_t),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (result != vk::Result::eSuccess) {
      spdlog::warn("getQueryPoolResults returned {}", vk::to_string(result));
      ts = {};
    }
  }
  // Mask off invalid high bits before subtracting.
  last_gpu_ns_ = static_cast<double>(mask(mask(ts[1]) - mask(ts[0]))) * timestamp_period_ns_;

  if (phases_enabled_) {
    const uint64_t t_done = now_ns();
    PhaseTimes ph{
        .record_ns = static_cast<double>(it->t_cmd_end - it->t_begin),
        .between_ns = static_cast<double>(it->t_submit - it->t_cmd_end),
        .flush_ns = static_cast<double>(it->t_flushed - it->t_submit),
        .submit_ns = static_cast<double>(it->t_submitted - it->t_flushed),
        .wait_ns = static_cast<double>(t_waited - it->t_submitted),
        .invalidate_ns = static_cast<double>(t_done - t_waited),
        .wall_ns = static_cast<double>(t_done - it->t_begin),
        .gpu_ns = last_gpu_ns_,
        .calibrated = calibrated && timestamp_valid_bits_ != 0,
        .recorded = it->recorded,
    };
    if (ph.calibrated) {
      // Device ticks -> CLOCK_MONOTONIC ns through the calibration pair, the tick difference
      // taken modulo timestampValidBits and sign-extended (the timestamps precede the pair).
      const unsigned shift = 64 - timestamp_valid_bits_;
      const auto host_of = [&](const uint64_t ticks) {
        const auto d = static_cast<int64_t>((ticks - cal_device) << shift) >> shift;
        return static_cast<double>(cal_host) + static_cast<double>(d) * timestamp_period_ns_;
      };
      ph.start_latency_ns = host_of(ts[0]) - static_cast<double>(it->t_flushed);
      ph.wake_ns = static_cast<double>(t_waited) - host_of(ts[1]);
      ph.calibration_deviation_ns = static_cast<double>(cal_deviation);
    }
    last_phases_ = ph;
  }

  slots_[it->slot].in_flight = false;
  pending_.erase(it);
}

uint64_t Sequence::mask(const uint64_t ticks) const {
  return timestamp_valid_bits_ >= 64 ? ticks : ticks & ((uint64_t{1} << timestamp_valid_bits_) - 1);
}

std::vector<VmaAllocation> Sequence::touched_allocations() const {
  std::vector<VmaAllocation> allocations;
  allocations.reserve(touched_.size());
  for (VkBuffer b : touched_) {
    for (const VulkanMemoryResource* mr : resources_) {
      if (VmaAllocation a = mr->find_allocation(b)) {
        allocations.push_back(a);
        break;
      }
    }
  }
  return allocations;
}

}  // namespace kiss_vk
