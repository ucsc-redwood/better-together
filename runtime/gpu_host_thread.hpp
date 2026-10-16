#pragma once
// ---------------------------------------------------------------------------
// runtime/gpu_host_thread -- where the thread that drives a Vulkan GPU chunk runs.
//
// That thread records, flushes, submits, waits and invalidates for every item. OMP chunks are
// pinned to their tier; this thread is left to the OS by default, and that stays the default:
// pinning it was measured (docs/reports-for-human/2026-10-16-vk-data-flow-fixes.md) and is not
// a win on every device. Tree pipeline, Big[1] | GPU[2-4] | Big[5-7], interval per item against
// unpinned: Pixel 7a +8 % (core 4), +13 % (cores 4,5), +3 % (core 6); Samsung -2 % (core 4),
// -43 % (cores 4,5,6) and -22 % (core 7), but those runs span 2x between rounds (108k-245k
// ticks) while the unpinned ones stay at ~245k. Pinned to a little core the Pixel was 71 %
// slower. So it is opt-in:
//
//   BT_GPU_THREAD_CORES unset  the OS places the thread (the default)
//   BT_GPU_THREAD_CORES=none   the same, explicitly
//   BT_GPU_THREAD_CORES=<id>[,<id>...]  pin to exactly those cores
//   BT_GPU_THREAD_CORES=auto   a medium or big core tier no CPU chunk of the schedule uses,
//                              medium first; none free: not pinned (never a little core, never
//                              the single "super" core)
// ---------------------------------------------------------------------------

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <exception>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "platform/registry/affinity.hpp"
#include "platform/registry/device_registry.hpp"
#include "runtime/schedule.hpp"

namespace bt {

// The tier choice itself, on any topology: `busy` are the tiers CPU chunks run on, `cores_of`
// the cores of a tier. Empty: do not pin.
[[nodiscard]] inline std::vector<int> choose_gpu_host_cores(
    const std::set<ProcessorType>& busy,
    const std::function<std::vector<int>(ProcessorType)>& cores_of) {
  for (const ProcessorType tier : {ProcessorType::kMediumCore, ProcessorType::kBigCore}) {
    if (busy.contains(tier)) continue;
    if (auto cores = cores_of(tier); !cores.empty()) return cores;
  }
  return {};
}

// `<id>[,<id>...]` -> ids; `none` or empty -> {}. A malformed list throws.
[[nodiscard]] inline std::vector<int> parse_core_list(const std::string& list) {
  std::vector<int> ids;
  if (list.empty() || list == "none") return ids;
  size_t pos = 0;
  while (pos <= list.size()) {
    const size_t comma = list.find(',', pos);
    const std::string token = list.substr(pos, comma == std::string::npos ? comma : comma - pos);
    size_t used = 0;
    const int id = std::stoi(token, &used);  // throws on a non-number
    if (used != token.size() || id < 0) throw std::invalid_argument("bad core id: " + token);
    ids.push_back(id);
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return ids;
}

// The cores for the GPU-driving thread of `sched` on this device (see BT_GPU_THREAD_CORES
// above); empty: leave it to the OS.
[[nodiscard]] inline std::vector<int> gpu_host_cores(const Schedule& sched,
                                                     const ExecutionModel gpu_em) {
  if (gpu_em != ExecutionModel::kVulkan) return {};  // measured for Vulkan only
  const char* env = std::getenv("BT_GPU_THREAD_CORES");
  if (env == nullptr) return {};
  const std::string value = env;
  if (value == "auto") {
    std::set<ProcessorType> busy;
    for (const auto& chunk : sched.chunks) {
      if (chunk.exec_model == ExecutionModel::kOMP && chunk.cpu_proc_type.has_value()) {
        busy.insert(*chunk.cpu_proc_type);
      }
    }
    return choose_gpu_host_cores(busy, [](const ProcessorType t) { return get_cores_by_type(t); });
  }
  try {
    return parse_core_list(value);
  } catch (const std::exception& e) {
    spdlog::warn("BT_GPU_THREAD_CORES={} ignored: {}", value, e.what());
    return {};
  }
}

// Pins the calling thread; a failure warns (it runs inside a worker thread, where a throw
// would terminate the process).
inline void pin_gpu_host_thread(const std::vector<int>& cores) {
  if (cores.empty()) return;
  try {
    bind_thread_to_cores(cores);
  } catch (const std::exception& e) {
    spdlog::warn("GPU host thread not pinned: {}", e.what());
  }
}

}  // namespace bt
