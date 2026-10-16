#pragma once
// ---------------------------------------------------------------------------
// runtime/placement -- builds pooled AppData with each buffer placed by the schedule
// (memory-placement spec, design B3). A buffer whose every accessor is a GPU stage of the
// schedule goes to the dispatcher's weights or GPU-local class; anything a CPU chunk or host
// code touches stays in the boundary class (host-cached), so CPU chunks never read a
// write-combined mapping. Dispatchers without buffer classes, or AppData without a
// PlaceBuffer constructor, get the plain AppData(get_mr()) as before.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <memory>
#include <set>
#include <type_traits>
#include <vector>

#include "platform/mem/mr_ptr.hpp"
#include "platform/util/buffer_placement.hpp"
#include "runtime/schedule.hpp"

// The stages a schedule runs on its GPU backend.
[[nodiscard]] inline std::set<int> gpu_stages_of(const Schedule& sched,
                                                 const ExecutionModel gpu_em) {
  std::set<int> stages;
  for (const auto& c : sched.chunks) {
    if (c.exec_model != gpu_em) continue;
    for (int s = c.start_stage; s <= c.end_stage; ++s) stages.insert(s);
  }
  return stages;
}

// Stages 1..n: every stage on the GPU (single-backend benchmarks, the GPU baseline).
[[nodiscard]] inline std::set<int> all_stages(const int n) {
  std::set<int> stages;
  for (int s = 1; s <= n; ++s) stages.insert(s);
  return stages;
}

// The placement for a set of GPU stages: by class where every accessor is one of them.
template <class Dispatcher>
[[nodiscard]] bt::PlaceBuffer schedule_placement(Dispatcher& disp, std::set<int> gpu_stages) {
  return [&disp, gpu_stages = std::move(gpu_stages)](
             const bt::BufferKind kind,
             const std::initializer_list<int> stages) -> std::pmr::memory_resource* {
    const bool gpu_only =
        std::ranges::all_of(stages, [&](const int s) { return gpu_stages.contains(s); });
    if (!gpu_only) return disp.get_mr(bt::BufferClass::kBoundary);
    return disp.get_mr(kind == bt::BufferKind::kWeights ? bt::BufferClass::kWeights
                                                        : bt::BufferClass::kGpuLocal);
  };
}

template <class AppData, class Dispatcher>
[[nodiscard]] std::unique_ptr<AppData> make_placed_appdata(Dispatcher& disp,
                                                           const std::set<int>& gpu_stages) {
  if constexpr (std::is_constructible_v<AppData, const bt::PlaceBuffer&> &&
                requires { disp.get_mr(bt::BufferClass::kWeights); }) {
    return std::make_unique<AppData>(schedule_placement(disp, gpu_stages));
  } else {
    return std::make_unique<AppData>(bt_pipe::as_mr_ptr(disp.get_mr()));
  }
}

// make_dataset, placed for the stages the schedule runs on the GPU.
template <class AppData, class Dispatcher>
[[nodiscard]] std::vector<std::unique_ptr<AppData>> make_placed_dataset(
    Dispatcher& disp, const size_t num_items, const std::set<int>& gpu_stages) {
  std::vector<std::unique_ptr<AppData>> result;
  result.reserve(num_items);
  for (size_t i = 0; i < num_items; ++i) {
    result.push_back(make_placed_appdata<AppData>(disp, gpu_stages));
  }
  return result;
}
