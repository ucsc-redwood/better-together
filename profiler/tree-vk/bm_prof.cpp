// ---------------------------------------------------------------------------
// bm_prof -- canonical-JSONL profiler for tree x Vulkan (isolated).
//
// One bm_prof binary per (app, backend) cell. The whole driver -- env knobs,
// per-cell registration, the calibrated measured loop, interference load, and
// JSONL emission -- lives in ../bm_prof_common.hpp (run_bm_prof); the Vulkan
// GPU-timer policy (on-GPU timestamp + wall-clock fallback) lives in
// ../bm_prof_vulkan.hpp. This cell supplies only its three points of variation:
// the app/backend identity, the Vulkan timer policy, and the app's OMP token.
// ---------------------------------------------------------------------------

#include "profiler/bm_prof_vulkan.hpp"
#include "const.hpp"  // DispatcherT (VulkanDispatcher), AppDataT, kNumStages

int main(int argc, char** argv) {
  return bt_prof::run_bm_prof<DispatcherT, AppDataT>(
      argc, argv, "tree", "vulkan", "vulkan", static_cast<int>(kNumStages),
      [](DispatcherT& disp) { return bt_prof::VulkanTimer<DispatcherT, AppDataT>{disp}; },
      [](auto&& cores, size_t n, AppDataT& app, int a, int b) {
        tree::omp::dispatch_multi_stage(cores, n, app, a, b);
      });
}
