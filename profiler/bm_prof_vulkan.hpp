#pragma once
// ---------------------------------------------------------------------------
// bm_prof_vulkan -- the Vulkan GPU-timer POLICY for run_bm_prof, shared by every
// <app>-vk/bm_prof.cpp. It owns the on-GPU timestamp path
// (seq->get_last_gpu_time_ns) and its host-wall-clock fallback
// (BT_PROF_GPU_WALLCLOCK=1, or a device without timestamp support: the full
// dispatch round-trip, so interference captures the submit/fence-wait host
// contention the GPU timestamp excludes).
//
// One instance is created ONCE per registered benchmark (run_bm_prof's MakeTimer
// factory). The Sequence pointer is cached once, mirroring the old loop.
// ---------------------------------------------------------------------------

#include "profiler/bm_prof_common.hpp"

namespace bt_prof {

// Disp/App are the cell's VulkanDispatcher / VkAppData_Safe.
template <class Disp, class App>
struct VulkanTimer {
  bool gpu_wall;
  bool ts_supported;

  explicit VulkanTimer(Disp& disp)
      : gpu_wall(env_int("BT_PROF_GPU_WALLCLOCK", 0) != 0),
        ts_supported(disp.get_seq()->gpu_timestamps_supported()) {}

  // On-GPU timing is active when the device supports timestamps and the caller
  // didn't force host wall-clock.
  bool gpu_timed() const { return ts_supported && !gpu_wall; }

  // On-GPU timestamp for one dispatch -> seconds.
  double time_gpu(Disp& disp, App& app, int s) {
    disp.dispatch_multi_stage(app, s, s);
    return disp.get_seq()->get_last_gpu_time_ns() * 1e-9;
  }

  // The wall path for the Vulkan PU: a plain dispatch round-trip (timed by the
  // caller's clock) -- no extra sync, matching the old hand-written loop.
  void dispatch_sync(Disp& disp, App& app, int s) { disp.dispatch_multi_stage(app, s, s); }
};

}  // namespace bt_prof
