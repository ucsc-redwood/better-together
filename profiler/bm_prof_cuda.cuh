#pragma once
// ---------------------------------------------------------------------------
// bm_prof_cuda -- the CUDA GPU-timer POLICY for run_bm_prof, shared by every
// <app>-cu/bm_prof.cu. It owns the cudaEvent path (GPU-elapsed) and its
// host-wall-clock fallback (BT_PROF_GPU_WALLCLOCK=1: dispatch + device sync, so
// interference captures the launch/sync host contention the GPU clock excludes).
//
// One instance is created ONCE per registered benchmark (run_bm_prof's MakeTimer
// factory), so the cudaEvents are created exactly once -- not per sample. The
// create/destroy of the events is unconditional, matching the old hand-written
// loop which made them at the top of every benchmark lambda.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "platform/engine/cuda/helpers.cuh"  // CheckCuda
#include "profiler/bm_prof_common.hpp"

namespace bt_prof {

// Disp/App are the cell's CudaDispatcher / AppData. CheckCuda comes from the
// app's cuda dispatcher header (pulled in via the cell's const.hpp).
template <class Disp, class App>
struct CudaTimer {
  bool gpu_wall;
  cudaEvent_t ev_start{};
  cudaEvent_t ev_stop{};

  CudaTimer() : gpu_wall(env_int("BT_PROF_GPU_WALLCLOCK", 0) != 0) {
    CheckCuda(cudaEventCreate(&ev_start));
    CheckCuda(cudaEventCreate(&ev_stop));
  }
  ~CudaTimer() {
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
  }
  CudaTimer(const CudaTimer&) = delete;
  CudaTimer& operator=(const CudaTimer&) = delete;

  // On-GPU timing is active unless the caller forced host wall-clock.
  bool gpu_timed() const { return !gpu_wall; }

  // GPU-elapsed (cudaEvent) for one dispatch -> seconds.
  double time_gpu(Disp& disp, App& app, int s) {
    CheckCuda(cudaEventRecord(ev_start, 0));
    disp.dispatch_multi_stage(app, s, s);
    CheckCuda(cudaEventRecord(ev_stop, 0));
    CheckCuda(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    return ms * 1e-3;
  }

  // The gpu_wall path: dispatch + a device sync (timed by the caller's clock).
  void dispatch_sync(Disp& disp, App& app, int s) {
    disp.dispatch_multi_stage(app, s, s);
    CheckCuda(cudaDeviceSynchronize());
  }
};

}  // namespace bt_prof
