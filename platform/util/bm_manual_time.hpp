#pragma once

#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdint>

// The Vulkan helper (run_vk_stage_timed) is only compiled in benchmark TUs that
// define BT_BM_VULKAN before including this header (and that pull in the
// kiss_vk::Sequence definition). This keeps the OMP/CUDA benchmark TUs free of
// any Vulkan dependency.
#ifdef BT_BM_VULKAN
#include "platform/engine/vulkan/sequence.hpp"
#endif

// Helpers to report device/cycle-accurate per-stage time to Google Benchmark
// via state.SetIterationTime() (requires the benchmark to be registered with
// ->UseManualTime()).
//
// For Vulkan we read the GPU-side timestamp recorded by kiss_vk::Sequence. When
// device timestamps are unsupported on the queue (timestampValidBits == 0) we
// transparently fall back to host wall-clock timing of the stage call.

namespace bt_bm {

#if defined(__aarch64__)
// ARM64 cycle counter (cntvct_el0 / cntfrq_el0) — matches the paper's host
// timing (record.hpp). Reused here directly to avoid pulling pipeline/schedule
// dependencies into the benchmark targets.
inline uint64_t host_cycles() {
  uint64_t c;
  asm volatile("mrs %0, cntvct_el0" : "=r"(c));
  return c;
}
inline uint64_t host_cycle_freq() {
  uint64_t f;
  asm volatile("mrs %0, cntfrq_el0" : "=r"(f));
  return f;
}
#endif

// Run `stage()` and report host stage time via state.SetIterationTime().
// On ARM64 this uses the cntvct_el0 cycle counter (paper methodology); elsewhere
// it falls back to steady_clock wall time.
template <typename StageFn>
inline void run_omp_stage_timed(benchmark::State& state, StageFn&& stage) {
#if defined(__aarch64__)
  const uint64_t freq = host_cycle_freq();
  if (freq != 0) {
    const uint64_t c0 = host_cycles();
    stage();
    const uint64_t c1 = host_cycles();
    state.SetIterationTime(static_cast<double>(c1 - c0) / static_cast<double>(freq));
    return;
  }
#endif
  const auto t0 = std::chrono::steady_clock::now();
  stage();
  const auto t1 = std::chrono::steady_clock::now();
  state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
}

#ifdef BT_BM_VULKAN
// Run `stage()` and report the device GPU time of the last submitted command
// buffer. Falls back to wall-clock when timestamps are unsupported.
template <typename StageFn>
inline void run_vk_stage_timed(benchmark::State& state, kiss_vk::Sequence* seq, StageFn&& stage) {
  if (seq->gpu_timestamps_supported()) {
    stage();
    const double gpu_ns = seq->get_last_gpu_time_ns();
    state.SetIterationTime(gpu_ns * 1e-9);
  } else {
    const auto t0 = std::chrono::steady_clock::now();
    stage();
    const auto t1 = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
  }
}
#endif  // BT_BM_VULKAN

}  // namespace bt_bm
