#pragma once
// ---------------------------------------------------------------------------
// bm_phases_vulkan -- where a Vulkan GPU chunk's time goes (design D1; submission spec
// "Per-chunk overhead is measured by phase"). Runs one chunk shape [start, end] over a
// small pool of items, one submission at a time, and reports per chunk the host phases of
// the Sequence (record / between / flush / submit / wait / invalidate), the GPU interval,
// and -- with calibrated timestamps -- the wait split into submit->GPU-start latency, GPU
// execution and GPU-end->wake-up latency. --gap-ms inserts an idle gap between chunks (the
// DVFS check). It also times each stage of the shape on its own, so the chunk's wall time
// can be set against the per-stage table sum the solver predicts with.
//
// Output: one CSV row per measured chunk, then `# summary` lines (medians, and the largest
// residual of each sum). Include after the cell's const.hpp (DispatcherT, AppDataT).
// ---------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "platform/registry/device_registry.hpp"
#include "runtime/placement.hpp"

namespace bt_phases {

inline double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::ranges::sort(v);
  return v[v.size() / 2];
}

template <class OmpDispatch>
inline int run(int argc, char** argv, const OmpDispatch& omp_dispatch) {
  PARSE_ARGS_BEGIN
  int start = 1;
  int end = static_cast<int>(kNumStages);
  int chunks = 200;
  int warmup = 20;
  int items = 4;
  int gap_ms = 0;
  app.add_option("-s,--start-stage", start, "First stage of the chunk");
  app.add_option("-e,--end-stage", end, "Last stage of the chunk");
  app.add_option("-n,--chunks", chunks, "Measured chunks");
  app.add_option("--warmup", warmup, "Unreported chunks first");
  app.add_option("--items", items, "Pooled items cycled through");
  app.add_option("--gap-ms", gap_ms, "Idle gap after each chunk (ms)");
  PARSE_ARGS_END

  spdlog::set_level(spdlog::level::off);
  setenv("BT_VK_PHASES", "1", 1);  // read by every Sequence at construction

  DispatcherT disp;
  std::set<int> gpu;
  for (int s = start; s <= end; ++s) gpu.insert(s);
  const auto pool = make_placed_dataset<AppDataT>(disp, static_cast<size_t>(items), gpu);
  if (start > 1) {  // genuinely-chained items need stages 1..start-1 to have run
    const auto& cores = get_cores_by_type(first_present_cpu_type());
    for (const auto& a : pool) omp_dispatch(cores, cores.size(), *a, 1, start - 1);
  }
  const auto seq = disp.make_seq();

  // Each stage on its own (GPU timestamp), the per-stage table the solver sums.
  std::vector<double> stage_gpu_ms;
  for (int s = start; s <= end; ++s) {
    std::vector<double> t;
    for (int i = 0; i < 50; ++i) {
      disp.dispatch_multi_stage(*pool[i % items], s, s, *seq);
      if (i >= 10) t.push_back(seq->get_last_gpu_time_ns() * 1e-6);
    }
    stage_gpu_ms.push_back(median(t));
  }
  double stage_sum_ms = 0;
  for (const double t : stage_gpu_ms) stage_sum_ms += t;

  std::printf(
      "chunk,dispatch_wall_ms,wall_ms,record_ms,between_ms,flush_ms,submit_ms,wait_ms,"
      "invalidate_ms,gpu_ms,calibrated,start_latency_ms,wake_ms,recorded,host_residual_ms,"
      "wait_residual_ms\n");
  std::vector<double> c_dispatch, c_wall, c_record, c_between, c_flush, c_submit, c_wait, c_inval,
      c_gpu, c_lat, c_wake;
  double max_host_residual = 0, max_wait_residual = 0;
  int recorded = 0, calibrated = 0;
  for (int i = 0; i < warmup + chunks; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    disp.dispatch_multi_stage(*pool[i % items], start, end, *seq);
    const double dispatch_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (gap_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
    if (i < warmup) continue;

    const kiss_vk::PhaseTimes& p = seq->last_phases();
    const double host_sum =
        p.record_ns + p.between_ns + p.flush_ns + p.submit_ns + p.wait_ns + p.invalidate_ns;
    const double host_residual = (host_sum - p.wall_ns) * 1e-6;
    const double wait_residual =
        p.calibrated ? (p.start_latency_ns + p.gpu_ns + p.wake_ns - p.wait_ns) * 1e-6 : 0;
    max_host_residual = std::max(max_host_residual, std::abs(host_residual));
    max_wait_residual = std::max(max_wait_residual, std::abs(wait_residual));
    recorded += p.recorded ? 1 : 0;
    calibrated += p.calibrated ? 1 : 0;
    std::printf("%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.4f,%.4f,%d,%.6f,%.6f\n",
                i - warmup,
                dispatch_ms,
                p.wall_ns * 1e-6,
                p.record_ns * 1e-6,
                p.between_ns * 1e-6,
                p.flush_ns * 1e-6,
                p.submit_ns * 1e-6,
                p.wait_ns * 1e-6,
                p.invalidate_ns * 1e-6,
                p.gpu_ns * 1e-6,
                p.calibrated ? 1 : 0,
                p.start_latency_ns * 1e-6,
                p.wake_ns * 1e-6,
                p.recorded ? 1 : 0,
                host_residual,
                wait_residual);
    c_dispatch.push_back(dispatch_ms);
    c_wall.push_back(p.wall_ns * 1e-6);
    c_record.push_back(p.record_ns * 1e-6);
    c_between.push_back(p.between_ns * 1e-6);
    c_flush.push_back(p.flush_ns * 1e-6);
    c_submit.push_back(p.submit_ns * 1e-6);
    c_wait.push_back(p.wait_ns * 1e-6);
    c_inval.push_back(p.invalidate_ns * 1e-6);
    c_gpu.push_back(p.gpu_ns * 1e-6);
    c_lat.push_back(p.start_latency_ns * 1e-6);
    c_wake.push_back(p.wake_ns * 1e-6);
  }

  std::printf("# summary stages=%d-%d gap_ms=%d chunks=%d items=%d calibrated=%d/%d recorded=%d\n",
              start,
              end,
              gap_ms,
              chunks,
              items,
              calibrated,
              chunks,
              recorded);
  std::printf(
      "# median_ms dispatch_wall=%.4f wall=%.4f record=%.4f between=%.4f flush=%.4f "
      "submit=%.4f wait=%.4f invalidate=%.4f gpu=%.4f start_latency=%.4f wake=%.4f\n",
      median(c_dispatch),
      median(c_wall),
      median(c_record),
      median(c_between),
      median(c_flush),
      median(c_submit),
      median(c_wait),
      median(c_inval),
      median(c_gpu),
      median(c_lat),
      median(c_wake));
  std::printf("# stage_gpu_ms");
  for (size_t k = 0; k < stage_gpu_ms.size(); ++k) {
    std::printf(" s%d=%.4f", start + static_cast<int>(k), stage_gpu_ms[k]);
  }
  std::printf(" sum=%.4f\n", stage_sum_ms);
  // The sums hold by construction (contiguous host timestamps; the wait split is anchored on
  // them), so the check on the calibration is that neither latency is ever negative.
  std::printf("# max_abs_residual_ms host=%.6f wait=%.6f\n", max_host_residual, max_wait_residual);
  std::printf("# min_ms start_latency=%.4f wake=%.4f\n",
              c_lat.empty() ? 0.0 : *std::ranges::min_element(c_lat),
              c_wake.empty() ? 0.0 : *std::ranges::min_element(c_wake));
  return 0;
}

}  // namespace bt_phases
