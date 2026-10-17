#pragma once
// ---------------------------------------------------------------------------
// bm_ranges_vulkan -- GPU stage-range records (gpu-chunk-cost-model design D2/D3): every
// contiguous stage range [a, b] of the app dispatched as ONE GPU chunk through the path a
// pipeline's GPU chunk uses (submit_multi_stage + complete_multi_stage on its own Sequence:
// every submission, the host cache maintenance and any host read-back between stages), timed
// on the host per chunk. Each range is measured back-to-back (gap 0 -> t0) and then with an
// idle gap of t0 before every chunk (-> t1), the two cadences the solver interpolates between.
//
//   * ranges run in a shuffled order (fixed seed), so drift spreads across them;
//   * samples follow bm-prof's calibrated time budget (BT_PROF_BUDGET_S and friends);
//   * each range gets its own pool of `--items` items placed as a pipeline whose GPU runs
//     exactly [a, b] (runtime/placement.hpp), primed for stages 1..a-1 on the GPU;
//   * BT_PROF_SCENARIO=interference runs bm-prof's background load: every present CPU tier
//     loops over the stages of the range being measured, on its own data;
//   * the GPU clock is sampled while each range runs (GpuClockSampler).
//
// Output: one bt.profiling-range/v0 JSONL record per (range, gap) on stdout
// (schemas/profiling-range.schema.json); logs go to stderr. Include after the cell's
// const.hpp (DispatcherT, AppDataT, kNumStages).
// ---------------------------------------------------------------------------

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include "profiler/bm_prof_common.hpp"

namespace bt_ranges {

template <class OmpDispatch>
inline int run(int argc, char** argv, const char* app_name, const OmpDispatch& omp_dispatch) {
  spdlog::set_default_logger(spdlog::stderr_color_mt("bt_ranges"));
  spdlog::set_level(spdlog::level::off);

  PARSE_ARGS_BEGIN
  int items = 4;
  app.add_option("--items", items, "Pooled items each range cycles through");
  PARSE_ARGS_END
  spdlog::set_level(spdlog::level::off);  // PARSE_ARGS_END logs the cores at info

  using bt_prof::env_int;
  const int run_id = env_int("BT_PROF_RUN", 1);
  const int warmup = env_int("BT_PROF_WARMUP", 20);
  const char* scenario = std::getenv("BT_PROF_SCENARIO");
  if (scenario == nullptr) scenario = "isolated";
  const bool interfere = std::string(scenario) == "interference";
  const int n = static_cast<int>(kNumStages);

  DispatcherT disp;

  // Interference: one background thread per present CPU tier, each on its own fully primed
  // item, loops over the stages of the range currently measured (`current` = a * 100 + b),
  // one stage per dispatch, so a range switch takes effect after one stage.
  std::atomic<int> current{100 + n};
  std::atomic<bool> stop{false};
  std::atomic<int> bg_ready{0};
  std::vector<std::unique_ptr<AppDataT>> bg_apps;
  std::vector<std::thread> bg;
  if (interfere) {
    for (auto& [name, pt] : bt_prof::present_cpu_pus()) {
      bg_apps.push_back(std::make_unique<AppDataT>(bt_pipe::as_mr_ptr(disp.get_mr())));
      AppDataT* a = bg_apps.back().get();
      disp.dispatch_multi_stage(*a, 1, n);
      bg.emplace_back(
          [a, cores = get_cores_by_type(pt), &current, &stop, &bg_ready, &omp_dispatch] {
            int s = 0;
            bool first = true;
            while (!stop.load(std::memory_order_relaxed)) {
              const int r = current.load(std::memory_order_relaxed);
              const int lo = r / 100, hi = r % 100;
              s = (s < lo || s >= hi) ? lo : s + 1;
              omp_dispatch(cores, cores.size(), *a, s, s);
              if (first) {
                bg_ready.fetch_add(1, std::memory_order_relaxed);
                first = false;
              }
            }
          });
    }
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(env_int("BT_PROF_SETTLE_MS", 200));
    while (bg_ready.load(std::memory_order_relaxed) < static_cast<int>(bg.size()) &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
  }

  std::vector<std::pair<int, int>> ranges;
  for (int a = 1; a <= n; ++a)
    for (int b = a; b <= n; ++b) ranges.emplace_back(a, b);
  std::mt19937 rng(114514);
  std::shuffle(ranges.begin(), ranges.end(), rng);

  const nlohmann::json run_provenance = bt_prof::base_provenance(warmup, "bt.ranges/v0");
  for (const auto& [a, b] : ranges) {
    current.store(a * 100 + b, std::memory_order_relaxed);

    std::set<int> gpu;
    for (int s = a; s <= b; ++s) gpu.insert(s);
    const auto pool = make_placed_dataset<AppDataT>(disp, static_cast<size_t>(items), gpu);
    if (a > 1)
      for (const auto& item : pool) disp.dispatch_multi_stage(*item, 1, a - 1);
    const auto seq = disp.make_seq();  // the chunk's own Sequence, as in the pipeline

    size_t next = 0;
    uint64_t last_ticket = 0;
    int submissions = 0;
    double gap_s = 0;
    auto time_once = [&]() -> double {
      if (gap_s > 0) std::this_thread::sleep_for(std::chrono::duration<double>(gap_s));
      AppDataT& item = *pool[next++ % pool.size()];
      const auto t0 = std::chrono::steady_clock::now();
      const uint64_t ticket = disp.submit_multi_stage(item, a, b, *seq);
      disp.complete_multi_stage(item, a, b, *seq, ticket);
      const double dt =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      submissions = static_cast<int>(ticket - last_ticket);
      last_ticket = ticket;
      return dt;
    };

    // Record every item's command buffers, then warm the GPU back-to-back as bm-prof does.
    for (int i = 0; i < items; ++i) time_once();
    const auto warm_until = std::chrono::steady_clock::now() +
                            std::chrono::milliseconds(env_int("BT_PROF_GPU_WARM_MS", 150));
    while (std::chrono::steady_clock::now() < warm_until) time_once();

    for (const bool idle : {false, true}) {
      std::vector<double> samples_s;
      bt_prof::GpuClockSampler sampler(env_int("BT_PROF_CLOCK_PERIOD_MS", 5));
      const bool abandoned = bt_prof::measure_calibrated(samples_s, time_once);
      const auto clock = sampler.stop();

      nlohmann::json provenance = run_provenance;
      if (abandoned) provenance["abandoned"] = true;
      provenance["items"] = items;
      provenance["gpu_clock_mhz"] = clock.mhz;
      provenance["gpu_clock_samples"] = clock.samples;
      provenance["gpu_clock_min_mhz"] = clock.min_mhz;
      provenance["gpu_clock_max_mhz"] = clock.max_mhz;
      const nlohmann::json timing = bt_prof::timing_json(samples_s);
      const nlohmann::json rec = {
          {"schema", "bt.profiling-range/v0"},
          {"device", g_device_id},
          {"app", app_name},
          {"backend", "vulkan"},
          {"scenario", scenario},
          {"run", run_id},
          {"start_stage", a},
          {"end_stage", b},
          {"pu", "vulkan"},
          {"gap_ms", gap_s * 1e3},
          {"submissions", submissions},
          {"timing", timing},
          {"provenance", provenance},
      };
      std::cout << rec.dump() << std::endl;                   // one line per (range, gap), flushed
      if (!idle) gap_s = timing["p50"].get<double>() * 1e-3;  // design D3: gap = t0
    }
  }

  stop.store(true);
  for (auto& t : bg) t.join();
  return 0;
}

}  // namespace bt_ranges
