#pragma once
// ---------------------------------------------------------------------------
// bm_baseline_common -- shared driver for the single-PU "no framework" baselines
// (all stages on one PU, plain google-benchmark loop). The 6 pipe/<cell>/
// bm_baseline.* were forked copies that drifted (the vk files registered default
// OMP + 3 CPU-tier variants + GPU via static BENCHMARK macros; the cu files only
// default OMP + GPU). This registers, uniformly, the GPU baseline + a baseline
// for EVERY CPU tier the device actually has -- the bm_prof_common.hpp pattern.
//
// Each cell's bm_baseline.{cpp,cu} is now just a main() that calls run_baselines
// with its types + dispatch closures. as_mr_ptr (pipeline_common.hpp, pulled in
// via const.hpp) normalizes the CUDA-ref vs Vulkan-pointer get_mr().
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include <string>
#include <vector>

#include "builtin-apps/app.hpp"  // parse_args, ProcessorType, get_cores_by_type, has_*_cores
#include "mr_ptr.hpp"            // bt_pipe::as_mr_ptr

namespace bt_baseline {

// omp_default(app, lo, hi)           : all stages on the default OMP cores
// omp_tier(cores, n, app, lo, hi)    : all stages pinned to one CPU tier
// gpu(disp, app, lo, hi)             : all stages on the GPU (Vulkan/CUDA)
template <class AppDataT, class DispatcherT, class OmpDefault, class OmpTier, class Gpu>
inline int run(int argc, char** argv, int n_stages, const std::string& app_label,
               const std::string& gpu_label, OmpDefault omp_default, OmpTier omp_tier, Gpu gpu) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::off);

  // GPU baseline.
  benchmark::RegisterBenchmark(
      (gpu_label + "/" + app_label + "/Baseline").c_str(),
      [=](benchmark::State& state) {
        DispatcherT disp;
        AppDataT app(bt_pipe::as_mr_ptr(disp.get_mr()));
        gpu(disp, app, 1, n_stages);  // warmup
        for (auto _ : state) gpu(disp, app, 1, n_stages);
      })
      ->Unit(benchmark::kMillisecond);

  // Default-OMP baseline (whatever cores OpenMP picks).
  benchmark::RegisterBenchmark(
      ("OMP/" + app_label + "/Baseline").c_str(),
      [=](benchmark::State& state) {
        DispatcherT disp;
        AppDataT app(bt_pipe::as_mr_ptr(disp.get_mr()));
        omp_default(app, 1, n_stages);  // warmup
        for (auto _ : state) omp_default(app, 1, n_stages);
      })
      ->Unit(benchmark::kMillisecond);

  // One baseline per CPU tier the device ACTUALLY has.
  struct Tier {
    const char* name;
    ProcessorType pt;
    bool present;
  };
  const Tier tiers[] = {
      {"LittleCores", ProcessorType::kLittleCore, has_lit_cores()},
      {"MediumCores", ProcessorType::kMediumCore, has_med_cores()},
      {"BigCores", ProcessorType::kBigCore, has_big_cores()},
  };
  for (const auto& t : tiers) {
    if (!t.present) continue;
    const ProcessorType pt = t.pt;
    benchmark::RegisterBenchmark(
        ("OMP/" + app_label + "/Baseline/" + t.name).c_str(),
        [=](benchmark::State& state) {
          DispatcherT disp;
          AppDataT app(bt_pipe::as_mr_ptr(disp.get_mr()));
          std::vector<int>& cores = get_cores_by_type(pt);
          omp_tier(cores, cores.size(), app, 1, n_stages);  // warmup
          for (auto _ : state) omp_tier(cores, cores.size(), app, 1, n_stages);
        })
        ->Unit(benchmark::kMillisecond);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}

}  // namespace bt_baseline
