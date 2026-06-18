// ---------------------------------------------------------------------------
// bm_prof -- canonical-JSONL profiler for cifar-sparse x Vulkan (isolated).
//
// One bm_prof binary per (app, backend) cell. It measures every PU the device
// has -- the Vulkan PU (GPU-timestamped) plus each present CPU tier -- and emits
// one self-describing JSONL record per (stage, pu) carrying the full timing
// DISTRIBUTION (p50/p95/p99/cv/...) and MEASURED provenance. Shared plumbing
// (stats, provenance, emission) lives in ../bm_prof_common.hpp; this file owns
// only the measured loop (VK gpu-timestamp path) and the app/backend identity.
//
// Sampling is a calibrated TIME budget (bm_prof_common::measure_calibrated), not
// a fixed iteration count -- slow cells (e.g. little-core conv) self-limit to a
// few samples; knobs: BT_PROF_BUDGET_S / _ABANDON_S / _MAX_CELL_S / _MIN_ITERS /
// _MAX_ITERS / _WARMUP, plus BT_PROF_RUN (the driver owns run numbering).
// stdout = JSONL only; all logs go to stderr. BT_PROF_SCENARIO selects isolated (default) or interference
// (saturate the other PUs with the same stage while measuring the target).
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "profiler/bm_prof_common.hpp"
#include "platform/registry/device_registry.hpp"  // parse_args, get_cores_by_type
#include "const.hpp"             // DispatcherT (VulkanDispatcher), AppDataT, kNumStages

using bt_prof::Cell;

int main(int argc, char** argv) {
  // stdout is reserved for JSONL records -- send every log line to stderr so a
  // captured run-NNN.jsonl is pure data regardless of what parse_args logs.
  spdlog::set_default_logger(spdlog::stderr_color_mt("bt_prof"));
  spdlog::set_level(spdlog::level::off);

  parse_args(argc, argv);  // populates g_device_id + g_big/med/lit_cores from the registry

  const int warmup = bt_prof::env_int("BT_PROF_WARMUP", 20);  // warmup cap (provenance)
  const int run_id = bt_prof::env_int("BT_PROF_RUN", 1);      // the driver owns run numbering
  // "isolated" (default) or "interference": saturate every OTHER present PU with
  // the same stage while the target is measured (the paper's BTPM condition).
  const char* scenario = std::getenv("BT_PROF_SCENARIO");
  if (!scenario) scenario = "isolated";
  const bool interfere = std::string(scenario) == "interference";
  // BT_PROF_GPU_WALLCLOCK=1 times the GPU PU by host wall-clock (full dispatch
  // round-trip) instead of the on-GPU timestamp -- so interference captures the
  // host-side contention (submit/fence-wait) the GPU timestamp excludes.
  const bool gpu_wall = bt_prof::env_int("BT_PROF_GPU_WALLCLOCK", 0) != 0;

  DispatcherT disp;  // one Vulkan engine for the whole process

  std::vector<Cell> cells = bt_prof::make_cells(static_cast<int>(kNumStages), "vulkan");

  for (Cell& cell : cells) {
    Cell* c = &cell;
    benchmark::RegisterBenchmark(
        (cell.pu + "/stage" + std::to_string(cell.stage)).c_str(),
        [c, &disp, interfere, gpu_wall](benchmark::State& state) {
          AppDataT app(disp.get_mr());
          const int s = c->stage;
          const bool is_vk = (c->pu == "vulkan");
          auto* seq = disp.get_seq();
          const bool gpu_ts = is_vk && seq->gpu_timestamps_supported() && !gpu_wall;

          // One measured run -> seconds. Prefer the on-GPU timestamp for Vulkan.
          auto time_once = [&]() -> double {
            if (gpu_ts) {
              disp.dispatch_multi_stage(app, s, s);
              return seq->get_last_gpu_time_ns() * 1e-9;
            }
            const auto t0 = std::chrono::steady_clock::now();
            if (is_vk)
              disp.dispatch_multi_stage(app, s, s);
            else
              cifar_sparse::omp::dispatch_multi_stage(
                  get_cores_by_type(c->cpu_pt), get_cores_by_type(c->cpu_pt).size(), app, s, s);
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          };

          // Interference: saturate every OTHER present PU with the same stage on
          // its OWN AppData (disjoint -> no cross-PU data race). At most one thread
          // ever touches the GPU dispatcher, matching the old BM_run_fully.
          std::atomic<bool> stop{false};
          std::vector<std::unique_ptr<AppDataT>> bg_apps;
          std::vector<std::thread> bg;
          if (interfere) {
            if (!is_vk) {  // GPU contends only when the target is a CPU tier
              bg_apps.push_back(std::make_unique<AppDataT>(disp.get_mr()));
              AppDataT* a = bg_apps.back().get();
              bg.emplace_back([&disp, a, s, &stop] {
                while (!stop.load(std::memory_order_relaxed)) disp.dispatch_multi_stage(*a, s, s);
              });
            }
            for (auto& [bname, bpt] : bt_prof::present_cpu_pus()) {
              if (!is_vk && bpt == c->cpu_pt) continue;  // skip the target CPU tier itself
              bg_apps.push_back(std::make_unique<AppDataT>(disp.get_mr()));
              AppDataT* a = bg_apps.back().get();
              auto cores = get_cores_by_type(bpt);
              bg.emplace_back([a, cores, s, &stop] {
                while (!stop.load(std::memory_order_relaxed))
                  cifar_sparse::omp::dispatch_multi_stage(cores, cores.size(), *a, s, s);
              });
            }
          }

          // Iterations(1): the calibrated helper owns warmup + adaptive sampling.
          for (auto _ : state) {
            c->abandoned = bt_prof::measure_calibrated(c->samples_s, time_once);
            double total = 0.0;
            for (double t : c->samples_s) total += t;
            state.SetIterationTime(total);
          }

          stop.store(true);
          for (auto& t : bg) t.join();
        })
        ->Unit(benchmark::kMillisecond)
        ->UseManualTime()
        ->Iterations(1);
  }

  bt_prof::NullReporter null_reporter;
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks(&null_reporter);
  benchmark::Shutdown();

  bt_prof::emit_jsonl(cells, "cifar-sparse", "vulkan", scenario, run_id, warmup);
  return 0;
}
