// ---------------------------------------------------------------------------
// bm_prof -- P1 prototype profiler (canonical JSONL store)
//
// Goal of this MVP: prove the new collection path end-to-end with a REAL
// harness-produced sample, so the schema + Python loader can be designed
// against ground truth instead of a hand-written example.
//
// What it does differently from bm_fully_vs_normal (which it does NOT replace
// yet -- the old flow stays intact, RFC Phase P3):
//   * timing engine is google-benchmark (warmup + fixed iterations), not a
//     hand-rolled chrono throughput loop;
//   * keeps the DISTRIBUTION (count/p50/p95/p99/mean/cv/min/max), not a single
//     throughput-mean -- that was the thing the z3 table was missing;
//   * `pu` is a self-describing field, not a CSV column position (kills the
//     enum<->index coupling);
//   * only measures PUs the device actually has (vulkan + present CPU tiers),
//     so absent PUs are simply absent rather than a silent 0.0;
//   * carries MEASURED provenance (git sha, /sys governor, timestamp). It does
//     NOT emit `throttled` -- we don't measure thermals yet, and a fabricated
//     `throttled:false` is worse than an omitted field.
//
// Scope kept deliberately minimal: scenario is "isolated" only. The co-runner
// (interference) load is the next increment -- it slots into the measured loop
// by saturating the OTHER PUs in background threads.
// ---------------------------------------------------------------------------

#include <benchmark/benchmark.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <string>
#include <vector>

#include "builtin-apps/app.hpp"  // parse_args, g_device_id, get_cores_by_type, has_*_cores
#include "const.hpp"             // DispatcherT (VulkanDispatcher), AppDataT, kNumStages

#ifndef BT_GIT_SHA
#define BT_GIT_SHA "unknown"
#endif

namespace {

// One (stage, pu) measurement cell; `samples_s` is filled by the gbench loop.
struct Cell {
  int stage;
  std::string pu;                 // "vulkan" | "big" | "medium" | "little"
  ProcessorType cpu_pt;           // unused when pu == "vulkan"
  std::vector<double> samples_s;  // raw per-iteration time, seconds
};

int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : fallback;
}

std::string read_governor() {
  std::ifstream f("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
  std::string g;
  if (f && std::getline(f, g)) return g;
  return "unknown";
}

std::string iso8601_utc_now() {
  const std::time_t tt = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&tt, &tm);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

std::string hostname() {
  char buf[256];
  return (gethostname(buf, sizeof buf) == 0) ? std::string(buf) : "unknown";
}

// Nearest-rank percentile on an ascending-sorted vector, q in [0,1].
double percentile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return 0.0;
  auto idx = static_cast<size_t>(std::ceil(q * sorted.size()));
  if (idx > 0) idx -= 1;
  if (idx >= sorted.size()) idx = sorted.size() - 1;
  return sorted[idx];
}

}  // namespace

int main(int argc, char** argv) {
  // stdout is reserved for JSONL records -- send every log line to stderr so a
  // captured run-NNN.jsonl is pure data regardless of what parse_args logs.
  spdlog::set_default_logger(spdlog::stderr_color_mt("bt_prof"));
  spdlog::set_level(spdlog::level::off);

  parse_args(argc, argv);  // populates g_device_id + g_big/med/lit_cores from the registry

  const int iters = env_int("BT_PROF_ITERS", 100);    // measured iterations / cell
  const int warmup = env_int("BT_PROF_WARMUP", 20);   // discarded warmup iterations
  const int run_id = env_int("BT_PROF_RUN", 1);       // the driver owns run numbering

  // One Vulkan engine for the whole process (mr + GPU timestamp sequence).
  DispatcherT disp;

  // Build the cell list from the PUs this device ACTUALLY has -- absent tiers
  // are never measured, so they never show up as a fake 0.0.
  std::vector<std::pair<std::string, ProcessorType>> cpu_pus;
  if (has_big_cores()) cpu_pus.emplace_back("big", ProcessorType::kBigCore);
  if (has_med_cores()) cpu_pus.emplace_back("medium", ProcessorType::kMediumCore);
  if (has_lit_cores()) cpu_pus.emplace_back("little", ProcessorType::kLittleCore);

  std::vector<Cell> cells;
  cells.reserve(kNumStages * (1 + cpu_pus.size()));  // reserve => stable &cell pointers
  for (int s = 1; s <= static_cast<int>(kNumStages); ++s) {
    cells.push_back({s, "vulkan", ProcessorType::kBigCore, {}});
    for (auto& [name, pt] : cpu_pus) cells.push_back({s, name, pt, {}});
  }

  for (Cell& cell : cells) {
    Cell* c = &cell;
    benchmark::RegisterBenchmark(
        (cell.pu + "/stage" + std::to_string(cell.stage)).c_str(),
        [c, &disp, warmup](benchmark::State& state) {
          AppDataT app(disp.get_mr());
          const int s = c->stage;
          const bool is_vk = (c->pu == "vulkan");
          auto* seq = disp.get_seq();
          const bool gpu_ts = is_vk && seq->gpu_timestamps_supported();

          auto run_once = [&] {
            if (is_vk)
              disp.dispatch_multi_stage(app, s, s);
            else
              cifar_dense::omp::dispatch_multi_stage(
                  get_cores_by_type(c->cpu_pt), get_cores_by_type(c->cpu_pt).size(), app, s, s);
          };

          for (int i = 0; i < warmup; ++i) run_once();

          c->samples_s.clear();
          c->samples_s.reserve(static_cast<size_t>(state.max_iterations));
          for (auto _ : state) {
            double t;
            if (gpu_ts) {
              run_once();
              t = seq->get_last_gpu_time_ns() * 1e-9;
            } else {
              const auto t0 = std::chrono::steady_clock::now();
              run_once();
              const auto t1 = std::chrono::steady_clock::now();
              t = std::chrono::duration<double>(t1 - t0).count();
            }
            state.SetIterationTime(t);
            c->samples_s.push_back(t);
          }
        })
        ->Unit(benchmark::kMillisecond)
        ->UseManualTime()
        ->Iterations(iters);
  }

  // Run gbench but swallow its console output -- stdout is reserved for JSONL.
  struct NullReporter : benchmark::BenchmarkReporter {
    bool ReportContext(const Context&) override { return true; }
    void ReportRuns(const std::vector<Run>&) override {}
  } null_reporter;
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks(&null_reporter);
  benchmark::Shutdown();

  // Emit one self-describing JSONL record per (stage, pu).
  const std::string gov = read_governor();
  const nlohmann::json provenance = {
      {"git_sha", BT_GIT_SHA},
      {"ts", iso8601_utc_now()},
      {"host", hostname()},
      {"freq_governor", gov},
      {"freq_locked", gov == "performance" || gov == "userspace"},
      {"warmup_iters", warmup},
      {"harness", "bt.prof/v0"},
  };

  for (Cell& cell : cells) {
    if (cell.samples_s.empty()) continue;
    std::vector<double> ms;
    ms.reserve(cell.samples_s.size());
    for (double s : cell.samples_s) ms.push_back(s * 1e3);
    std::sort(ms.begin(), ms.end());

    const double sum = std::accumulate(ms.begin(), ms.end(), 0.0);
    const double mean = sum / ms.size();
    double var = 0.0;
    for (double v : ms) var += (v - mean) * (v - mean);
    var /= ms.size();
    const double stddev = std::sqrt(var);

    const nlohmann::json rec = {
        {"schema", "bt.profiling/v0"},
        {"device", g_device_id},
        {"app", "cifar-dense"},
        {"backend", "vulkan"},
        {"scenario", "isolated"},
        {"run", run_id},
        {"stage", cell.stage},
        {"pu", cell.pu},
        {"timing",
         {{"unit", "ms"},
          {"count", ms.size()},
          {"p50", percentile(ms, 0.50)},
          {"p95", percentile(ms, 0.95)},
          {"p99", percentile(ms, 0.99)},
          {"mean", mean},
          {"cv", mean > 0 ? stddev / mean : 0.0},
          {"min", ms.front()},
          {"max", ms.back()}}},
        {"provenance", provenance},
    };
    std::cout << rec.dump() << "\n";
  }
  return 0;
}
