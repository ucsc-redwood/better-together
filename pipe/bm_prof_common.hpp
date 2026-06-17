#pragma once
// ---------------------------------------------------------------------------
// bm_prof_common -- backend-agnostic plumbing shared by every per-cell bm_prof.
//
// Each pipe/<app>-<backend>/bm_prof.{cpp,cu} owns only its MEASURED LOOP (the VK
// gpu-timestamp path vs the CUDA-event path differ) plus the app/backend strings
// and the OMP dispatch namespace. Everything that is identical across the whole
// (app x backend) matrix lives here: env knobs, MEASURED provenance, the
// percentile/stats math, and the canonical-JSONL record emission.
//
// See pipe/cifar-dense-vk/bm_prof.cpp for the design rationale (distribution over
// throughput-mean, self-describing `pu`, absent-PU = absent, measured provenance,
// scenario="isolated" only). Output schema: schemas/profiling-table.schema.json.
// ---------------------------------------------------------------------------

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "builtin-apps/app.hpp"  // ProcessorType, g_device_id, has_*_cores

// Build-time provenance: the bt_git_sha CMake target regenerates bt_git_sha.h each
// build and puts it on the bm-prof include path. Falls back to "unknown" when the
// header isn't present (e.g. a non-benchmark build that still includes this header).
#if __has_include("bt_git_sha.h")
#include "bt_git_sha.h"
#endif
#ifndef BT_GIT_SHA
#define BT_GIT_SHA "unknown"
#endif

namespace bt_prof {

// One (stage, pu) measurement cell; `samples_s` is filled by the gbench loop.
struct Cell {
  int stage;
  std::string pu;                 // "vulkan" | "cuda" | "big" | "medium" | "little"
  ProcessorType cpu_pt;           // unused when pu is the GPU backend
  std::vector<double> samples_s;  // raw per-iteration time, seconds
  bool abandoned = false;         // deep-sampling cut short because the cell is slow
};

inline int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : fallback;
}

inline double env_double(const char* name, double fallback) {
  const char* v = std::getenv(name);
  return v ? std::atof(v) : fallback;
}

// Calibrated time-budget sampling -- the cure for "little-core-on-dense" blowing
// up the run. `time_once()` runs the kernel ONCE and returns its time in seconds.
// We probe + warm up to estimate the per-iteration cost, then spend a fixed TIME
// budget rather than a fixed iteration COUNT: cheap kernels get many samples (up
// to MAX_ITERS), expensive ones get few (down to MIN_ITERS). A kernel whose cost
// already exceeds ABANDON_S is sampled only MIN_ITERS times and flagged; a hard
// per-cell ceiling (MAX_CELL_S) stops a pathological kernel after ~one sample.
// Either way the recorded value is a REAL measurement (a large one) -- the z3
// optimizer simply never picks it; we never fabricate a sentinel. Knobs (env):
//   BT_PROF_BUDGET_S (0.3) BT_PROF_ABANDON_S (0.25) BT_PROF_MAX_CELL_S (2.0)
//   BT_PROF_MIN_ITERS (5)  BT_PROF_MAX_ITERS (2000) BT_PROF_WARMUP (20)
// Returns true if the cell was abandoned (cost-capped).
template <class TimeOnce>
bool measure_calibrated(std::vector<double>& out, TimeOnce time_once) {
  const double budget = env_double("BT_PROF_BUDGET_S", 0.3);
  const double abandon = env_double("BT_PROF_ABANDON_S", 0.25);
  const double max_cell = env_double("BT_PROF_MAX_CELL_S", 2.0);
  const int min_it = env_int("BT_PROF_MIN_ITERS", 5);
  const int max_it = env_int("BT_PROF_MAX_ITERS", 2000);
  const int warm_max = env_int("BT_PROF_WARMUP", 20);

  double c = time_once();  // first call: pipeline/JIT warm + rough cost probe
  // Already hopeless on the very first run? Keep THAT one sample and bail -- a
  // slow kernel costs seconds per iteration, so probe+warmup+resampling is pure
  // waste. One real (large) measurement is all z3 needs to never pick this cell.
  if (c > abandon) {
    out.assign(1, c);
    return true;
  }

  // Cost-aware warmup: ~50ms total, so cheap kernels warm a lot, slow ones barely.
  const int warm = std::clamp(static_cast<int>(0.05 / std::max(c, 1e-9)), 1, warm_max);
  for (int i = 0; i < warm; ++i) c = time_once();  // discard warmup; last call refines cost

  const int n = std::clamp(static_cast<int>(budget / std::max(c, 1e-9)), min_it, max_it);
  out.clear();
  out.reserve(static_cast<size_t>(n));
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    const double t = time_once();
    out.push_back(t);
    acc += t;
    if (acc > max_cell) break;  // hard wall-time ceiling: a cell that drifts slow stops early
  }
  return false;
}

inline std::string read_governor() {
  std::ifstream f("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
  std::string g;
  if (f && std::getline(f, g)) return g;
  return "unknown";
}

inline std::string iso8601_utc_now() {
  const std::time_t tt = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&tt, &tm);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

inline std::string hostname() {
  char buf[256];
  return (gethostname(buf, sizeof buf) == 0) ? std::string(buf) : "unknown";
}

// Nearest-rank percentile on an ascending-sorted vector, q in [0,1].
inline double percentile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return 0.0;
  auto idx = static_cast<size_t>(std::ceil(q * sorted.size()));
  if (idx > 0) idx -= 1;
  if (idx >= sorted.size()) idx = sorted.size() - 1;
  return sorted[idx];
}

// The CPU tiers this device ACTUALLY has -- absent tiers are never measured, so
// they never show up as a fake 0.0.
inline std::vector<std::pair<std::string, ProcessorType>> present_cpu_pus() {
  std::vector<std::pair<std::string, ProcessorType>> pus;
  if (has_big_cores()) pus.emplace_back("big", ProcessorType::kBigCore);
  if (has_med_cores()) pus.emplace_back("medium", ProcessorType::kMediumCore);
  if (has_lit_cores()) pus.emplace_back("little", ProcessorType::kLittleCore);
  return pus;
}

// Build the (gpu + present-cpu) cell list for `n_stages` stages. `gpu_pu` is the
// backend PU name ("vulkan" | "cuda"); reserve() keeps the vector stable so a
// registered benchmark can hold a raw &cell.
inline std::vector<Cell> make_cells(int n_stages, const std::string& gpu_pu) {
  const auto cpu_pus = present_cpu_pus();
  std::vector<Cell> cells;
  cells.reserve(static_cast<size_t>(n_stages) * (1 + cpu_pus.size()));
  for (int s = 1; s <= n_stages; ++s) {
    cells.push_back({s, gpu_pu, ProcessorType::kBigCore, {}});
    for (auto& [name, pt] : cpu_pus) cells.push_back({s, name, pt, {}});
  }
  return cells;
}

// Emit one self-describing JSONL record per non-empty cell to stdout. stdout is
// reserved for data -- callers route logs to stderr.
inline void emit_jsonl(const std::vector<Cell>& cells, const char* app,
                       const char* backend, const char* scenario, int run_id, int warmup) {
  const std::string gov = read_governor();
  const nlohmann::json base_provenance = {
      {"git_sha", BT_GIT_SHA},
      {"ts", iso8601_utc_now()},
      {"host", hostname()},
      {"freq_governor", gov},
      {"freq_locked", gov == "performance" || gov == "userspace"},
      {"warmup_iters", warmup},
      {"harness", "bt.prof/v0"},
  };

  for (const Cell& cell : cells) {
    if (cell.samples_s.empty()) continue;
    nlohmann::json provenance = base_provenance;
    if (cell.abandoned) provenance["abandoned"] = true;  // deep-sampling cost-capped
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
        {"app", app},
        {"backend", backend},
        {"scenario", scenario},
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
}

// gbench reporter that swallows console output -- stdout is reserved for JSONL.
struct NullReporter : benchmark::BenchmarkReporter {
  bool ReportContext(const Context&) override { return true; }
  void ReportRuns(const std::vector<Run>&) override {}
};

}  // namespace bt_prof
