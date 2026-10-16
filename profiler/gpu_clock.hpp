#pragma once
// ---------------------------------------------------------------------------
// gpu_clock -- best-effort CURRENT GPU clock in MHz, -1 if unknown. The interference audit
// found the BTPM GPU column corrupted by DVFS (the GPU clocks up under the bg load while
// isolated runs at a low clock); recording the clock in provenance lets the loader /
// analysis see and disentangle that. Sources, in order (units differ per driver):
//   AMD RADV       the active ("*") line of pp_dpm_sclk, MHz
//   Adreno (kgsl)  gpuclk / devfreq cur_freq, Hz
//   Tegra, generic a devfreq `gpu` node's cur_freq, Hz
//   Mali (Pixel)   /sys/class/misc/mali0/device/cur_freq, kHz
//   Exynos sgpu    /sys/kernel/gpu/gpu_clock, then devfreq `*sgpu*`/cur_freq, both kHz
// A source reading 0 (the Samsung's while its GPU is idle and power-gated) counts as
// unknown, so the next one is tried. `root` prefixes every path (tests use a fake tree).
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace bt_prof {

inline int read_gpu_clock_mhz(const std::string& root = "") {
  for (const char* p :
       {"/sys/class/drm/card0/device/pp_dpm_sclk", "/sys/class/drm/card1/device/pp_dpm_sclk"}) {
    std::ifstream f(root + p);
    std::string line;
    while (f && std::getline(f, line)) {
      if (line.find('*') == std::string::npos) continue;  // not the active state
      auto u = line.find("Mhz");
      if (u == std::string::npos) u = line.find("MHz");
      if (u == std::string::npos) continue;
      size_t e = u;
      while (e > 0 && !std::isdigit(static_cast<unsigned char>(line[e - 1]))) --e;
      size_t b = e;
      while (b > 0 && std::isdigit(static_cast<unsigned char>(line[b - 1]))) --b;
      if (b < e) return std::atoi(line.substr(b, e - b).c_str());
    }
  }
  // A positive integer in `path`, divided by `per_mhz` (1e6 for Hz, 1e3 for kHz).
  auto read_scaled = [](const std::string& path, long per_mhz) {
    std::ifstream f(path);
    long v = 0;
    return (f && (f >> v) && v >= per_mhz) ? static_cast<int>(v / per_mhz) : -1;
  };
  for (const char* p : {"/sys/class/kgsl/kgsl-3d0/gpuclk",
                        "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq",
                        "/sys/devices/platform/gpu/devfreq/gpu/cur_freq",
                        "/sys/class/devfreq/gpu/cur_freq"}) {
    if (const int mhz = read_scaled(root + p, 1000000); mhz > 0) return mhz;
  }
  for (const char* p : {"/sys/class/misc/mali0/device/cur_freq", "/sys/kernel/gpu/gpu_clock"}) {
    if (const int mhz = read_scaled(root + p, 1000); mhz > 0) return mhz;
  }
  std::error_code ec;  // never throws: an unreadable /sys/class/devfreq just ends the scan
  for (std::filesystem::directory_iterator d(root + "/sys/class/devfreq", ec), end; !ec && d != end;
       d.increment(ec)) {
    if (d->path().filename().string().find("sgpu") == std::string::npos) continue;
    if (const int mhz = read_scaled(d->path().string() + "/cur_freq", 1000); mhz > 0) return mhz;
  }
  return -1;
}

// Samples read_gpu_clock_mhz() on its own thread every `period_ms` while a measured loop
// runs (design D5): the measuring thread blocks on the GPU, so a reading taken there would
// often see the idle clock, and on the Samsung the power-gated 0. stop() returns the median
// of the positive samples, -1 if there were none (no source, or the GPU never read busy).
class GpuClockSampler {
 public:
  struct Summary {
    int mhz = -1;  // median of the positive samples
    int min_mhz = -1;
    int max_mhz = -1;
    int samples = 0;  // positive samples taken
  };

  explicit GpuClockSampler(int period_ms)
      : thread_([this, period_ms] {
          std::unique_lock lock(mutex_);
          do {
            lock.unlock();
            const int mhz = read_gpu_clock_mhz();
            lock.lock();
            if (mhz > 0) samples_.push_back(mhz);
          } while (
              !cv_.wait_for(lock, std::chrono::milliseconds(period_ms), [this] { return stop_; }));
        }) {}
  GpuClockSampler(const GpuClockSampler&) = delete;
  GpuClockSampler& operator=(const GpuClockSampler&) = delete;
  ~GpuClockSampler() { (void)stop(); }

  Summary stop() {
    {
      const std::lock_guard lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    Summary s;
    if (samples_.empty()) return s;
    std::vector<int> v = samples_;
    std::sort(v.begin(), v.end());
    s.mhz = v[v.size() / 2];
    s.min_mhz = v.front();
    s.max_mhz = v.back();
    s.samples = static_cast<int>(v.size());
    return s;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::vector<int> samples_;
  std::thread thread_;  // last: starts once the members above exist
};

}  // namespace bt_prof
