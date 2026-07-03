// bm-runtime-null-cu -- EPCC-style NULL-KERNEL microbenchmark for the runtime.
//
// The pipeline machinery runs unchanged (SPSC ring, pool, two chunk workers), but
// the "kernels" are empty, so EVERYTHING measured is pure framework tax. Two chunks:
//   chunk0 (GPU driver thread): kStagesGpu empty kernel launches + one sync
//   chunk1 (OMP on CPU):        a calibrated ~kCpuWorkUs spin under omp parallel
//
// What it answers, per BT_NULL_MODE:
//   dev-stage   cudaDeviceSynchronize after EVERY empty launch (tree's old pattern)
//   dev-chunk   one cudaDeviceSynchronize per chunk (dispatch_multi_stage today)
//   stream      launches on a private stream + one cudaStreamSynchronize per chunk
//   none        no GPU work at all -> the pure SPSC/worker/handoff floor
// BT_NULL_BLOCKING=1 additionally sets cudaDeviceScheduleBlockingSync, so the sync
// SLEEPS instead of spinning -- if chunk1's throughput rises, the spinning sync was
// stealing a CPU core from the OMP chunk (the CPU/GPU-overlap question).
//
// Output: one summary line per run -- steady-state period per task, per-chunk busy
// p50, handoff gap p50 -- plus a PYTHON_DATA csv block for scripting.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "runtime/spsc_queue.hpp"

namespace {

__global__ void k_null() {}

constexpr int kStagesGpu = 6;
constexpr size_t kPool = 8;
constexpr double kCpuWorkUs = 100.0;

struct NullTask {
  int id = 0;
  // per-task timestamps (ns), filled by the workers
  long long c0_start = 0, c0_end = 0, c1_start = 0, c1_end = 0;
};

inline long long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Calibrated spin: busy-loop for roughly `us` microseconds of real work on the
// calling thread team (what a small OMP stage looks like to the runtime).
inline void spin_us(double us) {
  const long long t0 = now_ns();
  const long long dur = static_cast<long long>(us * 1000.0);
  while (now_ns() - t0 < dur) {
  }
}

double pct(std::vector<long long>& v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return static_cast<double>(v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))]);
}

}  // namespace

int main(int argc, char** argv) {
  int n_tasks = 512;
  for (int i = 1; i < argc - 1; ++i) {
    if (std::strcmp(argv[i], "--tasks") == 0) n_tasks = std::atoi(argv[i + 1]);
  }
  const char* mode_env = std::getenv("BT_NULL_MODE");
  const std::string mode = mode_env ? mode_env : "dev-chunk";
  const bool blocking = std::getenv("BT_NULL_BLOCKING") != nullptr;

  if (blocking) cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  cudaFree(nullptr);  // init context outside the measured region

  cudaStream_t stream{};
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  std::vector<NullTask> pool(kPool);
  std::vector<NullTask> tasks(static_cast<size_t>(n_tasks));

  SPSCQueue<NullTask*, kPool> q01, q1out;
  // Pre-fill: producer side is the pool itself (classic gen-logs shape).
  SPSCQueue<NullTask*, kPool> qfree;
  for (auto& t : pool) qfree.enqueue(&t);

  auto gpu_chunk = [&](NullTask* t) {
    t->c0_start = now_ns();
    if (mode != "none") {
      for (int s = 0; s < kStagesGpu; ++s) {
        if (mode == "stream") {
          k_null<<<1, 32, 0, stream>>>();
        } else {
          k_null<<<1, 32>>>();
        }
        if (mode == "dev-stage") cudaDeviceSynchronize();
      }
      if (mode == "dev-chunk") cudaDeviceSynchronize();
      if (mode == "stream") cudaStreamSynchronize(stream);
    }
    t->c0_end = now_ns();
  };

  auto cpu_chunk = [&](NullTask* t) {
    t->c1_start = now_ns();
    // 4 threads on the 6 little cores: with w0 + main also alive, a SPINNING sync
    // in w0 visibly inflates c1_busy (core contention) while a sleeping one doesn't.
#pragma omp parallel num_threads(4)
    { spin_us(kCpuWorkUs); }
    t->c1_end = now_ns();
  };

  const long long bench_t0 = now_ns();

  std::thread w0([&] {
    for (int i = 0; i < n_tasks; ++i) {
      NullTask* t = nullptr;
      while (!qfree.dequeue(t)) std::this_thread::yield();
      t->id = i;
      gpu_chunk(t);
      while (!q01.enqueue(t)) std::this_thread::yield();
    }
  });
  std::thread w1([&] {
    for (int i = 0; i < n_tasks; ++i) {
      NullTask* t = nullptr;
      while (!q01.dequeue(t)) std::this_thread::yield();
      cpu_chunk(t);
      tasks[static_cast<size_t>(i)] = *t;  // snapshot before the slot is reused
      while (!qfree.enqueue(t)) std::this_thread::yield();
    }
  });
  w0.join();
  w1.join();

  const double wall_ms = static_cast<double>(now_ns() - bench_t0) / 1e6;

  // Steady state: drop the first quarter.
  std::vector<long long> c0_busy, c1_busy, period, handoff;
  for (size_t i = tasks.size() / 4 + 1; i < tasks.size(); ++i) {
    c0_busy.push_back(tasks[i].c0_end - tasks[i].c0_start);
    c1_busy.push_back(tasks[i].c1_end - tasks[i].c1_start);
    period.push_back(tasks[i].c1_end - tasks[i - 1].c1_end);
    handoff.push_back(tasks[i].c1_start - tasks[i].c0_end);
  }

  std::printf(
      "mode=%s blocking=%d tasks=%d wall_ms=%.2f | period_p50_us=%.2f "
      "c0_busy_p50_us=%.2f c1_busy_p50_us=%.2f handoff_p50_us=%.2f handoff_p99_us=%.2f\n",
      mode.c_str(),
      blocking ? 1 : 0,
      n_tasks,
      wall_ms,
      pct(period, 0.5) / 1e3,
      pct(c0_busy, 0.5) / 1e3,
      pct(c1_busy, 0.5) / 1e3,
      pct(handoff, 0.5) / 1e3,
      pct(handoff, 0.99) / 1e3);

  cudaStreamDestroy(stream);
  return 0;
}
