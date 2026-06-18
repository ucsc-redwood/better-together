#pragma once
// ---------------------------------------------------------------------------
// bm_fully_common -- shared driver for the "normal vs fully-occupied" per-stage
// benchmark (single-PU latency vs latency when all 4 PUs hammer the same stage).
// The 6 pipe/<cell>/bm_fully_vs_normal.* were ~95% identical forks; the only
// per-cell variation is the OMP dispatch namespace, the GPU's ProcessorType +
// BmTable column, and the timer (wall-clock for Vulkan, cudaEvent for CUDA).
//
// Each cell's bm_fully_vs_normal.{cpp,cu} is now a thin main() that supplies its
// types (via const.hpp, included first) + those four knobs. Output is an analysis
// artifact (dump_tables_for_python / print_normal_benchmark_table); no script
// consumes it, so this is a behavior-preserving refactor.
// ---------------------------------------------------------------------------

#include <omp.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>

#include "builtin-apps/app.hpp"  // ProcessorType, g_*_cores, has_*_cores, get_cpu_cores_by_type
#include "table.hpp"

#ifdef __CUDACC__
#include "builtin-apps/common/cuda/helpers.cuh"  // CheckCuda (CudaEventTimer)
#endif

namespace bt_fully {

// ---- Timers ---------------------------------------------------------------
// Wall-clock (Vulkan + CPU-tier cells). Integer-ms cast to match the original.
struct WallTimer {
  std::chrono::high_resolution_clock::time_point t0_;
  void start() { t0_ = std::chrono::high_resolution_clock::now(); }
  [[nodiscard]] double stop_ms() const {
    const auto t1 = std::chrono::high_resolution_clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0_).count());
  }
};

#ifdef __CUDACC__
// CUDA-event timer (the cu cells) -- GPU-accurate, matches the original cu path.
// (The old code created but never destroyed the events; this frees them.)
struct CudaEventTimer {
  cudaEvent_t start_{}, stop_{};
  CudaEventTimer() {
    CheckCuda(cudaEventCreate(&start_));
    CheckCuda(cudaEventCreate(&stop_));
  }
  ~CudaEventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void start() { CheckCuda(cudaEventRecord(start_, 0)); }
  [[nodiscard]] double stop_ms() {
    CheckCuda(cudaEventRecord(stop_, 0));
    CheckCuda(cudaEventSynchronize(stop_));
    float ms = 0;
    CheckCuda(cudaEventElapsedTime(&ms, start_, stop_));
    return static_cast<double>(ms);
  }
};
#endif

// ---- Normal benchmark (one PU at a time) ----------------------------------
inline void run_normal_impl(LocalQueue& q,
                            const std::function<void(AppDataT*)>& func,
                            const int seconds_to_run) {
  std::atomic<bool> done = false;

  std::thread t1([&]() {
    while (!done.load(std::memory_order_relaxed)) {
      AppDataT* app = q.front();
      q.pop();
      func(app);
      q.push(app);  // After done -> push back for reuse
    }
  });

  std::this_thread::sleep_for(std::chrono::seconds(seconds_to_run));
  done.store(true);
  t1.join();
}

template <class Timer, class OmpDispatch>
inline void run_normal(BmTable<kNumStages>& table,
                       const ProcessorType gpu_pt,
                       const OmpDispatch& omp_dispatch,
                       const ProcessorType pt,
                       const int stage,
                       const int seconds_to_run,
                       const bool print_progress) {
  DispatcherT disp;
  const std::vector<AppDataPtr> dataset = make_dataset<AppDataT>(disp, kPoolSize);

  const auto cores_to_use_opt = get_cpu_cores_by_type(pt);
  if (pt != gpu_pt && cores_to_use_opt.has_value() && cores_to_use_opt->empty()) {
    SPDLOG_WARN("No cores to use for processor type: {}", static_cast<int>(pt));
    return;
  }

  LocalQueue q = make_queue_from_vector<LocalQueue>(dataset);

  Timer timer;
  timer.start();
  int total_processed = 0;

  if (pt == gpu_pt) {
    run_normal_impl(
        q,
        [&](AppDataT* app) {
          disp.dispatch_multi_stage(*app, stage, stage);
          total_processed++;
        },
        seconds_to_run);
  } else if (cores_to_use_opt.has_value()) {
    const auto cores_to_use = cores_to_use_opt.value();
    run_normal_impl(
        q,
        [&](AppDataT* app) {
          omp_dispatch(cores_to_use, cores_to_use.size(), *app, stage, stage);
          total_processed++;
        },
        seconds_to_run);
  } else {
    SPDLOG_WARN("Skipping processor type {} - no cores available", static_cast<int>(pt));
  }

  const double total_time = timer.stop_ms();

  if (print_progress) {
    const std::string pt_name = pt == ProcessorType::kLittleCore   ? "Little"
                                : pt == ProcessorType::kBigCore    ? "Big"
                                : pt == ProcessorType::kMediumCore ? "Medium"
                                : pt == ProcessorType::kVulkan     ? "Vulkan"
                                : pt == ProcessorType::kCuda       ? "CUDA"
                                                                   : "Unknown";
    fmt::print("Stage: {} by {}\n", stage, pt_name);
    fmt::print("\tCount \t{}\n", total_processed);
    fmt::print("\tAverage \t{:.4f} ms\n", total_time / total_processed);
    std::fflush(stdout);
  }

  if (total_processed > 0) {
    table.update_normal_table(stage - 1, static_cast<int>(pt), total_time / total_processed);
  }
}

// ---- Fully-occupied benchmark (all PUs hammer the same stage) --------------
template <class Timer, class OmpDispatch>
inline void run_fully(BmTable<kNumStages>& table,
                      const OmpDispatch& omp_dispatch,
                      const int gpu_col,
                      const char* gpu_name,
                      const int stage,
                      const int seconds_to_run,
                      const bool print_progress) {
  DispatcherT disp;
  const std::vector<AppDataPtr> dataset = make_dataset<AppDataT>(disp, kPoolSize);

  // Each PU thread gets a DISJOINT slice of the pool. Aliasing the same AppData
  // into every queue let a CPU thread and the GPU thread mutate the same tree
  // buffers concurrently -- a race tree's data-dependent stages can't tolerate.
  const size_t per = kPoolSize / 4;
  LocalQueue q_0, q_1, q_2, q_3;
  for (size_t i = 0 * per; i < 1 * per; ++i) q_0.push(dataset[i].get());
  for (size_t i = 1 * per; i < 2 * per; ++i) q_1.push(dataset[i].get());
  for (size_t i = 2 * per; i < 3 * per; ++i) q_2.push(dataset[i].get());
  for (size_t i = 3 * per; i < 4 * per; ++i) q_3.push(dataset[i].get());

  std::atomic<int> lit_processed(0);
  std::atomic<int> med_processed(0);
  std::atomic<int> big_processed(0);
  std::atomic<int> gpu_processed(0);

  std::vector<std::thread> threads;
  std::atomic<bool> done = false;

  Timer timer;
  timer.start();

  auto cpu_thread = [&](LocalQueue& q, std::vector<int>& cores, std::atomic<int>& counter) {
    while (!done.load(std::memory_order_relaxed)) {
      AppDataT* app = q.front();
      q.pop();
      omp_dispatch(cores, cores.size(), *app, stage, stage);
      counter++;
      q.push(app);
    }
  };

  if (has_lit_cores()) threads.emplace_back(cpu_thread, std::ref(q_0), std::ref(g_lit_cores),
                                            std::ref(lit_processed));
  if (has_med_cores()) threads.emplace_back(cpu_thread, std::ref(q_1), std::ref(g_med_cores),
                                            std::ref(med_processed));
  if (has_big_cores()) threads.emplace_back(cpu_thread, std::ref(q_2), std::ref(g_big_cores),
                                            std::ref(big_processed));

  // Always create the GPU thread.
  threads.emplace_back([&]() {
    while (!done.load(std::memory_order_relaxed)) {
      AppDataT* app = q_3.front();
      q_3.pop();
      disp.dispatch_multi_stage(*app, stage, stage);
      gpu_processed++;
      q_3.push(app);
    }
  });

  std::this_thread::sleep_for(std::chrono::seconds(seconds_to_run));
  done.store(true);
  for (auto& t : threads) t.join();

  const double total_time = timer.stop_ms();

  const auto lit_count = lit_processed.load();
  const auto med_count = med_processed.load();
  const auto big_count = big_processed.load();
  const auto gpu_count = gpu_processed.load();

  const auto lit_time = total_time / lit_count;
  const auto med_time = total_time / med_count;
  const auto big_time = total_time / big_count;
  const auto gpu_time = total_time / gpu_count;

  if (print_progress) {
    fmt::print("Stage: {}\n", stage);
    fmt::print("\tLittle \t{:.4f} ms \t({})\n", lit_time, lit_count);
    fmt::print("\tMedium \t{:.4f} ms \t({})\n", med_time, med_count);
    fmt::print("\tBig    \t{:.4f} ms \t({})\n", big_time, big_count);
    fmt::print("\t{} \t{:.4f} ms \t({})\n", gpu_name, gpu_time, gpu_count);
    std::fflush(stdout);
  }

  if (lit_count > 0) table.update_full_table(stage - 1, 0, lit_time);
  if (med_count > 0) table.update_full_table(stage - 1, 1, med_time);
  if (big_count > 0) table.update_full_table(stage - 1, 2, big_time);
  if (gpu_count > 0) table.update_full_table(stage - 1, gpu_col, gpu_time);
}

// ---- main ------------------------------------------------------------------
// gpu_pt/gpu_col/gpu_name select the GPU backend (Vulkan col 3 / CUDA col 4);
// Timer is WallTimer (vk) or CudaEventTimer (cu); omp_dispatch is the cell's
// <app>::omp::dispatch_multi_stage(cores, n, app, lo, hi).
template <class Timer, class OmpDispatch>
inline int run(int argc,
               char** argv,
               const ProcessorType gpu_pt,
               const int gpu_col,
               const char* gpu_name,
               const OmpDispatch& omp_dispatch) {
  PARSE_ARGS_BEGIN

  int start_stage = 1;
  int end_stage = static_cast<int>(kNumStages);
  int seconds_to_run = 10;
  bool print_progress = false;

  app.add_option("-s, --start-stage", start_stage, "Start stage");
  app.add_option("-e, --end-stage", end_stage, "End stage");
  app.add_option("-t, --seconds-to-run", seconds_to_run, "Seconds to run for normal benchmark");
  app.add_flag("-p, --print-progress", print_progress, "Print progress");

  PARSE_ARGS_END

  BmTable<kNumStages> table;
  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  spdlog::info("Running normal benchmark (one processor at a time)...");
  for (int stage = start_stage; stage <= end_stage; stage++) {
    if (has_lit_cores())
      run_normal<Timer>(table, gpu_pt, omp_dispatch, ProcessorType::kLittleCore, stage,
                        seconds_to_run, print_progress);
    if (has_med_cores())
      run_normal<Timer>(table, gpu_pt, omp_dispatch, ProcessorType::kMediumCore, stage,
                        seconds_to_run, print_progress);
    if (has_big_cores())
      run_normal<Timer>(table, gpu_pt, omp_dispatch, ProcessorType::kBigCore, stage,
                        seconds_to_run, print_progress);
    run_normal<Timer>(table, gpu_pt, omp_dispatch, gpu_pt, stage, seconds_to_run, print_progress);
  }

  spdlog::info("Running fully benchmark (each processor in isolation, all stages active)...");
  for (int stage = start_stage; stage <= end_stage; stage++) {
    run_fully<Timer>(table, omp_dispatch, gpu_col, gpu_name, stage, seconds_to_run, print_progress);
  }

  table.print_normal_benchmark_table(start_stage, end_stage);
  table.dump_tables_for_python(start_stage, end_stage);
  return 0;
}

}  // namespace bt_fully
