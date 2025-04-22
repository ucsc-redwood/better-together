#include <omp.h>

#include "../table.hpp"
#include "builtin-apps/app.hpp"  // for 'g_big_cores', 'g_med_cores', 'g_lit_cores'
#include "const.hpp"

// ----------------------------------------------------------------------------
// Clean Benchmark
// ----------------------------------------------------------------------------

static void BM_run_normal_impl(LocalQueue& q,
                               std::function<void(AppDataT*)> func,
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

  done.store(true);  // Signal all threads to stop

  t1.join();
}

static void BM_run_normal(BmTable<kNumStages>& table,
                          const ProcessorType pt,
                          const int stage,
                          const int seconds_to_run,
                          const bool print_progress = false) {
  DispatcherT disp;

  const std::vector<AppDataPtr> dataset = make_dataset(disp, kPoolSize);

  const auto cores_to_use_opt = get_cpu_cores_by_type(pt);

  if (pt != ProcessorType::kCuda && cores_to_use_opt.has_value() && cores_to_use_opt->empty()) {
    SPDLOG_WARN("No cores to use for processor type: {}", static_cast<int>(pt));
    return;
  }

  // prepare the queue
  LocalQueue q = make_queue_from_vector(dataset);

  cudaEvent_t start, stop;
  CheckCuda(cudaEventCreate(&start));
  CheckCuda(cudaEventCreate(&stop));
  CheckCuda(cudaEventRecord(start, 0));

  int total_processed = 0;

  // ----------------------------------------------------------------------------

  if (pt == ProcessorType::kCuda) {
    BM_run_normal_impl(
        q,
        [&](AppDataT* app) {
          disp.dispatch_multi_stage(*app, stage, stage);
          total_processed++;
        },
        seconds_to_run);
  } else if (cores_to_use_opt.has_value()) {
    const auto cores_to_use = cores_to_use_opt.value();

    BM_run_normal_impl(
        q,
        [&](AppDataT* app) {
          cifar_dense::omp::dispatch_multi_stage(
              cores_to_use, cores_to_use.size(), *app, stage, stage);
          total_processed++;
        },
        seconds_to_run);
  } else {
    SPDLOG_WARN("Skipping processor type {} - no cores available", static_cast<int>(pt));
  }

  // ----------------------------------------------------------------------------

  CheckCuda(cudaEventRecord(stop, 0));
  CheckCuda(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CheckCuda(cudaEventElapsedTime(&milliseconds, start, stop));

  const auto total_time = static_cast<double>(milliseconds);

  // map pt to name
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

// ----------------------------------------------------------------------------
// Fully Occupied Benchmark
// ----------------------------------------------------------------------------

static void BM_run_fully(BmTable<kNumStages>& table,
                         const int stage,
                         const int seconds_to_run,
                         const bool print_progress = false) {
  DispatcherT disp;

  const std::vector<AppDataPtr> dataset = make_dataset(disp, kPoolSize);

  LocalQueue q_0 = make_queue_from_vector(dataset);
  LocalQueue q_1 = make_queue_from_vector(dataset);
  LocalQueue q_2 = make_queue_from_vector(dataset);
  LocalQueue q_3 = make_queue_from_vector(dataset);

  // Use atomic counters for task tracking, but we'll only report on the one we're measuring
  std::atomic<int> lit_processed(0);
  std::atomic<int> med_processed(0);
  std::atomic<int> big_processed(0);
  std::atomic<int> cud_processed(0);

  std::vector<std::thread> threads;

  cudaEvent_t start, stop;
  CheckCuda(cudaEventCreate(&start));
  CheckCuda(cudaEventCreate(&stop));
  CheckCuda(cudaEventRecord(start, 0));

  // ----------------------------------------------------------------------------
  std::atomic<bool> done = false;

  if (has_lit_cores()) {
    threads.emplace_back([&]() {
      while (!done.load(std::memory_order_relaxed)) {
        AppDataT* app = q_0.front();
        q_0.pop();

        cifar_dense::omp::dispatch_multi_stage(LITTLE_CORES, *app, stage, stage);
        lit_processed++;

        q_0.push(app);
      }
    });
  }

  if (has_med_cores()) {
    threads.emplace_back([&]() {
      while (!done.load(std::memory_order_relaxed)) {
        AppDataT* app = q_1.front();
        q_1.pop();

        cifar_dense::omp::dispatch_multi_stage(MEDIUM_CORES, *app, stage, stage);
        med_processed++;

        q_1.push(app);
      }
    });
  }

  if (has_big_cores()) {
    threads.emplace_back([&]() {
      while (!done.load(std::memory_order_relaxed)) {
        AppDataT* app = q_2.front();
        q_2.pop();

        cifar_dense::omp::dispatch_multi_stage(BIG_CORES, *app, stage, stage);
        big_processed++;

        q_2.push(app);
      }
    });
  }

  // Always create Vulkan thread
  threads.emplace_back([&]() {
    while (!done.load(std::memory_order_relaxed)) {
      AppDataT* app = q_3.front();
      q_3.pop();

      disp.dispatch_multi_stage(*app, stage, stage);
      cud_processed++;

      q_3.push(app);
    }
  });

  std::this_thread::sleep_for(std::chrono::seconds(seconds_to_run));

  done.store(true);  // Signal all threads to stop

  for (auto& t : threads) t.join();

  // ----------------------------------------------------------------------------

  CheckCuda(cudaEventRecord(stop, 0));
  CheckCuda(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CheckCuda(cudaEventElapsedTime(&milliseconds, start, stop));

  const auto total_time = static_cast<double>(milliseconds);

  // Print human readable time to console
  const auto lit_count = lit_processed.load();
  const auto med_count = med_processed.load();
  const auto big_count = big_processed.load();
  const auto cud_count = cud_processed.load();

  const auto lit_time = total_time / lit_count;
  const auto med_time = total_time / med_count;
  const auto big_time = total_time / big_count;
  const auto cud_time = total_time / cud_count;

  if (print_progress) {
    fmt::print("Stage: {}\n", stage);
    fmt::print("\tLittle \t{:.4f} ms \t({})\n", lit_time, lit_count);
    fmt::print("\tMedium \t{:.4f} ms \t({})\n", med_time, med_count);
    fmt::print("\tBig    \t{:.4f} ms \t({})\n", big_time, big_count);
    fmt::print("\tCUDA   \t{:.4f} ms \t({})\n", cud_time, cud_count);
    std::fflush(stdout);
  }

  // update the table
  if (lit_count > 0) table.update_full_table(stage - 1, 0, lit_time);
  if (med_count > 0) table.update_full_table(stage - 1, 1, med_time);
  if (big_count > 0) table.update_full_table(stage - 1, 2, big_time);
  if (cud_count > 0) table.update_full_table(stage - 1, 4, cud_time);
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN

  int start_stage = 1;
  int end_stage = kNumStages;
  int seconds_to_run = 10;
  bool print_progress = false;

  app.add_option("-s, --start-stage", start_stage, "Start stage");
  app.add_option("-e, --end-stage", end_stage, "End stage");
  app.add_option("-t, --seconds-to-run", seconds_to_run, "Seconds to run for normal benchmark");
  app.add_flag("-p, --print-progress", print_progress, "Print progress");

  PARSE_ARGS_END

  BmTable<kNumStages> table;

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  // ----------------------------------------------------------------------------
  // Run normal benchmark (one processor type at a time)
  // ----------------------------------------------------------------------------

  spdlog::info("Running normal benchmark (one processor at a time)...");

  for (int stage = start_stage; stage <= end_stage; stage++) {
    if (has_lit_cores()) {
      BM_run_normal(table, ProcessorType::kLittleCore, stage, seconds_to_run, print_progress);
    }

    if (has_med_cores()) {
      BM_run_normal(table, ProcessorType::kMediumCore, stage, seconds_to_run, print_progress);
    }

    if (has_big_cores()) {
      BM_run_normal(table, ProcessorType::kBigCore, stage, seconds_to_run, print_progress);
    }

    BM_run_normal(table, ProcessorType::kCuda, stage, seconds_to_run, print_progress);
  }

  // ----------------------------------------------------------------------------
  // Run fully benchmark (each processor in isolation, all stages active)
  // ----------------------------------------------------------------------------

  spdlog::info("Running fully benchmark (each processor in isolation, all stages active)...");

  for (int stage = start_stage; stage <= end_stage; stage++) {
    BM_run_fully(table, stage, seconds_to_run, print_progress);
  }

  // ----------------------------------------------------------------------------
  // Print and dump tables
  // ----------------------------------------------------------------------------

  table.print_normal_benchmark_table(start_stage, end_stage);

  table.dump_tables_for_python(start_stage, end_stage);

  return 0;
}
