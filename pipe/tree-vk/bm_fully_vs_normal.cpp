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

  if (pt != ProcessorType::kVulkan && cores_to_use_opt.has_value() && cores_to_use_opt->empty()) {
    SPDLOG_WARN("No cores to use for processor type: {}", static_cast<int>(pt));
    return;
  }

  // prepare the queue
  LocalQueue q = make_queue_from_vector(dataset);

  const auto start = std::chrono::high_resolution_clock::now();

  int total_processed = 0;

  // ----------------------------------------------------------------------------

  if (pt == ProcessorType::kVulkan) {
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
          tree::omp::dispatch_multi_stage(cores_to_use, cores_to_use.size(), *app, stage, stage);
          total_processed++;
        },
        seconds_to_run);
  } else {
    SPDLOG_WARN("Skipping processor type {} - no cores available", static_cast<int>(pt));
  }

  // ----------------------------------------------------------------------------

  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  const auto total_time = static_cast<double>(duration.count());

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
  std::atomic<int> vul_processed(0);

  std::vector<std::thread> threads;

  const auto start = std::chrono::high_resolution_clock::now();

  // ----------------------------------------------------------------------------
  std::atomic<bool> done = false;

  if (has_lit_cores()) {
    threads.emplace_back([&]() {
      while (!done.load(std::memory_order_relaxed)) {
        AppDataT* app = q_0.front();
        q_0.pop();

        tree::omp::dispatch_multi_stage(LITTLE_CORES, *app, stage, stage);
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

        tree::omp::dispatch_multi_stage(MEDIUM_CORES, *app, stage, stage);
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

        tree::omp::dispatch_multi_stage(BIG_CORES, *app, stage, stage);
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
      vul_processed++;

      q_3.push(app);
    }
  });

  std::this_thread::sleep_for(std::chrono::seconds(seconds_to_run));

  done.store(true);  // Signal all threads to stop

  for (auto& t : threads) t.join();

  // ----------------------------------------------------------------------------

  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  const auto total_time = static_cast<double>(duration.count());

  // Print human readable time to console
  const auto lit_count = lit_processed.load();
  const auto med_count = med_processed.load();
  const auto big_count = big_processed.load();
  const auto vul_count = vul_processed.load();

  const auto lit_time = total_time / lit_count;
  const auto med_time = total_time / med_count;
  const auto big_time = total_time / big_count;
  const auto vul_time = total_time / vul_count;

  if (print_progress) {
    fmt::print("Stage: {}\n", stage);
    fmt::print("\tLittle \t{:.4f} ms \t({})\n", lit_time, lit_count);
    fmt::print("\tMedium \t{:.4f} ms \t({})\n", med_time, med_count);
    fmt::print("\tBig    \t{:.4f} ms \t({})\n", big_time, big_count);
    fmt::print("\tVulkan \t{:.4f} ms \t({})\n", vul_time, vul_count);
    std::fflush(stdout);
  }

  // update the table
  if (lit_count > 0) table.update_full_table(stage - 1, 0, lit_time);
  if (med_count > 0) table.update_full_table(stage - 1, 1, med_time);
  if (big_count > 0) table.update_full_table(stage - 1, 2, big_time);
  if (vul_count > 0) table.update_full_table(stage - 1, 3, vul_time);
}

// ----------------------------------------------------------------------------
// Main
//
// Run with (on Android)
// xmake r bm-fully-tree-vk --device 3A021JEHN02756 -p -t 10
//
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

    BM_run_normal(table, ProcessorType::kVulkan, stage, seconds_to_run, print_progress);
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

// // ----------------------------------------------------------------------------
// // Queue
// // ----------------------------------------------------------------------------

// std::atomic<bool> done = false;

// void reset_done_flag() { done.store(false); }

// void init_q(LocalQueue* q, DispatcherT& disp) {
//   constexpr int kPhysicalTasks = 10;

//   for (int i = 0; i < kPhysicalTasks; i++) {
//     q->push(new TaskT(disp.get_mr()));
//   }
// }

// void clean_up_q(LocalQueue* q) {
//   while (!q->empty()) {
//     TaskT* task = q->front();
//     q->pop();
//     delete task;
//   }
// }

// void similuation_thread(LocalQueue* q, std::function<void(TaskT*)> func) {
//   while (!done.load(std::memory_order_relaxed)) {
//     TaskT* task = q->front();
//     q->pop();

//     func(task);

//     // After done -> push back for reuse
//     task->reset();
//     q->push(task);
//   }
// }

// // ----------------------------------------------------------------------------
// // Normal Benchmark
// // ----------------------------------------------------------------------------

// static void BM_run_normal(BmTable<kNumStages>& table,
//                           const ProcessorType pt,
//                           const int stage,
//                           const int seconds_to_run,
//                           const bool print_progress = false) {
//   // prepare the vulkan dispatcher
//   DispatcherT disp;

//   // prepare the cpu cores to use
//   std::vector<int> cores_to_use;
//   if (pt == ProcessorType::kLittleCore) {
//     cores_to_use = g_lit_cores;
//   } else if (pt == ProcessorType::kBigCore) {
//     cores_to_use = g_big_cores;
//   } else if (pt == ProcessorType::kMediumCore) {
//     cores_to_use = g_med_cores;
//   }

//   if (pt != ProcessorType::kVulkan && cores_to_use.empty()) {
//     SPDLOG_WARN("No cores to use for processor type: {}", static_cast<int>(pt));
//     return;
//   }

//   // prepare the queue
//   LocalQueue q;
//   init_q(&q, disp);

//   reset_done_flag();

//   auto start = std::chrono::high_resolution_clock::now();

//   // ----------------------------------------------------------------------------

//   std::thread t1;
//   int total_processed = 0;  // Initialize here, now explicitly shown for clarity

//   if (pt == ProcessorType::kVulkan) {
//     // launch gpu thread
//     t1 = std::thread(similuation_thread, &q, [disp = &disp, &total_processed, stage](TaskT* task)
//     {
//       disp->dispatch_multi_stage(task->appdata, stage, stage);
//       total_processed++;
//     });
//   } else {
//     // launch cpu thread
//     t1 = std::thread(similuation_thread, &q, [cores_to_use, &total_processed, stage](TaskT* task)
//     {
//       tree::omp::dispatch_multi_stage(
//           cores_to_use, cores_to_use.size(), task->appdata, stage, stage);
//       total_processed++;
//     });
//   }

//   std::this_thread::sleep_for(std::chrono::seconds(seconds_to_run));

//   done.store(true);  // Signal all threads to stop

//   t1.join();

//   // ----------------------------------------------------------------------------

//   auto end = std::chrono::high_resolution_clock::now();
//   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

//   const auto total_time = static_cast<double>(duration.count());

//   // map pt to name
//   if (print_progress) {
//     const std::string pt_name = pt == ProcessorType::kLittleCore   ? "Little"
//                                 : pt == ProcessorType::kBigCore    ? "Big"
//                                 : pt == ProcessorType::kMediumCore ? "Medium"
//                                 : pt == ProcessorType::kVulkan     ? "Vulkan"
//                                                                    : "Unknown";

//     fmt::print("Stage: {} by {}\n", stage, pt_name);
//     fmt::print("\tCount \t{}\n", total_processed);
//     fmt::print("\tAverage \t{:.4f} ms\n", total_time / total_processed);
//     std::fflush(stdout);
//   }

//   clean_up_q(&q);

//   // map ProcessorType::kLittleCore = 0, ProcessorType::kBigCore = 1, ProcessorType::kMediumCore
//   =
//   // 2, ProcessorType::kVulkan = 3
//   if (total_processed > 0) {
//     table.update_normal_table(
//         stage - 1, static_cast<int>(pt), static_cast<double>(duration.count()) /
//         total_processed);
//   }
// }

// // ----------------------------------------------------------------------------
// // Fully Occupied Benchmark
// // ----------------------------------------------------------------------------

// static void BM_run_fully(BmTable<kNumStages>& table,
//                          const int stage,
//                          const int seconds_to_run,
//                          const bool print_progress = false) {
//   DispatcherT disp;

//   reset_done_flag();

//   LocalQueue q_0;
//   LocalQueue q_1;
//   LocalQueue q_2;
//   LocalQueue q_3;

//   init_q(&q_0, disp);
//   init_q(&q_1, disp);
//   init_q(&q_2, disp);
//   init_q(&q_3, disp);

//   // Use atomic counters for task tracking, but we'll only report on the one we're measuring
//   std::atomic<int> lit_processed(0);
//   std::atomic<int> med_processed(0);
//   std::atomic<int> big_processed(0);
//   std::atomic<int> vuk_processed(0);

//   std::vector<std::thread> threads;

//   auto start = std::chrono::high_resolution_clock::now();

//   // ----------------------------------------------------------------------------

//   if (!g_lit_cores.empty()) {
//     threads.emplace_back(similuation_thread, &q_0, [&lit_processed, stage](TaskT* task) {
//       tree::omp::dispatch_multi_stage(LITTLE_CORES, task->appdata, stage, stage);

//       lit_processed++;
//     });
//   }

//   if (!g_med_cores.empty()) {
//     threads.emplace_back(similuation_thread, &q_1, [&med_processed, stage](TaskT* task) {
//       tree::omp::dispatch_multi_stage(MEDIUM_CORES, task->appdata, stage, stage);

//       med_processed++;
//     });
//   }

//   if (!g_big_cores.empty()) {
//     threads.emplace_back(similuation_thread, &q_2, [&big_processed, stage](TaskT* task) {
//       tree::omp::dispatch_multi_stage(BIG_CORES, task->appdata, stage, stage);

//       big_processed++;
//     });
//   }

//   // Always create Vulkan thread
//   threads.emplace_back(
//       similuation_thread, &q_3, [disp = &disp, &vuk_processed, stage](TaskT* task) {
//         disp->dispatch_multi_stage(task->appdata, stage, stage);
//         vuk_processed++;
//       });

//   std::this_thread::sleep_for(std::chrono::seconds(seconds_to_run));

//   done.store(true);  // Signal all threads to stop

//   for (auto& t : threads) t.join();

//   // ----------------------------------------------------------------------------

//   auto end = std::chrono::high_resolution_clock::now();
//   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

//   clean_up_q(&q_0);
//   clean_up_q(&q_1);
//   clean_up_q(&q_2);
//   clean_up_q(&q_3);

//   // Print human readable time to console
//   const auto lit_count = lit_processed.load();
//   const auto med_count = med_processed.load();
//   const auto big_count = big_processed.load();
//   const auto vuk_count = vuk_processed.load();

//   const auto lit_time = static_cast<double>(duration.count()) / lit_count;
//   const auto med_time = static_cast<double>(duration.count()) / med_count;
//   const auto big_time = static_cast<double>(duration.count()) / big_count;
//   const auto vuk_time = static_cast<double>(duration.count()) / vuk_count;

//   if (print_progress) {
//     fmt::print("Stage: {}\n", stage);
//     fmt::print("\tLittle \t{:.4f} ms \t({})\n", lit_time, lit_count);
//     fmt::print("\tMedium \t{:.4f} ms \t({})\n", med_time, med_count);
//     fmt::print("\tBig    \t{:.4f} ms \t({})\n", big_time, big_count);
//     fmt::print("\tVulkan \t{:.4f} ms \t({})\n", vuk_time, vuk_count);
//     std::fflush(stdout);
//   }

//   // update the table
//   if (lit_count > 0) table.update_full_table(stage - 1, 0, lit_time);
//   if (med_count > 0) table.update_full_table(stage - 1, 1, med_time);
//   if (big_count > 0) table.update_full_table(stage - 1, 2, big_time);
//   if (vuk_count > 0) table.update_full_table(stage - 1, 3, vuk_time);
// }

// // ----------------------------------------------------------------------------
// // Main
// // ----------------------------------------------------------------------------

// int main(int argc, char** argv) {
//   PARSE_ARGS_BEGIN

//   int start_stage = 1;
//   int end_stage = kNumStages;
//   int seconds_to_run = 10;
//   bool print_progress = false;

//   app.add_option("-s, --start-stage", start_stage, "Start stage");
//   app.add_option("-e, --end-stage", end_stage, "End stage");
//   app.add_option("-t, --seconds-to-run", seconds_to_run, "Seconds to run for normal benchmark");
//   app.add_flag("-p, --print-progress", print_progress, "Print progress");

//   PARSE_ARGS_END

//   BmTable<kNumStages> table;

//   spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

//   // ----------------------------------------------------------------------------
//   // Run normal benchmark (one processor type at a time)
//   // ----------------------------------------------------------------------------

//   spdlog::info("Running normal benchmark (one processor at a time)...");

//   for (int stage = start_stage; stage <= end_stage; stage++) {
//     if (!g_lit_cores.empty()) {
//       BM_run_normal(table, ProcessorType::kLittleCore, stage, seconds_to_run, print_progress);
//     }
//     if (!g_med_cores.empty()) {
//       BM_run_normal(table, ProcessorType::kMediumCore, stage, seconds_to_run, print_progress);
//     }
//     if (!g_big_cores.empty()) {
//       BM_run_normal(table, ProcessorType::kBigCore, stage, seconds_to_run, print_progress);
//     }
//     BM_run_normal(table, ProcessorType::kVulkan, stage, seconds_to_run, print_progress);
//   }

//   // ----------------------------------------------------------------------------
//   // Run fully benchmark (each processor in isolation, all stages active)
//   // ----------------------------------------------------------------------------

//   spdlog::info("Running fully benchmark (each processor in isolation, all stages active)...");

//   for (int stage = start_stage; stage <= end_stage; stage++) {
//     BM_run_fully(table, stage, seconds_to_run, print_progress);
//   }

//   // ----------------------------------------------------------------------------
//   // Print and dump tables
//   // ----------------------------------------------------------------------------

//   table.print_normal_benchmark_table(start_stage, end_stage);

//   table.dump_tables_for_python(start_stage, end_stage);

//   return 0;
// }
