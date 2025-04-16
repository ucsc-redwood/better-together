#include <omp.h>

#include <queue>

#include "builtin-apps/app.hpp"  // for 'g_big_cores', 'g_medium_cores', 'g_little_cores'
#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/conf.hpp"
#include "builtin-apps/pipeline/task.hpp"
#include "common.hpp"

// ----------------------------------------------------------------------------
// Realistic Benchmark:  Stage
// ----------------------------------------------------------------------------

using MyTask = Task<cifar_sparse::v2::AppData>;

// Little: 0, Big: 2, Medium: 1, Vulkan: 3
constexpr int kLitIdx = 0;
constexpr int kMedIdx = 1;
constexpr int kBigIdx = 2;
constexpr int kVukIdx = 3;

constexpr int kNumStages = 9;

// ----------------------------------------------------------------------------
// Tables
// ----------------------------------------------------------------------------

// create a 2D table, kNumStages stage times 4 type of cores, initialize it with 0
// access by bm_table[stage][core_type] = value
std::array<std::array<double, 4>, kNumStages> bm_norm_table;
std::array<std::array<double, 4>, kNumStages> bm_full_table;

void init_tables() {
  for (int stage = 0; stage < kNumStages; stage++) {
    for (int processor = 0; processor < 4; processor++) {
      bm_norm_table[stage][processor] = 0.0;
      bm_full_table[stage][processor] = 0.0;
    }
  }
}

// ----------------------------------------------------------------------------
// Android
// ----------------------------------------------------------------------------

namespace android {

std::atomic<bool> done = false;

void reset_done_flag() { done.store(false); }

using MyQueue = std::queue<MyTask*>;

void init_q(MyQueue* q, cifar_sparse::vulkan::v2::VulkanDispatcher& disp) {
  constexpr int kPhysicalTasks = 10;

  for (int i = 0; i < kPhysicalTasks; i++) {
    q->push(new MyTask(disp.get_mr()));
  }
}

void clean_up_q(MyQueue* q) {
  while (!q->empty()) {
    MyTask* task = q->front();
    q->pop();
    delete task;
  }
}

void similuation_thread(MyQueue* q, std::function<void(MyTask*)> func) {
  while (!done.load(std::memory_order_relaxed)) {
    MyTask* task = q->front();
    q->pop();

    func(task);

    // After done -> push back for reuse
    task->reset();
    q->push(task);
  }
}

static void BM_run_normal(const ProcessorType pt, const int stage, const int seconds_to_run) {
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;

  // Reset the done flag before starting a new benchmark
  reset_done_flag();

  MyQueue q_little;
  init_q(&q_little, disp);

  MyQueue q_medium;
  init_q(&q_medium, disp);

  MyQueue q_big;
  init_q(&q_big, disp);

  MyQueue q_vulkan;
  init_q(&q_vulkan, disp);

  auto start = std::chrono::high_resolution_clock::now();

  int total_processed = 0;  // Initialize here, now explicitly shown for clarity

  auto gpu_func = [&disp, &total_processed, stage](MyTask* task) {
    disp.dispatch_multi_stage(task->appdata, stage, stage);
    total_processed++;
  };

  std::vector<int> cores_to_use;
  if (pt == ProcessorType::kLittleCore) {
    cores_to_use = g_little_cores;
  } else if (pt == ProcessorType::kBigCore) {
    cores_to_use = g_big_cores;
  } else if (pt == ProcessorType::kMediumCore) {
    cores_to_use = g_medium_cores;
  }

  auto cpu_func = [&total_processed, stage, cores_to_use](MyTask* task) {
    cifar_sparse::omp::v2::dispatch_multi_stage(
        cores_to_use, cores_to_use.size(), task->appdata, stage, stage);
    total_processed++;
  };

  std::thread t1;
  if (pt == ProcessorType::kVulkan) {
    if (cores_to_use.empty()) {
      // This is fine for Vulkan
    } else {
      SPDLOG_WARN("Unexpected cores for Vulkan processor type");
    }
    t1 = std::thread(similuation_thread, &q_vulkan, gpu_func);
  } else {
    if (cores_to_use.empty()) {
      SPDLOG_WARN("No cores to use for processor type: {}", static_cast<int>(pt));
      return;
    }
    t1 = std::thread(similuation_thread, &q_little, cpu_func);
  }

  std::this_thread::sleep_for(std::chrono::seconds(seconds_to_run));
  done.store(true);

  t1.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Stage: " << stage << std::endl;
  std::cout << "\tTotal processed: " << total_processed << std::endl;
  std::cout << "\tAverage time per task: " << duration.count() / total_processed << " ms"
            << std::endl;

  clean_up_q(&q_little);
  clean_up_q(&q_medium);
  clean_up_q(&q_big);
  clean_up_q(&q_vulkan);

  // update the table

  // map ProcessorType::kLittleCore = 0, ProcessorType::kBigCore = 1, ProcessorType::kMediumCore =
  // 2, ProcessorType::kVulkan = 3
  if (total_processed > 0) {
    bm_norm_table[stage - 1][static_cast<int>(pt)] =
        static_cast<double>(duration.count()) / total_processed;
  }
}

static void BM_run_fully(const int stage, const int seconds_to_run) {
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;

  reset_done_flag();

  MyQueue q_0;
  init_q(&q_0, disp);

  MyQueue q_1;
  init_q(&q_1, disp);

  MyQueue q_2;
  init_q(&q_2, disp);

  MyQueue q_3;
  init_q(&q_3, disp);

  auto start = std::chrono::high_resolution_clock::now();

  // Use atomic counters for task tracking, but we'll only report on the one we're measuring
  std::atomic<int> lit_processed(0);
  std::atomic<int> med_processed(0);
  std::atomic<int> big_processed(0);
  std::atomic<int> vuk_processed(0);

  // Create GPU task function
  auto gpu_func = [&disp, &vuk_processed, stage](MyTask* task) {
    disp.dispatch_multi_stage(task->appdata, stage, stage);
    vuk_processed++;
  };

  // Create CPU task functions for each core type
  auto lit_func = [&lit_processed, stage](MyTask* task) {
    if (g_little_cores.empty()) return;

    cifar_sparse::omp::v2::dispatch_multi_stage(
        g_little_cores, g_little_cores.size(), task->appdata, stage, stage);

    lit_processed++;
  };

  auto med_func = [&med_processed, stage](MyTask* task) {
    if (g_medium_cores.empty()) return;

    cifar_sparse::omp::v2::dispatch_multi_stage(
        g_medium_cores, g_medium_cores.size(), task->appdata, stage, stage);

    med_processed++;
  };

  auto big_func = [&big_processed, stage](MyTask* task) {
    if (g_big_cores.empty()) return;

    cifar_sparse::omp::v2::dispatch_multi_stage(
        g_big_cores, g_big_cores.size(), task->appdata, stage, stage);

    big_processed++;
  };

  std::vector<std::thread> threads;

  if (!g_little_cores.empty()) threads.emplace_back(similuation_thread, &q_0, lit_func);
  if (!g_medium_cores.empty()) threads.emplace_back(similuation_thread, &q_1, med_func);
  if (!g_big_cores.empty()) threads.emplace_back(similuation_thread, &q_2, big_func);

  threads.emplace_back(similuation_thread, &q_3, gpu_func);  // Always create Vulkan thread

  std::this_thread::sleep_for(std::chrono::seconds(seconds_to_run));

  done.store(true);  // Signal all threads to stop

  for (auto& t : threads) {
    t.join();
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  clean_up_q(&q_0);
  clean_up_q(&q_1);
  clean_up_q(&q_2);
  clean_up_q(&q_3);

  const auto lit_count = lit_processed.load();
  const auto med_count = med_processed.load();
  const auto big_count = big_processed.load();
  const auto vuk_count = vuk_processed.load();

  const auto lit_time = static_cast<double>(duration.count()) / lit_count;
  const auto med_time = static_cast<double>(duration.count()) / med_count;
  const auto big_time = static_cast<double>(duration.count()) / big_count;
  const auto vuk_time = static_cast<double>(duration.count()) / vuk_count;

  fmt::print("Stage: {}\n", stage);
  fmt::print("\tLittle \t{:.4f} ms \t({})\n", lit_time, lit_count);
  fmt::print("\tMedium \t{:.4f} ms \t({})\n", med_time, med_count);
  fmt::print("\tBig    \t{:.4f} ms \t({})\n", big_time, big_count);
  fmt::print("\tVulkan \t{:.4f} ms \t({})\n", vuk_time, vuk_count);
  std::fflush(stdout);

  if (lit_count > 0) bm_full_table[stage - 1][kLitIdx] = lit_time;
  if (med_count > 0) bm_full_table[stage - 1][kMedIdx] = med_time;
  if (big_count > 0) bm_full_table[stage - 1][kBigIdx] = big_time;
  if (vuk_count > 0) bm_full_table[stage - 1][kVukIdx] = vuk_time;
}

}  // namespace android

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

// e.g.,
// xmake r bm-table-cifar-sparse-vk --stage 1 --device-to-measure 3A021JEHN02756
// xmake r bm-table-cifar-sparse-vk --stage 1 --device jetson --device-to-measure jetson --full
void dump_tables_for_python(int start_stage, int end_stage) {
  // First dump a marker that can be easily grep'ed
  fmt::print("\n### PYTHON_DATA_START ###\n");

  // Dump normal benchmark data in CSV format
  fmt::print("# NORMAL_BENCHMARK_DATA\n");
  fmt::print("stage,little,medium,big,vulkan\n");
  for (int stage = start_stage; stage <= end_stage; stage++) {
    fmt::print("{},{:.4f},{:.4f},{:.4f},{:.4f}\n",
               stage,
               bm_norm_table[stage - 1][kLitIdx],
               bm_norm_table[stage - 1][kMedIdx],
               bm_norm_table[stage - 1][kBigIdx],
               bm_norm_table[stage - 1][kVukIdx]);
  }

  // Dump fully benchmark data in CSV format
  fmt::print("# FULLY_BENCHMARK_DATA\n");
  fmt::print("stage,little,medium,big,vulkan\n");
  for (int stage = start_stage; stage <= end_stage; stage++) {
    fmt::print("{},{:.4f},{:.4f},{:.4f},{:.4f}\n",
               stage,
               bm_full_table[stage - 1][kLitIdx],
               bm_full_table[stage - 1][kMedIdx],
               bm_full_table[stage - 1][kBigIdx],
               bm_full_table[stage - 1][kVukIdx]);
  }

  // Add raw data dump that includes all values directly for debugging
  fmt::print("# RAW_NORMAL_TABLE_DATA\n");
  for (int stage = start_stage; stage <= end_stage; stage++) {
    fmt::print("Stage {}: {:.4f} {:.4f} {:.4f} {:.4f}\n",
               stage,
               bm_norm_table[stage - 1][0],
               bm_norm_table[stage - 1][1],
               bm_norm_table[stage - 1][2],
               bm_norm_table[stage - 1][3]);
  }

  fmt::print("# RAW_FULLY_TABLE_DATA\n");
  for (int stage = start_stage; stage <= end_stage; stage++) {
    fmt::print("Stage {}: {:.4f} {:.4f} {:.4f} {:.4f}\n",
               stage,
               bm_full_table[stage - 1][0],
               bm_full_table[stage - 1][1],
               bm_full_table[stage - 1][2],
               bm_full_table[stage - 1][3]);
  }

  fmt::print("### PYTHON_DATA_END ###\n");
  std::fflush(stdout);
}

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN

  int start_stage = 1;
  int end_stage = kNumStages;
  int seconds_to_run = 10;

  app.add_option("-s, --start-stage", start_stage, "Start stage");
  app.add_option("-e, --end-stage", end_stage, "End stage");
  app.add_option("-t, --seconds-to-run", seconds_to_run, "Seconds to run for normal benchmark");

  PARSE_ARGS_END

  init_tables();

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  // // Run normal benchmark (one processor type at a time)
  // std::cout << "Running normal benchmark (one processor at a time)...\n";
  // for (int stage = start_stage; stage <= end_stage; stage++) {
  //   if (!g_little_cores.empty()) {
  //     android::BM_run_normal(ProcessorType::kLittleCore, stage, seconds_to_run);
  //   }
  //   if (!g_medium_cores.empty()) {
  //     android::BM_run_normal(ProcessorType::kMediumCore, stage, seconds_to_run);
  //   }
  //   if (!g_big_cores.empty()) {
  //     android::BM_run_normal(ProcessorType::kBigCore, stage, seconds_to_run);
  //   }
  //   android::BM_run_normal(ProcessorType::kVulkan, stage, seconds_to_run);
  // }

  // Run fully benchmark (each processor type in isolation, but with all stages active)
  std::cout << "Running fully benchmark (each processor in isolation, all stages active)...\n";
  for (int stage = start_stage; stage <= end_stage; stage++) {
    android::BM_run_fully(stage, seconds_to_run);
    // // Run for each processor type
    // if (!g_little_cores.empty()) {
    //   android::BM_run_fully(ProcessorType::kLittleCore, stage, seconds_to_run);
    // }
    // if (!g_medium_cores.empty()) {
    //   android::BM_run_fully(ProcessorType::kMediumCore, stage, seconds_to_run);
    // }
    // if (!g_big_cores.empty()) {
    //   android::BM_run_fully(ProcessorType::kBigCore, stage, seconds_to_run);
    // }
    // android::BM_run_fully(ProcessorType::kVulkan, stage, seconds_to_run);
  }
  // Print the normal benchmark table with higher precision
  fmt::print("\nNormal Benchmark Results Table (ms per task):\n");
  fmt::print("Stage | Little Core | Medium Core | Big Core | Vulkan \n");
  fmt::print("------|------------|-------------|----------|--------\n");
  for (int stage = start_stage; stage <= end_stage; stage++) {
    fmt::print("{:5} | {:11.4f} | {:11.4f} | {:8.4f} | {:6.4f}\n",
               stage,
               bm_norm_table[stage - 1][kLitIdx],
               bm_norm_table[stage - 1][kMedIdx],
               bm_norm_table[stage - 1][kBigIdx],
               bm_norm_table[stage - 1][kVukIdx]);
  }

  // Calculate sums for normal benchmark
  double little_norm_sum = 0, medium_norm_sum = 0, big_norm_sum = 0, vulkan_norm_sum = 0;
  for (int stage = start_stage; stage <= end_stage; stage++) {
    little_norm_sum += bm_norm_table[stage - 1][kLitIdx];
    medium_norm_sum += bm_norm_table[stage - 1][kMedIdx];
    big_norm_sum += bm_norm_table[stage - 1][kBigIdx];
    vulkan_norm_sum += bm_norm_table[stage - 1][kVukIdx];
  }

  // Print sum for normal benchmark
  fmt::print("\nNormal Benchmark - Sum of stages {}-{}:\n", start_stage, end_stage);
  fmt::print("Little Core: {:.4f} ms\n", little_norm_sum);
  fmt::print("Medium Core: {:.4f} ms\n", medium_norm_sum);
  fmt::print("Big Core: {:.4f} ms\n", big_norm_sum);
  fmt::print("Vulkan: {:.4f} ms\n", vulkan_norm_sum);

  // Print the fully benchmark table with higher precision
  fmt::print("\nFully Benchmark Results Table (ms per task):\n");
  fmt::print("Stage | Little Core | Medium Core | Big Core | Vulkan \n");
  fmt::print("------|------------|-------------|----------|--------\n");
  for (int stage = start_stage; stage <= end_stage; stage++) {
    fmt::print("{:5} | {:11.4f} | {:11.4f} | {:8.4f} | {:6.4f}\n",
               stage,
               bm_full_table[stage - 1][kLitIdx],
               bm_full_table[stage - 1][kMedIdx],
               bm_full_table[stage - 1][kBigIdx],
               bm_full_table[stage - 1][kVukIdx]);
  }

  // Calculate sums for fully benchmark
  double little_full_sum = 0, medium_full_sum = 0, big_full_sum = 0, vulkan_full_sum = 0;
  for (int stage = start_stage; stage <= end_stage; stage++) {
    little_full_sum += bm_full_table[stage - 1][kLitIdx];
    medium_full_sum += bm_full_table[stage - 1][kMedIdx];
    big_full_sum += bm_full_table[stage - 1][kBigIdx];
    vulkan_full_sum += bm_full_table[stage - 1][kVukIdx];
  }

  // Print sum for fully benchmark
  fmt::print("\nFully Benchmark - Sum of stages {}-{}:\n", start_stage, end_stage);
  fmt::print("Little Core: {:.4f} ms\n", little_full_sum);
  fmt::print("Medium Core: {:.4f} ms\n", medium_full_sum);
  fmt::print("Big Core: {:.4f} ms\n", big_full_sum);
  fmt::print("Vulkan: {:.4f} ms\n", vulkan_full_sum);

  // Compare normal vs fully
  fmt::print("\nPerformance Comparison (Fully vs Normal):\n");
  fmt::print("Processor  | Normal (ms) | Fully (ms) | Ratio\n");
  fmt::print("-----------|-------------|-----------|-------\n");

  auto print_comparison = [](const std::string& name, double normal, double fully) {
    if (normal > 0) {
      fmt::print("{:<11}| {:11.2f} | {:10.2f} | ", name, normal, fully);
      if (fully > 0) {
        fmt::print("{:5.2f}x\n", fully / normal);
      } else {
        fmt::print("N/A\n");
      }
    }
  };

  print_comparison("Little Core", little_norm_sum, little_full_sum);
  print_comparison("Medium Core", medium_norm_sum, medium_full_sum);
  print_comparison("Big Core", big_norm_sum, big_full_sum);
  print_comparison("Vulkan", vulkan_norm_sum, vulkan_full_sum);

  // Dump tables in a format that's easy to load in Python
  dump_tables_for_python(start_stage, end_stage);

  return 0;
}
