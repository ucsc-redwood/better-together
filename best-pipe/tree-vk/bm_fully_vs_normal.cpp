#include <omp.h>

#include <mutex>
#include <queue>
#include <thread>

#include "builtin-apps/app.hpp"  // for 'g_big_cores', 'g_medium_cores', 'g_little_cores'
#include "builtin-apps/conf.hpp"
#include "builtin-apps/pipeline/task.hpp"
#include "builtin-apps/tree/omp/dispatchers.hpp"
#include "builtin-apps/tree/vulkan/dispatchers.hpp"

// ----------------------------------------------------------------------------
// Realistic Benchmark:  Stage
// ----------------------------------------------------------------------------

using MyTask = Task<tree::vulkan::VkAppData_Safe>;

// Little: 0, Big: 2, Medium: 1, Vulkan: 3
constexpr int kLittleIdx = 0;
constexpr int kMediumIdx = 1;
constexpr int kBigIdx = 2;
constexpr int kVulkanIdx = 3;

constexpr int kNumStages = 9;

namespace android {

std::atomic<bool> done = false;

// create a 2D table, 9 stage times 4 type of cores, initialize it with 0
// access by bm_table[stage][core_type] = value
std::array<std::array<double, 4>, kNumStages> bm_norm_table;
std::array<std::array<double, 4>, kNumStages> bm_full_table;

// Reset the done flag
void reset_done_flag() { done.store(false); }

void init_q(std::queue<MyTask*>& q, tree::vulkan::VulkanDispatcher& disp) {
  constexpr int kPhysicalTasks = 10;

  for (int i = 0; i < kPhysicalTasks; i++) {
    q.push(new MyTask(disp.get_mr()));
  }
}

void clean_up_q(std::queue<MyTask*>& q) {
  while (!q.empty()) {
    MyTask* task = q.front();
    q.pop();
    delete task;
  }
}

void similuation_thread(tree::vulkan::VulkanDispatcher& disp,
                        std::mutex& mutex,
                        std::function<void(MyTask*)> func) {
  thread_local std::queue<MyTask*> q;
  {
    std::lock_guard<std::mutex> lock(mutex);
    init_q(q, disp);
  }

  while (!done.load(std::memory_order_relaxed)) {
    MyTask* task = q.front();
    q.pop();

    func(task);

    // After done -> push back for reuse
    task->reset();
    q.push(task);
  }

  clean_up_q(q);
}

// Add a warmup function to run before benchmarks
static void warmup_processors(int seconds_to_run) {
  std::cout << "Warming up processors for " << seconds_to_run << " seconds..." << std::endl;

  // Create dispatcher
  tree::vulkan::VulkanDispatcher disp;

  // Reset done flag
  reset_done_flag();

  // Create counters for each processor type
  std::atomic<int> little_count(0);
  std::atomic<int> medium_count(0);
  std::atomic<int> big_count(0);
  std::atomic<int> vulkan_count(0);

  // Create task functions
  auto little_func = [&little_count](MyTask* task) {
    if (g_little_cores.empty()) return;
    // Use a simple workload for warmup
    tree::omp::dispatch_multi_stage(g_little_cores, g_little_cores.size(), task->appdata, 1, 1);
    little_count++;
  };

  auto medium_func = [&medium_count](MyTask* task) {
    if (g_medium_cores.empty()) return;
    tree::omp::dispatch_multi_stage(g_medium_cores, g_medium_cores.size(), task->appdata, 1, 1);
    medium_count++;
  };

  auto big_func = [&big_count](MyTask* task) {
    if (g_big_cores.empty()) return;
    tree::omp::dispatch_multi_stage(g_big_cores, g_big_cores.size(), task->appdata, 1, 1);
    big_count++;
  };

  auto vulkan_func = [&disp, &vulkan_count](MyTask* task) {
    disp.dispatch_multi_stage(task->appdata, 1, 1);
    vulkan_count++;
  };

  // Create threads for all processor types
  std::vector<std::thread> threads;

  std::mutex mutex;

  if (!g_little_cores.empty()) {
    threads.emplace_back(similuation_thread, std::ref(disp), std::ref(mutex), little_func);
  }

  if (!g_medium_cores.empty()) {
    threads.emplace_back(similuation_thread, std::ref(disp), std::ref(mutex), medium_func);
  }

  if (!g_big_cores.empty()) {
    threads.emplace_back(similuation_thread, std::ref(disp), std::ref(mutex), big_func);
  }

  threads.emplace_back(similuation_thread, std::ref(disp), std::ref(mutex), vulkan_func);

  // Run warmup for specified time
  std::this_thread::sleep_for(std::chrono::seconds(seconds_to_run));

  // Signal threads to stop
  done.store(true);

  // Join all threads
  for (auto& t : threads) {
    t.join();
  }

  // Output warmup statistics
  std::cout << "Warmup completed: " << "Little=" << little_count.load() << ", "
            << "Medium=" << medium_count.load() << ", " << "Big=" << big_count.load() << ", "
            << "Vulkan=" << vulkan_count.load() << std::endl;
}

static void BM_run_normal(const ProcessorType pt, const int stage, const int seconds_to_run) {
  tree::vulkan::VulkanDispatcher disp;

  // Reset the done flag before starting a new benchmark
  reset_done_flag();

  // Short warmup before starting the actual benchmark
  std::atomic<int> warmup_count(0);
  std::function<void(MyTask*)> warmup_func;

  // Create appropriate function for the processor type
  if (pt == ProcessorType::kVulkan) {
    warmup_func = [&disp, &warmup_count, stage](MyTask* task) {
      disp.dispatch_multi_stage(task->appdata, stage, stage);
      warmup_count++;
    };
  } else {
    std::vector<int> cores_to_use;
    if (pt == ProcessorType::kLittleCore) {
      cores_to_use = g_little_cores;
    } else if (pt == ProcessorType::kBigCore) {
      cores_to_use = g_big_cores;
    } else if (pt == ProcessorType::kMediumCore) {
      cores_to_use = g_medium_cores;
    }

    if (cores_to_use.empty()) {
      return;
    }

    warmup_func = [&warmup_count, stage, cores_to_use](MyTask* task) {
      tree::omp::dispatch_multi_stage(
          cores_to_use, cores_to_use.size(), task->appdata, stage, stage);
      warmup_count++;
    };
  }

  // Run a short warmup
  const int warmup_seconds = 2;
  std::mutex mutex;
  std::thread warmup_thread(similuation_thread, std::ref(disp), std::ref(mutex), warmup_func);
  std::this_thread::sleep_for(std::chrono::seconds(warmup_seconds));
  done.store(true);
  warmup_thread.join();

  // Reset done flag for the actual benchmark
  reset_done_flag();

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
    tree::omp::dispatch_multi_stage(cores_to_use, cores_to_use.size(), task->appdata, stage, stage);
    total_processed++;
  };

  std::thread t1;
  if (pt == ProcessorType::kVulkan) {
    if (cores_to_use.empty()) {
      // This is fine for Vulkan
    } else {
      SPDLOG_WARN("Unexpected cores for Vulkan processor type");
    }
    t1 = std::thread(similuation_thread, std::ref(disp), std::ref(mutex), gpu_func);
  } else {
    if (cores_to_use.empty()) {
      SPDLOG_WARN("No cores to use for processor type: {}", static_cast<int>(pt));
      return;
    }
    t1 = std::thread(similuation_thread, std::ref(disp), std::ref(mutex), cpu_func);
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

  // update the table

  // map ProcessorType::kLittleCore = 0, ProcessorType::kBigCore = 1, ProcessorType::kMediumCore =
  // 2, ProcessorType::kVulkan = 3
  if (total_processed > 0) {
    bm_norm_table[stage - 1][static_cast<int>(pt)] =
        static_cast<double>(duration.count()) / total_processed;
  }
}

static void BM_run_fully(const ProcessorType pt_to_measure,
                         const int stage,
                         const int seconds_to_run) {
  // Create dispatchers for each processor type
  tree::vulkan::VulkanDispatcher disp;

  // Reset the done flag before starting a new benchmark
  reset_done_flag();

  // Short warmup before starting the actual benchmark
  std::atomic<int> little_warmup(0);
  std::atomic<int> medium_warmup(0);
  std::atomic<int> big_warmup(0);
  std::atomic<int> vulkan_warmup(0);

  // Create warmup functions
  auto little_warmup_func = [&little_warmup, stage](MyTask* task) {
    if (g_little_cores.empty()) return;
    tree::omp::dispatch_multi_stage(
        g_little_cores, g_little_cores.size(), task->appdata, stage, stage);
    little_warmup++;
  };

  auto medium_warmup_func = [&medium_warmup, stage](MyTask* task) {
    if (g_medium_cores.empty()) return;
    tree::omp::dispatch_multi_stage(
        g_medium_cores, g_medium_cores.size(), task->appdata, stage, stage);
    medium_warmup++;
  };

  auto big_warmup_func = [&big_warmup, stage](MyTask* task) {
    if (g_big_cores.empty()) return;
    tree::omp::dispatch_multi_stage(g_big_cores, g_big_cores.size(), task->appdata, stage, stage);
    big_warmup++;
  };

  auto vulkan_warmup_func = [&disp, &vulkan_warmup, stage](MyTask* task) {
    disp.dispatch_multi_stage(task->appdata, stage, stage);
    vulkan_warmup++;
  };

  // Run warmup with all processors
  std::vector<std::thread> warmup_threads;
  const int warmup_seconds = 2;
  std::mutex warmup_mutex;

  if (!g_little_cores.empty()) {
    warmup_threads.emplace_back(
        similuation_thread, std::ref(disp), std::ref(warmup_mutex), little_warmup_func);
  }

  if (!g_medium_cores.empty()) {
    warmup_threads.emplace_back(
        similuation_thread, std::ref(disp), std::ref(warmup_mutex), medium_warmup_func);
  }

  if (!g_big_cores.empty()) {
    warmup_threads.emplace_back(
        similuation_thread, std::ref(disp), std::ref(warmup_mutex), big_warmup_func);
  }

  warmup_threads.emplace_back(
      similuation_thread, std::ref(disp), std::ref(warmup_mutex), vulkan_warmup_func);

  std::this_thread::sleep_for(std::chrono::seconds(warmup_seconds));
  done.store(true);

  for (auto& t : warmup_threads) {
    t.join();
  }

  // Reset done flag for the actual benchmark
  reset_done_flag();

  auto start = std::chrono::high_resolution_clock::now();

  // Use atomic counters for task tracking, but we'll only report on the one we're measuring
  std::atomic<int> little_processed(0);
  std::atomic<int> medium_processed(0);
  std::atomic<int> big_processed(0);
  std::atomic<int> vulkan_processed(0);

  // Create GPU task function
  auto gpu_func = [&disp, &vulkan_processed, stage](MyTask* task) {
    disp.dispatch_multi_stage(task->appdata, stage, stage);
    vulkan_processed++;
  };

  // Create CPU task functions for each core type
  auto little_func = [&little_processed, stage](MyTask* task) {
    if (g_little_cores.empty()) return;
    tree::omp::dispatch_multi_stage(
        g_little_cores, g_little_cores.size(), task->appdata, stage, stage);
    little_processed++;
  };

  auto medium_func = [&medium_processed, stage](MyTask* task) {
    if (g_medium_cores.empty()) return;
    tree::omp::dispatch_multi_stage(
        g_medium_cores, g_medium_cores.size(), task->appdata, stage, stage);
    medium_processed++;
  };

  auto big_func = [&big_processed, stage](MyTask* task) {
    if (g_big_cores.empty()) return;
    tree::omp::dispatch_multi_stage(g_big_cores, g_big_cores.size(), task->appdata, stage, stage);
    big_processed++;
  };

  // Launch threads for all processor types
  std::vector<std::thread> threads;
  std::mutex benchmark_mutex;

  // Create threads for all available core types (we want all to run concurrently)
  if (!g_little_cores.empty()) {
    threads.emplace_back(
        similuation_thread, std::ref(disp), std::ref(benchmark_mutex), little_func);
  }

  if (!g_medium_cores.empty()) {
    threads.emplace_back(
        similuation_thread, std::ref(disp), std::ref(benchmark_mutex), medium_func);
  }

  if (!g_big_cores.empty()) {
    threads.emplace_back(similuation_thread, std::ref(disp), std::ref(benchmark_mutex), big_func);
  }

  // Always create Vulkan thread
  threads.emplace_back(similuation_thread, std::ref(disp), std::ref(benchmark_mutex), gpu_func);

  // Sleep for the specified time to let the benchmark run
  std::this_thread::sleep_for(std::chrono::seconds(seconds_to_run));

  // Signal all threads to stop
  done.store(true);

  // Wait for all threads to finish
  for (auto& t : threads) {
    t.join();
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Get the count for the processor type we're measuring
  int count = 0;
  std::string proc_name;

  switch (pt_to_measure) {
    case ProcessorType::kLittleCore:
      count = little_processed.load();
      proc_name = "Little Core";
      break;
    case ProcessorType::kMediumCore:
      count = medium_processed.load();
      proc_name = "Medium Core";
      break;
    case ProcessorType::kBigCore:
      count = big_processed.load();
      proc_name = "Big Core";
      break;
    case ProcessorType::kVulkan:
      count = vulkan_processed.load();
      proc_name = "Vulkan";
      break;
    default:
      proc_name = "Unknown";
  }

  // Report stats for the processor type we're measuring
  std::cout << "Stage: " << stage << " (" << proc_name << " - with all processors running)"
            << std::endl;
  std::cout << "\tTotal processed: " << count << std::endl;
  if (count > 0) {
    std::cout << "\tAverage time per task: " << duration.count() / count << " ms" << std::endl;
  } else {
    std::cout << "\tNo tasks processed" << std::endl;
  }

  // Only update the table entry for the processor type we're measuring
  if (count > 0) {
    bm_full_table[stage - 1][static_cast<int>(pt_to_measure)] =
        static_cast<double>(duration.count()) / count;
  }
}

}  // namespace android

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

// e.g.,
// xmake r bm-table-cifar-dense-vk --stage 1 --device-to-measure 3A021JEHN02756
// xmake r bm-table-cifar-dense-vk --stage 1 --device jetson --device-to-measure jetson --full

void dump_tables_for_python(int start_stage, int end_stage) {
  // Correct enum values based on debug output

  // First dump a marker that can be easily grep'ed
  std::cout << "\n### PYTHON_DATA_START ###" << std::endl;

  // Dump normal benchmark data in CSV format
  std::cout << "# NORMAL_BENCHMARK_DATA" << std::endl;
  std::cout << "stage,little,medium,big,vulkan" << std::endl;
  for (int stage = start_stage; stage <= end_stage; stage++) {
    std::cout << stage << "," << android::bm_norm_table[stage - 1][kLittleIdx] << ","
              << android::bm_norm_table[stage - 1][kMediumIdx] << ","
              << android::bm_norm_table[stage - 1][kBigIdx] << ","
              << android::bm_norm_table[stage - 1][kVulkanIdx] << std::endl;
  }

  // Dump fully benchmark data in CSV format
  std::cout << "# FULLY_BENCHMARK_DATA" << std::endl;
  std::cout << "stage,little,medium,big,vulkan" << std::endl;
  for (int stage = start_stage; stage <= end_stage; stage++) {
    std::cout << stage << "," << android::bm_full_table[stage - 1][kLittleIdx] << ","
              << android::bm_full_table[stage - 1][kMediumIdx] << ","
              << android::bm_full_table[stage - 1][kBigIdx] << ","
              << android::bm_full_table[stage - 1][kVulkanIdx] << std::endl;
  }

  // Add raw data dump that includes all values directly for debugging
  std::cout << "# RAW_NORMAL_TABLE_DATA" << std::endl;
  for (int stage = start_stage; stage <= end_stage; stage++) {
    std::cout << "Stage " << stage << ": ";
    for (int i = 0; i < 4; i++) {
      std::cout << android::bm_norm_table[stage - 1][i] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "# RAW_FULLY_TABLE_DATA" << std::endl;
  for (int stage = start_stage; stage <= end_stage; stage++) {
    std::cout << "Stage " << stage << ": ";
    for (int i = 0; i < 4; i++) {
      std::cout << android::bm_full_table[stage - 1][i] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "### PYTHON_DATA_END ###" << std::endl;
}

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN

  int start_stage = 1;
  int end_stage = kNumStages;
  int seconds_to_run = 10;
  int warmup_seconds = 1;

  app.add_option("-s, --start-stage", start_stage, "Start stage");
  app.add_option("-e, --end-stage", end_stage, "End stage");
  app.add_option("-t, --seconds-to-run", seconds_to_run, "Seconds to run for normal benchmark");
  app.add_option("-w, --warmup", warmup_seconds, "Seconds to warmup before benchmarks");

  PARSE_ARGS_END

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  // Initialize tables with 0
  for (int stage = 0; stage < kNumStages; stage++) {
    for (int processor = 0; processor < 4; processor++) {
      android::bm_norm_table[stage][processor] = 0.0;
      android::bm_full_table[stage][processor] = 0.0;
    }
  }

  // Global warmup to stabilize system state
  if (warmup_seconds > 0) {
    android::warmup_processors(warmup_seconds);
  }

  // Run normal benchmark (one processor type at a time)
  std::cout << "Running normal benchmark (one processor at a time)...\n";
  for (int stage = start_stage; stage <= end_stage; stage++) {
    if (!g_little_cores.empty()) {
      android::BM_run_normal(ProcessorType::kLittleCore, stage, seconds_to_run);
    }
    if (!g_medium_cores.empty()) {
      android::BM_run_normal(ProcessorType::kMediumCore, stage, seconds_to_run);
    }
    if (!g_big_cores.empty()) {
      android::BM_run_normal(ProcessorType::kBigCore, stage, seconds_to_run);
    }
    android::BM_run_normal(ProcessorType::kVulkan, stage, seconds_to_run);
  }

  // Run fully benchmark (each processor type in isolation, but with all stages active)
  std::cout << "Running fully benchmark (each processor in isolation, all stages active)...\n";
  for (int stage = start_stage; stage <= end_stage; stage++) {
    // Run for each processor type
    if (!g_little_cores.empty()) {
      android::BM_run_fully(ProcessorType::kLittleCore, stage, seconds_to_run);
    }
    if (!g_medium_cores.empty()) {
      android::BM_run_fully(ProcessorType::kMediumCore, stage, seconds_to_run);
    }
    if (!g_big_cores.empty()) {
      android::BM_run_fully(ProcessorType::kBigCore, stage, seconds_to_run);
    }
    android::BM_run_fully(ProcessorType::kVulkan, stage, seconds_to_run);
  }

  // Print the normal benchmark table with higher precision
  std::cout << "\nNormal Benchmark Results Table (ms per task):\n";
  std::cout << "Stage | Little Core | Medium Core | Big Core | Vulkan \n";
  std::cout << "------|------------|-------------|----------|--------\n";
  for (int stage = start_stage; stage <= end_stage; stage++) {
    std::cout << std::setw(5) << stage << " | " << std::fixed << std::setprecision(4)
              << std::setw(11) << android::bm_norm_table[stage - 1][kLittleIdx]
              << " | "  // Little Core
              << std::setw(11) << android::bm_norm_table[stage - 1][kMediumIdx]
              << " | "                                                                // Medium Core
              << std::setw(8) << android::bm_norm_table[stage - 1][kBigIdx] << " | "  // Big Core
              << std::setw(6) << android::bm_norm_table[stage - 1][kVulkanIdx]        // Vulkan
              << "\n";
  }

  // Calculate sums for normal benchmark
  double little_norm_sum = 0, medium_norm_sum = 0, big_norm_sum = 0, vulkan_norm_sum = 0;
  for (int stage = start_stage; stage <= end_stage; stage++) {
    little_norm_sum += android::bm_norm_table[stage - 1][kLittleIdx];
    medium_norm_sum += android::bm_norm_table[stage - 1][kMediumIdx];
    big_norm_sum += android::bm_norm_table[stage - 1][kBigIdx];
    vulkan_norm_sum += android::bm_norm_table[stage - 1][kVulkanIdx];
  }

  // Print sum for normal benchmark
  std::cout << "\nNormal Benchmark - Sum of stages " << start_stage << "-" << end_stage << ":"
            << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Little Core: " << little_norm_sum << " ms" << std::endl;
  std::cout << "Medium Core: " << medium_norm_sum << " ms" << std::endl;
  std::cout << "Big Core: " << big_norm_sum << " ms" << std::endl;
  std::cout << "Vulkan: " << vulkan_norm_sum << " ms" << std::endl;

  // Print the fully benchmark table with higher precision
  std::cout << "\nFully Benchmark Results Table (ms per task):\n";
  std::cout << "Stage | Little Core | Medium Core | Big Core | Vulkan \n";
  std::cout << "------|------------|-------------|----------|--------\n";
  for (int stage = start_stage; stage <= end_stage; stage++) {
    std::cout << std::setw(5) << stage << " | " << std::fixed << std::setprecision(4)
              << std::setw(11) << android::bm_full_table[stage - 1][kLittleIdx]
              << " | "  // Little Core
              << std::setw(11) << android::bm_full_table[stage - 1][kMediumIdx]
              << " | "                                                                // Medium Core
              << std::setw(8) << android::bm_full_table[stage - 1][kBigIdx] << " | "  // Big Core
              << std::setw(6) << android::bm_full_table[stage - 1][kVulkanIdx]        // Vulkan
              << "\n";
  }

  // Calculate sums for fully benchmark
  double little_full_sum = 0, medium_full_sum = 0, big_full_sum = 0, vulkan_full_sum = 0;
  for (int stage = start_stage; stage <= end_stage; stage++) {
    little_full_sum += android::bm_full_table[stage - 1][kLittleIdx];
    medium_full_sum += android::bm_full_table[stage - 1][kMediumIdx];
    big_full_sum += android::bm_full_table[stage - 1][kBigIdx];
    vulkan_full_sum += android::bm_full_table[stage - 1][kVulkanIdx];
  }

  // Print sum for fully benchmark
  std::cout << "\nFully Benchmark - Sum of stages " << start_stage << "-" << end_stage << ":"
            << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Little Core: " << little_full_sum << " ms" << std::endl;
  std::cout << "Medium Core: " << medium_full_sum << " ms" << std::endl;
  std::cout << "Big Core: " << big_full_sum << " ms" << std::endl;
  std::cout << "Vulkan: " << vulkan_full_sum << " ms" << std::endl;

  // Compare normal vs fully
  std::cout << "\nPerformance Comparison (Fully vs Normal):" << std::endl;
  std::cout << "Processor  | Normal (ms) | Fully (ms) | Ratio\n";
  std::cout << "-----------|-------------|-----------|-------\n";

  auto print_comparison = [](const std::string& name, double normal, double fully) {
    if (normal > 0) {
      std::cout << std::left << std::setw(11) << name << "| " << std::right << std::fixed
                << std::setprecision(2) << std::setw(11) << normal << " | " << std::setw(10)
                << fully << " | ";

      // Calculate ratio (fully / normal)
      if (fully > 0) {
        double ratio = fully / normal;
        std::cout << std::setw(5) << ratio << "x";
      } else {
        std::cout << "N/A";
      }
      std::cout << std::endl;
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
