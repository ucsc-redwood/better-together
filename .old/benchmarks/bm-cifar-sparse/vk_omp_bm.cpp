#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <ranges>
#include <set>
#include <string>

#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/conf.hpp"
#include "builtin-apps/resources_path.hpp"
#include "builtin-apps/spsc_queue.hpp"
#include "spdlog/common.h"

constexpr size_t kNumTasks = 100;
// Number of tasks to skip at the beginning (warmup) and end (cooldown)
constexpr size_t kWarmupTasks = 25;    // Skip first 25 tasks (warmup period)
constexpr size_t kCooldownTasks = 25;  // Skip last 25 tasks (cooldown period)

struct Record {
  int start_stage;
  int end_stage;
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  [[nodiscard]] std::chrono::duration<double> get_duration() const { return end_time - start_time; }
};

struct Task {
  static uint32_t uid_counter;
  uint32_t uid;

  // ----------------------------------
  cifar_sparse::AppData app_data;

  explicit Task(std::pmr::memory_resource* mr) : app_data(mr) { uid = uid_counter++; }

  // ----------------------------------
  // key: chunk_idx, value record
  std::array<Record, 5> records;

  void start_tick(ProcessorType pt) {
    records[static_cast<int>(pt)].start_time = std::chrono::high_resolution_clock::now();
  }

  void end_tick(ProcessorType pt) {
    records[static_cast<int>(pt)].end_time = std::chrono::high_resolution_clock::now();
  }

  void set_stages(ProcessorType pt, const int start_stage, const int end_stage) {
    records[static_cast<int>(pt)].start_stage = start_stage;
    records[static_cast<int>(pt)].end_stage = end_stage;
  }
};

uint32_t Task::uid_counter = 0;

using ProcessFunction = std::function<void(Task&)>;

// Worker thread that processes tasks and uses the tasks' built-in timing
static void task_timed_worker_thread(SPSCQueue<Task*, 1024>& in_queue,
                                     SPSCQueue<Task*, 1024>& out_queue,
                                     ProcessFunction process_function,
                                     const ProcessorType pt,
                                     const int start_stage,
                                     const int end_stage) {
  for (size_t i = 0; i < kNumTasks; ++i) {
    Task* task = nullptr;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

    task->set_stages(pt, start_stage, end_stage);
    task->start_tick(pt);

    process_function(*task);

    task->end_tick(pt);

    out_queue.enqueue(std::move(task));
  }
}

// Collect tasks from output queue for analysis
static std::vector<Task*> collect_tasks_from_queue(SPSCQueue<Task*, 1024>& queue) {
  std::vector<Task*> tasks;
  tasks.reserve(kNumTasks);

  for (size_t i = 0; i < kNumTasks; ++i) {
    Task* task = nullptr;
    while (!queue.dequeue(task)) {
      std::this_thread::yield();
    }
    tasks.push_back(task);
  }

  return tasks;
}

// ----------------------------------------------------------------
// Stages
// ----------------------------------------------------------------

constexpr auto kIterations = 1;

// Convert a processor type to a readable string
std::string processor_type_to_string(ProcessorType type) {
  switch (type) {
    case ProcessorType::kLittleCore:
      return "Little";
    case ProcessorType::kMediumCore:
      return "Medium";
    case ProcessorType::kBigCore:
      return "Big";
    case ProcessorType::kVulkan:
      return "Vulkan";
    case ProcessorType::kCuda:
      return "Cuda";
    default:
      return "Unknown";
  }
}

void print_task_timing_log_pretty(const std::vector<Task*>& tasks) {
  // Calculate the actual range of tasks to print
  size_t start_task = 0;
  size_t end_task = tasks.size();

  // Determine which processor types are used in the tasks
  std::set<int> used_processors;
  for (const auto& task : tasks) {
    for (int pt_idx = 0; pt_idx < 5; pt_idx++) {
      const auto& record = task->records[pt_idx];
      if (record.start_time != std::chrono::high_resolution_clock::time_point{} &&
          record.end_time != std::chrono::high_resolution_clock::time_point{}) {
        used_processors.insert(pt_idx);
      }
    }
  }

  if (used_processors.empty()) {
    std::cout << "No valid timing data found in any tasks.\n";
    return;
  }

  // Print the header
  std::cout << "\n┌───────────────────────────────────────────────────────────────┐\n";
  std::cout << "│                   TASK EXECUTION TIME REPORT                   │\n";
  std::cout << "├────────┬";

  // Create header for each processor type
  for (const auto& pt_idx : used_processors) {
    std::cout << "────────────────────┬";
  }
  std::cout << "\b│\n";

  // Print processor type names in header
  std::cout << "│ Task # │";
  for (const auto& pt_idx : used_processors) {
    std::cout << " " << std::setw(18) << std::left
              << processor_type_to_string(static_cast<ProcessorType>(pt_idx)) << "│";
  }
  std::cout << "\n├────────┼";

  // Complete the header separator
  for (const auto& pt_idx : used_processors) {
    std::cout << "────────────────────┼";
  }
  std::cout << "\b│\n";

  // Print each task's data
  for (size_t i = start_task; i < end_task; i++) {
    const auto& task = tasks[i];

    // Check if this task has any timing data
    bool has_data = false;
    for (const auto& pt_idx : used_processors) {
      const auto& record = task->records[pt_idx];
      if (record.start_time != std::chrono::high_resolution_clock::time_point{} &&
          record.end_time != std::chrono::high_resolution_clock::time_point{}) {
        has_data = true;
        break;
      }
    }

    if (!has_data) continue;

    // Print task ID
    std::cout << "│ " << std::setw(6) << task->uid << " │";

    // Print duration for each processor type
    for (const auto& pt_idx : used_processors) {
      const auto& record = task->records[pt_idx];
      if (record.start_time != std::chrono::high_resolution_clock::time_point{} &&
          record.end_time != std::chrono::high_resolution_clock::time_point{}) {
        double ms = record.get_duration().count() * 1000.0;
        std::cout << " " << std::setw(16) << std::fixed << std::setprecision(3) << ms << " ms │";
      } else {
        std::cout << " " << std::setw(18) << "-" << " │";
      }
    }
    std::cout << "\n";
  }

  // Print the footer
  std::cout << "└────────┴";
  for (const auto& pt_idx : used_processors) {
    std::cout << "────────────────────┴";
  }
  std::cout << "\b┘\n";

  // Print a summary of statistics
  std::cout << "\n┌───────────────────────────────────────────────────────────────┐\n";
  std::cout << "│                    PERFORMANCE STATISTICS                      │\n";
  std::cout << "├────────────┬──────────┬──────────┬──────────┬─────────────────┤\n";
  std::cout << "│ Processor  │   Min    │   Max    │   Avg    │  Total Tasks    │\n";
  std::cout << "├────────────┼──────────┼──────────┼──────────┼─────────────────┤\n";

  // Calculate and print statistics for each processor type
  for (const auto& pt_idx : used_processors) {
    ProcessorType proc_type = static_cast<ProcessorType>(pt_idx);
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    double sum_ms = 0.0;
    int count = 0;

    for (const auto& task : tasks) {
      const auto& record = task->records[pt_idx];
      if (record.start_time != std::chrono::high_resolution_clock::time_point{} &&
          record.end_time != std::chrono::high_resolution_clock::time_point{}) {
        double ms = record.get_duration().count() * 1000.0;
        if (ms > 0.0) {
          min_ms = std::min(min_ms, ms);
          max_ms = std::max(max_ms, ms);
          sum_ms += ms;
          count++;
        }
      }
    }

    if (count > 0) {
      double avg_ms = sum_ms / count;
      std::cout << "│ " << std::setw(10) << std::left << processor_type_to_string(proc_type)
                << " │ " << std::setw(8) << std::fixed << std::setprecision(3) << min_ms << " │ "
                << std::setw(8) << std::fixed << std::setprecision(3) << max_ms << " │ "
                << std::setw(8) << std::fixed << std::setprecision(3) << avg_ms << " │ "
                << std::setw(15) << count << " │\n";
    } else {
      std::cout << "│ " << std::setw(10) << std::left << processor_type_to_string(proc_type)
                << " │ " << std::setw(8) << "-" << " │ " << std::setw(8) << "-" << " │ "
                << std::setw(8) << "-" << " │ " << std::setw(15) << "0" << " │\n";
    }
  }

  std::cout << "└────────────┴──────────┴──────────┴──────────┴─────────────────┘\n\n";
}

void print_task_timing_log(const std::vector<Task*>& tasks) {
  for (const auto& task : tasks) {
    for (int pt_idx = 0; pt_idx < 5; pt_idx++) {
      const auto& record = task->records[pt_idx];
      if (record.start_time != std::chrono::high_resolution_clock::time_point{} &&
          record.end_time != std::chrono::high_resolution_clock::time_point{}) {
        std::cout << "Task " << task->uid << " on "
                  << processor_type_to_string(static_cast<ProcessorType>(pt_idx)) << " took "
                  << record.get_duration().count() * 1000.0 << " ms" << std::endl;
      }
    }
  }
}

double calculate_geomean(const std::vector<Task*>& tasks, const ProcessorType pt) {
  // Use ranges to filter and transform in one pipeline
  auto durations = tasks | std::ranges::views::filter([pt](const Task* task) {
                     const auto& record = task->records[static_cast<int>(pt)];
                     return record.start_time != std::chrono::high_resolution_clock::time_point{} &&
                            record.end_time != std::chrono::high_resolution_clock::time_point{};
                   }) |
                   std::ranges::views::transform([pt](const Task* task) {
                     return task->records[static_cast<int>(pt)].get_duration().count() * 1000.0;
                   }) |
                   std::ranges::views::filter([](double d) { return d > 0.0; });

  // Convert to vector to get size and use with algorithms
  std::vector<double> duration_vec(durations.begin(), durations.end());

  if (duration_vec.empty()) {
    return 0.0;
  }

  // For geometric mean, we need to compute the product of all elements and then take the nth root
  // To avoid overflow, we calculate the sum of logs and then exponentiate
  double log_sum = std::transform_reduce(
      duration_vec.begin(), duration_vec.end(), 0.0, std::plus<>(), [](double d) {
        return std::log(d);
      });

  return std::exp(log_sum / duration_vec.size());
}

// Add an overload that calculates geomean for all processor types and prints a summary
void print_geomean_summary(const std::vector<Task*>& tasks) {
  std::cout << "\n┌───────────────────────────────────────────┐\n";
  std::cout << "│        GEOMETRIC MEAN SUMMARY             │\n";
  std::cout << "├────────────┬────────────────────────────┤\n";
  std::cout << "│ Processor  │ Geometric Mean (ms)         │\n";
  std::cout << "├────────────┼────────────────────────────┤\n";

  for (int pt_idx = 0; pt_idx < 5; pt_idx++) {
    ProcessorType proc_type = static_cast<ProcessorType>(pt_idx);
    double geomean = calculate_geomean(tasks, proc_type);

    if (geomean > 0.0) {
      std::cout << "│ " << std::setw(10) << std::left << processor_type_to_string(proc_type)
                << " │ " << std::setw(26) << std::fixed << std::setprecision(6) << geomean
                << " │\n";
    } else {
      // Check if there's any data for this processor type
      bool has_data = false;
      for (const auto& task : tasks) {
        const auto& record = task->records[pt_idx];
        if (record.start_time != std::chrono::high_resolution_clock::time_point{} &&
            record.end_time != std::chrono::high_resolution_clock::time_point{}) {
          has_data = true;
          break;
        }
      }

      if (has_data) {
        std::cout << "│ " << std::setw(10) << std::left << processor_type_to_string(proc_type)
                  << " │ " << std::setw(26) << "INVALID DATA (zero values)" << " │\n";
      }
    }
  }

  std::cout << "└────────────┴────────────────────────────┘\n\n";
}

static void BM_Stage_Concurrent(const int stage, const ProcessorType processor_type) {
  cifar_sparse::vulkan::VulkanDispatcher disp;

  SPSCQueue<Task*, 1024> q_0;
  SPSCQueue<Task*, 1024> q_1;
  SPSCQueue<Task*, 1024> q_2;

  for (size_t i = 0; i < kNumTasks; ++i) {
    Task* task = new Task(disp.get_mr());
    q_0.enqueue(std::move(task));
  }

  std::vector<int> cores;
  int num_threads;

  if (processor_type == ProcessorType::kBigCore) {
    cores = g_big_cores;
    num_threads = cores.size();
  } else if (processor_type == ProcessorType::kMediumCore) {
    cores = g_med_cores;
    num_threads = cores.size();
  } else if (processor_type == ProcessorType::kLittleCore) {
    cores = g_lit_cores;
    num_threads = cores.size();
  } else {
    spdlog::error("Invalid processor type: {}", processor_type_to_string(processor_type));
    return;
  }

  // Using GPU to warmup and populate all stages before the OMP stage
  std::thread t1([&]() {
    task_timed_worker_thread(
        q_0,
        q_1,
        [&](Task& task) { disp.dispatch_multi_stage(task.app_data, 1, stage); },
        ProcessorType::kVulkan,
        1,
        stage);
  });

  // The CPU stage that we are interested in measuring
  std::thread t2([&]() {
    task_timed_worker_thread(
        q_1,
        q_2,
        [&](Task& task) {
          cifar_sparse::omp::dispatch_multi_stage(cores, num_threads, task.app_data, stage, stage);
        },
        processor_type,
        stage,
        stage);
  });

  t1.join();
  t2.join();

  // Collect tasks from output queue
  std::vector<Task*> completed_tasks = collect_tasks_from_queue(q_2);

  // Print geometric mean summary
  print_geomean_summary(completed_tasks);

  // Cleanup
  for (auto* task : completed_tasks) {
    delete task;
  }
}

// ----------------------------------------------------------------
// Main
// ----------------------------------------------------------------
int main(int argc, char** argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  BM_Stage_Concurrent(1, ProcessorType::kBigCore);
  BM_Stage_Concurrent(1, ProcessorType::kMediumCore);
  BM_Stage_Concurrent(1, ProcessorType::kLittleCore);

  BM_Stage_Concurrent(2, ProcessorType::kBigCore);
  BM_Stage_Concurrent(2, ProcessorType::kMediumCore);
  BM_Stage_Concurrent(2, ProcessorType::kLittleCore);

  BM_Stage_Concurrent(3, ProcessorType::kBigCore);
  BM_Stage_Concurrent(3, ProcessorType::kMediumCore);
  BM_Stage_Concurrent(3, ProcessorType::kLittleCore);

  BM_Stage_Concurrent(4, ProcessorType::kBigCore);
  BM_Stage_Concurrent(4, ProcessorType::kMediumCore);
  BM_Stage_Concurrent(4, ProcessorType::kLittleCore);

  BM_Stage_Concurrent(5, ProcessorType::kBigCore);
  BM_Stage_Concurrent(5, ProcessorType::kMediumCore);
  BM_Stage_Concurrent(5, ProcessorType::kLittleCore);

  BM_Stage_Concurrent(6, ProcessorType::kBigCore);
  BM_Stage_Concurrent(6, ProcessorType::kMediumCore);
  BM_Stage_Concurrent(6, ProcessorType::kLittleCore);

  BM_Stage_Concurrent(7, ProcessorType::kBigCore);
  BM_Stage_Concurrent(7, ProcessorType::kMediumCore);
  BM_Stage_Concurrent(7, ProcessorType::kLittleCore);

  BM_Stage_Concurrent(8, ProcessorType::kBigCore);
  BM_Stage_Concurrent(8, ProcessorType::kMediumCore);
  BM_Stage_Concurrent(8, ProcessorType::kLittleCore);

  BM_Stage_Concurrent(9, ProcessorType::kBigCore);
  BM_Stage_Concurrent(9, ProcessorType::kMediumCore);
  BM_Stage_Concurrent(9, ProcessorType::kLittleCore);

  return 0;
}
