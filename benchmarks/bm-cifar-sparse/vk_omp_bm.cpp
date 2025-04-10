#include <algorithm>
#include <limits>
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

struct Task {
  static uint32_t uid_counter;
  uint32_t uid;

  // ----------------------------------
  cifar_sparse::AppData app_data;

  explicit Task(std::pmr::memory_resource* mr) : app_data(mr) { uid = uid_counter++; }

  // ----------------------------------
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  ProcessorType processed_by;

  void start_tick() { start_time = std::chrono::high_resolution_clock::now(); }
  void end_tick() { end_time = std::chrono::high_resolution_clock::now(); }
  void set_processed_by(ProcessorType processor_type) { processed_by = processor_type; }
  [[nodiscard]] ProcessorType get_processed_by() const { return processed_by; }

  [[nodiscard]] std::chrono::duration<double> get_duration() const { return end_time - start_time; }
};

uint32_t Task::uid_counter = 0;

using ProcessFunction = std::function<void(Task&)>;

// Worker thread that processes tasks and uses the tasks' built-in timing
static void task_timed_worker_thread(SPSCQueue<Task*, 1024>& in_queue,
                                     SPSCQueue<Task*, 1024>& out_queue,
                                     ProcessFunction process_function,
                                     ProcessorType processor_type) {
  for (size_t i = 0; i < kNumTasks; ++i) {
    Task* task = nullptr;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

    // Set processor type and start timing
    task->set_processed_by(processor_type);
    task->start_tick();

    // Process the task
    process_function(*task);

    // End timing
    task->end_tick();

    // Forward processed task to output queue
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
    default:
      return "Unknown";
  }
}

void calculate_stats_from_tasks(const std::vector<Task*>& tasks,
                                const std::string_view& stage_name,
                                size_t warmup_tasks = kWarmupTasks,
                                size_t cooldown_tasks = kCooldownTasks) {
  if (tasks.empty()) {
    spdlog::error("No tasks to analyze for Stage {}", stage_name);
    return;
  }

  // Get processor type from first task (assuming all tasks run on same type)
  ProcessorType processor_type = tasks.front()->get_processed_by();
  std::string cores_name = processor_type_to_string(processor_type);

  // Print first few and last few durations
  spdlog::trace("First 5 durations:");
  for (size_t i = 0; i < std::min(size_t(5), tasks.size()); ++i) {
    spdlog::trace("Task {}: {:.6f} ms", i, tasks[i]->get_duration().count() * 1000.0);
  }

  if (tasks.size() > 10) {
    spdlog::trace("Last 5 durations:");
    for (size_t i = tasks.size() - 5; i < tasks.size(); ++i) {
      spdlog::trace("Task {}: {:.6f} ms", i, tasks[i]->get_duration().count() * 1000.0);
    }
  }

  // Handle edge cases where we don't have enough samples
  size_t begin_idx = warmup_tasks;
  size_t end_idx = tasks.size();

  if (begin_idx >= end_idx) {
    begin_idx = 0;  // Not enough samples, use all of them
  }

  if (cooldown_tasks > 0 && tasks.size() > cooldown_tasks) {
    end_idx = tasks.size() - cooldown_tasks;
    if (begin_idx >= end_idx) {
      end_idx = tasks.size();  // Not enough samples after warmup, use all remaining
    }
  }

  // Calculate statistics for the middle tasks (between warmup and cooldown)
  double sum = 0.0;
  double min_time = std::numeric_limits<double>::max();
  double max_time = 0.0;
  int valid_count = 0;
  double product = 1.0;
  int geomean_count = 0;

  for (size_t i = begin_idx; i < end_idx; ++i) {
    double ms = tasks[i]->get_duration().count() * 1000.0;

    // Skip any zero or negative durations in calculations
    if (ms > 0.0000001) {
      sum += ms;
      product *= ms;
      valid_count++;
      geomean_count++;
      min_time = std::min(min_time, ms);
      max_time = std::max(max_time, ms);
    }
  }

  double arithmetic_mean = valid_count > 0 ? sum / valid_count : 0.0;
  double geomean = geomean_count > 0 ? std::pow(product, 1.0 / geomean_count) : 0.0;

  spdlog::trace(
      "[{}] Stage {} Stats: Total tasks: {}, Measured tasks: {} (skipped {} warmup, {} cooldown)",
      cores_name,
      stage_name,
      tasks.size(),
      end_idx - begin_idx,
      begin_idx,
      tasks.size() - end_idx);
  spdlog::trace("[{}] Stage {} Valid measurements: {}", cores_name, stage_name, valid_count);
  spdlog::trace(
      "[{}] Stage {} Arithmetic Mean: {:.3f} ms", cores_name, stage_name, arithmetic_mean);
  spdlog::info("[{}] Stage {} Geometric Mean: {:.3f} ms", cores_name, stage_name, geomean);
  spdlog::trace(
      "[{}] Stage {} Min: {:.3f} ms, Max: {:.3f} ms", cores_name, stage_name, min_time, max_time);
}

#define PREPARE_DATA                           \
  cifar_sparse::vulkan::VulkanDispatcher disp; \
  SPSCQueue<Task*, 1024> q_0;                  \
  SPSCQueue<Task*, 1024> q_1;                  \
  for (size_t i = 0; i < kNumTasks; ++i) {     \
    Task* task = new Task(disp.get_mr());      \
    q_0.enqueue(std::move(task));              \
  }

static void BM_Stage1_Big() {
  PREPARE_DATA;

  auto cores = g_big_cores;
  auto num_threads = cores.size();

  task_timed_worker_thread(
      q_0,
      q_1,
      [&](Task& task) {
        cifar_sparse::omp::dispatch_multi_stage(cores, num_threads, task.app_data, 1, 1);
      },
      ProcessorType::kBigCore);

  // Collect tasks from output queue
  std::vector<Task*> completed_tasks = collect_tasks_from_queue(q_1);

  calculate_stats_from_tasks(completed_tasks, "Stage 1");

  // Cleanup
  for (auto* task : completed_tasks) {
    delete task;
  }
}

static void BM_Stage1_Medium() {
  PREPARE_DATA;

  auto cores = g_medium_cores;
  auto num_threads = cores.size();

  task_timed_worker_thread(
      q_0,
      q_1,
      [&](Task& task) {
        cifar_sparse::omp::dispatch_multi_stage(cores, num_threads, task.app_data, 1, 1);
      },
      ProcessorType::kMediumCore);

  // Collect tasks from output queue
  std::vector<Task*> completed_tasks = collect_tasks_from_queue(q_1);

  calculate_stats_from_tasks(completed_tasks, "Stage 1");

  // Cleanup
  for (auto* task : completed_tasks) {
    delete task;
  }
}

static void BM_Stage1_Little() {
  PREPARE_DATA;

  auto cores = g_little_cores;
  auto num_threads = cores.size();

  task_timed_worker_thread(
      q_0,
      q_1,
      [&](Task& task) {
        cifar_sparse::omp::dispatch_multi_stage(cores, num_threads, task.app_data, 1, 1);
      },
      ProcessorType::kLittleCore);

  // Collect tasks from output queue
  std::vector<Task*> completed_tasks = collect_tasks_from_queue(q_1);

  calculate_stats_from_tasks(completed_tasks, "Stage 1");

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

  BM_Stage1_Big();
  BM_Stage1_Medium();
  BM_Stage1_Little();

  return 0;
}
