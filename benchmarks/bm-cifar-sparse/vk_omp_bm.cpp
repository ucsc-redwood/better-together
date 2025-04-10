#include <limits>

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
};

uint32_t Task::uid_counter = 0;

using ProcessFunction = std::function<void(Task&)>;

// static void worker_thread(SPSCQueue<Task*, 1024>& in_queue,
//                           SPSCQueue<Task*, 1024>* out_queue,
//                           ProcessFunction process_function) {
//   for (size_t i = 0; i < kNumTasks; ++i) {
//     Task* task = nullptr;
//     while (!in_queue.dequeue(task)) {
//       std::this_thread::yield();
//     }

//     // Process the task (timing is handled in the provided process function)
//     process_function(*task);

//     // Forward processed task
//     if (out_queue) {
//       out_queue->enqueue(std::move(task));
//     }
//   }
// }

static void measured_worker_thread(std::vector<std::chrono::duration<double>>& durations,
                                   SPSCQueue<Task*, 1024>& in_queue,
                                   SPSCQueue<Task*, 1024>* out_queue,
                                   ProcessFunction process_function) {
  for (size_t i = 0; i < kNumTasks; ++i) {
    Task* task = nullptr;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

    // Process the task (timing is handled in the provided process function)
    auto start = std::chrono::high_resolution_clock::now();
    process_function(*task);
    auto end = std::chrono::high_resolution_clock::now();
    durations.push_back(end - start);

    // Forward processed task
    if (out_queue) {
      out_queue->enqueue(std::move(task));
    }
  }
}

// ----------------------------------------------------------------
// Stages
// ----------------------------------------------------------------

constexpr auto kIterations = 1;
// constexpr auto kRepetitions = 5;

void calculate_stats(const std::vector<std::chrono::duration<double>>& durations,
                     const std::string_view& stage_name,
                     const std::string_view& cores_name,
                     size_t warmup_tasks = kWarmupTasks,
                     size_t cooldown_tasks = kCooldownTasks) {
  // Print first few and last few durations to diagnose
  spdlog::info("First 5 durations:");
  for (size_t i = 0; i < std::min(size_t(5), durations.size()); ++i) {
    spdlog::info("Task {}: {:.6f} ms", i, durations[i].count() * 1000.0);
  }

  if (durations.size() > 10) {
    spdlog::info("Last 5 durations:");
    for (size_t i = durations.size() - 5; i < durations.size(); ++i) {
      spdlog::info("Task {}: {:.6f} ms", i, durations[i].count() * 1000.0);
    }
  }

  // Handle edge cases where we don't have enough samples
  size_t begin_idx = warmup_tasks;
  size_t end_idx = durations.size();

  if (begin_idx >= end_idx) {
    begin_idx = 0;  // Not enough samples, use all of them
  }

  if (cooldown_tasks > 0 && durations.size() > cooldown_tasks) {
    end_idx = durations.size() - cooldown_tasks;
    if (begin_idx >= end_idx) {
      end_idx = durations.size();  // Not enough samples after warmup, use all remaining
    }
  }

  // Calculate average in milliseconds for the middle tasks (between warmup and cooldown)
  double sum = 0.0;
  double min_time = std::numeric_limits<double>::max();
  double max_time = 0.0;
  int valid_count = 0;

  for (size_t i = begin_idx; i < end_idx; ++i) {
    const auto& duration = durations[i];
    double ms = duration.count() * 1000.0;

    // Skip any zero or negative durations in calculations
    if (ms > 0.0000001) {
      sum += ms;
      valid_count++;
      min_time = std::min(min_time, ms);
      max_time = std::max(max_time, ms);
    }
  }

  // Calculate geometric mean avoiding zeros
  double product = 1.0;
  int geomean_count = 0;
  for (size_t i = begin_idx; i < end_idx; ++i) {
    const auto& duration = durations[i];
    double ms = duration.count() * 1000.0;
    if (ms > 0.0000001) {
      product *= ms;
      geomean_count++;
    }
  }

  double arithmetic_mean = valid_count > 0 ? sum / valid_count : 0.0;
  double geomean = geomean_count > 0 ? std::pow(product, 1.0 / geomean_count) : 0.0;

  spdlog::info(
      "[{}] Stage {} Stats: Total tasks: {}, Measured tasks: {} (skipped {} warmup, {} cooldown)",
      cores_name,
      stage_name,
      durations.size(),
      end_idx - begin_idx,
      begin_idx,
      durations.size() - end_idx);
  spdlog::info("[{}] Stage {} Valid measurements: {}", cores_name, stage_name, valid_count);
  spdlog::info("[{}] Stage {} Arithmetic Mean: {:.3f} ms", cores_name, stage_name, arithmetic_mean);
  spdlog::info("[{}] Stage {} Geometric Mean: {:.3f} ms", cores_name, stage_name, geomean);
  spdlog::info(
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

  std::vector<std::chrono::duration<double>> durations;
  durations.reserve(kNumTasks);

  measured_worker_thread(durations, q_0, &q_1, [&](Task& task) {
    cifar_sparse::omp::dispatch_multi_stage(cores, num_threads, task.app_data, 1, 1);
  });

  calculate_stats(durations, "Stage 1", "Big");
}

static void BM_Stage1_Medium() {
  PREPARE_DATA;

  auto cores = g_medium_cores;
  auto num_threads = cores.size();

  std::vector<std::chrono::duration<double>> durations;
  durations.reserve(kNumTasks);

  measured_worker_thread(durations, q_0, &q_1, [&](Task& task) {
    cifar_sparse::omp::dispatch_multi_stage(cores, num_threads, task.app_data, 1, 1);
  });

  calculate_stats(durations, "Stage 1", "Medium");
}

static void BM_Stage1_Little() {
  PREPARE_DATA;

  auto cores = g_little_cores;
  auto num_threads = cores.size();

  std::vector<std::chrono::duration<double>> durations;
  durations.reserve(kNumTasks);

  measured_worker_thread(durations, q_0, &q_1, [&](Task& task) {
    cifar_sparse::omp::dispatch_multi_stage(cores, num_threads, task.app_data, 1, 1);
  });

  calculate_stats(durations, "Stage 1", "Little");
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
