#pragma once
#include <benchmark/benchmark.h>

#include <chrono>
#include <numeric>
#include <thread>

#include "../templates.hpp"
#include "../templates_vk.hpp"
#include "run_stages.hpp"
#include "task.hpp"

// Automatically generated benchmark code for Vulkan

namespace generated_schedules {
using bench_func_t = void (*)(benchmark::State &);
struct ScheduleRecord {
  const char *name;
  bench_func_t func;
};
}  // namespace generated_schedules

namespace device_3A021JEHN02756 {
static void BM_schedule_3A021JEHN02756_Tree_schedule_001(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<3, 3, ProcessorType::kLittleCore, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<4, 6>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_002(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, &q_2_3, omp::run_multiple_stages<6, 6, ProcessorType::kLittleCore, 4>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_003(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<3, 3, ProcessorType::kLittleCore, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<4, 5>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_004(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_005(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_006(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_007(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 6>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_008(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 6>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_009(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 6>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_010(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, &q_2_3, omp::run_multiple_stages<6, 6, ProcessorType::kLittleCore, 4>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_011(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_012(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 6>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_013(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_014(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 6>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_015(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_016(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 6>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_017(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<3, 5>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_018(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<3, 5>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_019(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<3, 6>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_020(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<3, 6>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_021(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<4, 5>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_022(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 3, ProcessorType::kBigCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<4, 5>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_023(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<4, 6>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_024(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 3, ProcessorType::kBigCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<4, 6>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_025(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<3, 4>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_026(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 3, ProcessorType::kBigCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<4, 4>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_027(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<3, 3, ProcessorType::kLittleCore, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<4, 4>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_028(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<3, 3, ProcessorType::kLittleCore, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<4, 5>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_029(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<3, 3, ProcessorType::kLittleCore, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<4, 6>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_030(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_031(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, &q_2_3, omp::run_multiple_stages<6, 6, ProcessorType::kLittleCore, 4>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_032(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_033(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 6>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_034(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_035(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_036(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 6>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_037(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_038(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_039(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 6>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_040(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<3, 4>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_041(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<4, 4>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_042(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<3, 3, ProcessorType::kLittleCore, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<4, 4>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_043(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_044(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_045(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_046(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_047(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_048(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_049(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, &q_2_3, omp::run_multiple_stages<5, 5, ProcessorType::kLittleCore, 4>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kBigCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_3A021JEHN02756_Tree_schedule_050(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, &q_2_3, omp::run_multiple_stages<5, 5, ProcessorType::kLittleCore, 4>);
    });
    std::thread t4([&]() {
      chunk<Task, tree::SafeAppData>(
          q_2_3, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// Table of schedules for this device:
static generated_schedules::ScheduleRecord schedule_table[] = {
    {"3A021JEHN02756_Tree_schedule_001", &BM_schedule_3A021JEHN02756_Tree_schedule_001},
    {"3A021JEHN02756_Tree_schedule_002", &BM_schedule_3A021JEHN02756_Tree_schedule_002},
    {"3A021JEHN02756_Tree_schedule_003", &BM_schedule_3A021JEHN02756_Tree_schedule_003},
    {"3A021JEHN02756_Tree_schedule_004", &BM_schedule_3A021JEHN02756_Tree_schedule_004},
    {"3A021JEHN02756_Tree_schedule_005", &BM_schedule_3A021JEHN02756_Tree_schedule_005},
    {"3A021JEHN02756_Tree_schedule_006", &BM_schedule_3A021JEHN02756_Tree_schedule_006},
    {"3A021JEHN02756_Tree_schedule_007", &BM_schedule_3A021JEHN02756_Tree_schedule_007},
    {"3A021JEHN02756_Tree_schedule_008", &BM_schedule_3A021JEHN02756_Tree_schedule_008},
    {"3A021JEHN02756_Tree_schedule_009", &BM_schedule_3A021JEHN02756_Tree_schedule_009},
    {"3A021JEHN02756_Tree_schedule_010", &BM_schedule_3A021JEHN02756_Tree_schedule_010},
    {"3A021JEHN02756_Tree_schedule_011", &BM_schedule_3A021JEHN02756_Tree_schedule_011},
    {"3A021JEHN02756_Tree_schedule_012", &BM_schedule_3A021JEHN02756_Tree_schedule_012},
    {"3A021JEHN02756_Tree_schedule_013", &BM_schedule_3A021JEHN02756_Tree_schedule_013},
    {"3A021JEHN02756_Tree_schedule_014", &BM_schedule_3A021JEHN02756_Tree_schedule_014},
    {"3A021JEHN02756_Tree_schedule_015", &BM_schedule_3A021JEHN02756_Tree_schedule_015},
    {"3A021JEHN02756_Tree_schedule_016", &BM_schedule_3A021JEHN02756_Tree_schedule_016},
    {"3A021JEHN02756_Tree_schedule_017", &BM_schedule_3A021JEHN02756_Tree_schedule_017},
    {"3A021JEHN02756_Tree_schedule_018", &BM_schedule_3A021JEHN02756_Tree_schedule_018},
    {"3A021JEHN02756_Tree_schedule_019", &BM_schedule_3A021JEHN02756_Tree_schedule_019},
    {"3A021JEHN02756_Tree_schedule_020", &BM_schedule_3A021JEHN02756_Tree_schedule_020},
    {"3A021JEHN02756_Tree_schedule_021", &BM_schedule_3A021JEHN02756_Tree_schedule_021},
    {"3A021JEHN02756_Tree_schedule_022", &BM_schedule_3A021JEHN02756_Tree_schedule_022},
    {"3A021JEHN02756_Tree_schedule_023", &BM_schedule_3A021JEHN02756_Tree_schedule_023},
    {"3A021JEHN02756_Tree_schedule_024", &BM_schedule_3A021JEHN02756_Tree_schedule_024},
    {"3A021JEHN02756_Tree_schedule_025", &BM_schedule_3A021JEHN02756_Tree_schedule_025},
    {"3A021JEHN02756_Tree_schedule_026", &BM_schedule_3A021JEHN02756_Tree_schedule_026},
    {"3A021JEHN02756_Tree_schedule_027", &BM_schedule_3A021JEHN02756_Tree_schedule_027},
    {"3A021JEHN02756_Tree_schedule_028", &BM_schedule_3A021JEHN02756_Tree_schedule_028},
    {"3A021JEHN02756_Tree_schedule_029", &BM_schedule_3A021JEHN02756_Tree_schedule_029},
    {"3A021JEHN02756_Tree_schedule_030", &BM_schedule_3A021JEHN02756_Tree_schedule_030},
    {"3A021JEHN02756_Tree_schedule_031", &BM_schedule_3A021JEHN02756_Tree_schedule_031},
    {"3A021JEHN02756_Tree_schedule_032", &BM_schedule_3A021JEHN02756_Tree_schedule_032},
    {"3A021JEHN02756_Tree_schedule_033", &BM_schedule_3A021JEHN02756_Tree_schedule_033},
    {"3A021JEHN02756_Tree_schedule_034", &BM_schedule_3A021JEHN02756_Tree_schedule_034},
    {"3A021JEHN02756_Tree_schedule_035", &BM_schedule_3A021JEHN02756_Tree_schedule_035},
    {"3A021JEHN02756_Tree_schedule_036", &BM_schedule_3A021JEHN02756_Tree_schedule_036},
    {"3A021JEHN02756_Tree_schedule_037", &BM_schedule_3A021JEHN02756_Tree_schedule_037},
    {"3A021JEHN02756_Tree_schedule_038", &BM_schedule_3A021JEHN02756_Tree_schedule_038},
    {"3A021JEHN02756_Tree_schedule_039", &BM_schedule_3A021JEHN02756_Tree_schedule_039},
    {"3A021JEHN02756_Tree_schedule_040", &BM_schedule_3A021JEHN02756_Tree_schedule_040},
    {"3A021JEHN02756_Tree_schedule_041", &BM_schedule_3A021JEHN02756_Tree_schedule_041},
    {"3A021JEHN02756_Tree_schedule_042", &BM_schedule_3A021JEHN02756_Tree_schedule_042},
    {"3A021JEHN02756_Tree_schedule_043", &BM_schedule_3A021JEHN02756_Tree_schedule_043},
    {"3A021JEHN02756_Tree_schedule_044", &BM_schedule_3A021JEHN02756_Tree_schedule_044},
    {"3A021JEHN02756_Tree_schedule_045", &BM_schedule_3A021JEHN02756_Tree_schedule_045},
    {"3A021JEHN02756_Tree_schedule_046", &BM_schedule_3A021JEHN02756_Tree_schedule_046},
    {"3A021JEHN02756_Tree_schedule_047", &BM_schedule_3A021JEHN02756_Tree_schedule_047},
    {"3A021JEHN02756_Tree_schedule_048", &BM_schedule_3A021JEHN02756_Tree_schedule_048},
    {"3A021JEHN02756_Tree_schedule_049", &BM_schedule_3A021JEHN02756_Tree_schedule_049},
    {"3A021JEHN02756_Tree_schedule_050", &BM_schedule_3A021JEHN02756_Tree_schedule_050},
};
static const size_t schedule_count = sizeof(schedule_table) / sizeof(schedule_table[0]);
}  // namespace device_3A021JEHN02756

namespace device_9b034f1b {
static void BM_schedule_9b034f1b_Tree_schedule_001(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kLittleCore, 3>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_002(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 6>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kLittleCore, 3>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_003(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kLittleCore, 3>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_004(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 6>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kLittleCore, 3>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_005(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_006(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_007(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 6>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_008(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<4, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_009(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 3, ProcessorType::kLittleCore, 3>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<4, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_010(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<3, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<4, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_011(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<3, 3, ProcessorType::kLittleCore, 3>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<4, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_012(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, nullptr, vulkan::run_gpu_stages<4, 7>);
    });

    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_013(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, nullptr, vulkan::run_gpu_stages<4, 7>);
    });

    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_014(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kLittleCore, 3>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_015(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kLittleCore, 3>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_016(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<3, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_017(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<3, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_018(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, nullptr, vulkan::run_gpu_stages<3, 7>);
    });

    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_019(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, nullptr, vulkan::run_gpu_stages<3, 7>);
    });

    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_020(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, nullptr, vulkan::run_gpu_stages<3, 7>);
    });

    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_021(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, nullptr, vulkan::run_gpu_stages<3, 7>);
    });

    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_022(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 4>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_023(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_024(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<4, 6>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_025(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, nullptr, vulkan::run_gpu_stages<4, 7>);
    });

    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_026(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, nullptr, vulkan::run_gpu_stages<4, 7>);
    });

    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_027(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<4, 4, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<5, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_028(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<3, 4, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<5, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_029(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<4, 5, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<6, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_030(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<4, 6, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<7, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_031(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<3, 5, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<6, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_032(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<3, 6, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<7, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_033(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 4, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<5, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_034(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_input, &q_0_1, vulkan::run_gpu_stages<1, 1>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 4, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kLittleCore, 3>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_035(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 5, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<6, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_036(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_input, &q_0_1, vulkan::run_gpu_stages<1, 1>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 5, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kLittleCore, 3>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_037(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_input, &q_0_1, vulkan::run_gpu_stages<1, 1>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 3, ProcessorType::kLittleCore, 3>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<4, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_038(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 3>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<4, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_039(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<4, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_040(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<4, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_041(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 6, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<7, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_042(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_input, &q_0_1, vulkan::run_gpu_stages<1, 1>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 6, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kLittleCore, 3>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_043(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_input, &q_0_1, vulkan::run_gpu_stages<1, 1>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<3, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_044(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<3, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_045(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<3, 7, ProcessorType::kMediumCore, 2>);
    });

    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_046(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 4, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<5, 5, ProcessorType::kLittleCore, 3>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<6, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_047(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 4, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<5, 5>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kLittleCore, 3>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_048(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 4, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<5, 6, ProcessorType::kLittleCore, 3>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, nullptr, vulkan::run_gpu_stages<7, 7>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_049(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 4, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<5, 6>);
    });
    std::thread t3([&]() {
      chunk<Task, tree::SafeAppData>(
          q_1_2, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kLittleCore, 3>);
    });

    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

static void BM_schedule_9b034f1b_Tree_schedule_050(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    moodycamel::ConcurrentQueue<Task *> q_0_1;

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 4, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kLittleCore, 3>);
    });

    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }  // for (auto _ : state)

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// Table of schedules for this device:
static generated_schedules::ScheduleRecord schedule_table[] = {
    {"9b034f1b_Tree_schedule_001", &BM_schedule_9b034f1b_Tree_schedule_001},
    {"9b034f1b_Tree_schedule_002", &BM_schedule_9b034f1b_Tree_schedule_002},
    {"9b034f1b_Tree_schedule_003", &BM_schedule_9b034f1b_Tree_schedule_003},
    {"9b034f1b_Tree_schedule_004", &BM_schedule_9b034f1b_Tree_schedule_004},
    {"9b034f1b_Tree_schedule_005", &BM_schedule_9b034f1b_Tree_schedule_005},
    {"9b034f1b_Tree_schedule_006", &BM_schedule_9b034f1b_Tree_schedule_006},
    {"9b034f1b_Tree_schedule_007", &BM_schedule_9b034f1b_Tree_schedule_007},
    {"9b034f1b_Tree_schedule_008", &BM_schedule_9b034f1b_Tree_schedule_008},
    {"9b034f1b_Tree_schedule_009", &BM_schedule_9b034f1b_Tree_schedule_009},
    {"9b034f1b_Tree_schedule_010", &BM_schedule_9b034f1b_Tree_schedule_010},
    {"9b034f1b_Tree_schedule_011", &BM_schedule_9b034f1b_Tree_schedule_011},
    {"9b034f1b_Tree_schedule_012", &BM_schedule_9b034f1b_Tree_schedule_012},
    {"9b034f1b_Tree_schedule_013", &BM_schedule_9b034f1b_Tree_schedule_013},
    {"9b034f1b_Tree_schedule_014", &BM_schedule_9b034f1b_Tree_schedule_014},
    {"9b034f1b_Tree_schedule_015", &BM_schedule_9b034f1b_Tree_schedule_015},
    {"9b034f1b_Tree_schedule_016", &BM_schedule_9b034f1b_Tree_schedule_016},
    {"9b034f1b_Tree_schedule_017", &BM_schedule_9b034f1b_Tree_schedule_017},
    {"9b034f1b_Tree_schedule_018", &BM_schedule_9b034f1b_Tree_schedule_018},
    {"9b034f1b_Tree_schedule_019", &BM_schedule_9b034f1b_Tree_schedule_019},
    {"9b034f1b_Tree_schedule_020", &BM_schedule_9b034f1b_Tree_schedule_020},
    {"9b034f1b_Tree_schedule_021", &BM_schedule_9b034f1b_Tree_schedule_021},
    {"9b034f1b_Tree_schedule_022", &BM_schedule_9b034f1b_Tree_schedule_022},
    {"9b034f1b_Tree_schedule_023", &BM_schedule_9b034f1b_Tree_schedule_023},
    {"9b034f1b_Tree_schedule_024", &BM_schedule_9b034f1b_Tree_schedule_024},
    {"9b034f1b_Tree_schedule_025", &BM_schedule_9b034f1b_Tree_schedule_025},
    {"9b034f1b_Tree_schedule_026", &BM_schedule_9b034f1b_Tree_schedule_026},
    {"9b034f1b_Tree_schedule_027", &BM_schedule_9b034f1b_Tree_schedule_027},
    {"9b034f1b_Tree_schedule_028", &BM_schedule_9b034f1b_Tree_schedule_028},
    {"9b034f1b_Tree_schedule_029", &BM_schedule_9b034f1b_Tree_schedule_029},
    {"9b034f1b_Tree_schedule_030", &BM_schedule_9b034f1b_Tree_schedule_030},
    {"9b034f1b_Tree_schedule_031", &BM_schedule_9b034f1b_Tree_schedule_031},
    {"9b034f1b_Tree_schedule_032", &BM_schedule_9b034f1b_Tree_schedule_032},
    {"9b034f1b_Tree_schedule_033", &BM_schedule_9b034f1b_Tree_schedule_033},
    {"9b034f1b_Tree_schedule_034", &BM_schedule_9b034f1b_Tree_schedule_034},
    {"9b034f1b_Tree_schedule_035", &BM_schedule_9b034f1b_Tree_schedule_035},
    {"9b034f1b_Tree_schedule_036", &BM_schedule_9b034f1b_Tree_schedule_036},
    {"9b034f1b_Tree_schedule_037", &BM_schedule_9b034f1b_Tree_schedule_037},
    {"9b034f1b_Tree_schedule_038", &BM_schedule_9b034f1b_Tree_schedule_038},
    {"9b034f1b_Tree_schedule_039", &BM_schedule_9b034f1b_Tree_schedule_039},
    {"9b034f1b_Tree_schedule_040", &BM_schedule_9b034f1b_Tree_schedule_040},
    {"9b034f1b_Tree_schedule_041", &BM_schedule_9b034f1b_Tree_schedule_041},
    {"9b034f1b_Tree_schedule_042", &BM_schedule_9b034f1b_Tree_schedule_042},
    {"9b034f1b_Tree_schedule_043", &BM_schedule_9b034f1b_Tree_schedule_043},
    {"9b034f1b_Tree_schedule_044", &BM_schedule_9b034f1b_Tree_schedule_044},
    {"9b034f1b_Tree_schedule_045", &BM_schedule_9b034f1b_Tree_schedule_045},
    {"9b034f1b_Tree_schedule_046", &BM_schedule_9b034f1b_Tree_schedule_046},
    {"9b034f1b_Tree_schedule_047", &BM_schedule_9b034f1b_Tree_schedule_047},
    {"9b034f1b_Tree_schedule_048", &BM_schedule_9b034f1b_Tree_schedule_048},
    {"9b034f1b_Tree_schedule_049", &BM_schedule_9b034f1b_Tree_schedule_049},
    {"9b034f1b_Tree_schedule_050", &BM_schedule_9b034f1b_Tree_schedule_050},
};
static const size_t schedule_count = sizeof(schedule_table) / sizeof(schedule_table[0]);
}  // namespace device_9b034f1b
