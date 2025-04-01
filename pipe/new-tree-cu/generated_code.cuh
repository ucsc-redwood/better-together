#pragma once
#include <benchmark/benchmark.h>

#include <chrono>
#include <numeric>
#include <thread>

#include "../templates.hpp"
#include "../templates_cu.hpp"
#include "run_stages.hpp"
#include "task.hpp"

// Automatically generated benchmark code for CUDA

namespace generated_schedules {
using bench_func_t = void (*)(benchmark::State &);
struct ScheduleRecord {
  const char *name;
  bench_func_t func;
};
}  // namespace generated_schedules

namespace device_jetson {
static void BM_schedule_jetson_Tree_schedule_001(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
      chunk<Task, tree::SafeAppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 4>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kLittleCore, 6>, mgr);
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

static void BM_schedule_jetson_Tree_schedule_002(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
      chunk<Task, tree::SafeAppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 5>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kLittleCore, 6>, mgr);
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

static void BM_schedule_jetson_Tree_schedule_003(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
      chunk<Task, tree::SafeAppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 6>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kLittleCore, 6>, mgr);
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

static void BM_schedule_jetson_Tree_schedule_004(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 6>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(q_0_1, nullptr, cuda::run_multiple_stages<3, 7>, mgr);
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

static void BM_schedule_jetson_Tree_schedule_005(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
      chunk<Task, tree::SafeAppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 3>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<4, 7, ProcessorType::kLittleCore, 6>, mgr);
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

static void BM_schedule_jetson_Tree_schedule_006(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 6>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(q_0_1, nullptr, cuda::run_multiple_stages<4, 7>, mgr);
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

static void BM_schedule_jetson_Tree_schedule_007(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
      chunk<Task, tree::SafeAppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 2>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<3, 7, ProcessorType::kLittleCore, 6>, mgr);
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

static void BM_schedule_jetson_Tree_schedule_008(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 6>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(q_0_1, nullptr, cuda::run_multiple_stages<2, 7>, mgr);
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

static void BM_schedule_jetson_Tree_schedule_009(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(q_input, nullptr, cuda::run_multiple_stages<1, 7>, mgr);
    });

    t1.join();

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

static void BM_schedule_jetson_Tree_schedule_010(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
          q_input, &q_0_1, omp::run_multiple_stages<1, 4, ProcessorType::kLittleCore, 6>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(q_0_1, nullptr, cuda::run_multiple_stages<5, 7>, mgr);
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

static void BM_schedule_jetson_Tree_schedule_011(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
          q_input, &q_0_1, omp::run_multiple_stages<1, 5, ProcessorType::kLittleCore, 6>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(q_0_1, nullptr, cuda::run_multiple_stages<6, 7>, mgr);
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

static void BM_schedule_jetson_Tree_schedule_012(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
          q_input, &q_0_1, omp::run_multiple_stages<1, 6, ProcessorType::kLittleCore, 6>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(q_0_1, nullptr, cuda::run_multiple_stages<7, 7>, mgr);
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

static void BM_schedule_jetson_Tree_schedule_013(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
      chunk<Task, tree::SafeAppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 1>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<2, 7, ProcessorType::kLittleCore, 6>, mgr);
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

static void BM_schedule_jetson_Tree_schedule_014(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, nullptr, omp::run_multiple_stages<1, 7, ProcessorType::kLittleCore, 6>, mgr);
    });

    t1.join();

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
    {"jetson_Tree_schedule_001", &BM_schedule_jetson_Tree_schedule_001},
    {"jetson_Tree_schedule_002", &BM_schedule_jetson_Tree_schedule_002},
    {"jetson_Tree_schedule_003", &BM_schedule_jetson_Tree_schedule_003},
    {"jetson_Tree_schedule_004", &BM_schedule_jetson_Tree_schedule_004},
    {"jetson_Tree_schedule_005", &BM_schedule_jetson_Tree_schedule_005},
    {"jetson_Tree_schedule_006", &BM_schedule_jetson_Tree_schedule_006},
    {"jetson_Tree_schedule_007", &BM_schedule_jetson_Tree_schedule_007},
    {"jetson_Tree_schedule_008", &BM_schedule_jetson_Tree_schedule_008},
    {"jetson_Tree_schedule_009", &BM_schedule_jetson_Tree_schedule_009},
    {"jetson_Tree_schedule_010", &BM_schedule_jetson_Tree_schedule_010},
    {"jetson_Tree_schedule_011", &BM_schedule_jetson_Tree_schedule_011},
    {"jetson_Tree_schedule_012", &BM_schedule_jetson_Tree_schedule_012},
    {"jetson_Tree_schedule_013", &BM_schedule_jetson_Tree_schedule_013},
    {"jetson_Tree_schedule_014", &BM_schedule_jetson_Tree_schedule_014},
};
static const size_t schedule_count = sizeof(schedule_table) / sizeof(schedule_table[0]);
}  // namespace device_jetson

namespace device_jetsonlowpower {
static void BM_schedule_jetsonlowpower_Tree_schedule_001(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
      chunk<Task, tree::SafeAppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 4>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kLittleCore, 4>, mgr);
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

static void BM_schedule_jetsonlowpower_Tree_schedule_002(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
      chunk<Task, tree::SafeAppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 5>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<6, 7, ProcessorType::kLittleCore, 4>, mgr);
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

static void BM_schedule_jetsonlowpower_Tree_schedule_003(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
      chunk<Task, tree::SafeAppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 6>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kLittleCore, 4>, mgr);
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

static void BM_schedule_jetsonlowpower_Tree_schedule_004(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 4>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(q_0_1, nullptr, cuda::run_multiple_stages<3, 7>, mgr);
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

static void BM_schedule_jetsonlowpower_Tree_schedule_005(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
          q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 4>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(q_0_1, nullptr, cuda::run_multiple_stages<4, 7>, mgr);
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

static void BM_schedule_jetsonlowpower_Tree_schedule_006(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(q_0_1, nullptr, cuda::run_multiple_stages<2, 7>, mgr);
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

static void BM_schedule_jetsonlowpower_Tree_schedule_007(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(q_input, nullptr, cuda::run_multiple_stages<1, 7>, mgr);
    });

    t1.join();

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

static void BM_schedule_jetsonlowpower_Tree_schedule_008(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
      chunk<Task, tree::SafeAppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 3>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<4, 7, ProcessorType::kLittleCore, 4>, mgr);
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

static void BM_schedule_jetsonlowpower_Tree_schedule_009(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
      chunk<Task, tree::SafeAppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 2>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<3, 7, ProcessorType::kLittleCore, 4>, mgr);
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

static void BM_schedule_jetsonlowpower_Tree_schedule_010(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
          q_input, &q_0_1, omp::run_multiple_stages<1, 4, ProcessorType::kLittleCore, 4>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(q_0_1, nullptr, cuda::run_multiple_stages<5, 7>, mgr);
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

static void BM_schedule_jetsonlowpower_Tree_schedule_011(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
          q_input, &q_0_1, omp::run_multiple_stages<1, 5, ProcessorType::kLittleCore, 4>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(q_0_1, nullptr, cuda::run_multiple_stages<6, 7>, mgr);
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

static void BM_schedule_jetsonlowpower_Tree_schedule_012(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
          q_input, &q_0_1, omp::run_multiple_stages<1, 6, ProcessorType::kLittleCore, 4>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(q_0_1, nullptr, cuda::run_multiple_stages<7, 7>, mgr);
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

static void BM_schedule_jetsonlowpower_Tree_schedule_013(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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
      chunk<Task, tree::SafeAppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 1>, mgr);
    });
    std::thread t2([&]() {
      chunk<Task, tree::SafeAppData>(
          q_0_1, nullptr, omp::run_multiple_stages<2, 7, ProcessorType::kLittleCore, 4>, mgr);
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

static void BM_schedule_jetsonlowpower_Tree_schedule_014(benchmark::State &state) {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::SafeAppData>(mr, num_tasks);

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

    std::thread t1([&]() {
      chunk<Task, tree::SafeAppData>(
          q_input, nullptr, omp::run_multiple_stages<1, 7, ProcessorType::kLittleCore, 4>, mgr);
    });

    t1.join();

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
    {"jetsonlowpower_Tree_schedule_001", &BM_schedule_jetsonlowpower_Tree_schedule_001},
    {"jetsonlowpower_Tree_schedule_002", &BM_schedule_jetsonlowpower_Tree_schedule_002},
    {"jetsonlowpower_Tree_schedule_003", &BM_schedule_jetsonlowpower_Tree_schedule_003},
    {"jetsonlowpower_Tree_schedule_004", &BM_schedule_jetsonlowpower_Tree_schedule_004},
    {"jetsonlowpower_Tree_schedule_005", &BM_schedule_jetsonlowpower_Tree_schedule_005},
    {"jetsonlowpower_Tree_schedule_006", &BM_schedule_jetsonlowpower_Tree_schedule_006},
    {"jetsonlowpower_Tree_schedule_007", &BM_schedule_jetsonlowpower_Tree_schedule_007},
    {"jetsonlowpower_Tree_schedule_008", &BM_schedule_jetsonlowpower_Tree_schedule_008},
    {"jetsonlowpower_Tree_schedule_009", &BM_schedule_jetsonlowpower_Tree_schedule_009},
    {"jetsonlowpower_Tree_schedule_010", &BM_schedule_jetsonlowpower_Tree_schedule_010},
    {"jetsonlowpower_Tree_schedule_011", &BM_schedule_jetsonlowpower_Tree_schedule_011},
    {"jetsonlowpower_Tree_schedule_012", &BM_schedule_jetsonlowpower_Tree_schedule_012},
    {"jetsonlowpower_Tree_schedule_013", &BM_schedule_jetsonlowpower_Tree_schedule_013},
    {"jetsonlowpower_Tree_schedule_014", &BM_schedule_jetsonlowpower_Tree_schedule_014},
};
static const size_t schedule_count = sizeof(schedule_table) / sizeof(schedule_table[0]);
}  // namespace device_jetsonlowpower
