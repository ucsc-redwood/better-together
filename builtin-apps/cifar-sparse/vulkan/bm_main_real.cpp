#include <omp.h>

#include "../omp/dispatchers.hpp"
#include "../sparse_appdata.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"
#include "builtin-apps/pipeline/task.hpp"
#include "builtin-apps/pipeline/worker.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// Realistic Benchmark: OMP Stage
// ----------------------------------------------------------------------------

using MyTask = Task<cifar_sparse::v2::AppData>;

static void BM_run_OMP_stage_full(const int stage, const ProcessorType core_type) {
  auto mr = std::pmr::new_delete_resource();
  std::vector<std::unique_ptr<MyTask>> preallocated_tasks;
  SPSCQueue<MyTask*, kPoolSize> free_task_pool;
  for (size_t i = 0; i < kPoolSize; ++i) {
    preallocated_tasks.emplace_back(std::make_unique<MyTask>(mr));
    free_task_pool.enqueue(preallocated_tasks.back().get());
  }

  // The cores we want to measure
  std::vector<int> cores_to_use;
  if (core_type == ProcessorType::kLittleCore) {
    cores_to_use = g_little_cores;
  } else if (core_type == ProcessorType::kMediumCore) {
    cores_to_use = g_medium_cores;
  } else {
    cores_to_use = g_big_cores;
  }

  const int num_threads = cores_to_use.size();

  // find the counter part of the cores
  std::vector<int> cores_to_use_counter_part_a;
  std::vector<int> cores_to_use_counter_part_b;
  int num_threads_counter_part_a;
  int num_threads_counter_part_b;

  if (core_type == ProcessorType::kLittleCore) {
    cores_to_use_counter_part_a = g_medium_cores;
    cores_to_use_counter_part_b = g_big_cores;

  } else if (core_type == ProcessorType::kMediumCore) {
    cores_to_use_counter_part_a = g_little_cores;
    cores_to_use_counter_part_b = g_big_cores;
  } else {
    cores_to_use_counter_part_a = g_little_cores;
    cores_to_use_counter_part_b = g_medium_cores;
  }

  num_threads_counter_part_a = cores_to_use_counter_part_a.size();
  num_threads_counter_part_b = cores_to_use_counter_part_b.size();

  SPSCQueue<MyTask*, kPoolSize> q_0_1;
  SPSCQueue<MyTask*, kPoolSize> q_1_2;

  // Main benchmark
  {
    // Only setup the records once
    RecordManager::instance().setup(100,
                                    {std::make_pair(0, ProcessorType::kLittleCore),
                                     std::make_pair(1, ProcessorType::kMediumCore),
                                     std::make_pair(2, ProcessorType::kBigCore)});

    auto t0 = std::thread(worker_thread_record<MyTask>,
                          0,
                          std::ref(free_task_pool),
                          std::ref(q_0_1),
                          [&](MyTask& task) {
                            cifar_sparse::omp::v2::dispatch_multi_stage(
                                cores_to_use, num_threads, task.appdata, stage, stage);
                          });

    auto t1 = std::thread(
        worker_thread_record<MyTask>, 1, std::ref(q_0_1), std::ref(q_1_2), [&](MyTask& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              cores_to_use_counter_part_a, num_threads_counter_part_a, task.appdata, stage, stage);
        });

    auto t2 = std::thread(
        worker_thread_record<MyTask>,
        2,
        std::ref(q_1_2),
        std::ref(free_task_pool),
        [&](MyTask& task) {
          cifar_sparse::omp::v2::dispatch_multi_stage(
              cores_to_use_counter_part_b, num_threads_counter_part_b, task.appdata, stage, stage);
        });

    t0.join();
    t1.join();
    t2.join();

    RecordManager::instance().print_processor_type_stats(core_type);
  }
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN

  int stage = 0;
  app.add_option("--stage", stage, "Stage to run")->required();
  std::string core_type_str;
  app.add_option("--core", core_type_str, "Core type to run")->required();

  PARSE_ARGS_END

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id != "3A021JEHN02756") {
    return 0;
  }

  if (core_type_str == "little") {
    BM_run_OMP_stage_full(stage, ProcessorType::kLittleCore);
  } else if (core_type_str == "medium") {
    BM_run_OMP_stage_full(stage, ProcessorType::kMediumCore);
  } else if (core_type_str == "big") {
    BM_run_OMP_stage_full(stage, ProcessorType::kBigCore);
  } else {
    throw std::invalid_argument("Invalid core type");
  }

  return 0;
}
