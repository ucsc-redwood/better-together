#include <omp.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-dense/omp/dispatchers.hpp"
#include "builtin-apps/cifar-dense/vulkan/dispatchers.hpp"
#include "builtin-apps/config_reader.hpp"
#include "builtin-apps/curl_json.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"
#include "builtin-apps/pipeline/task.hpp"
#include "builtin-apps/pipeline/worker.hpp"

using MyTask = Task<cifar_dense::AppData>;

// ----------------------------------------------------------------------------
// Schedule Auto
// ----------------------------------------------------------------------------

constexpr size_t kPoolSize = 32;

using QueueT = SPSCQueue<MyTask*, kPoolSize>;

static void BM_pipe_warmup(const Schedule schedule) {
  auto n_chunks = schedule.n_chunks();

  // Initialize the dispatcher and queues
  cifar_dense::vulkan::VulkanDispatcher disp;
  std::vector<std::unique_ptr<MyTask>> preallocated_tasks;
  std::vector<QueueT> queues(n_chunks);
  for (size_t i = 0; i < kPoolSize; ++i) {
    preallocated_tasks.emplace_back(std::make_unique<MyTask>(disp.get_mr()));
    queues[0].enqueue(preallocated_tasks.back().get());
  }

  // Run the schedule
  {
    std::vector<std::thread> threads;

    for (size_t i = 0; i < n_chunks; ++i) {
      QueueT& q_in = queues[i];
      QueueT& q_out = queues[(i + 1) % n_chunks];

      const int start = schedule.start_stage(i) + 1;
      const int end = schedule.end_stage(i);

      const ProcessorType pt = get_processor_type_from_chunk_config(schedule.chunks[i]);

      if (pt == ProcessorType::kVulkan) {
        threads.emplace_back(worker_thread<MyTask, kPoolSize>,
                             std::ref(q_in),
                             std::ref(q_out),
                             [disp = &disp, start, end](MyTask& task) {
                               disp->dispatch_multi_stage(task.appdata, start, end);
                             });
      } else {
        threads.emplace_back(worker_thread<MyTask, kPoolSize>,
                             std::ref(q_in),
                             std::ref(q_out),
                             [pt, start, end](MyTask& task) {
                               const auto cores = get_cores_by_type(pt);
                               cifar_dense::omp::dispatch_multi_stage(
                                   cores, cores.size(), task.appdata, start, end);
                             });
      }
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }
}

static void BM_pipe_cifar_dense_vk_schedule_auto(const Schedule schedule) {
  auto n_chunks = schedule.n_chunks();

  // Initialize the dispatcher and queues
  cifar_dense::vulkan::VulkanDispatcher disp;
  std::vector<std::unique_ptr<MyTask>> preallocated_tasks;
  std::vector<QueueT> queues(n_chunks);
  for (size_t i = 0; i < kPoolSize; ++i) {
    preallocated_tasks.emplace_back(std::make_unique<MyTask>(disp.get_mr()));
    queues[0].enqueue(preallocated_tasks.back().get());
  }

  Logger<kNumToProcess> logger;

  // Run the schedule
  {
    std::vector<std::thread> threads;

    for (size_t i = 0; i < n_chunks; ++i) {
      QueueT& q_in = queues[i];
      QueueT& q_out = queues[(i + 1) % n_chunks];

      const int start = schedule.start_stage(i) + 1;
      const int end = schedule.end_stage(i);

      const ProcessorType pt = get_processor_type_from_chunk_config(schedule.chunks[i]);

      if (pt == ProcessorType::kVulkan) {
        threads.emplace_back(worker_thread_record<MyTask, kPoolSize>,
                             i,
                             std::ref(logger),
                             std::ref(q_in),
                             std::ref(q_out),
                             [disp = &disp, start, end](MyTask& task) {
                               disp->dispatch_multi_stage(task.appdata, start, end);
                             });
      } else {
        threads.emplace_back(worker_thread_record<MyTask, kPoolSize>,
                             i,
                             std::ref(logger),
                             std::ref(q_in),
                             std::ref(q_out),
                             [pt, start, end](MyTask& task) {
                               const auto cores = get_cores_by_type(pt);
                               cifar_dense::omp::dispatch_multi_stage(
                                   cores, cores.size(), task.appdata, start, end);
                             });
      }
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

  logger.dump_records_for_python(schedule);
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN;

  std::string device_to_measure;
  app.add_option("--device-to-measure", device_to_measure, "Device to measure")->required();

  std::string schedule_url;
  app.add_option("--schedule-url", schedule_url, "Schedule URL")->required();

  size_t n_schedules_to_run;
  app.add_option("--n-schedules-to-run", n_schedules_to_run, "Number of schedules to run")
      ->default_val(10);

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id != device_to_measure) {
    return 0;
  }

  const auto json = fetch_json_from_url(schedule_url);

  const auto schedules = readSchedulesFromJson(json);

  auto n_schedules = schedules.size();

  spdlog::info("Loaded {} schedules", n_schedules);

  for (size_t i = 0; i < n_schedules; ++i) {
    std::cout << "--------------------------------" << std::endl;
    schedules[i].print(i);
  }

  BM_pipe_warmup(schedules[2]);

  spdlog::info("Running {}/{} schedules", n_schedules_to_run, n_schedules);

  for (size_t i = 0; i < n_schedules_to_run; ++i) {
    std::cout << "\n--------------------------------" << std::endl;
    BM_pipe_cifar_dense_vk_schedule_auto(schedules[i]);
  }

  return 0;
}
