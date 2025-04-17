#include <omp.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/config_reader.hpp"
#include "builtin-apps/curl_json.hpp"
#include "common.hpp"

// ----------------------------------------------------------------------------
// Schedule Auto
// ----------------------------------------------------------------------------
using QueueT = SPSCQueue<MyTask*, kPoolSize>;

static void BM_pipe_warmup(const Schedule schedule) {
  auto n_chunks = schedule.n_chunks();

  // Initialize the dispatcher and queues
  cifar_dense::vulkan::v2::VulkanDispatcher disp;
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
        threads.emplace_back(create_thread(q_in, q_out, disp, start, end));
      } else {
        threads.emplace_back(create_thread(q_in, q_out, get_cores_by_type(pt), start, end));
      }
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }
}

static void BM_pipe_cifar_dense_vk_schedule_auto(const size_t id, const Schedule schedule) {
  auto n_chunks = schedule.n_chunks();

  // Initialize the dispatcher and queues
  cifar_dense::vulkan::v2::VulkanDispatcher disp;
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
        threads.emplace_back(create_thread_record(i, logger, q_in, q_out, disp, start, end));
      } else {
        threads.emplace_back(
            create_thread_record(i, logger, q_in, q_out, get_cores_by_type(pt), start, end));
      }
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

  std::cout << "### Python Begin ###" << std::endl;

  logger.dump_records_for_python();

  schedule.print(id);

  std::cout << "### Python End ###" << std::endl;
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
      ->default_val(std::numeric_limits<size_t>::max());

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

  BM_pipe_warmup(schedules[4]);

  spdlog::info("Running {}/{} schedules", n_schedules_to_run, n_schedules);

  constexpr auto ignore_first_n_schedules = 4;
  for (size_t i = ignore_first_n_schedules; i < n_schedules_to_run; ++i) {
    BM_pipe_cifar_dense_vk_schedule_auto(i, schedules[i]);
  }

  return 0;
}
