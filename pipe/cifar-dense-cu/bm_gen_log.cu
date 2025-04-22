#include <omp.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/config_reader.hpp"
#include "builtin-apps/curl_json.hpp"
#include "const.hpp"

// ----------------------------------------------------------------------------
// Schedule Warmup
// ----------------------------------------------------------------------------

static void BM_pipe_warmup(const Schedule schedule) {
  auto n_chunks = schedule.n_chunks();

  DispatcherT disp;

  const std::vector<AppDataPtr> dataset = make_dataset(disp, kPoolSize);

  std::vector<QueueT> queues(n_chunks);
  for (size_t i = 0; i < kPoolSize; ++i) {
    queues[0].enqueue(dataset[i]);
  }

  // Run the schedule
  {
    std::vector<std::thread> threads;

    for (size_t chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
      QueueT& q_in = queues[chunk_id];
      QueueT& q_out = queues[(chunk_id + 1) % n_chunks];

      const int start = schedule.start_stage(chunk_id) + 1;
      const int end = schedule.end_stage(chunk_id);

      const ProcessorType pt = get_processor_type_from_chunk_config(schedule.chunks[chunk_id]);

      constexpr auto num_warmup_items = 5;

      if (pt == ProcessorType::kCuda) {
        threads.emplace_back(
            worker,
            std::ref(q_in),
            std::ref(q_out),
            [&disp, start, end](AppDataPtr& app) { disp.dispatch_multi_stage(*app, start, end); },
            num_warmup_items,
            chunk_id == n_chunks - 1);
      } else {
        threads.emplace_back(
            worker,
            std::ref(q_in),
            std::ref(q_out),
            [pt, start, end](AppDataPtr& app) {
              const auto cores = get_cores_by_type(pt);

              cifar_dense::omp::dispatch_multi_stage(cores, cores.size(), *app, start, end);
            },
            num_warmup_items,
            chunk_id == n_chunks - 1);
      }
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }
}

// ----------------------------------------------------------------------------
// Schedule Auto
// ----------------------------------------------------------------------------

static void BM_pipe_cifar_dense_vk_schedule_auto(const Schedule schedule) {
  auto n_chunks = schedule.n_chunks();

  DispatcherT disp;

  const std::vector<AppDataPtr> dataset = make_dataset(disp, kPoolSize);

  std::vector<QueueT> queues(n_chunks);
  for (size_t i = 0; i < kPoolSize; ++i) {
    queues[0].enqueue(dataset[i]);
  }

  Logger<kNumToProcess> logger;

  // Run the schedule
  std::vector<std::thread> threads;

  for (size_t chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
    QueueT& q_in = queues[chunk_id];
    QueueT& q_out = queues[(chunk_id + 1) % n_chunks];

    const int start = schedule.start_stage(chunk_id) + 1;
    const int end = schedule.end_stage(chunk_id);

    const ProcessorType pt = get_processor_type_from_chunk_config(schedule.chunks[chunk_id]);

    if (pt == ProcessorType::kCuda) {
      threads.emplace_back(
          worker_with_record,
          chunk_id,
          std::ref(logger),
          std::ref(q_in),
          std::ref(q_out),
          [&disp, start, end](AppDataPtr& app) {
            // Launch GPU
            disp.dispatch_multi_stage(*app, start, end);
          },
          kNumToProcess,
          chunk_id == n_chunks - 1);
    } else {
      threads.emplace_back(
          worker_with_record,
          chunk_id,
          std::ref(logger),
          std::ref(q_in),
          std::ref(q_out),
          [pt, start, end](AppDataPtr& app) {
            // Launch OMP
            const auto cores = get_cores_by_type(pt);
            cifar_dense::omp::dispatch_multi_stage(cores, cores.size(), *app, start, end);
          },
          kNumToProcess,
          chunk_id == n_chunks - 1);
    }
  }

  for (auto& thread : threads) {
    thread.join();
  }

  logger.dump_records_for_python(schedule);
}

// Test schedule

// ----------------------------------------------------------------------------
// Main
//
// Run with:
// xmake r bm-gen-logs-cifar-dense-cu -l off --device pc --device-to-measure pc --schedule url
//
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN;

  std::string device_to_measure;
  app.add_option("--device-to-measure", device_to_measure, "Device to measure")->required();

  std::string schedule_url;
  app.add_option("--schedule-url", schedule_url, "Schedule URL");

  size_t n_schedules_to_run = 0;  // 0 means run all schedules
  app.add_option(
      "--n-schedules-to-run", n_schedules_to_run, "Number of schedules to run (0 means run all)");

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id != device_to_measure) {
    return 0;
  }

  const Schedule test_schedule{
      .chunks =
          {
              {
                  .exec_model = ExecutionModel::kOMP,
                  .start_stage = 0,
                  .end_stage = 3,
                  .cpu_proc_type = ProcessorType::kLittleCore,
              },
              {
                  .exec_model = ExecutionModel::kCuda,
                  .start_stage = 4,
                  .end_stage = 6,
              },
          },
  };

  BM_pipe_warmup(test_schedule);

  // If schedule_url is empty or we fail to fetch the URL, just run warmup and quit
  if (schedule_url.empty()) {
    spdlog::info("No schedule URL provided. Running only warmup phase.");
    return 0;
  }

  std::vector<Schedule> schedules;
  try {
    const auto json = fetch_json_from_url(schedule_url);
    schedules = readSchedulesFromJson(json);
  } catch (const std::exception& e) {
    spdlog::error("Failed to fetch or parse schedules: {}", e.what());
    spdlog::info("Running only warmup phase.");
    return 0;
  }

  auto n_schedules = schedules.size();
  if (n_schedules == 0) {
    spdlog::info("No schedules found.");
    return 0;
  }

  spdlog::info("Loaded {} schedules", n_schedules);

  for (size_t i = 0; i < n_schedules; ++i) {
    std::cout << "--------------------------------" << std::endl;
    schedules[i].print(i);
  }

  // If n_schedules_to_run is 0 or greater than available schedules, run all schedules
  if (n_schedules_to_run == 0 || n_schedules_to_run > n_schedules) {
    n_schedules_to_run = n_schedules;
  }

  spdlog::info("Running {}/{} schedules", n_schedules_to_run, n_schedules);

  for (size_t i = 0; i < n_schedules_to_run; ++i) {
    std::cout << "\n--------------------------------" << std::endl;
    BM_pipe_cifar_dense_vk_schedule_auto(schedules[i]);
  }

  return 0;
}
