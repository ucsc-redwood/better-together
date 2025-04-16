#include <omp.h>

#include "builtin-apps/app.hpp"
#include "common.hpp"
#include "schedules.hpp"

// ----------------------------------------------------------------------------
// Schedule Auto
// ----------------------------------------------------------------------------
using QueueT = SPSCQueue<MyTask*, kPoolSize>;

static void BM_pipe_cifar_sparse_vk_schedule_auto(const Schedule schedule) {
  auto n_chunks = schedule.n_chunks();

  // Initialize the dispatcher and queues
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;
  std::vector<std::unique_ptr<MyTask>> preallocated_tasks;
  std::vector<QueueT> queues(n_chunks);
  for (size_t i = 0; i < kPoolSize; ++i) {
    preallocated_tasks.emplace_back(std::make_unique<MyTask>(disp.get_mr()));
    queues[0].enqueue(preallocated_tasks.back().get());
  }

  Logger<kNumToProcess> logger(schedule);

  // Run the schedule
  {
    std::vector<std::thread> threads;

    for (size_t i = 0; i < n_chunks; ++i) {
      QueueT& q_in = queues[i];
      QueueT& q_out = queues[(i + 1) % n_chunks];

      const int start = schedule.chunks[i].indices.front() + 1;
      const int end = schedule.chunks[i].indices.back() + 1;

      const ProcessorType pt = schedule.chunks[i].core;

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

  schedule.print();

  std::cout << "### Python End ###" << std::endl;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN

  // int schedule_id = 0;
  // app.add_option("--schedule", schedule_id, "Schedule to run")->required();

  PARSE_ARGS_END

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id != "3A021JEHN02756") {
    return 0;
  }

  // assert(schedule_id >= 1 && schedule_id <= 10);

  // Schedule schedule = schedules[schedule_id - 1];

  for (const auto& schedule : device_3A021JEHN02756::schedules) {
    BM_pipe_cifar_sparse_vk_schedule_auto(schedule);

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  return 0;
}
