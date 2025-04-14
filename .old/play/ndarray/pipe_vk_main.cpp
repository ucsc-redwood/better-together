
#include <functional>

#include "builtin-apps/app.hpp"
#include "builtin-apps/config_reader.hpp"
#include "omp/dispatchers.hpp"
#include "spsc_queue.hpp"
#include "vulkan/dispatcher.hpp"

constexpr size_t kNumTasks = 100;

struct Task {
  static uint32_t uid_counter;
  uint32_t uid;

  // ----------------------------------
  cifar_dense::AppDataBatch appdata;

  explicit Task(std::pmr::memory_resource* mr) : appdata(mr) { uid = uid_counter++; }
  // ----------------------------------
};

uint32_t Task::uid_counter = 0;

using ProcessFunction = std::function<void(Task&)>;

static void worker_thread(SPSCQueue<Task*, 1024>& in_queue,
                          SPSCQueue<Task*, 1024>* out_queue,
                          ProcessFunction process_function) {
  for (size_t i = 0; i < kNumTasks; ++i) {
    Task* task = nullptr;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

    process_function(*task);

    // Forward processed task
    if (out_queue) {
      out_queue->enqueue(std::move(task));
    }
  }
}

void process_chunk(const ChunkConfig& config,
                   SPSCQueue<Task*>& in_queue,
                   SPSCQueue<Task*>& out_queue,
                   cifar_dense::vulkan::VulkanDispatcher& disp) {
  if (config.exec_model == ExecutionModel::kOMP) {
    // OMP
    worker_thread(in_queue, &out_queue, [&](Task& task) {
      omp::dispatch_multi_stage(config.start_stage,
                                config.end_stage,
                                config.proc_type.value(),
                                config.num_threads.value(),
                                task.appdata);
    });

  } else if (config.exec_model == ExecutionModel::kGPU) {
    // GPU
    worker_thread(in_queue, &out_queue, [&](Task& task) {
      disp.dispatch_multi_stage(task.appdata, config.start_stage, config.end_stage);
    });

  } else {
    throw std::runtime_error("Invalid execution model");
  }
}

void execute_schedule(const std::vector<ChunkConfig>& chunk_configs) {
  cifar_dense::vulkan::VulkanDispatcher disp;

  std::vector<SPSCQueue<Task*>> concur_qs(chunk_configs.size() + 1);

  // Setting up the input queue
  for (size_t i = 0; i < kNumTasks; ++i) {
    concur_qs[0].enqueue(new Task(disp.get_mr()));
  }

  std::vector<std::thread> threads;

  // Benchmark start
  const auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < chunk_configs.size(); ++i) {
    threads.emplace_back(
        [&, i]() { process_chunk(chunk_configs[i], concur_qs[i], concur_qs[i + 1], disp); });
  }

  for (auto& t : threads) {
    t.join();
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  const double avg_task_time = duration.count() / kNumTasks;
  std::cout << "Time per task: " << avg_task_time << " ms" << std::endl;
}

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN;

  std::string schedule_file_path;
  app.add_option("-f,--file", schedule_file_path, "Schedule file path")->required();

  PARSE_ARGS_END;

  // ------------------------------------------------------------------------------------------------

  const auto chunks = readChunksFromJson(fs::path(schedule_file_path));
  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  execute_schedule(chunks);

  // ------------------------------------------------------------------------------------------------

  spdlog::info("Done");
  return 0;
}
