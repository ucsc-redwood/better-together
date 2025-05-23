#include <nvtx3/nvToolsExt.h>

#include <functional>

#include "builtin-apps/app.hpp"
#include "builtin-apps/common/cuda/cu_mem_resource.cuh"
#include "builtin-apps/config_reader.hpp"
#include "cuda/dispatchers.cuh"
#include "omp/dispatchers.hpp"
#include "spsc_queue.hpp"

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
                   cuda::CudaDispatcher<cuda::CudaPinnedResource>& disp) {
  if (config.exec_model == ExecutionModel::kOMP) {
    // OMP
    worker_thread(in_queue, &out_queue, [&](Task& task) {
      omp::dispatch_multi_stage(config.start_stage,
                                config.end_stage,
                                config.proc_type.value(),
                                config.num_threads.value(),
                                task.appdata);
    });

    // TODO: Implement OMP execution
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
  cuda::CudaDispatcher<cuda::CudaPinnedResource> disp;

  std::vector<SPSCQueue<Task*>> concur_qs(chunk_configs.size() + 1);

  // Setting up the input queue
  for (size_t i = 0; i < kNumTasks; ++i) {
    concur_qs[0].enqueue(new Task(&disp.get_mr()));
  }

  std::vector<std::thread> threads;

  // Benchmark start
  cudaEvent_t start, stop;
  float milliseconds = 0;
  CheckCuda(cudaEventCreate(&start));
  CheckCuda(cudaEventCreate(&stop));
  CheckCuda(cudaEventRecord(start, 0));

  nvtxRangePushA("Compute");

  for (int i = 0; i < chunk_configs.size(); ++i) {
    threads.emplace_back(
        [&, i]() { process_chunk(chunk_configs[i], concur_qs[i], concur_qs[i + 1], disp); });
  }

  for (auto& t : threads) {
    t.join();
  }

  nvtxRangePop();

  CheckCuda(cudaEventRecord(stop, 0));
  CheckCuda(cudaEventSynchronize(stop));
  CheckCuda(cudaEventElapsedTime(&milliseconds, start, stop));
  CheckCuda(cudaEventDestroy(start));
  CheckCuda(cudaEventDestroy(stop));

  double avg_task_time = milliseconds / kNumTasks;
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
