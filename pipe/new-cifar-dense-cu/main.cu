#include <concurrentqueue.h>
#include <spdlog/spdlog.h>

#include <thread>

#include "../templates.hpp"
#include "../templates_cu.cuh"
#include "task.hpp"

// ---------------------------------------------------------------------
#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-dense/cuda/dispatchers.cuh"
#include "builtin-apps/cifar-dense/omp/dispatchers.hpp"
#include "builtin-apps/common/cuda/manager.cuh"
// ---------------------------------------------------------------------

#include "../config_reader.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/common/cuda/helpers.cuh"

// ---------------------------------------------------------------------
// OMP
// ---------------------------------------------------------------------

namespace omp {

constexpr std::array<void (*)(cifar_dense::AppData &), 9> cpu_stages = {
    cifar_dense::omp::process_stage_1,
    cifar_dense::omp::process_stage_2,
    cifar_dense::omp::process_stage_3,
    cifar_dense::omp::process_stage_4,
    cifar_dense::omp::process_stage_5,
    cifar_dense::omp::process_stage_6,
    cifar_dense::omp::process_stage_7,
    cifar_dense::omp::process_stage_8,
    cifar_dense::omp::process_stage_9,
};

void run_multiple_stages(const int start_stage,
                         const int end_stage,
                         const ProcessorType pt,
                         const int num_threads,
                         cifar_dense::AppData &data) {
#pragma omp parallel num_threads(num_threads)
  {
    // Bind to core
    if (pt == ProcessorType::kLittleCore) {
      bind_thread_to_cores(g_little_cores);
    } else if (pt == ProcessorType::kMediumCore) {
      bind_thread_to_cores(g_medium_cores);
    } else if (pt == ProcessorType::kBigCore) {
      bind_thread_to_cores(g_big_cores);
    }

    for (int i = start_stage - 1; i < end_stage; ++i) {
      cpu_stages[i](data);
    }
  }
}

}  // namespace omp

// ---------------------------------------------------------------------
// CUDA
// ---------------------------------------------------------------------

namespace cuda {

constexpr std::array<void (*)(cifar_dense::AppData &), 9> gpu_stages = {
    cifar_dense::cuda::process_stage_1,
    cifar_dense::cuda::process_stage_2,
    cifar_dense::cuda::process_stage_3,
    cifar_dense::cuda::process_stage_4,
    cifar_dense::cuda::process_stage_5,
    cifar_dense::cuda::process_stage_6,
    cifar_dense::cuda::process_stage_7,
    cifar_dense::cuda::process_stage_8,
    cifar_dense::cuda::process_stage_9,
};

void run_multiple_stages(const int start_stage,
                         const int end_stage,
                         cifar_dense::AppData &data,
                         cuda::CudaManager &mgr) {
  for (int s = start_stage; s <= end_stage; ++s) {
    gpu_stages[s - 1](data);
  }

  CheckCuda(cudaStreamSynchronize(mgr.get_stream()));
}
}  // namespace cuda

// template <typename TaskType, typename AppDataType>
void process_chunk_detail(moodycamel::ConcurrentQueue<Task *> &q_in,
                          moodycamel::ConcurrentQueue<Task *> &q_out,
                          std::function<void(cifar_dense::AppData &)> func) {
  while (true) {
    Task *task = nullptr;
    if (q_in.try_dequeue(task)) {
      if (task == nullptr) {
        // Sentinel => pass it on if there's a next queue and stop
        q_out.enqueue(nullptr);
        break;
      }

      // -----------------------------------
      func(*task->data);
      // -----------------------------------

      // If there's a next queue, pass the task along
      q_out.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

// ---------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------

void process_chunk(const ChunkConfig &config,
                   moodycamel::ConcurrentQueue<Task *> &in_queue,
                   moodycamel::ConcurrentQueue<Task *> &out_queue,
                   cuda::CudaManager &mgr) {
  if (config.exec_model == ExecutionModel::kOMP) {
    process_chunk_detail(in_queue, out_queue, [&config](cifar_dense::AppData &data) {
      omp::run_multiple_stages(config.start_stage,
                               config.end_stage,
                               config.proc_type.value(),
                               config.num_threads.value(),
                               data);
    });

  } else if (config.exec_model == ExecutionModel::kGPU) {
    process_chunk_detail(in_queue, out_queue, [&config, &mgr](cifar_dense::AppData &data) {
      cuda::run_multiple_stages(config.start_stage, config.end_stage, data, mgr);
    });

  } else {
    throw std::runtime_error("Invalid execution model");
  }
}

// ---------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------

static void schedule_jetson_CifarDense(const std::vector<ChunkConfig> &chunk_configs) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  constexpr size_t num_tasks = 100;
  auto preallocated_data = init_appdata<cifar_dense::AppData>(&mgr.get_mr(), num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  // ---------------------------------------------------------------------

  std::vector<moodycamel::ConcurrentQueue<Task *>> concur_qs(chunk_configs.size() + 1);
  concur_qs[0] = std::move(q_input);

  std::vector<std::thread> threads;

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------

  for (int i = 0; i < chunk_configs.size(); ++i) {
    threads.emplace_back(
        [&, i]() { process_chunk(chunk_configs[i], concur_qs[i], concur_qs[i + 1], mgr); });
  }

  for (auto &t : threads) {
    t.join();
  }
  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Time per task: " << avg_task_time << " ms" << std::endl;
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char **argv) {
  PARSE_ARGS_BEGIN;

  // take a path to the schedule file
  std::string schedule_file_path;
  app.add_option("-f,--file", schedule_file_path, "Schedule file path")->required();

  PARSE_ARGS_END;

  auto chunks = readChunksFromJson(fs::path(schedule_file_path));

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  warmup();

  schedule_jetson_CifarDense(chunks);

  return 0;
}
