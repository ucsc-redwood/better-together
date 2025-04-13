#include <benchmark/benchmark.h>
#include <omp.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/config_reader.hpp"
#include "builtin-apps/curl_json.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"
#include "builtin-apps/pipeline/task.hpp"
#include "builtin-apps/pipeline/worker.hpp"

using MyTask = Task<cifar_sparse::v2::AppData>;

// ----------------------------------------------------------------------------
// Specialized Schedules
// ----------------------------------------------------------------------------

std::string g_schedule_base_url;
int g_schedule_id;

static void BM_pipe_cifar_sparse_vk_schedule_best(benchmark::State& state) {
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;
  std::vector<std::unique_ptr<MyTask>> preallocated_tasks;

  // --------------------------------------------------------------------------

  const auto schedule_json =
      fetch_json_from_url(make_full_url(g_device_id, "CifarSparse", g_schedule_id));
  const auto chunk_configs = readChunksFromJson(schedule_json);

  std::vector<SPSCQueue<MyTask*, kPoolSize>> concur_qs(chunk_configs.size() + 1);
  std::vector<std::thread> threads;

  for (size_t i = 0; i < kPoolSize; ++i) {
    preallocated_tasks.emplace_back(std::make_unique<MyTask>(disp.get_mr()));
    concur_qs[0].enqueue(preallocated_tasks.back().get());
  }

  for (auto _ : state) {
    for (size_t i = 0; i < chunk_configs.size(); ++i) {
      threads.emplace_back([&, i]() {
        worker_thread<MyTask>(
            std::ref(concur_qs[i]), std::ref(concur_qs[i + 1]), [&](MyTask& task) {
              if (chunk_configs[i].exec_model == ExecutionModel::kVulkan) {
                // GPU/Vulkan
                disp.dispatch_multi_stage(
                    task.appdata, chunk_configs[i].start_stage, chunk_configs[i].end_stage);
              } else {
                // CPU/OpenMP
                cifar_sparse::omp::v2::dispatch_multi_stage(
                    g_medium_cores, g_medium_cores.size(), task.appdata, 5, 5);
              }
            });
      });
    }

    for (auto& t : threads) {
      t.join();
    }
  }
  // --------------------------------------------------------------------------
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_best)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN;

  app.add_option("--base-dir", g_schedule_base_url, "Base directory")
      ->default_str(kDefaultScheduleBaseDir);
  app.add_option("-i,--index", g_schedule_id, "Schedule file index")->required();

  PARSE_ARGS_END;

  assert(g_schedule_id > 0);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup
  return 0;
}
