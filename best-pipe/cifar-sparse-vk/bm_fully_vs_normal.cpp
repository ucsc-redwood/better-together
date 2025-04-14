#include <omp.h>

#include "builtin-apps/app.hpp"  // for 'g_big_cores', 'g_medium_cores', 'g_little_cores'
#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/pipeline/record.hpp"
#include "builtin-apps/pipeline/task.hpp"
#include "builtin-apps/pipeline/worker.hpp"
#include "common.hpp"

// ----------------------------------------------------------------------------
// Realistic Benchmark:  Stage
// ----------------------------------------------------------------------------

using MyTask = Task<cifar_sparse::v2::AppData>;

namespace android {

static void BM_run_stage_full(const int stage_to_measure) {
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;

  // Main benchmark
  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kLittleCore},
                                        {1, ProcessorType::kMediumCore},
                                        {2, ProcessorType::kBigCore},
                                        {3, ProcessorType::kVulkan},
                                    });

    // Start simulation threads
    std::atomic<bool> done = false;

    // std::thread t0([&]() {
    //   thread_local MyTask task(disp.get_mr());
    //   auto start_time = std::chrono::steady_clock::now();

    //   while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() -
    //                                                           start_time)
    //              .count() < 5) {
    //     cifar_sparse::omp::v2::dispatch_multi_stage(g_little_cores,
    //                                                 g_little_cores.size(),
    //                                                 task.appdata,
    //                                                 stage_to_measure,
    //                                                 stage_to_measure);
    //   }

    //   done.store(true, std::memory_order_relaxed);
    // });

    std::thread t0([&]() {
      thread_local MyTask task(disp.get_mr());
      thread_local size_t counter = 0;

      while (!done.load(std::memory_order_relaxed)) {
        RecordManager::instance().start_tick(counter, 0);

        cifar_sparse::omp::v2::dispatch_multi_stage(g_little_cores,
                                                    g_little_cores.size(),
                                                    task.appdata,
                                                    stage_to_measure,
                                                    stage_to_measure);

        RecordManager::instance().end_tick(counter, 0);

        counter++;
      }
    });

    std::thread t1([&]() {
      thread_local MyTask task(disp.get_mr());
      thread_local size_t counter = 0;

      while (!done.load(std::memory_order_relaxed)) {
        RecordManager::instance().start_tick(counter, 1);

        cifar_sparse::omp::v2::dispatch_multi_stage(g_medium_cores,
                                                    g_medium_cores.size(),
                                                    task.appdata,
                                                    stage_to_measure,
                                                    stage_to_measure);

        RecordManager::instance().end_tick(counter, 1);

        counter++;
      }
    });

    std::thread t2([&]() {
      thread_local MyTask task(disp.get_mr());
      thread_local size_t counter = 0;

      while (!done.load(std::memory_order_relaxed)) {
        RecordManager::instance().start_tick(counter, 2);

        cifar_sparse::omp::v2::dispatch_multi_stage(
            g_big_cores, g_big_cores.size(), task.appdata, stage_to_measure, stage_to_measure);

        RecordManager::instance().end_tick(counter, 2);

        counter++;
      }
    });

    std::thread t3([&]() {
      thread_local MyTask task(disp.get_mr());
      thread_local size_t counter = 0;

      while (!done.load(std::memory_order_relaxed)) {
        RecordManager::instance().start_tick(counter, 3);

        disp.dispatch_multi_stage(task.appdata, stage_to_measure, stage_to_measure);

        RecordManager::instance().end_tick(counter, 3);

        counter++;
      }
    });

    std::this_thread::sleep_for(std::chrono::seconds(5));

    done.store(true, std::memory_order_relaxed);

    t0.join();
    t1.join();
    t2.join();
    t3.join();

    RecordManager::instance().print_processor_type_stats(ProcessorType::kLittleCore);
    RecordManager::instance().print_processor_type_stats(ProcessorType::kMediumCore);
    RecordManager::instance().print_processor_type_stats(ProcessorType::kBigCore);
    RecordManager::instance().print_processor_type_stats(ProcessorType::kVulkan);
  }
}

}  // namespace android

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

// e.g.,
// xmake r bm-table-cifar-sparse-vk --stage 1 --device-to-measure 3A021JEHN02756
// xmake r bm-table-cifar-sparse-vk --stage 1 --device jetson --device-to-measure jetson --full

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN

  std::string device_to_measure;
  app.add_option("--device-to-measure", device_to_measure, "Device to measure")->required();

  int stage = 0;
  app.add_option("--stage", stage, "Stage to run")->required();

  PARSE_ARGS_END

  // So that we don't output anything when the device is not the one we want to measure
  if (device_to_measure != g_device_id) {
    return 0;
  }

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  // Run for all processor types when using full benchmark
  android::BM_run_stage_full(stage);

  return 0;
}
