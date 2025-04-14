#include <omp.h>

#include "builtin-apps/app.hpp"  // for 'g_big_cores', 'g_medium_cores', 'g_little_cores'
#include "common.hpp"

// ----------------------------------------------------------------------------
// Realistic Benchmark:  Stage
// ----------------------------------------------------------------------------

using MyTask = Task<cifar_dense::v2::AppData>;

namespace android {

static void BM_run_stage_full(const int stage_to_measure) {
  SETUP_DATA;

  // Main benchmark
  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kLittleCore},
                                        {1, ProcessorType::kMediumCore},
                                        {2, ProcessorType::kBigCore},
                                        {3, ProcessorType::kVulkan},
                                    });

    // little core
    std::thread t0(
        worker_thread_record<MyTask>, 0, std::ref(q_0), std::ref(q_1), [&](MyTask& task) {
          cifar_dense::omp::v2::dispatch_multi_stage(g_little_cores,
                                                     g_little_cores.size(),
                                                     task.appdata,
                                                     stage_to_measure,
                                                     stage_to_measure);
        });

    // medium core
    std::thread t1(
        worker_thread_record<MyTask>, 1, std::ref(q_1), std::ref(q_2), [&](MyTask& task) {
          cifar_dense::omp::v2::dispatch_multi_stage(g_medium_cores,
                                                     g_medium_cores.size(),
                                                     task.appdata,
                                                     stage_to_measure,
                                                     stage_to_measure);
        });

    // big core
    std::thread t2(
        worker_thread_record<MyTask>, 2, std::ref(q_2), std::ref(q_3), [&](MyTask& task) {
          cifar_dense::omp::v2::dispatch_multi_stage(
              g_big_cores, g_big_cores.size(), task.appdata, stage_to_measure, stage_to_measure);
        });

    // vulkan core
    std::thread t3(
        worker_thread_record<MyTask>, 3, std::ref(q_3), std::ref(q_0), [&](MyTask& task) {
          disp.dispatch_multi_stage(task.appdata, stage_to_measure, stage_to_measure);
        });

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

static void BM_run_stage_non_full(const int stage_to_measure,
                                  const ProcessorType core_type_to_measure) {
  SETUP_DATA;

  // find cores
  std::vector<int> cores_to_use;
  ProcessorType counter_part_a;
  ProcessorType counter_part_b;
  ProcessorType counter_part_c;

  if (core_type_to_measure == ProcessorType::kLittleCore) {
    cores_to_use = g_little_cores;
    counter_part_a = ProcessorType::kMediumCore;
    counter_part_b = ProcessorType::kBigCore;
    counter_part_c = ProcessorType::kVulkan;
  } else if (core_type_to_measure == ProcessorType::kMediumCore) {
    cores_to_use = g_medium_cores;
    counter_part_a = ProcessorType::kLittleCore;
    counter_part_b = ProcessorType::kBigCore;
    counter_part_c = ProcessorType::kVulkan;
  } else if (core_type_to_measure == ProcessorType::kBigCore) {
    cores_to_use = g_big_cores;
    counter_part_a = ProcessorType::kLittleCore;
    counter_part_b = ProcessorType::kMediumCore;
    counter_part_c = ProcessorType::kVulkan;
  } else if (core_type_to_measure == ProcessorType::kVulkan) {
    counter_part_a = ProcessorType::kLittleCore;
    counter_part_b = ProcessorType::kMediumCore;
    counter_part_c = ProcessorType::kBigCore;
  } else {
    throw "asdasdasdasdasdasdasda";
  }

  // Main benchmark
  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, core_type_to_measure},
                                        {1, counter_part_a},
                                        {2, counter_part_b},
                                        {3, counter_part_c},
                                    });

    // Core to measure
    std::thread t0(
        worker_thread_record<MyTask>, 0, std::ref(q_0), std::ref(q_1), [&](MyTask& task) {
          if (core_type_to_measure == ProcessorType::kVulkan) {
            disp.dispatch_multi_stage(task.appdata, stage_to_measure, stage_to_measure);
          } else {
            cifar_dense::omp::v2::dispatch_multi_stage(cores_to_use,
                                                       cores_to_use.size(),
                                                       task.appdata,
                                                       stage_to_measure,
                                                       stage_to_measure);
          }
        });

    std::thread t1(worker_thread_record<MyTask>, 1, std::ref(q_1), std::ref(q_2), [&](MyTask&) {});
    std::thread t2(worker_thread_record<MyTask>, 2, std::ref(q_2), std::ref(q_3), [&](MyTask&) {});
    std::thread t3(worker_thread_record<MyTask>, 3, std::ref(q_3), std::ref(q_0), [&](MyTask&) {});

    t0.join();
    t1.join();
    t2.join();
    t3.join();

    RecordManager::instance().print_processor_type_stats(core_type_to_measure);
  }
}

}  // namespace android

// ----------------------------------------------------------------------------
// Jetson
// ----------------------------------------------------------------------------

namespace jetson {

static void BM_run_stage_full(const int stage_to_measure) {
  SETUP_DATA;

  // Main benchmark
  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, ProcessorType::kLittleCore},
                                        {1, ProcessorType::kVulkan},
                                    });

    // little core
    std::thread t0(
        worker_thread_record<MyTask>, 0, std::ref(q_0), std::ref(q_1), [&](MyTask& task) {
          cifar_dense::omp::v2::dispatch_multi_stage(g_little_cores,
                                                     g_little_cores.size(),
                                                     task.appdata,
                                                     stage_to_measure,
                                                     stage_to_measure);
        });

    // vulkan core
    std::thread t1(
        worker_thread_record<MyTask>, 1, std::ref(q_1), std::ref(q_0), [&](MyTask& task) {
          disp.dispatch_multi_stage(task.appdata, stage_to_measure, stage_to_measure);
        });

    t0.join();
    t1.join();

    RecordManager::instance().print_processor_type_stats(ProcessorType::kLittleCore);
    RecordManager::instance().print_processor_type_stats(ProcessorType::kVulkan);
  }
}

static void BM_run_stage_non_full(const int stage_to_measure,
                                  const ProcessorType target,
                                  const ProcessorType partner) {
  SETUP_DATA;

  // find cores
  std::vector<int> cores_to_use;
  if (target == ProcessorType::kLittleCore) {
    cores_to_use = g_little_cores;
  }

  // Main benchmark
  {
    RecordManager::instance().setup(kNumToProcess,
                                    {
                                        {0, target},
                                        {1, partner},
                                    });

    std::thread t0(
        worker_thread_record<MyTask>, 0, std::ref(q_0), std::ref(q_1), [&](MyTask& task) {
          if (target == ProcessorType::kVulkan) {
            disp.dispatch_multi_stage(task.appdata, stage_to_measure, stage_to_measure);
          } else {
            cifar_dense::omp::v2::dispatch_multi_stage(cores_to_use,
                                                       cores_to_use.size(),
                                                       task.appdata,
                                                       stage_to_measure,
                                                       stage_to_measure);
          }
        });

    std::thread t1(worker_thread_record<MyTask>, 1, std::ref(q_1), std::ref(q_0), [](MyTask&) {});

    t0.join();
    t1.join();

    RecordManager::instance().print_processor_type_stats(target);
  }
}

}  // namespace jetson

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

// e.g.,
// xmake r bm-table-cifar-dense-vk --stage 1 -l off --device-to-measure 3A021JEHN02756
// xmake r bm-table-cifar-dense-vk --stage 1 -l off --device jetson --device-to-measure jetson --full

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN

  std::string device_to_measure;
  app.add_option("--device-to-measure", device_to_measure, "Device to measure")->required();

  int stage = 0;
  app.add_option("--stage", stage, "Stage to run")->required();

  bool full = false;
  app.add_flag("--full", full, "Run fully occupied benchmark");

  PARSE_ARGS_END

  // So that we don't output anything when the device is not the one we want to measure
  if (device_to_measure != g_device_id) {
    return 0;
  }

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (device_to_measure == "jetson") {
    if (full) {
      jetson::BM_run_stage_full(stage);
    } else {
      jetson::BM_run_stage_non_full(stage, ProcessorType::kLittleCore, ProcessorType::kVulkan);
      jetson::BM_run_stage_non_full(stage, ProcessorType::kVulkan, ProcessorType::kLittleCore);
    }
  } else {
    if (full) {
      android::BM_run_stage_full(stage);
    } else {
      android::BM_run_stage_non_full(stage, ProcessorType::kLittleCore);
      android::BM_run_stage_non_full(stage, ProcessorType::kMediumCore);
      android::BM_run_stage_non_full(stage, ProcessorType::kBigCore);
      android::BM_run_stage_non_full(stage, ProcessorType::kVulkan);
    }
  }

  return 0;
}
