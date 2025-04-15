#include <omp.h>

#include "builtin-apps/app.hpp"  // for 'g_big_cores', 'g_medium_cores', 'g_little_cores'
#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/conf.hpp"
#include "builtin-apps/pipeline/record.hpp"
#include "builtin-apps/pipeline/task.hpp"
#include "builtin-apps/pipeline/worker.hpp"
#include "common.hpp"

// ----------------------------------------------------------------------------
// Realistic Benchmark:  Stage
// ----------------------------------------------------------------------------

using MyTask = Task<cifar_sparse::v2::AppData>;

namespace android {

std::atomic<bool> done = false;

// create a 2D table, 9 stage times 4 type of cores, initialize it with 0
// access by bm_table[stage][core_type] = value
std::array<std::array<int, 4>, 9> bm_table;

// Reset the done flag
void reset_done_flag() { done.store(false); }

void init_q(std::queue<MyTask*>& q, cifar_sparse::vulkan::v2::VulkanDispatcher& disp) {
  constexpr int kPhysicalTasks = 10;

  for (int i = 0; i < kPhysicalTasks; i++) {
    q.push(new MyTask(disp.get_mr()));
  }
}

void clean_up_q(std::queue<MyTask*>& q) {
  while (!q.empty()) {
    MyTask* task = q.front();
    q.pop();
    delete task;
  }
}

void similuation_thread(cifar_sparse::vulkan::v2::VulkanDispatcher& disp,
                        std::function<void(MyTask*)> func) {
  std::queue<MyTask*> q;
  init_q(q, disp);

  while (!done.load(std::memory_order_relaxed)) {
    MyTask* task = q.front();
    q.pop();

    func(task);

    // After done -> push back for reuse
    task->reset();
    q.push(task);
  }

  clean_up_q(q);
}

// static void BM_run_little_core(int start_stage, int end_stage, int seconds_to_run) {
static void BM_run_little_core(const ProcessorType pt, const int stage, const int seconds_to_run) {
  cifar_sparse::vulkan::v2::VulkanDispatcher disp;

  // Reset the done flag before starting a new benchmark
  reset_done_flag();

  auto start = std::chrono::high_resolution_clock::now();

  int total_processed = 0;  // Initialize here, now explicitly shown for clarity

  auto gpu_func = [&disp, &total_processed, stage](MyTask* task) {
    disp.dispatch_multi_stage(task->appdata, stage, stage);
    total_processed++;
  };

  std::vector<int> cores_to_use;
  if (pt == ProcessorType::kLittleCore) {
    cores_to_use = g_little_cores;
  } else if (pt == ProcessorType::kBigCore) {
    cores_to_use = g_big_cores;
  } else if (pt == ProcessorType::kMediumCore) {
    cores_to_use = g_medium_cores;
  }

  auto cpu_func = [&total_processed, stage, cores_to_use](MyTask* task) {
    cifar_sparse::omp::v2::dispatch_multi_stage(
        cores_to_use, cores_to_use.size(), task->appdata, stage, stage);
    total_processed++;
  };

  std::thread t1;
  if (pt == ProcessorType::kVulkan) {
    t1 = std::thread(similuation_thread, std::ref(disp), gpu_func);
  } else {
    t1 = std::thread(similuation_thread, std::ref(disp), cpu_func);
  }

  std::this_thread::sleep_for(std::chrono::seconds(seconds_to_run));
  done.store(true);

  t1.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  SPDLOG_DEBUG("Total processed: {}", total_processed);
  SPDLOG_DEBUG("Average time per task: {} ms", duration.count() / total_processed);

  // update the table

  // map ProcessorType::kLittleCore = 0, ProcessorType::kBigCore = 1, ProcessorType::kMediumCore =
  // 2, ProcessorType::kVulkan = 3
  bm_table[stage][static_cast<int>(pt)] = duration.count() / total_processed;
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

  // std::string device_to_measure;
  // app.add_option("--device-to-measure", device_to_measure, "Device to measure")->required();

  // int stage = 0;
  // app.add_option("--stage", stage, "Stage to run")->required();

  int start_stage = 1;
  int end_stage = 9;
  int seconds_to_run = 10;

  app.add_option("-s, --start-stage", start_stage, "Start stage");
  app.add_option("-e, --end-stage", end_stage, "End stage");
  app.add_option("-t, --seconds-to-run", seconds_to_run, "Seconds to run");

  PARSE_ARGS_END

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  // Run for all processor types when using full benchmark
  for (int stage = start_stage; stage <= end_stage; stage++) {
    android::BM_run_little_core(ProcessorType::kLittleCore, stage, seconds_to_run);
    android::BM_run_little_core(ProcessorType::kMediumCore, stage, seconds_to_run);
    android::BM_run_little_core(ProcessorType::kBigCore, stage, seconds_to_run);
    android::BM_run_little_core(ProcessorType::kVulkan, stage, seconds_to_run);
  }

  // Print the table in a nice 2D format
  std::cout << "\nBenchmark Results Table:\n";
  std::cout << "Stage | Little Core | Medium Core | Big Core | Vulkan \n";
  std::cout << "------|------------|-------------|----------|--------\n";
  for (int stage = start_stage; stage <= end_stage; stage++) {
    std::cout << std::setw(5) << stage << " | " << std::setw(11) << android::bm_table[stage][0]
              << " | " << std::setw(11) << android::bm_table[stage][2] << " | " << std::setw(8)
              << android::bm_table[stage][1] << " | " << std::setw(6) << android::bm_table[stage][3]
              << "\n";
  }
  // Time of Sum of all stage for each processor type
  int little_sum = 0, medium_sum = 0, big_sum = 0, vulkan_sum = 0;
  for (int stage = start_stage; stage <= end_stage; stage++) {
    little_sum += android::bm_table[stage][0];
    medium_sum += android::bm_table[stage][2];
    big_sum += android::bm_table[stage][1];
    vulkan_sum += android::bm_table[stage][3];
  }
  // print Sum of stage 1-9 for each processor type
  std::cout << "Sum of stage 1-9:" << std::endl;
  std::cout << "Little Core: " << little_sum << " ms" << std::endl;
  std::cout << "Medium Core: " << medium_sum << " ms" << std::endl;
  std::cout << "Big Core: " << big_sum << " ms" << std::endl;
  std::cout << "Vulkan: " << vulkan_sum << " ms" << std::endl;

  return 0;
}
