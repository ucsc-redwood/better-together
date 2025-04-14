#include "../templates_vk.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/tree/safe_tree_appdata.hpp"
#include "builtin-apps/tree/vulkan/dispatchers.hpp"
#include "builtin-apps/tree/vulkan/vk_appdata.hpp"
#include "run_stages.hpp"
#include "task.hpp"

int main(int argc, char **argv) {
  PARSE_ARGS_BEGIN;

  int schedule_index = 0;  // Default to first schedule
  app.add_option("-i,--index", schedule_index, "Schedule index (0-9, or -1 for all schedules)")
      ->required();

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::off);

  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  std::vector<tree::vulkan::VkAppData_Safe> preallocated_data =
      init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  // Queue definitions:
  moodycamel::ConcurrentQueue<Task *> q_0_1;
  moodycamel::ConcurrentQueue<Task *> q_1_2;

  std::vector<double> task_times;
  task_times.reserve(num_tasks);
  auto start_time = std::chrono::high_resolution_clock::now();

  // // Thread calls:
  // std::thread t1([&]() {
  //   chunk<Task, tree::vulkan::VkAppData>(q_input, &q_0_1, vulkan::run_gpu_stages<1, 1>);
  // });
  // std::thread t2([&]() {
  //   chunk<Task, tree::AppData>(
  //       q_0_1, &q_1_2, omp::run_multiple_stages<2, 4, ProcessorType::kLittleCore, 2>);
  // });
  // std::thread t3([&]() {
  //   chunk<Task, tree::AppData>(
  //       q_1_2, nullptr, omp::run_multiple_stages<5, 6, ProcessorType::kBigCore, 2>);
  // });

  // Thread calls:
  std::thread t1([&]() {
    chunk<Task, tree::SafeAppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kBigCore, 2>);
  });
  std::thread t2([&]() {
    chunk<Task, tree::vulkan::VkAppData_Safe>(q_0_1, &q_1_2, vulkan::run_gpu_stages<2, 4>);
  });
  std::thread t3([&]() {
    chunk<Task, tree::SafeAppData>(
        q_1_2, nullptr, omp::run_multiple_stages<5, 6, ProcessorType::kBigCore, 2>);
  });

  // Thread joins:
  t1.join();
  t2.join();
  t3.join();

  auto end_time = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
  task_times.push_back(elapsed / num_tasks);

  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();

  spdlog::info("Average task time: {} ms", avg_task_time);
  std::cout << "avg_task_time: " << avg_task_time << std::endl;

  // ---------------------------------------------------------------------

  spdlog::info("Done");

  return 0;
}
