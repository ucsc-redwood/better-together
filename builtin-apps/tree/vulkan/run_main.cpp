#include <queue>

#include "../omp/dispatchers.hpp"
#include "builtin-apps/app.hpp"
#include "dispatchers.hpp"
#include "spdlog/spdlog.h"

struct Task {
  uint32_t uid;
  tree::vulkan::VkAppData_Safe appdata;

  explicit Task(uint32_t uid, kiss_vk::VulkanMemoryResource::memory_resource* mr)
      : uid(uid), appdata(mr) {}
};

void process(std::queue<std::shared_ptr<Task>>& queue, tree::vulkan::VulkanDispatcher& disp) {
  while (!queue.empty()) {
    auto task = queue.front();
    queue.pop();

    spdlog::info("Dispatching task {}", task->uid);

    // disp.dispatch_multi_stage(task->appdata, 1, 7);
    tree::omp::dispatch_multi_stage(BIG_CORES, task->appdata, 1, 1);
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  tree::vulkan::VulkanDispatcher disp;

  constexpr auto n_tasks = 8;

  std::vector<std::shared_ptr<Task>> tasks;
  tasks.reserve(n_tasks);
  for (size_t i = 0; i < n_tasks; i++) {
    tasks.push_back(std::make_shared<Task>(i, disp.get_mr()));
  }

  std::queue<std::shared_ptr<Task>> queue;
  for (auto& task : tasks) {
    queue.push(task);
  }

  process(queue, disp);

  spdlog::info("Done");

  return 0;
}