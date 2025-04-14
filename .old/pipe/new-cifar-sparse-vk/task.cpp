// #include "task.hpp"

// #include <spdlog/spdlog.h>

// #include "../spsc_queue.hpp"

// [[nodiscard]] SPSCQueue<Task *, 1024> init_tasks(std::vector<cifar_sparse::AppData> &data) {
//   SPSCQueue<Task *, 1024> tasks;
//   for (auto &app_data : data) {
//     Task *task = new Task(&app_data);
//     tasks.enqueue(std::move(task));
//   }
//   tasks.enqueue(nullptr);
//   return std::move(tasks);
// }

// [[nodiscard]] moodycamel::ConcurrentQueue<Task *> init_tasks(
//     std::vector<cifar_sparse::AppData> &data, const size_t initial_capacity) {
//   // Initialize queue with reasonable capacity to avoid resizing
//   moodycamel::ConcurrentQueue<Task *> tasks(initial_capacity);

//   // Reserve space for all tasks plus sentinel
//   for (auto &app_data : data) {
//     Task *task = new Task(&app_data);
//     if (!tasks.enqueue(task)) {
//       delete task;
//       throw std::runtime_error("Failed to enqueue task");
//     }
//   }

//   // Add sentinel task
//   tasks.enqueue(nullptr);

//   return tasks;
// }
