// #pragma once

// #include <concurrentqueue.h>

// #include <functional>
// #include <vector>

// #include "builtin-apps/common/kiss-vk/vma_pmr.hpp"

// // ---------------------------------------------------------------------
// // Chunk building blocks
// // ---------------------------------------------------------------------

// template <typename TaskType, typename AppDataType>
// void chunk(moodycamel::ConcurrentQueue<TaskType *> &q_cur,
//            moodycamel::ConcurrentQueue<TaskType *> *q_next,
//            std::function<void(AppDataType &)> func) {
//   while (true) {
//     TaskType *task = nullptr;
//     if (q_cur.try_dequeue(task)) {
//       if (task == nullptr) {
//         // Sentinel => pass it on if there's a next queue and stop
//         if (q_next != nullptr) {
//           q_next->enqueue(nullptr);
//         }
//         break;
//       }

//       // -----------------------------------
//       func(*task->data);
//       // -----------------------------------

//       // If there's a next queue, pass the task along
//       if (q_next != nullptr) {
//         q_next->enqueue(task);
//       }
//     } else {
//       std::this_thread::yield();
//     }
//   }
// }

// // ---------------------------------------------------------------------
// // AppData initialization
// // ---------------------------------------------------------------------

// template <typename AppDataType>
// [[nodiscard]] std::vector<AppDataType> init_vk_appdata(
//     kiss_vk::VulkanMemoryResource::memory_resource *vk_mr, const size_t num_tasks) {
//   std::vector<AppDataType> all_data;
//   all_data.reserve(num_tasks);
//   for (size_t i = 0; i < num_tasks; ++i) {
//     all_data.emplace_back(vk_mr);  // Each has big vectors
//   }
//   return all_data;
// }
