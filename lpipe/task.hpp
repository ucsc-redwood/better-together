#pragma once

#include <numeric>
#include <vector>
#include <cstdint>

struct Task {
  uint32_t uid;
  std::vector<float> data;
};

[[nodiscard]] inline Task new_task(const size_t size) {
  Task task;
  static uint32_t uid_counter = 0;
  task.uid = uid_counter++;
  task.data.resize(size);
  std::iota(task.data.begin(), task.data.end(), 0.0f);
  return task;
}

// [[nodiscard]] inline Task new_sentinel() {
//   Task task;
//   task.is_sentinel = true;
//   return task;
// }

static_assert(std::is_move_constructible_v<Task>, "Task must be moveable");
