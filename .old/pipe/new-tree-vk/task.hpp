#pragma once

#include <concurrentqueue.h>

#include <vector>

#include "builtin-apps/tree/vulkan/vk_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure (new)
// ---------------------------------------------------------------------

struct Task {
  tree::vulkan::VkAppData_Safe *data;

  explicit Task(tree::vulkan::VkAppData_Safe *data) : data(data) {}
};

[[nodiscard]] moodycamel::ConcurrentQueue<Task *> init_tasks(
    std::vector<tree::vulkan::VkAppData_Safe> &data, size_t initial_capacity = 32);