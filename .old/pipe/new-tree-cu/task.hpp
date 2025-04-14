#pragma once

#include <concurrentqueue.h>

#include <vector>

#include "builtin-apps/tree/safe_tree_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure (new)
// ---------------------------------------------------------------------

struct Task {
  tree::SafeAppData *data;
  explicit Task(tree::SafeAppData *data) : data(data) {}
};

[[nodiscard]] moodycamel::ConcurrentQueue<Task *> init_tasks(std::vector<tree::SafeAppData> &data);