#pragma once

#include <concurrentqueue.h>

#include <vector>

#include "builtin-apps/cifar-dense/dense_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure (new)
// ---------------------------------------------------------------------

struct Task {
  cifar_dense::AppData *data;
  explicit Task(cifar_dense::AppData *data) : data(data) {}
};

[[nodiscard]] moodycamel::ConcurrentQueue<Task *> init_tasks(
    std::vector<cifar_dense::AppData> &data);