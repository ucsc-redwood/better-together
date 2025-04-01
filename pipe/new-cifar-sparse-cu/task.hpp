#pragma once

#include <concurrentqueue.h>

#include <vector>

#include "builtin-apps/cifar-sparse/sparse_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure (new)
// ---------------------------------------------------------------------

struct Task {
  cifar_sparse::AppData *data;
  explicit Task(cifar_sparse::AppData *data) : data(data) {}
};

[[nodiscard]] moodycamel::ConcurrentQueue<Task *> init_tasks(
    std::vector<cifar_sparse::AppData> &data);