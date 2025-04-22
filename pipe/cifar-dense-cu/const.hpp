#pragma once

#include <queue>

#include "builtin-apps/cifar-dense/cuda/dispatchers.cuh"
#include "builtin-apps/cifar-dense/omp/dispatchers.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"

// Application-specific constants
constexpr size_t kNumStages = 9;

using DispatcherT = cifar_dense::cuda::CudaDispatcher;
using AppDataT = cifar_dense::AppData;
using AppDataPtr = std::shared_ptr<AppDataT>;

// Pipeline-specific constants
constexpr size_t kPoolSize = 32;
constexpr size_t kNumToProcess = 100;

using QueueT = SPSCQueue<AppDataPtr, kPoolSize>;
using LocalQueue = std::queue<AppDataPtr>;

// Helper that creates a dataset of AppDataPtrs

[[nodiscard]] const std::vector<AppDataPtr> make_dataset(DispatcherT& disp,
                                                         const size_t num_items = kNumToProcess) {
  std::vector<AppDataPtr> result;
  result.reserve(num_items);

  for (size_t i = 0; i < num_items; ++i) {
    auto app = std::make_shared<cifar_dense::AppData>(&disp.get_mr());
    result.push_back(app);
  }

  return result;
}

[[nodiscard]] LocalQueue make_queue_from_vector(const std::vector<AppDataPtr>& vec) {
  return std::queue<AppDataPtr>(std::deque<AppDataPtr>(vec.begin(), vec.end()));
}

// ----------------------------------------------------------------------------
// Main Worker
// ----------------------------------------------------------------------------

template <typename QueueT>
void worker(QueueT& q_in,
            QueueT& q_out,
            std::function<void(AppDataPtr&)> func,
            const size_t num_items_to_process,
            const bool is_last = false) {
  for (size_t i = 0; i < num_items_to_process; ++i) {
    AppDataPtr app;
    while (!q_in.dequeue(app)) {
      std::this_thread::yield();
    }

    if (app == nullptr) {
      throw std::runtime_error("App is nullptr");
    }

    // ------------------------------------------------------------------------
    func(app);
    // ------------------------------------------------------------------------

    if (is_last) {
      app->reset();
    }

    while (!q_out.enqueue(app)) {
      std::this_thread::yield();
    }
  }
}
