#pragma once

#include <queue>

#include "builtin-apps/cifar-sparse/cuda/dispatchers.cuh"
#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/pipeline/record.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"

// Application-specific constants
constexpr size_t kNumStages = 9;

using DispatcherT = cifar_sparse::cuda::CudaDispatcher;
using AppDataT = cifar_sparse::AppData;
using AppDataPtr = std::unique_ptr<AppDataT>;

// Pipeline-specific constants
constexpr size_t kPoolSize = 32;
constexpr size_t kNumToProcess = 100;

using QueueT = SPSCQueue<AppDataT*, kPoolSize>;
using LocalQueue = std::queue<AppDataT*>;

// Helper that creates a dataset of AppDataPtrs

[[nodiscard]] const std::vector<AppDataPtr> make_dataset(DispatcherT& disp,
                                                         const size_t num_items = kNumToProcess) {
  std::vector<AppDataPtr> result;
  result.reserve(num_items);

  for (size_t i = 0; i < num_items; ++i) {
    auto app = std::make_unique<cifar_sparse::AppData>(&disp.get_mr());
    result.push_back(std::move(app));
  }

  return result;
}

[[nodiscard]] LocalQueue make_queue_from_vector(const std::vector<AppDataPtr>& vec) {
  LocalQueue q;
  for (const auto& item : vec) {
    q.push(item.get());
  }
  return q;
}

// ----------------------------------------------------------------------------
// Main Worker
// ----------------------------------------------------------------------------

void worker(QueueT& q_in,
            QueueT& q_out,
            std::function<void(AppDataT*)> func,
            const size_t num_items_to_process,
            const bool is_last = false) {
  for (size_t i = 0; i < num_items_to_process; ++i) {
    AppDataT* app = nullptr;
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

void worker_with_record(const int chunk_id,
                        Logger<kNumToProcess>& logger,
                        QueueT& q_in,
                        QueueT& q_out,
                        std::function<void(AppDataT*)> func,
                        const size_t num_items_to_process,
                        const bool is_last = false) {
  for (size_t processing_id = 0; processing_id < num_items_to_process; ++processing_id) {
    AppDataT* app = nullptr;
    while (!q_in.dequeue(app)) {
      std::this_thread::yield();
    }

    if (app == nullptr) {
      throw std::runtime_error("App is nullptr");
    }

    // ------------------------------------------------------------------------
    logger.start_tick(processing_id, chunk_id);
    func(app);
    logger.end_tick(processing_id, chunk_id);
    // ------------------------------------------------------------------------

    if (is_last) {
      app->reset();
    }

    while (!q_out.enqueue(app)) {
      std::this_thread::yield();
    }
  }
}
