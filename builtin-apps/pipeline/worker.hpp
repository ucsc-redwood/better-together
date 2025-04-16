#pragma once

#include <functional>
#include <thread>

#include "record.hpp"
#include "spsc_queue.hpp"

constexpr size_t kPoolSize = 32;
constexpr size_t kNumToProcess = 100;

// ----------------------------------------------------------------------------
// Normal worker thread
// ----------------------------------------------------------------------------

template <typename TaskT>
void worker_thread(SPSCQueue<TaskT*, kPoolSize>& in_queue,
                   SPSCQueue<TaskT*, kPoolSize>& out_queue,
                   std::function<void(TaskT&)> process_function) {
  for (size_t _ = 0; _ < kNumToProcess; ++_) {
    TaskT* task = nullptr;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

    // Process the task (timing is handled in the provided process function)
    process_function(*task);

    // Forward processed task
    while (!out_queue.enqueue(std::move(task))) {
      std::this_thread::yield();
    }
  }
}

// ----------------------------------------------------------------------------
// Worker thread with timing
// ----------------------------------------------------------------------------

template <typename TaskT>
void worker_thread_record(const int chunk_id,
                          Logger<kNumToProcess>& logger,
                          SPSCQueue<TaskT*, kPoolSize>& in_queue,
                          SPSCQueue<TaskT*, kPoolSize>& out_queue,
                          std::function<void(TaskT&)> process_function) {
  for (size_t processing_id = 0; processing_id < kNumToProcess; ++processing_id) {
    TaskT* task = nullptr;
    while (!in_queue.dequeue(task)) {
      std::this_thread::yield();
    }

    logger.start_tick(processing_id, chunk_id);

    process_function(*task);

    logger.end_tick(processing_id, chunk_id);

    while (!out_queue.enqueue(std::move(task))) {
      std::this_thread::yield();
    }
  }
}
