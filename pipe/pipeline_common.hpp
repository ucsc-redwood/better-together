#pragma once
// ---------------------------------------------------------------------------
// pipeline_common -- the worker/queue/dataset plumbing shared by every per-cell
// const.hpp. It was copy-pasted, ~95% identical, into all 6 pipe/<app>-<be>/
// const.hpp; a single race fix had to be applied six times. This lifts the
// identical parts here (the bm_prof_common.hpp pattern for the *_prof drivers).
//
// The INCLUDER (a per-cell const.hpp) must have already defined the cell types
// and constants this file references by name:
//   DispatcherT, AppDataT, AppDataPtr, QueueT, LocalQueue, kNumToProcess
// and pulled in Logger (builtin-apps/pipeline/record.hpp). Include this header
// at the END of const.hpp, after those declarations.
// ---------------------------------------------------------------------------

#include <spdlog/spdlog.h>

#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "mr_ptr.hpp"  // bt_pipe::as_mr_ptr (CUDA-ref vs Vulkan-pointer get_mr())

// Build a pool of fresh AppData, each backed by the dispatcher's memory resource.
[[nodiscard]] static inline const std::vector<AppDataPtr> make_dataset(
    DispatcherT& disp, const size_t num_items = kNumToProcess) {
  std::vector<AppDataPtr> result;
  result.reserve(num_items);
  for (size_t i = 0; i < num_items; ++i) {
    result.push_back(std::make_unique<AppDataT>(bt_pipe::as_mr_ptr(disp.get_mr())));
  }
  return result;
}

[[nodiscard]] static inline LocalQueue make_queue_from_vector(const std::vector<AppDataPtr>& vec) {
  LocalQueue q;
  for (const auto& item : vec) {
    q.push(item.get());
  }
  return q;
}

// ----------------------------------------------------------------------------
// Main Worker: pull an AppData from q_in, run func, push to q_out (reset on the
// last stage). Busy-yield SPSC queues.
// ----------------------------------------------------------------------------
// A throw escaping a std::thread body calls std::terminate (the main thread can't
// catch it). Catch it here, log it (the operator's signal — std::thread can't carry
// a default-arg sink, so we don't try to thread one through), and ALWAYS re-enqueue
// the item so the next worker in the SPSC ring keeps draining — catching-and-breaking
// would instead hang every downstream worker on its busy-yield dequeue.
static inline void worker(QueueT& q_in,
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
    try {
      func(app);
    } catch (const std::exception& e) {
      spdlog::error("worker: item {} threw: {}", i, e.what());
    } catch (...) {
      spdlog::error("worker: item {} threw unknown exception", i);
    }
    // ------------------------------------------------------------------------

    if (is_last) {
      app->reset();
    }

    while (!q_out.enqueue(app)) {
      std::this_thread::yield();
    }
  }
}

// See worker() above: a throw here (e.g. Logger OOB on a malformed >kMaxChunks
// schedule) would terminate the process from a worker thread. Catch it, log it with
// the chunk context, and re-enqueue so the SPSC ring doesn't deadlock.
static inline void worker_with_record(const int chunk_id,
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
    try {
      logger.start_tick(processing_id, chunk_id);
      func(app);
      logger.end_tick(processing_id, chunk_id);
    } catch (const std::exception& e) {
      spdlog::error("worker_with_record: chunk {} item {} threw: {}", chunk_id, processing_id,
                    e.what());
    } catch (...) {
      spdlog::error("worker_with_record: chunk {} item {} threw unknown exception", chunk_id,
                    processing_id);
    }
    // ------------------------------------------------------------------------

    if (is_last) {
      app->reset();
    }

    while (!q_out.enqueue(app)) {
      std::this_thread::yield();
    }
  }
}
