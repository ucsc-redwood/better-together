#pragma once
// ---------------------------------------------------------------------------
// runtime/pipeline -- the worker / queue / dataset plumbing for the concurrent
// SPSC ring. Was pipe/pipeline_common.hpp (a profiler dir); moved here so the
// runtime no longer includes UP into the profiler (breaks the runtime->profiler
// cycle), and TEMPLATIZED so it no longer relies on the includer pre-defining
// DispatcherT/AppDataT/QueueT/LocalQueue/kNumToProcess by name (the old fragile
// magic-typedef ODR contract). Callers now pass the concrete types explicitly.
// ---------------------------------------------------------------------------

#include <spdlog/spdlog.h>

#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "platform/mem/mr_ptr.hpp"  // bt_pipe::as_mr_ptr
#include "runtime/record.hpp"       // Logger (worker_with_record)

// Build a pool of fresh AppData, each backed by the dispatcher's memory resource.
template <class AppData, class Dispatcher>
[[nodiscard]] inline std::vector<std::unique_ptr<AppData>> make_dataset(Dispatcher& disp,
                                                                        const size_t num_items) {
  std::vector<std::unique_ptr<AppData>> result;
  result.reserve(num_items);
  for (size_t i = 0; i < num_items; ++i) {
    result.push_back(std::make_unique<AppData>(bt_pipe::as_mr_ptr(disp.get_mr())));
  }
  return result;
}

template <class LocalQueue, class AppData>
[[nodiscard]] inline LocalQueue make_queue_from_vector(
    const std::vector<std::unique_ptr<AppData>>& vec) {
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
// catch it). Catch it here, log it (the operator's signal -- std::thread can't carry
// a default-arg sink, so we don't try to thread one through), and ALWAYS re-enqueue
// the item so the next worker in the SPSC ring keeps draining -- catching-and-breaking
// would instead hang every downstream worker on its busy-yield dequeue.
template <class Queue, class AppData>
inline void worker(Queue& q_in,
                   Queue& q_out,
                   std::function<void(AppData*)> func,
                   const size_t num_items_to_process,
                   const bool is_last = false) {
  for (size_t i = 0; i < num_items_to_process; ++i) {
    AppData* app = nullptr;
    while (!q_in.dequeue(app)) {
      std::this_thread::yield();
    }

    // A null here would be an invariant violation (the rings are only ever fed non-null
    // items). This used to throw -- but a throw escaping a std::thread body calls
    // std::terminate, the very crash the try/catch below exists to prevent (review D2).
    // Log it and fall through to re-enqueue so the ring keeps draining instead of crashing.
    // ------------------------------------------------------------------------
    if (app != nullptr) {
      try {
        func(app);
      } catch (const std::exception& e) {
        spdlog::error("worker: item {} threw: {}", i, e.what());
      } catch (...) {
        spdlog::error("worker: item {} threw unknown exception", i);
      }
      if (is_last) {
        app->reset();
      }
    } else {
      spdlog::error("worker: item {} dequeued a null app (invariant violation)", i);
    }
    // ------------------------------------------------------------------------

    while (!q_out.enqueue(app)) {
      std::this_thread::yield();
    }
  }
}

// See worker() above: a throw here (e.g. Logger OOB on a malformed >kMaxChunks
// schedule) would terminate the process from a worker thread. Catch it, log it with
// the chunk context, and re-enqueue so the SPSC ring doesn't deadlock.
template <class Queue, class AppData, size_t N>
inline void worker_with_record(const int chunk_id,
                               Logger<N>& logger,
                               Queue& q_in,
                               Queue& q_out,
                               std::function<void(AppData*)> func,
                               const size_t num_items_to_process,
                               const bool is_last = false) {
  for (size_t processing_id = 0; processing_id < num_items_to_process; ++processing_id) {
    AppData* app = nullptr;
    while (!q_in.dequeue(app)) {
      std::this_thread::yield();
    }

    // Log (don't throw -> don't std::terminate the worker thread; review D2) a null app
    // and fall through to re-enqueue so the ring keeps draining.
    // ------------------------------------------------------------------------
    if (app != nullptr) {
      try {
        logger.start_tick(processing_id, chunk_id);
        func(app);
        logger.end_tick(processing_id, chunk_id);
      } catch (const std::exception& e) {
        spdlog::error(
            "worker_with_record: chunk {} item {} threw: {}", chunk_id, processing_id, e.what());
      } catch (...) {
        spdlog::error("worker_with_record: chunk {} item {} threw unknown exception",
                      chunk_id,
                      processing_id);
      }
      if (is_last) {
        app->reset();
      }
    } else {
      spdlog::error(
          "worker_with_record: chunk {} item {} dequeued a null app", chunk_id, processing_id);
    }
    // ------------------------------------------------------------------------

    while (!q_out.enqueue(app)) {
      std::this_thread::yield();
    }
  }
}
