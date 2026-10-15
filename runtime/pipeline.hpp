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

#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
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

// ----------------------------------------------------------------------------
// GPU-chunk worker with up to `depth` items in flight (design D3; submission spec "Several
// items may be in flight per GPU chunk"). Dequeues and submits while fewer than `depth`
// are outstanding (blocking only when none is), then completes the OLDEST and enqueues it
// downstream, so items leave in the order they entered. depth 1 is worker() exactly:
// dequeue, submit, wait, enqueue. on_submit/on_complete(item index) bracket an item's
// time in the chunk (worker_with_record's ticks). Same exception and null-item handling as
// worker(): log, and always hand the item on.
// ----------------------------------------------------------------------------
template <class Queue, class AppData>
inline void inflight_worker(Queue& q_in,
                            Queue& q_out,
                            const std::function<uint64_t(AppData*)>& submit,
                            const std::function<void(AppData*, uint64_t)>& complete,
                            const size_t num_items_to_process,
                            const bool is_last,
                            const size_t depth,
                            const std::function<void(size_t)>& on_submit = {},
                            const std::function<void(size_t)>& on_complete = {}) {
  struct InFlight {
    size_t index;
    AppData* app;
    bool submitted;
    uint64_t ticket;
  };
  std::deque<InFlight> in_flight;
  size_t dequeued = 0;
  for (size_t done = 0; done < num_items_to_process; ++done) {
    while (dequeued < num_items_to_process && in_flight.size() < depth) {
      AppData* app = nullptr;
      if (in_flight.empty()) {
        while (!q_in.dequeue(app)) std::this_thread::yield();
      } else if (!q_in.dequeue(app)) {
        break;  // nothing ready: complete the oldest instead of waiting here
      }
      InFlight f{.index = dequeued++, .app = app, .submitted = false, .ticket = 0};
      if (app == nullptr) {
        spdlog::error("inflight_worker: item {} dequeued a null app (invariant violation)",
                      f.index);
      } else {
        try {
          if (on_submit) on_submit(f.index);
          f.ticket = submit(app);
          f.submitted = true;
        } catch (const std::exception& e) {
          spdlog::error("inflight_worker: item {} submit threw: {}", f.index, e.what());
        } catch (...) {
          spdlog::error("inflight_worker: item {} submit threw unknown exception", f.index);
        }
      }
      in_flight.push_back(f);
    }

    const InFlight f = in_flight.front();
    in_flight.pop_front();
    if (f.submitted) {
      try {
        complete(f.app, f.ticket);
        if (on_complete) on_complete(f.index);
      } catch (const std::exception& e) {
        spdlog::error("inflight_worker: item {} threw: {}", f.index, e.what());
      } catch (...) {
        spdlog::error("inflight_worker: item {} threw unknown exception", f.index);
      }
    }
    if (is_last && f.app != nullptr) f.app->reset();
    while (!q_out.enqueue(f.app)) std::this_thread::yield();
  }
}

// BT_VK_INFLIGHT=<n>: the in-flight depth of every Vulkan GPU chunk (default 1, today's
// behaviour). Throws on anything but a positive integer.
[[nodiscard]] inline size_t vk_inflight_depth() {
  const char* v = std::getenv("BT_VK_INFLIGHT");
  if (v == nullptr) return 1;
  char* end = nullptr;
  const unsigned long depth = std::strtoul(v, &end, 10);
  if (end == v || *end != '\0' || depth == 0) {
    throw std::invalid_argument(std::string("BT_VK_INFLIGHT must be a positive integer, got ") + v);
  }
  return depth;
}

// The pool that lets every GPU chunk hold `depth` items and every CPU chunk one at once
// (design D3); a smaller pool does not deadlock, it just caps the depth reached.
[[nodiscard]] inline size_t min_pool_size(const size_t gpu_chunks,
                                          const size_t cpu_chunks,
                                          const size_t depth) {
  return depth * gpu_chunks + cpu_chunks;
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
