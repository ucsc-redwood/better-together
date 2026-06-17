#pragma once
// ---------------------------------------------------------------------------
// pipeline_test_executor -- the OMP instantiation of the runtime test harness.
// It fixes the concrete pipeline typedefs the shared worker()/make_dataset()
// reference by name (mirrors a per-cell const.hpp), pulls in pipeline_common.hpp
// (worker + make_dataset reused verbatim), then the backend-agnostic run_pipeline()
// from pipeline_test_runner.hpp. The vk/cu runtime tests do the same with their own
// (real GPU) types + a hybrid OMP|GPU schedule; this OMP-only path is the cheapest,
// locally-verifiable entry point.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <omp.h>

#include <cstddef>
#include <memory>
#include <memory_resource>
#include <queue>
#include <stdexcept>

#include "builtin-apps/app.hpp"
#include "builtin-apps/tree/omp/dispatchers.hpp"
#include "builtin-apps/tree/safe_tree_appdata.hpp"
#include "record.hpp"
#include "schedule.hpp"
#include "spsc_queue.hpp"

namespace bt_pipe_test {

// A trivial OMP "dispatcher": the only thing make_dataset()/pipeline_common.hpp
// asks of a DispatcherT is get_mr() (mirrors OmpRunner::Mr() in
// tree/omp/test_main.cpp -- every AppData is host memory on the OMP path).
struct OmpDispatcher {
  static std::pmr::memory_resource* get_mr() { return std::pmr::new_delete_resource(); }
  // Never reached on the OMP path (an OMP-only schedule has no gpu_em chunk); present
  // only so the templated run_pipeline() GPU branch compiles for this dispatcher type.
  void dispatch_multi_stage(tree::SafeAppData&, int, int) {
    throw std::logic_error("OmpDispatcher has no GPU dispatch path");
  }
};

}  // namespace bt_pipe_test

// ---------------------------------------------------------------------------
// Typedefs the shared worker()/make_dataset() reference by name. Mirrors a
// per-cell const.hpp. kPoolSize items in flight; the queue is sized to the next
// power of two >= kPoolSize (SPSC usable capacity is Size-1, one slot reserved),
// so enqueueing the whole pool into queues[0] never silently drops the last item.
// ---------------------------------------------------------------------------
using DispatcherT = bt_pipe_test::OmpDispatcher;
using AppDataT = tree::SafeAppData;
using AppDataPtr = std::unique_ptr<AppDataT>;
using QueueT = SPSCQueue<AppDataT*, 64>;  // pow2 >= kPoolSize(32) with a free slot
using LocalQueue = std::queue<AppDataT*>;

constexpr size_t kNumStages = 7;
constexpr size_t kPoolSize = 32;
constexpr size_t kNumToProcess = 100;

#include "../../pipe/pipeline_common.hpp"  // make_dataset + worker (reused verbatim)
#include "pipeline_test_runner.hpp"        // run_pipeline (backend-agnostic, after the above)
