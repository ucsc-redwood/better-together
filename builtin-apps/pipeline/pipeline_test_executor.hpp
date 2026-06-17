#pragma once
// ---------------------------------------------------------------------------
// pipeline_test_executor -- gtest-callable harness that drives the REAL
// concurrent worker/SPSC ring, lifted out of bt_gen_log::run_schedule()/warmup()
// (pipe/bm_gen_log_common.hpp). The per-cell bm_gen_log binaries embed that ring
// inside a main() that only emits a timing log; no test ever spun up the ring to
// CHECK the result. run_pipeline() is that same spawn-one-thread-per-chunk loop,
// callable from a TEST() with a per-item correctness check after the ring drains.
//
// POC scope: OMP-only (every chunk must be kOMP -- asserted). The GPU branch
// (gpu_em) is added when Category 3 needs it; until then keeping it out avoids
// pulling a GPU dispatcher into the OMP test binary.
//
// This header establishes the concrete pipeline typedefs the shared worker()/
// make_dataset() reference by name (DispatcherT, AppDataT, AppDataPtr, QueueT,
// LocalQueue, kNumToProcess), then includes pipeline_common.hpp -- exactly the
// contract a per-cell const.hpp fulfills, so worker()/make_dataset() are reused
// verbatim rather than re-implemented.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <omp.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <memory_resource>
#include <queue>
#include <thread>
#include <vector>

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

namespace bt_pipe_test {

// Drive `n_items` through the real concurrent ring described by `sched`:
//   - build a pool of `pool_size` AppData (each carries its own const golden,
//     built at construction from the fixed seed -- the differential oracle),
//   - pre-fill queues[0] with the whole pool (assert every enqueue succeeded),
//   - spawn one std::thread per chunk running the shared worker(), each OMP chunk
//     pinned to its tier's cores via omp_dispatch,
//   - join all, then run per_item_check on every pooled AppData (each item's _out
//     was written by the ring; reset() only bumps uid, so a stale _out from a
//     prior cycle that the ring failed to overwrite surfaces as a golden mismatch).
//
// Body mirrors bm_gen_log_common.hpp run_schedule() lines 36-69. Templated on the
// concrete types so later categories reuse it by swapping AppDataT/QueueT; the POC
// instantiates it with the tree OMP types defined above. OMP-only: every chunk
// must be kOMP (asserted -- no gpu_em branch).
template <class AppDataTArg, class DispatcherTArg, class QueueTArg, class OmpDispatch>
inline void run_pipeline(const Schedule& sched,
                         const size_t pool_size,
                         const size_t n_items,
                         OmpDispatch omp_dispatch,
                         const std::function<void(AppDataTArg&)>& per_item_check) {
  for (const auto& c : sched.chunks) {
    ASSERT_EQ(c.exec_model, ExecutionModel::kOMP) << "POC harness is OMP-only";
  }

  const auto n_chunks = sched.n_chunks();
  DispatcherTArg disp;
  const std::vector<std::unique_ptr<AppDataTArg>> dataset = make_dataset(disp, pool_size);

  std::vector<QueueTArg> queues(n_chunks);
  for (size_t i = 0; i < pool_size; ++i) {
    ASSERT_TRUE(queues[0].enqueue(dataset[i].get()))
        << "queue[0] full at item " << i << " -- pool_size exceeds QueueT capacity";
  }

  std::vector<std::thread> threads;
  for (size_t chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
    QueueTArg& q_in = queues[chunk_id];
    QueueTArg& q_out = queues[(chunk_id + 1) % n_chunks];
    const int start = sched.start_stage(chunk_id);
    const int end = sched.end_stage(chunk_id);
    const bool is_last = chunk_id == n_chunks - 1;

    const ProcessorType cpu_pt = get_processor_type_from_chunk_config(sched.chunks[chunk_id]);
    threads.emplace_back(
        worker, std::ref(q_in), std::ref(q_out),
        [&omp_dispatch, cpu_pt, start, end](AppDataTArg* app) {
          auto& cores = get_cores_by_type(cpu_pt);
          omp_dispatch(cores, cores.size(), *app, start, end);
        },
        n_items, is_last);
  }
  for (auto& t : threads) t.join();

  // After the ring drains, every pooled item's _out holds the result the pipeline
  // last wrote for it. Check each against its own inherited golden.
  for (const auto& item : dataset) {
    per_item_check(*item);
    if (::testing::Test::HasFatalFailure()) return;
  }
}

}  // namespace bt_pipe_test
