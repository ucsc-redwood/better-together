#pragma once
// ---------------------------------------------------------------------------
// pipeline_test_runner -- the backend-agnostic gtest harness that drives the REAL
// concurrent worker/SPSC ring (the same spawn-one-thread-per-chunk loop as
// bt_gen_log::run_schedule(), pipe/bm_gen_log_common.hpp), then runs a per-item
// correctness check after the ring drains.
//
// The INCLUDER must have already defined the pipeline typedefs the shared
// worker()/make_dataset() reference by name and pulled them in -- i.e. it must
// have included a const.hpp-equivalent + pipe/pipeline_common.hpp BEFORE this
// header (exactly the contract a per-cell const.hpp fulfills). run_pipeline() is
// templated on the concrete types, so the OMP test instantiates it with the OMP
// stub dispatcher and the vk/cu tests instantiate it with the real GPU dispatcher
// + its UMA memory resource -- the one harness covers every backend.
//
//   gpu_em  = the ExecutionModel that is "the GPU" for this binary (kVulkan for a
//             vk test, kCuda for a cu test; pass an unused value for an OMP-only
//             test). A chunk with that exec_model dispatches on the GPU dispatcher;
//             any other chunk is kOMP and runs omp_dispatch pinned to its tier.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "builtin-apps/app.hpp"  // get_cores_by_type, ProcessorType
#include "schedule.hpp"          // Schedule, ExecutionModel, get_processor_type_from_chunk_config

namespace bt_pipe_test {

// Drive `n_items` through the real concurrent ring described by `sched`:
//   - build a pool of `pool_size` AppData (each carries its own const golden, built
//     at construction from the fixed seed -- the differential oracle),
//   - pre-fill queues[0] with the whole pool (assert every enqueue succeeded),
//   - spawn one std::thread per chunk running the shared worker(): a GPU chunk
//     (exec_model == gpu_em) dispatches on the real GPU dispatcher (its UMA buffers),
//     an OMP chunk runs omp_dispatch pinned to its tier's cores,
//   - join all, then run per_item_check on every pooled AppData.
//
// A hybrid schedule (an OMP chunk + a GPU chunk) makes the OMP thread and the GPU
// thread process different items CONCURRENTLY out of the shared UMA pool -- the
// concurrent CPU+GPU + unified-memory-visibility path (§1/§7) the sequential
// oracle never exercises.
template <class AppDataTArg, class DispatcherTArg, class QueueTArg, class OmpDispatch>
inline void run_pipeline(const Schedule& sched,
                         const size_t pool_size,
                         const size_t n_items,
                         const ExecutionModel gpu_em,
                         OmpDispatch omp_dispatch,
                         const std::function<void(AppDataTArg&)>& per_item_check) {
  const auto n_chunks = sched.n_chunks();
  DispatcherTArg disp;
  const std::vector<std::unique_ptr<AppDataTArg>> dataset = make_dataset(disp, pool_size);

  std::vector<QueueTArg> queues(n_chunks);
  for (size_t i = 0; i < pool_size; ++i) {
    ASSERT_TRUE(queues[0].enqueue(dataset[i].get()))
        << "queue[0] full at item " << i << " -- pool_size exceeds QueueT capacity";
  }

  // Completion-edge sink: the LAST chunk records every item it finishes. Asserting
  // the count == n_items (and that all pool objects appear) catches a drop/dup that
  // the per-item golden check below cannot -- a later-cycle drop leaves the item's
  // _out holding an EARLIER cycle's still-correct result, so the golden compare
  // passes while throughput is silently wrong.
  std::vector<AppDataTArg*> completed;
  std::mutex completed_mu;

  std::vector<std::thread> threads;
  for (size_t chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
    QueueTArg& q_in = queues[chunk_id];
    QueueTArg& q_out = queues[(chunk_id + 1) % n_chunks];
    const int start = sched.start_stage(chunk_id);
    const int end = sched.end_stage(chunk_id);
    const bool is_last = chunk_id == n_chunks - 1;
    const ExecutionModel em = sched.chunks[chunk_id].exec_model;

    std::function<void(AppDataTArg*)> fn;
    if (em == gpu_em) {
      fn = [&disp, start, end](AppDataTArg* app) { disp.dispatch_multi_stage(*app, start, end); };
    } else {  // kOMP chunk pinned to its CPU tier
      const ProcessorType cpu_pt = get_processor_type_from_chunk_config(sched.chunks[chunk_id]);
      fn = [&omp_dispatch, cpu_pt, start, end](AppDataTArg* app) {
        auto& cores = get_cores_by_type(cpu_pt);
        omp_dispatch(cores, cores.size(), *app, start, end);
      };
    }
    if (is_last) {
      fn = [inner = std::move(fn), &completed, &completed_mu](AppDataTArg* app) {
        inner(app);
        std::lock_guard<std::mutex> lk(completed_mu);
        completed.push_back(app);
      };
    }
    threads.emplace_back(worker, std::ref(q_in), std::ref(q_out), std::move(fn), n_items, is_last);
  }
  for (auto& t : threads) t.join();

  // Completion-edge invariants: the last chunk must have finished exactly n_items and
  // every one of the pool_size distinct objects must have reached it (no orphan/starve).
  EXPECT_EQ(completed.size(), n_items)
      << "last chunk finished " << completed.size() << " of " << n_items
      << " items -- the ring dropped or duplicated items";
  const std::set<AppDataTArg*> distinct(completed.begin(), completed.end());
  EXPECT_EQ(distinct.size(), pool_size)
      << "only " << distinct.size() << " of " << pool_size
      << " pool objects reached the last chunk -- an orphaned/starved item";

  // After the ring drains, every pooled item's _out holds the result the pipeline
  // last wrote for it. Check each against its own inherited (distinct-seed) golden.
  for (const auto& item : dataset) {
    per_item_check(*item);
    if (::testing::Test::HasFatalFailure()) return;
  }
}

}  // namespace bt_pipe_test
