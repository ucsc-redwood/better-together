#pragma once
// ---------------------------------------------------------------------------
// runtime/pipeline_runner -- the backend-agnostic gtest harness that drives the REAL
// concurrent worker/SPSC ring (the same spawn-one-thread-per-chunk loop as
// bt_gen_log::run_schedule(), pipe/bm_gen_log_common.hpp), then runs a per-item
// correctness check after the ring drains. Was builtin-apps/pipeline/
// pipeline_test_runner.hpp; moved into runtime/ and re-threaded onto AppTraits.
//
// run_pipeline() is templated on the concrete types (the OMP test instantiates it with
// the OMP stub dispatcher; the vk/cu tests with the real GPU dispatcher + its UMA memory
// resource -- one harness, every backend). run_runtime_test<Dispatcher>() is the thin
// convenience that derives all of those from AppTraits<Dispatcher> (the compiler-checked
// contract) so a cell's test TU shrinks to: build a Schedule, define a per-item check,
// call run_runtime_test. The old pipeline_test_executor.hpp (which hardcoded
// tree::SafeAppData) is gone; OmpStubDispatcher below is its generic replacement.
//
//   gpu_em  = the ExecutionModel that is "the GPU" for this binary (kVulkan for a
//             vk test, kCuda for a cu test; an unused value for an OMP-only test). A
//             chunk with that exec_model dispatches on the GPU dispatcher; any other
//             chunk is kOMP and runs omp_dispatch pinned to its tier.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "platform/registry/device_registry.hpp"  // get_cores_by_type, ProcessorType
#include "runtime/app_traits.hpp"                 // AppTraits, BtRuntimeApp
#include "runtime/pipeline.hpp"                   // make_dataset, worker
#include "runtime/schedule.hpp"  // Schedule, ExecutionModel, get_processor_type_from_chunk_config

namespace bt_pipe_test {

// Generic OMP "dispatcher": the only thing make_dataset()/the ring asks of a Dispatcher on
// the OMP-only path is get_mr() (host memory -- every AppData is plain host memory on the
// OMP path). dispatch_multi_stage() is never reached (an OMP-only schedule has no gpu_em
// chunk) but must compile for the templated run_pipeline() GPU branch. Generic over AppData;
// replaces the tree-hardcoded stub the deleted pipeline_test_executor.hpp used to provide.
template <class AppData>
struct OmpStubDispatcher {
  static std::pmr::memory_resource* get_mr() { return std::pmr::new_delete_resource(); }
  void dispatch_multi_stage(AppData&, int, int) {
    throw std::logic_error("OmpStubDispatcher has no GPU dispatch path");
  }
};

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
  // The GPU dispatcher's command buffer/fence is shared across the per-chunk worker
  // threads, so a schedule with >1 GPU chunk would race it into VK_ERROR_DEVICE_LOST.
  // Reject up front (z3 never emits such a schedule) instead of crashing the device.
  if (const auto reason = first_concurrent_gpu_chunk(sched)) {
    FAIL() << *reason;
    return;
  }

  const auto n_chunks = sched.n_chunks();
  DispatcherTArg disp;
  const std::vector<std::unique_ptr<AppDataTArg>> dataset =
      make_dataset<AppDataTArg>(disp, pool_size);

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
    threads.emplace_back(worker<QueueTArg, AppDataTArg>,
                         std::ref(q_in),
                         std::ref(q_out),
                         std::move(fn),
                         n_items,
                         is_last);
  }

  // Watchdog: the workers busy-yield on dequeue/enqueue with no timeout, so a stalled
  // SPSC handoff would hang forever. Join on a helper thread and bound the wait. On
  // timeout the ring is deadlocked and the stuck workers still reference these stack
  // locals, so we can't safely detach/unwind -- abort with a diagnostic, turning an
  // infinite hang into a fast, informative failure. The bound is generous (slow
  // devices: cifar-sparse on Mali takes minutes) -- only a true deadlock trips it.
  constexpr auto kWatchdog = std::chrono::seconds(300);
  std::future<void> joined = std::async(std::launch::async, [&threads]() {
    for (auto& t : threads) t.join();
  });
  if (joined.wait_for(kWatchdog) != std::future_status::ready) {
    std::cerr << "\nFATAL: pipeline ring did not drain within " << kWatchdog.count()
              << "s -- deadlock (SPSC handoff stalled); last chunk completed " << completed.size()
              << "/" << n_items << " items. Aborting.\n";
    std::abort();
  }
  joined.get();

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

// Thin convenience: derive every type/constant from AppTraits<Dispatcher> (the
// compiler-checked contract) and drive the ring. A cell's runtime-test TU specializes
// AppTraits<its Dispatcher> + calls this; no magic-typedef preamble.
template <class Dispatcher>
  requires BtRuntimeApp<Dispatcher>
inline void run_runtime_test(
    const Schedule& sched,
    const std::function<void(typename AppTraits<Dispatcher>::AppData&)>& per_item_check) {
  using T = AppTraits<Dispatcher>;
  run_pipeline<typename T::AppData, Dispatcher, typename T::Queue>(
      sched, T::kPoolSize, T::kNumToProcess, T::kGpuExecModel, &T::omp_dispatch, per_item_check);
}

}  // namespace bt_pipe_test
