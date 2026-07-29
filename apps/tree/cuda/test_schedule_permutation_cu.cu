#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory_resource>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "apps/tree/omp/dispatchers.hpp"
#include "dispatchers.cuh"  // tree::cuda::CudaDispatcher
#include "platform/registry/device_registry.hpp"
#include "runtime/pipeline.hpp"  // make_dataset, worker_with_record
#include "runtime/record.hpp"    // Logger, now_cycles, get_counter_frequency
#include "runtime/schedule.hpp"
#include "runtime/spsc_queue.hpp"

// ----------------------------------------------------------------------------
// EXPERIMENTAL: exhaustive CPU/GPU schedule permutation + overlap coverage for
// the tree pipeline, on the genuinely-chained tree::AppData path (prior
// session). NOT part of the differential/oracle suite (test-tree-cu), the
// concurrency-mechanics suite (test-pipeline-e2e-cu), or the prior session's
// single-schedule proof-of-correctness (test-pipeline-chained-cu) -- this
// sweeps ALL 29 valid contiguous CPU/GPU stage splits (every GPU-range
// placement across the 7 stages, plus the all-CPU case), checking both:
//   (a) correctness -- every item's final octree matches a sequential OMP
//       reference, for every schedule, since stages now read/write shared
//       buffers directly rather than isolated golden/_out copies; and
//   (b) genuine overlap -- for every schedule with both a CPU and a GPU
//       chunk, the two PUs' measured work windows (from runtime/record.hpp's
//       Logger, already captured by worker_with_record for the production
//       Gantt-log tool) actually intersect in wall-clock time across the
//       majority of 5 repeated runs, not just once (real hardware timing is
//       noisy) and not fully serialized.
//
// The overlap math below ports dashboard/generate.py's `_coverage_time`
// sweep-line (already validated -- it's what produces the dashboard's
// existing measured-Gantt concurrency numbers) directly into C++, reading
// Logger::records_ (public) in-process -- no text-log round-trip needed.
//
// Deliberately excluded from ctest -L cuda (LABELS overridden to
// "experimental" in apps/tree/CMakeLists.txt) -- a one-off verification, not
// a maintained gate (spec FR-008).
// ----------------------------------------------------------------------------

namespace {

using QueueT = SPSCQueue<tree::AppData*, 64>;  // pow2 >= kPoolSize(32) with a free slot
using LoggerT = Logger<100>;

constexpr size_t kPoolSize = 32;
constexpr size_t kNumToProcess = 100;
constexpr int kNumStages = 7;
constexpr size_t kOverlapWarmupItems = 5;  // skip cold-start tasks, matches dashboard's PIPE_WARMUP
constexpr int kOverlapRepeats = 5;
constexpr int kOverlapRequiredHits =
    3;  // majority of 5 -- the clarification session's evidence bar

bool CudaAvailable() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// ----------------------------------------------------------------------------
// T002: every valid CPU/GPU schedule for the 7-stage tree pipeline -- the
// all-CPU schedule, plus every contiguous GPU stage range [gpu_start, gpu_end]
// with 1 <= gpu_start <= gpu_end <= 7 (28 combinations) = 29 schedules total.
// ----------------------------------------------------------------------------
std::vector<Schedule> generate_all_schedules() {
  std::vector<Schedule> out;

  {
    Schedule s;
    s.uid = "all-cpu";
    s.chunks = {{ExecutionModel::kOMP, 1, kNumStages, first_present_cpu_type()}};
    out.push_back(std::move(s));
  }

  for (int gpu_start = 1; gpu_start <= kNumStages; ++gpu_start) {
    for (int gpu_end = gpu_start; gpu_end <= kNumStages; ++gpu_end) {
      Schedule s;
      s.uid = "gpu-" + std::to_string(gpu_start) + "-" + std::to_string(gpu_end);
      if (gpu_start > 1) {
        s.chunks.push_back({ExecutionModel::kOMP, 1, gpu_start - 1, first_present_cpu_type()});
      }
      s.chunks.push_back({ExecutionModel::kCuda, gpu_start, gpu_end, std::nullopt});
      if (gpu_end < kNumStages) {
        s.chunks.push_back(
            {ExecutionModel::kOMP, gpu_end + 1, kNumStages, first_present_cpu_type()});
      }
      out.push_back(std::move(s));
    }
  }

  for (const auto& s : out) {
    validate_schedule_coverage(s, kNumStages);
    if (const auto reason = first_concurrent_gpu_chunk(s)) {
      ADD_FAILURE() << "generated invalid schedule [" << s.uid << "]: " << *reason;
    }
  }
  return out;
}

// ----------------------------------------------------------------------------
// T003: run one schedule once through the real concurrent runtime (pool of
// genuinely-chained tree::AppData, one worker_with_record thread per chunk),
// returning the pooled dataset (for correctness) and the populated Logger
// (for overlap measurement).
// ----------------------------------------------------------------------------
struct RunResult {
  std::vector<std::unique_ptr<tree::AppData>> dataset;
  LoggerT logger;
};

RunResult run_schedule_once(const Schedule& sched) {
  tree::cuda::CudaDispatcher disp;
  std::vector<std::unique_ptr<tree::AppData>> dataset =
      make_dataset<tree::AppData>(disp, kPoolSize);

  const auto n_chunks = sched.n_chunks();
  std::vector<QueueT> queues(n_chunks);
  for (size_t i = 0; i < kPoolSize; ++i) {
    // EXPECT (not ASSERT): this function returns RunResult, not void, and
    // ASSERT_* expands to a bare `return;` that only compiles in a void
    // function. A failed enqueue here would mean kPoolSize > QueueT's
    // capacity, an invariant that can't actually happen (64 > 32).
    EXPECT_TRUE(queues[0].enqueue(dataset[i].get())) << "queue[0] full seeding the pool";
  }

  LoggerT logger;
  std::vector<std::thread> threads;
  for (size_t chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
    QueueT& q_in = queues[chunk_id];
    QueueT& q_out = queues[(chunk_id + 1) % n_chunks];
    const int start = sched.start_stage(chunk_id);
    const int end = sched.end_stage(chunk_id);
    const bool is_last = chunk_id == n_chunks - 1;
    const ExecutionModel em = sched.chunks[chunk_id].exec_model;

    if (em == ExecutionModel::kCuda) {
      threads.emplace_back(
          worker_with_record<QueueT, tree::AppData, 100>,
          static_cast<int>(chunk_id),
          std::ref(logger),
          std::ref(q_in),
          std::ref(q_out),
          [&disp, start, end](tree::AppData* app) { disp.dispatch_multi_stage(*app, start, end); },
          kNumToProcess,
          is_last);
    } else {
      // Core-pinning intentionally unsupported for the AppData path (see
      // apps/tree/omp/dispatchers.hpp) -- dispatch_multi_stage(AppData&, ...)
      // manages its own parallelism internally.
      threads.emplace_back(
          worker_with_record<QueueT, tree::AppData, 100>,
          static_cast<int>(chunk_id),
          std::ref(logger),
          std::ref(q_in),
          std::ref(q_out),
          [start, end](tree::AppData* app) { tree::omp::dispatch_multi_stage(*app, start, end); },
          kNumToProcess,
          is_last);
    }
  }
  for (auto& t : threads) t.join();

  return RunResult{std::move(dataset), std::move(logger)};
}

// ----------------------------------------------------------------------------
// T004: reference check -- a fresh AppData seeded with the same input points
// as `item`, run through the OMP oracle sequentially, diffed against what the
// schedule's real concurrent run produced. Reused pattern from the prior
// session's test_pipeline_chained_cu.cu, tagged with the schedule's uid so a
// failure names exactly which of the 29 schedules it came from (FR-005).
// ----------------------------------------------------------------------------
void CheckItemChained(tree::AppData& item, const std::string& schedule_uid) {
  auto mr = std::pmr::new_delete_resource();
  tree::AppData ref(mr, item.get_n_input());
  ref.u_input_points_s0 = item.u_input_points_s0;

  tree::omp::run_stage_1(ref);
  tree::omp::run_stage_2(ref);
  tree::omp::run_stage_3(ref);
  tree::omp::run_stage_4(ref);
  tree::omp::run_stage_5(ref);
  tree::omp::run_stage_6(ref);
  tree::omp::run_stage_7(ref);

  ASSERT_EQ(item.get_n_unique(), ref.get_n_unique()) << "schedule [" << schedule_uid << "]";
  ASSERT_EQ(item.get_n_brt_nodes(), ref.get_n_brt_nodes()) << "schedule [" << schedule_uid << "]";
  ASSERT_EQ(item.get_n_octree_nodes(), ref.get_n_octree_nodes())
      << "schedule [" << schedule_uid << "]";

  const auto n = item.get_n_octree_nodes();
  bool all_zero = n > 0;
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(item.u_oct_child_node_mask_s7[i], ref.u_oct_child_node_mask_s7[i])
        << "schedule [" << schedule_uid << "] node_mask mismatch at node " << i;
    EXPECT_EQ(item.u_oct_child_leaf_mask_s7[i], ref.u_oct_child_leaf_mask_s7[i])
        << "schedule [" << schedule_uid << "] leaf_mask mismatch at node " << i;
    if (item.u_oct_child_node_mask_s7[i] != 0) all_zero = false;
  }
  EXPECT_FALSE(all_zero) << "schedule [" << schedule_uid << "] octree node_mask is all-zero";
}

// ----------------------------------------------------------------------------
// T006: port of dashboard/generate.py's _coverage_time(intervals, min_cover=2)
// -- total wall-time (ms) during which a CPU chunk's and the GPU chunk's work
// windows were BOTH active, for different tasks, across the whole run (minus
// a warmup prefix). Returns 0.0 when the two never overlap -- including,
// trivially, when one of the chunk-id lists is empty (the all-CPU/all-GPU
// boundary schedules, where overlap does not apply).
// ----------------------------------------------------------------------------
double MeasureConcurrentMs(const LoggerT& logger,
                           const std::vector<int>& cpu_chunk_ids,
                           int gpu_chunk_id,
                           uint64_t freq) {
  std::vector<std::pair<double, double>> intervals;

  auto collect = [&](int chunk_id) {
    for (size_t task = kOverlapWarmupItems; task < kNumToProcess; ++task) {
      const auto& rec = logger.records_[task][static_cast<size_t>(chunk_id)];
      if (rec.start == 0) continue;
      const double a = LoggerT::cycles_to_milliseconds(rec.start, freq);
      const uint64_t end_cycles = rec.end > rec.start ? rec.end : rec.start;
      const double b = LoggerT::cycles_to_milliseconds(end_cycles, freq);
      intervals.emplace_back(a, b);
    }
  };
  for (const int c : cpu_chunk_ids) collect(c);
  collect(gpu_chunk_id);

  if (intervals.empty()) return 0.0;

  std::vector<std::pair<double, int>> events;
  events.reserve(intervals.size() * 2);
  for (const auto& [a, b] : intervals) {
    events.emplace_back(a, 1);
    events.emplace_back(b, -1);
  }
  std::ranges::sort(events);

  double total = 0.0;
  int cov = 0;
  bool have_prev = false;
  double prev = 0.0;
  for (const auto& [t, d] : events) {
    if (have_prev && cov >= 2) total += t - prev;
    cov += d;
    prev = t;
    have_prev = true;
  }
  return total;
}

// ----------------------------------------------------------------------------
// T005 (US1): every one of the 29 schedules must produce correct output.
// Uses ADD_FAILURE-based checks (not a hard ASSERT that would stop the whole
// test), so one schedule failing doesn't prevent the rest from being checked
// and reported -- per the spec's edge case.
// ----------------------------------------------------------------------------
TEST(SchedulePermutation, AllScheduleCorrectness) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device";

  const auto schedules = generate_all_schedules();
  for (const auto& sched : schedules) {
    RunResult result = run_schedule_once(sched);
    for (auto& item : result.dataset) {
      CheckItemChained(*item, sched.uid);
    }
  }
}

// ----------------------------------------------------------------------------
// T007 (US2): every schedule with both a CPU and a GPU chunk must show
// genuine overlap in >= 3 of 5 repeated runs. The two all-one-PU boundary
// schedules are skipped (overlap is not applicable -- only one PU is ever
// active, per the spec's edge case).
// ----------------------------------------------------------------------------
TEST(SchedulePermutation, OverlapAcrossRepeatedRuns) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device";

  const auto schedules = generate_all_schedules();
  const uint64_t freq = get_counter_frequency();

  for (const auto& sched : schedules) {
    std::vector<int> cpu_chunk_ids;
    std::optional<int> gpu_chunk_id;
    for (size_t i = 0; i < sched.chunks.size(); ++i) {
      if (sched.chunks[i].exec_model == ExecutionModel::kCuda) {
        gpu_chunk_id = static_cast<int>(i);
      } else {
        cpu_chunk_ids.push_back(static_cast<int>(i));
      }
    }
    if (!gpu_chunk_id.has_value() || cpu_chunk_ids.empty()) {
      continue;  // all-CPU or all-GPU: overlap not applicable
    }

    int overlapping_runs = 0;
    for (int run = 0; run < kOverlapRepeats; ++run) {
      RunResult result = run_schedule_once(sched);
      const double concurrent_ms =
          MeasureConcurrentMs(result.logger, cpu_chunk_ids, *gpu_chunk_id, freq);
      if (concurrent_ms > 0.0) ++overlapping_runs;
    }
    EXPECT_GE(overlapping_runs, kOverlapRequiredHits)
        << "schedule [" << sched.uid << "] only overlapped in " << overlapping_runs << "/"
        << kOverlapRepeats << " runs";
  }
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  parse_args_test(argc, argv);
  spdlog::set_level(spdlog::level::off);
  return RUN_ALL_TESTS();
}
