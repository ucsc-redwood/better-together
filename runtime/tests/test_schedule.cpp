// Unit tests for the pipeline schedule contract — the correctness gate the
// pipeline path was missing. validate_schedule_coverage() must accept exactly
// the schedules whose chunks contiguously cover stages [1, n_stages] and reject
// every malformed/incomplete one (gaps, overlaps, dropped first/last stage,
// out-of-range, empty). The DroppedFirstStage case is the regression for the
// double-+1 scheduler bug (commit 3c6de54).
#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "runtime/config_reader.hpp"  // readSchedulesFromJson (the schedule consumer)
#include "runtime/pipeline.hpp"       // inflight_worker
#include "runtime/record.hpp"
#include "runtime/schedule.hpp"
#include "runtime/spsc_queue.hpp"

namespace {

// Build an OMP schedule from a list of [start_stage, end_stage] (1-based, inclusive).
Schedule make_schedule(const std::vector<std::pair<int, int>>& ranges,
                       const std::string& uid = "test") {
  Schedule s;
  s.uid = uid;
  for (const auto& [start, end] : ranges) {
    ChunkConfig c;
    c.exec_model = ExecutionModel::kOMP;
    c.start_stage = start;
    c.end_stage = end;
    c.cpu_proc_type = ProcessorType::kBigCore;
    s.chunks.push_back(c);
  }
  return s;
}

TEST(ScheduleCoverage, FullSingleChunkPasses) {
  EXPECT_NO_THROW(validate_schedule_coverage(make_schedule({{1, 7}}), 7));
}

TEST(ScheduleCoverage, ContiguousMultiChunkPasses) {
  EXPECT_NO_THROW(validate_schedule_coverage(make_schedule({{1, 3}, {4, 7}}), 7));
}

TEST(ScheduleCoverage, PerStageChunksPass) {
  EXPECT_NO_THROW(validate_schedule_coverage(
      make_schedule({{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}}), 7));
}

// Regression for the double-+1 bug (3c6de54): a chunk starting at 2 dropped stage 1.
TEST(ScheduleCoverage, DroppedFirstStageThrows) {
  EXPECT_THROW(validate_schedule_coverage(make_schedule({{2, 7}}), 7), std::runtime_error);
}

TEST(ScheduleCoverage, GapThrows) {
  EXPECT_THROW(validate_schedule_coverage(make_schedule({{1, 3}, {5, 7}}), 7), std::runtime_error);
}

TEST(ScheduleCoverage, OverlapThrows) {
  EXPECT_THROW(validate_schedule_coverage(make_schedule({{1, 4}, {4, 7}}), 7), std::runtime_error);
}

TEST(ScheduleCoverage, MissingLastStageThrows) {
  EXPECT_THROW(validate_schedule_coverage(make_schedule({{1, 6}}), 7), std::runtime_error);
}

TEST(ScheduleCoverage, ExceedsNStagesThrows) {
  EXPECT_THROW(validate_schedule_coverage(make_schedule({{1, 8}}), 7), std::runtime_error);
}

TEST(ScheduleCoverage, EmptyScheduleThrows) {
  EXPECT_THROW(validate_schedule_coverage(make_schedule({}), 7), std::runtime_error);
}

TEST(ScheduleCoverage, BackwardChunkThrows) {
  EXPECT_THROW(validate_schedule_coverage(make_schedule({{1, 0}}), 7), std::runtime_error);
}

// ---- first_unavailable_pu: skip-don't-crash on an absent PU --------------------
// Build a schedule with explicit per-chunk PUs (exec model + cpu tier).
namespace {
Schedule make_pu_schedule(const std::vector<std::pair<ExecutionModel, ProcessorType>>& pus) {
  Schedule s;
  s.uid = "test";
  int stage = 1;
  for (const auto& [em, pt] : pus) {
    ChunkConfig c;
    c.exec_model = em;
    c.start_stage = stage;
    c.end_stage = stage;
    if (em == ExecutionModel::kOMP) c.cpu_proc_type = pt;
    s.chunks.push_back(c);
    ++stage;
  }
  return s;
}
}  // namespace

TEST(SchedulePUs, AllPresentReturnsNullopt) {
  // Big-only device: a Big chunk is fine.
  auto big_only = [](ProcessorType pt) { return pt == ProcessorType::kBigCore; };
  EXPECT_FALSE(first_unavailable_pu(
                   make_pu_schedule({{ExecutionModel::kOMP, ProcessorType::kBigCore}}), big_only)
                   .has_value());
}

// The plan's gate: a Little chunk on the Big-only MiniPC must be reported (so the
// executor skips+warns), not run into an absent-tier throw in a worker thread.
TEST(SchedulePUs, LittleChunkOnBigOnlyReported) {
  auto big_only = [](ProcessorType pt) { return pt == ProcessorType::kBigCore; };
  const auto reason = first_unavailable_pu(
      make_pu_schedule({{ExecutionModel::kOMP, ProcessorType::kLittleCore}}), big_only);
  ASSERT_TRUE(reason.has_value());
  EXPECT_NE(reason->find("little"), std::string::npos);
}

TEST(SchedulePUs, ReportsFirstOffendingChunk) {
  auto big_only = [](ProcessorType pt) { return pt == ProcessorType::kBigCore; };
  const auto reason =
      first_unavailable_pu(make_pu_schedule({{ExecutionModel::kOMP, ProcessorType::kBigCore},
                                             {ExecutionModel::kOMP, ProcessorType::kMediumCore}}),
                           big_only);
  ASSERT_TRUE(reason.has_value());
  EXPECT_NE(reason->find("chunk 1"), std::string::npos);
}

TEST(SchedulePUs, GpuChunkPresenceFromPredicate) {
  // A predicate that lacks Vulkan flags a Vulkan chunk.
  auto cpu_only = [](ProcessorType pt) { return pt != ProcessorType::kVulkan; };
  EXPECT_TRUE(first_unavailable_pu(
                  make_pu_schedule({{ExecutionModel::kVulkan, ProcessorType::kVulkan}}), cpu_only)
                  .has_value());
}

// ---- first_concurrent_gpu_chunk: Vulkan may serve several chunks, CUDA only one ----
// Each Vulkan GPU chunk records into its own Sequence and submits under the Engine's
// submit mutex (runtime/gpu-chunks spec, design C2), so repeated Vulkan chunks are allowed.
// The CUDA dispatcher's single stream is still shared across chunk threads -> rejected.

TEST(ScheduleGpuReuse, SingleGpuChunkAllowed) {
  EXPECT_FALSE(first_concurrent_gpu_chunk(
                   make_pu_schedule({{ExecutionModel::kOMP, ProcessorType::kBigCore},
                                     {ExecutionModel::kVulkan, ProcessorType::kVulkan}}))
                   .has_value());
}

TEST(ScheduleGpuReuse, NoGpuChunkAllowed) {
  EXPECT_FALSE(first_concurrent_gpu_chunk(
                   make_pu_schedule({{ExecutionModel::kOMP, ProcessorType::kBigCore},
                                     {ExecutionModel::kOMP, ProcessorType::kLittleCore}}))
                   .has_value());
}

TEST(ScheduleGpuReuse, TwoVulkanChunksAllowed) {
  EXPECT_FALSE(first_concurrent_gpu_chunk(
                   make_pu_schedule({{ExecutionModel::kVulkan, ProcessorType::kVulkan},
                                     {ExecutionModel::kOMP, ProcessorType::kBigCore},
                                     {ExecutionModel::kVulkan, ProcessorType::kVulkan}}))
                   .has_value());
}

TEST(ScheduleGpuReuse, TwoCudaChunksRejected) {
  const auto reason =
      first_concurrent_gpu_chunk(make_pu_schedule({{ExecutionModel::kCuda, ProcessorType::kCuda},
                                                   {ExecutionModel::kOMP, ProcessorType::kBigCore},
                                                   {ExecutionModel::kCuda, ProcessorType::kCuda}}));
  ASSERT_TRUE(reason.has_value());
  EXPECT_NE(reason.value_or("").find("chunk 2"), std::string::npos);  // the second CUDA chunk
}

// ---- inflight_worker: up to `depth` items in flight, handed on in FIFO order ----------
// Submission spec, "Depth 2 preserves order and results" / "Depth 1 is the old behaviour"
// (design D3). Hardware-free: submit/complete are stand-ins that track how many items are
// outstanding and the order completions happen in.
struct InflightItem {
  int id = 0;
  void reset() {}
};

std::pair<std::vector<int>, size_t> RunInflight(const size_t depth, const int n) {
  std::vector<InflightItem> items(n);
  SPSCQueue<InflightItem*, 64> q_in;
  SPSCQueue<InflightItem*, 64> q_out;
  for (int i = 0; i < n; ++i) {
    items[i].id = i;
    EXPECT_TRUE(q_in.enqueue(&items[i]));
  }
  size_t outstanding = 0;
  size_t max_outstanding = 0;
  std::vector<int> completed;
  std::vector<int> submitted;
  inflight_worker<SPSCQueue<InflightItem*, 64>, InflightItem>(
      q_in,
      q_out,
      [&](InflightItem* item) {
        submitted.push_back(item->id);
        max_outstanding = std::max(max_outstanding, ++outstanding);
        return static_cast<uint64_t>(item->id) + 100;  // the ticket
      },
      [&](InflightItem* item, const uint64_t ticket) {
        EXPECT_EQ(ticket, static_cast<uint64_t>(item->id) + 100);
        --outstanding;
        completed.push_back(item->id);
      },
      n,
      /*is_last=*/false,
      depth);
  // Downstream sees the same order.
  for (int i = 0; i < n; ++i) {
    InflightItem* item = nullptr;
    EXPECT_TRUE(q_out.dequeue(item));
    EXPECT_EQ(item->id, completed[i]);
  }
  EXPECT_EQ(submitted, completed);
  return {completed, max_outstanding};
}

TEST(InflightWorker, Depth2KeepsTwoInFlightAndPreservesOrder) {
  const auto [order, max_outstanding] = RunInflight(2, 20);
  std::vector<int> expected(20);
  for (int i = 0; i < 20; ++i) expected[i] = i;
  EXPECT_EQ(order, expected);
  EXPECT_EQ(max_outstanding, 2u);
}

TEST(InflightWorker, Depth1IsSubmitThenWait) {
  const auto [order, max_outstanding] = RunInflight(1, 20);
  EXPECT_EQ(order.size(), 20u);
  EXPECT_EQ(max_outstanding, 1u);  // each item completes before the next is submitted
}

TEST(InflightWorker, DepthKnobAndPoolSize) {
  EXPECT_EQ(min_pool_size(/*gpu_chunks=*/2, /*cpu_chunks=*/1, /*depth=*/2), 5u);
  EXPECT_EQ(min_pool_size(1, 0, 1), 1u);
}

// ---- Case 5.2: a >kMaxChunks schedule must throw a clean std::out_of_range from the
// Logger bound check (caught by worker_with_record's try/catch -> process survives),
// not silently corrupt memory as the old hard-coded-4 array did. Test the guard directly.
TEST(PipelineRobust, LoggerRejectsOverflowChunk) {
  Logger<4> logger;
  EXPECT_NO_THROW(logger.start_tick(0, Logger<4>::kMaxChunks - 1));  // last valid chunk id
  EXPECT_THROW(logger.start_tick(0, Logger<4>::kMaxChunks), std::out_of_range);
  EXPECT_THROW(logger.end_tick(0, Logger<4>::kMaxChunks + 3), std::out_of_range);
}

// ---- readSchedulesFromJson: the C++ consumer of the schedule contract ----------
// New contract: chunks carry explicit 1-based-inclusive start_stage/end_stage (no
// stages[] array, no +1 base-shift). These guard config_reader.hpp against a
// regression of the off-by-one that previously dropped a stage.

TEST(ScheduleConsumer, ParsesNewFormatVerbatim) {
  const auto j = nlohmann::json::parse(R"([
    {"uid":"S1","chunks":[
      {"id":0,"core_type":"Big","start_stage":1,"end_stage":5,"time":1.0},
      {"id":1,"core_type":"GPU","start_stage":6,"end_stage":9,"time":2.0,"hardware":"gpu_cuda"}
    ]}
  ])");
  const auto scheds = readSchedulesFromJson(j);
  ASSERT_EQ(scheds.size(), 1u);
  ASSERT_EQ(scheds[0].chunks.size(), 2u);
  // Stages read verbatim — 1-based, no +1.
  EXPECT_EQ(scheds[0].chunks[0].start_stage, 1);
  EXPECT_EQ(scheds[0].chunks[0].end_stage, 5);
  EXPECT_EQ(scheds[0].chunks[0].exec_model, ExecutionModel::kOMP);
  EXPECT_EQ(scheds[0].chunks[0].cpu_proc_type, ProcessorType::kBigCore);
  EXPECT_EQ(scheds[0].chunks[1].start_stage, 6);
  EXPECT_EQ(scheds[0].chunks[1].end_stage, 9);
  EXPECT_EQ(scheds[0].chunks[1].exec_model, ExecutionModel::kCuda);
  // The parsed schedule satisfies the coverage gate for a 9-stage app.
  EXPECT_NO_THROW(validate_schedule_coverage(scheds[0], 9));
}

TEST(ScheduleConsumer, RejectsMissingStageBounds) {
  const auto j = nlohmann::json::parse(
      R"([{"uid":"S","chunks":[{"core_type":"Big","start_stage":1}]}])");  // no end_stage
  EXPECT_THROW(readSchedulesFromJson(j), std::runtime_error);
}

TEST(ScheduleConsumer, RejectsInvertedRange) {
  const auto j = nlohmann::json::parse(
      R"([{"uid":"S","chunks":[{"core_type":"Big","start_stage":5,"end_stage":2}]}])");
  EXPECT_THROW(readSchedulesFromJson(j), std::runtime_error);
}

// ---- Cross-tool round-trip contract (Lever B) ----------------------------------
// The committed fixture tests/fixtures/schedule.contract.json is the shared example
// of the schedule contract: the Python producer side validates it against
// schemas/schedule.schema.json (test_schedule_contract.py), and the C++ consumer
// side (here) must read it through readSchedulesFromJson + validate_schedule_coverage.
// If the consumer ever drifts from the contract shape, this reds; the Python test
// reds if the schema/producer drifts. BT_SCHEDULE_FIXTURE is the absolute path,
// passed by CMake.
#ifdef BT_SCHEDULE_FIXTURE
TEST(ScheduleContract, ConsumerReadsCommittedFixture) {
  std::ifstream f(BT_SCHEDULE_FIXTURE);
  ASSERT_TRUE(f.good()) << "contract fixture not found at " << BT_SCHEDULE_FIXTURE;
  nlohmann::json j;
  ASSERT_NO_THROW(f >> j);
  const auto scheds = readSchedulesFromJson(j);
  ASSERT_GE(scheds.size(), 1u);
  // The fixture's schedules each cover a 9-stage app contiguously.
  for (const auto& s : scheds) {
    EXPECT_NO_THROW(validate_schedule_coverage(s, 9)) << "fixture schedule " << s.uid;
  }
}
#endif

}  // namespace
