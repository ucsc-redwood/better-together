// Unit tests for the pipeline schedule contract — the correctness gate the
// pipeline path was missing. validate_schedule_coverage() must accept exactly
// the schedules whose chunks contiguously cover stages [1, n_stages] and reject
// every malformed/incomplete one (gaps, overlaps, dropped first/last stage,
// out-of-range, empty). The DroppedFirstStage case is the regression for the
// double-+1 scheduler bug (commit 3c6de54).
#include <gtest/gtest.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "runtime/config_reader.hpp"  // readSchedulesFromJson (the schedule consumer)
#include "runtime/record.hpp"
#include "runtime/schedule.hpp"

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

// ---- first_concurrent_gpu_chunk: a GPU backend may appear in at most one chunk ----
// The GPU dispatcher's single command buffer/fence is shared across the per-chunk
// worker threads, so >1 GPU chunk races it into VK_ERROR_DEVICE_LOST (diagnosed with
// GPU-assisted validation, 2026-06-17). z3 emits one contiguous GPU chunk per PU;
// this guard rejects any schedule that would spawn concurrent GPU dispatchers.

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

TEST(ScheduleGpuReuse, TwoVulkanChunksRejected) {
  const auto reason = first_concurrent_gpu_chunk(
      make_pu_schedule({{ExecutionModel::kVulkan, ProcessorType::kVulkan},
                        {ExecutionModel::kOMP, ProcessorType::kBigCore},
                        {ExecutionModel::kVulkan, ProcessorType::kVulkan}}));
  ASSERT_TRUE(reason.has_value());
  EXPECT_NE(reason->find("chunk 2"), std::string::npos);  // the offending (second Vulkan) chunk
}

TEST(ScheduleGpuReuse, TwoCudaChunksRejected) {
  EXPECT_TRUE(
      first_concurrent_gpu_chunk(make_pu_schedule({{ExecutionModel::kCuda, ProcessorType::kCuda},
                                                   {ExecutionModel::kOMP, ProcessorType::kBigCore},
                                                   {ExecutionModel::kCuda, ProcessorType::kCuda}}))
          .has_value());
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
