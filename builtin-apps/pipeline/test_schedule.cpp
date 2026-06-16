// Unit tests for the pipeline schedule contract — the correctness gate the
// pipeline path was missing. validate_schedule_coverage() must accept exactly
// the schedules whose chunks contiguously cover stages [1, n_stages] and reject
// every malformed/incomplete one (gaps, overlaps, dropped first/last stage,
// out-of-range, empty). The DroppedFirstStage case is the regression for the
// double-+1 scheduler bug (commit 3c6de54).
#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "builtin-apps/pipeline/schedule.hpp"

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

}  // namespace
