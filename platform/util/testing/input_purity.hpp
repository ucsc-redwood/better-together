#pragma once
// ----------------------------------------------------------------------------
// Input-purity check for CPU stage kernels (profiling/cost-table spec, "A stage is
// profiled on unmodified input"; cpu-chunk-costs design D2).
//
// bm-prof times a stage by running it again and again on one AppData. A stage that
// writes a buffer it reads (an in-place sort) is then timed on its own output from the
// second sample on, not on the input a pipeline item brings. ExpectStageInputPure runs
// one stage twice on an AppData whose earlier stages have already run and checks that
//   - every region the stage does not write equals its contents before the first run,
//     after each run, and
//   - every region it writes holds the same bytes after both runs.
// The app lists its regions (every buffer and scalar field) and, per stage, the names
// of the ones that stage writes; everything else counts as an input.
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace bt::testing {

// A named byte range of an AppData: a buffer's storage or a scalar field.
struct Region {
  std::string name;
  const void* data;
  std::size_t bytes;
};

template <class Vec>
[[nodiscard]] Region Buf(std::string name, const Vec& v) {
  return {std::move(name), v.data(), v.size() * sizeof(*v.data())};
}

template <class T>
[[nodiscard]] Region Scalar(std::string name, const T& x) {
  return {std::move(name), &x, sizeof(T)};
}

namespace detail {

using Bytes = std::vector<std::byte>;

inline std::vector<Bytes> Snapshot(const std::vector<Region>& regions) {
  std::vector<Bytes> out;
  out.reserve(regions.size());
  for (const Region& r : regions) {
    const auto* p = static_cast<const std::byte*>(r.data);
    out.emplace_back(p, p + r.bytes);
  }
  return out;
}

inline ::testing::AssertionResult SameBytes(const Region& now,
                                            const Bytes& then,
                                            const char* what) {
  if (now.bytes != then.size()) {
    return ::testing::AssertionFailure() << now.name << " " << what << ": size changed from "
                                         << then.size() << " to " << now.bytes << " bytes";
  }
  if (now.bytes == 0 || std::memcmp(now.data, then.data(), now.bytes) == 0) {
    return ::testing::AssertionSuccess();
  }
  const auto* p = static_cast<const std::byte*>(now.data);
  std::size_t first = 0;
  std::size_t n_diff = 0;
  for (std::size_t i = now.bytes; i-- > 0;) {
    if (p[i] != then[i]) {
      first = i;
      ++n_diff;
    }
  }
  return ::testing::AssertionFailure() << now.name << " " << what << ": " << n_diff << " of "
                                       << now.bytes << " bytes differ, the first at byte " << first;
}

}  // namespace detail

// `run(app, stage)` runs the stage the way the profiler does; `regions(app)` lists every
// region of the AppData; `outputs` names the regions this stage writes.
template <class App, class Run, class Regions>
void ExpectStageInputPure(
    App& app, const int stage, Run run, Regions regions, const std::set<std::string>& outputs) {
  for (const std::string& name : outputs) {
    bool found = false;
    for (const Region& r : regions(app)) found |= r.name == name;
    ASSERT_TRUE(found) << "stage " << stage << " output " << name << " is not a listed region";
  }
  const std::vector<detail::Bytes> before = detail::Snapshot(regions(app));
  run(app, stage);
  const std::vector<detail::Bytes> after_first = detail::Snapshot(regions(app));
  for (std::size_t i = 0; const Region& r : regions(app)) {
    if (!outputs.contains(r.name)) {
      EXPECT_TRUE(detail::SameBytes(r, before[i], "(input) after the first run"))
          << "stage " << stage;
    }
    ++i;
  }
  run(app, stage);
  for (std::size_t i = 0; const Region& r : regions(app)) {
    if (outputs.contains(r.name)) {
      EXPECT_TRUE(detail::SameBytes(r, after_first[i], "(output) second run vs first"))
          << "stage " << stage;
    } else {
      EXPECT_TRUE(detail::SameBytes(r, before[i], "(input) after the second run"))
          << "stage " << stage;
    }
    ++i;
  }
}

}  // namespace bt::testing
