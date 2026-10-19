#pragma once
// ----------------------------------------------------------------------------
// Pinning check for CPU chunks (runtime/cpu-chunks spec, "A CPU chunk runs only on its tier's
// cores"; tree-cpu-pinning design D2).
//
// ExpectChunkPinned runs an app's stage range once unpinned and then, through its pinned
// dispatch, on two one-tier core lists in turn (two disjoint halves of this process's CPUs),
// all on one thread, as the profiler does when it moves from one tier's cells to the next. For
// each pinned run it checks, from what BT_PIN_PROBE_HERE() recorded inside the stages, that
//   - every thread that ran stage work had an affinity mask inside the core list,
//   - at most `n` threads ran stage work (n = the list's size), and
//   - every region of the AppData equals the unpinned run's.
// Needs a binary built with BT_PIN_PROBE (see platform/util/pin_probe.hpp).
// ----------------------------------------------------------------------------

#ifndef BT_PIN_PROBE
#error "cpu_pinning.hpp needs BT_PIN_PROBE: the stages record nothing without it"
#endif

#include <gtest/gtest.h>
#include <omp.h>
#include <sched.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "platform/util/pin_probe.hpp"
#include "platform/util/testing/input_purity.hpp"

namespace bt::testing {

namespace detail {

inline std::vector<int> AllowedCpus() {
  std::vector<int> cpus;
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    for (int c = 0; c < CPU_SETSIZE; ++c)
      if (CPU_ISSET(c, &set)) cpus.push_back(c);
  }
  return cpus;
}

inline std::string Join(const std::vector<int>& v) {
  std::ostringstream s;
  for (std::size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
  return s.str();
}

struct PinnedRun {
  std::vector<int> cores{};
  std::vector<pin_probe::Seen> seen{};
  std::vector<Bytes> regions{};
};

}  // namespace detail

// `unpinned(app)` and `pinned(cores, n, app)` run the same stage range; `regions(app)` lists
// every region of the AppData.
template <class App, class Unpinned, class Pinned, class Regions>
void ExpectChunkPinned(App& app, Unpinned unpinned, Pinned pinned, Regions regions) {
  const std::vector<int> cpus = detail::AllowedCpus();
  if (cpus.size() < 2) GTEST_SKIP() << "needs at least 2 CPUs, has " << cpus.size();
  const auto half = static_cast<std::ptrdiff_t>(cpus.size() / 2);
  std::vector<detail::PinnedRun> runs = {{{cpus.begin(), cpus.begin() + half}, {}, {}},
                                         {{cpus.begin() + half, cpus.end()}, {}, {}}};
  std::vector<detail::Bytes> want;

  // A fresh thread, so the test thread's mask and OpenMP state stay as they were. Every run
  // has the whole machine as its default team.
  std::thread([&] {
    omp_set_num_threads(omp_get_num_procs());
    unpinned(app);
    want = detail::Snapshot(regions(app));
    static int armings = 0;
    for (detail::PinnedRun& r : runs) {
      {
        const std::lock_guard lock(pin_probe::mu);
        pin_probe::seen.clear();
      }
      pin_probe::epoch.store(++armings);
      pinned(r.cores, static_cast<int>(r.cores.size()), app);
      pin_probe::epoch.store(0);
      const std::lock_guard lock(pin_probe::mu);
      r.seen = pin_probe::seen;
      r.regions = detail::Snapshot(regions(app));
    }
  }).join();

  for (const detail::PinnedRun& r : runs) {
    const std::set<int> tier(r.cores.begin(), r.cores.end());
    std::set<pid_t> tids;
    std::size_t outside = 0;
    std::ostringstream report;
    for (const pin_probe::Seen& s : r.seen) {
      tids.insert(s.tid);
      const bool in = std::ranges::all_of(s.cpus, [&](int c) { return tier.contains(c); });
      outside += in ? 0 : 1;
      report << "\n  thread " << s.tid << ": mask {" << detail::Join(s.cpus) << "}"
             << (in ? "" : "  <- outside the core list");
    }
    SCOPED_TRACE("cores {" + detail::Join(r.cores) + "}, n " + std::to_string(r.cores.size()));
    ASSERT_FALSE(r.seen.empty()) << "no thread recorded stage work (is the probe compiled in?)";
    EXPECT_EQ(outside, 0u) << tids.size() << " threads ran stage work:" << report.str();
    EXPECT_LE(tids.size(), r.cores.size()) << "more threads ran stage work than the thread count";
    for (std::size_t i = 0; const Region& reg : regions(app)) {
      EXPECT_TRUE(r.regions[i].size() == want[i].size() &&
                  std::memcmp(r.regions[i].data(), want[i].data(), want[i].size()) == 0)
          << reg.name << " differs from the unpinned run";
      ++i;
    }
  }
}

}  // namespace bt::testing
