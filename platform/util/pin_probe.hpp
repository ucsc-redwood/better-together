#pragma once
// ----------------------------------------------------------------------------
// Test hook for runtime/cpu-chunks ("A CPU chunk runs only on its tier's cores").
//
// BT_PIN_PROBE_HERE() marks a place where a thread runs stage work. It compiles to nothing
// unless BT_PIN_PROBE is defined, which only the OMP test binaries that check pinning do
// (apps/*/CMakeLists.txt). There, while armed, it records once per thread and per arming the
// thread id and the thread's affinity mask (platform/util/testing/cpu_pinning.hpp reads them).
// ----------------------------------------------------------------------------

#ifdef BT_PIN_PROBE

#include <sched.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <vector>

namespace bt::pin_probe {

struct Seen {
  pid_t tid;
  std::vector<int> cpus;  // the thread's affinity mask
};

inline std::atomic<int> epoch{0};  // 0: disarmed
inline std::mutex mu;
inline std::vector<Seen> seen;

inline void record() noexcept {
  const int e = epoch.load(std::memory_order_relaxed);
  thread_local int last = 0;
  if (e == 0 || e == last) return;
  last = e;
  Seen s{gettid(), {}};
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    for (int c = 0; c < CPU_SETSIZE; ++c)
      if (CPU_ISSET(c, &set)) s.cpus.push_back(c);
  }
  const std::lock_guard lock(mu);
  seen.push_back(std::move(s));
}

}  // namespace bt::pin_probe

#define BT_PIN_PROBE_HERE() ::bt::pin_probe::record()

#else

#define BT_PIN_PROBE_HERE() ((void)0)

#endif
