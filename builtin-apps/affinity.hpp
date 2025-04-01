#pragma once

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>

#endif

#include <cstdlib>
#include <stdexcept>
#include <vector>

inline void bind_thread_to_cores(const std::vector<int>& core_ids) {
#if defined(_WIN32) || defined(_WIN64)
  // Windows implementation

  // Build a 64-bit mask from the given core IDs.
  // Note: 1ULL << core_id is valid for core_id in [0, 63].
  DWORD_PTR mask = 0;
  for (int core_id : core_ids) {
    mask |= (1ULL << core_id);
  }

  // Get a handle to the current thread.
  HANDLE thread = GetCurrentThread();

  // Apply the affinity mask to pin the thread.
  DWORD_PTR result = SetThreadAffinityMask(thread, mask);
  if (result == 0) {
    throw std::runtime_error("Failed to set thread affinity on Windows");
  }

#else
  // Linux (and other POSIX-like) implementation

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  for (int core_id : core_ids) {
    CPU_SET(core_id, &cpuset);
  }

  // sched_setaffinity for the current thread (pid=0).
  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
    // Print out the cpuset in a human readable way
    std::string cores_str;
    for (int i = 0; i < CPU_SETSIZE; i++) {
      if (CPU_ISSET(i, &cpuset)) {
        if (!cores_str.empty()) {
          cores_str += ", ";
        }
        cores_str += std::to_string(i);
      }
    }

    if (cores_str.empty()) {
      cores_str = "<empty>";
    }

    throw std::runtime_error("Failed to pin thread to cores on Linux: " + cores_str);
  }
#endif
}

// // Function to pin the current thread to a given core
// inline void bind_thread_to_cores(const std::vector<int>& core_ids) {
//   cpu_set_t cpuset;
//   CPU_ZERO(&cpuset);

//   for (int core_id : core_ids) {
//     CPU_SET(core_id, &cpuset);
//   }

//   if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
//     throw std::runtime_error("Failed to pin thread to cores");
//   }
// }

// inline void cpu_set_wrapper(int core, cpu_set_t* cpuset) {
//   // Now the macro expansion is isolated to this single call:
//   CPU_SET(core, cpuset);
// }

// template <int... Cores>
// void bind_thread_to_cores() {
//   cpu_set_t cpuset;
//   CPU_ZERO(&cpuset);

//   // Expand the wrapper call over each 'Core' in the pack:
//   (cpu_set_wrapper(Cores, &cpuset), ...);

//   if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
//     throw std::runtime_error("Failed to pin thread to cores");
//   }
// }
