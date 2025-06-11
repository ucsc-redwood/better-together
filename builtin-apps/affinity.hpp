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

#if defined(__APPLE__)
#include <mach/mach_init.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#endif

#if defined(__APPLE__)
#include <pthread.h>

#include <stdexcept>

inline void set_thread_qos_class(qos_class_t qos_class) {
  int err = pthread_set_qos_class_self_np(qos_class, 0);
  if (err != 0) {
    throw std::runtime_error("Failed to set thread QoS class on macOS");
  }
}
#endif

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

#elif defined(__APPLE__)

  // // Use the first core ID as a grouping tag
  // int affinity_tag = core_ids[0];

  // thread_affinity_policy_data_t policy = {affinity_tag};
  // kern_return_t result = thread_policy_set(mach_thread_self(),
  //                                          THREAD_AFFINITY_POLICY,
  //                                          (thread_policy_t)&policy,
  //                                          THREAD_AFFINITY_POLICY_COUNT);

  // if (result != KERN_SUCCESS) {
  //   throw std::runtime_error(
  //       "Failed to set thread affinity tag on macOS (tag = " + std::to_string(affinity_tag) +
  //       ")");
  // }

  set_thread_qos_class(QOS_CLASS_USER_INITIATED);

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
