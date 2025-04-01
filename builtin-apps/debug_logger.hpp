#pragma once

#include <omp.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include "app.hpp"
#include "resources_path.hpp"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

enum class LogKernelType {
  kOMP,
  kCUDA,
  kVK,
};

// ---------------------------------------------------------------------
// Old working design
// ---------------------------------------------------------------------

// template <LogKernelType kernel_type>
// inline void log_kernel(const int stage, const void *appdata_addr) {
//   if constexpr (kernel_type == LogKernelType::kOMP) {
//     uint64_t thread_id = 0;

// #if defined(_WIN32) || defined(_WIN64)
//     thread_id = (uint64_t)GetCurrentThreadId();
// #else
//     thread_id = (uint64_t)pthread_self();
// #endif
//     spdlog::debug("[omp][{}][thread {}/{}] process_stage_{}, app_data: {:p}",
//                   thread_id,
//                   omp_get_thread_num(),
//                   omp_get_num_threads(),
//                   stage,
//                   appdata_addr);
//   } else if constexpr (kernel_type == LogKernelType::kCUDA) {
//     spdlog::debug("[cuda] process_stage_{}, app_data: {:p}", stage, appdata_addr);
//   } else if constexpr (kernel_type == LogKernelType::kVK) {
//     spdlog::debug("[vk] process_stage_{}, app_data: {:p}", stage, appdata_addr);
//   }
// }

// ---------------------------------------------------------------------
// New design
// ---------------------------------------------------------------------

inline std::shared_ptr<spdlog::logger> getFileLogger() {
  // This static variable is initialized only once.
  auto log_dir = helpers::get_log_storage_location();
  static auto logger = spdlog::basic_logger_mt("file_logger", log_dir / "logs.txt");
  return logger;
}

template <LogKernelType kernel_type>
void log_kernel_impl(const int stage, const void *appdata_addr) {
  auto file_logger = getFileLogger();

  // Log the
  // 1) appdata_addr address
  // 2) stage number
  // 3) kernel type (OMP, CUDA, VK)
  // 4) (if OMP) core ID
  // 5) (if OMP) core type (big, medium, little)
  // 6) (if OMP) thread id (0..n_threads-1 using omp_get_thread_num())
  // 7) (if OMP) n_threads (using omp_get_num_threads())

  if constexpr (kernel_type == LogKernelType::kOMP) {
    // int core_id = -1;

    // Get core ID on Linux
    // core_id = sched_getcpu();
    uint64_t core_id = (uint64_t)pthread_self();

    file_logger->debug("[omp][Core: {}][Thread: {}/{}] [Stage: {}] [App: {:p}]",
                       core_id,
                       omp_get_thread_num() + 1,
                       omp_get_num_threads(),
                       stage,
                       appdata_addr);

  } else if constexpr (kernel_type == LogKernelType::kCUDA) {
    file_logger->debug("[cuda] [Stage: {}] [App: {:p}]", stage, appdata_addr);

  } else if constexpr (kernel_type == LogKernelType::kVK) {
    file_logger->debug("[vk] [Stage: {}] [App: {:p}]", stage, appdata_addr);
  }
}

template <LogKernelType kernel_type>
void log_kernel_console_impl(const int stage, const void *appdata_addr) {
  if constexpr (kernel_type == LogKernelType::kOMP) {
    spdlog::debug("[omp][Core: {}][Thread: {}/{}] [Stage: {}] [App: {:p}]",
                  (uint64_t)pthread_self(),
                  omp_get_thread_num() + 1,
                  omp_get_num_threads(),
                  stage,
                  appdata_addr);

  } else if constexpr (kernel_type == LogKernelType::kCUDA) {
    spdlog::debug("[cuda] [Stage: {}] [App: {:p}]", stage, appdata_addr);

  } else if constexpr (kernel_type == LogKernelType::kVK) {
    spdlog::debug("[vk] [Stage: {}] [App: {:p}]", stage, appdata_addr);
  }
}

// #ifdef NDEBUG
// #define LOG_KERNEL(kernel_type, stage, appdata) ((void)0)
// #else
#define LOG_KERNEL(kernel_type, stage, appdata)           \
  if (g_debug_filelogger) {                               \
    log_kernel_impl<kernel_type>(stage, appdata);         \
  } else {                                                \
    log_kernel_console_impl<kernel_type>(stage, appdata); \
  }
// #endif
