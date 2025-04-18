#pragma once

#include <omp.h>
#include <spdlog/spdlog.h>

enum class LogKernelType {
  kOMP,
  kCUDA,
  kVK,
};

// ---------------------------------------------------------------------
// New design
// ---------------------------------------------------------------------

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

constexpr bool kEnableLogging = true;

#define LOG_KERNEL(kernel_type, stage, appdata)           \
  if constexpr (kEnableLogging) {                         \
    log_kernel_console_impl<kernel_type>(stage, appdata); \
  }
