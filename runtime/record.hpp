#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "platform/registry/conf.hpp"
#include "schedule.hpp"

#if defined(__aarch64__)

// ========== ARM64 ==========
static inline uint64_t now_cycles() {
  uint64_t cycles;
  asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
  return cycles;
}

static inline uint32_t get_counter_frequency() {
  uint32_t freq;
  asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
  return freq;
}

#elif defined(__x86_64__) || defined(_M_X64)

// ========== x86_64 ==========
#include <x86intrin.h>

static inline uint64_t now_cycles() {
  return __rdtsc();  // Read time-stamp counter
}

static inline uint64_t get_counter_frequency() {
  // TSC frequency is not architectural; returning 0 (the old placeholder) made
  // every cycles->ms conversion divide by zero and silently yield inf/nan on
  // x86 (e.g. rocky-ryzen). Calibrate once against a steady clock instead.
  static const uint64_t tsc_hz = [] {
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t c0 = now_cycles();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const uint64_t c1 = now_cycles();
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    return static_cast<uint64_t>(static_cast<double>(c1 - c0) / seconds);
  }();
  return tsc_hz;
}

#elif defined(__arm__)

// ========== ARM 32-bit (armv7-a) ==========
// armv7 userspace usually can't read the CP15 generic timer (EL0 access is trapped on
// most Android kernels), so use the monotonic clock as a nanosecond "cycle" counter;
// the frequency is then 1e9 and the cycles->ms math is unchanged.
#include <time.h>

static inline uint64_t now_cycles() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
         static_cast<uint64_t>(ts.tv_nsec);
}

static inline uint64_t get_counter_frequency() {
  return 1000000000ull;  // ns ticks
}

#else
#error "Unsupported architecture"
#endif

struct Record {
  // ProcessorType processed_by;
  uint64_t start;  // Cycle counter when starting
  uint64_t end;    // Cycle counter when finishing
};

template <size_t NumToProcess>
struct Logger {
  // Worst case is one chunk per stage; the canonical AlexNetCIFAR has 11 stages,
  // so 16 leaves headroom for every app. The old hard-coded 4 silently corrupted
  // memory when a (machine-generated) schedule had more than four chunks.
  static constexpr size_t kMaxChunks = 16;

  // records_[processing_id][chunk_id]
  std::array<std::array<Record, kMaxChunks>, NumToProcess> records_;

  explicit Logger() { records_.fill(std::array<Record, kMaxChunks>{}); }

  void start_tick(const uint32_t processing_id, const int chunk_id) {
    if (static_cast<size_t>(chunk_id) >= kMaxChunks) {
      throw std::out_of_range("Logger::start_tick: chunk_id >= kMaxChunks");
    }
    records_[processing_id][chunk_id].start = now_cycles();
  }

  void end_tick(const uint32_t processing_id, const int chunk_id) {
    if (static_cast<size_t>(chunk_id) >= kMaxChunks) {
      throw std::out_of_range("Logger::end_tick: chunk_id >= kMaxChunks");
    }
    records_[processing_id][chunk_id].end = now_cycles();
  }

  void dump_records() const {
    for (size_t i = 0; i < records_.size(); ++i) {
      std::cout << "Task " << i << ":\n";
      for (size_t j = 0; j < kMaxChunks; ++j) {
        auto& rec = records_[i][j];

        std::cout << "  Chunk " << j << ":\n";
        std::cout << "    Start: " << rec.start << " cycles\n";
        std::cout << "    End: " << rec.end << " cycles\n";
        std::cout << "    Duration: " << rec.end - rec.start << " cycles\n";
      }
    }

    // dump frequency
    std::cout << "Frequency: " << get_counter_frequency() << " Hz\n";
    // how to convert cycles to microseconds?
    std::cout << "1 cycle = " << "1e6 / " << get_counter_frequency() << " us\n";
    std::cout << "1 cycle = " << "1e3 / " << get_counter_frequency() << " ms\n";
  }

  void dump_records_metadata() const {
    const auto freq = get_counter_frequency();  // in Hz
    std::cout << "# Frequency=" << freq << " Hz\n";
    std::cout << "# 1 cycle = " << (1e6 / freq) << " us\n";
    std::cout << "# 1 cycle = " << (1e3 / freq) << " ms\n";
  }

  void dump_records_for_python(const Schedule& schedule) const {
    std::cout << "### Python Begin ###" << std::endl;

    const auto freq = get_counter_frequency();  // in Hz
    // print Schedule UID
    std::cout << "Schedule_UID=" << schedule.uid << std::endl;
    std::cout << "Frequency=" << freq << " Hz\n";

    for (size_t i = 0; i < records_.size(); ++i) {
      for (size_t j = 0; j < kMaxChunks; ++j) {
        const auto& rec = records_[i][j];
        // Emit any cell that was actually ticked (start != 0); never-used cells are
        // zero-init and stay skipped. Clamp the duration so a sub-resolution-fast
        // stage (end == start) is still counted instead of vanishing and biasing the
        // surviving durations upward (it used to be dropped by `end > start`).
        if (rec.start != 0) {
          const uint64_t dur = rec.end > rec.start ? rec.end - rec.start : 0;
          std::cout << "Task=" << i << " Chunk="
                    << j
                    // << " Processor=" << processor_type_to_string(rec.processed_by)
                    << " Start=" << rec.start << " End=" << rec.end << " Duration=" << dur << "\n";
        }
      }
    }

    std::cout << "### Python End ###" << std::endl;
  }

  // Print statistics for a specific chunk ID
  void print_chunk_stats(const int chunk_id) const {
    std::vector<double> durations_ms;
    const auto counter_frequency = get_counter_frequency();

    // Collect all durations for the given chunk ID
    for (size_t i = 0; i < records_.size(); ++i) {
      const auto& record = records_[i][chunk_id];
      // Count any ticked cell (start != 0); clamp so a sub-resolution-fast stage
      // (end == start) contributes a 0 ms sample instead of being dropped — see
      // dump_records_for_python above.
      if (record.start != 0) {
        const auto cycles_elapsed = record.end > record.start ? record.end - record.start : 0;
        const auto ms_elapsed = cycles_to_milliseconds(cycles_elapsed, counter_frequency);
        durations_ms.push_back(ms_elapsed);
      }
    }

    if (durations_ms.empty()) {
      std::cout << "No records found for chunk ID: " << chunk_id << std::endl;
      std::cout << "CHUNK=" << chunk_id << " COUNT=0" << std::endl;
      return;
    }

    // Calculate total time
    const double total_time = std::accumulate(durations_ms.begin(), durations_ms.end(), 0.0);

    // Calculate average time
    const double avg_time = total_time / durations_ms.size();

    // Calculate min and max
    const auto [min_it, max_it] = std::minmax_element(durations_ms.begin(), durations_ms.end());
    const double min_time = *min_it;
    const double max_time = *max_it;

    // Calculate median
    std::vector<double> sorted_durations = durations_ms;
    std::ranges::sort(sorted_durations);

    const double median_time = sorted_durations.size() % 2 == 0
                                   ? (sorted_durations[sorted_durations.size() / 2 - 1] +
                                      sorted_durations[sorted_durations.size() / 2]) /
                                         2
                                   : sorted_durations[sorted_durations.size() / 2];

    // Calculate geometric mean
    double geomean = 0.0;
    if (!durations_ms.empty()) {
      double log_sum = 0.0;
      for (const auto& duration : durations_ms) {
        log_sum += std::log(duration > 0 ? duration : 0.000001);  // Avoid log(0)
      }
      geomean = std::exp(log_sum / durations_ms.size());
    }

    // Calculate standard deviation
    double variance = 0.0;
    for (const auto& duration : durations_ms) {
      variance += std::pow(duration - avg_time, 2);
    }
    variance /= durations_ms.size();
    const double std_dev = std::sqrt(variance);

    // Calculate coefficient of variation (CV)
    const double cv = std_dev / avg_time;

    // Calculate 90th, 95th, and 99th percentiles
    const double percentile_90 = sorted_durations[static_cast<int>(0.9 * sorted_durations.size())];
    const double percentile_95 = sorted_durations[static_cast<int>(0.95 * sorted_durations.size())];
    const double percentile_99 = sorted_durations[static_cast<int>(0.99 * sorted_durations.size())];

    // Human-friendly output first
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Statistics for Chunk " << chunk_id << ":\n";
    std::cout << "  Number of records: " << durations_ms.size() << "\n";
    std::cout << "  Total time: " << total_time << " ms\n";
    std::cout << "  Average time: " << avg_time << " ms\n";
    std::cout << "  Geometric mean: " << geomean << " ms\n";
    std::cout << "  Median: " << median_time << " ms\n";
    std::cout << "  Min: " << min_time << " ms\n";
    std::cout << "  Max: " << max_time << " ms\n";
    std::cout << "  Standard deviation: " << std_dev << " ms\n";
    std::cout << "  Coefficient of variation: " << cv << "\n";
    std::cout << "  Percentile 90: " << percentile_90 << " ms\n";
    std::cout << "  Percentile 95: " << percentile_95 << " ms\n";
    std::cout << "  Percentile 99: " << percentile_99 << " ms\n";

    // Machine-parsable format after
    std::cout << "CHUNK=" << chunk_id << "|COUNT=" << durations_ms.size() << "|TOTAL=" << total_time
              << "|AVG=" << avg_time << "|GEOMEAN=" << geomean << "|MEDIAN=" << median_time
              << "|MIN=" << min_time << "|MAX=" << max_time << "|STDDEV=" << std_dev << "|CV=" << cv
              << "|P90=" << percentile_90 << "|P95=" << percentile_95 << "|P99=" << percentile_99
              << std::endl;
  }

  // Print statistics for a chunk given its configuration
  void print_chunk_stats(const ChunkConfig& chunk_config, const int chunk_id) const {
    print_chunk_stats(chunk_id);

    // Additional information about the chunk configuration
    std::cout << "Chunk Configuration:\n";
    std::cout << "  Execution Model: ";
    switch (chunk_config.exec_model) {
      case ExecutionModel::kOMP:
        std::cout << "OMP";
        if (chunk_config.cpu_proc_type.has_value()) {
          std::cout << " (";
          switch (chunk_config.cpu_proc_type.value()) {
            case ProcessorType::kLittleCore:
              std::cout << "Little Core";
              break;
            case ProcessorType::kMediumCore:
              std::cout << "Medium Core";
              break;
            case ProcessorType::kBigCore:
              std::cout << "Big Core";
              break;
            default:
              std::cout << "Unknown Core";
          }
          std::cout << ")";
        }
        break;
      case ExecutionModel::kVulkan:
        std::cout << "Vulkan";
        break;
      case ExecutionModel::kCuda:
        std::cout << "CUDA";
        break;
    }
    std::cout << "\n";

    std::cout << "  Stages: " << chunk_config.start_stage << " to " << chunk_config.end_stage
              << "\n";
  }

  // Print statistics for all chunks in a schedule
  void print_schedule_stats(const Schedule& schedule) const {
    std::cout << "===============================================================\n";
    std::cout << "Schedule Statistics (UID: " << schedule.uid << ")\n";
    std::cout << "===============================================================\n";

    for (size_t i = 0; i < schedule.n_chunks(); ++i) {
      print_chunk_stats(schedule.chunks[i], static_cast<int>(i));
      std::cout << "\n";
    }

    std::cout << "===============================================================\n";
  }

  // write a helper function to convert cycles to microseconds
  static inline double cycles_to_microseconds(const uint64_t cycles, const uint64_t frequency) {
    return static_cast<double>(cycles) * 1e6 / static_cast<double>(frequency);
  }

  static inline double cycles_to_milliseconds(const uint64_t cycles, const uint64_t frequency) {
    return static_cast<double>(cycles) * 1e3 / static_cast<double>(frequency);
  }
};
