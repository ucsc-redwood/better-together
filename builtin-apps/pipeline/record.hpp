#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "../conf.hpp"
#include "schedule.hpp"

// Read the ARM virtual count register (CNTVCT_EL0)
static inline uint64_t now_cycles() {
  uint64_t cycles;
  asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
  return cycles;
}

// Read the counter frequency from CNTFRQ_EL0 for conversion if needed
static inline uint32_t get_counter_frequency() {
  uint32_t freq;
  asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
  return freq;
}

struct Record {
  // ProcessorType processed_by;
  uint64_t start;  // Cycle counter when starting
  uint64_t end;    // Cycle counter when finishing
};

template <size_t NumToProcess>
struct Logger {
  // 100x4
  // records_[processing_id][chunk_id]
  std::array<std::array<Record, 4>, NumToProcess> records_;

  // explicit Logger() {
  //   constexpr std::array<Record, 4> default_record_array = {{
  //       {0, 0},
  //       {0, 0},
  //       {0, 0},
  //       {0, 0},
  //   }};
  //   records_.fill(default_record_array);

  //   // records_.fill({{ProcessorType::kUndefined, 0, 0}});

  //   // for (size_t i = 0; i < NumToProcess; ++i) {
  //   //   for (size_t j = 0; j < schedule.chunks.size(); ++j) {
  //   //     records_[i][j].processed_by = schedule.chunks[j].exec_model;
  //   //   }
  //   // }
  // }

  Logger() = default;

  void start_tick(const uint32_t processing_id, const int chunk_id) {
    records_[processing_id][chunk_id].start = now_cycles();
  }

  void end_tick(const uint32_t processing_id, const int chunk_id) {
    records_[processing_id][chunk_id].end = now_cycles();
  }

  void dump_records() const {
    for (size_t i = 0; i < records_.size(); ++i) {
      std::cout << "Task " << i << ":\n";
      for (size_t j = 0; j < 4; ++j) {
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

  void dump_records_for_python() const {
    const auto freq = get_counter_frequency();  // in Hz
    for (size_t i = 0; i < records_.size(); ++i) {
      for (size_t j = 0; j < 4; ++j) {
        const auto& rec = records_[i][j];
        if (rec.end > rec.start) {  // skip empty chunks
          std::cout << "Task=" << i << " Chunk="
                    << j
                    // << " Processor=" << processor_type_to_string(rec.processed_by)
                    << " Start=" << rec.start << " End=" << rec.end
                    << " Duration=" << (rec.end - rec.start) << "\n";
        }
      }
    }

    std::cout << "# Frequency=" << freq << " Hz\n";
    std::cout << "# 1 cycle = " << (1e6 / freq) << " us\n";
    std::cout << "# 1 cycle = " << (1e3 / freq) << " ms\n";
  }

  // // write a function, given a ProcessorType, return the total time, average time, geomean,
  // median,
  // // min, max, standard deviation, cv
  // void print_processor_type_stats(const ProcessorType& processor_type) const {
  //   std::vector<double> durations_ms;
  //   const auto counter_frequency = get_counter_frequency();

  //   // Collect all durations for the given processor type
  //   for (const auto& record_array : records_) {
  //     for (const auto& record : record_array) {
  //       if (record.processed_by == processor_type && record.end > record.start) {
  //         const auto cycles_elapsed = record.end - record.start;
  //         const auto ms_elapsed = cycles_to_milliseconds(cycles_elapsed, counter_frequency);
  //         durations_ms.push_back(ms_elapsed);
  //       }
  //     }
  //   }

  //   if (durations_ms.empty()) {
  //     std::cout << "No records found for processor type: "
  //               << processor_type_to_string(processor_type) << std::endl;
  //     std::cout << "PROCESSOR=" << processor_type_to_string(processor_type) << " COUNT=0"
  //               << std::endl;
  //     return;
  //   }

  //   // Calculate total time
  //   const double total_time = std::accumulate(durations_ms.begin(), durations_ms.end(), 0.0);

  //   // Calculate average time
  //   const double avg_time = total_time / durations_ms.size();

  //   // Calculate min and max
  //   const auto [min_it, max_it] = std::minmax_element(durations_ms.begin(), durations_ms.end());
  //   const double min_time = *min_it;
  //   const double max_time = *max_it;

  //   // Calculate median
  //   std::vector<double> sorted_durations = durations_ms;
  //   std::ranges::sort(sorted_durations);

  //   const double median_time = sorted_durations.size() % 2 == 0
  //                                  ? (sorted_durations[sorted_durations.size() / 2 - 1] +
  //                                     sorted_durations[sorted_durations.size() / 2]) /
  //                                        2
  //                                  : sorted_durations[sorted_durations.size() / 2];

  //   // Calculate geometric mean
  //   double geomean = 0.0;
  //   if (!durations_ms.empty()) {
  //     double log_sum = 0.0;
  //     for (const auto& duration : durations_ms) {
  //       log_sum += std::log(duration > 0 ? duration : 0.000001);  // Avoid log(0)
  //     }
  //     geomean = std::exp(log_sum / durations_ms.size());
  //   }

  //   // Calculate standard deviation
  //   double variance = 0.0;
  //   for (const auto& duration : durations_ms) {
  //     variance += std::pow(duration - avg_time, 2);
  //   }
  //   variance /= durations_ms.size();
  //   const double std_dev = std::sqrt(variance);

  //   // Calculate coefficient of variation (CV)
  //   const double cv = std_dev / avg_time;

  //   // Calculate 90th, 95th, and 99th percentiles
  //   const double percentile_90 = sorted_durations[static_cast<int>(0.9 *
  //   sorted_durations.size())]; const double percentile_95 =
  //   sorted_durations[static_cast<int>(0.95 * sorted_durations.size())]; const double
  //   percentile_99 = sorted_durations[static_cast<int>(0.99 * sorted_durations.size())];

  //   // Human-friendly output first
  //   std::cout << "--------------------------------------------------------\n";
  //   std::cout << "Statistics for " << processor_type_to_string(processor_type) << ":\n";
  //   std::cout << "  Number of records: " << durations_ms.size() << "\n";
  //   std::cout << "  Total time: " << total_time << " ms\n";
  //   std::cout << "  Average time: " << avg_time << " ms\n";
  //   std::cout << "  Geometric mean: " << geomean << " ms\n";
  //   std::cout << "  Median: " << median_time << " ms\n";
  //   std::cout << "  Min: " << min_time << " ms\n";
  //   std::cout << "  Max: " << max_time << " ms\n";
  //   std::cout << "  Standard deviation: " << std_dev << " ms\n";
  //   std::cout << "  Coefficient of variation: " << cv << "\n";
  //   std::cout << "  Percentile 90: " << percentile_90 << " ms\n";
  //   std::cout << "  Percentile 95: " << percentile_95 << " ms\n";
  //   std::cout << "  Percentile 99: " << percentile_99 << " ms\n";

  //   // Machine-parsable format after
  //   std::cout << "PROCESSOR=" << processor_type_to_string(processor_type)
  //             << "|COUNT=" << durations_ms.size() << "|TOTAL=" << total_time << "|AVG=" <<
  //             avg_time
  //             << "|GEOMEAN=" << geomean << "|MEDIAN=" << median_time << "|MIN=" << min_time
  //             << "|MAX=" << max_time << "|STDDEV=" << std_dev << "|CV=" << cv
  //             << "|P90=" << percentile_90 << "|P95=" << percentile_95 << "|P99=" << percentile_99
  //             << std::endl;
  // }

  // write a helper function to convert cycles to microseconds
  static inline double cycles_to_microseconds(const uint64_t cycles, const uint32_t frequency) {
    return static_cast<double>(cycles) * 1e6 / frequency;
  }

  static inline double cycles_to_milliseconds(const uint64_t cycles, const uint32_t frequency) {
    return static_cast<double>(cycles) * 1e3 / frequency;
  }

  // static inline std::string processor_type_to_string(const ProcessorType processor_type) {
  //   switch (processor_type) {
  //     case ProcessorType::kLittleCore:
  //       return "Lut";
  //     case ProcessorType::kMediumCore:
  //       return "Med";
  //     case ProcessorType::kBigCore:
  //       return "Big";
  //     case ProcessorType::kVulkan:
  //       return "Vlk";
  //     default:
  //       return "Unknown";
  //   }
  // }
};
