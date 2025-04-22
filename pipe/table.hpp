#pragma once

#include <spdlog/spdlog.h>

#include <array>

// ----------------------------------------------------------------------------
// Tables
// ----------------------------------------------------------------------------

// create a 2D table, kNumStages stage times 5 type of cores, initialize it with 0
// access by bm_table[stage][core_type] = value
template <size_t kNumStages>
struct BmTable {
  static constexpr int kLitIdx = 0;
  static constexpr int kMedIdx = 1;
  static constexpr int kBigIdx = 2;
  static constexpr int kVukIdx = 3;
  static constexpr int kCudIdx = 4;

  std::array<std::array<double, 5>, kNumStages> bm_norm_table;
  std::array<std::array<double, 5>, kNumStages> bm_full_table;

  explicit BmTable() {
    for (size_t stage = 0; stage < kNumStages; stage++) {
      for (size_t processor = 0; processor < 5; processor++) {
        bm_norm_table[stage][processor] = 0.0;
        bm_full_table[stage][processor] = 0.0;
      }
    }
  }

  void update_normal_table(const int stage, const int processor_idx, const double value) {
    bm_norm_table[stage][processor_idx] = value;
  }

  void update_full_table(const int stage, const int processor_idx, const double value) {
    bm_full_table[stage][processor_idx] = value;
  }

  void dump_tables_for_python(int start_stage, int end_stage) {
    // First dump a marker that can be easily grep'ed
    fmt::print("\n### PYTHON_DATA_START ###\n");

    // Dump normal benchmark data in CSV format
    fmt::print("# NORMAL_BENCHMARK_DATA\n");
    fmt::print("stage,little,medium,big,vulkan,cuda\n");
    for (int stage = start_stage; stage <= end_stage; stage++) {
      fmt::print("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n",
                 stage,
                 bm_norm_table[stage - 1][kLitIdx],
                 bm_norm_table[stage - 1][kMedIdx],
                 bm_norm_table[stage - 1][kBigIdx],
                 bm_norm_table[stage - 1][kVukIdx],
                 bm_norm_table[stage - 1][kCudIdx]);
    }

    // Dump fully benchmark data in CSV format
    fmt::print("# FULLY_BENCHMARK_DATA\n");
    fmt::print("stage,little,medium,big,vulkan,cuda\n");
    for (int stage = start_stage; stage <= end_stage; stage++) {
      fmt::print("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n",
                 stage,
                 bm_full_table[stage - 1][kLitIdx],
                 bm_full_table[stage - 1][kMedIdx],
                 bm_full_table[stage - 1][kBigIdx],
                 bm_full_table[stage - 1][kVukIdx],
                 bm_full_table[stage - 1][kCudIdx]);
    }

    fmt::print("### PYTHON_DATA_END ###\n");
    std::fflush(stdout);
  }

  void print_normal_benchmark_table(int start_stage, int end_stage) {
    // Print the normal benchmark table with higher precision
    fmt::print("\nNormal Benchmark Results Table (ms per task):\n");
    fmt::print("Stage | Little Core | Medium Core | Big Core | Vulkan | CUDA\n");
    fmt::print("------|------------|-------------|----------|--------|-------\n");
    for (int stage = start_stage; stage <= end_stage; stage++) {
      fmt::print("{:5} | {:11.4f} | {:11.4f} | {:8.4f} | {:6.4f} | {:6.4f}\n",
                 stage,
                 bm_norm_table[stage - 1][kLitIdx],
                 bm_norm_table[stage - 1][kMedIdx],
                 bm_norm_table[stage - 1][kBigIdx],
                 bm_norm_table[stage - 1][kVukIdx],
                 bm_norm_table[stage - 1][kCudIdx]);
    }

    // Calculate sums for normal benchmark
    double lit_norm_sum = 0;
    double med_norm_sum = 0;
    double big_norm_sum = 0;
    double vul_norm_sum = 0;
    double cud_norm_sum = 0;

    for (int stage = start_stage; stage <= end_stage; stage++) {
      lit_norm_sum += bm_norm_table[stage - 1][kLitIdx];
      med_norm_sum += bm_norm_table[stage - 1][kMedIdx];
      big_norm_sum += bm_norm_table[stage - 1][kBigIdx];
      vul_norm_sum += bm_norm_table[stage - 1][kVukIdx];
      cud_norm_sum += bm_norm_table[stage - 1][kCudIdx];
    }

    // Print sum for normal benchmark
    fmt::print("\nNormal Benchmark - Sum of stages {}-{}:\n", start_stage, end_stage);
    fmt::print("Little Core: {:.4f} ms\n", lit_norm_sum);
    fmt::print("Medium Core: {:.4f} ms\n", med_norm_sum);
    fmt::print("Big Core: {:.4f} ms\n", big_norm_sum);
    fmt::print("Vulkan: {:.4f} ms\n", vul_norm_sum);
    fmt::print("CUDA: {:.4f} ms\n", cud_norm_sum);

    // Print the fully benchmark table with higher precision
    fmt::print("\nFully Benchmark Results Table (ms per task):\n");
    fmt::print("Stage | Little Core | Medium Core | Big Core | Vulkan | CUDA\n");
    fmt::print("------|-------------|-------------|----------|--------|-------\n");
    for (int stage = start_stage; stage <= end_stage; stage++) {
      fmt::print("{:5} | {:11.4f} | {:11.4f} | {:8.4f} | {:6.4f} | {:6.4f}\n",
                 stage,
                 bm_full_table[stage - 1][kLitIdx],
                 bm_full_table[stage - 1][kMedIdx],
                 bm_full_table[stage - 1][kBigIdx],
                 bm_full_table[stage - 1][kVukIdx],
                 bm_full_table[stage - 1][kCudIdx]);
    }

    // Calculate sums for fully benchmark
    double lit_full_sum = 0;
    double med_full_sum = 0;
    double big_full_sum = 0;
    double vul_full_sum = 0;
    double cud_full_sum = 0;

    for (int stage = start_stage; stage <= end_stage; stage++) {
      lit_full_sum += bm_full_table[stage - 1][kLitIdx];
      med_full_sum += bm_full_table[stage - 1][kMedIdx];
      big_full_sum += bm_full_table[stage - 1][kBigIdx];
      vul_full_sum += bm_full_table[stage - 1][kVukIdx];
      cud_full_sum += bm_full_table[stage - 1][kCudIdx];
    }

    // Print sum for fully benchmark
    fmt::print("\nFully Benchmark - Sum of stages {}-{}:\n", start_stage, end_stage);
    fmt::print("Little Core: {:.4f} ms\n", lit_full_sum);
    fmt::print("Medium Core: {:.4f} ms\n", med_full_sum);
    fmt::print("Big Core: {:.4f} ms\n", big_full_sum);
    fmt::print("Vulkan: {:.4f} ms\n", vul_full_sum);
    fmt::print("CUDA: {:.4f} ms\n", cud_full_sum);

    // Compare normal vs fully
    fmt::print("\nPerformance Comparison (Fully vs Normal):\n");
    fmt::print("Processor  | Normal (ms) | Fully (ms) | Ratio\n");
    fmt::print("-----------|-------------|------------|-------\n");

    auto print_comparison = [](const std::string& name, double normal, double fully) {
      if (normal > 0) {
        fmt::print("{:<11}| {:11.2f} | {:10.2f} | ", name, normal, fully);
        if (fully > 0) {
          fmt::print("{:5.2f}x\n", fully / normal);
        } else {
          fmt::print("N/A\n");
        }
      }
    };

    print_comparison("Little Core", lit_norm_sum, lit_full_sum);
    print_comparison("Medium Core", med_norm_sum, med_full_sum);
    print_comparison("Big Core", big_norm_sum, big_full_sum);
    print_comparison("Vulkan", vul_norm_sum, vul_full_sum);
    print_comparison("CUDA", cud_norm_sum, cud_full_sum);
  }
};
