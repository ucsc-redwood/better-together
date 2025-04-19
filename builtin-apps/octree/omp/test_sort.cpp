#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <vector>

#include "builtin-apps/app.hpp"
#include "sort.hpp"

class RadixSortTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Get the maximum number of threads available
    max_threads_ = omp_get_max_threads();

    // Random seed based on current time
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    rng_.seed(seed);
  }

  // Helper to generate different distributions
  enum class Distribution {
    SEQUENTIAL,    // 0, 1, 2, 3, ...
    REVERSE,       // n-1, n-2, n-3, ...
    RANDOM,        // Uniform random
    FEW_UNIQUE,    // Few unique values (high collision rate)
    MOSTLY_SORTED  // Mostly sorted with few out-of-place elements
  };

  std::vector<uint32_t> GenerateData(size_t size, Distribution dist) {
    std::vector<uint32_t> data(size);
    std::uniform_int_distribution<uint32_t> uniform_dist;
    std::uniform_int_distribution<uint32_t> few_vals;
    std::uniform_int_distribution<size_t> idx_dist;
    size_t num_swaps;

    switch (dist) {
      case Distribution::SEQUENTIAL:
        std::iota(data.begin(), data.end(), 0);
        break;

      case Distribution::REVERSE:
        std::iota(data.rbegin(), data.rend(), 0);
        break;

      case Distribution::RANDOM:
        uniform_dist = std::uniform_int_distribution<uint32_t>(0, UINT32_MAX);
        for (auto& val : data) {
          val = uniform_dist(rng_);
        }
        break;

      case Distribution::FEW_UNIQUE:
        few_vals = std::uniform_int_distribution<uint32_t>(0, 10);
        for (auto& val : data) {
          val = few_vals(rng_);
        }
        break;

      case Distribution::MOSTLY_SORTED:
        std::iota(data.begin(), data.end(), 0);
        // Swap ~5% of elements
        num_swaps = size * 0.05;
        idx_dist = std::uniform_int_distribution<size_t>(0, size - 1);
        for (size_t i = 0; i < num_swaps; ++i) {
          std::swap(data[idx_dist(rng_)], data[idx_dist(rng_)]);
        }
        break;
    }

    return data;
  }

  void VerifySortCorrectness(const std::vector<uint32_t>& input,
                             const std::vector<uint32_t>& sorted,
                             int num_threads) {
    // Check if sorted
    ASSERT_TRUE(std::is_sorted(sorted.begin(), sorted.end()))
        << "Output is not sorted with " << num_threads << " threads";

    // Check if permutation (same elements)
    std::vector<uint32_t> input_sorted = input;
    std::sort(input_sorted.begin(), input_sorted.end());

    ASSERT_EQ(input_sorted.size(), sorted.size()) << "Output size doesn't match input size";

    // Check element-by-element
    for (size_t i = 0; i < sorted.size(); ++i) {
      ASSERT_EQ(sorted[i], input_sorted[i]) << "Element mismatch at index " << i;
    }
  }

  std::mt19937 rng_;
  int max_threads_;
};

TEST_F(RadixSortTest, EmptyVector) {
  std::vector<uint32_t> empty_in;
  std::vector<uint32_t> empty_out;

  // Sort with different thread counts
  for (int threads = 1; threads <= std::min(4, max_threads_); ++threads) {
    dispatch_radix_sort(empty_in, empty_out, threads);
    EXPECT_TRUE(empty_out.empty()) << "Empty vector should remain empty";
  }
}

TEST_F(RadixSortTest, SingleElement) {
  std::vector<uint32_t> in = {42};
  std::vector<uint32_t> out(1);

  // Sort with different thread counts
  for (int threads = 1; threads <= std::min(4, max_threads_); ++threads) {
    dispatch_radix_sort(in, out, threads);
    EXPECT_EQ(out.size(), 1);
    EXPECT_EQ(out[0], 42);
  }
}

TEST_F(RadixSortTest, SmallSizes) {
  const std::vector<size_t> sizes = {2, 3, 10, 15};

  for (size_t size : sizes) {
    for (int threads = 1; threads <= std::min(4, max_threads_); ++threads) {
      auto input = GenerateData(size, Distribution::RANDOM);
      std::vector<uint32_t> output(size);

      dispatch_radix_sort(input, output, threads);

      VerifySortCorrectness(input, output, threads);
    }
  }
}

TEST_F(RadixSortTest, MediumSizes) {
  const std::vector<size_t> sizes = {1024, 4096, 8192};

  for (size_t size : sizes) {
    for (int threads = 1; threads <= std::min(8, max_threads_); ++threads) {
      auto input = GenerateData(size, Distribution::RANDOM);
      std::vector<uint32_t> output(size);

      dispatch_radix_sort(input, output, threads);

      VerifySortCorrectness(input, output, threads);
    }
  }
}

TEST_F(RadixSortTest, LargeSizes) {
  // Only test large sizes if not running in a constrained environment
  const std::vector<size_t> sizes = {65536, 262144};

  for (size_t size : sizes) {
    for (int threads : {1, 2, 4, 8, 16}) {
      if (threads > max_threads_) continue;

      auto input = GenerateData(size, Distribution::RANDOM);
      std::vector<uint32_t> output(size);

      dispatch_radix_sort(input, output, threads);

      VerifySortCorrectness(input, output, threads);
    }
  }
}

TEST_F(RadixSortTest, DifferentDistributions) {
  const size_t size = 10000;
  const std::vector<Distribution> distributions = {Distribution::SEQUENTIAL,
                                                   Distribution::REVERSE,
                                                   Distribution::RANDOM,
                                                   Distribution::FEW_UNIQUE,
                                                   Distribution::MOSTLY_SORTED};

  for (auto dist : distributions) {
    for (int threads : {1, 4, 8}) {
      if (threads > max_threads_) continue;

      auto input = GenerateData(size, dist);
      std::vector<uint32_t> output(size);

      dispatch_radix_sort(input, output, threads);

      VerifySortCorrectness(input, output, threads);
    }
  }
}

TEST_F(RadixSortTest, LittleCoresSorting) {
  // Skip if no little cores available
  if (g_lit_cores.empty()) {
    GTEST_SKIP() << "No little cores available on this device";
    return;
  }

  const size_t size = 10000;

  std::cout << "Testing with little cores (" << g_lit_cores.size() << " cores)" << std::endl;

  // Test with different thread counts up to the number of available little cores
  for (size_t threads = 1; threads <= g_lit_cores.size(); ++threads) {
    auto input = GenerateData(size, Distribution::RANDOM);
    std::vector<uint32_t> output(size);

    // Use only the first 'threads' little cores
    std::vector<int> cores_to_use(g_lit_cores.begin(), g_lit_cores.begin() + threads);

    std::cout << "  Using " << threads << " little cores: ";
    for (auto core : cores_to_use) {
      std::cout << core << " ";
    }
    std::cout << std::endl;

    dispatch_radix_sort(input, output, threads, cores_to_use);

    VerifySortCorrectness(input, output, threads);
  }
}

TEST_F(RadixSortTest, MediumCoresSorting) {
  // Skip if no medium cores available
  if (g_med_cores.empty()) {
    GTEST_SKIP() << "No medium cores available on this device";
    return;
  }

  const size_t size = 10000;

  std::cout << "Testing with medium cores (" << g_med_cores.size() << " cores)" << std::endl;

  // Test with different thread counts up to the number of available medium cores
  for (size_t threads = 1; threads <= g_med_cores.size(); ++threads) {
    auto input = GenerateData(size, Distribution::RANDOM);
    std::vector<uint32_t> output(size);

    // Use only the first 'threads' medium cores
    std::vector<int> cores_to_use(g_med_cores.begin(), g_med_cores.begin() + threads);

    std::cout << "  Using " << threads << " medium cores: ";
    for (auto core : cores_to_use) {
      std::cout << core << " ";
    }
    std::cout << std::endl;

    dispatch_radix_sort(input, output, threads, cores_to_use);

    VerifySortCorrectness(input, output, threads);
  }
}

TEST_F(RadixSortTest, BigCoresSorting) {
  // Skip if no big cores available
  if (g_big_cores.empty()) {
    GTEST_SKIP() << "No big cores available on this device";
    return;
  }

  const size_t size = 10000;

  std::cout << "Testing with big cores (" << g_big_cores.size() << " cores)" << std::endl;

  // Test with different thread counts up to the number of available big cores
  for (size_t threads = 1; threads <= g_big_cores.size(); ++threads) {
    auto input = GenerateData(size, Distribution::RANDOM);
    std::vector<uint32_t> output(size);

    // Use only the first 'threads' big cores
    std::vector<int> cores_to_use(g_big_cores.begin(), g_big_cores.begin() + threads);

    std::cout << "  Using " << threads << " big cores: ";
    for (auto core : cores_to_use) {
      std::cout << core << " ";
    }
    std::cout << std::endl;

    dispatch_radix_sort(input, output, threads, cores_to_use);

    VerifySortCorrectness(input, output, threads);
  }
}

TEST_F(RadixSortTest, CoreTypePerformanceComparison) {
// Skip in debug mode or if any core type is not available
#ifndef NDEBUG
  GTEST_SKIP() << "Skipping performance test in debug mode";
#endif

  bool skip_test = false;
  if (g_lit_cores.empty()) {
    std::cout << "Little cores not available - skipping some comparisons" << std::endl;
    skip_test = true;
  }
  if (g_med_cores.empty()) {
    std::cout << "Medium cores not available - skipping some comparisons" << std::endl;
    skip_test = true;
  }
  if (g_big_cores.empty()) {
    std::cout << "Big cores not available - skipping some comparisons" << std::endl;
    skip_test = true;
  }

  if (skip_test && g_lit_cores.empty() && g_med_cores.empty() && g_big_cores.empty()) {
    GTEST_SKIP() << "No core types available for testing";
    return;
  }

  const size_t size = 100000;  // 100K elements for timing
  auto input = GenerateData(size, Distribution::RANDOM);

  std::cout << "Core type performance comparison with " << size << " elements:" << std::endl;

  // Test each core type with a single thread
  std::map<std::string, double> times;

  if (!g_lit_cores.empty()) {
    std::vector<uint32_t> output(size);
    std::vector<uint32_t> input_copy = input;
    std::vector<int> cores = {g_lit_cores[0]};

    auto start = std::chrono::high_resolution_clock::now();
    dispatch_radix_sort(input_copy, output, 1, cores);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    times["Little"] = elapsed.count();
    std::cout << "  Little core (ID " << cores[0] << "): " << elapsed.count() << " ms" << std::endl;

    VerifySortCorrectness(input, output, 1);
  }

  if (!g_med_cores.empty()) {
    std::vector<uint32_t> output(size);
    std::vector<uint32_t> input_copy = input;
    std::vector<int> cores = {g_med_cores[0]};

    auto start = std::chrono::high_resolution_clock::now();
    dispatch_radix_sort(input_copy, output, 1, cores);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    times["Medium"] = elapsed.count();
    std::cout << "  Medium core (ID " << cores[0] << "): " << elapsed.count() << " ms" << std::endl;

    VerifySortCorrectness(input, output, 1);
  }

  if (!g_big_cores.empty()) {
    std::vector<uint32_t> output(size);
    std::vector<uint32_t> input_copy = input;
    std::vector<int> cores = {g_big_cores[0]};

    auto start = std::chrono::high_resolution_clock::now();
    dispatch_radix_sort(input_copy, output, 1, cores);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    times["Big"] = elapsed.count();
    std::cout << "  Big core (ID " << cores[0] << "): " << elapsed.count() << " ms" << std::endl;

    VerifySortCorrectness(input, output, 1);
  }

  // Compare performance across core types
  if (times.size() > 1) {
    std::cout << "Relative performance:" << std::endl;

    // Find the slowest core type as baseline
    auto slowest = std::max_element(times.begin(), times.end(), [](const auto& a, const auto& b) {
      return a.second < b.second;
    });

    for (const auto& [type, time] : times) {
      if (time > 0) {
        std::cout << "  " << type << " vs " << slowest->first << ": " << slowest->second / time
                  << "x faster" << std::endl;
      }
    }
  }
}

// Test scaling of speedup within each core type
TEST_F(RadixSortTest, LittleCoresScaling) {
// Skip in debug mode or if no little cores available
#ifndef NDEBUG
  GTEST_SKIP() << "Skipping performance test in debug mode";
#endif

  if (g_lit_cores.empty()) {
    GTEST_SKIP() << "No little cores available on this device";
    return;
  }

  const size_t size = 100000;  // 100K elements for timing
  auto input = GenerateData(size, Distribution::RANDOM);

  std::vector<double> times;

  std::cout << "Little cores scaling with " << size << " elements:" << std::endl;

  // First test with single core as baseline
  {
    std::vector<uint32_t> output(size);
    std::vector<uint32_t> input_copy = input;
    std::vector<int> cores = {g_lit_cores[0]};

    auto start = std::chrono::high_resolution_clock::now();
    dispatch_radix_sort(input_copy, output, 1, cores);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    times.push_back(elapsed.count());
    std::cout << "  1 little core (ID " << cores[0] << "): " << elapsed.count() << " ms"
              << std::endl;

    VerifySortCorrectness(input, output, 1);
  }

  // Now test with increasing number of cores
  for (size_t num_cores = 2; num_cores <= g_lit_cores.size(); ++num_cores) {
    std::vector<uint32_t> output(size);
    std::vector<uint32_t> input_copy = input;
    std::vector<int> cores(g_lit_cores.begin(), g_lit_cores.begin() + num_cores);

    auto start = std::chrono::high_resolution_clock::now();
    dispatch_radix_sort(input_copy, output, num_cores, cores);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    times.push_back(elapsed.count());
    std::cout << "  " << num_cores << " little cores: " << elapsed.count() << " ms" << std::endl;

    VerifySortCorrectness(input, output, num_cores);
  }

  // Report speedup
  if (!times.empty() && times[0] > 0) {
    std::cout << "Speedup relative to single little core:" << std::endl;
    for (size_t i = 0; i < times.size(); ++i) {
      std::cout << "  " << (i + 1) << " little cores: " << times[0] / times[i] << "x" << std::endl;
    }
  }
}

TEST_F(RadixSortTest, MediumCoresScaling) {
// Skip in debug mode or if no medium cores available
#ifndef NDEBUG
  GTEST_SKIP() << "Skipping performance test in debug mode";
#endif

  if (g_med_cores.empty()) {
    GTEST_SKIP() << "No medium cores available on this device";
    return;
  }

  const size_t size = 100000;  // 100K elements for timing
  auto input = GenerateData(size, Distribution::RANDOM);

  std::vector<double> times;

  std::cout << "Medium cores scaling with " << size << " elements:" << std::endl;

  // First test with single core as baseline
  {
    std::vector<uint32_t> output(size);
    std::vector<uint32_t> input_copy = input;
    std::vector<int> cores = {g_med_cores[0]};

    auto start = std::chrono::high_resolution_clock::now();
    dispatch_radix_sort(input_copy, output, 1, cores);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    times.push_back(elapsed.count());
    std::cout << "  1 medium core (ID " << cores[0] << "): " << elapsed.count() << " ms"
              << std::endl;

    VerifySortCorrectness(input, output, 1);
  }

  // Now test with increasing number of cores
  for (size_t num_cores = 2; num_cores <= g_med_cores.size(); ++num_cores) {
    std::vector<uint32_t> output(size);
    std::vector<uint32_t> input_copy = input;
    std::vector<int> cores(g_med_cores.begin(), g_med_cores.begin() + num_cores);

    auto start = std::chrono::high_resolution_clock::now();
    dispatch_radix_sort(input_copy, output, num_cores, cores);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    times.push_back(elapsed.count());
    std::cout << "  " << num_cores << " medium cores: " << elapsed.count() << " ms" << std::endl;

    VerifySortCorrectness(input, output, num_cores);
  }

  // Report speedup
  if (!times.empty() && times[0] > 0) {
    std::cout << "Speedup relative to single medium core:" << std::endl;
    for (size_t i = 0; i < times.size(); ++i) {
      std::cout << "  " << (i + 1) << " medium cores: " << times[0] / times[i] << "x" << std::endl;
    }
  }
}

TEST_F(RadixSortTest, BigCoresScaling) {
// Skip in debug mode or if no big cores available
#ifndef NDEBUG
  GTEST_SKIP() << "Skipping performance test in debug mode";
#endif

  if (g_big_cores.empty()) {
    GTEST_SKIP() << "No big cores available on this device";
    return;
  }

  const size_t size = 100000;  // 100K elements for timing
  auto input = GenerateData(size, Distribution::RANDOM);

  std::vector<double> times;

  std::cout << "Big cores scaling with " << size << " elements:" << std::endl;

  // First test with single core as baseline
  {
    std::vector<uint32_t> output(size);
    std::vector<uint32_t> input_copy = input;
    std::vector<int> cores = {g_big_cores[0]};

    auto start = std::chrono::high_resolution_clock::now();
    dispatch_radix_sort(input_copy, output, 1, cores);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    times.push_back(elapsed.count());
    std::cout << "  1 big core (ID " << cores[0] << "): " << elapsed.count() << " ms" << std::endl;

    VerifySortCorrectness(input, output, 1);
  }

  // Now test with increasing number of cores
  for (size_t num_cores = 2; num_cores <= g_big_cores.size(); ++num_cores) {
    std::vector<uint32_t> output(size);
    std::vector<uint32_t> input_copy = input;
    std::vector<int> cores(g_big_cores.begin(), g_big_cores.begin() + num_cores);

    auto start = std::chrono::high_resolution_clock::now();
    dispatch_radix_sort(input_copy, output, num_cores, cores);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    times.push_back(elapsed.count());
    std::cout << "  " << num_cores << " big cores: " << elapsed.count() << " ms" << std::endl;

    VerifySortCorrectness(input, output, num_cores);
  }

  // Report speedup
  if (!times.empty() && times[0] > 0) {
    std::cout << "Speedup relative to single big core:" << std::endl;
    for (size_t i = 0; i < times.size(); ++i) {
      std::cout << "  " << (i + 1) << " big cores: " << times[0] / times[i] << "x" << std::endl;
    }
  }
}

TEST_F(RadixSortTest, PerformanceScaling) {
// Skip in debug mode or CI environments
#ifndef NDEBUG
  GTEST_SKIP() << "Skipping performance test in debug mode";
#endif

  const size_t size = 1000000;  // 1M elements
  auto input = GenerateData(size, Distribution::RANDOM);

  std::vector<int> thread_counts = {1, 2, 4, 8, 16};
  std::vector<double> times;

  std::cout << "Performance scaling test with " << size << " elements:" << std::endl;

  for (int threads : thread_counts) {
    if (threads > max_threads_) continue;

    std::vector<uint32_t> output(size);
    std::vector<uint32_t> input_copy = input;  // Create a copy to preserve original data

    auto start = std::chrono::high_resolution_clock::now();

    dispatch_radix_sort(input_copy, output, threads);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "  " << threads << " threads: " << elapsed.count() << " ms" << std::endl;
    times.push_back(elapsed.count());

    VerifySortCorrectness(input, output, threads);
  }

  // Report speedup
  if (!times.empty() && times[0] > 0) {
    std::cout << "Speedup relative to single thread:" << std::endl;
    for (size_t i = 0; i < times.size(); ++i) {
      std::cout << "  " << thread_counts[i] << " threads: " << times[0] / times[i] << "x"
                << std::endl;
    }
  }
}

TEST_F(RadixSortTest, CompareWithStdSort) {
// Skip in debug mode or CI environments
#ifndef NDEBUG
  GTEST_SKIP() << "Skipping performance comparison test in debug mode";
#endif

  const std::vector<size_t> sizes = {10000, 100000, 1000000};

  std::cout << "Performance comparison with std::sort:" << std::endl;

  for (size_t size : sizes) {
    auto input = GenerateData(size, Distribution::RANDOM);
    std::vector<uint32_t> output(size);
    std::vector<uint32_t> std_sorted = input;

    // Measure std::sort time
    auto start_std = std::chrono::high_resolution_clock::now();
    std::sort(std_sorted.begin(), std_sorted.end());
    auto end_std = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_std = end_std - start_std;

    // Measure parallel radix sort with max threads
    int threads = std::min(16, max_threads_);
    auto start_radix = std::chrono::high_resolution_clock::now();
    dispatch_radix_sort(input, output, threads);
    auto end_radix = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_radix = end_radix - start_radix;

    std::cout << "Size " << size << ":" << std::endl;
    std::cout << "  std::sort: " << elapsed_std.count() << " ms" << std::endl;
    std::cout << "  radix_sort (" << threads << " threads): " << elapsed_radix.count() << " ms"
              << std::endl;
    std::cout << "  Speedup: " << elapsed_std.count() / elapsed_radix.count() << "x" << std::endl;

    // Verify correctness
    ASSERT_EQ(std_sorted, output) << "Radix sort result differs from std::sort";
  }
}

TEST_F(RadixSortTest, EdgeValues) {
  std::vector<uint32_t> edge_cases = {
      0,               // Zero
      1,               // One
      UINT32_MAX,      // Maximum value
      UINT32_MAX - 1,  // Almost maximum
      UINT32_MAX / 2,  // Middle value
      0x01010101,      // Repeated bytes
      0x80808080,      // High bits set
      0xFFFF0000,      // Upper half all ones
      0x0000FFFF       // Lower half all ones
  };

  // Test with different thread counts
  for (int threads : {1, 2, 4}) {
    if (threads > max_threads_) continue;

    // Create shuffled input with edge cases
    std::vector<uint32_t> input = edge_cases;
    std::shuffle(input.begin(), input.end(), rng_);

    std::vector<uint32_t> output(input.size());
    dispatch_radix_sort(input, output, threads);

    VerifySortCorrectness(input, output, threads);
  }
}

// Additional tests for heterogeneous core combinations
TEST_F(RadixSortTest, HeterogeneousCoreCombinations) {
  // Skip if not enough cores of different types are available
  if (g_lit_cores.empty() && g_med_cores.empty() && g_big_cores.empty()) {
    GTEST_SKIP() << "No cores available for testing";
    return;
  }

  const size_t size = 10000;
  auto input = GenerateData(size, Distribution::RANDOM);
  std::vector<uint32_t> output(size);

  // Combine cores of different types
  std::vector<int> mixed_cores;

  // Add cores from each available type
  if (!g_lit_cores.empty()) {
    mixed_cores.push_back(g_lit_cores[0]);
  }
  if (!g_med_cores.empty()) {
    mixed_cores.push_back(g_med_cores[0]);
  }
  if (!g_big_cores.empty()) {
    mixed_cores.push_back(g_big_cores[0]);
  }

  if (mixed_cores.size() < 2) {
    GTEST_SKIP() << "Not enough different core types for heterogeneous testing";
    return;
  }

  std::cout << "Testing with heterogeneous core combination: ";
  for (auto core : mixed_cores) {
    std::cout << core << " ";
  }
  std::cout << std::endl;

  // Test with the mixed core set
  dispatch_radix_sort(input, output, mixed_cores.size(), mixed_cores);

  VerifySortCorrectness(input, output, mixed_cores.size());
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
