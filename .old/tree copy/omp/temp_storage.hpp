#pragma once

#include <vector>

namespace tree::omp {

// ----------------------------------------------------------------------------
// New version
// ----------------------------------------------------------------------------

// Structure to hold all temporary storage needed for radix sort
template <typename T>
struct RadixSortTemp {
  const size_t n_elements;              // Number of elements to sort
  const int n_threads;                  // Number of threads to use
  static constexpr size_t RADIX = 256;  // Radix size (2^8)

  std::vector<T> temp_buffer;                          // Temporary buffer for elements
  std::vector<std::vector<size_t>> thread_histograms;  // Per-thread histograms
  std::vector<std::vector<size_t>> thread_offsets;     // Per-thread offsets
  std::vector<size_t> global_histogram;                // Global histogram
  std::vector<size_t> prefix_sum;                      // Prefix sum array

  // Constructor allocates all temporary storage
  RadixSortTemp(size_t n, int threads)
      : n_elements(n),
        n_threads(threads),
        temp_buffer(n),
        thread_histograms(threads, std::vector<size_t>(RADIX)),
        thread_offsets(threads, std::vector<size_t>(RADIX)),
        global_histogram(RADIX),
        prefix_sum(RADIX) {}

  // Calculate memory usage
  size_t get_memory_usage() const {
    size_t temp_buffer_size = n_elements * sizeof(T);
    size_t thread_histograms_size = n_threads * RADIX * sizeof(size_t);
    size_t thread_offsets_size = n_threads * RADIX * sizeof(size_t);
    size_t global_histogram_size = RADIX * sizeof(size_t);
    size_t prefix_sum_size = RADIX * sizeof(size_t);

    return temp_buffer_size + thread_histograms_size + thread_offsets_size + global_histogram_size +
           prefix_sum_size;
  }
};

// ----------------------------------------------------------------------------
// Old version
// ----------------------------------------------------------------------------

struct bucket {
  int n_elem;
  int index;  // [start : n_elem)
  int start;  // starting point in B array
};

struct TmpStorage {
 public:
  TmpStorage() = default;
  ~TmpStorage() = default;

  // Disallow copy if needed
  TmpStorage(const TmpStorage &) = delete;
  TmpStorage &operator=(const TmpStorage &) = delete;

  // Allow move semantics
  TmpStorage(TmpStorage &&) = default;
  TmpStorage &operator=(TmpStorage &&) = default;

  // getters
  [[nodiscard]] int *global_n_elem() { return h_global_n_elem.data(); }
  [[nodiscard]] int *global_starting_position() { return h_global_starting_position.data(); }
  [[nodiscard]] bucket *buckets() { return h_buckets.data(); }

  [[nodiscard]] bool is_allocated() const { return m_n_buckets > 0; }

  // Allocate memory for storage
  void allocate(int n_buckets, int num_threads) {
    m_n_buckets = n_buckets;
    m_num_threads = num_threads;

    h_global_n_elem.assign(n_buckets, 0);
    h_global_starting_position.assign(n_buckets, 0);

    // Each thread gets n_buckets, so total = n_buckets * num_threads
    h_buckets.assign(n_buckets * num_threads, bucket{});  // Default-initialize buckets
  }

  // // Reset the contents
  // void reset() {
  //   std::fill(global_n_elem.begin(), global_n_elem.end(), 0);
  //   std::fill(global_starting_position.begin(), global_starting_position.end(), 0);

  //   // Reset all buckets
  //   for (auto &b : buckets) {
  //     b = bucket{};  // Re-initialize each bucket to default state
  //   }
  // }

 private:
  int m_n_buckets = 0;
  int m_num_threads = 0;

  // Host memory
  std::vector<int> h_global_n_elem;
  std::vector<int> h_global_starting_position;
  std::vector<bucket> h_buckets;
};

}  // namespace tree::omp
