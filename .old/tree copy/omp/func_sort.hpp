#pragma once

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory_resource>
#include <vector>

#include "temp_storage.hpp"

namespace tree::omp {

// ----------------------------------------------------------------------------
// Merge sort version (new)
// ----------------------------------------------------------------------------

// Helper function to merge sorted segments
inline void merge_segments(const std::pmr::vector<uint32_t> &input,
                           std::pmr::vector<uint32_t> &output,
                           size_t start1,
                           size_t end1,
                           size_t start2,
                           size_t end2,
                           size_t output_start) {
  size_t i = start1, j = start2, k = output_start;
  while (i < end1 && j < end2) {
    if (input[i] <= input[j])
      output[k++] = input[i++];
    else
      output[k++] = input[j++];
  }
  while (i < end1) output[k++] = input[i++];
  while (j < end2) output[k++] = input[j++];
}

inline void parallel_sort(std::pmr::vector<uint32_t> &buffer_input,
                          std::pmr::vector<uint32_t> &buffer_output,
                          int thread_id,
                          int num_threads) {
  const size_t N = buffer_input.size();
  const size_t segment_size = (N + num_threads - 1) / num_threads;

  size_t start = thread_id * segment_size;
  size_t end = std::min(start + segment_size, N);

  // Step 1: Each thread sorts its segment
  if (start < end) {
    std::sort(buffer_input.begin() + start, buffer_input.begin() + end);
  }

#pragma omp barrier

  // Step 2: Iterative merging performed by a single thread after sorting
  if (thread_id == 0) {
    std::pmr::vector<uint32_t> *src = &buffer_input;
    std::pmr::vector<uint32_t> *dst = &buffer_output;

    for (size_t width = segment_size; width < N; width *= 2) {
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < N; i += 2 * width) {
        size_t start1 = i;
        size_t end1 = std::min(start1 + width, N);
        size_t start2 = end1;
        size_t end2 = std::min(start2 + width, N);

        merge_segments(*src, *dst, start1, end1, start2, end2, start1);
      }
      std::swap(src, dst);
    }

    if (src != &buffer_output) {
      buffer_output = *src;
    }
  }

#pragma omp barrier
}

// ----------------------------------------------------------------------------
// New version
// ----------------------------------------------------------------------------

// Parallel Radix Sort for unsigned integer types
template <typename T>
void parallel_radix_sort(const std::pmr::vector<T> &input,
                         std::pmr::vector<T> &output,
                         RadixSortTemp<T> &temp) {
  static_assert(std::is_unsigned<T>::value, "Radix sort requires unsigned integer type");

  const size_t n = input.size();
  std::copy(input.begin(), input.end(), output.begin());

  constexpr size_t RADIX_BITS = 8;
  constexpr size_t RADIX = 1 << RADIX_BITS;  // 256
  constexpr size_t MASK = RADIX - 1;         // 0xFF
  constexpr size_t NUM_PASSES = sizeof(T) * 8 / RADIX_BITS;

  // Get thread ID
  int thread_id = omp_get_thread_num();
  int num_threads = omp_get_num_threads();

  // For each byte
  for (size_t pass = 0; pass < NUM_PASSES; pass++) {
    const size_t shift = pass * RADIX_BITS;

    // Reset histograms for this thread
    std::fill(
        temp.thread_histograms[thread_id].begin(), temp.thread_histograms[thread_id].end(), 0);

// Barrier to ensure all threads have reset their histograms
#pragma omp barrier

// Build local histogram
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      size_t bin = (output[i] >> shift) & MASK;
      temp.thread_histograms[thread_id][bin]++;
    }

// Barrier to ensure all histograms are complete
#pragma omp barrier

// Thread 0 combines histograms and computes prefix sums
#pragma omp single
    {
      // Reset global histogram
      std::fill(temp.global_histogram.begin(), temp.global_histogram.end(), 0);

      // Combine histograms
      for (int t = 0; t < num_threads; t++) {
        for (size_t i = 0; i < RADIX; i++) {
          temp.global_histogram[i] += temp.thread_histograms[t][i];
        }
      }

      // Compute prefix sum
      temp.prefix_sum[0] = 0;
      for (size_t i = 1; i < RADIX; i++) {
        temp.prefix_sum[i] = temp.prefix_sum[i - 1] + temp.global_histogram[i - 1];
      }

      // Compute per-thread offsets
      for (size_t bin = 0; bin < RADIX; bin++) {
        size_t offset = temp.prefix_sum[bin];
        for (int t = 0; t < num_threads; t++) {
          temp.thread_offsets[t][bin] = offset;
          offset += temp.thread_histograms[t][bin];
        }
      }
    }

// Barrier to ensure prefix sums and offsets are ready
#pragma omp barrier

// Distribute elements to correct positions
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      size_t bin = (output[i] >> shift) & MASK;
      size_t pos = temp.thread_offsets[thread_id][bin]++;
      temp.temp_buffer[pos] = output[i];
    }

// Barrier to ensure all elements are distributed
#pragma omp barrier

// Copy back to output buffer
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      output[i] = temp.temp_buffer[i];
    }

// Barrier before next pass
#pragma omp barrier
  }
}

// ----------------------------------------------------------------------------
// Old version
// ----------------------------------------------------------------------------

// struct bucket {
//   int n_elem;
//   int index;  // [start : n_elem)
//   int start;  // starting point in B array
// };

[[deprecated("Use bucket_sort_v2 instead")]] inline int cmpfunc(const void *a, const void *b) {
  return (*(uint32_t *)a - *(uint32_t *)b);
}

[[deprecated("Use bucket_sort_v2 instead")]] inline void bucket_sort(

    uint32_t *A,
    uint32_t *B,  // for temporary storage

    int *global_n_elem,
    int *global_starting_position,

    struct bucket *buckets,

    const int dim,
    const int n_buckets,
    const int num_threads

) {
  //   uint32_t limit = 100000;

  // I got this number from running my program
  uint32_t limit = 1'073'741'600;
  uint32_t w = limit / n_buckets;

  int j, k;
  int local_index;        // [0 : n_buckets)
  int real_bucket_index;  // [0 : n_buckets * num_threads)
  int my_id = omp_get_thread_num();
  // int workload = dim / num_threads;
  // int prevoius_index;

#pragma omp for private(local_index)
  for (int i = 0; i < dim; i++) {
    local_index = A[i] / w;
    if (local_index > n_buckets - 1) local_index = n_buckets - 1;
    real_bucket_index = local_index + my_id * n_buckets;
    buckets[real_bucket_index].n_elem++;
  }

  int local_sum = 0;
  for (j = my_id; j < n_buckets * num_threads; j = j + num_threads) {
    local_sum += buckets[j].n_elem;
  }
  global_n_elem[my_id] = local_sum;

#pragma omp barrier

#pragma omp master
  {
    for (j = 1; j < n_buckets; j++) {
      global_starting_position[j] = global_starting_position[j - 1] + global_n_elem[j - 1];
      buckets[j].start = buckets[j - 1].start + global_n_elem[j - 1];
      buckets[j].index = buckets[j - 1].index + global_n_elem[j - 1];
    }
  }

#pragma omp barrier
  for (j = my_id + n_buckets; j < n_buckets * num_threads; j = j + num_threads) {
    int prevoius_index = j - n_buckets;
    buckets[j].start = buckets[prevoius_index].start + buckets[prevoius_index].n_elem;
    buckets[j].index = buckets[prevoius_index].index + buckets[prevoius_index].n_elem;
  }
#pragma omp barrier

  int b_index;

#pragma omp for private(b_index)
  for (int i = 0; i < dim; i++) {
    j = A[i] / w;
    if (j > n_buckets - 1) j = n_buckets - 1;
    k = j + my_id * n_buckets;
    b_index = buckets[k].index++;
    B[b_index] = A[i];
  }

#pragma omp for
  for (int i = 0; i < n_buckets; i++)
    qsort(B + global_starting_position[i], global_n_elem[i], sizeof(uint32_t), cmpfunc);

  // // I am not going to copy this back. (02/04/2025)
  // #pragma omp master
  // memcpy(A, B, sizeof(uint32_t) * dim);
}

// ----------------------------------------------------------------------------
// Old working version
// ----------------------------------------------------------------------------

// struct TempStorage {
//   explicit TempStorage(const int n_buckets, const int num_threads) {
//     global_n_elem = (int *)malloc(sizeof(int) * n_buckets);
//     global_starting_position = (int *)malloc(sizeof(int) * n_buckets);
//     memset(global_n_elem, 0, sizeof(int) * n_buckets);
//     memset(global_starting_position, 0, sizeof(int) * n_buckets);

//     // local buckets, n_buckets for each thread
//     buckets = (struct bucket *)calloc(n_buckets * num_threads, sizeof(struct bucket));
//   }

//   ~TempStorage() {
//     free(global_n_elem);
//     free(global_starting_position);
//     free(buckets);
//   }

//   int *global_n_elem;
//   int *global_starting_position;
//   struct bucket *buckets;
// };

// ----------------------------------------------------------------------------
// New version
// ----------------------------------------------------------------------------

// struct TmpStorage {
//  public:
//   TmpStorage() = default;
//   ~TmpStorage() = default;

//   // Disallow copy if needed
//   TmpStorage(const TmpStorage &) = delete;
//   TmpStorage &operator=(const TmpStorage &) = delete;

//   // Allow move semantics
//   TmpStorage(TmpStorage &&) = default;
//   TmpStorage &operator=(TmpStorage &&) = default;

//   // getters
//   [[nodiscard]] int *global_n_elem() { return h_global_n_elem.data(); }
//   [[nodiscard]] int *global_starting_position() { return h_global_starting_position.data(); }
//   [[nodiscard]] bucket *buckets() { return h_buckets.data(); }

//   [[nodiscard]] bool is_allocated() const { return m_n_buckets > 0; }

//   // Allocate memory for storage
//   void allocate(int n_buckets, int num_threads) {
//     m_n_buckets = n_buckets;
//     m_num_threads = num_threads;

//     h_global_n_elem.assign(n_buckets, 0);
//     h_global_starting_position.assign(n_buckets, 0);

//     // Each thread gets n_buckets, so total = n_buckets * num_threads
//     h_buckets.assign(n_buckets * num_threads, bucket{});  // Default-initialize buckets
//   }

//   // // Reset the contents
//   // void reset() {
//   //   std::fill(global_n_elem.begin(), global_n_elem.end(), 0);
//   //   std::fill(global_starting_position.begin(), global_starting_position.end(), 0);

//   //   // Reset all buckets
//   //   for (auto &b : buckets) {
//   //     b = bucket{};  // Re-initialize each bucket to default state
//   //   }
//   // }

//  private:
//   int m_n_buckets = 0;
//   int m_num_threads = 0;

//   std::vector<int> h_global_n_elem;
//   std::vector<int> h_global_starting_position;
//   std::vector<bucket> h_buckets;
// };

}  // namespace tree::omp
