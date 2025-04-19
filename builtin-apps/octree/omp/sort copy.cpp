
#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

// Constants for 32‑bit keys, 8 bits per pass
constexpr int BITS = 8;
constexpr int RADIX = 1 << BITS;                // 256 buckets
constexpr int PASSES = (32 + BITS - 1) / BITS;  // 4 passes

// buffer_in, buffer_out: size n
// local_hist:     length >= num_threads * RADIX
// local_offset:   length >= num_threads * RADIX
// global_hist:    length >= RADIX
// prefix:         length >= RADIX
//
// All arrays must be initialized (i.e. storage reserved) by the caller.
// No dynamic allocation, no vector creation, etc. happens here.
void radix_sort_omp(uint32_t* buffer_in,
                    uint32_t* buffer_out,
                    size_t n,
                    size_t* local_hist,
                    size_t* local_offset,
                    size_t* global_hist,
                    size_t* prefix) {
  uint32_t* in_ptr = buffer_in;
  uint32_t* out_ptr = buffer_out;

// Parallel region spans all passes so we pay the threading cost only once
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    size_t chunk = (n + nth - 1) / nth;
    size_t start = tid * chunk;
    size_t end = std::min(start + chunk, n);

    for (int pass = 0; pass < PASSES; ++pass) {
      uint32_t shift = pass * BITS;
      uint32_t mask = RADIX - 1;

      // 1) zero out this thread's local histogram
      size_t* lh = local_hist + size_t(tid) * RADIX;
      for (int d = 0; d < RADIX; ++d) lh[d] = 0;

      // 2) build local histogram
      for (size_t i = start; i < end; ++i) {
        uint32_t bucket = (in_ptr[i] >> shift) & mask;
        lh[bucket]++;
      }
#pragma omp barrier

// 3) single thread computes global_hist, prefix, and per-thread offsets
#pragma omp single
      {
        // reduce into global_hist
        for (int d = 0; d < RADIX; ++d) {
          size_t sum = 0;
          for (int t = 0; t < nth; ++t) {
            sum += local_hist[size_t(t) * RADIX + d];
          }
          global_hist[d] = sum;
        }
        // exclusive prefix-sum across buckets
        prefix[0] = 0;
        for (int d = 1; d < RADIX; ++d) {
          prefix[d] = prefix[d - 1] + global_hist[d - 1];
        }
        // compute each thread’s starting offset for each bucket
        for (int d = 0; d < RADIX; ++d) {
          size_t off = prefix[d];
          for (int t = 0; t < nth; ++t) {
            local_offset[size_t(t) * RADIX + d] = off;
            off += local_hist[size_t(t) * RADIX + d];
          }
        }
      }
#pragma omp barrier

      // 4) scatter each element into out_ptr at our thread’s offset
      size_t* lo = local_offset + size_t(tid) * RADIX;
      for (size_t i = start; i < end; ++i) {
        uint32_t bucket = (in_ptr[i] >> shift) & mask;
        out_ptr[lo[bucket]++] = in_ptr[i];
      }
#pragma omp barrier

// 5) swap buffers for next pass
#pragma omp single
      std::swap(in_ptr, out_ptr);
#pragma omp barrier
    }
  }  // end parallel

  // After 4 passes, data may live in buffer_in or buffer_out.
  // We guarantee result in buffer_out:
  if (in_ptr == buffer_in) {
    std::copy(buffer_in, buffer_in + n, buffer_out);
  }
}

int main() {
  constexpr auto n = 1024;
  constexpr auto num_threads = 8;

  std::vector<uint32_t> buffer_in(n);
  std::vector<uint32_t> buffer_out(n);

  std::iota(buffer_in.begin(), buffer_in.end(), 0);

  std::mt19937 g(114514);
  std::shuffle(buffer_in.begin(), buffer_in.end(), g);

  std::vector<size_t> local_hist(size_t(num_threads) * RADIX);
  std::vector<size_t> local_offset(size_t(num_threads) * RADIX);
  std::vector<size_t> global_hist(RADIX);
  std::vector<size_t> prefix(RADIX);

  // Now sort—no allocations happen inside this call:
  radix_sort_omp(buffer_in.data(),
                 buffer_out.data(),
                 n,
                 local_hist.data(),
                 local_offset.data(),
                 global_hist.data(),
                 prefix.data());

  for (size_t i = 0; i < n; ++i) {
    std::cout << buffer_out[i] << " ";
  }

  bool is_sorted = std::is_sorted(buffer_out.begin(), buffer_out.end());
  std::cout << "is_sorted: " << (is_sorted ? "true" : "false") << std::endl;

  return 0;
}