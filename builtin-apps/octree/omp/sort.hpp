#pragma once

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../affinity.hpp"

// Constants for 32‑bit keys, 8 bits per pass
constexpr int BITS = 8;
constexpr int RADIX = 1 << BITS;                // 256 buckets
constexpr int PASSES = (32 + BITS - 1) / BITS;  // 4 passes

static inline void radix_sort_in_parallel(uint32_t* buffer_in,
                                          uint32_t* buffer_out,
                                          size_t n,
                                          size_t* local_hist,    // size = num_threads * RADIX
                                          size_t* local_offset,  // size = num_threads * RADIX
                                          size_t* global_hist,   // size = RADIX
                                          size_t* prefix         // size = RADIX
) {
  int tid = omp_get_thread_num();
  int nth = omp_get_num_threads();
  auto lh = local_hist + size_t(tid) * RADIX;
  auto lo = local_offset + size_t(tid) * RADIX;

  for (int pass = 0; pass < PASSES; ++pass) {
    uint32_t shift = pass * BITS;
    uint32_t mask = RADIX - 1;

    // choose in/out buffers by pass parity
    uint32_t* from = (pass & 1) ? buffer_out : buffer_in;
    uint32_t* to = (pass & 1) ? buffer_in : buffer_out;

    // zero this thread’s histogram
    for (int d = 0; d < RADIX; ++d) lh[d] = 0;

// build local histograms
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      uint32_t bucket = (from[i] >> shift) & mask;
      lh[bucket]++;
    }
#pragma omp barrier

// reduce + prefix + per‑thread offsets
#pragma omp single
    {
      // global histogram
      for (int d = 0; d < RADIX; ++d) {
        size_t sum = 0;
        for (int t = 0; t < nth; ++t) sum += local_hist[size_t(t) * RADIX + d];
        global_hist[d] = sum;
      }
      // exclusive prefix
      prefix[0] = 0;
      for (int d = 1; d < RADIX; ++d) prefix[d] = prefix[d - 1] + global_hist[d - 1];
      // per‑thread offsets
      for (int d = 0; d < RADIX; ++d) {
        size_t off = prefix[d];
        for (int t = 0; t < nth; ++t) {
          local_offset[size_t(t) * RADIX + d] = off;
          off += local_hist[size_t(t) * RADIX + d];
        }
      }
    }
#pragma omp barrier

// scatter into the “to” buffer
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      uint32_t bucket = (from[i] >> shift) & mask;
      to[lo[bucket]++] = from[i];
    }
#pragma omp barrier
  }

// after an even number of passes (4), “buffer_in” holds the sorted data
// copy it back into buffer_out exactly once:
#pragma omp single
  {
    std::copy(buffer_in, buffer_in + n, buffer_out);
  }
#pragma omp barrier
}

static inline void dispatch_radix_sort(std::vector<uint32_t>& buffer_in,
                                       std::vector<uint32_t>& buffer_out,
                                       const size_t num_threads,
                                       const std::vector<int> core_ids = {}) {
  const size_t n = buffer_in.size();

  // –– 1) Static workspace (un‑sized on declaration) ––//
  static std::vector<size_t> local_hist;
  static std::vector<size_t> local_offset;
  static std::vector<size_t> global_hist;
  static std::vector<size_t> prefix;

  // –– 2) One‑time allocation & sizing ––//
  //      We only ever allocate when sizes change.
  static size_t last_threads = 0;
  if (last_threads != num_threads) {
    last_threads = num_threads;
    local_hist.assign(num_threads * RADIX, 0);
    local_offset.assign(num_threads * RADIX, 0);
    global_hist.assign(RADIX, 0);
    prefix.assign(RADIX, 0);
  } else {
    // –– 3) Zero out existing buffers before EACH call ––//
    std::ranges ::fill(local_hist, 0);
    std::ranges::fill(local_offset, 0);
    std::ranges::fill(global_hist, 0);
    std::ranges::fill(prefix, 0);
  }

#pragma omp parallel num_threads(num_threads)
  {
    // 1) pin this thread where you like:
    // pin_this_thread_to_core( omp_get_thread_num() );
    if (!core_ids.empty()) {
      bind_thread_to_cores(core_ids);
    }

    // 2) do the sort (all threads enter here together):
    radix_sort_in_parallel(buffer_in.data(),
                           buffer_out.data(),
                           n,
                           local_hist.data(),
                           local_offset.data(),
                           global_hist.data(),
                           prefix.data());
  }  // implicit barrier – now buffer_out is sorted
}