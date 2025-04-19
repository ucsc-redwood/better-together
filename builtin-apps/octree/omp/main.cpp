#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../../app.hpp"
#include "../appdata.hpp"
#include "dispatchers.hpp"

constexpr int RADIX = 256;
constexpr int BITS = 8;
constexpr int PASSES = (32 + BITS - 1) / BITS;  // =4

// void parallel_radix_sort(uint32_t* buffer_in, uint32_t* buffer_out, size_t n, uint32_t* hist) {
//   uint32_t* src = buffer_in;
//   uint32_t* dst = buffer_out;

//   // Find how many threads we'll actually use:
//   int num_threads = omp_get_max_threads();

//   for (int pass = 0; pass < PASSES; ++pass) {
//     int shift = pass * BITS;

// // 1) Zero per-thread histograms
// #pragma omp parallel
//     {
//       int tid = omp_get_thread_num();
//       uint32_t* my_hist = hist + tid * RADIX;
//       std::memset(my_hist, 0, RADIX * sizeof(uint32_t));
//     }

// // 2) Build per-thread histograms
// #pragma omp parallel
//     {
//       int tid = omp_get_thread_num();
//       uint32_t* my_hist = hist + tid * RADIX;

// #pragma omp for schedule(static)
//       for (size_t i = 0; i < n; ++i) {
//         uint32_t key = src[i];
//         uint32_t bucket = (key >> shift) & 0xFF;
//         my_hist[bucket]++;
//       }
//     }

//     // 3) Compute global bucket offsets
//     //    global_counts[b] = total number of keys in bucket b
//     //    hist[tid*RADIX + b] will be repurposed to the *starting* offset for thread tid, bucket
//     b uint32_t global_counts[RADIX] = {0};

//     // Sum up into global_counts
//     for (int t = 0; t < num_threads; ++t) {
//       uint32_t* my_hist = hist + t * RADIX;
//       for (int b = 0; b < RADIX; ++b) {
//         global_counts[b] += my_hist[b];
//       }
//     }

//     // Prefix‑sum global_counts to get the global bucket baselines
//     uint32_t bucket_starts[RADIX];
//     {
//       uint32_t sum = 0;
//       for (int b = 0; b < RADIX; ++b) {
//         bucket_starts[b] = sum;
//         sum += global_counts[b];
//       }
//     }

//     // Now compute each thread’s starting offset for each bucket
//     for (int b = 0; b < RADIX; ++b) {
//       uint32_t offset = bucket_starts[b];
//       for (int t = 0; t < num_threads; ++t) {
//         uint32_t cnt = hist[t * RADIX + b];
//         hist[t * RADIX + b] = offset;
//         offset += cnt;
//       }
//     }

// // 4) Scatter into dst using each thread’s local hist offsets
// #pragma omp parallel
//     {
//       int tid = omp_get_thread_num();
//       uint32_t* my_offsets = hist + tid * RADIX;

// #pragma omp for schedule(static)
//       for (size_t i = 0; i < n; ++i) {
//         uint32_t key = src[i];
//         uint32_t bucket = (key >> shift) & 0xFF;
//         uint32_t dst_idx = my_offsets[bucket]++;
//         dst[dst_idx] = key;
//       }
//     }

//     // 5) Ping‑pong for next pass
//     std::swap(src, dst);
//   }

//   // After an even number of passes (4), src points at buffer_in, dst at buffer_out.
//   // We need the final sorted data in buffer_out, so if src == buffer_out, we're done.
//   // Otherwise (src==buffer_in), we must copy from temp (which we never wrote) – this won’t
//   happen
//   // because we swapped exactly 4 times.  But to be safe:
//   if (src != buffer_out) {
//     // Copy sorted data from src->buffer_out
//     std::memcpy(buffer_out, src, n * sizeof(uint32_t));
//   }
// }






// Call from inside:
//   #pragma omp parallel num_threads(nthreads) shared(src,dst)
//     parallel_radix_sort_imp(src, dst, n, hist, nthreads);
//
void parallel_radix_sort_imp(uint32_t*& src,
                             uint32_t*& dst,
                             size_t     n,
                             uint32_t*  hist,
                             int        nthreads)
{
    int tid = omp_get_thread_num();

    for(int pass = 0; pass < PASSES; ++pass) {
        int shift = pass * BITS;

        // DEBUG: entry to pass
        #pragma omp single
        {
            printf("[DEBUG] Beginning pass %d: src=%p, dst=%p, nthreads=%d\n",
                   pass, (void*)src, (void*)dst, nthreads);
        }
        #pragma omp barrier

        // 1) zero this thread’s histogram
        for(int b = 0; b < RADIX; ++b)
            hist[tid * RADIX + b] = 0;
        #pragma omp barrier

        // 2) build per-thread histogram
        #pragma omp for schedule(static)
        for(size_t i = 0; i < n; ++i) {
            uint32_t bucket = (src[i] >> shift) & 0xFF;
            hist[tid * RADIX + bucket]++;
        }
        #pragma omp barrier

        // DEBUG: print this thread’s histogram total
        {
            uint32_t local_sum = 0;
            for(int b = 0; b < RADIX; ++b)
                local_sum += hist[tid * RADIX + b];
            printf("[DEBUG] tid=%d pass=%d histogram sum=%u\n",
                   tid, pass, local_sum);
        }
        #pragma omp barrier

        // 3) compute global bucket offsets (single thread)
        #pragma omp single
        {
            // reuse this array to accumulate then prefix-sum
            static uint32_t global_counts[RADIX];
            std::memset(global_counts, 0, sizeof(global_counts));

            // sum per-thread histograms
            for(int t = 0; t < nthreads; ++t) {
                uint32_t* th = hist + t * RADIX;
                for(int b = 0; b < RADIX; ++b)
                    global_counts[b] += th[b];
            }

            // prefix‑sum to get starting offsets
            uint32_t sum = 0;
            for(int b = 0; b < RADIX; ++b) {
                uint32_t cnt = global_counts[b];
                global_counts[b] = sum;
                sum += cnt;
            }

            // scatter those offsets back into each thread’s hist array
            for(int t = 0; t < nthreads; ++t) {
                uint32_t* th = hist + t * RADIX;
                for(int b = 0; b < RADIX; ++b) {
                    uint32_t cnt = th[b];
                    th[b] = global_counts[b];
                    global_counts[b] += cnt;
                }
            }
            printf("[DEBUG] completed global offset build on pass %d\n", pass);
        }
        #pragma omp barrier

        // 4) scatter keys into dst
        #pragma omp for schedule(static)
        for(size_t i = 0; i < n; ++i) {
            uint32_t key    = src[i];
            uint32_t bucket = (key >> shift) & 0xFF;
            uint32_t dst_idx = hist[tid*RADIX + bucket]++;
            dst[dst_idx] = key;

            // DEBUG: for the first few elements only
            if (i < 3 && tid == 0) {
                printf("[DEBUG] tid=0 pass=%d i=%zu --> dst[%u]=%u\n",
                       pass, i, dst_idx, key);
            }
        }
        #pragma omp barrier

        // DEBUG: after scatter, show ptrs before swap
        #pragma omp single
        {
            printf("[DEBUG] pass %d before swap: src=%p, dst=%p\n",
                   pass, (void*)src, (void*)dst);
            std::swap(src, dst);
            printf("[DEBUG] pass %d after  swap: src=%p, dst=%p\n",
                   pass, (void*)src, (void*)dst);
        }
        #pragma omp barrier
    }
}


int main(int argc, char** argv) {
  parse_args(argc, argv);

  //   spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  //   auto mr = std::pmr::new_delete_resource();
  //   octree::AppData appdata(mr);

  // #pragma omp parallel
  //   {
  //     octree::omp::run_stage_1(appdata);
  //     octree::omp::run_stage_2(appdata);
  //   }

  //   // print first 10 points
  //   for (size_t i = 0; i < 10; i++) {
  //     std::cout << "[pos " << i << "]\t" << appdata.u_positions[i].x << ", "
  //               << appdata.u_positions[i].y << ", " << appdata.u_positions[i].z << ", "
  //               << appdata.u_positions[i].w << std::endl;
  //   }
  //   std::cout << std::endl;

  //   // print first 10 morton codes
  //   for (size_t i = 0; i < 10; i++) {
  //     std::cout << "[morton " << i << "]\t" << appdata.u_morton_codes[i] << std::endl;
  //   }
  //   std::cout << std::endl;

  //   //

  constexpr auto n = 1024;
  constexpr auto nthreads = 8;

  std::pmr::vector<uint32_t> buffer_in(n);
  std::pmr::vector<uint32_t> buffer_out(n);
  std::iota(buffer_in.begin(), buffer_in.end(), 0);
  std::reverse(buffer_in.begin(), buffer_in.end());

  std::vector<uint32_t> hist(nthreads * RADIX);


  //   parallel_radix_sort(buffer_in.data(), buffer_out.data(), n, hist.data());

  // allocate input, output, and all intermediate buffers

#pragma omp parallel num_threads(nthreads)
  { parallel_radix_sort_imp(buffer_in.data(), buffer_out.data(), n, hist.data(), nthreads); }

  // print first 10 sorted morton codes
  for (size_t i = 0; i < 10; i++) {
    std::cout << "[morton " << i << "]\t" << buffer_out[i] << std::endl;
  }
  std::cout << std::endl;

  bool is_sorted = std::is_sorted(buffer_out.begin(), buffer_out.end());
  std::cout << "is_sorted: " << (is_sorted ? "true" : "false") << std::endl;

  return 0;
}
