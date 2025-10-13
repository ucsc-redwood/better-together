#pragma once

#include <stddef.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <emmintrin.h>
// For x86: Flush the cache line using the CLFLUSH instruction and ensure ordering with a fence.
inline void flushCacheLine(void* ptr) {
  _mm_clflush(ptr);
  _mm_mfence();  // Ensure the flush completes before further memory operations.
}
#elif defined(__aarch64__)
// For ARM64: Use inline assembly to clean and invalidate the data cache line.
inline void flushCacheLine(void* ptr) {
  // Clean and invalidate the cache line containing 'ptr' to the point of coherency.
  asm volatile("dc civac, %0" ::"r"(ptr) : "memory");
  // Data Synchronization Barrier to ensure the operation completes.
  asm volatile("dsb ish" ::: "memory");
}
#else
#error "Unsupported architecture"
#endif
