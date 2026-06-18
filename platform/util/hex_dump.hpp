#pragma once

#include <memory_resource>
#include <vector>

void dumpHexRaw(const void* ptr,
                std::size_t num_bytes,
                std::size_t max_bytes = 64,
                std::size_t bytes_per_line = 16);
template <typename T>
void dumpHex(const std::vector<T>& vec,
             std::size_t max_bytes = 64,
             std::size_t bytes_per_line = 16) {
  dumpHexRaw(vec.data(), vec.size() * sizeof(T), max_bytes, bytes_per_line);
}

template <typename T>
void dumpHex(const std::pmr::vector<T>& vec,
             std::size_t max_bytes = 64,
             std::size_t bytes_per_line = 16) {
  dumpHexRaw(vec.data(), vec.size() * sizeof(T), max_bytes, bytes_per_line);
}

// ────────────────────────────────────────────────────────
// 1) Raw‐pointer “compressed” dump
//    - ptr:     start of region
//    - num_bytes: total bytes in region
//    - block_size: number of bytes to fold into one summary
//    - max_blocks: maximum number of blocks to print
//    - blocks_per_line: how many summary‐bytes per output line
// ────────────────────────────────────────────────────────
void dumpCompressedRaw(const void* ptr,
                       std::size_t num_bytes,
                       std::size_t block_size = 16,
                       std::size_t max_blocks = 256,
                       std::size_t blocks_per_line = 32);

template <typename T>
void dumpCompressed(const std::vector<T>& vec,
                    std::size_t block_size = 16,
                    std::size_t max_blocks = 256,
                    std::size_t blocks_per_line = 32) {
  dumpCompressedRaw(vec.data(), vec.size() * sizeof(T), block_size, max_blocks, blocks_per_line);
}

template <typename T>
void dumpCompressed(const std::pmr::vector<T>& vec,
                    std::size_t block_size = 16,
                    std::size_t max_blocks = 256,
                    std::size_t blocks_per_line = 32) {
  dumpCompressedRaw(vec.data(), vec.size() * sizeof(T), block_size, max_blocks, blocks_per_line);
}
