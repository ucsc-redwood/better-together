#include "hex_dump.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>

void dumpHexRaw(const void* ptr,
                std::size_t num_bytes,
                std::size_t max_bytes,
                std::size_t bytes_per_line) {
  const auto* byte_ptr = static_cast<const uint8_t*>(ptr);
  std::size_t to_print = std::min(num_bytes, max_bytes);

  std::cout << "Hex dump (" << to_print << "/" << num_bytes << " bytes):\n";

  for (std::size_t i = 0; i < to_print; ++i) {
    if (i % bytes_per_line == 0) {
      std::cout << std::setw(4) << std::setfill('0') << std::hex << i << ": ";
    }

    std::cout << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(byte_ptr[i])
              << ' ';

    if ((i + 1) % bytes_per_line == 0) {
      std::cout << '\n';
    }
  }

  if (to_print % bytes_per_line != 0) {
    std::cout << '\n';
  }

  if (to_print < num_bytes) {
    std::cout << "... (truncated)\n";
  }

  std::cout << std::dec << std::setfill(' ') << std::endl;
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
                       std::size_t block_size,
                       std::size_t max_blocks,
                       std::size_t blocks_per_line) {
  auto* byte_ptr = static_cast<const uint8_t*>(ptr);
  std::size_t total_blocks = (num_bytes + block_size - 1) / block_size;
  std::size_t to_print_blocks = std::min(total_blocks, max_blocks);

  std::cout << "Compressed dump: " << to_print_blocks << "/" << total_blocks << " blocks (each "
            << block_size << " bytes)\n";

  for (std::size_t b = 0; b < to_print_blocks; ++b) {
    // compute XOR of this block
    uint8_t summary = 0;
    std::size_t start = b * block_size;
    std::size_t end = std::min(start + block_size, num_bytes);
    for (std::size_t i = start; i < end; ++i) {
      summary ^= byte_ptr[i];
    }

    // at start of each line, print byte‐offset of this block
    if (b % blocks_per_line == 0) {
      std::cout << std::setw(6) << std::setfill('0') << std::hex << (b * block_size) << ": ";
    }

    // print the 1‐byte summary
    std::cout << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(summary) << ' ';

    if ((b + 1) % blocks_per_line == 0) {
      std::cout << '\n';
    }
  }

  if (to_print_blocks % blocks_per_line != 0) {
    std::cout << '\n';
  }

  if (to_print_blocks < total_blocks) {
    std::cout << "... (truncated)\n";
  }
  // restore formatting
  std::cout << std::dec << std::setfill(' ') << std::endl;
}
