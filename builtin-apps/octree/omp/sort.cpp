#include "sort.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

int main() {
  size_t n = 1024;
  std::vector<uint32_t> buffer_in(n), buffer_out(n);

  // fill buffer_in…
  std::iota(buffer_in.begin(), buffer_in.end(), 0);
  std::shuffle(buffer_in.begin(), buffer_in.end(), std::mt19937(114514));

  dispatch_radix_sort(buffer_in, buffer_out, 8);

  bool is_sorted = std::is_sorted(buffer_out.begin(), buffer_out.end());
  printf("is_sorted: %s\n", is_sorted ? "true" : "false");

  return 0;
}
