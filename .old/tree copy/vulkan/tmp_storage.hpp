// #pragma once

// #include <memory_resource>
// #include <vector>

// namespace tree::vulkan {

// struct TmpStorage {
//   TmpStorage(std::pmr::memory_resource* mr, const size_t n_input)
//       : u_contributes(n_input, mr),
//         u_out_idx(n_input, mr),
//         u_sums(n_input, mr),
//         u_prefix_sums(n_input, mr) {}

//   // for remove duplicates
//   std::pmr::vector<uint32_t> u_contributes;
//   std::pmr::vector<uint32_t> u_out_idx;

//   // for prefix sum
//   std::pmr::vector<uint32_t> u_sums;
//   std::pmr::vector<uint32_t> u_prefix_sums;
// };

// }  // namespace tree::vulkan
