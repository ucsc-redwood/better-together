// #pragma once

// namespace cuda {

// namespace kernels {

// /**
//  * @brief Find Duplicate elements in a sorted array, using 'discontinuity', and
//  * write to the u_flag_heads array. This is the first step in the process of
//  * finding unique elements in an array.
//  *
//  * @param u_keys The input array
//  * @param u_flag_heads the discontinuity flags
//  * @param n
//  */
// __global__ void k_FindDups(const unsigned int *u_keys, int *u_flag_heads, int n);

// /**
//  * @brief After prefix sum on the u_flag_heads array, move the unique elements
//  * in new order.
//  *
//  * @param u_keys The input array
//  * @param u_flag_heads_sums needs to be prefix summed
//  * @param n
//  * @param u_keys_out
//  * @param n_unique_out May be nullptr.
//  */
// __global__ void k_MoveDups(const unsigned int *u_keys,
//                            const int *u_flag_heads_sums,
//                            int n,
//                            unsigned int *u_keys_out,
//                            int *n_unique_out = nullptr);

// }  // namespace kernels

// }  // namespace cuda
