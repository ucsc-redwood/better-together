// #include "03_unique.cuh"
// #include "agents/unique_agent.cuh"

// namespace cuda {

// namespace kernels {

// __global__ void k_FindDups(const unsigned int *u_keys, int *u_flag_heads, const int n) {
//   __shared__ agents::UniqueAgent::TempStorage temp_storage;

//   agents::UniqueAgent agent(n);
//   agent.Process_FindDups(temp_storage, u_keys, u_flag_heads, n);
// }

// __global__ void k_MoveDups(const unsigned int *u_keys,
//                            const int *u_flag_heads_sums,
//                            const int n,
//                            unsigned int *u_keys_out,
//                            int *n_unique_out) {
//   agents::UniqueAgent agent(n);
//   agent.Process_MoveDups(u_keys, u_flag_heads_sums, n, u_keys_out, n_unique_out);
// }

// }  // namespace kernels

// }  // namespace cuda
