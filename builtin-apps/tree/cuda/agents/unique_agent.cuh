#pragma once

#include <device_launch_parameters.h>

#include <cub/cub.cuh>

namespace cuda {

namespace agents {

// ----------------------------------------------------------------------------
// Agent class
// ----------------------------------------------------------------------------

struct UniqueAgent {
  static constexpr auto n_threads = 256;
  static constexpr auto items_per_thread = 4;
  static constexpr auto tile_size = n_threads * items_per_thread;

  // Note: 'unsigned int' for the data
  using BlockLoad =
      cub::BlockLoad<unsigned int, n_threads, items_per_thread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockDiscontinuity = cub::BlockDiscontinuity<unsigned int, n_threads>;

  // Note: 'int' for 'flag_heads'
  using BlockStore =
      cub::BlockStore<int, n_threads, items_per_thread, cub::BLOCK_STORE_WARP_TRANSPOSE>;

  using TempStorage = union {
    BlockLoad::TempStorage load;
    BlockDiscontinuity::TempStorage discontinuity;
    BlockStore::TempStorage store;
  };

  __device__ __forceinline__ UniqueAgent(const size_t n) : n(n) {}

  __device__ __forceinline__ void Process_FindDups(TempStorage &temp_storage,
                                                   const unsigned int *u_keys,
                                                   int *u_flag_heads,
                                                   const int n) {
    const auto num_tiles = cub::DivideAndRoundUp(n, tile_size);

    for (auto tile_idx = blockIdx.x; tile_idx < num_tiles; tile_idx += gridDim.x) {
      const auto tile_offset = tile_idx * tile_size;

      // Have thread0 obtain the predecessor item for the entire tile
      unsigned int tile_predecessor_item;
      if (threadIdx.x == 0)
        tile_predecessor_item = (tile_offset == 0) ? u_keys[0] : u_keys[tile_offset - 1];

      unsigned int thread_data[items_per_thread];
      int head_flags[items_per_thread];

      BlockLoad(temp_storage.load).Load(u_keys + tile_offset, thread_data);
      __syncthreads();

      BlockDiscontinuity(temp_storage.discontinuity)
          .FlagHeads(head_flags, thread_data, cub::Inequality(), tile_predecessor_item);
      __syncthreads();

      BlockStore(temp_storage.store).Store(u_flag_heads + tile_offset, head_flags);
      __syncthreads();
    }
  }

  __device__ __forceinline__ void Process_MoveDups(const unsigned int *u_keys,
                                                   const int *u_flag_heads_sums,
                                                   const int n,
                                                   unsigned int *u_keys_out,
                                                   int *n_unique_out = nullptr) {
    // 'tile' == 'block' in this kernel
    const auto num_tiles = cub::DivideAndRoundUp(n, blockDim.x);

    for (auto tile_idx = blockIdx.x; tile_idx < num_tiles; tile_idx += gridDim.x) {
      const auto idx = tile_idx * blockDim.x + threadIdx.x;
      if (idx < n) {
        u_keys_out[u_flag_heads_sums[idx]] = u_keys[idx];
      }
    }

    // we want to save number of unique elements if n_unique_out is not null
    if (n_unique_out != nullptr && threadIdx.x == 0) {
      *n_unique_out = u_flag_heads_sums[n - 1] + 1;
    }
  }

  size_t n;
};

}  // namespace agents

}  // namespace cuda
