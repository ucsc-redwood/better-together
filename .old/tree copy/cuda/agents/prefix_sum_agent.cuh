#pragma once

#include <device_launch_parameters.h>

#include <cub/cub.cuh>

namespace cuda {

namespace agents {

// ----------------------------------------------------------------------------
// Agent class
// ----------------------------------------------------------------------------

template <typename T>
struct PrefixSumAgent {
  struct BlockPrefixCallbackOp {
    T running_total;

    __device__ __forceinline__ explicit BlockPrefixCallbackOp(const T running_total)
        : running_total(running_total) {}

    // Callback operator to be entered by the first warp of threads in the
    // block. Thread-0 is responsible for returning a value for seeding the
    // block-wide scan.
    __device__ __forceinline__ T operator()(const T block_aggregate) {
      T old_prefix = running_total;
      running_total += block_aggregate;
      return old_prefix;
    }
  };

  // these configuration are same across kernels
  static constexpr auto n_threads = 128;
  static constexpr auto items_per_thread = 8;

  static constexpr auto tile_size = n_threads * items_per_thread;

  using BlockLoad = cub::BlockLoad<T, n_threads, items_per_thread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockScan = cub::BlockScan<T, n_threads>;
  using BlockStore =
      cub::BlockStore<T, n_threads, items_per_thread, cub::BLOCK_STORE_WARP_TRANSPOSE>;

  using TempStorage_LoadStore = union {
    typename BlockLoad::TempStorage load;
    typename BlockStore::TempStorage store;
  };

  using TempStorage_LoadScanStore = union {
    typename BlockLoad::TempStorage load;
    typename BlockScan::TempStorage scan;
    typename BlockStore::TempStorage store;
  };

  // Constructor
  __device__ __forceinline__ explicit PrefixSumAgent(const size_t n) : n(n) {}

  __device__ __forceinline__ void Process_LocalPrefixSums(TempStorage_LoadScanStore& temp_storage,
                                                          const T* u_input,
                                                          T* u_local_sums,
                                                          volatile T* u_auxiliary = nullptr) {
    const auto num_tiles = cub::DivideAndRoundUp(n, tile_size);

    // Process full tiles (num_tiles - 1)
    for (auto tile_idx = blockIdx.x; tile_idx < num_tiles; tile_idx += gridDim.x) {
      T thread_data[items_per_thread];

      BlockLoad(temp_storage.load).Load(u_input + tile_idx * tile_size, thread_data);
      __syncthreads();

      [[maybe_unused]] T aggregate;
      BlockScan(temp_storage.scan).ExclusiveSum(thread_data, thread_data, aggregate);
      __syncthreads();

      // if (u_auxiliary && threadIdx.x == 0) {
      //   u_auxiliary[tile_idx] = aggregate;
      // }

      BlockStore(temp_storage.store).Store(u_local_sums + tile_idx * tile_size, thread_data);
      __syncthreads();
    }

    // I know this sucks, but this give the correct result

    // if (u_auxiliary) {
    //   for (auto tile_idx = threadIdx.x; tile_idx < num_tiles - 1;
    //        tile_idx += n_threads) {
    //     u_auxiliary[tile_idx] =
    //         u_local_sums[tile_idx * tile_size + tile_size - 1] + 1;
    //   }
    //   u_auxiliary[num_tiles - 1] = u_local_sums[n - 1] + 1;
    // }

    if (u_auxiliary && threadIdx.x == 0) {
      for (auto tile_idx = 0; tile_idx < num_tiles - 1; ++tile_idx) {
        u_auxiliary[tile_idx] = u_local_sums[tile_idx * tile_size + tile_size - 1] + 1;
      }
      u_auxiliary[num_tiles - 1] = u_local_sums[n - 1] + 1;
    }
  }

  __device__ __forceinline__ void Process_SingleBlockExclusiveScan(
      TempStorage_LoadScanStore& temp_storage, const T* u_input, T* u_output, T init = T()) {
    BlockPrefixCallbackOp prefix_op(init);

    for (auto my_block_offset = 0; my_block_offset < n; my_block_offset += tile_size) {
      T thread_data[items_per_thread];

      BlockLoad(temp_storage.load).Load(u_input + my_block_offset, thread_data, n);
      __syncthreads();

      BlockScan(temp_storage.scan).ExclusiveSum(thread_data, thread_data, prefix_op);
      __syncthreads();

      BlockStore(temp_storage.store).Store(u_output + my_block_offset, thread_data, n);
      __syncthreads();
    }
  }

  // is this safe to use local_sums as input and output? (in-place) I think it
  // is
  __device__ __forceinline__ void Process_GlobalPrefixSum(TempStorage_LoadStore& temp_storage,
                                                          const T* u_local_sums,
                                                          const T* u_auxiliary_summed,
                                                          T* u_global_sums) {
    const auto num_tiles = cub::DivideAndRoundUp(n, tile_size);

    for (auto tile_idx = blockIdx.x; tile_idx < num_tiles; tile_idx += gridDim.x) {
      T thread_data[items_per_thread];

      BlockLoad(temp_storage.load).Load(u_local_sums + tile_idx * tile_size, thread_data, n);
      __syncthreads();

#pragma unroll
      for (auto i = 0; i < items_per_thread; ++i) {
        if (const auto idx = tile_idx * tile_size + i; idx < n) {
          thread_data[i] += u_auxiliary_summed[tile_idx];
        }
      }

      BlockStore(temp_storage.store).Store(u_global_sums + tile_idx * tile_size, thread_data, n);
      __syncthreads();
    }
  }

  size_t n;
};

}  // namespace agents

}  // namespace cuda
