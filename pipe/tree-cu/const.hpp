#pragma once

#include <queue>

#include "builtin-apps/pipeline/record.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"
#include "builtin-apps/tree/cuda/dispatchers.cuh"
#include "builtin-apps/tree/omp/dispatchers.hpp"

// Application-specific constants
constexpr size_t kNumStages = 7;

using DispatcherT = tree::cuda::CudaDispatcher;
using AppDataT = tree::SafeAppData;
using AppDataPtr = std::unique_ptr<AppDataT>;

// Pipeline-specific constants
constexpr size_t kPoolSize = 32;
constexpr size_t kNumToProcess = 100;

using QueueT = SPSCQueue<AppDataT*, kPoolSize>;
using LocalQueue = std::queue<AppDataT*>;

// Shared worker / queue / dataset plumbing (was duplicated in all 6 cells).
#include "runtime/pipeline.hpp"
