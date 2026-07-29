#pragma once

#include <queue>

#include "apps/tree/cuda/dispatchers.cuh"
#include "apps/tree/omp/dispatchers.hpp"
#include "runtime/record.hpp"
#include "runtime/spsc_queue.hpp"

// Application-specific constants
constexpr size_t kNumStages = 7;

using DispatcherT = tree::cuda::CudaDispatcher;
// Compact, genuinely-chained path (see apps/tree/tree_appdata.hpp) -- every
// production profiling tool here is generic over AppDataT, so this one alias
// is the whole switch. SafeAppData (golden-decoupled, ~2x this size) remains
// the differential/oracle path for apps/tree/cuda/test_main.cu and the
// runtime-mechanics test test_pipeline_main_cu.cu, which bind their own
// AppData type directly and don't go through this file.
using AppDataT = tree::AppData;
using AppDataPtr = std::unique_ptr<AppDataT>;

// Pipeline-specific constants
constexpr size_t kPoolSize = 32;
constexpr size_t kNumToProcess = 100;

using QueueT = SPSCQueue<AppDataT*, kPoolSize>;
using LocalQueue = std::queue<AppDataT*>;

// Shared worker / queue / dataset plumbing (was duplicated in all 6 cells).
#include "runtime/pipeline.hpp"
