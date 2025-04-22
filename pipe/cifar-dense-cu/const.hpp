#pragma once

#include <queue>

#include "builtin-apps/cifar-dense/cuda/dispatchers.cuh"
#include "builtin-apps/cifar-dense/omp/dispatchers.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"
// #include "builtin-apps/pipeline/task.hpp"

// Application-specific constants

constexpr size_t kNumStages = 9;

using DispatcherT = cifar_dense::cuda::CudaDispatcher;
using AppDataT = cifar_dense::AppData;
// using TaskT = Task<AppDataT>;
using AppDataPtr = std::shared_ptr<AppDataT>;

// Pipeline-specific constants

constexpr size_t kPoolSize = 16;
constexpr size_t kNumToProcess = 100;

// using QueueT = SPSCQueue<TaskT*, kPoolSize>;
// using LocalQueue = std::queue<TaskT*>;
