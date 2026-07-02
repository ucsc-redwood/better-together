#pragma once

#include <queue>

#include "apps/cifar-sparse/cuda/dispatchers.cuh"
#include "apps/cifar-sparse/omp/dispatchers.hpp"
#include "runtime/record.hpp"
#include "runtime/spsc_queue.hpp"

// Application-specific constants
constexpr size_t kNumStages = 11;

using DispatcherT = cifar_sparse::cuda::CudaDispatcher;
using AppDataT = cifar_sparse::AppData;
using AppDataPtr = std::unique_ptr<AppDataT>;

// Pipeline-specific constants
// AlexNetCIFAR AppData is ~250 MB (two 4096x4096 FC weights); pool x 250 MB must fit a
// 7.4 GB Jetson alongside the OS.
constexpr size_t kPoolSize = 6;
constexpr size_t kNumToProcess = 100;

using QueueT = SPSCQueue<AppDataT*, kPoolSize>;
using LocalQueue = std::queue<AppDataT*>;

// Shared worker / queue / dataset plumbing (was duplicated in all 6 cells).
#include "runtime/pipeline.hpp"
