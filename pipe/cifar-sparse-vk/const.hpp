#pragma once

#include <queue>

#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/pipeline/record.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"

// Application-specific constants
constexpr size_t kNumStages = 9;

using DispatcherT = cifar_sparse::vulkan::VulkanDispatcher;
using AppDataT = cifar_sparse::AppData;
using AppDataPtr = std::unique_ptr<AppDataT>;

// Pipeline-specific constants
constexpr size_t kPoolSize = 16;
constexpr size_t kNumToProcess = 100;

using QueueT = SPSCQueue<AppDataT*, kPoolSize>;
using LocalQueue = std::queue<AppDataT*>;

// Shared worker / queue / dataset plumbing (was duplicated in all 6 cells).
#include "../pipeline_common.hpp"
