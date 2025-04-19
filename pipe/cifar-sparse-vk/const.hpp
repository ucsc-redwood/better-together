#pragma once

#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"
#include "builtin-apps/pipeline/spsc_queue.hpp"
#include "builtin-apps/pipeline/task.hpp"

// Application-specific constants

constexpr size_t kNumStages = 9;

using DispatcherT = cifar_sparse::vulkan::VulkanDispatcher;
using AppDataT = cifar_sparse::AppData;
using TaskT = Task<AppDataT>;

// Pipeline-specific constants

constexpr size_t kPoolSize = 16;
constexpr size_t kNumToProcess = 100;

using QueueT = SPSCQueue<TaskT*, kPoolSize>;
using LocalQueue = std::queue<TaskT*>;
