#pragma once

#include "builtin-apps/pipeline/spsc_queue.hpp"
#include "builtin-apps/pipeline/task.hpp"
#include "builtin-apps/tree/omp/dispatchers.hpp"
#include "builtin-apps/tree/vulkan/dispatchers.hpp"

// Application-specific constants

constexpr size_t kNumStages = 7;

using DispatcherT = tree::vulkan::VulkanDispatcher;
using AppDataT = tree::vulkan::VkAppData_Safe;
using TaskT = Task<AppDataT>;

// Pipeline-specific constants

constexpr size_t kPoolSize = 16;
constexpr size_t kNumToProcess = 100;

using QueueT = SPSCQueue<TaskT*, kPoolSize>;
using LocalQueue = std::queue<TaskT*>;

// Processor-specific constants
