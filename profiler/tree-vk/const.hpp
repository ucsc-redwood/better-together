#pragma once

#include <queue>

#include "runtime/record.hpp"
#include "runtime/spsc_queue.hpp"
#include "apps/tree/omp/dispatchers.hpp"
#include "apps/tree/vulkan/dispatchers.hpp"

// Application-specific constants
constexpr size_t kNumStages = 7;

using DispatcherT = tree::vulkan::VulkanDispatcher;
using AppDataT = tree::vulkan::VkAppData_Safe;
using AppDataPtr = std::unique_ptr<AppDataT>;

// Pipeline-specific constants
constexpr size_t kPoolSize = 16;
constexpr size_t kNumToProcess = 100;

using QueueT = SPSCQueue<AppDataT*, kPoolSize>;
using LocalQueue = std::queue<AppDataT*>;

// Shared worker / queue / dataset plumbing (was duplicated in all 6 cells).
#include "runtime/pipeline.hpp"
