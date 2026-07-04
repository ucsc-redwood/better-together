#pragma once

#include <queue>

#include "apps/tree/omp/dispatchers.hpp"
#include "apps/tree/vulkan/dispatchers.hpp"
#include "runtime/record.hpp"
#include "runtime/spsc_queue.hpp"

// Application-specific constants
constexpr size_t kNumStages = 7;

using DispatcherT = tree::vulkan::VulkanDispatcher;
// Compact, genuinely-chained path (see apps/tree/vulkan/vk_appdata.hpp) -- every
// production profiling tool here is generic over AppDataT, so this one alias is
// the whole switch. VkAppData_Safe (golden-decoupled) remains the differential/
// oracle path for apps/tree/vulkan/test_main.cpp's TreeDiffVulkan suite and
// test_pipeline_main_vk.cpp, which bind their own AppData type directly and
// don't go through this file. Mirrors profiler/tree-cu/const.hpp's Phase 2
// switch.
using AppDataT = tree::vulkan::VkAppData;
using AppDataPtr = std::unique_ptr<AppDataT>;

// Pipeline-specific constants
constexpr size_t kPoolSize = 16;
constexpr size_t kNumToProcess = 100;

using QueueT = SPSCQueue<AppDataT*, kPoolSize>;
using LocalQueue = std::queue<AppDataT*>;

// Shared worker / queue / dataset plumbing (was duplicated in all 6 cells).
#include "runtime/pipeline.hpp"
