#pragma once

#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/tree/omp/dispatchers.hpp"
#include "builtin-apps/tree/vulkan/dispatchers.hpp"

template <int N>
concept AllowedStage = (N >= 1 && N <= 7);

// ---------------------------------------------------------------------
// CPU stages
// ---------------------------------------------------------------------

namespace omp {

constexpr std::array<void (*)(tree::SafeAppData &), 7> cpu_stages = {
    tree::omp::process_stage_1,
    tree::omp::process_stage_2,
    tree::omp::process_stage_3,
    tree::omp::process_stage_4,
    tree::omp::process_stage_5,
    tree::omp::process_stage_6,
    tree::omp::process_stage_7,
};

template <int Start, int End, ProcessorType PT, int NThreads>
  requires AllowedStage<Start> && AllowedStage<End> && (Start <= End)
void run_multiple_stages(tree::SafeAppData &data) {
#pragma omp parallel num_threads(NThreads)
  {
    // Bind to core
    if constexpr (PT == ProcessorType::kLittleCore) {
      bind_thread_to_cores(g_little_cores);
    } else if constexpr (PT == ProcessorType::kMediumCore) {
      bind_thread_to_cores(g_medium_cores);
    } else if constexpr (PT == ProcessorType::kBigCore) {
      bind_thread_to_cores(g_big_cores);
    }

    // don't know if this has any effect
    for (int s = Start; s <= End; ++s) {
      cpu_stages[s - 1](data);
    }
  }
}

}  // namespace omp

// ---------------------------------------------------------------------
// GPU stages
// ---------------------------------------------------------------------

namespace vulkan {

template <int Start, int End>
  requires AllowedStage<Start> && AllowedStage<End> && (Start <= End)
void run_gpu_stages(tree::vulkan::VkAppData_Safe &data) {
  // Generate a compile-time sequence for the range [start_stage, end_stage]
  []<std::size_t... I>(std::index_sequence<I...>, tree::vulkan::VkAppData_Safe &data) {
    ((tree::vulkan::Singleton::getInstance().run_stage<Start + I>(data)), ...);
  }(std::make_index_sequence<End - Start + 1>{}, data);
}

}  // namespace vulkan
