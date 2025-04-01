#pragma once

#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"

template <int N>
concept AllowedStage = (N >= 1 && N <= 9);

// ---------------------------------------------------------------------
// CPU stages
// ---------------------------------------------------------------------

namespace omp {

constexpr std::array<void (*)(cifar_sparse::AppData &), 9> cpu_stages = {
    cifar_sparse::omp::process_stage_1,
    cifar_sparse::omp::process_stage_2,
    cifar_sparse::omp::process_stage_3,
    cifar_sparse::omp::process_stage_4,
    cifar_sparse::omp::process_stage_5,
    cifar_sparse::omp::process_stage_6,
    cifar_sparse::omp::process_stage_7,
    cifar_sparse::omp::process_stage_8,
    cifar_sparse::omp::process_stage_9,
};

template <int Start, int End, ProcessorType PT, int NThreads>
  requires AllowedStage<Start> && AllowedStage<End> && (Start <= End)
void run_multiple_stages(cifar_sparse::AppData &data) {
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
void run_gpu_stages(cifar_sparse::AppData &data) {
  // Generate a compile-time sequence for the range [start_stage, end_stage]
  []<std::size_t... I>(std::index_sequence<I...>, cifar_sparse::AppData &data) {
    ((cifar_sparse::vulkan::Singleton::getInstance().run_stage<Start + I>(data)), ...);
  }(std::make_index_sequence<End - Start + 1>{}, data);
}

}  // namespace vulkan
