#pragma once
// ---------------------------------------------------------------------------
// runtime/app_traits -- the compiler-checked contract for the concurrent runtime
// test, replacing the old pre-#include magic-typedef ODR contract (the includer
// used to #define DispatcherT/AppDataT/QueueT/... by bare name before pulling in
// pipeline_common.hpp). A runtime-test cell now specializes AppTraits and the
// BtRuntimeApp concept turns a wrong/missing field into a NAMED diagnostic.
//
// Keyed on the DISPATCHER type, because that is the only type that uniquely
// identifies an (app, backend) cell: tree's OMP and CUDA cells SHARE
// tree::SafeAppData, and all three of cifar's backends SHARE one cifar_*::AppData,
// so AppData cannot be the key. Each Dispatcher (OmpStubDispatcher<App>,
// <app>::cuda::CudaDispatcher, <app>::vulkan::VulkanDispatcher) is distinct.
//
//   using AppData         -- the concrete pooled type (host SafeAppData or a UMA subclass)
//   using Queue           -- the SPSC queue of AppData* for this cell
//   kNumStages            -- stage count (app-level)
//   kPoolSize             -- items in flight
//   kNumToProcess         -- items pushed through the ring
//   kGpuExecModel         -- the ExecutionModel that IS "the GPU" for this cell
//                            (kVulkan / kCuda; an unused value for an OMP-only cell)
//   static omp_dispatch(cores, n, AppData&, start, end)
//                         -- forward to this app's OMP multi-stage dispatcher
// ---------------------------------------------------------------------------

#include <concepts>
#include <cstddef>

#include "builtin-apps/pipeline/schedule.hpp"  // ExecutionModel

template <class Dispatcher>
struct AppTraits;  // primary template intentionally left undefined

template <class D>
concept BtRuntimeApp = requires {
  typename AppTraits<D>::AppData;
  typename AppTraits<D>::Queue;
  { AppTraits<D>::kNumStages } -> std::convertible_to<int>;
  { AppTraits<D>::kPoolSize } -> std::convertible_to<std::size_t>;
  { AppTraits<D>::kNumToProcess } -> std::convertible_to<std::size_t>;
  { AppTraits<D>::kGpuExecModel } -> std::convertible_to<ExecutionModel>;
};
