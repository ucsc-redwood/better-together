#pragma once

#include <cstdlib>
#include <optional>
#include <string_view>

// The BT_VK_* knobs the engine reads, in one place. Defaults and effects are documented in
// docs/instruction-for-ai/03-unit-testing.md and 06-end-to-end-scheduling.md. Knobs are read
// where they are used, never cached at startup: tests change them between engines.
// (BT_VK_INFLIGHT is read by runtime/pipeline.hpp, outside the engine.)
namespace kiss_vk::env {

inline constexpr const char* kValidation = "BT_VK_VALIDATION";   // 1: validation layer
inline constexpr const char* kPhases = "BT_VK_PHASES";           // 1: phase timing
inline constexpr const char* kBarrierAll = "BT_VK_BARRIER_ALL";  // 1: barrier before every dispatch
inline constexpr const char* kForceDescriptorFallback =
    "BT_VK_FORCE_DESCRIPTOR_FALLBACK";                // 1: transient descriptor pools
inline constexpr const char* kQueue = "BT_VK_QUEUE";  // universal | compute
inline constexpr const char* kQueuePriority = "BT_VK_QUEUE_PRIORITY";  // low | medium | high
inline constexpr const char* kWeightsPlacement =
    "BT_VK_WEIGHTS_PLACEMENT";  // default | device_local | uncached
inline constexpr const char* kGpuLocalPlacement =
    "BT_VK_GPU_LOCAL_PLACEMENT";  // default | device_local | uncached (gpu_local class)
inline constexpr const char* kBoundaryPlacement =
    "BT_VK_BOUNDARY_PLACEMENT";  // experiment: default | device_local | uncached (boundary class)
inline constexpr const char* kPipelineCacheDir = "BT_VK_PIPELINE_CACHE_DIR";  // default: "."
inline constexpr const char* kSubgroupSize = "BT_VK_SUBGROUP_SIZE";  // test hook: pin a size

// The variable's value, or nullopt when unset.
[[nodiscard]] inline std::optional<std::string_view> get(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr ? std::optional<std::string_view>(v) : std::nullopt;
}

// True only for the value "1".
[[nodiscard]] inline bool flag(const char* name) {
  const auto v = get(name);
  return v.has_value() && *v == "1";
}

}  // namespace kiss_vk::env
