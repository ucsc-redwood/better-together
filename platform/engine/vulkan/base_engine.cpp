#include "base_engine.hpp"

#include <spdlog/spdlog.h>

#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

// ----------------------------------------------------------------------------
// Global variables
// ----------------------------------------------------------------------------

VmaAllocator g_vma_allocator = nullptr;

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace kiss_vk {

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------

BaseEngine::BaseEngine(const bool enable_validation_layer)
    : compute_queue_family_index_(std::numeric_limits<uint32_t>::max()) {
  initialize_dynamic_loader();

  if (enable_validation_layer) {
    request_validation_layer();
  }

  // A constructor that throws never runs ~BaseEngine, so a failure after create_instance()
  // (e.g. no iGPU, or a missing required feature) would leak the instance/device. Release
  // what exists, then rethrow.
  try {
    create_instance();
    create_physical_device();
    create_device();

    initialize_vma_allocator();
    owns_vma_allocator_ = true;
  } catch (...) {
    destroy();
    throw;
  }
}

// ----------------------------------------------------------------------------
// Destructor
// ----------------------------------------------------------------------------

BaseEngine::~BaseEngine() { destroy(); }

void BaseEngine::destroy() noexcept {
  // Tear down explicitly and in order. Previously only the VMA allocator was
  // destroyed; the vk::Device and vk::Instance were leaked, so the Vulkan loader's
  // own static/atexit teardown raced and segfaulted on exit on Tegra (§9) — after the
  // results were already flushed. By the time this base dtor runs, the derived
  // Engine's memory resource (buffers) and the dispatcher's sequences/algorithms
  // (command/descriptor pools) are already destroyed (declaration/destruction order),
  // so draining and destroying the device + instance here is safe.
  if (device_) {
    // waitIdle can throw (vulkan-hpp), which in a destructor would std::terminate.
    try {
      device_.waitIdle();
    } catch (const std::exception& e) {
      spdlog::warn("BaseEngine teardown: device.waitIdle() failed: {}", e.what());
    }
  }
  if (owns_vma_allocator_ && g_vma_allocator) {
    vmaDestroyAllocator(g_vma_allocator);
    owns_vma_allocator_ = false;
    // Null the global so a second BaseEngine destruction (or any later use via
    // vma_pmr) can't double-free / use-after-free this handle. NOTE: g_vma_allocator
    // is still a single global shared across engines, so two BaseEngines must not
    // coexist; this guard makes the common create→destroy→create sequence safe.
    g_vma_allocator = nullptr;
  }
  if (device_) {
    device_.destroy();
    device_ = nullptr;
  }
  if (instance_) {
    instance_.destroy();
    instance_ = nullptr;
  }
}

// ----------------------------------------------------------------------------
// Initialize dynamic loader
// ----------------------------------------------------------------------------

void BaseEngine::initialize_dynamic_loader() {
  dl_ = DynamicLoader();

  // Load the Vulkan library
  vkGetInstanceProcAddr_ = dl_.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
  if (!vkGetInstanceProcAddr_) {
    throw std::runtime_error("Failed to load vkGetInstanceProcAddr!");
  }

  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr_);
}

// ----------------------------------------------------------------------------
// Instance
// ----------------------------------------------------------------------------

void BaseEngine::create_instance() {
  constexpr vk::ApplicationInfo appInfo{
      .pApplicationName = "Vulkan Compute Example",
      .applicationVersion = 1,
      .pEngineName = "No Engine",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_3,
  };

  const vk::InstanceCreateInfo instance_create_info{
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(enabled_layers_.size()),
      .ppEnabledLayerNames = enabled_layers_.data(),
  };

  std::string layers;
  for (const char* layer : enabled_layers_)
    layers += (layers.empty() ? "" : ", ") + std::string(layer);
  spdlog::debug("Vulkan instance layers enabled: {}", layers.empty() ? "none" : layers);

  // Create the instance using the default dispatcher
  instance_ = createInstance(instance_create_info);

  // Initialize instance-specific function pointers
  VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);
}

// ----------------------------------------------------------------------------
// Validation layer
// ----------------------------------------------------------------------------

void BaseEngine::request_validation_layer() {
  constexpr auto validationLayerName = "VK_LAYER_KHRONOS_validation";

  const auto availableLayers = vk::enumerateInstanceLayerProperties();

  spdlog::debug("Available Vulkan Layers:");
  for (const auto& layer : availableLayers) {
    spdlog::debug("- {}", layer.layerName.data());
  }

  bool layerFound = std::ranges::any_of(availableLayers, [validationLayerName](const auto& layer) {
    return std::strcmp(layer.layerName.data(), validationLayerName) == 0;
  });

  if (!layerFound) {
    spdlog::warn(
        "Validation layer requested but not available, continuing without "
        "it...");
    return;
  }

  enabled_layers_.push_back(validationLayerName);
}

// ----------------------------------------------------------------------------
// Physical device
// ----------------------------------------------------------------------------

void BaseEngine::create_physical_device(vk::PhysicalDeviceType type) {
  if (!instance_) {
    throw std::runtime_error("Instance is not valid");
  }

  // Get all physical devices
  const auto physicalDevices = instance_.enumeratePhysicalDevices();
  if (physicalDevices.empty()) {
    throw std::runtime_error("No Vulkan-compatible physical devices found.");
  }

  // Try to find an integrated GPU
  const auto integrated_gpu = std::ranges::find_if(physicalDevices, [type](const auto& device) {
    return device.getProperties().deviceType == type;
  });

  if (integrated_gpu == physicalDevices.end()) {
    // Surface WHAT was enumerated so the failure is diagnosable (the discrete-GPU
    // build box hits this) instead of an opaque message (review #13).
    for (const auto& dev : physicalDevices) {
      const auto props = dev.getProperties();
      spdlog::error("  available device: {} (type {})",
                    props.deviceName.data(),
                    vk::to_string(props.deviceType));
    }
    throw std::runtime_error(
        "No integrated GPU found (kiss-vk requires an iGPU; see the "
        "enumerated devices above)");
  }

  physical_device_ = *integrated_gpu;

  spdlog::info("Using integrated GPU: {}", physical_device_.getProperties().deviceName.data());
}

bool has_integrated_gpu() noexcept {
  // Constructing a throwaway engine reuses the exact loader+enumeration path: it succeeds
  // only if an integrated GPU was selected and has every required feature, and throws
  // "No integrated GPU found", "... lacks required Vulkan features: ..." (or a loader
  // error) otherwise. Probe once and cache; validation layer off to stay quiet.
  static const bool available = [] {
    try {
      BaseEngine probe(/*enable_validation_layer=*/false);
      return true;
    } catch (...) {
      return false;
    }
  }();
  return available;
}

// ----------------------------------------------------------------------------
// Device
// ----------------------------------------------------------------------------

// The device features the engine requires -- exactly what the committed shaders' SPIR-V
// capabilities (Int8, StorageBuffer8BitAccess), the VMA allocator
// (VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT) and LocalSizeId (the SPIR-V 1.6 form of
// `local_size_*_id`, VUID-RuntimeSpirv-LocalSizeId-06434) need, plus synchronization2 and
// timelineSemaphore for the Sequence's barriers, timestamps and tickets. Only these are
// enabled.
namespace {
template <typename Features>
struct RequiredFeature {
  const char* name = nullptr;
  vk::Bool32 Features::* member = nullptr;
};

constexpr std::array<RequiredFeature<vk::PhysicalDeviceVulkan12Features>, 4>
    kRequiredVulkan12Features{{
        {"storageBuffer8BitAccess", &vk::PhysicalDeviceVulkan12Features::storageBuffer8BitAccess},
        {"shaderInt8", &vk::PhysicalDeviceVulkan12Features::shaderInt8},
        {"bufferDeviceAddress", &vk::PhysicalDeviceVulkan12Features::bufferDeviceAddress},
        // The Sequence's per-submission tickets (design D3).
        {"timelineSemaphore", &vk::PhysicalDeviceVulkan12Features::timelineSemaphore},
    }};

constexpr std::array<RequiredFeature<vk::PhysicalDeviceVulkan13Features>, 2>
    kRequiredVulkan13Features{{
        {"maintenance4", &vk::PhysicalDeviceVulkan13Features::maintenance4},
        // vkCmdPipelineBarrier2 / vkCmdWriteTimestamp2 (Sequence, design A1).
        {"synchronization2", &vk::PhysicalDeviceVulkan13Features::synchronization2},
    }};
}  // namespace

std::vector<std::string> missing_required_features(
    const vk::PhysicalDeviceVulkan12Features& supported_12,
    const vk::PhysicalDeviceVulkan13Features& supported_13) {
  std::vector<std::string> missing;
  for (const auto& f : kRequiredVulkan12Features) {
    if (supported_12.*f.member != VK_TRUE) missing.emplace_back(f.name);
  }
  for (const auto& f : kRequiredVulkan13Features) {
    if (supported_13.*f.member != VK_TRUE) missing.emplace_back(f.name);
  }
  return missing;
}

void BaseEngine::create_device() {
  if (!physical_device_) {
    throw std::runtime_error("Physical device is not valid");
  }

  // Queue family (design C1). Every family is logged; BT_VK_QUEUE=universal (default, today's
  // choice: the first family with compute) or compute (the first with compute and no
  // graphics, falling back with a warning where none exists).
  const auto families = physical_device_.getQueueFamilyProperties();
  for (uint32_t i = 0; i < families.size(); ++i) {
    spdlog::debug("Queue family {}: flags {}, queueCount {}, timestampValidBits {}",
                  i,
                  vk::to_string(families[i].queueFlags),
                  families[i].queueCount,
                  families[i].timestampValidBits);
  }
  const auto first_family = [&](const auto& pred) -> std::optional<uint32_t> {
    for (uint32_t i = 0; i < families.size(); ++i) {
      if (pred(families[i].queueFlags)) return i;
    }
    return std::nullopt;
  };
  const auto universal = first_family(
      [](const vk::QueueFlags f) { return static_cast<bool>(f & vk::QueueFlagBits::eCompute); });
  if (!universal) throw std::runtime_error("No queue family supports compute operations.");
  compute_queue_family_index_ = *universal;

  const char* queue_env = std::getenv("BT_VK_QUEUE");
  const std::string_view queue_kind = queue_env != nullptr ? queue_env : "universal";
  if (queue_kind != "universal" && queue_kind != "compute") {
    throw std::runtime_error("BT_VK_QUEUE must be universal or compute, got " +
                             std::string(queue_kind));
  }
  if (queue_kind == "compute") {
    if (const auto compute_only = first_family([](const vk::QueueFlags f) {
          return (f & vk::QueueFlagBits::eCompute) && !(f & vk::QueueFlagBits::eGraphics);
        })) {
      compute_queue_family_index_ = *compute_only;
    } else {
      spdlog::warn("BT_VK_QUEUE=compute: no compute-only queue family; using family {} ({})",
                   compute_queue_family_index_,
                   vk::to_string(families[compute_queue_family_index_].queueFlags));
    }
  }

  // Create a logical device with a compute queue
  constexpr float queuePriority = 1.0f;
  vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
      .queueFamilyIndex = compute_queue_family_index_,
      .queueCount = 1,
      .pQueuePriorities = &queuePriority,
  };

  // Subgroup size control is core 1.3; its structs must not be chained on an older device.
  const auto device_props = physical_device_.getProperties();
  const bool is_vulkan_13 = device_props.apiVersion >= VK_API_VERSION_1_3;

  // Query -> check -> enable a separate minimal struct (never pass the queried struct:
  // that would enable every supported feature).
  vk::PhysicalDeviceVulkan13Features supported_13{};
  vk::PhysicalDeviceVulkan12Features supported_12{.pNext = is_vulkan_13 ? &supported_13 : nullptr};
  vk::PhysicalDeviceFeatures2 supported{.pNext = &supported_12};
  physical_device_.getFeatures2(&supported);

  // Not 1.3: supported_13 stays zeroed, so the 1.3 features are reported missing.
  if (const auto missing = missing_required_features(supported_12, supported_13);
      !missing.empty()) {
    std::string names;
    for (const auto& name : missing) names += (names.empty() ? "" : ", ") + name;
    throw std::runtime_error(std::string(physical_device_.getProperties().deviceName.data()) +
                             " lacks required Vulkan features: " + names);
  }

  // Subgroup size (design D5): the device's default size, pinned on subgroup-dependent
  // pipelines when subgroupSizeControl covers COMPUTE (VUID-...-pNext-02755 needs both).
  vk::PhysicalDeviceSubgroupSizeControlProperties size_control{};
  vk::PhysicalDeviceSubgroupProperties subgroup{.pNext = is_vulkan_13 ? &size_control : nullptr};
  vk::PhysicalDeviceProperties2 props2{.pNext = &subgroup};
  physical_device_.getProperties2(&props2);

  const bool pinnable =
      supported_13.subgroupSizeControl == VK_TRUE &&
      (size_control.requiredSubgroupSizeStages & vk::ShaderStageFlagBits::eCompute);
  const bool full_subgroups = supported_13.computeFullSubgroups == VK_TRUE;
  compute_limits_ = ComputeLimits{
      .subgroup_size = subgroup.subgroupSize,
      .min_subgroup_size = is_vulkan_13 ? size_control.minSubgroupSize : subgroup.subgroupSize,
      .max_subgroup_size = is_vulkan_13 ? size_control.maxSubgroupSize : subgroup.subgroupSize,
      .subgroup_size_pinned = pinnable,
      .full_subgroups = full_subgroups,
      .max_compute_workgroup_subgroups = pinnable ? size_control.maxComputeWorkgroupSubgroups : 0,
      .max_work_group_size = device_props.limits.maxComputeWorkGroupSize,
      .max_work_group_invocations = device_props.limits.maxComputeWorkGroupInvocations,
  };
  spdlog::debug(
      "Subgroup size {} ({}): device default {}, min {}, max {}, subgroupSizeControl {}, "
      "requiredSubgroupSizeStages {}, maxComputeWorkgroupSubgroups {}",
      compute_limits_.subgroup_size,
      pinnable ? "pinned" : "not pinned",
      subgroup.subgroupSize,
      size_control.minSubgroupSize,
      size_control.maxSubgroupSize,
      supported_13.subgroupSizeControl == VK_TRUE,
      vk::to_string(size_control.requiredSubgroupSizeStages),
      size_control.maxComputeWorkgroupSubgroups);

  // Optional features are enabled only when supported (design D3).
  vk::PhysicalDeviceVulkan13Features enabled_13{.subgroupSizeControl = pinnable,
                                                .computeFullSubgroups = full_subgroups};
  vk::PhysicalDeviceVulkan12Features enabled_12{.pNext = &enabled_13};
  for (const auto& f : kRequiredVulkan12Features) enabled_12.*f.member = VK_TRUE;
  for (const auto& f : kRequiredVulkan13Features) enabled_13.*f.member = VK_TRUE;
  spdlog::debug(
      "Vulkan 1.3 features: synchronization2 supported {} enabled {}, computeFullSubgroups "
      "supported {} enabled {}",
      supported_13.synchronization2 == VK_TRUE,
      enabled_13.synchronization2 == VK_TRUE,
      supported_13.computeFullSubgroups == VK_TRUE,
      enabled_13.computeFullSubgroups == VK_TRUE);

  // VK_KHR_push_descriptor is optional (design D6): enabled when the device has it, unless
  // BT_VK_FORCE_DESCRIPTOR_FALLBACK=1 asks for the transient-pool path.
  const auto device_extensions = physical_device_.enumerateDeviceExtensionProperties();
  const auto has_extension = [&](const char* name) {
    return std::ranges::any_of(device_extensions, [name](const auto& ext) {
      return std::strcmp(ext.extensionName.data(), name) == 0;
    });
  };
  const bool has_push_descriptor = has_extension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  const char* force_env = std::getenv("BT_VK_FORCE_DESCRIPTOR_FALLBACK");
  const bool force_fallback = force_env != nullptr && std::string_view(force_env) == "1";
  const bool use_push_descriptor = has_push_descriptor && !force_fallback;
  std::vector<const char*> extensions;
  if (use_push_descriptor) extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

  // Calibrated timestamps (design D1) correlate the GPU's timestamps with the host's
  // CLOCK_MONOTONIC for the phase timing; optional, KHR or the older EXT.
  const char* calibrated_extension = has_extension(VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)
                                         ? VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME
                                     : has_extension(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)
                                         ? VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME
                                         : nullptr;
  bool calibrated_domains = false;
  if (calibrated_extension != nullptr) {
    const bool khr =
        calibrated_extension == std::string_view(VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
    const auto get_domains = reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR>(
        instance_.getProcAddr(khr ? "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR"
                                  : "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"));
    uint32_t n = 0;
    std::vector<VkTimeDomainKHR> domains;
    if (get_domains != nullptr && get_domains(physical_device_, &n, nullptr) == VK_SUCCESS) {
      domains.resize(n);
      get_domains(physical_device_, &n, domains.data());
    }
    calibrated_domains =
        std::ranges::find(domains, VK_TIME_DOMAIN_DEVICE_KHR) != domains.end() &&
        std::ranges::find(domains, VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR) != domains.end();
    if (calibrated_domains) extensions.push_back(calibrated_extension);
  }

  // Global queue priority (design C1), best-effort: BT_VK_QUEUE_PRIORITY=low|medium|high
  // through VK_KHR_global_priority (or the EXT), warning and continuing at the default
  // priority when the device lacks it or the driver refuses it (VK_ERROR_NOT_PERMITTED).
  std::optional<vk::QueueGlobalPriorityKHR> priority;
  std::string_view priority_name = "default";
  if (const char* v = std::getenv("BT_VK_QUEUE_PRIORITY")) {
    priority_name = v;
    if (priority_name == "low") {
      priority = vk::QueueGlobalPriorityKHR::eLow;
    } else if (priority_name == "medium") {
      priority = vk::QueueGlobalPriorityKHR::eMedium;
    } else if (priority_name == "high") {
      priority = vk::QueueGlobalPriorityKHR::eHigh;
    } else {
      throw std::runtime_error("BT_VK_QUEUE_PRIORITY must be low, medium or high, got " +
                               std::string(priority_name));
    }
  }
  const char* priority_extension =
      has_extension(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME)   ? VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME
      : has_extension(VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME) ? VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME
                                                             : nullptr;
  if (priority && priority_extension == nullptr) {
    spdlog::warn(
        "BT_VK_QUEUE_PRIORITY={}: the device has no global queue priority extension; "
        "using the default priority",
        priority_name);
    priority.reset();
    priority_name = "default";
  }
  const vk::DeviceQueueGlobalPriorityCreateInfoKHR priority_info{
      .globalPriority = priority.value_or(vk::QueueGlobalPriorityKHR::eMedium),
  };

  const auto create = [&](const bool with_priority) {
    std::vector<const char*> exts = extensions;
    if (with_priority) exts.push_back(priority_extension);
    deviceQueueCreateInfo.pNext = with_priority ? &priority_info : nullptr;
    const vk::DeviceCreateInfo deviceCreateInfo{
        .pNext = &enabled_12,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &deviceQueueCreateInfo,
        .enabledExtensionCount = static_cast<uint32_t>(exts.size()),
        .ppEnabledExtensionNames = exts.data(),
    };
    return physical_device_.createDevice(deviceCreateInfo);
  };
  try {
    device_ = create(priority.has_value());
  } catch (const vk::SystemError& e) {
    if (!priority || e.code().value() != static_cast<int>(VK_ERROR_NOT_PERMITTED_KHR)) throw;
    spdlog::warn(
        "BT_VK_QUEUE_PRIORITY={}: the driver refused that global queue priority "
        "(VK_ERROR_NOT_PERMITTED); using the default priority",
        priority_name);
    priority_name = "default";
    device_ = create(false);
  }
  spdlog::debug("Compute queue: family {} ({}, BT_VK_QUEUE={}), priority {}",
                compute_queue_family_index_,
                vk::to_string(families[compute_queue_family_index_].queueFlags),
                queue_kind,
                priority_name);

  if (calibrated_domains) {
    calibrated_timestamps_fn_ =
        reinterpret_cast<PFN_vkGetCalibratedTimestampsKHR>(device_.getProcAddr(
            calibrated_extension == std::string_view(VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)
                ? "vkGetCalibratedTimestampsKHR"
                : "vkGetCalibratedTimestampsEXT"));
  }
  spdlog::debug(
      "Calibrated timestamps: {}",
      calibrated_timestamps_fn_ != nullptr
          ? std::string(calibrated_extension) + " (device + CLOCK_MONOTONIC)"
      : calibrated_extension != nullptr
          ? std::string(calibrated_extension) + " lacks the device/CLOCK_MONOTONIC domains"
          : std::string("not supported; GPU and host phases stay uncorrelated"));

  if (use_push_descriptor) {
    // From the device, not the instance-level dispatcher: an extension command.
    push_descriptor_fn_ = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
        device_.getProcAddr("vkCmdPushDescriptorSetKHR"));
    if (!push_descriptor_fn_) throw std::runtime_error("vkCmdPushDescriptorSetKHR not found");
  }
  spdlog::debug("Descriptor binding: {}",
                use_push_descriptor ? "push descriptors (VK_KHR_push_descriptor)"
                : force_fallback
                    ? "transient descriptor pool (forced by BT_VK_FORCE_DESCRIPTOR_FALLBACK=1)"
                    : "transient descriptor pool (VK_KHR_push_descriptor not supported)");

  // Memory placement (design B1): every heap and memory type, for the per-device tables.
  const auto mem = physical_device_.getMemoryProperties();
  for (uint32_t h = 0; h < mem.memoryHeapCount; ++h) {
    spdlog::debug("Memory heap {}: {} MiB, flags {}",
                  h,
                  mem.memoryHeaps[h].size >> 20,
                  vk::to_string(mem.memoryHeaps[h].flags));
  }
  for (uint32_t t = 0; t < mem.memoryTypeCount; ++t) {
    spdlog::debug("Memory type {}: heap {}, flags {}",
                  t,
                  mem.memoryTypes[t].heapIndex,
                  vk::to_string(mem.memoryTypes[t].propertyFlags));
  }

  // looks like I don't need this
  compute_queue_ = device_.getQueue(compute_queue_family_index_, 0);
}

void BaseEngine::check_calibration() {
  if (calibrated_timestamps_fn_ == nullptr) return;
  // A queue timestamp written between two calibration pairs must lie between their device
  // ticks. RADV 25.2.7 on the 780M (PHOENIX) puts it ~2.9e9 ticks (29 s) later, a constant
  // offset between the two counters; mapping through it would make every latency garbage.
  const uint32_t valid_bits = queue_family_timestamp_bits();
  if (valid_bits == 0) return;
  const vk::QueryPool query = device_.createQueryPool(
      vk::QueryPoolCreateInfo{.queryType = vk::QueryType::eTimestamp, .queryCount = 1});
  const vk::CommandPool pool = device_.createCommandPool(
      vk::CommandPoolCreateInfo{.queueFamilyIndex = compute_queue_family_index_});
  const vk::Fence fence = device_.createFence({});
  const vk::CommandBuffer cmd = device_
                                    .allocateCommandBuffers(vk::CommandBufferAllocateInfo{
                                        .commandPool = pool,
                                        .level = vk::CommandBufferLevel::ePrimary,
                                        .commandBufferCount = 1,
                                    })
                                    .front();
  cmd.begin(vk::CommandBufferBeginInfo{});
  cmd.resetQueryPool(query, 0, 1);
  cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, query, 0);
  cmd.end();

  uint64_t before = 0, after = 0, host = 0, deviation = 0, ts = 0;
  bool ok = calibrate(before, host, deviation);
  compute_queue_.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &cmd}, fence);
  ok = ok && device_.waitForFences(1, &fence, VK_TRUE, UINT64_MAX) == vk::Result::eSuccess;
  ok = ok && calibrate(after, host, deviation);
  ok = ok &&
       device_.getQueryPoolResults(query,
                                   0,
                                   1,
                                   sizeof(ts),
                                   &ts,
                                   sizeof(ts),
                                   vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait) ==
           vk::Result::eSuccess;
  device_.destroyFence(fence);
  device_.destroyCommandPool(pool);
  device_.destroyQueryPool(query);

  const uint64_t mask = valid_bits >= 64 ? ~uint64_t{0} : (uint64_t{1} << valid_bits) - 1;
  ts &= mask;
  if (ok && ts >= (before & mask) && ts <= (after & mask)) return;
  spdlog::warn(
      "Calibrated timestamps: a queue timestamp ({}) lies outside the calibrated device ticks "
      "around it ({}..{}, offset {} ticks); GPU and host phases stay uncorrelated",
      ts,
      before & mask,
      after & mask,
      static_cast<int64_t>(ts - (before & mask)));
  calibrated_timestamps_fn_ = nullptr;
}

uint32_t BaseEngine::queue_family_timestamp_bits() const {
  return physical_device_.getQueueFamilyProperties()[compute_queue_family_index_]
      .timestampValidBits;
}

bool BaseEngine::calibrate(uint64_t& device_ticks,
                           uint64_t& host_ns,
                           uint64_t& deviation_ns) const {
  if (calibrated_timestamps_fn_ == nullptr) return false;
  const std::array<VkCalibratedTimestampInfoKHR, 2> infos{{
      {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
       .pNext = nullptr,
       .timeDomain = VK_TIME_DOMAIN_DEVICE_KHR},
      {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
       .pNext = nullptr,
       .timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR},
  }};
  std::array<uint64_t, 2> stamps{};
  if (calibrated_timestamps_fn_(device_, 2, infos.data(), stamps.data(), &deviation_ns) !=
      VK_SUCCESS) {
    return false;
  }
  device_ticks = stamps[0];
  host_ns = stamps[1];
  return true;
}

void BaseEngine::override_subgroup_size(const uint32_t size) {
  const auto& l = compute_limits_;
  if (!l.subgroup_size_pinned || std::popcount(size) != 1 || size < l.min_subgroup_size ||
      size > l.max_subgroup_size) {
    throw std::runtime_error(fmt::format(
        "Subgroup size {} cannot be pinned on this device (pinning {}, supported {}..{})",
        size,
        l.subgroup_size_pinned ? "available" : "unavailable",
        l.min_subgroup_size,
        l.max_subgroup_size));
  }
  spdlog::debug("Subgroup size {} (pinned, overrides device default {})", size, l.subgroup_size);
  compute_limits_.subgroup_size = size;
}

// ----------------------------------------------------------------------------
// VMA allocator
// ----------------------------------------------------------------------------

void BaseEngine::initialize_vma_allocator() const {
  SPDLOG_TRACE("BaseEngine::initialize_vma_allocator");

  if (!physical_device_ || !device_ || !instance_) {
    throw std::runtime_error("Physical device, device, or instance is not valid");
  }

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#endif
  const VmaVulkanFunctions vulkan_functions{
      .vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
      .vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr,
  };
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

  const VmaAllocatorCreateInfo vma_allocator_create_info{
      .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice = physical_device_,
      .device = device_,
      .preferredLargeHeapBlockSize = 0,  // Let VMA use default size
      .pAllocationCallbacks = nullptr,
      .pDeviceMemoryCallbacks = nullptr,
      .pHeapSizeLimit = nullptr,
      .pVulkanFunctions = &vulkan_functions,
      // .pVulkanFunctions = nullptr,
      .instance = instance_,
      .vulkanApiVersion = VK_API_VERSION_1_3,
      .pTypeExternalMemoryHandleTypes = nullptr};

  if (vmaCreateAllocator(&vma_allocator_create_info, &g_vma_allocator) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create VMA allocator");
  }
}

}  // namespace kiss_vk
