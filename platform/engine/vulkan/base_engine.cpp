#include "base_engine.hpp"

#include <spdlog/spdlog.h>

#include <array>
#include <limits>
#include <string>
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
// capabilities (Int8, StorageBuffer8BitAccess) and the VMA allocator
// (VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT) need. Only these are enabled.
namespace {
struct RequiredFeature {
  const char* name;
  vk::Bool32 vk::PhysicalDeviceVulkan12Features::* member;
};

constexpr std::array<RequiredFeature, 3> kRequiredVulkan12Features{{
    {"storageBuffer8BitAccess", &vk::PhysicalDeviceVulkan12Features::storageBuffer8BitAccess},
    {"shaderInt8", &vk::PhysicalDeviceVulkan12Features::shaderInt8},
    {"bufferDeviceAddress", &vk::PhysicalDeviceVulkan12Features::bufferDeviceAddress},
}};
}  // namespace

std::vector<std::string> missing_required_features(
    const vk::PhysicalDeviceVulkan12Features& supported) {
  std::vector<std::string> missing;
  for (const auto& f : kRequiredVulkan12Features) {
    if (supported.*f.member != VK_TRUE) missing.emplace_back(f.name);
  }
  return missing;
}

void BaseEngine::create_device(vk::QueueFlags queue_flags) {
  if (!physical_device_) {
    throw std::runtime_error("Physical device is not valid");
  }

  auto queueFamilyProperties = physical_device_.getQueueFamilyProperties();

  compute_queue_family_index_ = std::distance(
      queueFamilyProperties.begin(),
      std::find_if(queueFamilyProperties.begin(),
                   queueFamilyProperties.end(),
                   [queue_flags](const auto& qfp) { return qfp.queueFlags & queue_flags; }));

  if (compute_queue_family_index_ == queueFamilyProperties.size()) {
    throw std::runtime_error("No queue family supports compute operations.");
  }

  // Create a logical device with a compute queue
  constexpr float queuePriority = 1.0f;
  const vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
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

  if (const auto missing = missing_required_features(supported_12); !missing.empty()) {
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
  compute_limits_ = ComputeLimits{
      .subgroup_size = subgroup.subgroupSize,
      .subgroup_size_pinned = pinnable,
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
  vk::PhysicalDeviceVulkan13Features enabled_13{.subgroupSizeControl = pinnable};
  vk::PhysicalDeviceVulkan12Features enabled_12{.pNext = pinnable ? &enabled_13 : nullptr};
  for (const auto& f : kRequiredVulkan12Features) enabled_12.*f.member = VK_TRUE;

  const vk::DeviceCreateInfo deviceCreateInfo{
      .pNext = &enabled_12,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &deviceQueueCreateInfo,
  };

  device_ = physical_device_.createDevice(deviceCreateInfo);

  // looks like I don't need this
  compute_queue_ = device_.getQueue(compute_queue_family_index_, 0);
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
