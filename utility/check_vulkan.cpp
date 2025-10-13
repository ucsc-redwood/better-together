#include <volk.h>

#include <iostream>
#include <vector>

int main() {
  // Initialize Volk
  if (volkInitialize() != VK_SUCCESS) {
    std::cerr << "[X] volkInitialize() failed!" << std::endl;
    return 1;
  }

  uint32_t version = volkGetInstanceVersion();
  std::cout << "[+] Volk initialized. Vulkan version: " << VK_VERSION_MAJOR(version) << "."
            << VK_VERSION_MINOR(version) << "." << VK_VERSION_PATCH(version) << "\n";

  // Vulkan application info
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Vulkan Init Test";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "None";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  // Instance create info (no layers/extensions for now)
  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  // macOS quirk: Vulkan needs VK_KHR_portability_enumeration
  const char* extensions[] = {
      "VK_KHR_surface", "VK_MVK_macos_surface", "VK_KHR_portability_enumeration"};
  createInfo.enabledExtensionCount = 3;
  createInfo.ppEnabledExtensionNames = extensions;
  createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

  // Create the Vulkan instance
  VkInstance instance;
  if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
    std::cerr << "[X] Failed to create Vulkan instance!" << std::endl;
    return 1;
  }

  std::cout << "[+] Vulkan instance created!\n";

  // Load instance-level function pointers
  volkLoadInstance(instance);

  // Enumerate physical devices
  uint32_t gpuCount = 0;
  vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr);
  if (gpuCount == 0) {
    std::cerr << "[X] No Vulkan-compatible GPUs found!" << std::endl;
    vkDestroyInstance(instance, nullptr);
    return 1;
  }

  std::vector<VkPhysicalDevice> devices(gpuCount);
  vkEnumeratePhysicalDevices(instance, &gpuCount, devices.data());

  std::cout << "[+] Found " << gpuCount << " Vulkan-compatible device(s):\n";

  for (uint32_t i = 0; i < gpuCount; ++i) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devices[i], &props);
    std::cout << "  - " << props.deviceName << " (API " << VK_VERSION_MAJOR(props.apiVersion) << "."
              << VK_VERSION_MINOR(props.apiVersion) << "." << VK_VERSION_PATCH(props.apiVersion)
              << ")\n";
  }

  // Clean up
  vkDestroyInstance(instance, nullptr);
  return 0;
}
