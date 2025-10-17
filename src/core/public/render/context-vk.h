#pragma once

#ifdef AVK_OS_WINDOWS
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(AVK_OS_MACOS)  // TODO: also iOS and iPadOS
#define VK_USE_PLATFORM_METAL_EXT
#elif defined(AVK_OS_ANDROID)
#define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(AVK_OS_LINUX)
#if defined(AVK_USE_XCB)
#define VK_USE_PLATFORM_XCB_KHR
#else
#define VK_USE_PLATFORM_WAYLAND_KHR
#endif
#else
#error "ADD SUPPORT"
#endif

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

#ifdef AVK_OS_WINDOWS
#include <WinDef.h>
#elif defined(AVK_OS_MACOS)
#error "TODO"
#elif defined(AVK_OS_ANDROID)
#error "TODO"
#elif defined(AVK_OS_LINUX)
#error "TODO X11 and Wayland"
#else
#error "ADD SUPPORT"
#endif

namespace avk {

struct WindowHDRInfo {
  bool hdrEnabled = false;
  bool wideGamutEnabled = false;
  // scale factor to display SDR content on HDR display
  float sdrWhiteLevel = 0.0f;
};

struct ContextVkParams {
  // TODO add more
  WindowHDRInfo hdrInfo;
#ifdef AVK_OS_WINDOWS
  HWND window;
#elif defined(AVK_OS_MACOS)
#error "TODO"
#elif defined(AVK_OS_ANDROID)
#error "TODO"
#elif defined(AVK_OS_LINUX)
#error "TODO X11 and Wayland"
#else
#error "ADD SUPPORT"
#endif
};

enum class ContextResult {
  Error = 0,
  Success = 1,
};

struct DevicePropertiesFeatures {
  VkPhysicalDeviceFeatures2 features2{};
  VkPhysicalDeviceVulkan11Features features11{};
  VkPhysicalDeviceVulkan12Features features12{};
  VkPhysicalDeviceProperties2 properties2{};
  VkPhysicalDeviceVulkan11Properties properties11{};
  VkPhysicalDeviceVulkan12Properties properties12{};
  VkPhysicalDeviceVulkan13Properties properties13{};
  VkPhysicalDeviceDriverProperties driverProperties{};
};

// kept for links
// struct DeviceProperties {
//   // common properties
//   VkPhysicalDeviceType deviceType;
//
//   // Vulkan 1.1 properties
//   // subgroup has to do with compute shaders, links:
//   // - http://vkguide.dev/docs/gpudriven/compute_shaders/
//   // - https://www.khronos.org/blog/vulkan-subgroup-tutorial
//   // uint32_t subgroupSize; // See 1.3 properties
//   VkShaderStageFlags subgroupSupportedStages;
//   VkSubgroupFeatureFlags subgroupSupportedOperations;
//   VkBool32 subgroupQuadOperationsInAllStages;
//   uint32_t maxPerSetDescriptors;
//   VkDeviceSize maxMemoryAllocationSize;
//
//   // Vulkan 1.2 properties
//   char driverName[VK_MAX_DRIVER_NAME_SIZE];
//   char driverInfo[VK_MAX_DRIVER_INFO_SIZE];
//
//   // Vulkan 1.3 properties
//   uint32_t minSubgroupSize;
//   uint32_t maxSubgroupSize;
//   uint32_t maxComputeWorkgroupSubgroups;
// };

struct Extensions {
  std::vector<VkExtensionProperties> extensions;
  std::vector<char const*> enabled;

  inline bool isSupported(char const* name) const {
    for (VkExtensionProperties const& ext : extensions) {
      if (strcmp(ext.extensionName, name) == 0) {
        return true;
      }
    }
    return false;
  }

  template <typename StrIt>
  inline bool isSupported(StrIt&& beg, StrIt&& end) const {
    for (auto it = beg; it != end; ++it) {
      if (!isSupported(*it)) {
        return false;
      }
    }
    return true;
  }

  inline bool enable(char const* name) {
    bool supported = isSupported(name);
    if (supported) {
      enabled.push_back(name);
      // TODO add logging
      return true;
    }
    // TODO add logging
    return false;
  }

  template <typename StrIt>
  inline bool enable(StrIt&& beg, StrIt&& end) {
    bool failure = false;
    for (auto it = beg; it != end; ++it) {
      char const* name = *it;
      failure |= !enable(name);
    }
    return !failure;
  }

  inline bool isEnabled(char const* name) const {
    for (char const* enabledName : enabled) {
      if (strcmp(enabledName, name) == 0) {
        return true;
      }
    }
    return false;
  }
};

struct SwapchainDataVk {
  // handle to image presented to the user
  VkImage image = VK_NULL_HANDLE;
  // format of the swapchain
  VkSurfaceFormatKHR format = {};
  // color space of the swapchain
  VkColorSpaceKHR colorSpace = {};
  // presentation mode of the swapchain
  VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
  // resolution of the image
  VkExtent2D extent = {};
  // semaphore to wait before updating the image
  VkSemaphore acquireSemaphore = VK_NULL_HANDLE;
  // semaphore to signal after the image has been updated
  VkSemaphore presentSemaphore = VK_NULL_HANDLE;
  // Fence to signal after the image has been updated
  VkFence submissionFence = VK_NULL_HANDLE;
  // factor to scale SDR content on HDR display
  float sdrWhiteLevel = 0.0f;
};

struct InstanceVkExtensionTable {
  // TODO add extension specific function pointers which need to be cached
};

struct InstanceVk {
  VkInstance instance = VK_NULL_HANDLE;
#ifdef AVK_DEBUG
  VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
#endif
  Extensions extensions;
  std::unique_ptr<InstanceVkExtensionTable> extensionTable = nullptr;
};

struct DeviceVkExtensionTable {
  // TODO add extension specific function pointers which need to be cached
  PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR = nullptr;
  PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR = nullptr;
  PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR vkGetPhysicalDeviceSurfaceCapabilities2KHR = nullptr;
  PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR = nullptr;
};

struct DeviceVk {
  static uint32_t constexpr InvalidQueueFamilyIndex = UINT32_MAX;
  static uint32_t constexpr QueueFamilyIndicesCount = 4;

  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  Extensions extensions;
  std::unique_ptr<DevicePropertiesFeatures> propertiesFeatures = nullptr;
  std::unique_ptr<DeviceVkExtensionTable> extensionTable = nullptr;

  // Vulkan specification mandates that at least one queue family must support
  // graphics and compute
  uint32_t graphicsComputeQueueFamily = InvalidQueueFamilyIndex;
  VkQueue graphicsComputeQueue = VK_NULL_HANDLE;

  // A compute, not graphics queue family signals that there's GPU hardware
  // which can work in parallel with a graphics/ray tracing pipeline on a
  // graphics queue family
  uint32_t computeAsyncQueueFamily = InvalidQueueFamilyIndex;
  VkQueue computeAsyncQueue = VK_NULL_HANDLE;

  // Transfer only queue to separate transfer and computation/graphics work
  uint32_t transferQueueFamily = InvalidQueueFamilyIndex;
  VkQueue transferQueue = VK_NULL_HANDLE;

  // presentation queue family, might be the same as graphicsComputeQueueFamily,
  // but we explicitly avoid using a compute only or transfer queue family for
  // presentation unless necessary
  uint32_t presentQueueFamily = InvalidQueueFamilyIndex;
  VkQueue presentQueue = VK_NULL_HANDLE;

  // some bools for optional device extensions
  bool extSwapchainMaintenance1 = false;
};

class ContextVk {
 public:
  ContextVk(ContextVkParams const& params);
  ContextVk(ContextVk const&) = delete;
  ContextVk(ContextVk&&) noexcept = delete;
  ContextVk& operator=(ContextVk const&) = delete;
  ContextVk& operator=(ContextVk&&) noexcept = delete;
  ~ContextVk() noexcept;

  ContextResult initializeDrawingContext();
  ContextResult recreateSwapchain(bool useHDR);

 private:
  bool initInstanceExtensions();
  bool createInstance(uint32_t vulkanApiVersion);
  bool physicalDeviceSupport(DeviceVk& physicalDevice) const;
  std::vector<std::string_view> anyMissingCapabilities(
      DeviceVk& physicalDevice) const;
  bool createSurface();

  bool selectPhysicalDevice(
      std::vector<std::string_view> const& requiredExtensions);
  bool createDevice(std::vector<std::string_view> const& requiredExtensions);

#ifdef AVK_OS_WINDOWS
  HWND m_hWindow;
#elif defined(AVK_OS_MACOS)
#error "TODO"
#elif defined(AVK_OS_ANDROID)
#error "TODO"
#elif defined(AVK_OS_LINUX)
#error "TODO X11 and Wayland"
#else
#error "ADD SUPPORT"
#endif

  // Instance level data
  InstanceVk m_instance;
  DeviceVk m_device;

  // Display data
  VkSurfaceKHR m_surface = VK_NULL_HANDLE;
  VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
  SwapchainDataVk m_swapchainData;
};

}  // namespace avk