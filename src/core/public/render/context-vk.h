#pragma once

#include <functional>
#include <map>
#include <mutex>
#include <type_traits>

#include "utils/mixins.h"

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

#define VK_NO_PROTOTYPES
#include <Volk/volk.h>
#include <vma/vk_mem_alloc.h>
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

struct VmaVulkanFunctions;

namespace avk {

inline bool vkCheck(VkResult res) {
  if (res != ::VK_SUCCESS) {
    // TODO
    return false;
  }
  return true;
}

inline VkExternalMemoryHandleTypeFlagBits externalMemoryVkFlags() {
#ifdef AVK_OS_WINDOWS
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#elif AVK_OS_ANDROID
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
         VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
#elif AVK_OS_MACOS
  // TODO insert metal handles
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#elif AVK_OS_LINUX
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
         VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
#else
  return 0;
#endif
}

struct WindowHDRInfo {
  bool hdrEnabled = false;
  bool wideGamutEnabled = false;
  // scale factor to display SDR content on HDR display
  float sdrWhiteLevel = 0.0f;
};

struct ContextVkParams {
  // TODO add more
  WindowHDRInfo hdrInfo;
  bool useHDR = false;
#ifdef AVK_OS_WINDOWS
  HWND window = nullptr;
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
      const char* name = nullptr;
      const auto& val = *it;

      if constexpr (std::is_same_v<std::decay_t<decltype(val)>, const char*>) {
        name = val;
      } else if constexpr (std::is_same_v<std::decay_t<decltype(val)>,
                                          std::string>) {
        name = val.c_str();
      } else if constexpr (std::is_same_v<std::decay_t<decltype(val)>,
                                          std::string_view>) {
        name = val.data();
      } else {
        static_assert(sizeof(val) == 0, "Unsupported string type");
      }

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
  // resolution of the image
  VkExtent2D extent = {};
  // semaphore to wait before updating the image
  VkSemaphore acquireSemaphore = VK_NULL_HANDLE;
  // semaphore to signal after the image has been updated
  VkSemaphore presentSemaphore = VK_NULL_HANDLE;
  // Fence to signal after the image has been updated
  VkFence submissionFence = VK_NULL_HANDLE;
  // factor to scale SDR content on HDR display
  float sdrWhiteLevel = 1.0f;
  // index of current swapchain image
  uint32_t imageIndex = 0;

  inline operator bool() { return image != VK_NULL_HANDLE; }
};

struct InstanceVk {
  VkInstance instance = VK_NULL_HANDLE;
#ifdef AVK_DEBUG
  VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
#endif
  Extensions extensions;
};

// TODO VmaMemoryPool for external memory for images and for external memory
// for buffers

struct DeviceVk : public NonCopyable {
  DeviceVk() = default;
  // rember to update these
  DeviceVk(DeviceVk&&) noexcept;
  DeviceVk& operator=(DeviceVk&&) noexcept;

  static uint32_t constexpr InvalidQueueFamilyIndex = UINT32_MAX;
  static uint32_t constexpr QueueFamilyIndicesCount = 4;

  VmaAllocator vmaAllocator = VK_NULL_HANDLE;
  std::mutex queueMutex;

  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  Extensions extensions;
  std::unique_ptr<DevicePropertiesFeatures> propertiesFeatures = nullptr;

  union Families {
    struct FamiliesStruct {
      uint32_t graphicsCompute;
      uint32_t computeAsync;
      uint32_t transfer;
      uint32_t present;
    } family;
    uint32_t families[4];
  } queueIndices;

  // Vulkan specification mandates that at least one queue family must support
  // graphics and compute
  VkQueue graphicsComputeQueue = VK_NULL_HANDLE;

  // A compute, not graphics queue family signals that there's GPU hardware
  // which can work in parallel with a graphics/ray tracing pipeline on a
  // graphics queue family
  VkQueue computeAsyncQueue = VK_NULL_HANDLE;

  // Transfer only queue to separate transfer and computation/graphics work
  VkQueue transferQueue = VK_NULL_HANDLE;

  // presentation queue family, might be the same as graphicsComputeQueueFamily,
  // but we explicitly avoid using a compute only or transfer queue family for
  // presentation unless necessary
  VkQueue presentQueue = VK_NULL_HANDLE;

  // some bools for optional device extensions
  bool extSwapchainMaintenance1 = false;
  bool extSwapchainColorspace = false;
};

// discard mechanism inspired by blender's GHOST
struct FrameDiscard {
  std::vector<VkSwapchainKHR> swapchains;
  std::vector<VkSemaphore> semaphores;

  FrameDiscard() {
    swapchains.reserve(16);
    semaphores.reserve(16);
  }

  void destroy(DeviceVk const& device);
};

// A command pool needs to be created per frame per thread
struct Frame : public NonCopyable {
  Frame() = default;
  // TODO: multiple frames in flight support
  // fence signaled when previous use of the frame has finished rendering,
  // meaning we can acquire a new image and semaphores can be reused
  VkFence submissionFence = VK_NULL_HANDLE;
  // semaphore to acquire a image. being signaled when image is ready to be
  // updated
  VkSemaphore acquireSemaphore = VK_NULL_HANDLE;

  FrameDiscard discard;

  void destroy(DeviceVk const& device);
};

struct SwapchainImage {
  VkImage image = VK_NULL_HANDLE;
  // signaled when image is ready to be presented
  VkSemaphore presentSemaphore = VK_NULL_HANDLE;

  void destroy(DeviceVk const& device);
};

class ContextVk : public NonMoveable {
 public:
  ContextVk(ContextVkParams const& params);
  ~ContextVk() noexcept;

  ContextResult initializeDrawingContext();
  ContextResult recreateSwapchain(bool useHDR);

  VkFence getFence();
  ContextResult swapBufferRelease();
  ContextResult swapBufferAcquire();
  void setSwapBufferCallbacks(
      std::function<void(const SwapchainDataVk*)> drawCallback,
      std::function<void(void)> acquireCallback,
      std::function<void(VkImage const*, uint32_t, VkFormat, VkExtent2D)>
          recreationCallback);

  inline VmaAllocator getAllocator() const { return m_vmaAllocator; }
  inline InstanceVk const& instance() const { return m_instance; }
  inline DeviceVk const& device() const { return m_device; }
  inline InstanceVk& instance() { return m_instance; }
  inline DeviceVk& device() { return m_device; }

  SwapchainDataVk getSwapchainData() const;

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

  bool initializeFrameData();
  void destroySwapchainPresentFences(VkSwapchainKHR swapchain);
  bool destroySwapchain();

  void setPresentFence(VkSwapchainKHR swapchain, VkFence presentFence);

  static uint32_t constexpr InvalidSwapchainImageIndex = UINT32_MAX;

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
  VmaVulkanFunctions* m_vmaVulkanFunctions = nullptr;
  VmaAllocator m_vmaAllocator;

  // Display data
  VkSurfaceKHR m_surface = VK_NULL_HANDLE;
  VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
  std::vector<Frame> m_frameData;
  std::vector<SwapchainImage> m_swapchainImages;
  std::vector<VkImage> m_vkImages;  // temp storage for swapchain images
  uint32_t m_acquiredSwapchainImageIndex = InvalidSwapchainImageIndex;
  std::unique_ptr<WindowHDRInfo> m_hdrInfo = nullptr;
  bool m_hdrEnabled;
  VkExtent2D m_currentExtent;
  VkSurfaceFormatKHR m_surfaceFormat;

  // callbacks to customize rendering procedure
  std::function<void(const SwapchainDataVk*)> m_swapBufferDrawCallback;
  std::function<void(void)> m_swapBufferAcquiredCallback;
  std::function<void(VkImage const* /*images*/, uint32_t /*num images*/,
                     VkFormat /*format*/, VkExtent2D /*image extent*/)>
      m_swapchainRecreationCallback =
          [](VkImage const*, uint32_t, VkFormat, VkExtent2D) {};

  std::map<VkSwapchainKHR, std::vector<VkFence>> m_presentFences;
  std::vector<VkFence> m_fencePile;

  uint64_t m_renderFrame = 0;
  uint64_t m_imageCount = 0;
};

}  // namespace avk