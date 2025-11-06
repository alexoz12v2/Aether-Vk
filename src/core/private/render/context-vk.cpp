
// TODO only if sanitizers are enabled (only on clang), then ignore them
// TODO Apply per function -> Every function calling vulkan must not use cfi
#include "render/context-vk.h"

#include <vulkan/vulkan_core.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <unordered_set>
#include <utility>

#include "utils/integer.h"


#ifdef AVK_OS_WINDOWS
#  include <Windows.h>
// https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute
#  include <dwmapi.h>
#elif defined(AVK_OS_MACOS)
#  error "TODO"
#elif defined(AVK_OS_ANDROID)
#  error "TODO"
#elif defined(AVK_OS_LINUX)
#  error "TODO X11 and Wayland"
#else
#  error "ADD SUPPORT"
#endif

// TODO better logging
#ifdef AVK_DEBUG
static VkBool32 VKAPI_PTR
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageTypes,
              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
              [[maybe_unused]] void* pUserData) {
  if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    std::cout << pCallbackData->messageIdNumber
              << " Validation Layer: Error: " << pCallbackData->pMessageIdName
              << ": " << pCallbackData->pMessage << std::endl;
  } else if (messageSeverity &
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    std::cout << pCallbackData->messageIdNumber
              << " Validation Layer: Warning: " << pCallbackData->pMessageIdName
              << ": " << pCallbackData->pMessage << std::endl;
  } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    std::cout << pCallbackData->messageIdNumber
              << " Validation Layer: Information: "
              << pCallbackData->pMessageIdName << ": "
              << pCallbackData->pMessage << std::endl;
  } else if (messageTypes & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) {
    std::cout << pCallbackData->messageIdNumber
              << " Validation Layer: Performance warning: "
              << pCallbackData->pMessageIdName << ": "
              << pCallbackData->pMessage << std::endl;
  } else if (messageSeverity &
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
    std::cout << pCallbackData->messageIdNumber
              << " Validation Layer: Verbose: " << pCallbackData->pMessageIdName
              << ": " << pCallbackData->pMessage << std::endl;
  }
  return VK_FALSE;
}
#endif

// TODO: Maybe in Debug only?
static void VKAPI_PTR allocateDeviceMemoryCallback(
    VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory,
    VkDeviceSize size, [[maybe_unused]] void* pUserData) {
  std::cout << "VMA Allocation (" << allocator
            << "): memoryType: " << memoryType << ", memory: " << memory
            << " size: " << size << std::endl;
}

/// Callback function called before vkFreeMemory.
static void VKAPI_PTR freeDeviceMemoryCallback(
    VmaAllocator allocator, uint32_t memoryType, VkDeviceMemory memory,
    VkDeviceSize size, [[maybe_unused]] void* pUserData) {
  std::cout << "VMA Free     (" << allocator << "): memoryType: " << memoryType
            << ", memory: " << memory << " size: " << size << std::endl;
}

namespace avk {

#if 0
DeviceVk::DeviceVk(DeviceVk&& other) noexcept {
  std::lock_guard<std::mutex> lk{other.queueMutex};
  vmaAllocator = std::exchange(vmaAllocator, other.vmaAllocator);
  physicalDevice = std::exchange(physicalDevice, other.physicalDevice);
  device = std::exchange(device, other.device);
  extensions = std::exchange(extensions, other.extensions);
  propertiesFeatures = std::move(other.propertiesFeatures);
  queueIndices.family.graphicsCompute =
      std::exchange(queueIndices.family.graphicsCompute,
                    other.queueIndices.family.graphicsCompute);
  graphicsComputeQueue =
      std::exchange(graphicsComputeQueue, other.graphicsComputeQueue);
  queueIndices.family.computeAsync = std::exchange(
      queueIndices.family.computeAsync, other.queueIndices.family.computeAsync);
  computeAsyncQueue = std::exchange(computeAsyncQueue, other.computeAsyncQueue);
  queueIndices.family.transfer = std::exchange(
      queueIndices.family.transfer, other.queueIndices.family.transfer);
  transferQueue = std::exchange(transferQueue, other.transferQueue);
  queueIndices.family.present = std::exchange(
      queueIndices.family.present, other.queueIndices.family.present);
  presentQueue = std::exchange(presentQueue, other.presentQueue);
  extSwapchainMaintenance1 =
      std::exchange(extSwapchainMaintenance1, other.extSwapchainMaintenance1);
  extSwapchainColorspace =
      std::exchange(extSwapchainColorspace, other.extSwapchainColorspace);
}

DeviceVk& DeviceVk::operator=(DeviceVk&& other) noexcept {
  // Note: We are not destroying the allocator and the device
  std::lock_guard<std::mutex> lk{other.queueMutex};
  vmaAllocator = std::exchange(vmaAllocator, other.vmaAllocator);
  physicalDevice = std::exchange(physicalDevice, other.physicalDevice);
  device = std::exchange(device, other.device);
  extensions = std::exchange(extensions, other.extensions);
  propertiesFeatures.release();
  propertiesFeatures = std::move(other.propertiesFeatures);
  queueIndices.family.graphicsCompute =
      std::exchange(queueIndices.family.graphicsCompute,
                    other.queueIndices.family.graphicsCompute);
  graphicsComputeQueue =
      std::exchange(graphicsComputeQueue, other.graphicsComputeQueue);
  queueIndices.family.computeAsync = std::exchange(
      queueIndices.family.computeAsync, other.queueIndices.family.computeAsync);
  computeAsyncQueue = std::exchange(computeAsyncQueue, other.computeAsyncQueue);
  queueIndices.family.transfer = std::exchange(
      queueIndices.family.transfer, other.queueIndices.family.transfer);
  transferQueue = std::exchange(transferQueue, other.transferQueue);
  queueIndices.family.present = std::exchange(
      queueIndices.family.present, other.queueIndices.family.present);
  presentQueue = std::exchange(presentQueue, other.presentQueue);
  extSwapchainMaintenance1 =
      std::exchange(extSwapchainMaintenance1, other.extSwapchainMaintenance1);
  extSwapchainColorspace =
      std::exchange(extSwapchainColorspace, other.extSwapchainColorspace);

  return *this;
}
#endif

bool DeviceVk::getQueueUsage(VkQueue queue) {
  int32_t expectFree = 0;
  auto it = queuesStateMap.find(queue);
  if (it == queuesStateMap.end()) {
    assert(false);
    return false;
  }
  std::atomic<int32_t>& state = it->second;
  while (true) {
    if (!state.compare_exchange_weak(expectFree, 1,
                                     std::memory_order_relaxed) &&
        !expectFree) {
      // TODO maybe. add boost dependency and add yield?
      std::this_thread::yield();
      continue;
    }
    state.store(1, std::memory_order_acquire);
    break;
  }
  return true;
}

bool DeviceVk::freeQueueUsage(VkQueue queue) {
  int32_t expectUsed = 1;
  auto it = queuesStateMap.find(queue);
  if (it == queuesStateMap.end()) {
    assert(false);
    return false;
  }
  std::atomic<int32_t>& state = it->second;
  return state.compare_exchange_strong(expectUsed, 0,
                                       std::memory_order_release);
}

// ---------------------------- FRAME -----------------------------

void FrameDiscard::destroy(DeviceVk const& device) AVK_NO_CFI {
  while (!swapchains.empty()) {
    VkSwapchainKHR swapchain = swapchains.back();
    assert(swapchain != VK_NULL_HANDLE);
    swapchains.pop_back();
    vkDestroySwapchainKHR(device.device, swapchain, nullptr);
  }
  while (!semaphores.empty()) {
    VkSemaphore semaphore = semaphores.back();
    assert(semaphore != VK_NULL_HANDLE);
    semaphores.pop_back();
    vkDestroySemaphore(device.device, semaphore, nullptr);
  }
}

void Frame::destroy(DeviceVk const& device) AVK_NO_CFI {
  assert(submissionFence != VK_NULL_HANDLE);
  vkDestroyFence(device.device, submissionFence, nullptr);
  submissionFence = VK_NULL_HANDLE;
  assert(acquireSemaphore != VK_NULL_HANDLE);
  vkDestroySemaphore(device.device, acquireSemaphore, nullptr);
  acquireSemaphore = VK_NULL_HANDLE;
  discard.destroy(device);
}

void SwapchainImage::destroy(DeviceVk const& device) AVK_NO_CFI {
  assert(presentSemaphore != VK_NULL_HANDLE);
  vkDestroySemaphore(device.device, presentSemaphore, nullptr);
  presentSemaphore = VK_NULL_HANDLE;
  assert(image != VK_NULL_HANDLE);
  // Swapchain Images are destroyed with VkDestroySwapchainKHR
  image = VK_NULL_HANDLE;
}

// ---------------------------- CONTEXT -----------------------------

ContextVk::~ContextVk() noexcept AVK_NO_CFI {
  if (m_instance.instance != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(m_device.device);
    for (VkFence fence : m_fencePile) {
      vkDestroyFence(m_device.device, fence, nullptr);
    }
    m_fencePile.clear();

    destroySwapchain();

    vmaDestroyAllocator(m_device.vmaAllocator);

    if (m_vmaVulkanFunctions) {
      delete m_vmaVulkanFunctions;
    }

    if (m_device.device != VK_NULL_HANDLE) {
      vkDestroyDevice(m_device.device, nullptr);
      m_device.device = VK_NULL_HANDLE;
    }

    if (m_surface != VK_NULL_HANDLE) {
      vkDestroySurfaceKHR(m_instance.instance, m_surface, nullptr);
      m_surface = VK_NULL_HANDLE;
    }
#ifdef AVK_DEBUG
    vkDestroyDebugUtilsMessengerEXT(m_instance.instance,
                                    m_instance.debugMessenger, nullptr);
#endif

    vkDestroyInstance(m_instance.instance, nullptr);
    m_instance.instance = VK_NULL_HANDLE;
  } else {
    if (m_vmaVulkanFunctions) {
      delete m_vmaVulkanFunctions;
    }
  }
}

ContextVk::ContextVk() {
  m_frameData.reserve(16);
  m_swapchainImages.reserve(16);
  m_vkImages.reserve(16);

  // TODO: Initialize VkAllocationCallbacks to integrate eventual Host Memory
  // allocation strategy
}

bool ContextVk::initInstanceExtensions() AVK_NO_CFI {
  uint32_t count = 0;
  vkCheck(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));
  m_instance.extensions.extensions.resize(count);
  return vkCheck(vkEnumerateInstanceExtensionProperties(
      nullptr, &count, m_instance.extensions.extensions.data()));
}

bool ContextVk::createInstance(uint32_t vulkanApiVersion) AVK_NO_CFI {
  // instance extensions to enable: portabilty, surface, platform surface
  // - (PORTABILITY) Vulkan will consider devices that aren’t fully conformant
  //   such as MoltenVk to be identified as a conformant implementation. When
  //   this happens, use the VkPhysicalDevicePortabilitySubsetPropertiesKHR
  //   extension with the vkGetPhysicalDeviceFeatures2 as detailed below to
  //   get the list of supported/unsupported features.
  //   https://github.com/KhronosGroup/Vulkan-Samples/blob/main/samples/api/hello_triangle_1_3/hello_triangle_1_3.cpp
  bool const portabilitySupported = m_instance.extensions.isSupported(
      VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  if (portabilitySupported) {
    m_instance.extensions.enable(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  }

#ifdef AVK_DEBUG
  bool const debugSupported =
      m_instance.extensions.isSupported(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  if (debugSupported) {
    m_instance.extensions.enable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
#endif

  if (!m_instance.extensions.enable(VK_KHR_SURFACE_EXTENSION_NAME)) {
    return false;
  }

  m_device.extSwapchainColorspace =
      m_instance.extensions.enable(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);

  // 2 extensions used in swapchain recreation
  if (!m_instance.extensions.enable(
          VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME)) {
    return false;
  }
  if (!m_instance.extensions.enable(
          VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME)) {
    return false;
  }
#ifdef AVK_OS_WINDOWS
  if (!m_instance.extensions.enable(VK_KHR_WIN32_SURFACE_EXTENSION_NAME)) {
    return false;
  }
#elif defined(AVK_OS_MACOS)
  if (!m_instance.extensions.enable(VK_EXT_METAL_SURFACE_EXTENSION_NAME)) {
    return false;
  }
#elif defined(AVK_OS_ANDROID)
#  error "TODO"
#elif defined(AVK_OS_LINUX)
#  error "TODO X11 and Wayland"
#else
#  error "ADD SUPPORT"
#endif

  // validation layers
#ifdef AVK_DEBUG
  std::vector<char const*> validationLayers;
  validationLayers.reserve(16);
  const char* validationLayerName = "VK_LAYER_KHRONOS_validation";
  uint32_t layerCount = 0;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
  for (const auto& layerProperties : availableLayers) {
    if (strcmp(layerProperties.layerName, validationLayerName) == 0) {
      validationLayers.push_back(validationLayerName);
    }
  }
#endif

  // Warning: VK_MAKE_API_VERSION(0, 1, 3, 2) necessary to use dynamic
  // rendering
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pNext = nullptr;
  appInfo.pApplicationName = "AetherVK";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "AetherVK";
  appInfo.apiVersion = vulkanApiVersion;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pNext = nullptr;
#ifdef AVK_DEBUG
  if (debugSupported) {
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    debugCreateInfo.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugCreateInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugCreateInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugCreateInfo.pfnUserCallback = debugCallback;
    debugCreateInfo.pUserData = nullptr;

    createInfo.pNext = &debugCreateInfo;
  }
#endif

  createInfo.flags = 0;
  createInfo.pApplicationInfo = &appInfo;
#ifdef AVK_DEBUG
  createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
  createInfo.ppEnabledLayerNames =
      validationLayers.size() ? validationLayers.data() : nullptr;
#else
  createInfo.enabledLayerCount = 0;
  createInfo.ppEnabledLayerNames = nullptr;
#endif

  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(m_instance.extensions.enabled.size());
  createInfo.ppEnabledExtensionNames = m_instance.extensions.enabled.data();

  if (portabilitySupported) {
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  }

  bool res =
      vkCheck(vkCreateInstance(&createInfo, nullptr, &m_instance.instance));
#ifdef AVK_DEBUG
  if (res && debugSupported) {
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    debugCreateInfo.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugCreateInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugCreateInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugCreateInfo.pfnUserCallback = debugCallback;
    debugCreateInfo.pUserData = nullptr;

    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(m_instance.instance,
                              "vkCreateDebugUtilsMessengerEXT"));
    if (func != nullptr) {
      res = vkCheck(func(m_instance.instance, &debugCreateInfo, nullptr,
                         &m_instance.debugMessenger));
    }
  }
#endif

  // allocate function pointers to skip trampoline on vulkan loader
  volkLoadInstance(m_instance.instance);

  return res;
}

bool ContextVk::createSurface() AVK_NO_CFI {
#ifdef VK_USE_PLATFORM_WIN32_KHR
  VkWin32SurfaceCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
  createInfo.hwnd = m_hWindow;
  createInfo.hinstance = GetModuleHandleW(nullptr);
  auto const vkCreateWin32SurfaceKHR =
      reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(vkGetInstanceProcAddr(
          m_instance.instance, "vkCreateWin32SurfaceKHR"));
  return vkCreateWin32SurfaceKHR &&
         vkCheck(vkCreateWin32SurfaceKHR(m_instance.instance, &createInfo,
                                         nullptr, &m_surface));
#elif defined(VK_USE_PLATFORM_METAL_EXT)
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
#elif defined(VK_USE_PLATFORM_XCB_KHR)
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
#else
#  error "ADD SUPPORT"
#endif
}

bool ContextVk::physicalDeviceSupport(DeviceVk& physicalDevice) const
    AVK_NO_CFI {
  // TODO add buffer device address extension
  // inspiration from
  // https://github.com/blender/blender/blob/main/source/blender/gpu/vulkan/vk_backend.cc#L48
  // for now, just query about core 1.1, 1.2, 1.3 features and device driver
  // properties (the kind you can get with nvidia-smi, for example)
  VkPhysicalDeviceProperties2 properties{};
  VkPhysicalDeviceVulkan11Properties properties11{};
  VkPhysicalDeviceVulkan12Properties properties12{};
  VkPhysicalDeviceVulkan13Properties properties13{};
  VkPhysicalDeviceDriverProperties driverProperties{};  // Vulkan 1.2

  // set all sTypes and pNexts
  properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
  properties12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
  properties13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
  driverProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;

  properties.pNext = &properties11;
  properties11.pNext = &properties12;
  properties12.pNext = &properties13;
  properties13.pNext = &driverProperties;

  // in properties 1.1, we are interested in max descriptor set count,
  // subgroup properties and max allocation
  vkGetPhysicalDeviceProperties2(physicalDevice.physicalDevice, &properties);

  // just check that the device actually supports Vulkan 1.3
  uint32_t const apiVersion =
      VK_MAKE_API_VERSION(0, driverProperties.conformanceVersion.major,
                          driverProperties.conformanceVersion.minor,
                          driverProperties.conformanceVersion.subminor);
  if (apiVersion < VK_API_VERSION_1_3) {
    return false;
  }

  return true;
}

std::vector<std::string_view> ContextVk::anyMissingCapabilities(
    DeviceVk& physicalDevice) const AVK_NO_CFI {
  std::vector<std::string_view> missing;
  missing.reserve(64);

  // TODO: avoid queries using Vulkan Versioned feature structs. Stick to those
  // from either extensions or 1.0
  VkPhysicalDeviceFeatures2 features2{};
  VkPhysicalDeviceVulkan11Features features11{};
  VkPhysicalDeviceVulkan12Features features12{};
  VkPhysicalDeviceVulkan13Features features13{};
  VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT
      featuresSwapchainMaintenance1{};

  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  featuresSwapchainMaintenance1.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;

  features2.pNext = &features11;
  features11.pNext = &features12;
  features12.pNext = &features13;
  features13.pNext = &featuresSwapchainMaintenance1;

  // we require the same device features from Blender's initialization, which
  // are the common one
  vkGetPhysicalDeviceFeatures2(physicalDevice.physicalDevice, &features2);
  features13.pNext = nullptr;

  // WARNING: Keep this in sync with device creation function
  if (features2.features.geometryShader == VK_FALSE)
    missing.push_back("geometryShader");
  if (features2.features.tessellationShader == VK_FALSE)
    missing.push_back("tessellationShader");
  if (features2.features.depthBiasClamp == VK_FALSE)
    missing.push_back("depth bias clamp");
  if (features12.timelineSemaphore == VK_FALSE)
    missing.push_back("timeline semaphores");
  if (features12.bufferDeviceAddress == VK_FALSE)
    missing.push_back("buffer device address");
  if (features2.features.imageCubeArray == VK_FALSE)
    missing.push_back("image cube array");
  if (features2.features.drawIndirectFirstInstance == VK_FALSE)
    missing.push_back("draw indirect first instance");
  if (features11.shaderDrawParameters == VK_FALSE)
    missing.push_back("shader draw parameters");
  if (features2.features.sampleRateShading == VK_FALSE)
    missing.push_back("sample rate shading");
  if (features2.features.fragmentStoresAndAtomics == VK_FALSE)
    missing.push_back("fragment stores and atomics");
  if (features12.timelineSemaphore == VK_FALSE)
    missing.push_back("Timeline Semaphores");
  // Note: No Shader Clip (ClipDistance decorator) or Shader Cull (CullDistance
  // decorator) capability for SPIR-V Shaders

  // TODO these two are optional
  if (features13.dynamicRendering == VK_FALSE)
    missing.push_back("dynamic rendering");
  if (features13.synchronization2 == VK_FALSE)
    missing.push_back("synchronization 2");
  if (featuresSwapchainMaintenance1.swapchainMaintenance1 == VK_TRUE)
    physicalDevice.extSwapchainMaintenance1 = true;

  // device extensions: swapchain, dynamic rendering
  uint32_t vkExtensionCount = 0;
  vkEnumerateDeviceExtensionProperties(physicalDevice.physicalDevice, nullptr,
                                       &vkExtensionCount, nullptr);
  std::vector<VkExtensionProperties> vkExtensions(vkExtensionCount);
  std::unordered_set<std::string_view> extensions;
  vkEnumerateDeviceExtensionProperties(physicalDevice.physicalDevice, nullptr,
                                       &vkExtensionCount, vkExtensions.data());
  for (const auto& ext : vkExtensions) {
    extensions.insert(ext.extensionName);
  }

  if (extensions.find(VK_KHR_SWAPCHAIN_EXTENSION_NAME) == extensions.end())
    missing.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  if (extensions.find(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) ==
      extensions.end())
    missing.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);

  return missing;
}

bool static deviceExtensionSupported(
    DeviceVk& physicalDevice,
    std::vector<std::string_view> const& requiredExtensions) AVK_NO_CFI {
  uint32_t vkExtensionCount = 0;
  vkEnumerateDeviceExtensionProperties(physicalDevice.physicalDevice, nullptr,
                                       &vkExtensionCount, nullptr);
  std::vector<VkExtensionProperties> vkExtensions(vkExtensionCount);
  vkEnumerateDeviceExtensionProperties(physicalDevice.physicalDevice, nullptr,
                                       &vkExtensionCount, vkExtensions.data());

  std::unordered_set<std::string_view> extensions;
  for (const auto& ext : vkExtensions) {
    extensions.insert(ext.extensionName);
    physicalDevice.extensions.extensions.push_back(ext);
  }

  for (auto const& reqExt : requiredExtensions) {
    if (extensions.find(reqExt) == extensions.end()) {
      return false;
    }
  }

  return true;
}

bool ContextVk::selectPhysicalDevice(
    std::vector<std::string_view> const& requiredExtensions) AVK_NO_CFI {
  int32_t bestScore = 0;
  int32_t bestIndex = -1;

  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(m_instance.instance, &deviceCount, nullptr);
  if (deviceCount == 0) {
    return false;
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(m_instance.instance, &deviceCount, devices.data());

  for (size_t i = 0; i < devices.size(); ++i) {
    int32_t score = 0;
    DeviceVk currentDevice;
    currentDevice.physicalDevice = devices[i];

    if (!physicalDeviceSupport(currentDevice) ||
        !anyMissingCapabilities(currentDevice).empty() ||
        !deviceExtensionSupported(currentDevice, requiredExtensions)) {
      continue;
    }

    VkPhysicalDeviceProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkGetPhysicalDeviceProperties2(currentDevice.physicalDevice, &props);

    switch (props.properties.deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        score += 400;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        score += 300;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        score += 200;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_CPU:
        score += 100;
        break;
      default:
        break;
    }

    if (score > bestScore) {
      m_device = std::move(currentDevice);

      bestScore = score;
      bestIndex = static_cast<int32_t>(i);
    }
  }

  return bestIndex != -1;
}

static void initializeGenericQueueFamilies(VkInstance instance,
                                           DeviceVk& device,
                                           VkSurfaceKHR surface) AVK_NO_CFI {
  uint32_t queueFamilyCount = 0;
  assert(vkGetPhysicalDeviceQueueFamilyProperties2);
  vkGetPhysicalDeviceQueueFamilyProperties2(device.physicalDevice,
                                            &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties2> queueFamilies(queueFamilyCount);
  // video properties are possible future uses
  std::vector<VkQueueFamilyVideoPropertiesKHR> queueFamiliesVideo(
      queueFamilyCount);
  for (uint32_t i = 0; i < queueFamilyCount; ++i) {
    queueFamilies[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
    queueFamilies[i].pNext = &queueFamiliesVideo[i];

    queueFamiliesVideo[i].sType =
        VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR;
  }

  device.queueIndices.family.graphicsCompute =
      DeviceVk::InvalidQueueFamilyIndex;
  device.queueIndices.family.computeAsync = DeviceVk::InvalidQueueFamilyIndex;
  device.queueIndices.family.transfer = DeviceVk::InvalidQueueFamilyIndex;
  device.queueIndices.family.present = DeviceVk::InvalidQueueFamilyIndex;

  vkGetPhysicalDeviceQueueFamilyProperties2(
      device.physicalDevice, &queueFamilyCount, queueFamilies.data());
  if (device.queueIndices.family.graphicsCompute ==
      DeviceVk::InvalidQueueFamilyIndex) {
    // look for a queue that supports graphics and compute
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
      if ((queueFamilies[i].queueFamilyProperties.queueFlags &
           VK_QUEUE_GRAPHICS_BIT) != 0 &&
          (queueFamilies[i].queueFamilyProperties.queueFlags &
           VK_QUEUE_COMPUTE_BIT) != 0) {
        device.queueIndices.family.graphicsCompute = i;
        break;
      }
    }
  }

  if (device.queueIndices.family.computeAsync ==
      DeviceVk::InvalidQueueFamilyIndex) {
    // try to look for a queue that supports compute but not graphics, if not
    // found, default to graphicsComputeQueueFamily
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
      if ((queueFamilies[i].queueFamilyProperties.queueFlags &
           VK_QUEUE_COMPUTE_BIT) != 0 &&
          (queueFamilies[i].queueFamilyProperties.queueFlags &
           VK_QUEUE_GRAPHICS_BIT) == 0) {
        device.queueIndices.family.computeAsync = i;
        break;
      }
    }
    if (device.queueIndices.family.computeAsync ==
        DeviceVk::InvalidQueueFamilyIndex) {
      device.queueIndices.family.computeAsync =
          device.queueIndices.family.computeAsync;
    }
  }

  if (device.queueIndices.family.transfer ==
      DeviceVk::InvalidQueueFamilyIndex) {
    // try to look for a queue that supports transfer but not graphics and
    // compute, if not found, default to graphicsComputeQueueFamily
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
      if ((queueFamilies[i].queueFamilyProperties.queueFlags &
           VK_QUEUE_TRANSFER_BIT) != 0 &&
          (queueFamilies[i].queueFamilyProperties.queueFlags &
           VK_QUEUE_GRAPHICS_BIT) == 0 &&
          (queueFamilies[i].queueFamilyProperties.queueFlags &
           VK_QUEUE_COMPUTE_BIT) == 0) {
        device.queueIndices.family.transfer = i;
        break;
      }
    }
    if (device.queueIndices.family.transfer ==
        DeviceVk::InvalidQueueFamilyIndex) {
      for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if ((queueFamilies[i].queueFamilyProperties.queueFlags &
             VK_QUEUE_TRANSFER_BIT) != 0) {
          device.queueIndices.family.transfer = i;
          break;
        }
      }
    }
  }

  if (surface != VK_NULL_HANDLE &&
      device.queueIndices.family.present == DeviceVk::InvalidQueueFamilyIndex) {
    auto const vkGetPhysicalDeviceSurfaceSupportKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
            vkGetInstanceProcAddr(instance,
                                  "vkGetPhysicalDeviceSurfaceSupportKHR"));
    // look for a queue that supports present
    VkBool32 presentSupport = VK_FALSE;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
      vkGetPhysicalDeviceSurfaceSupportKHR(device.physicalDevice, i, surface,
                                           &presentSupport);
      if (presentSupport == VK_TRUE) {
        device.queueIndices.family.present = i;
        break;
      }
    }
  }

  assert(device.queueIndices.family.graphicsCompute !=
         DeviceVk::InvalidQueueFamilyIndex);
  assert(device.queueIndices.family.computeAsync !=
         DeviceVk::InvalidQueueFamilyIndex);
  assert(device.queueIndices.family.transfer !=
         DeviceVk::InvalidQueueFamilyIndex);
  assert(device.queueIndices.family.present !=
         DeviceVk::InvalidQueueFamilyIndex);
}

bool ContextVk::createDevice(
    std::vector<std::string_view> const& requiredExtensions) AVK_NO_CFI {
  // to call vkCreateDevice: 1) extensions 2) features 3) queues
  initializeGenericQueueFamilies(m_instance.instance, m_device, m_surface);

  assert(requiredExtensions.size() > 0 && "No Device Extensions?");
  if (!m_device.extensions.enable(requiredExtensions.begin(),
                                  requiredExtensions.end())) {
    return false;
  }

  // Desktop devices should be fine. Android only from Android 14 onwards
  // m_device.extSwapchainMaintenance1 feature checked by anyMissingCapabilities
  if (!m_device.extSwapchainMaintenance1 ||
      !m_device.extensions.enable(
          VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME)) {
    m_device.extSwapchainMaintenance1 = false;
  }

  // how many queues we'll request: 1 per family for now
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

  float queuePriority = 1.0f;
  std::unordered_set<uint32_t> const uniqueQueueFamilies{
      m_device.queueIndices.family.graphicsCompute,
      m_device.queueIndices.family.computeAsync,
      m_device.queueIndices.family.transfer,
      m_device.queueIndices.family.present};
  queueCreateInfos.resize(uniqueQueueFamilies.size());

  uint32_t index = 0;
  for (uint32_t queueFamily : uniqueQueueFamilies) {
    uint32_t const i = index++;
    queueCreateInfos[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfos[i].pNext = nullptr;
    queueCreateInfos[i].flags = 0;
    queueCreateInfos[i].queueCount = 1;  // TODO Maybe more?
    queueCreateInfos[i].pQueuePriorities = &queuePriority;
    queueCreateInfos[i].queueFamilyIndex = queueFamily;
  }
  m_device.queuesStateMap.reserve(index);

  // create features (the ones we use here have been tested)
  VkPhysicalDeviceFeatures2 featuresBase{};
  featuresBase.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeature{};
  dynamicRenderingFeature.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
  VkPhysicalDeviceVulkan12Features features12{};
  features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  VkPhysicalDeviceSynchronization2FeaturesKHR sync2Features{};
  sync2Features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
  VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT
      swapchainMaintenance1Feature{};
  swapchainMaintenance1Feature.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;

  featuresBase.pNext = &dynamicRenderingFeature;
  dynamicRenderingFeature.pNext = &features12;
  features12.pNext = &sync2Features;
  sync2Features.pNext = &swapchainMaintenance1Feature;

  // list of used features (TODO: synchronization2, textureCompression,
  // sparseResidency, pipelineStatisticsQuery)
  featuresBase.features.geometryShader = VK_TRUE;
  featuresBase.features.tessellationShader = VK_TRUE;
  featuresBase.features.fillModeNonSolid = VK_TRUE;  // wireframe rendering
  featuresBase.features.depthBiasClamp = VK_TRUE;
  featuresBase.features.sampleRateShading = VK_TRUE;
  dynamicRenderingFeature.dynamicRendering = VK_TRUE;
  features12.bufferDeviceAddress = VK_TRUE;
  features12.timelineSemaphore = VK_TRUE;
  sync2Features.synchronization2 = VK_TRUE;
  if (m_device.extSwapchainMaintenance1) {
    swapchainMaintenance1Feature.swapchainMaintenance1 = VK_TRUE;
  }

  // features we require are checked by physicalDeviceSupport and
  // anyMissingCapabilities and in device creation it's deprecated
  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.pNext = &featuresBase;
  createInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());
  createInfo.pQueueCreateInfos = queueCreateInfos.data();
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(m_device.extensions.enabled.size());
  createInfo.ppEnabledExtensionNames = m_device.extensions.enabled.data();

  if (!vkCheck(vkCreateDevice(m_device.physicalDevice, &createInfo, nullptr,
                              &m_device.device))) {
    return false;
  }

  // if you use a pNext or flag on create info, use vkGetDeviceQueue2
  // TODO handle queues better: make an effort into having the present queue
  // different from all the rest (hence store number of queues for each queue
  // family, and if family of presentation) is the same for others, then write
  // it out
  vkGetDeviceQueue(m_device.device,
                   m_device.queueIndices.family.graphicsCompute, 0,
                   &m_device.graphicsComputeQueue);
  vkGetDeviceQueue(m_device.device, m_device.queueIndices.family.computeAsync,
                   0, &m_device.computeAsyncQueue);
  vkGetDeviceQueue(m_device.device, m_device.queueIndices.family.transfer, 0,
                   &m_device.transferQueue);
  vkGetDeviceQueue(m_device.device, m_device.queueIndices.family.present, 0,
                   &m_device.presentQueue);

  m_device.queuesStateMap.try_emplace(m_device.graphicsComputeQueue, 1);
  m_device.queuesStateMap.try_emplace(m_device.computeAsyncQueue, 1);
  m_device.queuesStateMap.try_emplace(m_device.transferQueue, 1);
  m_device.queuesStateMap.try_emplace(m_device.presentQueue, 1);

  // populate extension function pointers to save one dispatch from the vulkan
  // loader
  assert(m_device.device != VK_NULL_HANDLE);
  volkLoadDevice(m_device.device);
  std::cout << "VOLK LOAD DEVICE " << std::endl;

  // create instance of VmaVulkanFunctions
  VmaDeviceMemoryCallbacks vmaMemoryCallbacks{};
  vmaMemoryCallbacks.pfnAllocate = allocateDeviceMemoryCallback;
  vmaMemoryCallbacks.pfnFree = freeDeviceMemoryCallback;

  if (!m_vmaVulkanFunctions) {
    m_vmaVulkanFunctions = new VmaVulkanFunctions;
  }
  memset(m_vmaVulkanFunctions, 0, sizeof(VmaVulkanFunctions));

  VmaAllocatorCreateInfo vmaAllocatorCreateInfo{};
  vmaAllocatorCreateInfo.physicalDevice = m_device.physicalDevice;
  vmaAllocatorCreateInfo.device = m_device.device;
  vmaAllocatorCreateInfo.instance = m_instance.instance;
  vmaAllocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
  vmaAllocatorCreateInfo.pDeviceMemoryCallbacks = &vmaMemoryCallbacks;

  // fill vmaAllocatorCreateInfo.pVulkanFunctions
  if (!vkCheck(vmaImportVulkanFunctionsFromVolk(&vmaAllocatorCreateInfo,
                                                m_vmaVulkanFunctions))) {
    std::cerr
        << "\033[31m------ FAILURE -- vmaImportVulkanFunctionsFromVolk\033[0m"
        << std::endl;
    return false;
  }
  vmaAllocatorCreateInfo.pVulkanFunctions = m_vmaVulkanFunctions;
  if (!vkCheck(vmaCreateAllocator(&vmaAllocatorCreateInfo,
                                  &m_device.vmaAllocator))) {
    std::cerr << "\033[31m------ FAILURE -- vmaCreateAllocator\033[0m"
              << std::endl;
    return false;
  }

  return true;
}

ContextResult ContextVk::initializeDrawingContext(ContextVkParams const& params)
    AVK_NO_CFI {
  // params
#ifdef AVK_OS_WINDOWS
  m_hWindow = params.window;
#elif defined(AVK_OS_MACOS)
#  error "TODO"
#elif defined(AVK_OS_ANDROID)
#  error "TODO"
#elif defined(AVK_OS_LINUX)
#  error "TODO X11 and Wayland"
#else
#  error "ADD SUPPORT"
#endif
  // TODO REMOVE LOGGING AND DO IT PROPERLY (Controlled by both macro and
  // runtime or just runtime)
  m_hdrInfo = std::make_unique<WindowHDRInfo>(params.hdrInfo);

  // find vulkan loader
  if (volkInitialize() != VK_SUCCESS) {
    std::cerr << "Couldn't Initialize Volk" << std::endl;
    return ContextResult::Error;
  }

  initInstanceExtensions();

  // instance and surface
  uint32_t vulkanApiVersion = VK_API_VERSION_1_3;
  if (!createInstance(vulkanApiVersion)) {
    std::cerr << "Couldn't Initialize Instance" << std::endl;
    return ContextResult::Error;
  }
  if (!createSurface()) {
    std::cerr << "Couldn't Initialize Surface" << std::endl;
    return ContextResult::Error;
  }

  // for reference, look at guide and
  std::vector<std::string_view> requiredExtensions;
  requiredExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
  // added in createDevice
  // requiredExtensions.push_back(VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
  if (!selectPhysicalDevice(requiredExtensions)) {
    std::cerr << "Couldn't Select proper VkPhysicalDevice" << std::endl;
    return ContextResult::Error;
  }
  if (!createDevice(requiredExtensions)) {
    std::cerr << "Couldn't Create proper VkDevice" << std::endl;
    return ContextResult::Error;
  }
  return ContextResult::Success;
}

static bool selectSurfaceFormat(DeviceVk const& device, VkSurfaceKHR surface,
                                bool useHDR,
                                VkSurfaceFormatKHR& outFormat) AVK_NO_CFI {
  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device.physicalDevice, surface,
                                       &formatCount, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(device.physicalDevice, surface,
                                       &formatCount, formats.data());

  // Select the best format (from blender)
  static VkSurfaceFormatKHR constexpr preferredFormats[] = {
      {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT},
      {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
      {VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_HDR10_ST2084_EXT},
      {VK_FORMAT_A2B10G10R10_UNORM_PACK32,
       VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT},
      {VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_HDR10_ST2084_EXT},
      {VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
      {VK_FORMAT_R8G8B8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
  };

  for (auto const& preferredFormat : preferredFormats) {
    if (!useHDR &&
        (preferredFormat.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT ||
         preferredFormat.colorSpace ==
             VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT)) {
      continue;
    }
    for (auto const& availableFormat : formats) {
      if (availableFormat.format == preferredFormat.format &&
          availableFormat.colorSpace == preferredFormat.colorSpace) {
        outFormat = availableFormat;
        return true;
      }
    }
  }

  outFormat = formats[0];
  return true;
}

static bool selectPresentMode(bool vsyncOff, DeviceVk const& device,
                              VkSurfaceKHR surface,
                              VkPresentModeKHR& outPresentMode) AVK_NO_CFI {
  uint32_t presentModeCount = 0;

  vkGetPhysicalDeviceSurfacePresentModesKHR(device.physicalDevice, surface,
                                            &presentModeCount, nullptr);
  std::vector<VkPresentModeKHR> presentModes(presentModeCount);
  vkGetPhysicalDeviceSurfacePresentModesKHR(
      device.physicalDevice, surface, &presentModeCount, presentModes.data());

  // Select the best present mode
  if (vsyncOff) {
    for (VkPresentModeKHR mode : presentModes) {
      if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
        outPresentMode = mode;
        return true;
      }
    }
  } else {
    for (VkPresentModeKHR mode : presentModes) {
      if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
        outPresentMode = mode;
        return true;
      }
    }
  }

  outPresentMode = VK_PRESENT_MODE_FIFO_KHR;
  return true;
}

#ifdef _WIN32
static VkExtent2D getWindowExtent(HWND hWnd) {
  VkExtent2D extent{};

  RECT client{}, dwm{};
  GetClientRect(hWnd, &client);

  if (SUCCEEDED(DwmGetWindowAttribute(hWnd, DWMWA_EXTENDED_FRAME_BOUNDS, &dwm,
                                      sizeof(dwm)))) {
    // True visible rectangle on screen (DWM composition area)
    extent.width =
        std::max<uint32_t>(static_cast<uint32_t>(dwm.right - dwm.left), 1);
    extent.height =
        std::max<uint32_t>(static_cast<uint32_t>(dwm.bottom - dwm.top), 1);
  } else {
    // Standard window path (client area only)
    extent.width = std::max<uint32_t>(client.right - client.left, 1);
    extent.height = std::max<uint32_t>(client.bottom - client.top, 1);
  }

  std::cout << "\033[31m" << "getWindowExtent: " << extent.width << "x"
            << extent.height << "\033[0m" << std::endl;
  return extent;
}
#endif

ContextResult ContextVk::recreateSwapchain(
    bool useHDR, VkExtent2D const* overrideExtent) AVK_NO_CFI {
  // Don't bother resizing until the user finishes resizing the window
  while (isResizing.load(std::memory_order_relaxed)) {
    // wait (no yield processor cause this is part of the render loop)
  }

  if (m_callbacks) {
    m_callbacks->onSwapchainRecreationStarted(*this);
  }

  // Notice: No waiting here, because we are pushing the swapchain
  // onto the frame's discard buffer
  // frame's discard buffer is flushed the next time you try to acquire the
  // frame

  // TODO synchronization with timeline semaphore and/or present fences
  // VkSurfaceCapabilitiesKHR.currentTransform=VkSwapchainCreateInfoKHR.preTransform
  if (!selectSurfaceFormat(m_device, m_surface, useHDR, m_surfaceFormat)) {
    return ContextResult::Error;
  }
  // TODO make vsync configurable
  VkPresentModeKHR presentMode;
  if (!selectPresentMode(false, m_device, m_surface, presentMode)) {
    return ContextResult::Error;
  }

  // 1. query surface capabilities for the given present mode on the surface
  VkSurfacePresentScalingCapabilitiesEXT scalingCapabilities{};
  scalingCapabilities.sType =
      VK_STRUCTURE_TYPE_SURFACE_PRESENT_SCALING_CAPABILITIES_EXT;
  VkSurfaceCapabilities2KHR surfaceCapabilities2{};
  surfaceCapabilities2.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR;
  surfaceCapabilities2.pNext = &scalingCapabilities;

  VkSurfacePresentModeEXT presentModeInfo{};
  presentModeInfo.sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_EXT;
  presentModeInfo.presentMode = presentMode;
  VkPhysicalDeviceSurfaceInfo2KHR surfaceInfo2{};
  surfaceInfo2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR;
  surfaceInfo2.pNext = &presentModeInfo;
  surfaceInfo2.surface = m_surface;

  VkSurfaceCapabilitiesKHR capabilities{};
  if (m_device.extSwapchainMaintenance1) {
    vkGetPhysicalDeviceSurfaceCapabilities2KHR(
        m_device.physicalDevice, &surfaceInfo2, &surfaceCapabilities2);
    capabilities = surfaceCapabilities2.surfaceCapabilities;
  } else {
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_device.physicalDevice,
                                              m_surface, &capabilities);
  }

  m_hdrEnabled = useHDR;
  if (overrideExtent) {
    m_currentExtent = *overrideExtent;
  } else {
    m_currentExtent = capabilities.currentExtent;
    // TODO see if not necessary
    if (m_currentExtent.width == UINT32_MAX) {
#ifdef VK_USE_PLATFORM_WIN32_KHR
      m_currentExtent = getWindowExtent(m_hWindow);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
#  error "TODO Wayland surface extent wayland window info"
      // special value that means that the size is determined by the extent of
      // the surface
      m_currentExtent.width = 1280;
      m_currentExtent.height = 720;
#else
#  error "TODO Other platforms grab surface information"
#endif
    }

    if (capabilities.minImageExtent.width > m_currentExtent.width) {
      m_currentExtent.width = capabilities.minImageExtent.width;
    }
    if (capabilities.minImageExtent.height > m_currentExtent.height) {
      m_currentExtent.height = capabilities.minImageExtent.height;
    }

    if (m_device.extSwapchainMaintenance1) {
      if (scalingCapabilities.minScaledImageExtent.width >
          m_currentExtent.width) {
        m_currentExtent.width = scalingCapabilities.minScaledImageExtent.width;
      }
      if (scalingCapabilities.minScaledImageExtent.height >
          m_currentExtent.height) {
        m_currentExtent.height =
            scalingCapabilities.minScaledImageExtent.height;
      }
    }
  }

  assert(m_currentExtent.width > 0);
  assert(m_currentExtent.height > 0);

  // 2. Mark for discard swapchain resources (wait on semaphores and discard
  // old images, to be discarded on the next frame usage)
  FrameDiscard* currentDiscard = nullptr;
  if (m_frameData.size() > m_renderFrame) {
    currentDiscard = &m_frameData[m_renderFrame].discard;
    for (SwapchainImage& swapchainImage : m_swapchainImages) {
      currentDiscard->semaphores.push_back(swapchainImage.presentSemaphore);
      swapchainImage.presentSemaphore = VK_NULL_HANDLE;
    }
  }

  // double buffering on FIFO or other, triple buffering on MAILBOX
  // minimage = 0 -> no limit
  uint32_t desiredImageCount =
      presentMode == VK_PRESENT_MODE_MAILBOX_KHR ? 3 : 2;
  if (capabilities.minImageCount != 0 &&
      desiredImageCount < capabilities.minImageCount) {
    desiredImageCount = capabilities.minImageCount;
  }
  if (capabilities.maxImageCount != 0 &&
      desiredImageCount > capabilities.maxImageCount) {
    desiredImageCount = capabilities.maxImageCount;
  }

  VkSwapchainKHR const oldSwapchain = m_swapchain;

  // on first frame size may be incorrect. Stretch on first swapchain creation,
  // then uniform scaling
  VkPresentScalingFlagBitsEXT const presentScaling =
      oldSwapchain == VK_NULL_HANDLE ? VK_PRESENT_SCALING_STRETCH_BIT_EXT
                                     : VK_PRESENT_SCALING_ONE_TO_ONE_BIT_EXT;
  VkSwapchainPresentModesCreateInfoEXT presentModesCreateInfo{};
  presentModesCreateInfo.sType =
      VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT;
  presentModesCreateInfo.pNext = nullptr;
  presentModesCreateInfo.presentModeCount = 1;
  presentModesCreateInfo.pPresentModes = &presentMode;

  VkSwapchainPresentScalingCreateInfoEXT presentScalingCreateInfo{};
  presentScalingCreateInfo.sType =
      VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_EXT;
  presentScalingCreateInfo.pNext = &presentModesCreateInfo;
  // scale from top left (min, max)
  presentScalingCreateInfo.scalingBehavior =
      scalingCapabilities.supportedPresentScaling & presentScaling;
  presentScalingCreateInfo.presentGravityX =
      scalingCapabilities.supportedPresentGravityX &
      VK_PRESENT_GRAVITY_MIN_BIT_EXT;
  presentScalingCreateInfo.presentGravityY =
      scalingCapabilities.supportedPresentGravityY &
      VK_PRESENT_GRAVITY_MAX_BIT_EXT;

  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  if (m_device.extSwapchainMaintenance1) {
    createInfo.pNext = &presentScalingCreateInfo;
  }
  createInfo.surface = m_surface;
  createInfo.minImageCount = desiredImageCount;
  createInfo.imageFormat = m_surfaceFormat.format;
  createInfo.imageColorSpace = m_surfaceFormat.colorSpace;
  createInfo.imageExtent = m_currentExtent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                          (useHDR ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
  createInfo.preTransform = capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = oldSwapchain;
  // TODO better if graphics and present are different
  // CORRECTION: not necessary if swapchain images are transfer destination
  createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  createInfo.queueFamilyIndexCount = 0;
  createInfo.pQueueFamilyIndices = nullptr;

  if (!vkCheck(vkCreateSwapchainKHR(m_device.device, &createInfo, nullptr,
                                    &m_swapchain))) {
    std::cerr << "\033[31mvkCreateSwapchainKHR\033[0m" << std::endl;
    return ContextResult::Error;
  }
  std::cout << "\033[31m"
               "vkCreateSwapchainKHR -> "
            << m_currentExtent.width << "x" << m_currentExtent.height
            << "\033[0m" << std::endl;

  uint32_t actualImageCount = 0;
  vkGetSwapchainImagesKHR(m_device.device, m_swapchain, &actualImageCount,
                          nullptr);
  if (actualImageCount > m_frameData.size()) {
    assert(actualImageCount <= m_frameData.capacity());
    m_frameData.resize(actualImageCount);
  }
  m_swapchainImages.resize(actualImageCount);
  m_vkImages.resize(actualImageCount);
  vkGetSwapchainImagesKHR(m_device.device, m_swapchain, &actualImageCount,
                          m_vkImages.data());
  for (uint32_t i = 0; i < actualImageCount; ++i) {
    m_swapchainImages[i].image = m_vkImages[i];
  }

  // new semaphores if needed (image count may be increased)
  for (Frame& frame : m_frameData) {
    if (frame.acquireSemaphore != VK_NULL_HANDLE) {
      assert(currentDiscard);
      currentDiscard->semaphores.push_back(frame.acquireSemaphore);
    }
    frame.acquireSemaphore = VK_NULL_HANDLE;
  }
  if (oldSwapchain != VK_NULL_HANDLE) {
    assert(currentDiscard);
    currentDiscard->swapchains.push_back(oldSwapchain);
  }

  initializeFrameData();

  m_imageCount = actualImageCount;

  // callback (recreate depth images, recreate swapchain image views)
  if (m_callbacks) {
    m_callbacks->onSwapchainRecreationCallback(
        *this, m_vkImages.data(), static_cast<uint32_t>(m_vkImages.size()),
        m_surfaceFormat.format, m_currentExtent);
  }

  return ContextResult::Success;
}

// TODO multiple frames in flight
bool ContextVk::initializeFrameData() AVK_NO_CFI {
  // create semaphores for presenting swapchain images
  VkSemaphoreCreateInfo semaphoreCreateInfo{};
  semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkFenceCreateInfo fenceCreateInfo{};
  fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // start signaled

  // create semaphores for acquiring swapchain images + fence for submission
  for (SwapchainImage& swapchainImage : m_swapchainImages) {
    // VK_KHR_swapchain_maintenance1 can reuse semaphores
    if (swapchainImage.presentSemaphore == VK_NULL_HANDLE) {
      if (!vkCheck(vkCreateSemaphore(m_device.device, &semaphoreCreateInfo,
                                     nullptr,
                                     &swapchainImage.presentSemaphore))) {
        return false;
      }
    }
  }

  for (Frame& frame : m_frameData) {
    if (frame.acquireSemaphore == VK_NULL_HANDLE) {
      if (!vkCheck(vkCreateSemaphore(m_device.device, &semaphoreCreateInfo,
                                     nullptr, &frame.acquireSemaphore))) {
        return false;
      }
    }
    if (frame.submissionFence == VK_NULL_HANDLE) {
      if (!vkCheck(vkCreateFence(m_device.device, &fenceCreateInfo, nullptr,
                                 &frame.submissionFence))) {
        return false;
      }
    }
  }

  return true;
}

void ContextVk::destroySwapchainPresentFences(VkSwapchainKHR swapchain)
    AVK_NO_CFI {
  std::vector<VkFence> const& fences = m_presentFences[swapchain];
  if (!fences.empty()) {
    vkWaitForFences(m_device.device, static_cast<uint32_t>(fences.size()),
                    fences.data(), VK_TRUE, UINT64_MAX);
    for (VkFence fence : fences) {
      vkDestroyFence(m_device.device, fence, nullptr);
    }
  }
  m_presentFences.erase(swapchain);
}

bool ContextVk::destroySwapchain() AVK_NO_CFI {
  if (m_swapchain != VK_NULL_HANDLE) {
    destroySwapchainPresentFences(m_swapchain);
    vkDestroySwapchainKHR(m_device.device, m_swapchain, nullptr);
  }
  vkDeviceWaitIdle(m_device.device);
  for (SwapchainImage& swapchainImage : m_swapchainImages) {
    swapchainImage.destroy(m_device);
  }
  m_swapchainImages.clear();
  for (Frame& frame : m_frameData) {
    for (VkSwapchainKHR swapchain : frame.discard.swapchains) {
      destroySwapchainPresentFences(swapchain);
    }
    frame.destroy(m_device);
  }
  m_frameData.clear();
  return true;
}

void ContextVk::setPresentFence(VkSwapchainKHR swapchain,
                                VkFence presentFence) AVK_NO_CFI {
  if (presentFence == VK_NULL_HANDLE) {
    return;
  }
  m_presentFences[swapchain].push_back(presentFence);
  // recycle signaled fences
  for (auto& [otherSwapchain, fences] : m_presentFences) {
    auto it = std::remove_if(
        fences.begin(), fences.end(), [this](const VkFence fence) AVK_NO_CFI {
          // TODO remove if not worth it
          if (fence == VK_NULL_HANDLE) {
            assert(false &&
                   "There shouldn't be any null fences on present fences");
            return true;
          }
          if (vkGetFenceStatus(m_device.device, fence) == VK_NOT_READY) {
            return false;
          }
          vkResetFences(m_device.device, 1, &fence);
          m_fencePile.push_back(fence);
          return true;
        });
    fences.erase(it, fences.end());
  }
}

VkFence ContextVk::getFence() AVK_NO_CFI {
  if (!m_fencePile.empty()) {
    VkFence fence = m_fencePile.back();
    m_fencePile.pop_back();
    return fence;
  }
  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo createInfo{};  // created waiting
  createInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  vkCreateFence(m_device.device, &createInfo, nullptr, &fence);
  return fence;
}

SwapchainDataVk ContextVk::getSwapchainData() const AVK_NO_CFI {
  SwapchainImage const& swapchainImage =
      m_swapchainImages[m_acquiredSwapchainImageIndex];
  Frame const& submissionFrame = m_frameData[m_renderFrame];

  SwapchainDataVk swapchainData;
  swapchainData.image = swapchainImage.image;
  swapchainData.format = m_surfaceFormat;
  swapchainData.extent = m_currentExtent;
  swapchainData.submissionFence = submissionFrame.submissionFence;
  swapchainData.acquireSemaphore = submissionFrame.acquireSemaphore;
  swapchainData.presentSemaphore = swapchainImage.presentSemaphore;
  swapchainData.sdrWhiteLevel = m_hdrInfo ? m_hdrInfo->sdrWhiteLevel : 1.f;
  swapchainData.imageIndex = m_acquiredSwapchainImageIndex;
  return swapchainData;
}

ContextResult ContextVk::swapBufferRelease() AVK_NO_CFI {
  // minimized window doesn't have a swapchain and swapchain image. in this case
  // callback with empty data
  if (m_swapchain == VK_NULL_HANDLE) {
    SwapchainDataVk data{};
    if (m_callbacks) {
      m_callbacks->onSwapBufferDrawCallback(*this, &data);
    }
    return ContextResult::Success;
  }

  if (m_acquiredSwapchainImageIndex == UINT32_MAX) {
    return ContextResult::Error;
  }

  SwapchainImage& swapchainImage =
      m_swapchainImages[m_acquiredSwapchainImageIndex];
  Frame& submissionFrame = m_frameData[m_renderFrame];
  const bool useHDRSwapchain =
      m_hdrInfo && m_hdrInfo->hdrEnabled && m_device.extSwapchainColorspace;

  SwapchainDataVk swapchainData;
  swapchainData.image = swapchainImage.image;
  swapchainData.format = m_surfaceFormat;
  swapchainData.extent = m_currentExtent;
  swapchainData.submissionFence = submissionFrame.submissionFence;
  swapchainData.acquireSemaphore = submissionFrame.acquireSemaphore;
  swapchainData.presentSemaphore = swapchainImage.presentSemaphore;
  swapchainData.sdrWhiteLevel = m_hdrInfo ? m_hdrInfo->sdrWhiteLevel : 1.f;
  swapchainData.imageIndex = m_acquiredSwapchainImageIndex;

  vkResetFences(m_device.device, 1, &submissionFrame.submissionFence);
  if (m_callbacks) {
    m_callbacks->onSwapBufferDrawCallback(*this, &swapchainData);
  }

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &swapchainImage.presentSemaphore;
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &m_swapchain;
  presentInfo.pImageIndices = &m_acquiredSwapchainImageIndex;
  presentInfo.pResults = nullptr;

  VkResult presentResult = VK_SUCCESS;
  {
    // TODO
    // std::scoped_lock<std::mutex> lock{m_device.queueMutex};
    VkSwapchainPresentFenceInfoEXT fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT;
    VkFence presentFence = VK_NULL_HANDLE;
    if (m_device.extSwapchainMaintenance1) {
      presentFence = getFence();
      fenceInfo.swapchainCount = 1;
      fenceInfo.pFences = &presentFence;
      presentInfo.pNext = &fenceInfo;
    }
    presentResult = vkQueuePresentKHR(m_device.presentQueue, &presentInfo);
    setPresentFence(m_swapchain, presentFence);
  }
  m_acquiredSwapchainImageIndex = ContextVk::InvalidSwapchainImageIndex;

  if (presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
      presentResult == VK_SUBOPTIMAL_KHR) {
    std::cout << "\033[81m"
              << (presentResult == VK_ERROR_OUT_OF_DATE_KHR
                      ? "VK_ERROR_OUT_OF_DATE_KHR"
                      : "VK_SUBOPTIMAL_KHR")
              << "\033[0m" << std::endl;
    recreateSwapchain(useHDRSwapchain);
    return ContextResult::Success;
  }

  if (presentResult != VK_SUCCESS && presentResult != VK_SUBOPTIMAL_KHR) {
    // TODO log error
    return ContextResult::Error;
  }

  return ContextResult::Success;
}

ContextResult ContextVk::swapBufferAcquire() AVK_NO_CFI {
  if (m_acquiredSwapchainImageIndex != ContextVk::InvalidSwapchainImageIndex) {
    assert(false);
    return ContextResult::Error;
  }

  // called after all draw calls in hte application, signals ready to
  // 1. submit commands for draw calls to device
  // 2. begin building the next frame.
  //  - assumes sumbission fence in current frame has been signaled
  //    so, we wait for the next frame sumbission fence to be signaled
  //  - pass current frame to callback for command buffer submission
  //    the callback should use the frame's fence as submission fence
  //  - since the callback is called after we wait for the next frame to be
  //    complete, it is safe in the callback to clean up resources associated
  //    to the next frame
  m_renderFrame = (m_renderFrame + 1) % m_frameData.size();
  Frame& submissionFrameData = m_frameData[m_renderFrame];
  // wait for previous submission of this frame. presentating can
  // happen in parallel, but acquiring needs to happen when the frame acquire
  // semaphore has been signaled and waited for
  if (submissionFrameData.submissionFence != VK_NULL_HANDLE) {
    vkWaitForFences(m_device.device, 1, &submissionFrameData.submissionFence,
                    true, UINT64_MAX);
    // TODO not resetting? is it correct? if Submit signals this?
  }
  for (VkSwapchainKHR swapchain : submissionFrameData.discard.swapchains) {
    destroySwapchainPresentFences(swapchain);
  }
  submissionFrameData.discard.destroy(m_device);
  const bool useHDRSwapchain =
      m_hdrInfo && (m_hdrInfo->wideGamutEnabled || m_hdrInfo->hdrEnabled) &&
      m_device.extSwapchainColorspace;
  // if HDR changed (TODO not possible now), recreate swapchain
  if (m_swapchain != VK_NULL_HANDLE && useHDRSwapchain != m_hdrEnabled) {
    recreateSwapchain(useHDRSwapchain);
  }
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
// Wayland doesn't provide WSI with windowing, hence cannot detect whether
// size changed unless you do that directiy:
// https://docs.vulkan.org/spec/latest/chapters/VK_KHR_surface/wsi.html#platformCreateSurface_wayland
#  error "TODO"
  if (recreateSwapchain) {
    recreateSwapchain(useHDRSwapchain);
  }
#endif
  // if previous window was minimized there is no valid swapchain (TODO/Check),
  // when window is brought up again we might need to recreate the swapchain
  if (m_swapchain == VK_NULL_HANDLE) {
    recreateSwapchain(useHDRSwapchain);
  }

  // acquire next image. Swapchain can become invalid if you then minimize the
  // window (TODO)
  uint32_t imageIndex = 0;
  if (m_swapchain != VK_NULL_HANDLE) {
    // NVIDIA/Wayland receives a out of date swapchain whn acquiring the next
    // swapchain image, instead of having it when calling vkQueuePresentKHR
    VkResult acquireResult = VK_ERROR_OUT_OF_DATE_KHR;
    while (m_swapchain != VK_NULL_HANDLE &&
           (acquireResult == VK_ERROR_OUT_OF_DATE_KHR ||
            acquireResult == VK_SUBOPTIMAL_KHR)) {
      acquireResult = vkAcquireNextImageKHR(
          m_device.device, m_swapchain, UINT64_MAX,
          submissionFrameData.acquireSemaphore, VK_NULL_HANDLE, &imageIndex);
      if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR ||
          acquireResult == VK_SUBOPTIMAL_KHR) {
        recreateSwapchain(useHDRSwapchain);
      }
    }
  }

  // called even if swapchain discarded on minimized window, such that we can
  // free up resources
  if (m_callbacks) {
    m_callbacks->onSwapBufferAcquiredCallback(*this);
  }

  if (m_swapchain == VK_NULL_HANDLE) {
    // TODO log minimized window
    return ContextResult::Success;
  }

  m_acquiredSwapchainImageIndex = imageIndex;
  return ContextResult::Success;
}

}  // namespace avk
