#include "render/context-vk.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnullability-extension"

#ifndef VOLK_HEADER_VERSION
#define VOLK_HEADER_VERSION
#endif
#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>

#pragma GCC diagnostic pop

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <unordered_set>

#ifdef AVK_OS_WINDOWS
#include <Windows.h>
#elif defined(AVK_OS_MACOS)
#error "TODO"
#elif defined(AVK_OS_ANDROID)
#error "TODO"
#elif defined(AVK_OS_LINUX)
#error "TODO X11 and Wayland"
#else
#error "ADD SUPPORT"
#endif

static inline bool vkCheck(VkResult res) {
  if (res != ::VK_SUCCESS) {
    // TODO
    return false;
  }
  return true;
}

// TODO better logging
// TODO all device functions on table

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
// ---------------------------- FRAME -----------------------------

void FrameDiscard::destroy(DeviceVk const& device) {
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

void Frame::destroy(DeviceVk const& device) {
  assert(submissionFence != VK_NULL_HANDLE);
  vkDestroyFence(device.device, submissionFence, nullptr);
  submissionFence = VK_NULL_HANDLE;
  assert(acquireSemaphore != VK_NULL_HANDLE);
  vkDestroySemaphore(device.device, acquireSemaphore, nullptr);
  acquireSemaphore = VK_NULL_HANDLE;
  discard.destroy(device);
}

void SwapchainImage::destroy(DeviceVk const& device) {
  assert(presentSemaphore != VK_NULL_HANDLE);
  vkDestroySemaphore(device.device, presentSemaphore, nullptr);
  presentSemaphore = VK_NULL_HANDLE;
  assert(image != VK_NULL_HANDLE);
  vkDestroyImage(device.device, image, nullptr);
  image = VK_NULL_HANDLE;
}

// ---------------------------- CONTEXT -----------------------------

ContextVk::~ContextVk() noexcept {
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

    vkDestroyInstance(m_instance.instance, nullptr);
    m_instance.instance = VK_NULL_HANDLE;
  } else {
    if (m_vmaVulkanFunctions) {
      delete m_vmaVulkanFunctions;
    }
  }
}

ContextVk::ContextVk(ContextVkParams const& params)
    :
#ifdef AVK_OS_WINDOWS
      m_hWindow(params.window),
#elif defined(AVK_OS_MACOS)
#error "TODO"
#elif defined(AVK_OS_ANDROID)
#error "TODO"
#elif defined(AVK_OS_LINUX)
#error "TODO X11 and Wayland"
#else
#error "ADD SUPPORT"
#endif
      m_hdrInfo(std::make_unique<WindowHDRInfo>(params.hdrInfo)) {

  m_frameData.reserve(16);
  m_swapchainImages.reserve(16);
  m_vkImages.reserve(16);
  initInstanceExtensions();
  // TODO: Initialize VkAllocationCallbacks to integrate eventual Host Memory
  // allocation strategy
}

bool ContextVk::initInstanceExtensions() {
  uint32_t count = 0;
  vkCheck(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));
  m_instance.extensions.extensions.resize(count);
  return vkCheck(vkEnumerateInstanceExtensionProperties(
      nullptr, &count, m_instance.extensions.extensions.data()));
}

bool ContextVk::createInstance(uint32_t vulkanApiVersion) {
  if (!volkInitialize()) {
    return false;
  }

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
          VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME)) {
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
#error "TODO"
#elif defined(AVK_OS_LINUX)
#error "TODO X11 and Wayland"
#else
#error "ADD SUPPORT"
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

bool ContextVk::createSurface() {
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
#error "ADD SUPPORT"
#endif
}

bool ContextVk::physicalDeviceSupport(DeviceVk& physicalDevice) const {
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

  if (!physicalDevice.propertiesFeatures) {
    physicalDevice.propertiesFeatures =
        std::make_unique<DevicePropertiesFeatures>();
  }

  physicalDevice.propertiesFeatures->properties2 = properties;
  physicalDevice.propertiesFeatures->properties11 = properties11;
  physicalDevice.propertiesFeatures->properties12 = properties12;
  physicalDevice.propertiesFeatures->properties13 = properties13;
  physicalDevice.propertiesFeatures->driverProperties = driverProperties;

  return true;
}

std::vector<std::string_view> ContextVk::anyMissingCapabilities(
    DeviceVk& physicalDevice) const {
  std::vector<std::string_view> missing;
  missing.reserve(64);

  VkPhysicalDeviceFeatures2 features2{};
  VkPhysicalDeviceVulkan11Features features11{};
  VkPhysicalDeviceVulkan12Features features12{};

  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

  features2.pNext = &features11;
  features11.pNext = &features12;

  // we require the same device features from Blender's initialization, which
  // are the common one
  vkGetPhysicalDeviceFeatures2(physicalDevice.physicalDevice, &features2);
  if (features2.features.geometryShader == VK_FALSE)
    missing.push_back("geometryShader");
  if (features2.features.tessellationShader == VK_FALSE)
    missing.push_back("tessellationShader");
  if (features2.features.logicOp == VK_FALSE)
    missing.push_back("logical operations");
  if (features2.features.dualSrcBlend == VK_FALSE)
    missing.push_back("dual source blending");
  if (features2.features.imageCubeArray == VK_FALSE)
    missing.push_back("image cube array");
  if (features2.features.multiDrawIndirect == VK_FALSE)
    missing.push_back("multi draw indirect");
  if (features2.features.multiViewport == VK_FALSE)
    missing.push_back("multi viewport");
  if (features2.features.shaderClipDistance == VK_FALSE)
    missing.push_back("shader clip distance");
  if (features2.features.drawIndirectFirstInstance == VK_FALSE)
    missing.push_back("draw indirect first instance");
  if (features2.features.fragmentStoresAndAtomics == VK_FALSE)
    missing.push_back("fragment stores and atomics");
  if (features11.shaderDrawParameters == VK_FALSE)
    missing.push_back("shader draw parameters");
  if (features12.timelineSemaphore == VK_FALSE)
    missing.push_back("timeline semaphores");
  if (features12.bufferDeviceAddress == VK_FALSE)
    missing.push_back("buffer device address");

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

  if (!physicalDevice.propertiesFeatures) {
    physicalDevice.propertiesFeatures =
        std::make_unique<DevicePropertiesFeatures>();
  }

  physicalDevice.propertiesFeatures->features2 = features2;
  physicalDevice.propertiesFeatures->features11 = features11;
  physicalDevice.propertiesFeatures->features12 = features12;

  return missing;
}

bool static deviceExtensionSupported(
    DeviceVk& physicalDevice,
    std::vector<std::string_view> const& requiredExtensions) {
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
    std::vector<std::string_view> const& requiredExtensions) {
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

    assert(currentDevice.propertiesFeatures);
    switch (
        currentDevice.propertiesFeatures->properties2.properties.deviceType) {
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
                                           VkSurfaceKHR surface) {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties2(device.physicalDevice,
                                            &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties2> queueFamilies(queueFamilyCount);
  // family ownership and video properties are possible future uses
  std::vector<VkQueueFamilyOwnershipTransferPropertiesKHR>
      queueFamiliesTransferOwnership(queueFamilyCount);
  std::vector<VkQueueFamilyVideoPropertiesKHR> queueFamiliesVideo(
      queueFamilyCount);
  for (uint32_t i = 0; i < queueFamilyCount; ++i) {
    queueFamilies[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
    queueFamilies[i].pNext = &queueFamiliesTransferOwnership[i];

    queueFamiliesTransferOwnership[i].sType =
        VK_STRUCTURE_TYPE_QUEUE_FAMILY_OWNERSHIP_TRANSFER_PROPERTIES_KHR;
    queueFamiliesTransferOwnership[i].pNext = &queueFamiliesVideo[i];

    queueFamiliesVideo[i].sType =
        VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR;
  }

  vkGetPhysicalDeviceQueueFamilyProperties2(
      device.physicalDevice, &queueFamilyCount, queueFamilies.data());
  if (device.graphicsComputeQueueFamily == DeviceVk::InvalidQueueFamilyIndex) {
    // look for a queue that supports graphics and compute
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
      if ((queueFamilies[i].queueFamilyProperties.queueFlags &
           VK_QUEUE_GRAPHICS_BIT) != 0 &&
          (queueFamilies[i].queueFamilyProperties.queueFlags &
           VK_QUEUE_COMPUTE_BIT) != 0) {
        device.graphicsComputeQueueFamily = i;
        break;
      }
    }
  }

  if (device.computeAsyncQueueFamily == DeviceVk::InvalidQueueFamilyIndex) {
    // try to look for a queue that supports compute but not graphics, if not
    // found, default to graphicsComputeQueueFamily
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
      if ((queueFamilies[i].queueFamilyProperties.queueFlags &
           VK_QUEUE_COMPUTE_BIT) != 0 &&
          (queueFamilies[i].queueFamilyProperties.queueFlags &
           VK_QUEUE_GRAPHICS_BIT) == 0) {
        device.computeAsyncQueueFamily = i;
        break;
      }
    }
    if (device.computeAsyncQueueFamily == DeviceVk::InvalidQueueFamilyIndex) {
      device.computeAsyncQueueFamily = device.graphicsComputeQueueFamily;
    }
  }

  if (device.transferQueueFamily == DeviceVk::InvalidQueueFamilyIndex) {
    // try to look for a queue that supports transfer but not graphics and
    // compute, if not found, default to graphicsComputeQueueFamily
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
      if ((queueFamilies[i].queueFamilyProperties.queueFlags &
           VK_QUEUE_TRANSFER_BIT) != 0 &&
          (queueFamilies[i].queueFamilyProperties.queueFlags &
           VK_QUEUE_GRAPHICS_BIT) == 0 &&
          (queueFamilies[i].queueFamilyProperties.queueFlags &
           VK_QUEUE_COMPUTE_BIT) == 0) {
        device.transferQueueFamily = i;
        break;
      }
    }
    if (device.transferQueueFamily == DeviceVk::InvalidQueueFamilyIndex) {
      for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if ((queueFamilies[i].queueFamilyProperties.queueFlags &
             VK_QUEUE_TRANSFER_BIT) != 0) {
          device.transferQueueFamily = i;
          break;
        }
      }
    }
  }

  assert(device.graphicsComputeQueueFamily !=
         DeviceVk::InvalidQueueFamilyIndex);
  assert(device.computeAsyncQueueFamily != DeviceVk::InvalidQueueFamilyIndex);
  assert(device.transferQueueFamily != DeviceVk::InvalidQueueFamilyIndex);

  if (surface != VK_NULL_HANDLE &&
      device.presentQueueFamily == DeviceVk::InvalidQueueFamilyIndex) {
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
        device.presentQueueFamily = i;
        break;
      }
    }
  }

  assert(device.presentQueueFamily != DeviceVk::InvalidQueueFamilyIndex);
}

bool ContextVk::createDevice(
    std::vector<std::string_view> const& requiredExtensions) {
  // to call vkCreateDevice: 1) extensions 2) features 3) queues
  initializeGenericQueueFamilies(m_instance.instance, m_device, m_surface);

  if (m_device.extensions.enable(requiredExtensions.begin(),
                                 requiredExtensions.end())) {
    return false;
  }

  // Desktop devices should be fine. Android only from Android 14 onwards
  if (m_device.extensions.enable(
          VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME)) {
    m_device.extSwapchainMaintenance1 = true;
  }

  // how many queues we'll request: 1 per family for now
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos(
      DeviceVk::QueueFamilyIndicesCount);
  std::vector<uint32_t> queueFamilyIndices(DeviceVk::QueueFamilyIndicesCount);
  queueFamilyIndices[0] = m_device.graphicsComputeQueueFamily;
  queueFamilyIndices[1] = m_device.computeAsyncQueueFamily;
  queueFamilyIndices[2] = m_device.transferQueueFamily;
  queueFamilyIndices[3] = m_device.presentQueueFamily;

  float queuePriority = 1.0f;

  for (size_t i = 0; i < queueCreateInfos.size(); ++i) {
    queueCreateInfos[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfos[i].pNext = nullptr;
    queueCreateInfos[i].flags = 0;
    queueCreateInfos[i].queueCount = 1;  // TODO Maybe more?
    queueCreateInfos[i].pQueuePriorities = &queuePriority;
    queueCreateInfos[i].queueFamilyIndex = queueFamilyIndices[i];
  }

  // features we require are checked by physicalDeviceSupport and
  // anyMissingCapabilities and in device creation it's deprecated
  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
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
  vkGetDeviceQueue(m_device.device, m_device.graphicsComputeQueueFamily, 0,
                   &m_device.graphicsComputeQueue);
  vkGetDeviceQueue(m_device.device, m_device.computeAsyncQueueFamily, 0,
                   &m_device.computeAsyncQueue);
  vkGetDeviceQueue(m_device.device, m_device.transferQueueFamily, 0,
                   &m_device.transferQueue);
  vkGetDeviceQueue(m_device.device, m_device.presentQueueFamily, 0,
                   &m_device.presentQueue);

  // populate extension function pointers to save one dispatch from the vulkan
  // loader
  volkLoadDevice(m_device.device);

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
  if (!vkCheck(vmaImportVulkanFunctionsFromVolk(&vmaAllocatorCreateInfo, m_vmaVulkanFunctions)) {
    return false;
  }
  vmaAllocatorCreateInfo.pVulkanFunctions = m_vmaVulkanFunctions;
  if (!vkCheck(vmaCreateAllocator(&vmaAllocatorCreateInfo, &m_device.vmaAllocator))) {
    return false;
  }

  return true;
}

ContextResult ContextVk::initializeDrawingContext() {
  uint32_t vulkanApiVersion = VK_API_VERSION_1_3;
  if (!createInstance(vulkanApiVersion)) {
    return ContextResult::Error;
  }
  if (!createSurface()) {
    return ContextResult::Error;
  }

  // for reference, look at guide and
  std::vector<std::string_view> requiredExtensions;
  requiredExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
  if (!selectPhysicalDevice(requiredExtensions)) {
    return ContextResult::Error;
  }
  if (!createDevice(requiredExtensions)) {
    return ContextResult::Error;
  }
  return ContextResult::Success;
}

static bool selectSurfaceFormat(DeviceVk const& device, VkSurfaceKHR surface,
                                bool useHDR, VkSurfaceFormatKHR& outFormat) {
  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device.physicalDevice, surface,
                                       &formatCount, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(device.physicalDevice, surface,
                                       &formatCount, formats.data());

  // Select the best format (from blender)
  static VkSurfaceFormatKHR constexpr preferredFormats[] = {
      {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT},
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
                              VkPresentModeKHR& outPresentMode) {
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

ContextResult ContextVk::recreateSwapchain(bool useHDR) {
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
  VkSurfacePresentScalingCapabilitiesKHR scalingCapabilities{};
  scalingCapabilities.sType =
      VK_STRUCTURE_TYPE_SURFACE_PRESENT_SCALING_CAPABILITIES_KHR;
  VkSurfaceCapabilities2KHR surfaceCapabilities2{};
  surfaceCapabilities2.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR;
  surfaceCapabilities2.pNext = &scalingCapabilities;

  VkSurfacePresentModeKHR presentModeInfo{};
  presentModeInfo.sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_KHR;
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
  m_currentExtent = capabilities.currentExtent;
  if (m_currentExtent.width == UINT32_MAX) {
#ifndef VK_USE_PLATFORM_WAYLAND_KHR
    // special value that means that the size is determined by the extent of
    // the surface
    m_currentExtent.width = 1280;
    m_currentExtent.height = 720;
#else
#error \
    "TODO Wayland surface extent handling taking it from the wayland window info"
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
      m_currentExtent.height = scalingCapabilities.minScaledImageExtent.height;
    }
  }

  assert(m_currentExtent.width > 0);
  assert(m_currentExtent.height > 0);

  // 2. Mark for discard swapchain resources (wait on semaphores and discard
  // old images, to be discarded on the next frame usage)
  FrameDiscard& currentDiscard = m_frameData[m_renderFrame].discard;
  for (SwapchainImage& swapchainImage : m_swapchainImages) {
    currentDiscard.semaphores.push_back(swapchainImage.presentSemaphore);
    swapchainImage.presentSemaphore = VK_NULL_HANDLE;
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
  VkPresentScalingFlagBitsKHR const presentScaling =
      oldSwapchain == VK_NULL_HANDLE ? VK_PRESENT_SCALING_STRETCH_BIT_KHR
                                     : VK_PRESENT_SCALING_ONE_TO_ONE_BIT_KHR;
  VkSwapchainPresentModesCreateInfoKHR presentModesCreateInfo{};
  presentModesCreateInfo.sType =
      VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_KHR;
  presentModesCreateInfo.pNext = nullptr;
  presentModesCreateInfo.presentModeCount = 1;
  presentModesCreateInfo.pPresentModes = &presentMode;

  VkSwapchainPresentScalingCreateInfoKHR presentScalingCreateInfo{};
  presentScalingCreateInfo.sType =
      VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_KHR;
  presentScalingCreateInfo.pNext = &presentModesCreateInfo;
  // scale from top left (min, max)
  presentScalingCreateInfo.scalingBehavior =
      scalingCapabilities.supportedPresentScaling & presentScaling;
  presentScalingCreateInfo.presentGravityX =
      scalingCapabilities.supportedPresentGravityX &
      VK_PRESENT_GRAVITY_MIN_BIT_KHR;
  presentScalingCreateInfo.presentGravityY =
      scalingCapabilities.supportedPresentGravityY &
      VK_PRESENT_GRAVITY_MAX_BIT_KHR;

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
  createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  createInfo.queueFamilyIndexCount = 0;
  createInfo.pQueueFamilyIndices = nullptr;

  if (!vkCheck(vkCreateSwapchainKHR(m_device.device, &createInfo, nullptr,
                                    &m_swapchain))) {
    return ContextResult::Error;
  }

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
      currentDiscard.semaphores.push_back(frame.acquireSemaphore);
    }
    frame.acquireSemaphore = VK_NULL_HANDLE;
  }
  if (oldSwapchain != VK_NULL_HANDLE) {
    currentDiscard.swapchains.push_back(oldSwapchain);
  }

  initializeFrameData();

  m_imageCount = actualImageCount;
  return ContextResult::Success;
}

// TODO multiple frames in flight
bool ContextVk::initializeFrameData() {
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

void ContextVk::destroySwapchainPresentFences(VkSwapchainKHR swapchain) {
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

bool ContextVk::destroySwapchain() {
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
                                VkFence presentFence) {
  if (presentFence) {
    return;
  }
  m_presentFences[swapchain].push_back(presentFence);
  // recycle signaled fences
  for (auto& [otherSwapchain, fences] : m_presentFences) {
    auto it = std::remove_if(
        fences.begin(), fences.end(), [this](const VkFence fence) {
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

VkFence ContextVk::getFence() {
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

void ContextVk::setSwapBufferCallbacks(
    std::function<void(const SwapchainDataVk*)> drawCallback,
    std::function<void(void)> acquireCallback) {
  m_swapBufferDrawCallback = drawCallback;
  m_swapBufferAcquiredCallback = acquireCallback;
}

ContextResult ContextVk::swapBufferRelease() {
  // minimized window doesn't have a swapchain and swapchain image. in this case
  // callback with empty data
  if (m_swapchain == VK_NULL_HANDLE) {
    SwapchainDataVk data{};
    if (m_swapBufferDrawCallback) {
      m_swapBufferDrawCallback(&data);
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

  vkResetFences(m_device.device, 1, &submissionFrame.submissionFence);
  if (m_swapBufferDrawCallback) {
    m_swapBufferDrawCallback(&swapchainData);
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
    std::scoped_lock<std::mutex> lock{m_device.queueMutex};
    VkSwapchainPresentFenceInfoKHR fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR;
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
    recreateSwapchain(useHDRSwapchain);
    return ContextResult::Success;
  }
  if (presentResult != VK_SUCCESS) {
    // TODO log error
    return ContextResult::Error;
  }

  return ContextResult::Success;
}

ContextResult ContextVk::swapBufferAcquire() {
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
#error "TODO"
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
  if (m_swapBufferAcquiredCallback) {
    m_swapBufferAcquiredCallback();
  }

  if (m_swapchain == VK_NULL_HANDLE) {
    // TODO log minimized window
    return ContextResult::Success;
  }

  m_acquiredSwapchainImageIndex = imageIndex;
  return ContextResult::Success;
}

}  // namespace avk