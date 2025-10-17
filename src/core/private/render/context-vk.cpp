#include "render/context-vk.h"

#include <cassert>
#include <cstddef>
#include <iostream>
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

namespace avk {
ContextVk::~ContextVk() noexcept {
  if (m_swapchain != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(m_device.device);
    m_device.extensionTable->vkDestroySwapchainKHR(m_device.device, m_swapchain,
                                                   nullptr);
    m_swapchain = VK_NULL_HANDLE;
  }

  if (m_device.device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(m_device.device);
    vkDestroyDevice(m_device.device, nullptr);
    m_device.device = VK_NULL_HANDLE;
  }

  if (m_surface != VK_NULL_HANDLE) {
    auto const vkDestroySurfaceKHR = reinterpret_cast<PFN_vkDestroySurfaceKHR>(
        vkGetInstanceProcAddr(m_instance.instance, "vkDestroySurfaceKHR"));
    assert(vkDestroySurfaceKHR);
    vkDestroySurfaceKHR(m_instance.instance, m_surface, nullptr);
    m_surface = VK_NULL_HANDLE;
  }

  if (m_instance.instance != VK_NULL_HANDLE) {
    vkDestroyInstance(m_instance.instance, nullptr);
    m_instance.instance = VK_NULL_HANDLE;
  }
}

ContextVk::ContextVk(ContextVkParams const& params)
    :
#ifdef AVK_OS_WINDOWS
      m_hWindow(params.window)
#elif defined(AVK_OS_MACOS)
#error "TODO"
#elif defined(AVK_OS_ANDROID)
#error "TODO"
#elif defined(AVK_OS_LINUX)
#error "TODO X11 and Wayland"
#else
#error "ADD SUPPORT"
#endif
{
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
  // instance extensions to enable: portabilty, surface, platform surface
  // - (PORTABILITY) Vulkan will consider devices that aren’t fully conformant
  //   such as MoltenVk to be identified as a conformant implementation. When
  //   this happens, use the VkPhysicalDevicePortabilitySubsetPropertiesKHR
  //   extension with the vkGetPhysicalDeviceFeatures2 as detailed below to get
  //   the list of supported/unsupported features.
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

#ifdef AVK_OS_WINDOWS
  if (!m_instance.extensions.enable(VK_KHR_SURFACE_EXTENSION_NAME)) {
    return false;
  }
  if (!m_instance.extensions.enable(VK_KHR_WIN32_SURFACE_EXTENSION_NAME)) {
    return false;
  }
#elif defined(AVK_OS_MACOS)
#error "TODO"
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

  // Warning: VK_MAKE_API_VERSION(0, 1, 3, 2) necessary to use dynamic rendering
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

  // in properties 1.1, we are interested in max descriptor set count, subgroup
  // properties and max allocation
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

  // populate extension function pointers to save one dispatch from the vulkan loader

  m_device.extensionTable = std::make_unique<DeviceVkExtensionTable>();
  m_device.extensionTable->vkDestroySwapchainKHR =
      reinterpret_cast<PFN_vkDestroySwapchainKHR>(
          vkGetDeviceProcAddr(m_device.device, "vkDestroySwapchainKHR"));
  m_device.extensionTable->vkCreateSwapchainKHR =
      reinterpret_cast<PFN_vkCreateSwapchainKHR>(
          vkGetDeviceProcAddr(m_device.device, "vkCreateSwapchainKHR"));
  m_device.extensionTable->vkGetPhysicalDeviceSurfacePresentModesKHR =
      reinterpret_cast<PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>(
          vkGetInstanceProcAddr(m_instance.instance,
                                "vkGetPhysicalDeviceSurfacePresentModesKHR"));

  if (m_device.extSwapchainMaintenance1) {
    m_device.extensionTable->vkGetPhysicalDeviceSurfaceCapabilities2KHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR>(
            vkGetInstanceProcAddr(
                m_instance.instance,
                "vkGetPhysicalDeviceSurfaceCapabilities2KHR"));
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

static bool selectSurfaceFormat(VkPhysicalDevice physicalDevice,
                                VkSurfaceKHR surface, bool useHDR,
                                VkSurfaceFormatKHR& outFormat) {
  return false;
}

static bool selectPresentMode(bool vsyncOff, VkPhysicalDevice physicalDevice,
                              VkSurfaceKHR surface,
                              VkPresentModeKHR& outPresentMode) {
  return false;
}

ContextResult ContextVk::recreateSwapchain(bool useHDR) {
  // VkSurfaceCapabilitiesKHR.currentTransform =
  // VkSwapchainCreateInfoKHR.preTransform
}

}  // namespace avk