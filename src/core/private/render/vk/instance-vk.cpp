#include "render/vk/instance-vk.h"

#include <vulkan/vulkan_core.h>

#include "render/vk/common-vk.h"

namespace avk::vk {

// ------------------------- Static Functions --------------------------------
// TODO better logging
#ifdef AVK_DEBUG
static VkBool32 VKAPI_PTR
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageTypes,
              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
              [[maybe_unused]] void* pUserData) {
  if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    std::cout << pCallbackData->messageIdNumber
              << " Debug Utils Messenger: Error: "
              << pCallbackData->pMessageIdName << ": "
              << pCallbackData->pMessage << std::endl;
  } else if (messageSeverity &
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    std::cout << pCallbackData->messageIdNumber
              << " Debug Utils Messenger: Warning: "
              << pCallbackData->pMessageIdName << ": "
              << pCallbackData->pMessage << std::endl;
  } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    std::cout << pCallbackData->messageIdNumber
              << " Debug Utils Messenger: Information: "
              << pCallbackData->pMessageIdName << ": "
              << pCallbackData->pMessage << std::endl;
  } else if (messageTypes & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) {
    std::cout << pCallbackData->messageIdNumber
              << " Debug Utils Messenger: Performance warning: "
              << pCallbackData->pMessageIdName << ": "
              << pCallbackData->pMessage << std::endl;
  } else if (messageSeverity &
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
    std::cout << pCallbackData->messageIdNumber
              << " Debug Utils Messenger: Verbose: "
              << pCallbackData->pMessageIdName << ": "
              << pCallbackData->pMessage << std::endl;
  }
  return VK_FALSE;
}
#endif

// ----------------------------- Instance Implementation ---------------------

Instance::Instance() AVK_NO_CFI {
  static uint32_t constexpr BaselineVulkanVersion = VK_API_VERSION_1_1;

  AVK_EXT_CHECK(volkInitialize());

  // 1. initialize instance extensions
  Extensions extensions;
  uint32_t count = 0;
  VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));
  extensions.extensions.resize(count);
  VK_CHECK(vkEnumerateInstanceExtensionProperties(
      nullptr, &count, extensions.extensions.data()));

  // 2. Add baseline extensions
  // opt: VK_KHR_portability_enumeration
  // Instance extension to support instances over a translation layer
  // (Metal on Apple devices)
  if (extensions.isSupported(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
    extensions.enable(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    m_portabilityEnumeration = true;
  }
  // VK_KHR_get_surface_capabilities2 for extensible queries on surfaces
  AVK_EXT_CHECK(
      extensions.enable(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME));
  // VK_EXT_surface_maintenance1 for pNext present mode and scaling caps
  AVK_EXT_CHECK(extensions.enable(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME));
#ifdef AVK_DEBUG
  // opt: VK_EXT_debug_utils for messenger callbacks
  if (extensions.isSupported(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
    extensions.enable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    m_debugUtils = true;
  }
#endif
  // VK_KHR_surface
  AVK_EXT_CHECK(extensions.enable(VK_KHR_SURFACE_EXTENSION_NAME));
#ifdef VK_USE_PLATFORM_WIN32_KHR
  AVK_EXT_CHECK(extensions.enable(VK_KHR_WIN32_SURFACE_EXTENSION_NAME));
#elif defined(VK_USE_PLATFORM_METAL_EXT)
  AVK_EXT_CHECK(extensions.enable(VK_EXT_METAL_SURFACE_EXTENSION_NAME));
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
  AVK_EXT_CHECK(extensions.enable(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME));
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
  AVK_EXT_CHECK(extensions.enable(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME));
#else
#  error "Not Supported"
#endif
  // VK_EXT_swapchain_colorspace to add support for non more color spaces
  AVK_EXT_CHECK(extensions.enable(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME));

  // -- Deprecation State --
  // (Core from 1.1) VK_KHR_external_fence_capabilities
  // (Core from 1.1) VK_KHR_external_memory_capabilities
  // (Core from 1.1) VK_KHR_external_semaphore_capabilities
  // (Core from 1.1) VK_KHR_get_physical_device_properties2

  // 3. Create Instance
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "AetherVk";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "AetherVk";
  appInfo.apiVersion = BaselineVulkanVersion;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.flags = m_portabilityEnumeration
                         ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
                         : 0;
#ifdef AVK_DEBUG
  // define validation layers
  uint32_t layerCount = 0;
  std::vector<char const*> validationLayers;
  validationLayers.reserve(16);
  char const* const desiredLayer = "VK_LAYER_KHRONOS_validation";
  VK_CHECK(vkEnumerateInstanceLayerProperties(&layerCount, nullptr));
  std::vector<VkLayerProperties> layerProperties(layerCount);
  VK_CHECK(
      vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data()));
  for (const auto& layerProp : layerProperties) {
    if (strcmp(layerProp.layerName, desiredLayer)) {
      validationLayers.push_back(layerProp.layerName);
    }
  }

  createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
  createInfo.ppEnabledLayerNames = validationLayers.data();

  // define, if supported, Debug Messenger
  VkDebugUtilsMessengerCreateInfoEXT messengerCreateInfo{};
  if (m_debugUtils) {
    messengerCreateInfo.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    messengerCreateInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messengerCreateInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messengerCreateInfo.pfnUserCallback = debugCallback;
    messengerCreateInfo.pUserData = nullptr;

    createInfo.pNext = &messengerCreateInfo;
  }
#endif

  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(extensions.enabled.size());
  createInfo.ppEnabledExtensionNames = extensions.enabled.data();

  VK_CHECK(vkCreateInstance(&createInfo, nullptr, &m_instance));

#ifdef AVK_DEBUG
  if (m_debugUtils) {
    PFN_vkCreateDebugUtilsMessengerEXT const pfnCreateDebugUtilsMessengerEXT =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            m_instance, "vkCreateDebugUtilsMessengerEXT");
    VK_CHECK(pfnCreateDebugUtilsMessengerEXT(m_instance, &messengerCreateInfo,
                                             nullptr, &m_messenger));
  }
#endif

  volkLoadInstanceOnly(m_instance);
}

Instance::~Instance() noexcept AVK_NO_CFI {
  if (m_instance == VK_NULL_HANDLE) return;
  if (m_debugUtils && m_messenger != VK_NULL_HANDLE) {
    PFN_vkDestroyDebugUtilsMessengerEXT const pfnDestroyDebugUtilsMessengerEXT =
        (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            m_instance, "vkDestroyDebugUtilsMessengerEXT");
    pfnDestroyDebugUtilsMessengerEXT(m_instance, m_messenger, nullptr);
  }

  vkDestroyInstance(m_instance, nullptr);
}

}  // namespace avk::vk