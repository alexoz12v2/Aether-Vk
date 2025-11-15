#include "render/vk/instance-vk.h"

#include <vulkan/vulkan_core.h>

#include "render/vk/common-vk.h"

// std
#include <algorithm>
#include <unordered_set>

namespace avk::vk {

// ------------------------- Static Functions --------------------------------
// some older devices don't have VK_EXT_debug_utils, and we can't reliably
// detect it (see below), hence we prefer enabling VK_EXT_debug_report and pray
// that it works 👌
#ifdef AVK_DEBUG
#  if 0
static VkBool32 VKAPI_PTR
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageTypes,
              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
              [[maybe_unused]] void* pUserData) {
  if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    LOGE << pCallbackData->messageIdNumber
         << " Debug Utils Messenger: Error: " << pCallbackData->pMessageIdName
         << ": " << pCallbackData->pMessage << std::endl;
  } else if (messageSeverity &
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    LOGW << pCallbackData->messageIdNumber
         << " Debug Utils Messenger: Warning: " << pCallbackData->pMessageIdName
         << ": " << pCallbackData->pMessage << std::endl;
  } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    LOGE << pCallbackData->messageIdNumber
         << " Debug Utils Messenger: Information: "
         << pCallbackData->pMessageIdName << ": " << pCallbackData->pMessage
         << std::endl;
  } else if (messageTypes & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) {
    LOGI << pCallbackData->messageIdNumber
         << " Debug Utils Messenger: Performance warning: "
         << pCallbackData->pMessageIdName << ": " << pCallbackData->pMessage
         << std::endl;
  } else if (messageSeverity &
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
    LOGI << pCallbackData->messageIdNumber
         << " Debug Utils Messenger: Verbose: " << pCallbackData->pMessageIdName
         << ": " << pCallbackData->pMessage << std::endl;
  }
  return VK_FALSE;
}
#  endif

static VkBool32 VKAPI_PTR debugReportCallback(
    VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType,
    uint64_t object, size_t location, int32_t messageCode,
    const char *pLayerPrefix, const char *pMessage,
    [[maybe_unused]] void *pUserData) {
  const char *severity = "INFO";
  if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
    severity = "ERROR";
  else if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT)
    severity = "WARNING";
  else if (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)
    severity = "PERF";
  else if (flags & VK_DEBUG_REPORT_DEBUG_BIT_EXT)
    severity = "DEBUG";

  LOGI << "[" << severity << "]"
       << "[" << (pLayerPrefix ? pLayerPrefix : "Unknown") << "] " << pMessage
       << "\n\t"
       << "  ObjectType: [VkDebugReportObjectTypeEXT::]" << objectType
       << "  Object: " << std::hex << object << "  Location: " << location
       << "  Code: " << messageCode << "  Flags: " << flags << std::dec
       << std::endl;

  // Returning VK_FALSE tells Vulkan to continue normally
  return VK_FALSE;
}

#  if 0  // left as reference
// Note:
// <https://docs.vulkan.org/refpages/latest/refpages/source/vkEnumerateInstanceExtensionProperties.html>
// "When pLayerName parameter is NULL, only extensions provided by the Vulkan
// implementation or by implicitly enabled layers are returned."
// implicitly enabled layers = There can be instance extensions enabled by said
// implicit layers, but actually *NOT* supported by the current Vulkan
// implementation. Example: When using RenderDoc, my Xiaomi 22126RN91Y, which
// *doesn't* support `VK_EXT_debug_utils`, reports it as a valid instance when
// enumerating instance extensions with `pLayerName = nullptr`.
// This means that to get actually supported extensions, we need to query
// extensions for everything first, and then filter out extensions provided by
// each layer?
// TODO Problem: This removes also VK_EXT_debug_report, which is supported
// Solution Now: Just pray that  VK_EXT_debug_report is always supported and not
// a sham given by implicit layers True solution: spawn a subprocess with
// environment variable `VK_LOADER_LAYERS_ENABLE=none` (? Untested)
static std::unordered_set<std::string> getDriverOnlyExtensions() {
#    define P "[Instance::DebuggingExtensions] "
  // 1. All layers
  uint32_t layerCount = 0;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  std::vector<VkLayerProperties> layers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, layers.data());

  std::unordered_set<std::string> layerExtensions;

  // 2. Collect extensions from each layer
  for (const auto& layer : layers) {
    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(layer.layerName, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateInstanceExtensionProperties(layer.layerName, &extCount,
                                           exts.data());

    LOGI << P "(layer.layerName)" << std::endl;
    for (VkExtensionProperties const& extName : exts) {
      LOGI << P "  - " << extName.extensionName << std::endl;
    }

    for (auto& e : exts) layerExtensions.insert(e.extensionName);
  }

  // 3. Combined extension list (driver + implicit layers)
  uint32_t extCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
  std::vector<VkExtensionProperties> all(extCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &extCount, all.data());

  // 4. Subtract layer-provided extensions
  std::unordered_set<std::string> driverOnly;
  for (auto& e : all) {
    if (!layerExtensions.count(e.extensionName)) {
      driverOnly.insert(e.extensionName);
    }
  }

  LOGI << P "filtered extensions: (" << driverOnly.size() << ")" << std::endl;
  for (std::string const& extName : driverOnly) {
    LOGI << P "  - " << extName << std::endl;
  }
#    undef P
  return driverOnly;
}
#  endif

#endif

// ----------------------------- Instance Implementation ---------------------

Instance::Instance() AVK_NO_CFI {
  static uint32_t constexpr BaselineVulkanVersion = VK_API_VERSION_1_1;

  VK_CHECK(volkInitialize());

  uint32_t apiVersion;
  VkResult res =
      vkEnumerateInstanceVersion(&apiVersion);  // Vulkan 1.1+ function
  if (res < 0) {
    LOGE << "[Instance] Couldn't query max instance version vulkan"
         << std::endl;
    AVK_EXT_CHECK(false);
  }
  LOGI << "Vulkan instance version Supported: " << VK_VERSION_MAJOR(apiVersion)
       << '.' << VK_VERSION_MINOR(apiVersion) << '.'
       << VK_VERSION_PATCH(apiVersion) << std::endl;
  if (apiVersion < BaselineVulkanVersion) {
    LOGE << "[Instance] API version lower than baseline, aborting" << std::endl;
    AVK_EXT_CHECK(false);
  }

  // 1. initialize instance extensions
  Extensions extensions;
  uint32_t count = 0;
  VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));
  extensions.extensions.resize(count);
  VK_CHECK(vkEnumerateInstanceExtensionProperties(
      nullptr, &count, extensions.extensions.data()));

#ifdef AVK_DEBUG
  // (Android) When VK_LAYER_RENDERDOC_ARM_Capture is detected as a global
  // layer by the system, vkEnumerateInstanceExtensionProperties will report
  // non-existent instance extensions! (Example, on Xiaomi 22126RN91Y,
  // VK_EXT_debug_utils, VK_EXT_validation_features, VK_EXT_layer_settings are
  // *NOT* actually supported)
  LOGI << "[Instance::CreateInstance] List instance extensions: ("
       << extensions.extensions.size() << ")" << std::endl;
  for (VkExtensionProperties const &instExt : extensions.extensions) {
    LOGI << "[Instance::CreateInstance]   - " << instExt.extensionName
         << std::endl;
  }
#endif

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
  // TODO Android: If it's not supported, show an error message "Update to
  // Android 14 or higher" VK_EXT_surface_maintenance1 for pNext present mode
  // and scaling caps
  AVK_EXT_CHECK(extensions.enable(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME));
#ifdef AVK_DEBUG
  // opt: VK_EXT_debug_utils for messenger callbacks
  if (extensions.isSupported(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)) {
    LOGI << "[Instance::CreateInstance] Found VK_EXT_debug_report" << std::endl;
    extensions.enable(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    m_debugReport = true;
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
  std::vector<char const *> validationLayers;
  validationLayers.reserve(16);
  VK_CHECK(vkEnumerateInstanceLayerProperties(&layerCount, nullptr));
  LOGI << "[Instance] Found " << layerCount << " Validation Layers"
       << std::endl;
  std::vector<VkLayerProperties> layerProperties(layerCount);
  VK_CHECK(
      vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data()));
  // Desktop might have lots of layers causing error VK_ERROR_OUT_OF_HOST_MEMORY
  // if all enabled
#  define AVK_DEBUG_VALIDATION_LAYER_ONLY
#  ifdef AVK_DEBUG_VALIDATION_LAYER_ONLY
  char const *const desiredLayer = "VK_LAYER_KHRONOS_validation";
  for (const auto &layerProp : layerProperties) {
    LOGI << "[Instance]  - " << layerProp.layerName << std::endl;
    if (strcmp(layerProp.layerName, desiredLayer) == 0) {
      LOGI << "[Instance]  - ADDED " << layerProp.layerName << std::endl;
      validationLayers.push_back(layerProp.layerName);
    }
  }
#  else
  // RenderDoc uses its own layer, hence don't filter it
  for (const auto &layerProp : layerProperties) {
    LOGI << "[Instance]  - " << layerProp.layerName << std::endl;
    validationLayers.push_back(layerProp.layerName);
  }
#  endif

  createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
  createInfo.ppEnabledLayerNames = validationLayers.data();

  // TODO add check for AGI (need newer phone though) and NVIDIA Insight
  // Graphics
  bool hasRenderdoc = std::any_of(
      validationLayers.cbegin(), validationLayers.cend(),
      [](const char *layer) {
        return std::string(layer).find("RENDERDOC") != std::string::npos;
      });
  if (hasRenderdoc) {
    LOGW << "[Instance::CreateInstance] Since renderdoc was detected, do not "
            "add any debugging extension"
         << std::endl;
    m_debugReport = false;
  }

  // define, if supported, Debug Messenger
  VkDebugReportCallbackCreateInfoEXT debugInfo{};
  if (m_debugReport) {
    debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
    debugInfo.flags =
        VK_DEBUG_REPORT_INFORMATION_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT |
        VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
        VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_DEBUG_BIT_EXT;
    debugInfo.pfnCallback = debugReportCallback;
    debugInfo.pUserData = nullptr;

    // crash, why?
    createInfo.pNext = &debugInfo;
  }
#endif

  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(extensions.enabled.size());
  createInfo.ppEnabledExtensionNames = extensions.enabled.data();
  createInfo.pApplicationInfo = &appInfo;

  VK_CHECK(vkCreateInstance(&createInfo, nullptr, &m_instance));

#ifdef AVK_DEBUG
  auto *const pfnCreateDebugReportCallbackEXT =
      reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(
          vkGetInstanceProcAddr(m_instance, "vkCreateDebugReportCallbackEXT"));
  if (!pfnCreateDebugReportCallbackEXT) {
    LOGW << "[Instance::CreateInstance] While this device should support"
            " VK_EXT_debug_report, 'vkCreateDebugReportCallbackEXT' was found "
            "to be NULL"
         << std::endl;
    m_debugReport = false;
  }
  if (m_debugReport) {
    // this is before volkLoadInstanceOnly, so PFN
    VK_CHECK(pfnCreateDebugReportCallbackEXT(m_instance, &debugInfo, nullptr,
                                             &m_debugCallback));
  }
#endif

  volkLoadInstanceOnly(m_instance);
  LOGI << "[Instance] Correctly created Vulkan Instance Version "
       << VK_VERSION_MAJOR(BaselineVulkanVersion) << '.'
       << VK_VERSION_MINOR(BaselineVulkanVersion) << '.'
       << VK_VERSION_PATCH(BaselineVulkanVersion) << std::endl;
  uint32_t instanceVersion = 0;
  vkEnumerateInstanceVersion(&instanceVersion);
  LOGI << "Effective instance version: " << VK_VERSION_MAJOR(instanceVersion)
       << "." << VK_VERSION_MINOR(instanceVersion) << "."
       << VK_VERSION_PATCH(instanceVersion) << std::endl;
}

Instance::~Instance() noexcept AVK_NO_CFI {
  if (m_instance == VK_NULL_HANDLE) return;
#ifdef AVK_DEBUG
  if (m_debugReport && m_debugCallback != VK_NULL_HANDLE) {
    vkDestroyDebugReportCallbackEXT(m_instance, m_debugCallback, nullptr);
  }
#endif

  vkDestroyInstance(m_instance, nullptr);
  volkFinalize();
}

}  // namespace avk::vk