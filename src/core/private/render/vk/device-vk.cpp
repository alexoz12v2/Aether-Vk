#include "render/vk/device-vk.h"

// os specific
#if defined(__APPLE__)
#  include <TargetConditionals.h>
#endif

// core
#include "os/avk-core-macros.h"
#include "render/vk/common-vk.h"

// std and external
#include <algorithm>
#include <cassert>
#include <memory>
#include <unordered_set>
#include <vector>

using namespace avk;
using namespace avk::vk;

// TODO prefer debug_utils on anything different from Android,
// TODO insert optional support for debug names in object creation

// -- Used Device Extensions Deprecation State --
// (1.1) `VK_KHR_maintenance1`
// (1.1) `VK_KHR_maintenance2`
// (1.1) `VK_KHR_maintenance3`
// (1.1) `VK_KHR_dedicated_allocation` (through VMA)
// (1.1) `VK_KHR_bind_memory2` (through VMA)
// (1.1) `VK_KHR_sampler_ycbcr_conversion`
// (1.1) `VK_KHR_external_memory`
// (1.1) `VK_KHR_external_fence`
// (1.1) `VK_KHR_external_semaphore`
// (1.1) `VK_KHR_shader_draw_parameters`
// (1.1) `VK_KHR_get_memory_requirements2` (through VMA)

// ---------------------------------------------------------------------------

struct OptionalFeatures {
  // placed in order of priority
  bool textureCompressionASTC_LDR;
  bool textureCompressionBC;
  bool textureCompressionETC2;

  bool swapchainMaintenance1;
  bool memoryBudget;

  bool isSoC;
};

// ---------------------------------------------------------------------------

static void debugPrintDeviceExtensions(
    std::vector<char const *> const &extensions) {
#define PREFIX "[Device::createDevice::debugPrintDeviceExtensions] "
  LOGI << PREFIX "Enabled Extensions: " << extensions.size() << std::endl;
  for (char const *extName : extensions) {
    LOGI << PREFIX "  " << extName << std::endl;
  }
#undef PREFIX
}

static void fillRequiredExtensions(
    std::vector<char const *> &requiredExtensions) {
#if defined(__linux__) || defined(__ANDROID__)
  requiredExtensions.push_back(VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
#if defined(__linux__)
  requiredExtensions.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
#endif
#if defined(__ANDROID__)
  requiredExtensions.push_back(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
  requiredExtensions.push_back(
      VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME);
#endif
#if defined(_WIN32)
  requiredExtensions.push_back(VK_KHR_EXTERNAL_FENCE_WIN32_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
  requiredExtensions.push_back(VK_EXT_FULL_SCREEN_EXCLUSIVE_EXTENSION_NAME);
#endif
#if defined(__APPLE__)
  requiredExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  requiredExtensions.push_back(VK_EXT_EXTERNAL_MEMORY_METAL_EXTENSION_NAME);
#endif
  requiredExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
  // see anyRequiredFeatureMissing for more information. Basically, if you have
  // a RENDERDOC Vulkan Layer present and bufferDeviceAddressCaptureReplay is
  // not a supported feature, bufferAddress will be filtered out
  // requiredExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  requiredExtensions.push_back(
      VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME);
  requiredExtensions.push_back(VK_EXT_INLINE_UNIFORM_BLOCK_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
  requiredExtensions.push_back(
      VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME);
}

static std::vector<std::string> anyRequiredFeaturesMissing(VkPhysicalDevice dev)
    AVK_NO_CFI {
  std::vector<std::string> missing;
  missing.reserve(16);
  // Prepare Structs to query Necessary Features
  VkPhysicalDeviceFeatures2 features{};
  VkPhysicalDeviceVulkanMemoryModelFeaturesKHR vulkanMemoryModel{};
  VkPhysicalDeviceUniformBufferStandardLayoutFeaturesKHR ubStandardLayout{};
  VkPhysicalDeviceTimelineSemaphoreFeaturesKHR timelineSemaphoreFeat{};
  VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bufferDeviceAddressFeat{};
  VkPhysicalDeviceInlineUniformBlockFeaturesEXT inlineUniformFeat{};
  VkPhysicalDeviceShaderDrawParametersFeatures shaderDrawFeat{};

  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  vulkanMemoryModel.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR;
  ubStandardLayout.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES_KHR;
  timelineSemaphoreFeat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR;
  bufferDeviceAddressFeat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
  inlineUniformFeat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES_EXT;
  shaderDrawFeat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES;

  features.pNext = &vulkanMemoryModel;
  vulkanMemoryModel.pNext = &ubStandardLayout;
  ubStandardLayout.pNext = &timelineSemaphoreFeat;
  timelineSemaphoreFeat.pNext = &bufferDeviceAddressFeat;
  bufferDeviceAddressFeat.pNext = &inlineUniformFeat;
  inlineUniformFeat.pNext = &shaderDrawFeat;

  vkGetPhysicalDeviceFeatures2(dev, &features);

  if (!vulkanMemoryModel.vulkanMemoryModel)
    missing.push_back("vulkanMemoryModel");
  if (!ubStandardLayout.uniformBufferStandardLayout)
    missing.push_back("uniformBufferStandardLayout");
  if (!timelineSemaphoreFeat.timelineSemaphore)
    missing.push_back("timelineSemaphore");
  if (!inlineUniformFeat.inlineUniformBlock)
    missing.push_back("inlineUniformBlock");
  if (!shaderDrawFeat.shaderDrawParameters)
    missing.push_back("shaderDrawParameters");
  // TODO remove
  if (!features.features.geometryShader) missing.push_back("geometryShader");

  // While bufferDeviceAddress is useful when you are doing GPU-driven
  // rendering or complex setups on compute shaders, as it allows you to work
  // with raw pointers to GPU memory instead of relying on indexing schemes,
  // If you have a RENDERDOC implicit Vulkan Layer active in the current
  // session, it disables this feature if bufferDeviceAddressCaptureReplay is
  // not supported, altering the "normal" result of
  // `vkGetPhysicalDeviceFeatures2`. Since my Xiaomi 22126RN91Y, Android 14,
  // doesn't support bufferDeviceAddressCaptureReplay, we'll try to avoid
  // this extension
  // if (!bufferDeviceAddressFeat.bufferDeviceAddress)
  //   missing.push_back("bufferDeviceAddress");

  return missing;
}

static void setOptionalFeaturesForDevice(
    VkPhysicalDevice dev, OptionalFeatures &outOptFeatures) AVK_NO_CFI {
  VkPhysicalDeviceFeatures2 features{};
  VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT swapMain1Feat{};

  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  swapMain1Feat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;

  features.pNext = &swapMain1Feat;

  vkGetPhysicalDeviceFeatures2(dev, &features);

  outOptFeatures.swapchainMaintenance1 = swapMain1Feat.swapchainMaintenance1;
  outOptFeatures.textureCompressionASTC_LDR =
      features.features.textureCompressionASTC_LDR;
  outOptFeatures.textureCompressionBC = features.features.textureCompressionBC;
  outOptFeatures.textureCompressionETC2 =
      features.features.textureCompressionETC2;
}

// buffers there too can be beneficial
static bool isSoC(VkPhysicalDevice dev, VkPhysicalDeviceType deviceType) {
  // assumes gpu is integrated. We say that it's a SoC chip if
  // - is a mobile platform
  // - OR all memory heaps are DEVICE_LOCAL_BIT, with all types with
  // DEVICE_LOCAL_BIT
  if (deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
    return false;
  }
#if defined(__ANDROID__)
  return true;
#elif defined(__APPLE__) && TARGET_OS_IOS
  return true;
#endif
  // for desktop, check whether the integrated GPU has all heaps with
  // DEVICE_LOCAL_BIT, with all types with the DEVICE_LOCAL_BIT
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(dev, &props);
  // - if all heaps are device local
  for (uint32_t heapIndex = 0; heapIndex < props.memoryHeapCount; ++heapIndex) {
    if (!(props.memoryHeaps[heapIndex].flags &
          VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
      return false;
    }
    if (props.memoryHeaps[heapIndex].flags &
        VK_MEMORY_HEAP_MULTI_INSTANCE_BIT) {
      return false;
    }
  }

  return true;
}

static void setCompressedFormats(
    VkPhysicalDevice chosen, OptionalFeatures &outOptFeatures,
    utils::SampledImageCompressedFormats &comprFormats) AVK_NO_CFI {
  LOGI << "[Device::choosePhysicalDevice::setCompressedFormats]" << std::endl;
  VkFormat proposedLinear = VK_FORMAT_UNDEFINED;
  VkFormat proposedsRGB = VK_FORMAT_UNDEFINED;
  VkFormatProperties2KHR linearFormatProperties{};
  linearFormatProperties.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2_KHR;
  VkFormatProperties2KHR sRGBFormatProperties{};
  sRGBFormatProperties.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2_KHR;

  VkFormatFeatureFlags const sampled_BlitSrc_Transfer =
      VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
  // now set extensions to be used on create info and populate comporession
  // formats
  if (outOptFeatures.textureCompressionASTC_LDR) {
    proposedLinear = VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
    proposedsRGB = VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
  } else if (outOptFeatures.textureCompressionBC) {
    // BPTC Texture Compression for unsigned, normalized images (BC7)
    // https://wikis.khronos.org/opengl/BPTC_Texture_Compression
    proposedLinear = VK_FORMAT_BC7_UNORM_BLOCK;
    proposedsRGB = VK_FORMAT_BC7_SRGB_BLOCK;
  } else if (outOptFeatures.textureCompressionETC2) {
    proposedLinear = VK_FORMAT_EAC_R11_UNORM_BLOCK;
    proposedsRGB = VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
  }

  if (proposedLinear != VK_FORMAT_UNDEFINED &&
      proposedsRGB != VK_FORMAT_UNDEFINED) {
    vkGetPhysicalDeviceFormatProperties2(chosen, proposedLinear,
                                         &linearFormatProperties);
    vkGetPhysicalDeviceFormatProperties2(chosen, proposedsRGB,
                                         &sRGBFormatProperties);
    if ((linearFormatProperties.formatProperties.linearTilingFeatures &
         sampled_BlitSrc_Transfer) &&
        (linearFormatProperties.formatProperties.optimalTilingFeatures &
         sampled_BlitSrc_Transfer)) {
      comprFormats.linear_R = proposedLinear;
      if (proposedLinear != VK_FORMAT_EAC_R11_UNORM_BLOCK) {  // special case
        comprFormats.linear_RGB = proposedLinear;
      } else {
        vkGetPhysicalDeviceFormatProperties2(
            chosen, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, &linearFormatProperties);
        if ((linearFormatProperties.formatProperties.linearTilingFeatures &
             sampled_BlitSrc_Transfer) &&
            (linearFormatProperties.formatProperties.optimalTilingFeatures &
             sampled_BlitSrc_Transfer)) {
          comprFormats.linear_RGB = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
        }
      }
    }

    if ((sRGBFormatProperties.formatProperties.linearTilingFeatures &
         sampled_BlitSrc_Transfer) &&
        (sRGBFormatProperties.formatProperties.optimalTilingFeatures &
         sampled_BlitSrc_Transfer)) {
      comprFormats.sRGB_RGBA = VK_FORMAT_BC7_SRGB_BLOCK;
    }
  }
}

[[nodiscard]] static VkPhysicalDevice choosePhysicalDevice(
    Instance *instance, Extensions &outExtensions,
    OptionalFeatures &outOptFeatures,
    utils::SampledImageCompressedFormats &comprFormats) AVK_NO_CFI {
  assert(instance);
  uint32_t physicalDeviceCount = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance->handle(), &physicalDeviceCount,
                                      nullptr));
  AVK_EXT_CHECK(physicalDeviceCount);
  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
  VK_CHECK(vkEnumeratePhysicalDevices(instance->handle(), &physicalDeviceCount,
                                      physicalDevices.data()));
  LOGI << "[Device::choosePhysicalDevice] Enumerated " << physicalDeviceCount
       << " Vulkan Capable Devices" << std::endl;

  VkPhysicalDevice chosen = VK_NULL_HANDLE;
  uint32_t maxScoreSoFar = 0;

  // prepare list of required extensions. We'll note the ones which we will use,
  // but won't activate explicitly since we are on Vulkan 1.1 and have been
  // promoted
  std::vector<char const *> requiredExtensions;
  requiredExtensions.reserve(64);
  fillRequiredExtensions(requiredExtensions);
  std::vector<VkExtensionProperties> extensionProperties;
  extensionProperties.reserve(256);

  for (VkPhysicalDevice dev : physicalDevices) {
    LOGI << "[Device::choosePhysicalDevice] Examining Physical Device "
         << std::hex << dev << std::dec << std::endl;
    // check if required extensions are supported
    uint32_t score = 0;
    uint32_t devExtCount = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(dev, nullptr, &devExtCount,
                                                  nullptr));
    extensionProperties.clear();
    extensionProperties.resize(devExtCount);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(dev, nullptr, &devExtCount,
                                                  extensionProperties.data()));
    for (char const *requiredExtName : requiredExtensions) {
      auto const it =
          std::find_if(extensionProperties.cbegin(), extensionProperties.cend(),
                       [requiredExtName](VkExtensionProperties const &prop) {
                         return strncmp(prop.extensionName, requiredExtName,
                                        VK_MAX_EXTENSION_NAME_SIZE) == 0;
                       });
      if (it == extensionProperties.cend()) {
        continue;
      }
    }
    LOGI << "[Device::choosePhysicalDevice] Physical Device " << std::hex << dev
         << std::dec << " Supports all required extensions" << std::endl;

    // check if required features are supported
    if (auto missing = anyRequiredFeaturesMissing(dev); !missing.empty()) {
      LOGW << "[Device::choosePhysicalDevice] device " << std::hex << dev
           << std::dec << "Misses features: ";
      for (std::string const &feat : missing) {
        LOGW << "\n\t" << feat << '\n';
      }
      LOGW << std::flush;
      continue;
    }
    LOGI << "[Device::choosePhysicalDevice] Physical Device " << std::hex << dev
         << std::dec << " Supports all required Features " << std::endl;

    // evaluate device type
    VkPhysicalDeviceProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkGetPhysicalDeviceProperties2(dev, &props);

    switch (props.properties.deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        LOGI << "[Device::choosePhysicalDevice] Physical Device " << std::hex
             << dev << std::dec << " VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"
             << std::endl;
        score += 400;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        LOGI << "[Device::choosePhysicalDevice] Physical Device " << std::hex
             << dev << std::dec << " VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU"
             << std::endl;
        score += 300;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        score += 200;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_CPU:
        continue;  // reject CPUs
        break;
      default:
        continue;  // shouldn't reach here
        break;
    }

    // if valid device, compare its score
    if (score > maxScoreSoFar) {
      LOGI << "[Device::choosePhysicalDevice] Physical Device " << std::hex
           << dev << std::dec << " Highest Score" << std::endl;
      chosen = dev;
      setOptionalFeaturesForDevice(dev, outOptFeatures);
      outOptFeatures.isSoC = isSoC(dev, props.properties.deviceType);
    }
  }

  AVK_EXT_CHECK(chosen != VK_NULL_HANDLE);

  setCompressedFormats(chosen, outOptFeatures, comprFormats);

  uint32_t devExtCount = 0;
  VK_CHECK(vkEnumerateDeviceExtensionProperties(chosen, nullptr, &devExtCount,
                                                nullptr));
  outExtensions.extensions.resize(devExtCount);
  VK_CHECK(vkEnumerateDeviceExtensionProperties(
      chosen, nullptr, &devExtCount, outExtensions.extensions.data()));
  for (char const *extName : requiredExtensions) {
    if (!outExtensions.enable(extName)) {
      LOGE << "[Device::CreateDevice from chosen] Extension '" << extName
           << "' is missing" << std::endl;
      AVK_EXT_CHECK(false);
    }
  }
  // if you can have present fences, then enable them
  if (outOptFeatures.swapchainMaintenance1) {
    if (!outExtensions.enable(VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME)) {
      LOGE << "[Device::CreateDevice from chosen] swapchainMaintenance1 "
              "feature enabled but extension not found"
           << std::endl;
      AVK_EXT_CHECK(false);
    }
  }
  // VK_EXT_memory_budget allows VMA library to be more precise when estimating
  // memory budget
  if (outExtensions.isSupported(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
    outExtensions.enable(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    outOptFeatures.memoryBudget = true;
    LOGI << "[Device::choosePhysicalDevice] VK_EXT_memory_budget supported, "
            "creating VMA Allocator with "
            "VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT"
         << std::endl;
  }

  LOGI << "[Device::choosePhysicalDevice] Physical Device " << std::hex
       << chosen << std::dec << " chosen" << std::endl;
  return chosen;
}

static bool getPresentationSupport(
    [[maybe_unused]] VkInstance instance,
    [[maybe_unused]] VkPhysicalDevice physicalDevice,
    [[maybe_unused]] uint32_t index,
    [[maybe_unused]] Surface const *surface) AVK_NO_CFI {
#ifdef VK_USE_PLATFORM_WIN32_KHR
  auto *pfnGetPhysicalDeviceWin32PresentationSupportKHR =
      reinterpret_cast<PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR>(
          vkGetInstanceProcAddr(
              instance, "vkGetPhysicalDeviceWin32PresentationSupportKHR"));
  return pfnGetPhysicalDeviceWin32PresentationSupportKHR(physicalDevice, index);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
  auto *pfnGetPhysicalDeviceWaylandPresentationSupportKHR =
      reinterpret_cast<PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR>(
          vkGetInstanceProcAddr(
              instance, "vkGetPhysicalDeviceWaylandPresentationSupportKHR"));
  return pfnGetPhysicalDeviceWaylandPresentationSupportKHR(
      physicalDevice, index, surface->display());
#elif defined(VK_USE_PLATFORM_METAL_EXT) || defined(VK_USE_PLATFORM_ANDROID_KHR)
  return true;
#else
#  error "Add support for this WSI platform"
#endif
}

/// For now, support finding only the one queue family and queue which supports
/// presentation
/// Note: Queue Priorities on each element still to populate
static std::vector<VkDeviceQueueCreateInfo> newDeviceQueuesCreateInfos(
    VkInstance instance, VkPhysicalDevice physicalDevice,
    Surface const *surface,
    utils::QueueFamilyMap &outQueueFamilies) AVK_NO_CFI {
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  queueCreateInfos.reserve(4);
  queueCreateInfos.push_back({});
  VkDeviceQueueCreateInfo &createInfo = queueCreateInfos.back();
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  createInfo.queueCount = 1;

  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice, &count, nullptr);
  std::vector<VkQueueFamilyProperties2> queueProperties{count};
  for (VkQueueFamilyProperties2 &prop : queueProperties) {
    prop.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2_KHR;
  }
  vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice, &count,
                                            queueProperties.data());

  uint32_t familyIndex = 0;
  for (VkQueueFamilyProperties2 const &props : queueProperties) {
    uint32_t const index = familyIndex++;
    bool const supportsPresent =
        getPresentationSupport(instance, physicalDevice, index, surface);
    bool const graphicsComputeTransfer =
        props.queueFamilyProperties.queueFlags &
        (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT);
    if (supportsPresent && graphicsComputeTransfer) {
      outQueueFamilies.universalGraphics = index;
      familyIndex = index;
      break;
    }
  }
  AVK_EXT_CHECK(familyIndex < queueProperties.size());

  createInfo.queueFamilyIndex = familyIndex;
  return queueCreateInfos;
}

static VkDevice createDevice(VkInstance instance,
                             VkPhysicalDevice physicalDevice,
                             Extensions const &extensions,
                             OptionalFeatures const &optFeatures,
                             Surface const *surface,
                             utils::QueueFamilyMap &queueFamilies) AVK_NO_CFI {
  // enable all the necessary features
  VkPhysicalDeviceFeatures2KHR features{};
  VkPhysicalDeviceVulkanMemoryModelFeaturesKHR vulkanMemoryModel{};
  VkPhysicalDeviceUniformBufferStandardLayoutFeaturesKHR ubStandardLayout{};
  VkPhysicalDeviceTimelineSemaphoreFeaturesKHR timelineSemaphoreFeat{};
  VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bufferDeviceAddressFeat{};
  VkPhysicalDeviceInlineUniformBlockFeaturesEXT inlineUniformFeat{};
  VkPhysicalDeviceShaderDrawParametersFeatures shaderDrawFeat{};
  VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT swapchainMaintenance1Feat{};

  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR;
  vulkanMemoryModel.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR;
  ubStandardLayout.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES_KHR;
  timelineSemaphoreFeat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR;
  bufferDeviceAddressFeat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR;
  inlineUniformFeat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES_EXT;
  shaderDrawFeat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES;
  swapchainMaintenance1Feat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;

  features.pNext = &vulkanMemoryModel;
  vulkanMemoryModel.pNext = &ubStandardLayout;
  ubStandardLayout.pNext = &timelineSemaphoreFeat;
  timelineSemaphoreFeat.pNext = &bufferDeviceAddressFeat;
  bufferDeviceAddressFeat.pNext = &inlineUniformFeat;
  inlineUniformFeat.pNext = &shaderDrawFeat;
  if (optFeatures.swapchainMaintenance1) {
    shaderDrawFeat.pNext = &swapchainMaintenance1Feat;
  }

  // WARNING: Keep in sync with functions
  //   `anyRequiredFeaturesMissing` and `setOptionalFeaturesForDevice`
  vulkanMemoryModel.vulkanMemoryModel = VK_TRUE;
  ubStandardLayout.uniformBufferStandardLayout = VK_TRUE;
  bufferDeviceAddressFeat.bufferDeviceAddress = VK_TRUE;
  timelineSemaphoreFeat.timelineSemaphore = VK_TRUE;
  inlineUniformFeat.inlineUniformBlock = VK_TRUE;
  shaderDrawFeat.shaderDrawParameters = VK_TRUE;
  if (optFeatures.swapchainMaintenance1) {
    swapchainMaintenance1Feat.swapchainMaintenance1 = VK_TRUE;
  }
  if (optFeatures.textureCompressionASTC_LDR) {
    features.features.textureCompressionASTC_LDR = VK_TRUE;
  }
  if (optFeatures.textureCompressionBC) {
    features.features.textureCompressionBC = VK_TRUE;
  }
  if (optFeatures.textureCompressionETC2) {
    features.features.textureCompressionETC2 = VK_TRUE;
  }
  // TODO remove
  features.features.geometryShader = VK_TRUE;

  // Create the device::General Setup
  VkDevice device = VK_NULL_HANDLE;
  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.pNext = &features;
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(extensions.enabled.size());
  createInfo.ppEnabledExtensionNames = extensions.enabled.data();

  // Create the device::Queues (single)
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos =
      newDeviceQueuesCreateInfos(instance, physicalDevice, surface,
                                 queueFamilies);
  float const highestPriority = 1.0f;
  for (VkDeviceQueueCreateInfo &queueCreateInfo : queueCreateInfos) {
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &highestPriority;
  }

  createInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());
  createInfo.pQueueCreateInfos = queueCreateInfos.data();

  // TODO: If bufferDeviceAddress and bufferDeviceAddressCaptureReplay are
  // both supported features, add that as an optional extension
  // (just for renderdoc though, not to be used in code)

  VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device));
  debugPrintDeviceExtensions(extensions.enabled);
  return device;
}

static VmaAllocator newVmaAllocator(VkInstance instance,
                                    uint32_t vulkanApiVersion,
                                    VkPhysicalDevice physicalDevice,
                                    VkDevice device,
                                    VmaVulkanFunctions *vmaVulkanFunctions,
                                    bool memoryBudget) AVK_NO_CFI {
  assert(vulkanApiVersion >= VK_API_VERSION_1_1);
  VmaAllocatorCreateInfo allocatorCreateInfo{};
  // `VK_KHR_dedicated_allocation` promoted from 1.1
  allocatorCreateInfo.flags |=
      VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
  // `VK_KHR_buffer_device_address` in use
  // (TODO better with proper extension detection or leave it like this)
  // allocatorCreateInfo.flags |=
  // VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT; `VK_KHR_bind_memory2`
  // promoted from 1.1
  allocatorCreateInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
  // `VK_KHR_EXTERNAL_MEMORY_WIN32` is the only exporting extension supported by
  // VMA, so, for consistency, I'd rather handle it myself
  if (memoryBudget) {
    allocatorCreateInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
  }
  allocatorCreateInfo.instance = instance;
  allocatorCreateInfo.device = device;
  allocatorCreateInfo.physicalDevice = physicalDevice;
  allocatorCreateInfo.vulkanApiVersion = vulkanApiVersion;

  VK_CHECK(vmaImportVulkanFunctionsFromVolk(&allocatorCreateInfo,
                                            vmaVulkanFunctions));
  allocatorCreateInfo.pVulkanFunctions = vmaVulkanFunctions;

  VmaAllocator allocator = VK_NULL_HANDLE;
  VK_CHECK(vmaCreateAllocator(&allocatorCreateInfo, &allocator));
  return allocator;
}

// ---------------------------------------------------------------------------

namespace avk::vk {

Device::Device(Instance *instance, Surface *surface) AVK_NO_CFI
    : m_deps({instance, surface}) {
  assert(instance && surface);
  Extensions extensions;
  OptionalFeatures optFeatures{};

  // 1. Select physical device which supports baseline and bookkeeping of opt
  m_physicalDevice = choosePhysicalDevice(m_deps.instance, extensions,
                                          optFeatures, m_comprFormats);
  m_isSoC = optFeatures.isSoC;
  m_swapchainMaintenance1 = optFeatures.swapchainMaintenance1;

  // 2. Device creation, extract graphics/compute/transfer/present queue, load
  // table
  LOGI << "[Device] Creating Device" << std::endl;
  m_device =
      createDevice(m_deps.instance->handle(), m_physicalDevice, extensions,
                   optFeatures, m_deps.surface, m_queueFamilies);
  LOGI << "[Device] Created Device " << std::hex << m_device << std::dec
       << " | Selected GRAPHICS Queue Family as "
       << m_queueFamilies.universalGraphics << std::endl;
  LOGI << "[Device] Allocating Volk Device Function Table" << std::endl;
  m_table = std::make_unique<VolkDeviceTable>();
  AVK_EXT_CHECK(m_table);
  memset(m_table.get(), 0, sizeof(VolkDeviceTable));
  volkLoadDeviceTable(m_table.get(), m_device);
  LOGI << "[Device] Getting Queue 0 From GRAPHICS" << std::endl;
  // on my device, Xiaomi 22126RN91Y, Mali-G62 MC2, vkGetDeviceQueue2 returns 0
  m_table->vkGetDeviceQueue(m_device, m_queueFamilies.universalGraphics, 0,
                            &m_queue);
  LOGI << "[Device] Got Queue " << std::hex << m_queue << std::dec << std::endl;

  // memory heap information for VMA (Memory Budget Tracking)
  VkPhysicalDeviceMemoryProperties2 memoryProperties{};
  memoryProperties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
  vkGetPhysicalDeviceMemoryProperties2(m_physicalDevice, &memoryProperties);
  m_heapBudgets.resize(memoryProperties.memoryProperties.memoryHeapCount);
  memset(m_heapBudgets.data(), 0,
         sizeof(VmaBudget) * memoryProperties.memoryProperties.memoryHeapCount);

  // 3. create allocator and eventually pool
  LOGI << "[Device] Creating VMA Allocator" << std::endl;
  m_vmaVulkanFunctions = std::make_unique<VmaVulkanFunctions>();
  memset(m_vmaVulkanFunctions.get(), 0, sizeof(VmaVulkanFunctions));
  m_vmaAllocator = newVmaAllocator(
      instance->handle(), instance->vulkanApiVersion(), m_physicalDevice,
      m_device, m_vmaVulkanFunctions.get(), optFeatures.memoryBudget);
}

Device::~Device() noexcept AVK_NO_CFI {
  if (!*this) {
    return;
  }

  m_table->vkDeviceWaitIdle(m_device);
  vmaDestroyAllocator(m_vmaAllocator);
  m_table->vkDestroyDevice(m_device, nullptr);
}

void Device::refreshMemoryBudgets(uint32_t frameIndex) {
  vmaSetCurrentFrameIndex(m_vmaAllocator, frameIndex);
  vmaGetHeapBudgets(m_vmaAllocator, m_heapBudgets.data());
}

}  // namespace avk::vk