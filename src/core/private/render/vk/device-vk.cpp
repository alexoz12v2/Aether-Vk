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
#include <vector>

using namespace avk;
using namespace avk::vk;

// TODO insert logging macro
// TODO insert presentation queue query
// TODO insert optional support for `VK_EXT_memory_budget` W/ VMA

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

  bool isSoC;
};

// ---------------------------------------------------------------------------

static void fillRequiredExtensions(
    std::vector<char const*>& requiredExtensions) {
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
  requiredExtensions.push_back(VK_EXT_EXTERNAL_MEMORY_METAL);
#endif
  requiredExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  requiredExtensions.push_back(
      VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME);
  requiredExtensions.push_back(VK_EXT_INLINE_UNIFORM_BLOCK_EXTENSION_NAME);
  requiredExtensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
}

static bool anyRequiredFeaturesMissing(VkPhysicalDevice dev) AVK_NO_CFI {
  // Prepare Structs to query Necessary Features
  VkPhysicalDeviceFeatures2KHR features{};
  VkPhysicalDeviceVulkanMemoryModelFeaturesKHR vulkanMemoryModel{};
  VkPhysicalDeviceUniformBufferStandardLayoutFeaturesKHR ubStandardLayout{};
  VkPhysicalDeviceTimelineSemaphoreFeaturesKHR timelineSemaphoreFeat{};
  VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bufferDeviceAddressFeat{};
  VkPhysicalDeviceInlineUniformBlockFeaturesEXT inlineUniformFeat{};

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

  features.pNext = &vulkanMemoryModel;
  vulkanMemoryModel.pNext = &ubStandardLayout;
  ubStandardLayout.pNext = &timelineSemaphoreFeat;
  timelineSemaphoreFeat.pNext = &bufferDeviceAddressFeat;
  bufferDeviceAddressFeat.pNext = &inlineUniformFeat;

  vkGetPhysicalDeviceFeatures2KHR(dev, &features);

  if (!vulkanMemoryModel.vulkanMemoryModel) {
    return false;
  } else if (!ubStandardLayout.uniformBufferStandardLayout) {
    return false;
  } else if (!bufferDeviceAddressFeat.bufferDeviceAddress) {
    return false;
  } else if (!timelineSemaphoreFeat.timelineSemaphore) {
    return false;
  } else if (!inlineUniformFeat.inlineUniformBlock) {
    return false;
  }

  return true;
}

static void setOptionalFeaturesForDevice(
    VkPhysicalDevice dev, OptionalFeatures& outOptFeatures) AVK_NO_CFI {
  VkPhysicalDeviceFeatures2KHR features{};
  VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT swapMain1Feat{};

  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR;
  swapMain1Feat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;

  features.pNext = &swapMain1Feat;

  vkGetPhysicalDeviceFeatures2KHR(dev, &features);

  outOptFeatures.swapchainMaintenance1 = swapMain1Feat.swapchainMaintenance1;
  outOptFeatures.textureCompressionASTC_LDR =
      features.features.textureCompressionASTC_LDR;
  outOptFeatures.textureCompressionBC = features.features.textureCompressionBC;
  outOptFeatures.textureCompressionETC2 =
      features.features.textureCompressionETC2;
}

// TODO later: test on a integrated (laptop) and see whether not using staging
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
    VkPhysicalDevice chosen, OptionalFeatures& outOptFeatures,
    utils::SampledImageCompressedFormats& comprFormats) AVK_NO_CFI {
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
    vkGetPhysicalDeviceFormatProperties2KHR(chosen, proposedLinear,
                                            &linearFormatProperties);
    vkGetPhysicalDeviceFormatProperties2KHR(chosen, proposedsRGB,
                                            &sRGBFormatProperties);
    if ((linearFormatProperties.formatProperties.linearTilingFeatures &
         sampled_BlitSrc_Transfer) &&
        (linearFormatProperties.formatProperties.optimalTilingFeatures &
         sampled_BlitSrc_Transfer)) {
      comprFormats.linear_R = proposedLinear;
      if (proposedLinear != VK_FORMAT_EAC_R11_UNORM_BLOCK) {  // special case
        comprFormats.linear_RGB = proposedLinear;
      } else {
        vkGetPhysicalDeviceFormatProperties2KHR(
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
    Instance* instance, Extensions& outExtensions,
    OptionalFeatures& outOptFeatures,
    utils::SampledImageCompressedFormats& comprFormats) AVK_NO_CFI {
  if (instance != VK_NULL_HANDLE) {
    return VK_NULL_HANDLE;
  }

  uint32_t physicalDeviceCount = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance->handle(), &physicalDeviceCount,
                                      nullptr));
  AVK_EXT_CHECK(physicalDeviceCount);
  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
  VK_CHECK(vkEnumeratePhysicalDevices(instance->handle(), &physicalDeviceCount,
                                      physicalDevices.data()));

  VkPhysicalDevice chosen = VK_NULL_HANDLE;
  uint32_t maxScoreSoFar = 0;

  // prepare list of required extensions. We'll note the ones which we will use,
  // but won't activate explicitly since we are on Vulkan 1.1 and have been
  // promoted
  std::vector<char const*> requiredExtensions;
  requiredExtensions.reserve(64);
  fillRequiredExtensions(requiredExtensions);
  std::vector<VkExtensionProperties> extensionProperties;
  extensionProperties.reserve(256);

  for (VkPhysicalDevice dev : physicalDevices) {
    // check if required extensions are supported
    uint32_t score = 0;
    uint32_t devExtCount = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(dev, nullptr, &devExtCount,
                                                  nullptr));
    extensionProperties.clear();
    extensionProperties.resize(devExtCount);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(dev, nullptr, &devExtCount,
                                                  extensionProperties.data()));
    for (char const* requiredExtName : requiredExtensions) {
      auto const it =
          std::find_if(extensionProperties.cbegin(), extensionProperties.cend(),
                       [requiredExtName](VkExtensionProperties const& prop) {
                         return strncmp(prop.extensionName, requiredExtName,
                                        VK_MAX_EXTENSION_NAME_SIZE) == 0;
                       });
      if (it == extensionProperties.cend()) {
        continue;
      }
    }

    // check if required features are supported
    if (anyRequiredFeaturesMissing(dev)) {
      continue;
    }

    // evaluate device type
    VkPhysicalDeviceProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkGetPhysicalDeviceProperties2(dev, &props);

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
        continue;  // reject CPUs
        break;
      default:
        continue;  // shouldn't reach here
        break;
    }

    // if valid device, compare its score
    if (score > maxScoreSoFar) {
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
  for (char const* extName : requiredExtensions) {
    AVK_EXT_CHECK(outExtensions.enable(extName));
  }
  if (outOptFeatures.swapchainMaintenance1) {
    AVK_EXT_CHECK(
        outExtensions.enable(VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME));
  }

  return chosen;
}

static bool getPresentationSupport(
    [[maybe_unused]] VkPhysicalDevice physicalDevice,
    [[maybe_unused]] uint32_t index,
    [[maybe_unused]] Surface const* surface) AVK_NO_CFI {
#ifdef VK_USE_PLATFORM_WIN32_KHR
  return vkGetPhysicalDeviceWin32PresentationSupportKHR(physicalDevice, index);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
  return vkGetPhysicalDeviceWaylandPresentationSupportKHR(physicalDevice, index,
                                                          surface->display());
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
    VkPhysicalDevice physicalDevice, Surface const* surface,
    utils::QueueFamilyMap& outQueueFamilies) AVK_NO_CFI {
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  queueCreateInfos.reserve(4);
  queueCreateInfos.push_back({});
  VkDeviceQueueCreateInfo& createInfo = queueCreateInfos.back();
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  createInfo.queueCount = 1;

  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties2KHR(physicalDevice, &count, nullptr);
  std::vector<VkQueueFamilyProperties2> queueProperties{count};
  for (VkQueueFamilyProperties2& prop : queueProperties) {
    prop.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2_KHR;
  }
  vkGetPhysicalDeviceQueueFamilyProperties2KHR(physicalDevice, &count,
                                               queueProperties.data());

  uint32_t familyIndex = 0;
  for (VkQueueFamilyProperties2 const& props : queueProperties) {
    // TODO add per WSI type surface support check
    uint32_t const index = familyIndex++;
    bool const supportsPresent =
        getPresentationSupport(physicalDevice, index, surface);
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

static VkDevice createDevice(VkPhysicalDevice physicalDevice,
                             Extensions const& extensions,
                             OptionalFeatures const& optFeatures,
                             Surface const* surface,
                             utils::QueueFamilyMap& queueFamilies) AVK_NO_CFI {
  // enable all the necessary features
  VkPhysicalDeviceFeatures2KHR features{};
  VkPhysicalDeviceVulkanMemoryModelFeaturesKHR vulkanMemoryModel{};
  VkPhysicalDeviceUniformBufferStandardLayoutFeaturesKHR ubStandardLayout{};
  VkPhysicalDeviceTimelineSemaphoreFeaturesKHR timelineSemaphoreFeat{};
  VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bufferDeviceAddressFeat{};
  VkPhysicalDeviceInlineUniformBlockFeaturesEXT inlineUniformFeat{};
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
  swapchainMaintenance1Feat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;

  features.pNext = &vulkanMemoryModel;
  vulkanMemoryModel.pNext = &ubStandardLayout;
  ubStandardLayout.pNext = &timelineSemaphoreFeat;
  timelineSemaphoreFeat.pNext = &bufferDeviceAddressFeat;
  bufferDeviceAddressFeat.pNext = &inlineUniformFeat;
  if (optFeatures.swapchainMaintenance1) {
    inlineUniformFeat.pNext = &swapchainMaintenance1Feat;
  }

  // WARNING: Keep in sync with functions
  //   `anyRequiredFeaturesMissing` and `setOptionalFeaturesForDevice`
  vulkanMemoryModel.vulkanMemoryModel = VK_TRUE;
  ubStandardLayout.uniformBufferStandardLayout = VK_TRUE;
  bufferDeviceAddressFeat.bufferDeviceAddress = VK_TRUE;
  timelineSemaphoreFeat.timelineSemaphore = VK_TRUE;
  inlineUniformFeat.inlineUniformBlock = VK_TRUE;
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
      newDeviceQueuesCreateInfos(physicalDevice, surface, queueFamilies);
  float const highestPriority = 1.0f;
  for (VkDeviceQueueCreateInfo& queueCreateInfo : queueCreateInfos) {
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &highestPriority;
  }

  createInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());
  createInfo.pQueueCreateInfos = queueCreateInfos.data();

  VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device));
  return device;
}

static VmaAllocator newVmaAllocator(
    VkInstance instance, uint32_t vulkanApiVersion,
    VkPhysicalDevice physicalDevice, VkDevice device,
    VmaVulkanFunctions* vmaVulkanFunctions) AVK_NO_CFI {
  assert(vulkanApiVersion >= VK_API_VERSION_1_1);
  VmaAllocatorCreateInfo allocatorCreateInfo{};
  // `VK_KHR_dedicated_allocation` promoted from 1.1
  allocatorCreateInfo.flags |=
      VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
  // `VK_KHR_buffer_device_address` in use
  allocatorCreateInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  // `VK_KHR_bind_memory2` promoted from 1.1
  allocatorCreateInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
  // `VK_KHR_EXTERNAL_MEMORY_WIN32` is the only exporting extension supported by
  // VMA, so, for consistency, I'd rather handle it myself
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

Device::Device(Instance* instance, Surface* surface) AVK_NO_CFI
    : m_deps({instance, surface}) {
  assert(instance);
  Extensions extensions;
  OptionalFeatures optFeatures{};

  // 1. Select physical device which supports baseline and bookkeeping of opt
  m_physicalDevice = choosePhysicalDevice(m_deps.instance, extensions,
                                          optFeatures, m_comprFormats);
  m_isSoC = optFeatures.isSoC;
  m_swapchainMaintenance1 = optFeatures.swapchainMaintenance1;

  // 2. Device creation, extract graphics/compute/transfer/present queue, load
  // table
  m_device = createDevice(m_physicalDevice, extensions, optFeatures,
                          m_deps.surface, m_queueFamilies);
  VkDeviceQueueInfo2 queueInfo{};
  queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2;
  queueInfo.queueFamilyIndex = m_queueFamilies.universalGraphics;
  queueInfo.queueIndex = 0;
  vkGetDeviceQueue2(m_device, &queueInfo, &m_queue);
  m_table = std::make_unique<VolkDeviceTable>();
  AVK_EXT_CHECK(m_table);
  volkLoadDeviceTable(m_table.get(), m_device);

  // 3. create allocator and pool
  m_vmaVulkanFunctions = std::make_unique<VmaVulkanFunctions>();
  m_vmaAllocator =
      newVmaAllocator(instance->handle(), instance->vulkanApiVersion(),
                      m_physicalDevice, m_device, m_vmaVulkanFunctions.get());
}

Device::~Device() noexcept AVK_NO_CFI {
  if (!*this) {
    return;
  }

  m_table->vkDeviceWaitIdle(m_device);
  vmaDestroyAllocator(m_vmaAllocator);
  m_table->vkDestroyDevice(m_device, nullptr);
}

}  // namespace avk::vk