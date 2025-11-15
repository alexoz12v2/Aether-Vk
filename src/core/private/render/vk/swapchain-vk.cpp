#include "render/vk/swapchain-vk.h"

// our stuff
#include "utils/integer.h"

// os specific
#ifdef AVK_OS_WINDOWS
#  include <Windows.h>
#else
#  include <sched.h>
#endif

// standard
#include <algorithm>
#include <vector>

// library
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

using namespace avk;
using namespace avk::vk;

inline constexpr VkImageUsageFlags s_SwapchainImageUsage =
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

// ---------------------------------------------------------------------------
static VkSurfaceFormatKHR selectSurfaceFormat(VkPhysicalDevice physicalDevice,
                                              VkSurfaceKHR surface) AVK_NO_CFI {
  // TODO Maybe: If necessary to support HDR, you need to query for colorspace
  // too
  static VkFormat constexpr preferredFormats[] = {
      VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
      VK_FORMAT_A8B8G8R8_UNORM_PACK32};

  uint32_t formatCount = 0;
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface,
                                                &formatCount, nullptr));
  AVK_EXT_CHECK(formatCount);
  std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(
      physicalDevice, surface, &formatCount, surfaceFormats.data()));

  uint32_t index = 0;
  for (auto const &formt : preferredFormats) {
    bool const found =
        surfaceFormats.cend() !=
            std::find_if(surfaceFormats.cbegin(), surfaceFormats.cend(),
                         [formt](VkSurfaceFormatKHR const &surfFmr) {
                           return formt == surfFmr.format;
                         });
    if (found) {
      break;
    }
    ++index;
  }
  if (index >= surfaceFormats.size()) {
    index = 0;
    if (surfaceFormats[0].format == VK_FORMAT_UNDEFINED) {
      surfaceFormats[0].format = VK_FORMAT_B8G8R8A8_UNORM;
    }
  }
  return surfaceFormats[index];
}

static VkPresentModeKHR selectPresentMode(VkPhysicalDevice physicalDevice,
                                          VkSurfaceKHR surface, bool vsync) {
  uint32_t presentModeCount = 0;
  VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(
      physicalDevice, surface, &presentModeCount, nullptr));
  std::vector<VkPresentModeKHR> presentModes{presentModeCount};
  VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(
      physicalDevice, surface, &presentModeCount, presentModes.data()));
  VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;  // supported by spec
  if (vsync) {
    // search for MAILBOX
    if (std::any_of(presentModes.cbegin(), presentModes.cend(),
                    [](VkPresentModeKHR pMode) {
                      return pMode == VK_PRESENT_MODE_MAILBOX_KHR;
                    })) {
      presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    }
  } else {
    // search for IMMEDIATE
    if (std::any_of(presentModes.cbegin(), presentModes.cend(),
                    [](VkPresentModeKHR pMode) {
                      return pMode == VK_PRESENT_MODE_IMMEDIATE_KHR;
                    })) {
      presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
  }

  return presentMode;
}

static VkExtent2D extentVkClamp(VkExtent2D extentCurr, VkExtent2D extentMin,
                                VkExtent2D extentMax) {
  VkExtent2D result{};
  result.width = extentCurr.width > extentMax.width
                 ? extentMax.width
                 : (extentCurr.width < extentMin.width ? extentMin.width
                                                       : extentCurr.width);
  result.height =
      extentCurr.height > extentMax.height
      ? extentMax.height
      : (extentCurr.height < extentMin.height ? extentMin.height
                                              : extentCurr.height);
  return result;
}

static inline bool extentVkSpecialValue(VkExtent2D extent) {
  static uint32_t constexpr Special = 0xFFFFFFFFU;
  return extent.width == Special && extent.height == Special;
}

// ---------------------------------------------------------------------------

namespace avk::vk::utils {

void FrameDiscard::destroy(const Device *device,
                           VkInstance instance) AVK_NO_CFI {
  auto const *const vkDevApi = device->table();
  VkDevice const dev = device->device();
  while (!swapchains.empty()) {
    VkSwapchainKHR handle = swapchains.back();
    swapchains.pop_back();
    vkDevApi->vkDestroySwapchainKHR(dev, handle, nullptr);
  }
  while (!surfaces.empty()) {
    VkSurfaceKHR surface = surfaces.back();
    surfaces.pop_back();
    vkDestroySurfaceKHR(instance, surface, nullptr);
  }
  while (!semaphores.empty()) {
    VkSemaphore handle = semaphores.back();
    semaphores.pop_back();
    vkDevApi->vkDestroySemaphore(dev, handle, nullptr);
  }
  while (!imageViews.empty()) {
    VkImageView handle = imageViews.back();
    imageViews.pop_back();
    vkDevApi->vkDestroyImageView(dev, handle, nullptr);
  }
}

void FrameDiscard::discardSwapchainImages(
    std::vector<utils::SwapchainImage> &images) {
  for (utils::SwapchainImage &swapchainImage : images) {
    assert(swapchainImage.presentSemaphore && swapchainImage.imageView);
    semaphores.push_back(swapchainImage.presentSemaphore);
    imageViews.push_back(swapchainImage.imageView);
    swapchainImage.presentSemaphore = VK_NULL_HANDLE;
  }
}

void Frame::destroy(const Device *device, VkInstance instance) AVK_NO_CFI {
  auto const *vkDevApi = device->table();
  VkDevice const dev = device->device();
  if (submissionFence != VK_NULL_HANDLE) {
    vkDevApi->vkDestroyFence(dev, submissionFence, nullptr);
    submissionFence = VK_NULL_HANDLE;
  }
  if (acquireSemaphore != VK_NULL_HANDLE) {
    vkDevApi->vkDestroySemaphore(dev, acquireSemaphore, nullptr);
    acquireSemaphore = VK_NULL_HANDLE;
  }
  discard.destroy(device, instance);
}

}  // namespace avk::vk::utils

namespace avk::vk {

void Swapchain::recreateSwapchain() AVK_NO_CFI {
#define PREFIX "[Swapchain::recreateSwapchain] "
  auto const *const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();
  uint32_t const currentFrame = frameIndex();

  // TODO: These values never change for a given surface, we may cache them
  // 1. Query surface formats and select one
  VkSurfaceFormatKHR const surfaceFormat = selectSurfaceFormat(
      m_deps.device->physicalDevice(), m_deps.surface->handle());
  VkPresentModeKHR const presentMode = selectPresentMode(
      m_deps.device->physicalDevice(), m_deps.surface->handle(), true);
  LOGI << PREFIX "Surface Format and Present Mode Selected" << std::endl;
  // 2. Query surface capabiliities
  // TODO Windows: Insert VkSurfaceCapabilitiesFullScreenExclusiveEXT
  // Note: Removed scaling capabilities because we won't be creating a swapchain
  // with VkSwapchainPresentScalingCreateInfoKHR, as it may strain the Android
  // device's DPU
  VkSurfaceCapabilities2KHR surfCaps{};
  surfCaps.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR;

  // TODO Windows: Insert VkSurfaceFullScreenExclusiveWin32InfoEXT
  VkSurfacePresentModeEXT presentModeExt{};
  presentModeExt.sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_EXT;
  presentModeExt.presentMode = presentMode;
  VkPhysicalDeviceSurfaceInfo2KHR surfInfo{};
  surfInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR;
  surfInfo.surface = m_deps.surface->handle();
  surfInfo.pNext = &presentModeExt;

  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilities2KHR(
      m_deps.device->physicalDevice(), &surfInfo, &surfCaps));
  // get the `preTransform` (saved after successful creation)
  LOGI << PREFIX "Queried Surface capabilities" << std::endl;
  VkSurfaceTransformFlagBitsKHR const transform =
      surfCaps.surfaceCapabilities.currentTransform;
  // get appropriate extent
  VkExtent2D extent{};
  if (extentVkSpecialValue(surfCaps.surfaceCapabilities.currentExtent)) {
    extent = m_deps.surface->internalExtent();
  } else {
    extent = surfCaps.surfaceCapabilities.currentExtent;
  }
  extent = extentVkClamp(extent, surfCaps.surfaceCapabilities.minImageExtent,
                         surfCaps.surfaceCapabilities.maxImageExtent);
  // extent: Manage rotation to perform Pre-Rotation
  switch (transform) {
    case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR:
    case VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR:
    case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR:
    case VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR:
      std::swap(extent.width,
                extent.height);
      break;
    default:break;
  }

  // get image count (3 -> vsync, 2 -> no vsync)
  uint32_t imageCount = presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR ? 2 : 3;
  LOGI << PREFIX "Min Image Count: "
       << surfCaps.surfaceCapabilities.minImageCount << std::endl;
  if (imageCount <
      surfCaps.surfaceCapabilities.minImageCount)  // min >= 1 by spec
    imageCount = surfCaps.surfaceCapabilities.minImageCount;
  else if (uint32_t max = surfCaps.surfaceCapabilities.maxImageCount;
      max && imageCount > max)  // max = 0 if no limit by spec
    imageCount = surfCaps.surfaceCapabilities.maxImageCount;

  LOGI << PREFIX "Computed Extent: " << extent.width << 'x' << extent.height
       << ", transform, imageCount: " << imageCount << std::endl;

  // 3. Mark for discard the current swapchain and present semaphores. On the
  // next round in which the current frame is examined, it is safe to destroy
  // them
  // TODO: recycle semaphores if present fences?
  utils::FrameDiscard *frameDiscard = nullptr;
  if (currentFrame <= m_frames.size() && m_images.size() > 0) {
    frameDiscard = &m_frames[currentFrame].discard;
    // sweep semaphores (swapchain done after creation)
    frameDiscard->discardSwapchainImages(m_images);
  }

  // 4. Create Info
  // is it necessary to handle scaling or support multiple present modes?
  // Vulkan-Samples doesn't do that
  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = m_deps.surface->handle();
  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = surfaceFormat.format;
  createInfo.imageArrayLayers = 1;
  createInfo.imageColorSpace = surfaceFormat.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  // check image flags (color attachment is mandatory by standard)
  AVK_EXT_CHECK(
      surfCaps.surfaceCapabilities.supportedUsageFlags &
          (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
  createInfo.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
#ifdef AVK_WINDOW_TRANSPARENCY
  // check for either pre multiplied or post multiplied and inherit
  VkCompositeAlphaFlagBitsKHR compositeAlpha =
      VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;  // OS handling + ...
  AVK_EXT_CHECK(surfCaps.surfaceCapabilities.supportedCompositeAlpha &
                VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR);
  if (surfCaps.surfaceCapabilities.supportedCompositeAlpha &
      VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) {
    compositeAlpha = static_cast<VkCompositeAlphaFlagBitsKHR>(
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR |
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR);
  } else if (surfCaps.surfaceCapabilities.supportedCompositeAlpha &
             VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) {
    compositeAlpha = static_cast<VkCompositeAlphaFlagBitsKHR>(
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR |
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR);
  }
  AVK_EXT_CHECK(compositeAlpha != VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR);
  createInfo.compositeAlpha = compositeAlpha;
#else
  if (surfCaps.surfaceCapabilities.supportedCompositeAlpha &
      VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) {
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  } else if (surfCaps.surfaceCapabilities.supportedCompositeAlpha &
      VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) {
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
  } else {
    showErrorScreenAndExit("No Alpha Composition mode supported");
  }
#endif
  // some pixels from the swapchain may be written over by OS specific
  // more generally, docs suggest setting to true unless you read back
  // from image
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = m_swapchain;
  createInfo.preTransform = transform;
  createInfo.presentMode = presentMode;

  // 5. Create and formally decommission. Note: Presentation of acquired images
  // for a decommissioned swapchain go through, but fail, so make sure to NOT
  // acquire images from decommissioned swapchain
  // - before calling recreate, wait on the two atomic flags
  LOGI << PREFIX "About to create Swapchain Before atomic polling" << std::endl;
  // Busy wait until it's ok to recreate after resize
  while (true) {
    bool const val = m_stillResizing.load(std::memory_order_relaxed);
    if (val) {
#ifdef AVK_OS_WINDOWS
      YieldProcessor();
#else
      sched_yield();
#endif
    } else if (!m_stillResizing.load(std::memory_order_acquire)) {
      break;
    }
  }
  LOGI << PREFIX "About to create Swapchain After atomic polling" << std::endl;

  VkSwapchainKHR swapchain = VK_NULL_HANDLE;

  VkResult res =
      vkDevApi->vkCreateSwapchainKHR(dev, &createInfo, nullptr, &swapchain);
  VK_CHECK(res);
  LOGI << PREFIX "Just Created Pool" << std::endl;

  // 6. bookkeeping and cleanup
  m_surfaceFormat = surfaceFormat;
  m_extent = extent;
  m_currentTransform = transform;
  std::swap(swapchain, m_swapchain);
  LOGI << PREFIX "Bookkeeping variables done" << std::endl;

  // 7. Resources Creation: Frame synchronization primitives, image views
  uint32_t swapchainImageCount = 0;
  VK_CHECK(vkDevApi->vkGetSwapchainImagesKHR(dev, m_swapchain,
                                             &swapchainImageCount, nullptr));
  LOGI << PREFIX "Acquired " << swapchainImageCount << " Count of images"
       << std::endl;
  m_images.resize(swapchainImageCount);
  m_tmpImages.resize(swapchainImageCount);
  VK_CHECK(vkDevApi->vkGetSwapchainImagesKHR(
      dev, m_swapchain, &swapchainImageCount, m_tmpImages.data()));
  LOGI << PREFIX "Acquired " << swapchainImageCount << " Swapchain Images"
       << std::endl;

  VkSemaphoreCreateInfo semCreateInfo{};
  semCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkImageViewCreateInfo imgViewCreateInfo{};
  imgViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  imgViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  imgViewCreateInfo.format = surfaceFormat.format;
  imgViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imgViewCreateInfo.subresourceRange.baseArrayLayer = 0;
  imgViewCreateInfo.subresourceRange.baseMipLevel = 0;
  imgViewCreateInfo.subresourceRange.layerCount = 1;
  imgViewCreateInfo.subresourceRange.levelCount = 1;

  for (uint32_t imageIndex = 0; imageIndex < swapchainImageCount;
       ++imageIndex) {
    m_images[imageIndex].image = m_tmpImages[imageIndex];
    // semaphore from decommissioned swapchain should have been discarded
    // you may not recycle them safely without waiting for a present fence,
    // which requires VK_EXT_swapchain_maintenance1, which isn't supported on
    // all Android Devices hence: Recreate semaphores
    AVK_EXT_CHECK(m_images[imageIndex].presentSemaphore == VK_NULL_HANDLE);
    VK_CHECK(vkDevApi->vkCreateSemaphore(
        dev, &semCreateInfo, nullptr, &m_images[imageIndex].presentSemaphore));
    // Since imageUsage contains COLOR_ATTACHMENT_BIT, we can create image
    // views associated to the swapchain image
    imgViewCreateInfo.image = m_images[imageIndex].image;
    VK_CHECK(vkDevApi->vkCreateImageView(dev, &imgViewCreateInfo, nullptr,
                                         &m_images[imageIndex].imageView));
  }
  m_tmpImages.clear();
  LOGI << PREFIX
          "Created Image Views and Present Semaphores for swapchain images"
       << std::endl;

  // discard all frame acquire semaphores and swapchain itself
  for (utils::Frame &frame : m_frames) {
    if (frame.acquireSemaphore != VK_NULL_HANDLE && frameDiscard) {
      frameDiscard->semaphores.push_back(frame.acquireSemaphore);
      frame.acquireSemaphore = VK_NULL_HANDLE;
    }
  }
  // discard decommissioned swapchain
  if (swapchain != VK_NULL_HANDLE && frameDiscard) {
    frameDiscard->swapchains.push_back(swapchain);
  }
  // don't delete current frames in flight
  if (swapchainImageCount > m_frames.size()) {
    m_frames.resize(swapchainImageCount);
  }
  // refresh frame data
  VkFenceCreateInfo fenceCreateInfo{};
  fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (utils::Frame &frame : m_frames) {
    // semaphores for old frames have been discarded, recreate them (+ new ones)
    if (frame.acquireSemaphore == VK_NULL_HANDLE) {
      VK_CHECK(vkDevApi->vkCreateSemaphore(dev, &semCreateInfo, nullptr,
                                           &frame.acquireSemaphore));
    }
    // fences for old frames are still here, and it's safe to reuse them
    if (frame.submissionFence == VK_NULL_HANDLE) {
      VK_CHECK(vkDevApi->vkCreateFence(dev, &fenceCreateInfo, nullptr,
                                       &frame.submissionFence));
    }
  }
  LOGI << PREFIX
          "Created submission Fence and acquire semaphore for each frame in flight "
          "("
       << m_frames.size() << ')' << std::endl;
#undef PREFIX
};

Swapchain::Swapchain(Instance *instance, Surface *surface,
                     Device *device) AVK_NO_CFI
    : m_deps({instance, surface, device}) {
  recreateSwapchain();
}

Swapchain::~Swapchain() noexcept AVK_NO_CFI {
  if (!*this) {
    return;
  }
  auto const *vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();
  // 1. Destroy Present Semaphores (assumes no images are acquired/presenting)
  m_tmpImages.clear();
  for (utils::SwapchainImage &image : m_images) {
    // WARNING: Supposes that presentation has finished (wait idle or present
    // fence in place)
    if (image.presentSemaphore != VK_NULL_HANDLE) {
      vkDevApi->vkDestroySemaphore(dev, image.presentSemaphore, nullptr);
    }
    if (image.imageView != VK_NULL_HANDLE) {
      vkDevApi->vkDestroyImageView(dev, image.imageView, nullptr);
    }
  }
  m_images.clear();
  // 2. Destroy current swapchain
  vkDevApi->vkDestroySwapchainKHR(dev, m_swapchain, nullptr);
  // 3. Destroy Frames (synchronization primitives and decommissioned handles)
  for (utils::Frame &frame : m_frames) {
    frame.destroy(m_deps.device, m_deps.instance->handle());
  }
  m_frames.clear();
}

void Swapchain::signalNextFrame() {
  // tick next image/frame indices
  m_frameIndex = (m_frameIndex + 1) % m_frames.size();
}

VkResult Swapchain::acquireNextImage(VkFence acquireFence) AVK_NO_CFI {
  auto const *const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();
  uint32_t const frameIndex = m_frameIndex;
  uint64_t const timeout = acquireFence != VK_NULL_HANDLE ? 0 : UINT64_MAX;

  // wait for previous submission: When using Multiple Frames in Flight
  // we need to make sure to not break frame-dependant data
  VkResult res = VK_SUCCESS;
  if (m_frames[frameIndex].submissionFence != VK_NULL_HANDLE) {
    do {
      res = vkDevApi->vkWaitForFences(
          dev, 1, &m_frames[frameIndex].submissionFence, VK_TRUE, UINT64_MAX);
      // timeout is actually a success code (>0), hence check won't catch that
      VK_CHECK(res);
    } while (res == VK_TIMEOUT);
    VK_CHECK(vkDevApi->vkResetFences(
        dev, 1, &m_frames[frameIndex].submissionFence));
  }

  // TODO: HDR Support: check if display changed its color space.
  // recreate the swapchain if it did

  res = vkDevApi->vkAcquireNextImageKHR(dev, m_swapchain, timeout,
                                        m_frames[frameIndex].acquireSemaphore,
                                        acquireFence, &m_imageIndex);
  if (res < 0 && res != VK_ERROR_OUT_OF_DATE_KHR) {
    VK_CHECK(res);
  }

  // VK_SUCCESS, VK_NOT_READY, VK_SUBOPTIMAL_KHR, VK_TIMEOUT,
  // VK_ERROR_OUT_OF_DATE_KHR
  return res;
}

utils::SwapchainData Swapchain::swapchainData() const {
  utils::SwapchainData data{};
  data.acquireSemaphore = m_frames[frameIndex()].acquireSemaphore;
  data.presentSemaphore = m_images[m_imageIndex].presentSemaphore;
  data.submissionFence = m_frames[frameIndex()].submissionFence;
  data.image = m_images[m_imageIndex].image;
  data.imageView = m_images[m_imageIndex].imageView;
  return data;
}

utils::SurfacePreRotation Swapchain::preRotation() const {
  utils::SurfacePreRotation result{};

  switch (m_currentTransform) {
    case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR:
    case VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR:
      result.preRotate = glm::rotate(glm::mat4(1.f),
                                     glm::radians(90.f),
                                     glm::vec3(0, 0, 1));
      break;

    case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR:
    case VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR:
      result.preRotate = glm::rotate(glm::mat4(1.f),
                                     glm::radians(180.f),
                                     glm::vec3(0, 0, 1));
      break;

    case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR:
    case VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR:
      result.preRotate = glm::rotate(glm::mat4(1.f),
                                     glm::radians(270.f),
                                     glm::vec3(0, 0, 1));
      break;
    case VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR:
    default:result.preRotate = glm::mat4(1.f);
      break;

  }

  return result;
}

void Swapchain::forceDiscardToCurrentFrame(VkSurfaceKHR lostSurface) {
  if (m_frameIndex < m_frames.size()) {
    if (!m_images.empty()) {
      m_frames[m_frameIndex].discard.discardSwapchainImages(m_images);
      m_images.clear();
    }
    if (m_swapchain) {
      m_frames[m_frameIndex].discard.swapchains.push_back(m_swapchain);
      m_swapchain = VK_NULL_HANDLE;
      if (lostSurface) {
        m_frames[m_frameIndex].discard.surfaces.push_back(lostSurface);
      }
    }
  };
}

}  // namespace avk::vk