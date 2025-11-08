#include "render/vk/swapchain-vk.h"

// os specific

// standard
#include <algorithm>
#include <vector>

using namespace avk;
using namespace avk::vk;

inline constexpr VkImageUsageFlags s_SwapchainImageUsage =
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

// ---------------------------------------------------------------------------
static VkSurfaceFormatKHR selectSurfaceFormat(VkPhysicalDevice physicalDevice,
                                              VkSurfaceKHR surface) AVK_NO_CFI {
  static VkSurfaceFormatKHR constexpr preferredFormats[] = {
      {VK_FORMAT_R8G8B8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
      {VK_FORMAT_B8G8R8A8_UNORM, VK_COLORSPACE_SRGB_NONLINEAR_KHR}};

  uint32_t formatCount = 0;
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface,
                                                &formatCount, nullptr));
  std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(
      physicalDevice, surface, &formatCount, surfaceFormats.data()));

  uint32_t index = 0;
  for (auto const& formt : preferredFormats) {
    bool const found =
        surfaceFormats.cend() !=
        std::find_if(surfaceFormats.cbegin(), surfaceFormats.cend(),
                     [formt](VkSurfaceFormatKHR const& surfFmr) {
                       return formt.colorSpace == surfFmr.colorSpace &&
                              formt.format == surfFmr.format;
                     });
    if (found) {
      break;
    }
    ++index;
  }

  return surfaceFormats[index >= surfaceFormats.size() ? 0 : index];
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

// ---------------------------------------------------------------------------

namespace avk::vk {

VkResult Swapchain::recreateSwapchain() AVK_NO_CFI {
  // 1. Query surface formats and select one
  [[maybe_unused]] VkSurfaceFormatKHR const surfaceFormat = selectSurfaceFormat(
      m_deps.device->physicalDevice(), m_deps.surface->handle());
  [[maybe_unused]] VkPresentModeKHR const presentMode = selectPresentMode(
      m_deps.device->physicalDevice(), m_deps.surface->handle(), true);
  // 2. Query surface capabiliities
  VkSurfaceCapabilities2KHR surfCaps{};
  VkPhysicalDeviceSurfaceInfo2KHR surfInfo{};
  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilities2KHR(
      m_deps.device->physicalDevice(), &surfInfo, &surfCaps));

  return VK_SUCCESS;
}

Swapchain::Swapchain(Instance* instance, Surface* surface,
                     Device* device) AVK_NO_CFI
    : m_deps({instance, surface, device}) {}

Swapchain::~Swapchain() noexcept AVK_NO_CFI {
  if (!*this) {
    return;
  }
  auto const* vkDevApi = m_deps.device->table();
  vkDevApi->vkDestroySwapchainKHR(m_deps.device->device(), m_swapchain,
                                  nullptr);
}

}  // namespace avk::vk