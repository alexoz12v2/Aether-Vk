#include "render/vk/swapchain-vk.h"

#include <vulkan/vulkan_core.h>

// os specific
// standard

using namespace avk;
using namespace avk::vk;

inline constexpr VkImageUsageFlags s_SwapchainImageUsage =
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------

namespace avk::vk {

VkResult Swapchain::recreateSwapchain() AVK_NO_CFI {
  // 1. Query surface formats and select one
  // 2. Query surface capabiliities

  return VK_SUCCESS;
}

Swapchain::Swapchain(Device* device, Surface* surface) AVK_NO_CFI
    : m_deps({device, surface}) {}

Swapchain::~Swapchain() noexcept AVK_NO_CFI {
  if (!*this) {
    return;
  }
  auto const* vkDevApi = m_deps.device->table();
  vkDevApi->vkDestroySwapchainKHR(m_deps.device->device(), m_swapchain,
                                  nullptr);
}

}  // namespace avk::vk