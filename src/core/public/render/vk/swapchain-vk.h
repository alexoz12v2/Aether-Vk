#pragma once

#include "render/vk/device-vk.h"
#include "render/vk/surface-vk.h"
#include "utils/mixins.h"

namespace avk::vk::utils {}

namespace avk::vk {

class Swapchain : public NonMoveable {
 public:
  Swapchain(Device* device, Surface* surface);
  /// WARN: Externally synchronized
  ~Swapchain() noexcept;

  /// WARN: To be externally synchronized with UI interaction eg desktop window
  /// resizing: recreate swapchain only at the end
  VkResult recreateSwapchain();

  inline VkSwapchainKHR handle() const { return m_swapchain; }
  inline operator bool() const { return m_swapchain != VK_NULL_HANDLE; }

 private:
  // dependencies which must outlive this object
  struct Deps {
    Device* device;  // check swapchain_maintenance and colorspace
    Surface* surface;
  } m_deps;

  VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
  // TODO current present index
};

}  // namespace avk::vk