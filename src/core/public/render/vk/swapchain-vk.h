#pragma once

#include "render/vk/device-vk.h"
#include "render/vk/surface-vk.h"
#include "utils/mixins.h"

// std
#include <vector>

namespace avk::vk::utils {

/// Swapchain/Present Semaphore discard mechanism inspired by
/// blender's GHOST:
/// - when recreating the swapchain
struct FrameDiscard {
  std::vector<VkSwapchainKHR> swapchains;
  std::vector<VkSemaphore> semaphores;

  FrameDiscard() {
    swapchains.reserve(16);
    semaphores.reserve(16);
  }

  // void destroy(Device const* device);
};

// A command pool needs to be created per frame per thread
struct Frame : public NonCopyable {
  Frame() = default;
  // fence signaled when previous use of the frame has finished rendering,
  // meaning we can acquire a new image and semaphores can be reused
  VkFence submissionFence = VK_NULL_HANDLE;

  // semaphore to acquire a image. being signaled when image is ready to be
  // updated
  VkSemaphore acquireSemaphore = VK_NULL_HANDLE;

  // mechanism to handle delayed out of date swapchain destruction
  FrameDiscard discard;

  // to be called once the frame on which there was a resize is acquired again
  // (after destroying present fences if in use)
  // void destroy(Device const* device);
};

struct SwapchainImage {
  VkImage image = VK_NULL_HANDLE;
  // signaled when image is ready to be presented
  VkSemaphore presentSemaphore = VK_NULL_HANDLE;

  // void destroy(Device const* device);
};

}  // namespace avk::vk::utils

namespace avk::vk {

class Swapchain : public NonMoveable {
 public:
  Swapchain(Instance* instance, Surface* surface, Device* device);
  /// WARN: Externally synchronized
  ~Swapchain() noexcept;

  /// WARN: To be externally synchronized with UI interaction eg desktop window
  /// resizing: recreate swapchain only at the end
  VkResult recreateSwapchain();

  inline operator bool() const { return m_swapchain != VK_NULL_HANDLE; }
  inline void unlockResize() {
    m_stillResizing.store(true, std::memory_order_relaxed);
  }
  inline void lockResize() {
    m_stillResizing.store(true, std::memory_order_relaxed);
  }
  inline void shouldResize() {
    m_forceResize.store(true, std::memory_order_relaxed);
  }

  inline VkSwapchainKHR handle() const { return m_swapchain; }
  inline VkSurfaceTransformFlagsKHR transform() const {
    return m_currentTransform;
  }
  inline VkSurfaceFormatKHR surfaceFormat() const { return m_surfaceFormat; }
  inline VkExtent2D extent() const { return m_extent; }
  inline size_t imageCount() const { return m_images.size(); }

 private:
  // dependencies which must outlive this object
  struct Deps {
    Instance* instance;
    Surface* surface;
    Device* device;  // check swapchain_maintenance and colorspace
  } m_deps;

  // handles and objects
  VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
  std::vector<utils::SwapchainImage> m_images;

  // bookkeeping
  std::vector<utils::Frame> m_frames;
  [[maybe_unused]] uint32_t m_frameIndex = -1;
  [[maybe_unused]] uint32_t m_acquiredImageIndex = -1;

  // parameters
  VkSurfaceTransformFlagsKHR m_currentTransform =
      VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  VkSurfaceFormatKHR m_surfaceFormat{};
  VkExtent2D m_extent{};

  // handle control over when we should resize
  std::atomic_bool m_stillResizing = false;
  std::atomic_bool m_forceResize = false;
};

}  // namespace avk::vk