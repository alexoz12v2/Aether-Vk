#pragma once

#include "render/vk/device-vk.h"
#include "render/vk/surface-vk.h"
#include "utils/mixins.h"

// std
#include <vector>

// libraries
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

namespace avk::vk::utils {

struct SwapchainData {
  /// swapchain image handle
  VkImage image;
  /// swapchain image view handle
  VkImageView imageView;
  /// to signal when we submit
  VkFence submissionFence;
  /// semaphore to wait for when submitting
  VkSemaphore acquireSemaphore;
  /// semaphore to signal when submitting, wait when presenting
  VkSemaphore presentSemaphore;
};

struct SwapchainImage;

/// Swapchain/Present Semaphore discard mechanism inspired by
/// blender's GHOST:
/// - when recreating the swapchain
struct FrameDiscard {
  std::vector<VkSwapchainKHR> swapchains;
  std::vector<VkSemaphore> semaphores;
  std::vector<VkImageView> imageViews;

  FrameDiscard() {
    swapchains.reserve(16);
    semaphores.reserve(16);
    imageViews.reserve(16);
  }

  void discardSwapchainImages(std::vector<utils::SwapchainImage>& images);

  void destroy(Device const* device);
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
  void destroy(Device const* device);
};

struct SwapchainImage {
  VkImage image = VK_NULL_HANDLE;
  VkImageView imageView = VK_NULL_HANDLE;
  // signaled when image is ready to be presented
  VkSemaphore presentSemaphore = VK_NULL_HANDLE;
};

struct SurfacePreRotation {
  /// Rotation to apply to camera to account for screen orientation
  /// \example:
  ///   glm::mat4 view = glm::mat4_cast(preRot.cameraRotation) * baseView;
  /// \note see if it is correct or if I should take complex conjugate
  glm::quat cameraRotation;
  /// Projection matrix adjustment to account for mirroring
  /// \example:
  ///    glm::mat4 proj = preRot.projectionAdjust * baseProjection;
  glm::mat4 projectionAdjust;
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
  /// NOTE: After this, you should check the transform of the swapchain
  /// if a 90 or 270 rotation occurred (mirrored or not), camera should be
  /// rotated
  void recreateSwapchain();

  /// Force swapchain discard (response to a surface lost)
  /// This is externally synchronized with respect to the other functions
  /// such as `recreateSwapchain` and `acquireNextImage`
  void forceDiscardToCurrentFrame();

  /// vkAcquireNextImageKHR. If a non null fence is given, timeout is zero
  /// and you should wait for this fence before submitting the renderPass
  /// and/or presenting the image. Otherwise, it waits synchronously
  /// for a swapchain image to be available
  VkResult acquireNextImage(VkFence acquireFence = VK_NULL_HANDLE);

  inline operator bool() const { return m_swapchain != VK_NULL_HANDLE; }
  // TODO move to application and rendering coordinator
  inline void unlockResize() {
    m_stillResizing.store(false, std::memory_order_release);
  }
  inline void lockResize() {
    m_stillResizing.store(true, std::memory_order_release);
  }

  inline VkSwapchainKHR handle() const { return m_swapchain; }
  inline VkSurfaceTransformFlagsKHR transform() const {
    return m_currentTransform;
  }
  inline VkSurfaceFormatKHR surfaceFormat() const { return m_surfaceFormat; }
  inline VkExtent2D extent() const { return m_extent; }
  inline size_t imageCount() const { return m_images.size(); }
  inline size_t frameCount() const { return m_frames.size(); }

  static uint32_t constexpr InvalidIndex = -1;
  /// Get the index of the last successfully acquired image
  /// Note: May be the index from a decommissioned swapchain if you didn't call
  /// acquireNextImage after a recreation
  /// Note: If you don't wait for fence after acquisition, this is wrong
  inline uint32_t imageIndex() const { return m_imageIndex; }
  inline uint32_t frameIndex() const { return m_frameIndex; }
  /// While Image Index is updated by vkAcquireNextImageKHR, frame index is
  /// tracked by us, and should be called at the *End* of the frame
  void signalNextFrame();
  /// To be called after waiting for image acquisition
  utils::SwapchainData swapchainData() const;

  utils::SurfacePreRotation preRotation() const;

  inline VkImage imageAt(size_t index) const {
    return index < m_images.size() ? m_images[index].image : VK_NULL_HANDLE;
  }
  inline VkImageView imageViewAt(size_t index) const {
    return index < m_images.size() ? m_images[index].imageView : VK_NULL_HANDLE;
  }

 private:
  // dependencies which must outlive this object
  struct Deps {
    Instance* instance;
    Surface* surface;
    Device* device;
  } m_deps;

  // handles and objects
  VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
  std::vector<utils::SwapchainImage> m_images;
  uint32_t m_imageIndex = 0;

  // bookkeeping
  std::vector<utils::Frame> m_frames;
  uint32_t m_frameIndex = 0;

  // parameters
  VkSurfaceTransformFlagsKHR m_currentTransform =
      VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  VkSurfaceFormatKHR m_surfaceFormat{};
  VkExtent2D m_extent{};

  // handle control over when we should resize
  std::atomic_bool m_stillResizing = false;

  // temp storage for images
  std::vector<VkImage> m_tmpImages;
};

}  // namespace avk::vk