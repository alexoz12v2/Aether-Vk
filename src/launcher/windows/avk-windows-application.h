#pragma once

// AVK Core
#include "app/avk-application.h"
#include "render/experimental/avk-staging-transient-manager.h"
#include "render/testing/avk-primitives.h"

// library
#include <shared_mutex>

// os specific
#include <Windef.h>

// from this module
#include "avk-win32-window.h"

namespace avk {

class WindowsApplication : public ApplicationBase {
 public:
  WindowsApplication() = default;
  ~WindowsApplication() noexcept override;

  WindowPayload WindowPayload;
  HWND PrimaryWindow = nullptr;
  HANDLE RenderThread = INVALID_HANDLE_VALUE;
  HANDLE UpdateThread = INVALID_HANDLE_VALUE;

 protected:
  void doOnSaveState() override;
  void doOnRestoreState() override;
  void RTdoOnDeviceRegained() override;
  void RTdoOnWindowInit() override;
  VkResult RTdoOnRender(const vk::utils::SwapchainData& swapchainData) override;
  void RTdoOnResize() override;

  vk::SurfaceSpec doSurfaceSpec() override;
  [[noreturn]] void RTdoOnDeviceLost() override;
  [[noreturn]] void RTdoOnSurfaceLost() override;
  void RTdoEarlySurfaceRegained() override;
  void RTdoLateSurfaceRegained() override;

  void UTdoOnFixedUpdate() override;
  void UTdoOnUpdate() override;
  void UTdoOnInit() override;

 private:
  void createConstantVulkanResources();
  void destroyConstantVulkanResources();
  void cleanupVulkanResources();
  void createVulkanResources();

 private:
  // TODO refactor
  vk::GraphicsInfo m_graphicsInfo;
  VkPipeline m_graphicsPipeline = VK_NULL_HANDLE;

  VkImageView m_depthView = VK_NULL_HANDLE;

  std::vector<VkFramebuffer> m_framebuffers;
  std::vector<uint64_t> m_commandBufferIds;

  // TODO move to base if works well
  experimental::StagingTransientManager m_staging;

  // stuff to refactor
  std::shared_mutex m_swapState;
  float m_angle = 0.f;
  Camera m_RTcamera{};
  glm::mat4 m_UTcamera{};  // Update thread only has view. proj owned by render
  std::vector<Camera> m_pushCameras;  // STABLE, FIF
  VkDescriptorSet m_cubeDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorUpdateTemplateKHR m_descriptorUpdateTemplate = VK_NULL_HANDLE;
};

}  // namespace avk
