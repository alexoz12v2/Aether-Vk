#pragma once

#include "app/avk-application.h"

// os specific
#include <Windef.h>

#include "avk-win32-window.h"

namespace avk {

// TODO check max push constant size is 256
// TODO refactor
struct Camera {
  glm::mat4 view;
  glm::mat4 proj;
};

class WindowsApplication : public ApplicationBase {
 public:
  WindowsApplication() = default;
  ~WindowsApplication() noexcept override;

  WindowPayload WindowPayload;
  HWND PrimaryWindow = nullptr;
  HANDLE RenderThread = INVALID_HANDLE_VALUE;

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

 private:
  void createConstantVulkanResources();
  void destroyConstantVulkanResources();
  void cleanupVulkanResources();
  void createVulkanResources();

  // TODO refactor
  vk::GraphicsInfo m_graphicsInfo;
  VkPipeline m_graphicsPipeline = VK_NULL_HANDLE;

  VkImageView m_depthView = VK_NULL_HANDLE;

  std::vector<VkFramebuffer> m_framebuffers;
  std::vector<uint64_t> m_commandBufferIds;

  // stuff to refactor
  Camera m_camera;
  std::vector<Camera> m_pushCameras;  // STABLE, FIF
  VkDescriptorSet m_cubeDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorUpdateTemplateKHR m_descriptorUpdateTemplate = VK_NULL_HANDLE;
};

}  // namespace avk
