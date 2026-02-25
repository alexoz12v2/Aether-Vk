#pragma once

#include "app/avk-application.h"
#include "render/experimental/avk-ktx2-textures.h"
#include "render/experimental/avk-staging-transient-manager.h"
#include "render/testing/avk-primitives.h"

namespace avk {

class MacosApplication : public ApplicationBase {
 public:
  MacosApplication();
  ~MacosApplication() noexcept override;

  pthread_t RenderThread = 0;
  pthread_t UpdateThread = 0;
  CAMetalLayer* MetalLayer = 0;

  void signalDisplayLinkReady();

 protected:
  void doOnSaveState() override;
  void doOnRestoreState() override;
  void UTdoOnFixedUpdate() override;
  void UTdoOnUpdate() override;
  void UTdoOnInit() override;
  void RTdoOnDeviceLost() override;
  void RTdoOnDeviceRegained() override;
  void RTdoOnWindowInit() override;
  VkResult RTdoOnRender(const vk::utils::SwapchainData& swapchainData) override;
  void RTdoOnResize() override;
  void RTdoOnSurfaceLost() override;
  void RTdoEarlySurfaceRegained() override;
  void RTdoLateSurfaceRegained() override;
  vk::SurfaceSpec doSurfaceSpec() override;

 private:
  void createConstantVulkanResources();
  void destroyConstantVulkanResources();
  void cleanupVulkanResources();
  void createVulkanResources();

  // Communication with display link
  std::atomic_bool m_displayLinkReady = false;
  // TODO refactor
  vk::GraphicsInfo m_graphicsInfo{};
  vk::GraphicsInfo m_skyboxGraphicsInfo{};
  VkPipeline m_graphicsPipeline = VK_NULL_HANDLE;
  VkPipeline m_skyboxPipeline = VK_NULL_HANDLE;
  DelayedConstruct<experimental::TextureLoaderKTX2> m_textureLoader;

  VkImageView m_depthView = VK_NULL_HANDLE;
  experimental::TextureInfo m_cubeTexInfo{};
  VkSampler m_cubeSampler = VK_NULL_HANDLE;

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

  // TODO better descriptor set management
  // -- main graphics pipeline
  VkDescriptorSet m_cubeDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorUpdateTemplateKHR m_descriptorUpdateTemplate = VK_NULL_HANDLE;

  // -- skybox
  VkDescriptorSet m_skyboxDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_skyboxDescriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorUpdateTemplateKHR m_skyboxDescriptorUpdateTemplate =
      VK_NULL_HANDLE;
};

}  // namespace avk