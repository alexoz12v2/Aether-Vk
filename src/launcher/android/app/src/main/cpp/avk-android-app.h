#pragma once

// our utils
#include "utils/mixins.h"

// our render (TODO add a vk/all.h)
#include "render/vk/command-pools.h"
#include "render/vk/descriptor-pools.h"
#include "render/vk/device-vk.h"
#include "render/vk/discard-pool.h"
#include "render/vk/instance-vk.h"
#include "render/vk/pipeline-info.h"
#include "render/vk/pipeline-pool-vk.h"
#include "render/vk/surface-vk.h"
#include "render/vk/swapchain-vk.h"

// application
#include "app/avk-application.h"

// JNI/Android stuff
#include <jni.h>

// libraries
#include <glm/glm.hpp>

// standard library
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

struct android_app;

namespace avk {

struct Camera {
  glm::mat4 view;
  glm::mat4 proj;
};

class AndroidApp : public ApplicationBase {
 public:
  AndroidApp(android_app *app, JNIEnv *jniEnv);
  ~AndroidApp() noexcept override;

  pthread_t RenderThread = 0;

 protected:
  void RTdoOnWindowInit() override;
  VkResult RTdoOnRender(vk::utils::SwapchainData const &swapchainData) override;
  void RTdoOnResize() override;
  void RTdoOnSurfaceLost() override;
  void RTdoEarlySurfaceRegained() override;
  void RTdoLateSurfaceRegained() override;
  vk::SurfaceSpec doSurfaceSpec() override;
  void doOnSaveState() override;
  void doOnRestoreState() override;
  void RTdoOnDeviceLost() override;
  void RTdoOnDeviceRegained() override;

 private:
  // stuff to refactor
  [[maybe_unused]] std::vector<Camera> m_pushCameras;// STABLE, FIF
  [[maybe_unused]] Camera m_camera;
  [[maybe_unused]] VkDescriptorSet m_cubeDescriptorSet = VK_NULL_HANDLE;
  [[maybe_unused]] VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
  [[maybe_unused]] VkDescriptorUpdateTemplateKHR m_descriptorUpdateTemplate = VK_NULL_HANDLE;

  // constant on resize
  vk::GraphicsInfo m_graphicsInfo;
  // variable on resize
  std::vector<VkFramebuffer> m_framebuffers;
  std::vector<uint64_t> m_commandBufferIds;
  VkPipeline m_graphicsPipeline = VK_NULL_HANDLE;
  VkImageView m_depthView = VK_NULL_HANDLE;

  void createConstantVulkanResources();
  void destroyConstantVulkanResources();
  void cleanupVulkanResources();
  void createVulkanResources();

  // app state
  const android_app *m_app = nullptr;
  JNIEnv *m_jniEnv = nullptr;
};

}  // namespace avk