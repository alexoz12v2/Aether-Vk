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

// library
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

struct android_app;

namespace avk {

class AndroidApp : public ApplicationBase {
 public:
  AndroidApp(android_app* app);
  ~AndroidApp() noexcept override;

 protected:
  void RTdoOnWindowInit() override;
  VkResult RTdoOnRender(vk::utils::SwapchainData const& swapchainData) override;
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

  // constant on resize
  vk::GraphicsInfo m_graphicsInfo;
  VkBuffer m_vertexBuffer = VK_NULL_HANDLE;
  VmaAllocation m_vertexAlloc = VK_NULL_HANDLE;
  // variable on resize
  std::vector<VkFramebuffer> m_framebuffers;
  std::vector<uint64_t> m_commandBufferIds;
  VkPipeline m_graphicsPipeline = VK_NULL_HANDLE;
  VkImage m_depthImage = VK_NULL_HANDLE;
  VmaAllocation m_depthAlloc = VK_NULL_HANDLE;
  VkImageView m_depthView = VK_NULL_HANDLE;

  void createConstantVulkanResources();
  void destroyConstantVulkanResources();
  void cleanupVulkanResources();
  void createVulkanResources();

  // app state
  const android_app* m_app = nullptr;
  JNIEnv* m_jniEnv = nullptr;
};

}  // namespace avk