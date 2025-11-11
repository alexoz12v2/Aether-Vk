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

// JNI/Android stuff
#include <jni.h>

struct android_app;

namespace avk {

class AndroidApp : public NonMoveable {
 public:
  AndroidApp(android_app* app);
  ~AndroidApp() noexcept;

  void onWindowInit();
  void onRender();
  void onResize();

  inline void pauseRendering() {
    m_shouldRender.store(false, std::memory_order_relaxed);
  }
  inline void resumeRendering() {
    m_shouldRender.store(true, std::memory_order_relaxed);
  }

  inline vk::Instance* vkInstance() { return m_vkInstance.get(); }
  inline vk::Surface* vkSurface() { return m_vkSurface.get(); }
  inline vk::Device* vkDevice() { return m_vkDevice.get(); }
  inline vk::Swapchain* vkSwapchain() { return m_vkSwapchain.get(); }
  inline bool windowInitialized() { return m_windowInit; }

 private:
  // vulkan stuff
  DelayedConstruct<vk::Instance> m_vkInstance;
  DelayedConstruct<vk::Device> m_vkDevice;
  // on android, there's only one of these, hence we integrate it here
  DelayedConstruct<vk::Surface> m_vkSurface;
  DelayedConstruct<vk::Swapchain> m_vkSwapchain;
  DelayedConstruct<vk::DiscardPool> m_vkDiscardPool;
  DelayedConstruct<vk::CommandPools> m_vkCommandPools;
  DelayedConstruct<vk::DescriptorPools> m_vkDescriptorPools;
  DelayedConstruct<vk::PipelinePool> m_vkPipelines;

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

  // useless state
  uint64_t m_timeline = 0;
  bool m_windowInit = false;
  std::atomic_bool m_shouldRender = false;
};

}  // namespace avk