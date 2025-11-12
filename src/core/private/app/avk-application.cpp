#include "app/avk-application.h"

namespace avk {

ApplicationBase::ApplicationBase() {
  m_vkInstance.create();
  LOGI << "[ApplicationBase] Vulkan Instance " << std::hex
       << m_vkInstance.get()->handle() << std::dec << " Created" << std::endl;
}

ApplicationBase::~ApplicationBase() noexcept {
  LOGI << "[AndroidApp] Destructor Running ..." << std::endl;
  if (m_windowInit) {
    auto const* const vkDevApi = m_vkDevice.get()->table();
    vkDevApi->vkDeviceWaitIdle(m_vkDevice.get()->device());

    // resource handling mechanisms
    RTdestroyDeviceAndDependencies();
    m_vkSurface.destroy();
  }
  m_vkInstance.destroy();
}

void ApplicationBase::onWindowInit() AVK_NO_CFI {
  // if `m_windowInit` is true, this is not the first time the primary window
  // has been initialized. On desktop platforms, this should never happen. On
  // mobile platforms this means that the window has been brought back up

  // Note: Since we are recreating the surface, and the queue family we acquired
  // depends on it, we should, normally we should recompute the proper queue
  // family used for presentation. We won't do that, because both Android and
  // iOS ensure that every queue family is presentation capable by specification
  // reference:
  // https://docs.vulkan.org/spec/latest/chapters/VK_KHR_surface/wsi.html

  if (m_windowInit.load(std::memory_order_acquire)) {
    m_renderCoordinator.surfaceLost.store(false, std::memory_order_release);
    m_renderCoordinator.cv.notify_all();
  } else {
    m_renderCoordinator.shouldInitialize.store(true, std::memory_order_release);
  }
}

void ApplicationBase::onResize() {
  m_vkSwapchain.get()->recreateSwapchain();
  RTdoOnResize();
}

void ApplicationBase::onSurfaceLost() {
  m_renderCoordinator.surfaceLost.store(true, std::memory_order_release);
}

void ApplicationBase::RTonRender() AVK_NO_CFI {
  auto const* const vkDevApi = m_vkDevice.get()->table();
  VkDevice const dev = m_vkDevice.get()->device();

  if (!m_shouldRender.load(std::memory_order_relaxed)) {
    return;
  }
  // check for destruction of discarded resources. Monitor ticks every N frame
  // to collect resources which are not used anymore calling the discard pool
  m_vkDiscardPoolMonitor.get()->onFrame();

  // acquire a swapchain image (handle resize pt.1)
  VkResult res = m_vkSwapchain.get()->acquireNextImage();
  if (res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR) {
    onResize();
    return;
  }
  auto const swapchainData = m_vkSwapchain.get()->swapchainData();
  if (swapchainData.submissionFence != VK_NULL_HANDLE) {
    VK_CHECK(vkDevApi->vkWaitForFences(dev, 1, &swapchainData.submissionFence,
                                       VK_TRUE, UINT64_MAX));
    VK_CHECK(vkDevApi->vkResetFences(dev, 1, &swapchainData.submissionFence));
  }

  res = RTdoOnRender(swapchainData);
  if (res == VK_ERROR_DEVICE_LOST) {
    RThandleDeviceLost();
  }
  VK_CHECK(res);

  // queue present
  VkSwapchainKHR const swapchain = m_vkSwapchain.get()->handle();
  uint32_t const imageIndex = m_vkSwapchain.get()->imageIndex();
  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &swapchain;
  presentInfo.pImageIndices = &imageIndex;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &swapchainData.presentSemaphore;

  res = vkDevApi->vkQueuePresentKHR(m_vkDevice.get()->queue(), &presentInfo);
  if (res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR) {
    onResize();
  } else if (res == VK_ERROR_DEVICE_LOST) {
    RThandleDeviceLost();
  } else if (res == VK_ERROR_SURFACE_LOST_KHR) {
    RThandleSurfaceLost();
  } else {  // TODO VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT
    VK_CHECK(res);
  }

  m_vkSwapchain.get()->signalNextFrame();
  m_timeline++;
}

void ApplicationBase::onSaveState() { doOnSaveState(); }

void ApplicationBase::onRestoreState() { doOnRestoreState(); }

void ApplicationBase::RThandleDeviceLost() {
  RTdoOnDeviceLost();
  m_vkCommandPools.get()->threadShutdown();
  m_vkDiscardPool.get()->destroyDiscardedResources(true);
  RTdestroyDeviceAndDependencies();
  RTcreateDeviceAndDependencies();
  // if we are still alive, then the device was regained
  RTdoOnDeviceRegained();
}

void ApplicationBase::RTdestroyDeviceAndDependencies() {
  // resource handling mechanisms
  m_vkPipelines.destroy();
  m_vkCommandPools.destroy();
  m_vkDescriptorPools.destroy();
  m_vkDiscardPoolMonitor.destroy();
  // last before main 3
  m_vkDiscardPool.destroy();

  // main 3 handles
  m_vkSwapchain.destroy();
  m_vkDevice.destroy();
}

void ApplicationBase::RTcreateDeviceAndDependencies() {
#define PREFIX "[ApplicationBase::RTcreateDeviceAndDependencies] "
  m_vkDevice.create(m_vkInstance.get(), m_vkSurface.get());
  LOGI << PREFIX "Vulkan Device Created" << std::endl;
  m_vkSwapchain.create(m_vkInstance.get(), m_vkSurface.get(), m_vkDevice.get());
  LOGI << PREFIX "Vulkan Swapchain Created" << std::endl;

  m_vkDiscardPool.create(m_vkInstance.get(), m_vkDevice.get());
  m_vkDiscardPoolMonitor.create(m_vkDiscardPool.get());
  LOGI << PREFIX "Discard Pool Created" << std::endl;
  m_vkCommandPools.create(
      m_vkDevice.get(), m_vkDevice.get()->universalGraphicsQueueFamilyIndex());
  LOGI << PREFIX "Command Pools Created" << std::endl;
  m_vkDescriptorPools.create(m_vkDevice.get());
  LOGI << PREFIX "Descriptor Pools Created" << std::endl;
  m_vkPipelines.create(m_vkDevice.get());
  LOGI << PREFIX "Pipeline Pool Created" << std::endl;
#undef PREFIX
}

void ApplicationBase::RThandleSurfaceLost() {
#define PREFIX "[ApplicationBase::RThandleSurfaceLost] "
  // Releases all command pools. Might be not optimal, but this
  // event should be rare
  m_vkCommandPools.get()->discardActivePool(m_vkDiscardPool.get(), m_timeline);
  // This cleans up also depth/stencil attachment, which isn't
  // strictly necessary here, but we don't care
  RTdoEarlySurfaceRegained();
  m_vkSwapchain.get()->forceDiscardToCurrentFrame();
  m_vkDiscardPool.get()->discardSurface(m_vkSurface.get()->handle(),
                                        m_timeline);
  m_vkSurface.get()->reset();
  m_vkSwapchain.destroy();
  m_vkSurface.destroy();
  vk::SurfaceSpec surfSpec = doSurfaceSpec();
  m_vkSurface.create(m_vkInstance.get(), surfSpec);
  m_vkSwapchain.create(m_vkInstance.get(), m_vkSurface.get(), m_vkDevice.get());
  LOGI << PREFIX "Swapchain And Surface Recreated" << std::endl;
  RTdoLateSurfaceRegained();
#undef PREFIX
}

void ApplicationBase::RTwindowInit() {
#define PREFIX "[ApplicationBase::onWindowInit] "
  vk::SurfaceSpec surfSpec = doSurfaceSpec();
  m_vkSurface.create(m_vkInstance.get(), surfSpec);
  AVK_EXT_CHECK(*m_vkSurface.get());
  LOGI << PREFIX "Vulkan Surface " << std::hex << m_vkSurface.get()->handle()
       << std::dec << " Created" << std::endl;
  RTcreateDeviceAndDependencies();
  RTdoOnWindowInit();

  m_windowInit.store(true, std::memory_order_release);
  m_renderCoordinator.renderRunning.store(true, std::memory_order_release);
}
#undef PREFIX

}  // namespace avk
