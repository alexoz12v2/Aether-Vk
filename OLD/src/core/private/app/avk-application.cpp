#include "app/avk-application.h"

namespace avk {

// --------------------------- Static impl details ------------------------
static void addPlatformEvents(avk::EventSystem *eventSystem) {
#if defined(AVK_OS_ANDROID) || defined(AVK_OS_IOS)
  eventSystem->addEvent(events::EvTouchBegin);
  eventSystem->addEvent(events::EvTouchEnd);
  eventSystem->addEvent(events::EvTouchMove);
  eventSystem->addEvent(events::EvTouchCancel);
  eventSystem->addEvent(events::EvAppPause);
  eventSystem->addEvent(events::EvAppResume);
  eventSystem->addEvent(events::EvAppLowMemory);
#elif defined(AVK_OS_WINDOWS) || defined(AVK_OS_LINUX) || defined(AVK_OS_MACOS)
  eventSystem->addEvent(events::EvKeyDown);
  eventSystem->addEvent(events::EvKeyUp);
  eventSystem->addEvent(events::EvKeyRepeat);
  eventSystem->addEvent(events::EvMouseButtonDown);
  eventSystem->addEvent(events::EvMouseButtonUp);
  eventSystem->addEvent(events::EvMouseMove);
  eventSystem->addEvent(events::EvMouseScrollUp);
  eventSystem->addEvent(events::EvMouseScrollDown);
  eventSystem->addEvent(events::EvMouseEnter);
  eventSystem->addEvent(events::EvMouseLeave);
#else
#  error "Unrecognized OS platoform"
#endif
  // common events
  eventSystem->addEvent(events::EvTextInput);
  eventSystem->addEvent(events::EvWindowResize);
  eventSystem->addEvent(events::EvWindowClose);
  eventSystem->addEvent(events::EvWindowFocus);
  eventSystem->addEvent(events::EvWindowLostFocus);
  eventSystem->addEvent(events::EvWindowMinimize);
  eventSystem->addEvent(events::EvWindowMaximize);
  eventSystem->addEvent(events::EvWindowRestore);
}

// ----------------------------- Entry Points ----------------------------

void ApplicationBase::RTmain(ApplicationBase *app) {
  LOGI << "[RenderThread] started with window ready" << std::endl;
  app->RTwindowInit();
  LOGI << "[RenderThread] rendering started" << std::endl;
  // keep running while rendering
  while (app->RTshouldRun()) {
    if (app->RTsurfaceWasLost()) {
      app->RTsurfaceLost();
    }
    // this shouldn't return `true` if the application is
    // currently in the pause state
    if (app->RTshouldUpdate()) {
      // successfully claimed state: render it
      app->RTonRender();
    } else {
      // if CAS failed or nothing to render, fallthrough and wait
      // LOGI << "AVK Render Thread NO UPDATE, GOING TO WAIT" << std::endl;
      app->RTwaitForNextRound();
      // LOGI << "AVK Render Thread WOKE UP" << std::endl;
    }
  }
  app->RTsignalExit();
  LOGI << "[RenderThread] exiting" << std::endl;
}

void ApplicationBase::UTmain(ApplicationBase *app) {
  assert(app);
  LOGI << "[UpdateThread] Started Update Thread" << std::endl;
  app->UTonInit();
  while (app->UTshouldRun()) {
    // update timings from last frame
    app->Time.UTupdate();
    while (app->Time.needsFixedUpdate()) {
      app->Time.UTfixedUpdate();
      app->UTonFixedUpdate();
    }
    app->UTonUpdate();
  }
  LOGI << "[UpdateThread] Exiting Update Thread" << std::endl;
}

// ---------------------------- Class Implementation -------------------------

ApplicationBase::ApplicationBase() {
  // Vulkan State Initialization
  m_vkInstance.create();
  LOGI << "[ApplicationBase] Vulkan Instance " << std::hex
       << m_vkInstance.get()->handle() << std::dec << " Created" << std::endl;
}

ApplicationBase::~ApplicationBase() noexcept AVK_NO_CFI {
  LOGI << "[ApplicationBase] Destructor Running ..." << std::endl;
  if (m_windowInit) {
    auto const *const vkDevApi = m_vkDevice->table();
    vkDevApi->vkDeviceWaitIdle(m_vkDevice->device());

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
  } else {
    // Event System Initialization
    // ensure render thread and update thread, when started, can register events
    addPlatformEvents(&m_eventSystem);

    // start update thread only after the events from the platform have been
    // added
    m_updateCoordinator.updateShouldRun.store(true, std::memory_order_release);

    // why is this a heap allocated size_t and not a atomic? Because I've tried
    // with it, and it was fine until I added the global/implicit Vulkan layers
    // from RENDERDOC. how is this correlated? No Idea🤪
    {
      std::lock_guard lk{m_renderCoordinator.mtx};
      *m_renderCoordinator.shouldInitialize = 1;
    }
  }
  LOGI << "m_renderCoordinator.shouldInitialize: "
       << *m_renderCoordinator.shouldInitialize << std::endl;
  m_renderCoordinator.cv.notify_all();
}

void ApplicationBase::onResize() {
  if (windowInitializedOnce()) {
    // since you check for resizes on `vkAcquireNextImageKHR` and on
    // `vkQueuePresentKHR`, command buffers should already be either clean or
    // already submitted. In the latter case we would need to wait for the
    // signal semaphore of their `vkQueueSubmit`. Therefore, to be sure, we
    // discard the command pool
    m_vkCommandPools.get()->discardActivePool(m_vkDiscardPool.get(),
                                              m_timeline);
    m_vkSwapchain.get()->recreateSwapchain();
    RTdoOnResize();
  }
}

void ApplicationBase::onSurfaceLost() {
  m_renderCoordinator.surfaceLost.store(true, std::memory_order_release);
  // wake the render thread up such that it can see that the surface was lost
  m_renderCoordinator.cv.notify_all();
}

void ApplicationBase::RTonRender() AVK_NO_CFI {
  auto const *const vkDevApi = m_vkDevice->table();

  // TODO per panel refresh flags
  if (!m_shouldRender.load(std::memory_order_relaxed)) {
    return;
  }
  // check for destruction of discarded resources. Monitor ticks every N frame
  // to collect resources which are not used anymore calling the discard pool
  m_vkDiscardPoolMonitor.get()->onFrame();

  // acquire a swapchain image (handle resize pt.1)
  VkResult res = m_vkSwapchain.get()->acquireNextImage();
  if (res == VK_ERROR_SURFACE_LOST_KHR) {
    RTsurfaceLost();
    return;
  }
  if (res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR) {
    LOGI << "[RT Render] onResize from acquire" << std::endl;
    onResize();
    return;
  }
  auto const swapchainData = m_vkSwapchain.get()->swapchainData();
#if 0  // done by acquisition TODO remove
  if (swapchainData.submissionFence != VK_NULL_HANDLE) {
    VK_CHECK(vkDevApi->vkWaitForFences(dev, 1, &swapchainData.submissionFence,
                                       VK_TRUE, UINT64_MAX));
    VK_CHECK(vkDevApi->vkResetFences(dev, 1, &swapchainData.submissionFence));
  }
#endif
  // reset the fence only after a successful acquisition, avoiding fence freeze
  // when reacquiring for the same frame after a resize
  VK_CHECK(vkDevApi->vkResetFences(m_vkDevice->device(), 1,
                                   &swapchainData.submissionFence));

  // prepare VMA for querying memory budget
  m_vkDevice->refreshMemoryBudgets(m_vkSwapchain.get()->frameIndex());
  [[maybe_unused]] auto const &memoryBudgets = m_vkDevice->heapBudgets();
#ifdef AVK_DEBUG
  if ((m_timeline % 30000) == 0) {
    uint32_t heapIndex = 0;
    for (VmaBudget const &heapBudget : memoryBudgets) {
      uint32_t const heap = heapIndex++;
      LOGI << "[Render Thread::OnRender, Memory Usage] (heap " << heap
           << ") USAGE: " << heapBudget.usage
           << " B | AVAIL: " << heapBudget.budget
           << " B\n\tblockCount: " << heapBudget.statistics.blockCount
           << " | blockBytes :" << heapBudget.statistics.blockBytes
           << " | allocationCount: " << heapBudget.statistics.allocationCount
           << " | allocationBytes: " << heapBudget.statistics.allocationBytes
           << std::endl;
    }
  }
#endif

  // call overridden rendering function
  res = RTdoOnRender(swapchainData);
  if (res == VK_ERROR_DEVICE_LOST) {
    RThandleDeviceLost();
    return;
  }
  VK_CHECK(res);
  // increment timeline on successful submit
  m_timeline++;

  // increment frame index after submission (fence is what we care about)
  // NOT presentation (otherwise might deadlock on submission fence)
  m_vkSwapchain.get()->signalNextFrame();

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

  res = vkDevApi->vkQueuePresentKHR(m_vkDevice->queue(), &presentInfo);
  if (res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR) {
    LOGI << "[RT Render] onResize from presentation" << std::endl;
    onResize();
    LOGI << "[RT Render] AFTER onResize from presentation" << std::endl;
  } else if (res == VK_ERROR_DEVICE_LOST) {
    RThandleDeviceLost();
    return;
  } else if (res == VK_ERROR_SURFACE_LOST_KHR) {
    RTsurfaceLost();
    return;
  } else {  // TODO VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT
    VK_CHECK(res);
  }
}

void ApplicationBase::onSaveState() { doOnSaveState(); }

void ApplicationBase::onRestoreState() { doOnRestoreState(); }

void ApplicationBase::RThandleDeviceLost() {
  RTdoOnDeviceLost();
  // Note: We are not discarding resources, because the user is responsible to
  // clean them up
  m_vkCommandPools.get()->threadShutdown();
  m_vkDiscardPool.get()->destroyDiscardedResources(true);
  RTdestroyDeviceAndDependencies();
  // crash if physical device is lost
  RTcreateDeviceAndDependencies();
  // if we are still alive, then the device was regained
  RTdoOnDeviceRegained();
  // reset the timeline when the device is reacquired!
  m_timeline = 0;
}

void ApplicationBase::RTdestroyDeviceAndDependencies() {
  // resource managers (Note: user is responsible to dump their things)
  m_bufferManager.destroy();
  m_imageManager.destroy();

  // resource handling mechanisms
  m_vkPipelines.destroy();
  m_vkCommandPools.destroy();
  m_vkDescriptorPools.destroy();
  m_vkDiscardPoolMonitor.destroy();
  // last before main 3
  m_vkDiscardPool.destroy();

  // main 3 handles
  m_vkSwapchain.destroy();
  m_vkDevice.reset();
}

void ApplicationBase::RTcreateDeviceAndDependencies() {
#define PREFIX "[ApplicationBase::RTcreateDeviceAndDependencies] "
  m_vkDevice.emplace(m_vkInstance.get(), m_vkSurface.get());
  LOGI << PREFIX "Vulkan Device Created" << std::endl;

  // if you use a non-standard layout class directly stored in memory, the
  // derived class won't see it properly (neither will protected/public getters)
  // leave this so that if errors of such kind happen, you know the cause
  LOGI << PREFIX
      "Want to see some magic?:\n\t"
      "Direct Access    = 0x"
       << (void *)&*m_vkDevice << "\n\tProtected Getter = 0x"
       << (void *)vkDevice() << "\n\tPrivate Getter   = 0x"
       << (void *)vkDevicePrivate() << std::endl;

  // note: the initial create swapchain might get a surface lost
  // if you quickly exit from the application after entering. We don't care.
  m_vkSwapchain.create(m_vkInstance.get(), m_vkSurface.get(), vkDevice());
  LOGI << PREFIX "Vulkan Swapchain Created" << std::endl;

  m_vkDiscardPool.create(m_vkInstance.get(), vkDevice());
  m_vkDiscardPoolMonitor.create(m_vkDiscardPool.get());
  LOGI << PREFIX "Discard Pool Created" << std::endl;
  m_vkCommandPools.create(vkDevice(),
                          m_vkDevice->universalGraphicsQueueFamilyIndex());
  LOGI << PREFIX "Command Pools Created" << std::endl;
  m_vkDescriptorPools.create(vkDevice());
  LOGI << PREFIX "Descriptor Pools Created" << std::endl;
  m_vkPipelines.create(vkDevice());
  LOGI << PREFIX "Pipeline Pool Created" << std::endl;
  m_bufferManager.create(vkDevice());
  m_imageManager.create(vkDevice());
  LOGI << PREFIX "[Experimental] Buffer/Image Manager created" << std::endl;
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
  m_vkSwapchain.get()->forceDiscardToCurrentFrame(m_vkSurface.get()->handle());
  m_vkSurface.get()->reset();
  m_vkSwapchain.destroy();
  m_vkSurface.destroy();
  vk::SurfaceSpec surfSpec = doSurfaceSpec();
  m_vkSurface.create(m_vkInstance.get(), surfSpec);
  m_vkSwapchain.create(m_vkInstance.get(), m_vkSurface.get(), vkDevice());
  LOGI << PREFIX "Swapchain And Surface Recreated" << std::endl;
  RTdoLateSurfaceRegained();
#undef PREFIX
}

void ApplicationBase::RTwindowInit() {
#define PREFIX "[ApplicationBase::onWindowInit] "
  LOGI << PREFIX "Window Init Started on render thread" << std::endl;
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

void ApplicationBase::publishPlatformEvent(const Event &ev) {
  m_eventSystem.publish(ev);
}

#undef PREFIX

}  // namespace avk
