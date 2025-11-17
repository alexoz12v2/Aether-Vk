#pragma once

#include "utils/mixins.h"

// rendering
#include "render/vk/command-pools.h"
#include "render/vk/descriptor-pools.h"
#include "render/vk/device-vk.h"
#include "render/vk/discard-pool.h"
#include "render/vk/instance-vk.h"
#include "render/vk/pipeline-pool-vk.h"
#include "render/vk/surface-vk.h"
#include "render/vk/swapchain-vk.h"

// rendering stuff which might change
#include "render/experimental/avk-basic-buffer-manager.h"
#include "render/experimental/avk-basic-image-manager.h"

// library
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "os/avk-time.h"

namespace avk {

struct RenderCoordinator : public NonMoveable {
  RenderCoordinator() = default;
#ifdef AVK_DEBUG
  inline ~RenderCoordinator() noexcept {
    LOGI << "------------------------------ RenderCoordinator Destructor "
            "-------------------------------------------"
         << std::endl;
  }
#endif

  static constexpr uint32_t MaxWaitMillis = 16;

  /// Variable to notify the Render thread it's ok to start
  /// initializing the Vulkan related rendering context
  /// example:
  ///  - Windows -> `HWND` + COM Apartment created on UI Thread
  ///  - Android -> `APP_CMD_INIT_WINDOW` received for the first time
  std::unique_ptr<size_t> shouldInitialize = std::make_unique<size_t>(0);

  /// Variable to notify the render thread the primary window was
  /// lost, hence it should handle that
  std::atomic_bool surfaceLost = false;

  /// variable to notify whether render thread should continue running
  /// or if should stop.
  std::atomic_bool renderRunning = false;

  /// similar to timeline mechanism for vulkan synchronization. communicates
  /// the "version" for the simulation which is currently produced
  std::atomic_uint64_t stateVersion = 0;

  /// similar to timeline mechanism for vulkan synchronization. communicates
  /// the "version" for the simulation which was lastly consumed -> Render only
  /// if there are updates in the state To achieve traditional games behaviour,
  /// just update this thing every time
  std::atomic_uint64_t consumedVersion = 0;

  /// mutex and condition variable are to be signaled when the producer of the
  /// new state is finished and will be waited for on the render thread
  mutable std::mutex mtx;

  /// mutex and condition variable are to be signaled when the producer of the
  /// new state is finished and will be waited for on the render thread
  std::condition_variable cv;
};

struct UpdateCoordinator : public NonMoveable {
  UpdateCoordinator() = default;

  std::atomic_bool updateShouldRun = true;
};

class ApplicationBase : public NonMoveable {
 public:  // Threads Starting Procedures
  static void RTmain(ApplicationBase *app);
  static void UTmain(ApplicationBase *app);

 public:
  /// Initialized to have fixed update for 60 PFS, max delta of 24 FPS, 1 scale
  /// \warning should never be reassigned
  os::TimeInfo Time{16'667, 41'666, 1.f};

 public:
  ApplicationBase();
  virtual ~ApplicationBase() noexcept;

  // -------------------- Update Thread - Main Thread Coordination -------
  inline bool UTshouldRun() const {
    return m_updateCoordinator.updateShouldRun.load(std::memory_order_acquire);
  }

  inline void signalStopUpdating() {
    m_updateCoordinator.updateShouldRun.store(false, std::memory_order_release);
  }

  inline void UTonFixedUpdate() { UTdoOnFixedUpdate(); }

  inline void UTonInit() { UTdoOnInit(); }

  inline void UTonUpdate() {
    UTdoOnUpdate();
    signalStateUpdated();
    // TODO frameRate Governing if necessary
  }

  // -------------------- Main Thread Events ------------------------------
  void onWindowInit();
  void onResize();

  /// function called on `APP_CMD_WINDOW_TERM` on Android (iOS TODO)
  /// meaning when the user goes to some other Activity/Window, but
  /// this one stays in memory
  void onSurfaceLost();

  /// Function which implements the structure of an iteration of
  /// a render loop. It should run in its own thread
  void RTonRender();

  /// function which should be called from the application
  /// whenever it is necessary to persist some state
  /// examples:
  /// - Desktop: based on a autosave timer
  /// - Android: `APP_CMD_PAUSE` (`onPause`), which signals to stop rendering
  ///   and that the Activity might get nuked once it stays out of focus for
  ///   enough time
  /// - Note: Another Android Example of pause is when an activity goes from
  ///   full screen to Picture in Picture mode:
  ///   - `APP_CMD_PAUSE`->`APP_CMD_LOST_FOCUS`->`APP_CMD_INIT_WINDOW`
  void onSaveState();

  /// Function which should be called whenever a persistent state
  /// is loaded into the application
  /// - Desktop: load a stored configuration
  /// - Android: `APP_CMD_RESUME` (`onResume`), which is called whenever the app
  /// needs to restore its state
  void onRestoreState();

  /// Desktop method to signal to swapchain to defer recreation unti resize
  /// exited
  void onEnterResize() {
    assert(*m_vkSwapchain.get());
    m_vkSwapchain.get()->lockResize();
  }

  void onExitResize() {
    assert(*m_vkSwapchain.get());
    m_vkSwapchain.get()->unlockResize();
  }

  inline void pauseRendering() {
    LOGI << "[Application] Paused Rendering" << std::endl;
    m_shouldRender.store(false, std::memory_order_relaxed);
  }
  inline void resumeRendering() {
    LOGI << "[Application] (Re)Started Rendering" << std::endl;
    m_shouldRender.store(true, std::memory_order_relaxed);
  }

  /// Signal to render thread it's time to stop rendering for good
  inline void signalStopRendering() {
    m_renderCoordinator.renderRunning.store(false, std::memory_order_seq_cst);
    m_renderCoordinator.cv.notify_all();
  }
  /// Signal to render thread the next state is ready
  /// TODO Now it's on the main thread, it should be on the update thread
  inline void signalStateUpdated() {
    m_renderCoordinator.stateVersion.fetch_add(1, std::memory_order_release);
    m_renderCoordinator.cv.notify_all();
  }

  /// whether primary window initalization happened. Useful on
  /// derived classes destrouctor
  inline bool windowInitializedOnce() const {
    return m_windowInit.load(std::memory_order_relaxed);
  }

  /// Called by the render thread to know when it's time to initialize itself
  inline void RTwaitReadyForInit() {
    std::unique_lock lk{m_renderCoordinator.mtx};
    m_renderCoordinator.cv.wait(lk, [this]() {
      bool const ready = *m_renderCoordinator.shouldInitialize == 1;
      return ready;
    });
    LOGI << "[RT WOKE UP] Woke up: " << *m_renderCoordinator.shouldInitialize
         << std::endl;
  }

  void RTwindowInit();

  inline bool RTshouldRun() const {
    return m_renderCoordinator.renderRunning.load(std::memory_order_acquire);
  }

  /// Android's bionic libc++ doesn't have pthread_timedjoin_np,
  /// hence render pthread about to exit acknowledges the termination request
  inline void RTsignalExit() {
    m_renderCoordinator.renderRunning.store(true, std::memory_order_release);
  }

  inline bool checkRTAck() const {
    return m_renderCoordinator.renderRunning.load(std::memory_order_acquire);
  }

  inline bool RTshouldUpdate() {
    uint64_t const lastConsumed =
        m_renderCoordinator.consumedVersion.load(std::memory_order_acquire);
    uint64_t const latestState =
        m_renderCoordinator.stateVersion.load(std::memory_order_acquire);

    if (latestState <= lastConsumed) {
      return false;
    }

    // try to automatically consumed state
    uint64_t expected = lastConsumed;
    return m_renderCoordinator.consumedVersion.compare_exchange_strong(
        expected, latestState, std::memory_order_acq_rel,
        std::memory_order_acquire);
  }

  inline void RTwaitForNextRound() {
    // nothing to render or lost CAS, so wait to be notified or timeout
    std::unique_lock lk{m_renderCoordinator.mtx};
    m_renderCoordinator.cv.wait_for(
        lk, std::chrono::milliseconds(RenderCoordinator::MaxWaitMillis),
        [coord = &m_renderCoordinator]() {
          // if surface lost wake up and handle that before being killed!
          return coord->surfaceLost.load(std::memory_order_relaxed) ||
                 (!coord->renderRunning.load(std::memory_order_acquire) ||
                  coord->stateVersion.load(std::memory_order_acquire) >
                      coord->consumedVersion.load(std::memory_order_acquire));
        });
  }

  inline bool RTsurfaceWasLost() const {
    if (m_renderCoordinator.surfaceLost.load(std::memory_order_relaxed)) {
      return m_renderCoordinator.surfaceLost.load(std::memory_order_acquire);
    }
    return false;
  }

  inline void RTsurfaceLost() {
    assert(m_renderCoordinator.surfaceLost.load(std::memory_order_relaxed));
    LOGI << "[RenderThread::surfaceLost]" << std::endl;
    static constexpr uint32_t TimeoutWakeupMillis = 256;
    RTdoOnSurfaceLost();
    while (m_renderCoordinator.surfaceLost.load(std::memory_order_acquire)) {
      std::unique_lock lk{m_renderCoordinator.mtx};
      m_renderCoordinator.cv.wait_for(
          lk, std::chrono::milliseconds(TimeoutWakeupMillis),
          [coord = &m_renderCoordinator]() {
            return !coord->surfaceLost.load(std::memory_order_acquire);
          });
    }
    LOGI << "[RenderThread::surfaceLost] Surface regained" << std::endl;
    // only after surface was regained, recreate everything
    RThandleSurfaceLost();
  }

 protected:  // virtual interface
  // ----------------- Main Thread Interface -----------------------------
  virtual void doOnSaveState() = 0;
  virtual void doOnRestoreState() = 0;

  // --------------- Update Thread Interface -----------------------------
  virtual void UTdoOnFixedUpdate() = 0;
  virtual void UTdoOnUpdate() = 0;
  virtual void UTdoOnInit() = 0;

  // --------------- Render Thread Interface -----------------------------
  /// called when `vkQueueSubmit` or `vkQueuePresentKHR` returns
  /// `VK_ERROR_DEVICE_LOST` it should cleanup any resources used by the
  /// application *before* the default behaviour kicks in, which is destroy
  /// everything and recreate everything
  virtual void RTdoOnDeviceLost() = 0;
  virtual void RTdoOnDeviceRegained() = 0;

  /// called after first Primary window initialization
  /// earliest point in which all primary vulkan handles and pools are available
  virtual void RTdoOnWindowInit() = 0;

  /// called after successful acquisition of a swapchain Image
  /// This function is responsible to submit to the graphics/present
  /// queue some work which will signal the `presentSemaphore` inside
  /// the `swapchainData`
  /// should return the `VkResult` of the last submission, or result of the
  /// first failed submission as it will handle checking for
  /// `VK_ERROR_DEVICE_LOST` and call the callback
  virtual VkResult RTdoOnRender(
      vk::utils::SwapchainData const &swapchainData) = 0;

  /// Function to react on a resize/rotation of the screen
  virtual void RTdoOnResize() = 0;

  /// exampls: call `threadShutdown` on all other threads for command buffer
  /// pool
  virtual void RTdoOnSurfaceLost() = 0;

  /// called after the surface is regained, on Android it would
  /// be on a `APP_CMD_WINDOW_INIT`, *before* destroying the current
  /// `VkSwapchainKHR` and `VkSurfaceKHR`
  /// This should be used to cleanup any resources still linked
  /// to the current surface handle
  /// example: `threadShutdown` for command buffers
  virtual void RTdoEarlySurfaceRegained() = 0;
  /// called after the surface is regained, on Android it would
  /// be on a `APP_CMD_WINDOW_INIT`, *After* destroying the current
  /// `VkSwapchainKHR` and `VkSurfaceKHR`
  /// This should be used to recreate any resources linked
  /// to the previous surface/swapchain handle
  virtual void RTdoLateSurfaceRegained() = 0;
  virtual vk::SurfaceSpec doSurfaceSpec() = 0;

 protected:
  // --------------- getters for subclasses ----------------------------
  inline VolkDeviceTable const *vkDevTable() const {
    return m_vkDevice.get()->table();
  }
  inline VkPhysicalDevice vkPhysicalDeviceHandle() const {
    return m_vkDevice.get()->physicalDevice();
  }
  inline VmaAllocator vmaAllocator() const {
    return m_vkDevice.get()->vmaAllocator();
  }
  inline VkDevice vkDeviceHandle() const { return m_vkDevice.get()->device(); }
  inline uint64_t timeline() const { return m_timeline; }
  inline vk::Instance *vkInstance() { return m_vkInstance.get(); };
  inline vk::Device *vkDevice() { return m_vkDevice.get(); };
  inline vk::Surface *vkSurface() { return m_vkSurface.get(); };
  inline vk::Swapchain *vkSwapchain() { return m_vkSwapchain.get(); };
  inline vk::DiscardPool *vkDiscardPool() { return m_vkDiscardPool.get(); };
  inline vk::DiscardPoolMonitor *vkDiscardPoolMonitor() {
    return m_vkDiscardPoolMonitor.get();
  };
  inline vk::CommandPools *vkCommandPools() { return m_vkCommandPools.get(); };
  inline vk::DescriptorPools *vkDescriptorPools() {
    return m_vkDescriptorPools.get();
  };
  inline vk::PipelinePool *vkPipelines() { return m_vkPipelines.get(); };

  inline experimental::BufferManager *bufferManager() {
    return m_bufferManager.get();
  }
  inline experimental::ImageManager *imageManager() {
    return m_imageManager.get();
  }

  inline RenderCoordinator &renderCoordinator() { return m_renderCoordinator; }
  inline UpdateCoordinator &updateCoordinator() { return m_updateCoordinator; }

 private:
  void RThandleSurfaceLost();
  void RThandleDeviceLost();
  void RTdestroyDeviceAndDependencies();
  void RTcreateDeviceAndDependencies();

 private:
  // ----------- Vulkan Basic State -------------
  /// Basic RAII wrapper for VkInstance (using volk)
  DelayedConstruct<vk::Instance> m_vkInstance;
  /// Basic RAII wrapper for VkDevice and VmaAllocator
  /// depends on: `m_vkInstance`, `m_vkSurface`
  DelayedConstruct<vk::Device> m_vkDevice;
  /// Basic RAII wrapper for VkSurface
  /// depends on: `m_vkInstance`
  DelayedConstruct<vk::Surface> m_vkSurface;
  /// Basic RAII wrapper for VkSwapchainKHR
  /// depends on: `m_vkSurface`, `m_vkDevice`
  DelayedConstruct<vk::Swapchain> m_vkSwapchain;
  /// Timeline Semaphore based mechanism to enqueue
  /// Vulkan resources to be destroyed or reused once their signaling timeline
  /// value is reached
  DelayedConstruct<vk::DiscardPool> m_vkDiscardPool;
  /// helper class which, every N frames (see constructor), checks if
  /// `m_vkDiscardPool` size is over a certain threshold, and if yes, destroys
  /// discarded resources depends on: `m_vkDiscardPool`
  DelayedConstruct<vk::DiscardPoolMonitor> m_vkDiscardPoolMonitor;
  /// manager for pseudo-Thread Local Storage (map from tid to struct)
  /// to handle `VkCommandPool` and `VkCommandBuffer` caching
  /// depends on: `m_vkDevice`
  DelayedConstruct<vk::CommandPools> m_vkCommandPools;
  /// manager for descriptor pools. Should be created by one thread at a time
  DelayedConstruct<vk::DescriptorPools> m_vkDescriptorPools;
  /// manager for `VkPipeline` and `VkPipelineCache` objects for compute
  /// pipelines and graphics pipelines
  DelayedConstruct<vk::PipelinePool> m_vkPipelines;

  // ----------- Vulkan: Resource Management --------------
  DelayedConstruct<experimental::BufferManager> m_bufferManager;
  DelayedConstruct<experimental::ImageManager> m_imageManager;

  // ------------ Render Thread - Main Thread -------------
  /// A bunch of synchronization primitives to facilitate signaling
  /// between the main(Window/UI) thread and Render Thread
  RenderCoordinator m_renderCoordinator;

  // ------------ Update Thread - Main Thread --------------
  UpdateCoordinator m_updateCoordinator;

  // ------- basic state ------------------------
  /// timeline value for the current frame being rendered
  uint64_t m_timeline = 0;
  /// indicates whether the primary window was initialized at least
  /// once. On Mobile Platforms, this allows the window initialization
  /// code to distinguish between fresh creation or regain of focus
  std::atomic_bool m_windowInit = false;
  /// indicates whether the application should render the current timeline frame
  /// if this is `false`, the `RTonRender` function returns immediately without
  /// calling the `RTdoOnRender` function, and doesn't increment the timeline
  std::atomic_bool m_shouldRender = false;
};

}  // namespace avk