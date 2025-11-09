#pragma once

// our utils
#include "utils/mixins.h"

// our render
#include "render/vk/device-vk.h"
#include "render/vk/instance-vk.h"
#include "render/vk/surface-vk.h"
#include "render/vk/swapchain-vk.h"

struct android_app;

namespace avk {

class AndroidApp : public NonMoveable {
 public:
  AndroidApp(android_app* app);
  ~AndroidApp() noexcept;

  void onWindowInit();

  inline vk::Instance* vkInstance() { return m_vkInstance.get(); }
  inline vk::Surface* vkSurface() { return m_vkSurface.get(); }
  inline vk::Device* vkDevice() { return m_vkDevice.get(); }
  inline vk::Swapchain* vkSwapchain() { return m_vkSwapchain.get(); }
  inline bool windowInitialized() { return m_windowInit; }

 private:
  DelayedConstruct<vk::Instance> m_vkInstance;
  DelayedConstruct<vk::Device> m_vkDevice;
  // on android, there's only one of these, hence we integrate it here
  DelayedConstruct<vk::Surface> m_vkSurface;
  DelayedConstruct<vk::Swapchain> m_vkSwapchain;
  const android_app* m_app = nullptr;

  // useless state
  bool m_windowInit = false;
};

}  // namespace avk