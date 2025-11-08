#pragma once

// our utils
#include "utils/mixins.h"

// our render
#include "render/vk/device-vk.h"
#include "render/vk/instance-vk.h"
#include "render/vk/surface-vk.h"

struct android_app;

namespace avk {

class AndroidApp : NonMoveable {
 public:
  AndroidApp(android_app* app);
  ~AndroidApp() noexcept;

  inline vk::Instance* vkInstance() { return m_vkInstance.get(); }
  inline vk::Surface* vkSurface() { return m_vkSurface.get(); }
  inline vk::Device* vkDevice() { return m_vkDevice.get(); }

 private:
  DelayedConstruct<vk::Instance> m_vkInstance;
  // on android, there's only one of these, hence we integrate it here
  DelayedConstruct<vk::Surface> m_vkSurface;
  DelayedConstruct<vk::Device> m_vkDevice;
  const android_app* m_app = nullptr;
};

}  // namespace avk