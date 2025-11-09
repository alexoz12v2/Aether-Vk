#include "avk-android-app.h"

// android stuff
#include <game-activity/native_app_glue/android_native_app_glue.h>

// our stuff
#include "os/avk-log.h"

namespace avk {

AndroidApp::AndroidApp(android_app* app) : m_app(app) {
  m_vkInstance.create();
  LOGI << "[AndroidApp] Vulkan Instance " << std::hex
       << m_vkInstance.get()->handle() << std::dec << " Created" << std::endl;
  AVK_EXT_CHECK(*m_vkInstance.get());
}

AndroidApp::~AndroidApp() noexcept {
  if (m_windowInit) {
    m_vkSwapchain.destroy();
    m_vkDevice.destroy();
    m_vkSurface.destroy();
  }
  m_vkInstance.destroy();
}

void AndroidApp::onWindowInit() {
  vk::SurfaceSpec surfSpec{};
  surfSpec.window = m_app->window;
  m_vkSurface.create(m_vkInstance.get(), surfSpec);
  AVK_EXT_CHECK(*m_vkSurface.get());
  LOGI << "[AndroidApp] Vulkan Surface " << std::hex
       << m_vkSurface.get()->handle() << std::dec << " Created" << std::endl;
  m_vkDevice.create(m_vkInstance.get(), m_vkSurface.get());
  LOGI << "[AndroidApp] Vulkan Device Created" << std::endl;
  m_vkSwapchain.create(m_vkInstance.get(), m_vkSurface.get(), m_vkDevice.get());
  LOGI << "[AndroidApp] Vulkan Swapchain Created" << std::endl;
  m_windowInit = true;
}

}  // namespace avk
