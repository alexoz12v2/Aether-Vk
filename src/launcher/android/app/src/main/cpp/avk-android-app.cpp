#include "avk-android-app.h"

// android stuff
#include <game-activity/native_app_glue/android_native_app_glue.h>

// our stuff
#include "os/avk-log.h"

namespace avk {

AndroidApp::AndroidApp(android_app* app) : m_app(app) {
  m_vkInstance.create();
  LOGI << "[AndroidApp] Vulkan Instance Created" << std::endl;
  AVK_EXT_CHECK(*m_vkInstance.get());
  vk::SurfaceSpec surfSpec{};
  surfSpec.window = m_app->window;
  m_vkSurface.create(m_vkInstance.get(), surfSpec);
  AVK_EXT_CHECK(*m_vkSurface.get());
  LOGI << "[AndroidApp] Vulkan Surface Created" << std::endl;
  m_vkDevice.create(m_vkInstance.get(), m_vkSurface.get());
  LOGI << "[AndroidApp] Vulkan Device Created" << std::endl;
}

AndroidApp::~AndroidApp() noexcept {
  m_vkDevice.destroy();
  m_vkSurface.destroy();
  m_vkInstance.destroy();
}

}  // namespace avk
