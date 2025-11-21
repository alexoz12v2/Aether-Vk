#include "render/vk/surface-vk.h"

// TODO vsync and dpi support from native APIs
#ifdef VK_USE_PLATFORM_WIN32_KHR
#  include <Windows.h>
#  include <dwmapi.h>
#endif
#ifdef VK_USE_PLATFORM_ANDROID_KHR
#  include <android/native_window.h>
#endif
#ifdef VK_USE_PLATFORM_METAL_EXT
#include "avk-metal-helpers.h"
#endif

namespace avk::vk {

Surface::Surface(Instance* instance, SurfaceSpec const& spec) AVK_NO_CFI
    : m_deps({instance, spec}) {
  // create surface with specific method
#ifdef VK_USE_PLATFORM_WIN32_KHR
  VkWin32SurfaceCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
  createInfo.hinstance = spec.instance;
  createInfo.hwnd = spec.window;

  VK_CHECK(vkCreateWin32SurfaceKHR(instance->handle(), &createInfo, nullptr,
                                   &m_surface));
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
  VkAndroidSurfaceCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
  createInfo.window = spec.window;
  if (!spec.window) { // TODO make it an assert
    showErrorScreenAndExit("Invalid ANativeWindow");
  }
  VK_CHECK(vkCreateAndroidSurfaceKHR(instance->handle(), &createInfo, nullptr,
                                     &m_surface));
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
  VkWaylandSurfaceCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
  createInfo.display = spec.display;
  createInfo.surface = spec.surface;

  VK_CHECK(vkCreateWaylandSurfaceKHR(instance->handle(), &createInfo, nullptr,
                                     &m_surface));
#elif defined(VK_USE_PLATFORM_METAL_EXT)
  VkMetalSurfaceCreateInfoEXT createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
  createInfo.pLayer = spec.layer;

  VK_CHECK(vkCreateMetalSurfaceEXT(instance->handle(), &createInfo, nullptr,
                                   &m_surface));
#else
#  error "Add support for this WSI platform"
#endif
}

Surface::~Surface() noexcept AVK_NO_CFI {
  if (*this) {
    vkDestroySurfaceKHR(m_deps.instance->handle(), m_surface, nullptr);
  }
}

// TODO: Instead of querying for extent at swapchain recreation, we could
// store it proactively at message/event callback
VkExtent2D Surface::internalExtent() const {
  VkExtent2D extent{};
#ifdef VK_USE_PLATFORM_WIN32_KHR
  RECT extendedFrameBounds{};
  // if window styles and procedure are properly setup, these 2 values
  // (dwm and client) should always be equal (assuming WM_NCCALCSIZE cancels
  // non-client area)
  if (!SUCCEEDED(DwmGetWindowAttribute(
          m_deps.internal.window, DWMWA_EXTENDED_FRAME_BOUNDS,
          &extendedFrameBounds, sizeof(extendedFrameBounds)))) {
    GetClientRect(m_deps.internal.window, &extendedFrameBounds);
  }
  extent.width = extendedFrameBounds.right - extendedFrameBounds.left;
  extent.height = extendedFrameBounds.bottom - extendedFrameBounds.top;
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
  LOGI << "[Surface::internalExtent] computing ANativeWindow extent"
       << std::endl;
  int32_t aWidth = ANativeWindow_getWidth(m_deps.internal.window);
  int32_t aHeight = ANativeWindow_getHeight(m_deps.internal.window);
  AVK_EXT_CHECK(aWidth > 0 && aHeight > 0);
  extent.width = static_cast<uint32_t>(aWidth);
  extent.height = static_cast<uint32_t>(aHeight);
  LOGI << "[Surface::internalExtent] computed ANativeWindow extent: " << aWidth
       << 'x' << aHeight << std::endl;
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
#  error "TODO"
#elif defined(VK_USE_PLATFORM_METAL_EXT)
  extent = avkGetDrawableExtent(m_deps.internal.layer);
#else
#endif
  return extent;
}

}  // namespace avk::vk