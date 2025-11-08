#include "render/vk/surface-vk.h"

// TODO vsync and dpi support from native APIs

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
  auto* pfnCreateAndroidSurfaceKHR =
      reinterpret_cast<PFN_vkCreateAndroidSurfaceKHR>(vkGetInstanceProcAddr(
          instance->handle(), "vkCreateAndroidSurfaceKHR"));
  AVK_EXT_CHECK(pfnCreateAndroidSurfaceKHR);
  VK_CHECK(pfnCreateAndroidSurfaceKHR(instance->handle(), &createInfo, nullptr,
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

}  // namespace avk::vk