#pragma once

#include "render/vk/instance-vk.h"
#include "utils/mixins.h"

struct ANativeWindow;

namespace avk::vk {

struct SurfaceSpec {
#ifdef VK_USE_PLATFORM_WIN32_KHR
  HWND window;
  HINSTANCE instance;
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
  ANativeWindow* window;
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
  struct wl_display* display;
  struct wl_surface* surface;
#elif defined(VK_USE_PLATFORM_METAL_EXT)
  const CAMetalLayer* layer;
#else
#  error "Add support for this WSI platform"
#endif
};

/// Helper struct to pack necessary data for Windows System Integration for
/// Our supported platforms. It is assumed that the handles contained within it
/// are valid for the lifetime of this struct
class Surface : public NonMoveable {
 public:
  Surface(Instance* instance, SurfaceSpec const& spec);
  ~Surface() noexcept;

  inline VkSurfaceKHR handle() const { return m_surface; }
  inline operator bool() const { return m_surface != VK_NULL_HANDLE; }

  VkExtent2D internalExtent() const;

#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
  inline struct wl_display* display() const { return m_internal.display; }
#endif

 private:
  // dependencies which must outlive the object
  struct Deps {
    Instance* instance;
    SurfaceSpec internal;
  } m_deps;

  VkSurfaceKHR m_surface = VK_NULL_HANDLE;
};

}  // namespace avk::vk