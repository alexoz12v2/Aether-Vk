#include "avk-windows-application.h"

// system
#include <Windows.h>

namespace avk {
WindowsApplication::WindowsApplication(HWND primaryWindow)
    : m_primaryHwnd(primaryWindow) {}

void WindowsApplication::doOnSaveState() {}

void WindowsApplication::doOnRestoreState() {}

void WindowsApplication::RTdoOnDeviceLost() {
  showErrorScreenAndExit("Desktop should never lose device");
}

void WindowsApplication::RTdoOnDeviceRegained() {}

void WindowsApplication::RTdoOnWindowInit() {}

VkResult WindowsApplication::RTdoOnRender(
    [[maybe_unused]] const vk::utils::SwapchainData& swapchainData) {
  return VK_ERROR_DEVICE_LOST;
}

void WindowsApplication::RTdoOnResize() {}

 void WindowsApplication::RTdoOnSurfaceLost() {
  showErrorScreenAndExit(
      "VkSurfaceKHR lost: How did the"
      " primary HWND get destroyed?");
}

void WindowsApplication::RTdoEarlySurfaceRegained() {
  // never called
}

void WindowsApplication::RTdoLateSurfaceRegained() {
  // never called
}

vk::SurfaceSpec WindowsApplication::doSurfaceSpec() {
  vk::SurfaceSpec spec{};
  spec.instance = GetModuleHandleW(nullptr);
  spec.window = m_primaryHwnd;
  return spec;
}

}  // namespace avk