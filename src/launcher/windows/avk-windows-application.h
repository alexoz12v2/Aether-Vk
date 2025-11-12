#pragma once

#include "app/avk-application.h"

// os specific
#include <Windef.h>

namespace avk {

class WindowsApplication : public ApplicationBase {
 public:
  explicit WindowsApplication(HWND primaryWindow);

 protected:
  void doOnSaveState() override;
  void doOnRestoreState() override;
  void RTdoOnDeviceRegained() override;
  void RTdoOnWindowInit() override;
  VkResult RTdoOnRender(const vk::utils::SwapchainData& swapchainData) override;
  void RTdoOnResize() override;

  vk::SurfaceSpec doSurfaceSpec() override;
  [[noreturn]] void RTdoOnDeviceLost() override;
  [[noreturn]] void RTdoOnSurfaceLost() override;
  void RTdoEarlySurfaceRegained() override;
  void RTdoLateSurfaceRegained() override;

 private:
  /// windows related handles. This class doesn't own it, as the UI/Main
  /// thread is responsible for handling its message pump
  HWND m_primaryHwnd;
};

}  // namespace avk
