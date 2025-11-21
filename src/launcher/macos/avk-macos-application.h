#pragma once

#include "app/avk-application.h"

namespace avk {

class MacosApplication : public ApplicationBase {
 public:
  MacosApplication();
  ~MacosApplication() noexcept override;

 protected:
  void doOnSaveState() override;
  void doOnRestoreState() override;
  void UTdoOnFixedUpdate() override;
  void UTdoOnUpdate() override;
  void UTdoOnInit() override;
  void RTdoOnDeviceLost() override;
  void RTdoOnDeviceRegained() override;
  void RTdoOnWindowInit() override;
  VkResult RTdoOnRender(const vk::utils::SwapchainData& swapchainData) override;
  void RTdoOnResize() override;
  void RTdoOnSurfaceLost() override;
  void RTdoEarlySurfaceRegained() override;
  void RTdoLateSurfaceRegained() override;
  vk::SurfaceSpec doSurfaceSpec() override;

 private:
};

}  // namespace avk