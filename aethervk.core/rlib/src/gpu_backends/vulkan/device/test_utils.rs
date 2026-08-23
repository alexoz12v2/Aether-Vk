#[cfg(test)]
use crate::{
  gpu::{
    self, DeviceAdditionalParams, PresentationEngineParams, RenderDeviceHandle, RenderFrontend,
    ScopedCommandBuffer, ScopedRenderPass, VULKAN_RENDER_BACKEND,
  },
  gpu_backends::new_render_frontend,
  types::RuntimeParams,
};
#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
pub fn setup_assets_dir() {
  if let Ok(mut errors) = crate::gpu_backends::vulkan::utils::VULKAN_ERROR_MESSAGES.lock() {
    errors.clear();
  }
  crate::gpu::set_asset_dir_for_tests();
}

#[cfg(test)]
pub fn setup_render_frontend_for_tests(
  with_windowless: bool,
) -> (
  Arc<aethervk_oshal_rlib::os::pool::ThreadPool>,
  RenderFrontend,
  RenderDeviceHandle,
  Option<crate::gpu::PresentationEngineHandle>,
) {
  fn panic_on_validation_error(msg: &str) {
    println!("Vulkan validation error occurred during testing: {}", msg);
  }

  use std::sync::mpsc;
  let (tx, rx) = mpsc::channel();

  let th = aethervk_oshal_rlib::os::thread::Builder::new()
    .stack_size(8 * 1024 * 1024)
    .spawn(move || {
      let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
      let pool_arc = Arc::new(pool);

      let runtime_params = Box::new(RuntimeParams {
        render_backend_params: heapless::index_map::FnvIndexMap::new(),
        validation_error_callback: Some(panic_on_validation_error as fn(&str)),
      });

      let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

      let additional_params = DeviceAdditionalParams::new();
      let render_device_handle =
        render_frontend.write().init_device(0, &additional_params).unwrap();

      render_frontend
        .with_device(render_device_handle, |device| {
          device.wire_callbacks(Arc::clone(&pool_arc))
        })
        .unwrap();

      let presentation_engine = if with_windowless {
        let width = 256;
        let height = 256;

        let params = PresentationEngineParams::windowless(width, height);
        Some(
          render_frontend
            .with_device(render_device_handle, |device| {
              let pe = device.create_presentation_engine(&params)?;
              device.generate_sky()?;
              crate::types::GpuResult::Ok(pe)
            })
            .unwrap(),
        )
      } else {
        None
      };

      tx.send((
        pool_arc,
        render_frontend,
        render_device_handle,
        presentation_engine,
      ))
      .expect("Failed to send setup data from thread");
    });

  let _ = th.unwrap().join();
  rx.recv().expect("Failed to receive setup data")
}
