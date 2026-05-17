//! test_ui_text module.

use super::*;
use crate::{
  gpu::{
    self, DeviceAdditionalParams, PresentationEngineParams, RenderDeviceHandle, RenderFrontend,
    ScopedCommandBuffer, ScopedRenderPass, VULKAN_RENDER_BACKEND, new_render_frontend,
  },
  types::RuntimeParams,
};
use heapless::index_map::FnvIndexMap;
use std::sync::Arc;

fn setup_assets_dir() {
  if let Ok(mut errors) = crate::gpu_backends::vulkan::utils::VULKAN_ERROR_MESSAGES.lock() {
    errors.clear();
  }

  let mut home_dir = std::env::current_exe().unwrap();
  let mut iter = 0;
  while !home_dir.join("assets").is_dir() && iter < 32 {
    home_dir.pop();
    iter += 1;
  }
  *crate::gpu::ASSET_DIR.write() = Some(home_dir.join("assets").to_str().unwrap().to_string());
}

fn setup_render_frontend_for_tests(
  with_windowless: bool,
) -> (
  Arc<aethervk_oshal_rlib::os::pool::ThreadPool>,
  RenderFrontend,
  RenderDeviceHandle,
  Option<PresentationEngineHandle>,
) {
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }

  use std::sync::mpsc;
  let (tx, rx) = mpsc::channel();

  let th =
    aethervk_oshal_rlib::os::thread::Builder::new().stack_size(8 * 1024 * 1024).spawn(move || {
      let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
      let pool_arc = Arc::new(pool);

      let runtime_params = Box::new(RuntimeParams {
        render_backend_params: FnvIndexMap::new(),
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
              device.init_archetypes(pe)?;
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

#[test]
fn test_render_ui_and_text() {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();
  let [width, height] = render_frontend
    .with_device(render_device_handle, |device| {
      device.get_presentation_engine_extent(presentation_engine)
    })
    .unwrap();

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

      let asset_font_path = format!(
        "{}/fonts/JetBrainsMono-Regular.ttf",
        crate::gpu::ASSET_DIR.read().as_ref().unwrap()
      );
      let atlas = crate::scene::text::FontAtlas::from_path(&asset_font_path, 32.0)
        .expect("Failed to load asset font");

      let font_hash = atlas.hash_metadata();

      {
        let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

        let font_id = device.allocate_rasterized_font_atlas(
          cmd_buffer_handle,
          font_hash,
          alloc::sync::Arc::new(atlas),
        )?;

        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

        device.set_viewport(
          cmd_buffer_handle,
          &gpu::Viewport::from_extent([width, height]),
        )?;
        device.set_scissor(
          cmd_buffer_handle,
          &gpu::Rect2D::from_extent([width, height]),
        )?;

        device.render_ui_rect(
          cmd_buffer_handle,
          [0.2, 0.2, 0.8, 1.0],
          [-0.5, -0.5],
          [1.0, 1.0],
        )?;

        device.prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer_handle)?;

        #[rustfmt::skip]
        let view_proj = [
          2.0 / width as f32, 0.0, 0.0, 0.0,
          0.0, 2.0 / height as f32, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0,
          -1.0, -1.0, 0.0, 1.0,
        ];

        device.render_text(
          cmd_buffer_handle,
          "Test UI Text",
          [50.0, 50.0],
          view_proj,
          (font_hash, font_id),
          24.0,
          [1.0, 1.0, 1.0, 1.0],
        )?;

        scoped_rp.end()?;
        device.record_windowless_download(cmd_buffer_handle, task_id)?;
      }

      device.present(
        presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      )?;

      while !device.is_task_completed(task_id)? {
        std::thread::sleep(std::time::Duration::from_millis(10));
      }
      device.success_task(task_id);

      let mut buffer = vec![0u8; (width * height * 4) as usize];
      device.read_windowless_download(task_id, &mut buffer)?;

      // Assert that not all pixels are black (some UI rendering occurred)
      let sum: u64 = buffer.iter().map(|&b| b as u64).sum();
      assert!(sum > 0, "Rendered UI/Text buffer is completely empty!");

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}
