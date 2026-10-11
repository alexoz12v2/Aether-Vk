//! Sky image overlay (HUD) frame hooks: a user telescope image ([`crate::scene::sky_image`])
//! registered on the sky, drawn above the scene and below the UI (composite subpass 2, between
//! `draw_composite` and the UI batch).
//!
//! - [`prepare_sky_image`]: render thread, before the render pass (uploads on the frame's command
//!   buffer when the scene's image changed).
//! - [`draw_sky_image`]: [`crate::gpu::frame::render_frame`], subpass 2.
//!
//! The draw uses the camera's real projection and its rotation-only view (`proj_f64 · view_f64`,
//! the matrices every layer shares; not the sky's fixed-90° `sky_view_proj` with its parallax
//! lag), the mesh at infinity (`w = 0`): stars stay put whatever the camera's position. Perspective
//! cameras only (an orthographic camera has no directions at infinity).
use crate::{
  gpu::{
    CommandBufferHandle, PresentationEngineHandle, RenderDevice,
    frame::{CameraProjectionParams, CameraRenderData, RenderScene},
  },
  types::GpuResult,
};

/// `proj · view` of the camera for directions at infinity (the view is rotation only in RTE),
/// column major; `None` for an orthographic camera.
pub fn sky_image_view_proj(camera: &CameraRenderData) -> Option<[f32; 16]> {
  match camera.projection_params {
    CameraProjectionParams::Perspective { .. } => {
      Some((camera.proj_f64 * camera.view_f64).to_mat4_f32().into())
    }
    CameraProjectionParams::Orthographic { .. } => None,
  }
}

/// Render thread, before the render pass: syncs presentation engine `pe` with the sky image of
/// the scene whose store key is `scene_key` (`scene::sky_image::scene_key(&render_frame.scene)`).
/// Errors are logged, the frame goes on without the overlay.
pub fn prepare_sky_image(
  device: &crate::gpu_backends::vulkan::device::Device,
  cmd: ash::vk::CommandBuffer,
  pe: PresentationEngineHandle,
  scene_key: usize,
) {
  let entry = crate::scene::sky_image::store_get(scene_key);
  if let Err(e) = device.sky_overlay_prepare(cmd, pe, scene_key, entry.as_ref()) {
    aethervk_oshal_rlib::log!("[SkyOverlay] prepare failed: {:?}", e);
  }
}

/// Composite subpass: draws the presentation engine's sky image, if any, visible and the camera
/// is a perspective one.
pub fn draw_sky_image(
  device: &dyn RenderDevice,
  cmd_buffer: CommandBufferHandle,
  handle: PresentationEngineHandle,
  render_scene: &RenderScene,
) -> GpuResult<()> {
  let Some(vk_device) =
    device.as_any().downcast_ref::<crate::gpu_backends::vulkan::device::Device>()
  else {
    return Ok(());
  };
  let Some(view_proj) = sky_image_view_proj(&render_scene.camera_data) else {
    return Ok(());
  };
  vk_device.cmd_draw_sky_overlay(cmd_buffer, handle, view_proj).map(|_| ())
}
