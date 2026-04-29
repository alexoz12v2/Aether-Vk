use aethervk_core_rlib::gpu::{FrameCancelGuard, RenderScene, ScopedCommandBuffer, ScopedRenderPass};
use aethervk_core_rlib::{
  gpu::{self, RenderDevice},
  scene::{EntityId, Scene},
  types::GpuResult,
};
use std::sync::{mpsc, Arc};
use test_utils::scene_to_render_scene;
use aethervk_oshal_rlib::math::vector::{Vector, Vector3};

pub struct RenderPacket {
  pub camera_entity: EntityId,
  pub window_size: winit::dpi::PhysicalSize<u32>,
  pub outlines_enabled: bool,
  pub is_command_prompt_open: bool,
  pub console_open_progress: f32,
  pub console_scroll_offset: usize,
  pub command_history: std::collections::VecDeque<String>,
  pub current_command: String,
}

pub fn start_render_thread(
  render_rx: mpsc::Receiver<Option<RenderPacket>>,
  scene_shared: Arc<Scene>,
  render_frontend: gpu::RenderFrontend,
  render_device_handle: gpu::RenderDeviceHandle,
  presentation_engine: gpu::PresentationEngineHandle,
  font_id: (u64, u32),
) -> std::thread::JoinHandle<()> {
  std::thread::spawn(move || {
    while let Ok(Some(packet)) = render_rx.recv() {
      let res = render_frontend.with_device(render_device_handle, |device| {
        scene_to_render_scene(
          &scene_shared,
          device,
          presentation_engine,
          packet.camera_entity,
          packet.outlines_enabled,
        )
        .and_then(|render_scene| {
          render_payload(device, presentation_engine, packet, render_scene, font_id)
        })
      });
      if let Err(e) = res {
        println!("Render error: {:?}", e);
      }
    }
  })
}

fn render_payload(
  device: &dyn RenderDevice,
  presentation_engine: gpu::PresentationEngineHandle,
  payload: RenderPacket,
  render_scene: RenderScene,
  font_id: (u64, u32),
) -> GpuResult<()> {
  device.start_frame()?;
  let acquire_result = device.acquire_next_image(presentation_engine)?;
  if acquire_result.status.needs_resize() {
    device.resize_presentation_engine(
      presentation_engine,
      payload.window_size.width,
      payload.window_size.height,
    )?;
    return Ok(());
  }
  let present_guard = FrameCancelGuard::new(device, presentation_engine, acquire_result);

  // --- Start of safely scoped GPU Operations ---

  let raw_cmd_buffer = device.get_command_buffer()?;
  let cmd_guard = ScopedCommandBuffer::new(device, raw_cmd_buffer, None)?;
  let cmd_buffer = cmd_guard.cmd();
  if let Some(sun_call) = &render_scene.sun_call {
    // TODO move to kernels
    device.update_sun(cmd_buffer, sun_call.entity, (128, 128, 128))?;
  }

  device.begin_render_pass(cmd_buffer, presentation_engine, &acquire_result)?;
  let rp_guard = ScopedRenderPass::new(device, cmd_buffer);

  let extent = device.get_presentation_engine_extent(presentation_engine)?;
  device.set_viewport(cmd_buffer, &gpu::Viewport::from_extent(extent))?;
  device.set_scissor(cmd_buffer, &gpu::Rect2D::from_extent(extent))?;

  device.render_frame(cmd_buffer, &render_scene)?;

  let screen_extent = [
    payload.window_size.width as f32,
    payload.window_size.height as f32,
  ];

  if !render_scene.measurement_calls.is_empty() {
    device.prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer)?;

    let view_proj = render_scene.camera_data.view_proj;
    for m in &render_scene.measurement_calls {
      let p1 = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
        m.p1[0], m.p1[1], m.p1[2],
      );
      let p2 = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
        m.p2[0], m.p2[1], m.p2[2],
      );

      use aethervk_oshal_rlib::math::vector::Vector;
      let mid = p1 + (p2 - p1) * 0.5;

      // Add a slight upward offset in world space
      let offset_mid =
        mid + aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(0.0, 0.0, 5.0);

      if let Some((screen_x, screen_y)) = aethervk_core_rlib::math::from_world_space_to_screen_space(
        offset_mid,
        view_proj,
        (screen_extent[0], screen_extent[1]),
      ) {
        let distance = (p2 - p1).length() as f64;
        let text = crate::logic_thread::format_distance(distance, m.significant_digits);

        // Convert screen coords to NDC for text renderer
        let ndc_x = (screen_x / screen_extent[0]) * 2.0 - 1.0;
        let ndc_y = (screen_y / screen_extent[1]) * 2.0 - 1.0;

        let _ = device.render_text(
          cmd_buffer,
          &text,
          [ndc_x, ndc_y],
          screen_extent,
          font_id,
          m.points,
          [1.0, 1.0, 1.0, 1.0],
        );
      }
    }
  }

  // TODO use Calculate slide-in animation offset
  let slide_y = -1.0 + (payload.console_open_progress * 1.0); // Ranges from -1.0 (hidden above screen) to 0.0 (fully visible)
  let base_y = 0.18 + slide_y;

  if payload.console_open_progress > 0.0 {
    let width = 2.0; // full screen width in NDC
    let height = 1.0; // half screen height in NDC (total is 2.0)
    let box_y = -1.0 - height + (payload.console_open_progress * height);

    device.render_ui_rect(
      cmd_buffer,
      [0.05, 0.1, 0.05, 0.7],
      [-1.0, box_y],
      [width, height],
    )?;

    let mut console_text = String::new();
    let max_lines = 12; // Further reduced to prevent any overlap
    let history_len = payload.command_history.len();
    let scroll = payload
      .console_scroll_offset
      .min(history_len.saturating_sub(max_lines));
    let start_idx = history_len.saturating_sub(max_lines + scroll);
    let end_idx = history_len.saturating_sub(scroll);

    for cmd in payload
      .command_history
      .iter()
      .skip(start_idx)
      .take(end_idx - start_idx)
    {
      console_text.push_str(cmd);
      console_text.push('\n');
    }

    // Position the prompt at the very bottom, and start the text history well above it.
    let prompt_y = box_y + height - 0.08;
    let text_start_y = box_y + 0.05;

    let screen_extent = [
      payload.window_size.width as f32,
      payload.window_size.height as f32,
    ];

    device.prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer)?;

    device.render_text(
      cmd_buffer,
      &console_text,
      [-0.98, text_start_y],
      screen_extent,
      font_id,
      14.0, // Slightly smaller font to fit better
      [0.8, 0.8, 0.8, 1.0],
    )?;

    let mut prompt_text = String::from("> ");
    prompt_text.push_str(&payload.current_command);
    prompt_text.push('_');

    device.render_text(
      cmd_buffer,
      &prompt_text,
      [-0.98, prompt_y],
      screen_extent,
      font_id,
      16.0,
      [1.0, 1.0, 0.2, 1.0],
    )?;
  }

  // Explictly end and submit. Bypasses the Drop trait's automatic closure.
  rp_guard.end()?;
  cmd_guard.submit()?;
  present_guard.defuse();

  // --- End of safely scoped GPU Operations ---

  let present_status = device.present(
    presentation_engine,
    acquire_result.image_index as usize,
    acquire_result.frame_index as usize,
  )?;

  if present_status.needs_resize() {
    device.resize_presentation_engine(
      presentation_engine,
      payload.window_size.width,
      payload.window_size.height,
    )?;
  }

  Ok(())
}
