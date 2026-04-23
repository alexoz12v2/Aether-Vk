use aethervk_core_rlib::{
  gpu::{self, RenderDevice},
  scene::{
    CameraComponent, CursorComponent, EntityId, PhysicalMeshComponent, RenderableDataRef, Scene,
    TransformComponent,
  },
  types::GpuResult,
};
use aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use std::sync::{Arc, mpsc};
use aethervk_core_rlib::gpu::{ScopedCommandBuffer, ScopedRenderPass};

pub struct RenderItem {
  pub entity_id: EntityId,
  pub model_matrix: Mat4x4f32,
}

pub struct RenderPacket {
  pub render_items: Vec<RenderItem>,
  pub camera_transform: TransformComponent,
  pub camera_component: CameraComponent,
  pub window_size: winit::dpi::PhysicalSize<u32>,
  pub outlines_enabled: bool,
  pub is_command_prompt_open: bool,
  pub console_open_progress: f32,
  pub console_scroll_offset: usize,
  pub command_history: std::collections::VecDeque<String>,
  pub current_command: String,
}

#[repr(C)]
struct RenderPayloadData<'a> {
  packet: &'a mut RenderPacket,
  presentation_engine: gpu::PresentationEngineHandle,
  scene: &'a Scene,
  cursor_entity: EntityId,
  sun_entity: EntityId,
  assets_dir: &'a std::path::PathBuf,
}

pub fn start_render_thread(
  render_rx: mpsc::Receiver<Option<RenderPacket>>,
  scene_shared: Arc<Scene>,
  render_frontend: gpu::RenderFrontend,
  render_device_handle: gpu::RenderDeviceHandle,
  presentation_engine: gpu::PresentationEngineHandle,
  cursor_entity: EntityId,
  sun_entity: EntityId,
  assets_dir: std::path::PathBuf,
) -> std::thread::JoinHandle<()> {
  std::thread::spawn(move || {
    while let Ok(Some(mut packet)) = render_rx.recv() {
      let scene_guard = scene_shared.as_ref();
      let mut c_payload = RenderPayloadData {
        packet: &mut packet,
        presentation_engine,
        scene: &scene_guard,
        cursor_entity,
        sun_entity,
        assets_dir: &assets_dir,
      };

      let res = render_frontend.with_device(render_device_handle, |device| {
        render_payload_ffi(device, &mut c_payload)
      });
      if let Err(e) = res {
        println!("Render error: {:?}", e);
      }
    }
  })
}

fn render_payload_ffi(device: &dyn RenderDevice, payload: &mut RenderPayloadData) -> GpuResult<()> {
  device.start_frame()?;
  let acquire_result = device.acquire_next_image(payload.presentation_engine)?;
  if acquire_result.status.needs_resize() {
    device.resize_presentation_engine(
      payload.presentation_engine,
      payload.packet.window_size.width,
      payload.packet.window_size.height,
    )?;
    return Ok(());
  }

  let mut render_scene = gpu::frame::RenderScene::new((
    payload.packet.camera_transform,
    payload.packet.camera_component,
  ));

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::SunComponent, _>(|entity, comp| {
      if let Some(transform) = payload.scene.global_transform(entity) {
        render_scene.sun = Some((entity, *comp, transform));
      }
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::SkyComponent, _>(|entity, comp| {
      render_scene.sky = Some((entity, *comp));
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::GridComponent, _>(|entity, comp| {
      render_scene.grid = Some((entity, *comp));
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::CursorComponent, _>(|entity, comp| {
      if let Some(transform) = payload.scene.global_transform(entity) {
        render_scene
          .add_renderable(
            device,
            entity,
            transform.to_mat4(),
            RenderableDataRef::Cursor(comp),
            payload.presentation_engine,
            "Cursor",
            false,
            [1.0, 1.0, 1.0, 1.0],
          )
          .unwrap();
      }
    });

  for item in &payload.packet.render_items {
    payload.scene.with_component(
      item.entity_id,
      |mesh: &PhysicalMeshComponent| -> GpuResult<()> {
        let mut draw_outline = payload.packet.outlines_enabled;
        let mut outline_color = [0.2, 0.5, 1.0, 1.0]; // Default blueish

        let is_selected = payload
          .scene
          .with_component(
            item.entity_id,
            |_c: &aethervk_core_rlib::scene::SelectedComponent| {},
          )
          .is_some();
        let is_following = payload
          .scene
          .with_component(
            item.entity_id,
            |_c: &aethervk_core_rlib::scene::FollowingComponent| {},
          )
          .is_some();

        if is_following {
          draw_outline = true;
          outline_color = [1.0, 0.0, 0.0, 1.0]; // Red
        } else if is_selected {
          draw_outline = true;
          outline_color = [1.0, 1.0, 1.0, 1.0]; // White
        }

        render_scene
          .add_renderable(
            device,
            item.entity_id,
            item.model_matrix,
            RenderableDataRef::PhysicalMesh(mesh),
            payload.presentation_engine,
            "Comet",
            draw_outline,
            outline_color,
          )
          .unwrap();
        Ok(())
      },
    );
  }

  payload.scene.with_component(
    payload.cursor_entity,
    |cursor: &CursorComponent| -> GpuResult<()> {
      let t = payload
        .scene
        .global_transform(payload.cursor_entity)
        .unwrap();
      render_scene
        .add_renderable(
          device,
          payload.cursor_entity,
          t.to_mat4(),
          RenderableDataRef::Cursor(cursor),
          payload.presentation_engine,
          "Cursor",
          false,
          [1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
      Ok(())
    },
  );

  let mut sun_opt = None;
  payload.scene.with_component(
    payload.sun_entity,
    |sun_comp: &aethervk_core_rlib::scene::SunComponent| {
      sun_opt = Some(*sun_comp);
    },
  );
  let sun_comp = sun_opt.unwrap();
  let sun_transform = payload.scene.global_transform(payload.sun_entity).unwrap();

  let mut sky_opt = None;
  payload
    .scene
    .query1::<aethervk_core_rlib::scene::SkyComponent, _>(|id, comp| {
      sky_opt = Some((id, *comp));
    });

  let mut grid_opt = None;
  payload
    .scene
    .query1::<aethervk_core_rlib::scene::GridComponent, _>(|id, comp| {
      grid_opt = Some((id, *comp));
    });

  render_scene.sun = Some((payload.sun_entity, sun_comp, sun_transform.into()));
  if let Some((id, comp)) = sky_opt {
    render_scene.sky = Some((id, comp));
  }
  if let Some((id, comp)) = grid_opt {
    render_scene.grid = Some((id, comp));
  }

  // --- Start of safely scoped GPU Operations ---

  let raw_cmd_buffer = device.get_command_buffer()?;
  let cmd_guard = ScopedCommandBuffer::new(device, raw_cmd_buffer)?;
  let cmd_buffer = cmd_guard.cmd();

  device.update_sun(cmd_buffer, payload.sun_entity, &sun_comp)?;
  device.begin_render_pass(cmd_buffer, payload.presentation_engine, &acquire_result)?;

  // Protect the active render pass. If any `?` happens below, `rp_guard` ends the pass automatically.
  let rp_guard = ScopedRenderPass::new(device, cmd_buffer);

  let extent = device.get_presentation_engine_extent(payload.presentation_engine)?;
  let root_viewport = gpu::Viewport {
    x: 0.0,
    y: 0.0,
    width: extent[0] as f32,
    height: extent[1] as f32,
    min_depth: 0.0,
    max_depth: 1.0,
  };
  device.set_viewport(cmd_buffer, &root_viewport)?;
  device.set_scissor(
    cmd_buffer,
    &gpu::Rect2D {
      offset: [0, 0],
      extent,
    },
  )?;

  device.render_frame(cmd_buffer, &render_scene)?;

  let mut planets = Vec::new();
  if let Some(sun_transform) = payload.scene.global_transform(payload.sun_entity) {
    planets.push((sun_transform.position, 0.06, [1.0, 1.0, 0.2, 1.0]));
  }
  use aethervk_oshal_rlib::math::matrix::Matrix;
  use aethervk_oshal_rlib::math::vector::{Vector4};
  for item in &payload.packet.render_items {
    let col = item.model_matrix.column(3).unwrap();
    let pos =
      aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([col.x(), col.y(), col.z()]);
    planets.push((pos, 0.02, [0.8, 0.8, 0.8, 1.0]));
  }

  let mut player_pos =
    aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]);
  if let Some(cursor_transform) = payload.scene.global_transform(payload.cursor_entity) {
    player_pos = cursor_transform.position;
  }

  let max_dist = 60000.0; // Pluto is around 59000 units away
  let _ = device.render_minimap(cmd_buffer, player_pos, max_dist, &planets);

  // Compute view matrix to print
  let view =
      <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::from_columns(
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(1.0, 0.0, 0.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, -1.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 1.0, 0.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, 0.0, 1.0),
      ) * <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::from_quat_custom_frame(
        payload.packet.camera_transform.rotation.conjugate(),
      ) * <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::translation(payload.packet.camera_transform.position * -1.0);

  let view_proj = payload.packet.camera_component.projection * view;

  let mut all_bvh_nodes = Vec::new();
  for item in &payload.packet.render_items {
    let mut dbg_states = None;
    payload.scene.with_component(
      item.entity_id,
      |dbg: &aethervk_core_rlib::scene::BvhDebugComponent| {
        dbg_states = Some(dbg.node_render_states.clone());
      },
    );

    if let Some(states) = dbg_states {
      payload.scene.with_component(
        item.entity_id,
        |mesh: &aethervk_core_rlib::scene::PhysicalMeshComponent| {
          if let Some(bvh) = &mesh.mesh.bvh {
            for (i, &render) in states.iter().enumerate() {
              if render && i < bvh.nodes.len() {
                all_bvh_nodes.push((bvh.nodes[i].bound.clone(), item.model_matrix));
              }
            }
          }
        },
      );
    }
  }
  if !all_bvh_nodes.is_empty() {
    let _ = device.render_bvh(
      cmd_buffer,
      &all_bvh_nodes,
      view_proj.into(),
      payload.presentation_engine,
    );
  }

  // TODO use Calculate slide-in animation offset
  let slide_y = -1.0 + (payload.packet.console_open_progress * 1.0); // Ranges from -1.0 (hidden above screen) to 0.0 (fully visible)
  let base_y = 0.18 + slide_y;

  if payload.packet.console_open_progress > 0.0 {
    let font_path_buf = payload.assets_dir.join("fonts/JetBrainsMono-Bold.ttf");
    let font_path = font_path_buf.to_str().unwrap();

    let width = 2.0; // full screen width in NDC
    let height = 1.0; // half screen height in NDC (total is 2.0)
    let box_y = -1.0 - height + (payload.packet.console_open_progress * height);

    let _ = device.render_ui_rect(
      cmd_buffer,
      [0.05, 0.1, 0.05, 0.7],
      [-1.0, box_y],
      [width, height],
      payload.presentation_engine,
    );

    let mut console_text = String::new();
    let max_lines = 12; // Further reduced to prevent any overlap
    let history_len = payload.packet.command_history.len();
    let scroll = payload
      .packet
      .console_scroll_offset
      .min(history_len.saturating_sub(max_lines));
    let start_idx = history_len.saturating_sub(max_lines + scroll);
    let end_idx = history_len.saturating_sub(scroll);

    for cmd in payload
      .packet
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

    let _ = device.render_text(
      cmd_buffer,
      &console_text,
      font_path,
      14.0, // Slightly smaller font to fit better
      [0.8, 0.8, 0.8, 1.0],
      [-0.98, text_start_y],
      payload.presentation_engine,
    );

    let mut prompt_text = String::from("> ");
    prompt_text.push_str(&payload.packet.current_command);
    prompt_text.push('_');

    let _ = device.render_text(
      cmd_buffer,
      &prompt_text,
      font_path,
      16.0,
      [1.0, 1.0, 0.2, 1.0],
      [-0.98, prompt_y],
      payload.presentation_engine,
    );
  }

  // Explictly end and submit. Bypasses the Drop trait's automatic closure.
  rp_guard.end()?;
  cmd_guard.submit()?;

  // --- End of safely scoped GPU Operations ---

  let present_status = device.present(
    payload.presentation_engine,
    acquire_result.image_index as usize,
    acquire_result.frame_index as usize,
  )?;

  if present_status.needs_resize() {
    device.resize_presentation_engine(
      payload.presentation_engine,
      payload.packet.window_size.width,
      payload.packet.window_size.height,
    )?;
  }

  Ok(())
}
