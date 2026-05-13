use aethervk_core_rlib::gpu::PresentationEngineHandle;
use aethervk_core_rlib::scene::CameraProjection;
use aethervk_core_rlib::scene::camera::SceneCameraExt;
use aethervk_core_rlib::scene::camera::QuatToEulerAngles;
use aethervk_core_rlib::scene::ui::UiBuilder;
use aethervk_core_rlib::scene::{
  CursorComponent, EntityId, HiddenComponent, PhysicalMeshComponent, TransformComponent,
};
use aethervk_core_rlib::simulation_api::SimulationContext;
use aethervk_core_rlib::simulation_api::structs::{SimulationTaskResult, TaskStatusCode};
use aethervk_core_rlib::types::EngineResult;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::vec4::Quat;
use std::sync::Arc;
use test_utils::sim_app::{SimulationDelegate, run_simulation_app};
use winit::event::{ElementState, MouseButton};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

#[derive(PartialEq, Clone, Copy)]
enum Mode {
  Normal,
  Paint,
}

#[derive(PartialEq, Clone, Copy)]
enum Submode {
  Color,
  Distribution,
}

struct PaintDelegate {
  mode: Mode,
  submode: Submode,
  brush_radius: f32,
  color_val: [f32; 3],
  dist_val: f32,

  mesh_entity: Option<EntityId>,
  camera_entity: Option<EntityId>,
  cursor_entity: Option<EntityId>,
  view_center: Option<EntityId>,
  ui_text_entity: Option<EntityId>,

  mesh_id: Option<aethervk_core_rlib::gpu::RenderableInstanceId>,

  is_left_mouse_down: bool,
  is_shift_down: bool,
  mouse_x: f64,
  mouse_y: f64,
  window_width: f32,
  window_height: f32,

  input_buffer: String,
  input_active: bool,

  font_atlas: Option<Arc<aethervk_core_rlib::scene::text::FontAtlas>>,
  font_hash: u64,
  
  emissive_initialized: bool,
}

impl PaintDelegate {
  fn new() -> Self {
    let font_data = include_bytes!("../../../assets/fonts/JetBrainsMono-Regular.ttf");
    let atlas = aethervk_core_rlib::scene::text::FontAtlas::from_slice(font_data, 48.0)
      .expect("Failed to load font");
    let hash = atlas.hash_metadata();

    Self {
      mode: Mode::Normal,
      submode: Submode::Color,
      brush_radius: 0.01,
      color_val: [1.0, 0.0, 0.0],
      dist_val: 1.0,

      mesh_entity: None,
      camera_entity: None,
      cursor_entity: None,
      view_center: None,
      ui_text_entity: None,

      mesh_id: None,

      is_left_mouse_down: false,
      is_shift_down: false,
      mouse_x: 0.0,
      mouse_y: 0.0,
      window_width: 800.0,
      window_height: 600.0,

      input_buffer: String::new(),
      input_active: false,

      font_atlas: Some(Arc::new(atlas)),
      font_hash: hash,
      emissive_initialized: false,
    }
  }

  fn update_ui_text(&self, ctx: &mut SimulationContext, scene_id: u64) {
    if let Some(ui_e) = self.ui_text_entity {
      let scene_ctx = ctx.get_scene(scene_id).unwrap();
      let active_scene = scene_ctx.write();

      let mode_str = match self.mode {
        Mode::Normal => "Normal",
        Mode::Paint => "Paint",
      };
      let submode_str = match self.submode {
        Submode::Color => "Color",
        Submode::Distribution => "Distribution",
      };

      let mut text = format!(
        "Mode: {} | Submode: {} | Rad: {:.3}\nColor: {:?} | Dist: {:.2}",
        mode_str, submode_str, self.brush_radius, self.color_val, self.dist_val
      );

      if self.input_active {
         text.push_str("\n--- INPUT FORM ---\n");
         if self.submode == Submode::Color {
             text.push_str("Enter RGB values (e.g. 1.0 0.0 0.0): ");
         } else {
             text.push_str("Enter Distribution value (0.0 to 1.0): ");
         }
         text.push_str(&self.input_buffer);
      }

      active_scene
        .scene
        .with_component_mut::<aethervk_core_rlib::scene::ui::ScreenSpaceTextComponent, _, _>(
          ui_e,
          |text_comp| {
            text_comp.text = text;
          },
        );
        
      if let Some(mesh_id) = self.mesh_entity {
          active_scene.scene.with_component_mut::<aethervk_core_rlib::scene::PhysicalMeshComponent, _, _>(mesh_id, |c| {
              c.paint_display_mode = match self.mode {
                  Mode::Normal => 0, // NONE
                  Mode::Paint => match self.submode {
                      Submode::Color => 1, // COLOR
                      Submode::Distribution => 2, // DISTRIBUTION
                  }
              }
          });
      }
    }
  }

  fn apply_paint(&mut self, ctx: &mut SimulationContext, scene_id: u64) {
    if self.mode != Mode::Paint {
      return;
    }
    if let (Some(cam), Some(mesh_id)) = (self.camera_entity, self.mesh_id) {
      let scene_arc = ctx.get_scene(scene_id).unwrap();
      let scene_read = scene_arc.read();
      let camera_entity =
        scene_read.entity_map.iter().find(|&(_, v)| *v == cam).map(|(k, _)| *k).unwrap();
      drop(scene_read);
      let ndc_x = (self.mouse_x as f32 / self.window_width) * 2.0 - 1.0;
      let ndc_y = (self.mouse_y as f32 / self.window_height) * 2.0 - 1.0;
      println!("DEBUG: apply_paint at mouse ({}, {}) -> NDC ({}, {})", self.mouse_x, self.mouse_y, ndc_x, ndc_y);

      // TODO version with EntityId
      if let Ok(task_id) = ctx.raycast_ndc(scene_id, camera_entity, ndc_x, ndc_y) {
        while ctx.task_manager.read().get_status(task_id.get()) == aethervk_core_rlib::simulation_api::structs::TaskStatusCode::Pending {
          core::hint::spin_loop();
        }
        let task_result = ctx.task_manager.write().take_result(task_id.get());
        let hit_res = match task_result {
          Some(SimulationTaskResult::Raycast(hit)) => {
            println!("DEBUG: Raycast returned: hit={}", hit.is_some());
            hit
          },
          _ => {
            println!("DEBUG: Raycast failed or unexpected task result");
            None
          }
        };
        if let Some(hit) = hit_res {
          if let Some(cursor_e) = self.cursor_entity {
            let scene_arc = ctx.get_scene(scene_id).unwrap();
            let mut active_scene = scene_arc.write();
            active_scene.scene.with_component_mut(cursor_e, |t: &mut TransformComponent| {
                t.position = hit.p;
            });
            let _ = active_scene.scene.remove_component::<aethervk_core_rlib::scene::HiddenComponent>(cursor_e);
          }
          if let Some(ptr) = ctx.get_emissive_paint_image_mapped_ptr(mesh_id) {
            let cx = (hit.uv[0] * 1024.0) as i32;
            let cy = (hit.uv[1] * 1024.0) as i32;
            let radius_px = (self.brush_radius * 1024.0) as i32;

            let min_x = (cx - radius_px).max(0);
            let max_x = (cx + radius_px).min(1023);
            let min_y = (cy - radius_px).max(0);
            let max_y = (cy + radius_px).min(1023);

            for y in min_y..=max_y {
              for x in min_x..=max_x {
                let dx = (x - cx) as f32;
                let dy = (y - cy) as f32;
                let dist = (dx * dx + dy * dy).sqrt();
                let r_px = radius_px as f32;

                if dist <= r_px {
                  let t = dist / r_px;
                  let falloff = 1.0 - t * t * (3.0 - 2.0 * t); // Smoothstep hermite

                  let offset = ((y * 1024 + x) * 4) as isize;
                  unsafe {
                    if self.submode == Submode::Color {
                      let r = ptr.offset(offset);
                      let g = ptr.offset(offset + 1);
                      let b = ptr.offset(offset + 2);

                      let curr_r = *r as f32 / 255.0;
                      let curr_g = *g as f32 / 255.0;
                      let curr_b = *b as f32 / 255.0;

                      let target_r = if self.is_shift_down { 128.0 / 255.0 } else { self.color_val[0] };
                      let target_g = if self.is_shift_down { 128.0 / 255.0 } else { self.color_val[1] };
                      let target_b = if self.is_shift_down { 128.0 / 255.0 } else { self.color_val[2] };

                      let new_r = curr_r + (target_r - curr_r) * falloff;
                      let new_g = curr_g + (target_g - curr_g) * falloff;
                      let new_b = curr_b + (target_b - curr_b) * falloff;

                      *r = (new_r.clamp(0.0, 1.0) * 255.0) as u8;
                      *g = (new_g.clamp(0.0, 1.0) * 255.0) as u8;
                      *b = (new_b.clamp(0.0, 1.0) * 255.0) as u8;
                    } else {
                      let a = ptr.offset(offset + 3);
                      let curr_a = *a as f32 / 255.0;
                      
                      let sign = if self.is_shift_down { -1.0 } else { 1.0 };
                      // Make distribution mode additive (like an airbrush)
                      let new_a = curr_a + sign * self.dist_val * falloff * 0.05;
                      *a = (new_a.clamp(0.0, 1.0) * 255.0) as u8;
                    }
                  }
                }
              }
            }
          }
        } else {
          if let Some(cursor_e) = self.cursor_entity {
             let _ = ctx.get_scene(scene_id).unwrap().write().scene.add_component(cursor_e, aethervk_core_rlib::scene::HiddenComponent {});
          }
        }
      }
    }
  }
}

impl SimulationDelegate for PaintDelegate {
  fn create_scene(&mut self, ctx: &mut SimulationContext) -> EngineResult<u64> {
    ctx.create_empty_scene()
  }

  fn on_setup(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    _pe_handle: PresentationEngineHandle,
    window: &Window,
  ) -> EngineResult<()> {
    self.window_width = window.inner_size().width as f32;
    self.window_height = window.inner_size().height as f32;

    let assets_dir = test_utils::cycle_get_asset_path_from_exe(true);
    let model_path = assets_dir.join("Comet.glb");
    let comet = aethervk_core_rlib::simulation::comet::load_comet_from_gltf(
      model_path.to_str().unwrap(),
      false,
    )
    .expect("Failed to load comet");

    let comet_arc = Arc::from(comet);

    let scene_ctx = ctx.get_scene(scene_id).unwrap();
    let mut active_scene = scene_ctx.write();
    let root_entity = active_scene.root_entity;

    let mesh_e = active_scene.scene.spawn_entity("comet");
    active_scene.scene.set_parent(mesh_e, Some(root_entity));

    active_scene
      .scene
      .add_component(
        mesh_e,
        TransformComponent {
          position: Vec3f32::from_array([0.0, 0.0, 0.0]),
          rotation: Quat::identity(),
          scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
        },
      )
      .unwrap();

    active_scene
      .scene
      .add_component(
        mesh_e,
        PhysicalMeshComponent {
          asset_path: "".into(),
          mesh: comet_arc.clone(),
          emissive_intensity: 1.0,
          emissive_color: [1.0, 1.0, 1.0],
          use_new_path: true,
          paint_display_mode: 0,
        },
      )
      .unwrap();

    active_scene.register_entity(mesh_e);
    self.mesh_entity = Some(mesh_e);
    let mesh_instance_id = aethervk_core_rlib::gpu::RenderableInstanceId::from_physical_mesh(comet_arc.id);
    self.mesh_id = Some(mesh_instance_id);

    let cursor_e = active_scene.scene.spawn_entity("cursor");
    active_scene.scene.set_parent(cursor_e, Some(root_entity));
    active_scene.scene.add_component(cursor_e, TransformComponent {
      position: Vec3f32::from_array([0.0, 0.0, 0.0]),
      rotation: Quat::identity(),
      scale: Vec3f32::from_array([0.02, 0.02, 0.02]),
    }).unwrap();
    // Add HiddenComponent so it doesn't show up until we raycast
    active_scene.scene.add_component(cursor_e, aethervk_core_rlib::scene::HiddenComponent {}).unwrap();
    active_scene.scene.add_component(cursor_e, CursorComponent {}).unwrap();
    self.cursor_entity = Some(cursor_e);

    let view_center = active_scene.scene.spawn_entity("view_center");
    active_scene.scene.set_parent(view_center, Some(root_entity));
    active_scene.scene.add_component(view_center, TransformComponent {
      position: Vec3f32::from_array([0.0, 0.0, 0.0]),
      rotation: Quat::identity(),
      scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
    }).unwrap();
    self.view_center = Some(view_center);

    let cam_e = active_scene.scene.spawn_entity("camera");
    active_scene.scene.set_parent(cam_e, Some(root_entity));
    
    let pitch = -core::f32::consts::PI / 4.0;
    let yaw = 0.0;
    let q = Quat::from_pitch_and_yaw_radians(pitch, yaw);
    let offset = q.rotate_vector(Vec3f32::from_array([0.0, 10.0, 0.0]));
    
    active_scene
      .scene
      .add_component(
        cam_e,
        TransformComponent {
          position: offset,
          rotation: q,
          scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
        },
      )
      .unwrap();
    active_scene
      .scene
      .add_component(
        cam_e,
        aethervk_core_rlib::scene::CameraComponent {
          projection: CameraProjection::Perspective {
            fov: 45.0,
            aspect_ratio: self.window_width / self.window_height,
            near: 0.1,
            far: 100.0,
          },
        },
      )
      .unwrap();
    active_scene.scene.orbit_camera(cam_e, view_center, 0.0, 0.0).unwrap();
    active_scene.register_entity(cam_e);
    self.camera_entity = Some(cam_e);
    
    let grid_e = active_scene.scene.spawn_entity("grid");
    active_scene.scene.set_parent(grid_e, Some(root_entity));
    active_scene.scene.add_component(grid_e, TransformComponent::default()).unwrap();
    active_scene.scene.add_component(grid_e, aethervk_core_rlib::scene::GridComponent {}).unwrap();

    active_scene.presentation_engines.write().get_mut(&_pe_handle).unwrap().camera_entity = Some(cam_e);

    let ui_builder = UiBuilder::new(&active_scene.scene);
    let ui_text = ui_builder.build_text(
      "status_text",
      "Mode: Normal | Submode: Color | Rad: 0.010\nColor: [1.0, 0.0, 0.0] | Dist: 1.00",
      [10.0, 30.0],
      self.font_atlas.clone().unwrap(),
      self.font_hash,
      [1.0, 1.0, 1.0, 1.0], // White text
      24.0,
    );
    active_scene.scene.set_parent(ui_text, Some(root_entity));
    active_scene.register_entity(ui_text);
    self.ui_text_entity = Some(ui_text);

    drop(active_scene); // Release lock before sending message

    let _ = ctx.threads.logic_thread.tx().try_send(
      aethervk_core_rlib::simulation_api::structs::LogicCommand::SetSceneTimeScale {
        scene_id,
        scale: aethervk_core_rlib::simulation_api::structs::TimeScale::RealTime,
      },
    );
    let _ = ctx.threads.logic_thread.tx().try_send(
      aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id },
    );

    Ok(())
  }

  fn on_about_to_wait(&mut self, ctx: &mut SimulationContext, scene_id: u64, _delta_time: f32) {
    if !self.emissive_initialized {
        if let Some(mesh_instance_id) = self.mesh_id {
            if let Some(ptr) = ctx.get_emissive_paint_image_mapped_ptr(mesh_instance_id) {
                unsafe {
                    // 1024x1024 RGBA is 4MB. Fill with 128 (grey)
                    core::ptr::write_bytes(ptr, 128, 1024 * 1024 * 4);
                }
                self.emissive_initialized = true;
            }
        }
    }

    if self.is_left_mouse_down && self.mode == Mode::Paint {
      self.apply_paint(ctx, scene_id);
    }
  }

  fn on_keyboard_input(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    event: &winit::event::KeyEvent,
    _modifiers: winit::keyboard::ModifiersState,
  ) {
    self.is_shift_down = _modifiers.shift_key();
    
    if event.state == ElementState::Pressed {
      if self.input_active {
         match event.physical_key {
            PhysicalKey::Code(KeyCode::Enter) | PhysicalKey::Code(KeyCode::NumpadEnter) => {
               self.input_active = false;
               if self.submode == Submode::Color {
                   let parts: Vec<&str> = self.input_buffer.trim().split_whitespace().collect();
                   if parts.len() >= 3 {
                       if let (Ok(r), Ok(g), Ok(b)) = (parts[0].parse::<f32>(), parts[1].parse::<f32>(), parts[2].parse::<f32>()) {
                           self.color_val = [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)];
                       }
                   }
               } else {
                   if let Ok(d) = self.input_buffer.trim().parse::<f32>() {
                       self.dist_val = d.clamp(0.0, 1.0);
                   }
               }
               self.update_ui_text(ctx, scene_id);
            }
            PhysicalKey::Code(KeyCode::Backspace) => {
               self.input_buffer.pop();
               self.update_ui_text(ctx, scene_id);
            }
            PhysicalKey::Code(KeyCode::Space) => {
               self.input_buffer.push(' ');
               self.update_ui_text(ctx, scene_id);
            }
            PhysicalKey::Code(KeyCode::Period) => {
               self.input_buffer.push('.');
               self.update_ui_text(ctx, scene_id);
            }
            PhysicalKey::Code(c) => {
               let char_opt = match c {
                   KeyCode::Digit0 => Some('0'),
                   KeyCode::Digit1 => Some('1'),
                   KeyCode::Digit2 => Some('2'),
                   KeyCode::Digit3 => Some('3'),
                   KeyCode::Digit4 => Some('4'),
                   KeyCode::Digit5 => Some('5'),
                   KeyCode::Digit6 => Some('6'),
                   KeyCode::Digit7 => Some('7'),
                   KeyCode::Digit8 => Some('8'),
                   KeyCode::Digit9 => Some('9'),
                   KeyCode::KeyA => Some('A'),
                   KeyCode::KeyB => Some('B'),
                   KeyCode::KeyC => Some('C'),
                   KeyCode::KeyD => Some('D'),
                   KeyCode::KeyE => Some('E'),
                   KeyCode::KeyF => Some('F'),
                   KeyCode::KeyG => Some('G'),
                   KeyCode::KeyH => Some('H'),
                   KeyCode::KeyI => Some('I'),
                   KeyCode::KeyJ => Some('J'),
                   KeyCode::KeyK => Some('K'),
                   KeyCode::KeyL => Some('L'),
                   KeyCode::KeyM => Some('M'),
                   KeyCode::KeyN => Some('N'),
                   KeyCode::KeyO => Some('O'),
                   KeyCode::KeyP => Some('P'),
                   KeyCode::KeyQ => Some('Q'),
                   KeyCode::KeyR => Some('R'),
                   KeyCode::KeyS => Some('S'),
                   KeyCode::KeyT => Some('T'),
                   KeyCode::KeyU => Some('U'),
                   KeyCode::KeyV => Some('V'),
                   KeyCode::KeyW => Some('W'),
                   KeyCode::KeyX => Some('X'),
                   KeyCode::KeyY => Some('Y'),
                   KeyCode::KeyZ => Some('Z'),
                   _ => None,
               };
               if let Some(ch) = char_opt {
                   self.input_buffer.push(ch);
                   self.update_ui_text(ctx, scene_id);
               }
            }
            _ => {}
         }
         return;
      }
      
      println!("DEBUG: on_keyboard_input key pressed: {:?}", event.physical_key);
      match event.physical_key {
        PhysicalKey::Code(KeyCode::Tab) => {
          self.mode = match self.mode {
            Mode::Normal => Mode::Paint,
            Mode::Paint => Mode::Normal,
          };
          self.update_ui_text(ctx, scene_id);
        }
        PhysicalKey::Code(KeyCode::KeyM) => {
          if self.mode == Mode::Paint {
            self.submode = match self.submode {
              Submode::Color => Submode::Distribution,
              Submode::Distribution => Submode::Color,
            };
            self.update_ui_text(ctx, scene_id);
          }
        }
        PhysicalKey::Code(KeyCode::KeyC) => {
          if self.mode == Mode::Paint {
            if self.submode == Submode::Color {
              if self.color_val[0] > 0.5 && self.color_val[1] < 0.5 {
                self.color_val = [0.0, 1.0, 0.0];
              } else if self.color_val[1] > 0.5 && self.color_val[2] < 0.5 {
                self.color_val = [0.0, 0.0, 1.0];
              } else {
                self.color_val = [1.0, 0.0, 0.0];
              }
            } else {
              self.dist_val = if self.dist_val >= 1.0 {
                0.0
              } else {
                self.dist_val + 0.25
              };
            }
            self.update_ui_text(ctx, scene_id);
          }
        }
        PhysicalKey::Code(KeyCode::Digit0) => {
          if let (Some(cam_e), Some(center_e)) = (self.camera_entity, self.view_center) {
            let scene_ctx = ctx.get_scene(scene_id).unwrap();
            let active_scene = scene_ctx.write();
            
            let cur_pos = active_scene.scene.with_component(center_e, |t: &aethervk_core_rlib::scene::TransformComponent| t.position).unwrap_or(Vec3f32::from_array([0.0, 0.0, 0.0]));
            
            // Set camera 10 units away along Y axis (since forward is -Y)
            // and apply a 45 degree pitch so it looks down at the object.
            let pitch = -core::f32::consts::PI / 4.0;
            let yaw = 0.0;
            let q = Quat::from_pitch_and_yaw_radians(pitch, yaw);
            let offset = q.rotate_vector(Vec3f32::from_array([0.0, 10.0, 0.0]));
            
            let _ = active_scene.scene.with_component_mut(cam_e, |t: &mut aethervk_core_rlib::scene::TransformComponent| {
               t.position = cur_pos + offset;
               t.rotation = q;
            });
          }
        }
        PhysicalKey::Code(KeyCode::KeyG) => {
          if self.mode == Mode::Paint {
             self.input_active = true;
             self.input_buffer.clear();
             self.update_ui_text(ctx, scene_id);
          }
        }
        _ => {}
      }
    }
  }

  fn on_mouse_input(
    &mut self,
    _ctx: &mut SimulationContext,
    _scene_id: u64,
    button: MouseButton,
    state: ElementState,
    _mouse_pos: (f64, f64),
  ) {
    if button == MouseButton::Left {
      self.is_left_mouse_down = state == ElementState::Pressed;
    }
  }

  fn on_cursor_moved(
    &mut self,
    _ctx: &mut SimulationContext,
    _scene_id: u64,
    pos: winit::dpi::PhysicalPosition<f64>,
  ) {
    self.mouse_x = pos.x;
    self.mouse_y = pos.y;
  }
  fn on_mouse_motion(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    delta: (f64, f64),
    middle_mouse_down: bool,
    shift_down: bool,
    ctrl_down: bool,
  ) {
    self.is_shift_down = shift_down;
    if self.mode == Mode::Normal && middle_mouse_down {
      if let (Some(cam_e), Some(center_e)) = (self.camera_entity, self.view_center) {
        let scene_ctx = ctx.get_scene(scene_id).unwrap();
        let active_scene = scene_ctx.read();
        if shift_down {
          // Pan: move camera local and center
          let _ = active_scene.scene.pan_camera_and_cursor(
            cam_e,
            center_e,
            delta.0 as f32 * 0.01,
            delta.1 as f32 * 0.01,
          );
        } else if ctrl_down {
          // Zoom: move camera local forward/backward (-Y is forward, so delta.1 > 0 means push forward, we apply negative delta.1 to Y? Wait, if they scroll up, delta.1 is usually positive, which means zoom IN -> forward -> -Y. So we use delta.1 * -0.05 on the Y axis)
          let _ = active_scene.scene.translate_camera_local(
            cam_e,
            aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([
              0.0,
              delta.1 as f32 * -0.05,
              0.0,
            ]),
          );
        } else {
          // Orbit
          let _ = active_scene.scene.orbit_camera(
            cam_e,
            center_e,
            delta.0 as f32 * 0.01,
            delta.1 as f32 * 0.01,
          );
        }
      }
    }
  }

  fn on_mouse_wheel(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    delta: winit::event::MouseScrollDelta,
  ) {
    if self.mode == Mode::Paint {
      let scroll_amount = match delta {
        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
        winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y / 50.0) as f32, // roughly scale pixels to lines
      };

      // Increase or decrease brush radius logarithmically
      self.brush_radius *= 1.0 + (scroll_amount * 0.1);
      
      // Clamp the radius to sane bounds (e.g. 0.001 to 0.5)
      self.brush_radius = self.brush_radius.clamp(0.001, 0.5);
      
      self.update_ui_text(ctx, scene_id);
    }
  }

  fn on_resize(&mut self, _ctx: &mut SimulationContext, _scene_id: u64, width: u32, height: u32) {
    self.window_width = width as f32;
    self.window_height = height as f32;
  }
}

fn main() {
  run_simulation_app("Paint Test", PaintDelegate::new());
}
