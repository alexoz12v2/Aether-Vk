use aethervk_core_rlib::{
  gpu::PresentationEngineHandle,
  scene::camera::QuatToEulerAngles,
  simulation_api::SimulationContext,
  types::EngineResult,
};
use aethervk_oshal_rlib::math::{
  matrix::SquareMatrix,
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, vec4::Quat, Vector, Vector3, Vector4},
};
use std::sync::Arc;
use test_utils::{
  cycle_get_asset_path_from_exe,
  sim_app::{SimulationDelegate, run_simulation_app},
};
use winit::window::Window;

struct SpawnCometDelegate {
  camera_ext_entity: u64,
  was_micro: Option<bool>,
}

impl SimulationDelegate for SpawnCometDelegate {
  fn create_scene(&mut self, ctx: &mut SimulationContext) -> EngineResult<u64> {
    ctx.create_default_scene(true)
  }

  fn on_setup(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    pe_handle: PresentationEngineHandle,
    _window: &Window,
  ) -> EngineResult<()> {
    let assets_dir = cycle_get_asset_path_from_exe(true);
    let comet_path = assets_dir.join("Comet.glb");
    aethervk_core_rlib::gpu::ASSET_DIR
      .write()
      .replace(assets_dir.to_string_lossy().to_string());

    let loaded_mesh = aethervk_core_rlib::simulation::comet::load_comet_from_gltf(
      comet_path.to_str().unwrap(),
      false,
      None,
    )
    .expect("Failed to load comet mesh");

    let model_id = {
      let mut scenes = ctx.scenes.write();
      scenes.import_model_from_mesh("comet_model".to_string(), loaded_mesh)
    };

    let (_lca_ext, _comet_ext) = ctx.spawn_comet_internal(
      scene_id,
      model_id,
      "comet",
      Vec3f32::from_components(0.01, 0.0, 0.0), // 0.01 AU along +x
      Quat::identity(),
      1.0,    // radius_km = 1 km
      1000.0, // mass_kg
      0,      // physics_type = static
    )?;

    let scene_lock = ctx.get_scene(scene_id).unwrap();
    {
      let scene_read = scene_lock.read();

      let camera_int = {
        let mut found = None;
        for (_ext_id, &int_id) in scene_read.entity_map.iter() {
          if scene_read
            .scene
            .has_component::<aethervk_core_rlib::scene::CameraComponent>(int_id)
            .into()
          {
            found = Some(int_id);
            break;
          }
        }
        found.expect("No camera found in scene")
      };

      for (&ext_id, &int_id) in scene_read.entity_map.iter() {
        if int_id == camera_int {
          self.camera_ext_entity = ext_id;
          break;
        }
      }

      scene_read
        .presentation_engines
        .write()
        .get_mut(&pe_handle)
        .unwrap()
        .camera_entity = Some(camera_int);

      let cam_pos = Vec3f32::from_components(0.01001, 0.0, 0.0);
      let rot = <Quat as aethervk_oshal_rlib::math::quaternion::Quaternion>::from_vector_and_scalar(
        Vec3f32::from_components(0.0, 0.0, -std::f32::consts::FRAC_1_SQRT_2),
        std::f32::consts::FRAC_1_SQRT_2,
      );
      let _ = scene_read.scene.with_component_mut(camera_int, |t: &mut aethervk_core_rlib::scene::TransformComponent| {
        t.position = cam_pos;
        t.rotation = rot;
      });
      let _ = scene_read.scene.with_component_mut(camera_int, |c: &mut aethervk_core_rlib::scene::CameraComponent| {
        c.focus_distance = 0.00001;
        // Fix for invisible sun: Avalonia default near plane is 0.1 AU, but we spawn at 0.01 AU!
        if let aethervk_core_rlib::scene::CameraProjection::Perspective { ref mut near, ref mut far, .. } = c.projection {
          *near = 0.00001; // 0.00001 AU = 1500 km
          *far = 10000.0;
        }
      });
      
      let mut cursor_ent = None;
      if let Some((id, _)) = scene_read.scene.query1_first_res::<aethervk_core_rlib::scene::CursorComponent, _, _>(|id, _| Some(id)) {
        cursor_ent = Some(id);
      }
      if let Some(id) = cursor_ent {
        let _ = scene_read.scene.with_component_mut(id, |t: &mut aethervk_core_rlib::scene::TransformComponent| {
          t.position = Vec3f32::from_components(0.01, 0.0, 0.0);
        });
      }
    }

    {
      let scene_read = scene_lock.read();
      scene_read.time_state.write().is_playing = true;
    }

    Ok(())
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
    let scene = ctx.get_scene(scene_id).unwrap();
    let camera_entity = scene.read().get_entity(self.camera_ext_entity).expect(&format!(
      "No camera entity with id {} in scene {}",
      self.camera_ext_entity, scene_id
    ));

    let logic_command = test_utils::command::process_mouse_motion_camera_commands(
      delta,
      middle_mouse_down,
      shift_down,
      ctrl_down,
      camera_entity,
      Arc::clone(&scene),
    );

    if let Some(logic_command) = logic_command {
      let _ = ctx.threads.logic_thread.tx().try_send(logic_command);
    }
  }

  fn on_mouse_wheel(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    delta: winit::event::MouseScrollDelta,
  ) {
    let amount = match delta {
      winit::event::MouseScrollDelta::LineDelta(_, y) => y * 0.5,
      winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y * 0.01) as f32,
    };

    let scene = ctx.get_scene(scene_id).unwrap();
    let camera_entity = scene.read().get_entity(self.camera_ext_entity).unwrap();

    let _ = ctx.threads.logic_thread.tx().try_send(
      aethervk_core_rlib::simulation_api::structs::LogicCommand::ZoomCamera(
        aethervk_core_rlib::simulation_api::structs::ZoomCamera {
          camera_entity,
          scene,
          amount,
        },
      ),
    );
  }

  fn on_keyboard_input(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    event: &winit::event::KeyEvent,
    _modifiers: winit::keyboard::ModifiersState,
  ) {
    if event.state != winit::event::ElementState::Pressed {
      return;
    }

    let (target_pos, offset) = match event.logical_key.as_ref() {
      winit::keyboard::Key::Character("f") | winit::keyboard::Key::Character("F") => {
        (Some(Vec3f32::from_components(0.01, 0.0, 0.0)), Some(0.00005)) // 7,500 km away from comet
      }
      winit::keyboard::Key::Character("0") => {
        (Some(Vec3f32::from_components(0.0, 0.0, 0.0)), Some(0.02)) // 0.02 AU away from sun
      }
      winit::keyboard::Key::Character("m") | winit::keyboard::Key::Character("M") => {
        let scene = ctx.get_scene(scene_id).unwrap();
        let scene_read = scene.read();
        if let Some(camera_ent) = scene_read.get_entity(self.camera_ext_entity) {
          let mut curr = camera_ent;
          print!("[DEBUG] Camera {:?} parent hierarchy: ", camera_ent);
          while let Some(parent) = scene_read.scene.get_parent(curr) {
             print!("{:?} -> ", parent);
             curr = parent;
          }
          println!("(root)");
        }
        (None, None)
      }
      _ => (None, None),
    };

    if let Some(pos) = target_pos {
      let scene = ctx.get_scene(scene_id).unwrap();
      let scene_read = scene.read();
      
      let mut cursor_ent = None;
      if let Some((id, _)) = scene_read.scene.query1_first_res::<aethervk_core_rlib::scene::CursorComponent, _, _>(|id, _| Some(id)) {
        cursor_ent = Some(id);
      }
      if let Some(id) = cursor_ent {
        let _ = scene_read.scene.with_component_mut(id, |t: &mut aethervk_core_rlib::scene::TransformComponent| {
          t.position = pos;
        });
      }
      
      if let Some(camera_int) = scene_read.get_entity(self.camera_ext_entity) {
        // Reparent to root so that our local position is treated as macro scale (AU).
        // The logic thread will automatically reparent it back to the micro frame
        // if the new global position happens to fall inside its SOI.
        let root_entity = scene_read.root_entity;
        scene_read.scene.set_parent(camera_int, Some(root_entity));

        if let Some(off) = offset {
          let _ = scene_read.scene.with_component_mut(camera_int, |t: &mut aethervk_core_rlib::scene::TransformComponent| {
            t.position = Vec3f32::from_components(pos.x() + off, pos.y(), pos.z());
            // Look towards -X, Up is +Z
            t.rotation = <aethervk_oshal_rlib::math::vector::vec4::Quat as aethervk_oshal_rlib::math::quaternion::Quaternion>::from_vector_and_scalar(
              Vec3f32::from_components(0.0, 0.0, -std::f32::consts::FRAC_1_SQRT_2),
              std::f32::consts::FRAC_1_SQRT_2,
            );
          });
        }
        let _ = scene_read.scene.with_component_mut(camera_int, |c: &mut aethervk_core_rlib::scene::CameraComponent| {
          c.focus_distance = offset.unwrap_or(0.1);
        });
      }
    }
  }

  fn on_about_to_wait(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    _delta_time: f32,
  ) {
    let scene = ctx.get_scene(scene_id).unwrap();
    let scene_read = scene.read();
    if let Some(camera_entity) = scene_read.get_entity(self.camera_ext_entity) {
      if let Some(t) = scene_read.scene.with_component(camera_entity, |t: &aethervk_core_rlib::scene::TransformComponent| t.clone()) {
        let dist = (t.position - Vec3f32::from_components(0.01, 0.0, 0.0)).length();
        let is_micro = dist < 0.005;

        if let Some(was_micro) = self.was_micro {
          if was_micro != is_micro {
            let color = "\x1b[1;31m"; // Bold Red
            let reset = "\x1b[0m";
            if is_micro {
              println!("{}*** CAMERA ENTERED MICRO FRAME (dist to comet: {:.6} AU) ***{}", color, dist, reset);
            } else {
              println!("{}*** CAMERA EXITED MICRO FRAME (dist to comet: {:.6} AU) ***{}", color, dist, reset);
            }
          }
        }
        self.was_micro = Some(is_micro);

        use core::sync::atomic::{AtomicU64, Ordering};
        static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);
        let frame = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        if frame % 120 == 0 {
          let (pitch, yaw) = t.rotation.to_pitch_yaw();
          println!("[CAMERA] Frame {} | Pos: ({:.6}, {:.6}, {:.6}) | Rot Quat: (w={:.4}, x={:.4}, y={:.4}, z={:.4}) | Euler: (pitch={:.2}°, yaw={:.2}°)",
                   frame,
                   t.position.x(), t.position.y(), t.position.z(),
                   t.rotation.0.w(), t.rotation.0.x(), t.rotation.0.y(), t.rotation.0.z(),
                   pitch.to_degrees(), yaw.to_degrees());
        }
      }
    }
  }
}

fn main() {
  let _assets_dir = cycle_get_asset_path_from_exe(true);
  let delegate = SpawnCometDelegate {
    camera_ext_entity: 0,
    was_micro: None,
  };
  run_simulation_app("AetherVk Comet Spawn Test", delegate);
}
