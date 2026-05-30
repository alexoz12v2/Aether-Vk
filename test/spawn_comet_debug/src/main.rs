use aethervk_core_rlib::{
  gpu::PresentationEngineHandle,
  simulation_api::SimulationContext,
  types::EngineResult,
};
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{Vector3, vec3::Vec3f32, vec4::Quat},
};
use std::sync::Arc;
use test_utils::{
  cycle_get_asset_path_from_exe,
  sim_app::{SimulationDelegate, run_simulation_app},
};
use winit::window::Window;

struct SpawnCometDelegate {
  camera_ext_entity: u64,
}

impl SimulationDelegate for SpawnCometDelegate {
  fn create_scene(&mut self, ctx: &mut SimulationContext) -> EngineResult<u64> {
    // Use create_default_scene(true) to get the full Avalonia-like scene:
    // sun (emissive mesh), camera with look_at, sky, grid, cursor —
    // all on the macro frame (root), exactly as the Avalonia app does.
    ctx.create_default_scene(true)
  }

  ///    root (Macro, depth_layer=0)
  ///    ├── sun        — emissive UV sphere mesh at origin, scale = sun_radius/0.6
  ///    ├── camera     — perspective, look_at from first octant toward origin
  ///    │   └── sky    — sky cubemap (child of camera)
  ///    ├── grid       — infinite Z=0 grid
  ///    ├── cursor     — 3D cursor at origin
  ///    └── comet_microframe (Micro, depth_layer=1, pos=(0.01, 0, 0) AU, SOI=0.005 AU)
  ///          └── comet  — PhysicalMesh, radius=1km, mesh_scale=2.497
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

    // ── Load comet mesh ─────────────────────────────────────────────────────
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

    // ── Spawn comet micro-frame at 0.01 AU (+x), SOI = 0.04 AU ──────────
    // spawn_comet_internal creates:
    //   root
    //     └── comet_microframe  (ReferenceFrameComponent, Micro, depth_layer=1)
    //           └── comet        (PhysicalMeshComponent, scaled by radius_km/bounding_sphere)
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

    // ── Assign the scene's fallback camera to the presentation engine ────
    // create_default_scene already created a camera named "camera" on the
    // macro frame. Find it and assign it to the PE.
    let scene_lock = ctx.get_scene(scene_id).unwrap();
    {
      let scene_read = scene_lock.read();

      // Find the camera entity: iterate the entity_map to find the one
      // that has a CameraComponent.
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

      // Store the external ID so we can reference it later for controls
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
    }

    // Start the simulation clock
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
}

fn main() {
  let _assets_dir = cycle_get_asset_path_from_exe(true);
  let delegate = SpawnCometDelegate {
    camera_ext_entity: 0,
  };
  run_simulation_app("AetherVk Comet Spawn Test", delegate);
}