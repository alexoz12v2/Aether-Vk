use aethervk_core_rlib::{
  gpu::PresentationEngineHandle,
  scene::{PhysicalMeshComponent, SunComponent, TransformComponent},
  simulation_api::SimulationContext,
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{Vector3, vec3::Vec3f32, vec4::Quat},
};
use rfd::FileDialog;
use std::sync::Arc;
use test_utils::{
  cycle_get_asset_path_from_exe,
  sim_app::{SimulationDelegate, run_simulation_app},
};
use winit::window::Window;

struct MeshViewerDelegate {
  camera_ext_entity: u64,
}

impl SimulationDelegate for MeshViewerDelegate {
  fn on_setup(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    pe_handle: PresentationEngineHandle,
    _window: &Window,
  ) -> EngineResult<()> {
    let file_path = FileDialog::new().add_filter("GLTF/GLB", &["glb", "gltf"]).pick_file();
    let file_path = match file_path {
      Some(p) => p,
      None => {
        println!("No file selected, exiting.");
        return Err(EngineError::InvalidOperation("no file selected"));
      }
    };
    let loaded_mesh = aethervk_core_rlib::simulation::comet::load_comet_from_gltf(
      file_path.to_str().unwrap(),
      true,
    )
    .expect("Failed to load mesh");

    // create mesh and move camera -5
    let scene_lock = ctx.get_scene(scene_id).unwrap();
    {
      let scene_read = scene_lock.read();
      let camera_entity = scene_read
        .presentation_engines
        .read()
        .get(&pe_handle)
        .unwrap()
        .camera_entity
        .unwrap();
      self.camera_ext_entity = scene_read
        .entity_map
        .iter()
        .find(|&(_, v)| *v == camera_entity)
        .map(|(k, _)| *k)
        .unwrap();
      ctx.set_transform_component(
        scene_id,
        self.camera_ext_entity,
        Vec3f32::from_components(0.0, 5.0, 0.0),
        Quat::identity(),
        Vec3f32::one(),
      )?;
    }
    {
      let mut scene_write = scene_lock.write();
      let mesh_entity = scene_write.scene.spawn_entity("mesh");
      scene_write.scene.add_component(mesh_entity, TransformComponent::default())?;
      scene_write.scene.add_component(
        mesh_entity,
        PhysicalMeshComponent {
          asset_path: file_path.to_str().unwrap().to_string(),
          mesh: Arc::from(loaded_mesh),
          emissive_intensity: 0.0,
          emissive_color: [0.0, 0.0, 0.0],
          use_new_path: false,
          paint_display_mode: 0,
          sphere_center: [0.0, 0.0, 0.0],
          sphere_radius: 1.0,
          grid_color: [0.0, 0.0, 0.0],
          grid_density: 1.0,
          rotational_model: None,
        },
      )?;
      scene_write.register_entity(mesh_entity);

      let sun_entity = scene_write.scene.spawn_entity("sun");
      scene_write.scene.add_component(
        sun_entity,
        TransformComponent {
          position: Vec3f32::from_components(10.0, 10.0, 10.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )?;
      scene_write.scene.add_component(
        sun_entity,
        SunComponent {
          resolution: (128, 128, 128),
          radius: 1.0,
        },
      )?;
      scene_write.register_entity(sun_entity);
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
      "There is not camera entity with id {} in scene {}",
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
}

fn main() {
  let _assets_dir = cycle_get_asset_path_from_exe(true);
  let delegate = MeshViewerDelegate {
    camera_ext_entity: 0,
  };
  run_simulation_app("AetherVk Mesh Viewer", delegate);
}
