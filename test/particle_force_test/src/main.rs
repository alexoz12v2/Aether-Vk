use aethervk_core_rlib::gpu::PresentationEngineHandle;
use aethervk_core_rlib::scene::{CameraComponent, TransformComponent, EntityId};
use aethervk_core_rlib::scene::ui::{ScreenSpaceTextComponent, Transform2DComponent, UiComponent};
use aethervk_core_rlib::simulation_api::components_api::{CameraParams, PerspectiveCameraParams};
use aethervk_core_rlib::simulation_api::SimulationContext;
use aethervk_core_rlib::types::{EngineResult, GpuResult};
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::{vec3::Vec3f32, vec4::Quat, Vector, Vector3};
use rand::Rng;
use rayon::prelude::*;
use std::sync::Arc;
use test_utils::cycle_get_asset_path_from_exe;
use test_utils::sim_app::{run_simulation_app, SimulationDelegate};
use winit::window::Window;

struct ForceTestDelegate {
  camera_entity: u64,
  quad_entity: Option<EntityId>,
  particle_sys_entity: Option<EntityId>,
  ui_text_entity: Option<EntityId>,
  font_atlas: Option<Arc<aethervk_core_rlib::scene::text::FontAtlas>>,
  font_hash: u64,
  startup_time: std::time::Instant,
}

impl SimulationDelegate for ForceTestDelegate {
  fn create_scene(&mut self, ctx: &mut SimulationContext) -> EngineResult<u64> {
    ctx.create_empty_scene()
  }

  fn on_setup(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    pe_handle: PresentationEngineHandle,
    _window: &Window,
  ) -> EngineResult<()> {
    let root_entity = ctx.spawn_entity(scene_id, "root").unwrap();
    ctx
      .add_transform_component(
        scene_id,
        root_entity,
        Vec3f32::zero(),
        Quat::identity(),
        Vec3f32::one(),
      )
      .unwrap();

    let scene_ctx = ctx.get_scene(scene_id).unwrap();
    let scene_ctx_write = scene_ctx.write();
    let scene = &scene_ctx_write.scene;

    // Spawn Quad to initialize hierarchy so get_root works!
    let quad_entity = scene.spawn_entity("quad");
    self.quad_entity = Some(quad_entity);
    let root_entity_id = scene_ctx_write.get_entity(root_entity).unwrap();
    scene.set_parent(quad_entity, Some(root_entity_id));
    scene.add_component(quad_entity, TransformComponent {
      position: Vec3f32::from_components(-8.0, -15.0, 0.0),
      rotation: Quat::identity(),
      scale: Vec3f32::one(),
    }).unwrap();

    let quad_mesh = aethervk_core_rlib::simulation::comet::generate_quad(Vec3f32::from_components(1.0, 0.0, 0.0), 5.0);
    let quad_mesh_arc = Arc::new(quad_mesh);
    scene.add_component(quad_entity, aethervk_core_rlib::scene::PhysicalMeshComponent {
      asset_path: "procedural_quad".to_string(),
      mesh: quad_mesh_arc.clone(),
      emissive_intensity: -1.0,
      emissive_color: [0.5, 0.5, 0.5],
      use_new_path: true,
      paint_display_mode: 0,
    }).unwrap();

    drop(scene_ctx_write); // Drop write lock to call add_perspective_camera

    let camera_entity =
      ctx.add_perspective_camera(scene_id, pe_handle, "camera", 45.0, 0.1, 1000.0).unwrap().get();
    ctx.set_parent(scene_id, camera_entity, root_entity).unwrap();
    ctx
      .set_transform_component(
        scene_id,
        camera_entity,
        Vec3f32::from_components(0.0, 0.0, 0.0),
        Quat::identity(),
        Vec3f32::one(),
      )
      .unwrap();
    self.camera_entity = camera_entity;

    // Load font for UI
    let asset_dir = test_utils::cycle_get_asset_path_from_exe(true);
    let font_path = test_utils::get_monospace_font_path_from_asset_path(&asset_dir);
    let font_atlas = aethervk_core_rlib::scene::text::FontAtlas::from_path(font_path.to_str().unwrap(), 32.0).unwrap();
    let font_hash = font_atlas.hash_metadata();
    let font_arc = Arc::new(font_atlas);
    self.font_atlas = Some(font_arc.clone());
    self.font_hash = font_hash;

    let scene_ctx_write2 = scene_ctx.write();
    let scene = &scene_ctx_write2.scene;

    // Setup Force Emitter
    let emitter_entity = scene.spawn_entity("force_emitter");
    scene.add_component(emitter_entity, TransformComponent {
      position: Vec3f32::from_components(-8.0, -15.0, 0.0), // Origin
      rotation: Quat::identity(),
      scale: Vec3f32::one(),
    }).unwrap();

    scene.add_component(emitter_entity, aethervk_core_rlib::scene::ForceEmitterComponent::Planar {
        normal: Vec3f32::from_components(1.0, 0.0, 0.0),
        base_force: 0.005,
        trunc_distance: 10.0,
    }).unwrap();

    // Setup Particles natively on the Quad Entity!
    let config = aethervk_core_rlib::scene::particles::ParticleEmitterComponent {
        uv_distribution: aethervk_core_rlib::math::distribution::Distribution2D::new(&[1.0, 1.0, 1.0, 1.0], 2, 2),
        delta: 16000,
        max_particles: 1_000_000,
        velocity_intensity: aethervk_core_rlib::scene::particles::GaussianParams {
          mean: 5.0,
          std_dev: 2.0,
          min: 0.0,
          max: 10.0,
        },
        emission_count: aethervk_core_rlib::scene::particles::GaussianParams {
          mean: 1600.0,
          std_dev: 200.0,
          min: 0.0,
          max: 2000.0,
        },
        particle_radius: 0.05,
        density: 1.0,
        lifetime: 10_000_000, // 10 seconds
        color: [0.2, 0.6, 1.0, 0.8],
        beta: 0.0,
        use_particle2: true, // Requested Particle2 pipeline
    };
    scene.add_component(quad_entity, config).unwrap();
    
    let particle_sys = aethervk_core_rlib::scene::particles::ParticleSystemComponent::new(1_000_000);
    scene.add_component(quad_entity, particle_sys).unwrap();

    // Add UI Text
    let ui_text_entity = scene.spawn_entity("ui_text");
    self.ui_text_entity = Some(ui_text_entity);
    let mut t2d = Transform2DComponent::default();
    t2d.local_position = [20.0, 20.0];
    t2d.global_depth = 2;
    scene.add_component(ui_text_entity, t2d).unwrap();
    scene.add_component(ui_text_entity, ScreenSpaceTextComponent {
       text: "Particles: 0".to_string(),
       font_atlas: font_arc.clone(),
       font_hash,
       color: [1.0, 1.0, 1.0, 1.0],
       points: 24.0,
       use_new_path: true,
    }).unwrap();

    // Un-pause simulation out of the gate
    let _ = ctx.threads.logic_thread.tx().try_send(aethervk_core_rlib::simulation_api::structs::LogicCommand::SetSceneTimeScale { 
        scene_id,
        scale: aethervk_core_rlib::simulation_api::structs::TimeScale::RealTime,
    });
    let _ = ctx.threads.logic_thread.tx().try_send(aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id });

    Ok(())
  }

  fn on_about_to_wait(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    _delta_time: f32,
  ) {
    let scene_ctx = ctx.get_scene(scene_id).unwrap();
    let scene_ctx_read = scene_ctx.read();
    let scene = &scene_ctx_read.scene;

    let mut count = 0;
    if let Some(quad_id) = self.quad_entity {
        let _ = scene.with_component::<aethervk_core_rlib::scene::particles::ParticleSystemComponent, _, _>(quad_id, |ps| {
            count = ps.particles.read().len();
        });
    }

    if count > 0 {
        let elapsed = std::time::Instant::now().duration_since(self.startup_time).as_millis();
        if elapsed % 1000 < 30 {
            println!("Active Particles: {}", count);
        }
    }

    // Update UI text
    if let Some(ui_id) = self.ui_text_entity {
        let _ = scene.with_component_mut::<ScreenSpaceTextComponent, _, _>(ui_id, |text_comp| {
           text_comp.text = format!("Particles: {}", count);
        });
    }
  }
}

fn main() {
  let delegate = ForceTestDelegate {
    camera_entity: 0,
    quad_entity: None,
    particle_sys_entity: None,
    ui_text_entity: None,
    font_atlas: None,
    font_hash: 0,
    startup_time: std::time::Instant::now(),
  };

  run_simulation_app("Particle Force Test", delegate);
}