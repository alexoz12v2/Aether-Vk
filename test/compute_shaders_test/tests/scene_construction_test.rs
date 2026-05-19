use aethervk_core_rlib::{
  gpu,
  scene::{PhysicalMeshComponent, TransformComponent},
  simulation_api::SimulationContext,
};
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{Vector3, vec3::Vec3f32, vec4::Quat},
};
use std::sync::Arc;
use test_utils::cycle_get_asset_path_from_exe;

#[test]
fn test_scene_construction() {
  let assets_dir = std::path::PathBuf::from("../../assets");

  // 1. Initialize Simulation Context
  let simulation_context = SimulationContext::startup(gpu::VULKAN_RENDER_BACKEND, None).unwrap();
  let scene_id = simulation_context.create_default_scene().unwrap();

  // 2. Load Mesh
  let model_path = assets_dir.join("Comet.glb");
  let comet = aethervk_core_rlib::simulation::comet::load_comet_from_gltf(
    model_path.to_str().unwrap(),
    false,
    None,
  )
  .expect("Failed to load comet");

  let initial_rotation = if let Some(ref axes) = comet.pa_basis_bf {
    Quat::from_rotation_matrix(axes)
  } else {
    Quat::identity()
  };

  // 3. Populate Scene
  let scene_ctx = simulation_context.get_scene(scene_id).unwrap();
  let mut active_scene = scene_ctx.write();
  let root_entity = active_scene.root_entity;

  // A. Add Rigid Body (PhysicalMeshComponent)
  let mesh_entity = active_scene.scene.spawn_entity("comet_rigid_body");
  active_scene
    .scene
    .add_component(
      mesh_entity,
      TransformComponent {
        position: Vec3f32::from_components(10.0, 0.0, 0.0),
        rotation: initial_rotation,
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();

  active_scene
    .scene
    .add_component(
      mesh_entity,
      PhysicalMeshComponent {
        asset_path: "".to_string(),
        mesh: Arc::from(comet),
        emissive_intensity: 0.0,
        emissive_color: [0.0, 0.0, 0.0],
        use_new_path: false,
        paint_display_mode: 0,
        sphere_center: [0.0, 0.0, 0.0],
        sphere_radius: 1.0,
        grid_color: [0.0, 0.0, 0.0],
        grid_density: 1.0,
      },
    )
    .unwrap();

  active_scene.scene.set_parent(mesh_entity, Some(root_entity));
  active_scene.register_entity(mesh_entity);

  // B. Add Particle System
  let uv_dist =
    aethervk_core_rlib::simulation::utils::generate_gaussian_distribution(64, 0.5, 0.5, 0.5, 0.5);
  let particle_system_entity = active_scene.scene.spawn_entity("particle_emitter");

  active_scene
    .scene
    .add_component(
      particle_system_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 10.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();

  active_scene
    .scene
    .add_component(
      particle_system_entity,
      aethervk_core_rlib::scene::ParticleSystemComponent::new(10_000_000),
    )
    .unwrap();

  // mesh entity will emit particles
  active_scene
    .scene
    .add_component(
      mesh_entity,
      aethervk_core_rlib::scene::ParticleEmitterComponent {
        uv_distribution: uv_dist,
        delta: 100_000,
        max_particles: 100_000,
        velocity_intensity: aethervk_core_rlib::scene::GaussianParams {
          mean: 0.5,
          std_dev: 0.1,
          min: 0.0,
          max: 1.0,
        },
        emission_count: aethervk_core_rlib::scene::GaussianParams {
          mean: 100.0,
          std_dev: 20.0,
          min: 10.0,
          max: 200.0,
        },
        particle_radius: 1.0,
        density: 1000.0,
        lifetime: 5_000_000,
        color: [1.0, 0.5, 0.0, 1.0],
        beta: 0.1,
        use_particle2: false,
      },
    )
    .unwrap();

  active_scene.scene.set_parent(particle_system_entity, Some(root_entity));
  active_scene.register_entity(particle_system_entity);

  println!(
    "Successfully constructed a scene with a PhysicalMeshComponent (RigidBody) and a ParticleSystemComponent!"
  );
}
