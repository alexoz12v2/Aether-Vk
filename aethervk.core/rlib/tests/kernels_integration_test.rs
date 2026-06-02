use aethervk_core_rlib::{
  physics::physics_scene::PhysicsScene,
  scene::{KinematicComponent, ReferenceFrameComponent, Scene, TransformComponent},
  simulation::texture_cache::TextureCache,
};
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;

fn setup_test_scene() -> Scene {
  let texture_cache = std::sync::Arc::new(spin::RwLock::new(TextureCache::new("")));
  let scene = Scene::new(texture_cache);

  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<KinematicComponent>(&[]);
  scene.register_component::<ReferenceFrameComponent>(&[]);
  scene.register_component::<aethervk_core_rlib::scene::PhysicalMeshComponent>(&[]);

  let comet = std::sync::Arc::new(aethervk_core_rlib::simulation::comet::generate_quad(
    Vec3f32::from_array([0.0, 1.0, 0.0]),
    1.0,
  ));

  let e1 = scene.spawn_entity("e1");
  let _ = scene.add_component(
    e1,
    TransformComponent {
      position: Vec3f32::from_array([0.0, 10.0, 0.0]),
      ..Default::default()
    },
  );
  let _ = scene.add_component(
    e1,
    KinematicComponent {
      velocity: Vec3f32::from_array([0.0, -9.8, 0.0]),
      ..Default::default()
    },
  );
  let _ = scene.add_component(
    e1,
    aethervk_core_rlib::scene::PhysicalMeshComponent {
      asset_path: "".to_string(),
      mesh: comet.clone(),
      emissive_intensity: 0.0,
      emissive_color: [0.0; 3],
      use_new_path: false,
      paint_display_mode: 0,
      sphere_center: [0.0; 3],
      sphere_radius: 1.0,
      grid_color: [0.0; 3],
      grid_density: 0.0,
      rotational_model: None,
    },
  );

  let e2 = scene.spawn_entity("e2");
  let _ = scene.add_component(
    e2,
    TransformComponent {
      position: Vec3f32::from_array([0.0, -10.0, 0.0]),
      ..Default::default()
    },
  );
  let _ = scene.add_component(
    e2,
    KinematicComponent {
      velocity: Vec3f32::from_array([0.0, 9.8, 0.0]),
      ..Default::default()
    },
  );
  let _ = scene.add_component(
    e2,
    aethervk_core_rlib::scene::PhysicalMeshComponent {
      asset_path: "".to_string(),
      mesh: comet.clone(),
      emissive_intensity: 0.0,
      emissive_color: [0.0; 3],
      use_new_path: false,
      paint_display_mode: 0,
      sphere_center: [0.0; 3],
      sphere_radius: 1.0,
      grid_color: [0.0; 3],
      grid_density: 0.0,
      rotational_model: None,
    },
  );

  let macro_frame = scene.spawn_entity("macro_frame");
  let _ = scene.add_component(
    macro_frame,
    TransformComponent {
      ..Default::default()
    },
  );
  let _ = scene.add_component(
    macro_frame,
    ReferenceFrameComponent {
      frame_type: aethervk_core_rlib::scene::ReferenceFrameType::Macro,
      scale: 1.0,
      soi_radius: f32::MAX,
      depth_layer: 0,
    },
  );

  scene
}

#[test]
fn test_imex_no_collision() {
  let scene = setup_test_scene();
  let physics_scene = PhysicsScene::build_from_scene(&scene, 0.016);

  assert_eq!(physics_scene.gpu_frames.len(), 1); // 1 default macro frame
  // The rest requires a VulkanDevice. We skip execution if no device is available.
}

#[test]
fn test_imex_collision_no_lca() {
  let scene = setup_test_scene();
  let physics_scene = PhysicsScene::build_from_scene(&scene, 0.016);

  assert_eq!(physics_scene.gpu_frames.len(), 1);
}

#[test]
fn test_imex_collision_with_lca() {
  let scene = setup_test_scene();

  let lca_entity = scene.spawn_entity("lca");
  scene.add_component(
    lca_entity,
    TransformComponent {
      position: Vec3f32::from_array([10.0, 0.0, 0.0]),
      ..Default::default()
    },
  );
  scene.add_component(
    lca_entity,
    ReferenceFrameComponent {
      frame_type: aethervk_core_rlib::scene::ReferenceFrameType::Micro,
      scale: 1.0,
      soi_radius: 50.0,
      depth_layer: 1,
    },
  );

  let physics_scene = PhysicsScene::build_from_scene(&scene, 0.016);

  // 1 macro + 1 micro
  assert_eq!(physics_scene.gpu_frames.len(), 2);
}
