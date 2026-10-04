use super::*;
use crate::{
  scene::{
    CameraComponent, Scene, TransformComponent,
    ui::{Transform2DComponent, UiComponent},
  },
  simulation::texture_cache::TextureCache,
};
use alloc::sync::Arc;
use parking_lot::RwLock;

#[test]
fn test_ui_layout_relative_placement() {
  let tex_cache = Arc::new(RwLock::new(TextureCache::new("test_tex_cache_ui")));
  let scene = Scene::new(tex_cache);
  scene.register_all_crate_components();

  // 1. Root Background Panel
  let bg_entity = scene.spawn_entity("Background");
  let mut bg_t2d = Transform2DComponent::default();
  bg_t2d.local_position = [0.0, 0.0];
  bg_t2d.size = [1000.0, 1000.0];
  scene.add_component(bg_entity, bg_t2d).unwrap();
  scene.add_component(bg_entity, UiComponent::default()).unwrap();

  // 2. Child Panel
  let child_panel = scene.spawn_entity("Child");
  scene.set_parent(child_panel, Some(bg_entity));
  let mut child_t2d = Transform2DComponent::default();
  child_t2d.local_position = [100.0, 50.0];
  child_t2d.size = [200.0, 200.0];
  scene.add_component(child_panel, child_t2d).unwrap();
  scene.add_component(child_panel, UiComponent::default()).unwrap();

  // 3. Grandchild Panel (Anchored to Bottom-Right of Child)
  let gc_panel = scene.spawn_entity("GrandChild");
  scene.set_parent(gc_panel, Some(child_panel));
  let mut gc_t2d = Transform2DComponent::default();
  gc_t2d.anchor_min = [1.0, 1.0];
  gc_t2d.pivot = [1.0, 1.0];
  gc_t2d.local_position = [-10.0, -10.0]; // 10px padding from right-bottom corner
  gc_t2d.size = [50.0, 50.0];
  scene.add_component(gc_panel, gc_t2d).unwrap();
  scene.add_component(gc_panel, UiComponent::default()).unwrap();

  // Run layout pass directly
  crate::scene::ui::update_ui_layouts(&scene, [1000.0, 1000.0]);

  // Verify background
  scene
    .with_component::<Transform2DComponent, _, _>(bg_entity, |t| {
      assert_eq!(t.global_bounds[0..2], [0.0, 0.0]);
      assert_eq!(t.global_bounds[2..4], [1000.0, 1000.0]);
    })
    .unwrap();

  // Verify child
  scene
    .with_component::<Transform2DComponent, _, _>(child_panel, |t| {
      assert_eq!(t.global_bounds[0..2], [100.0, 50.0]);
      assert_eq!(t.global_bounds[2..4], [200.0, 200.0]);
    })
    .unwrap();

  // Verify grandchild
  // Parent pos: (100, 50), size: (200, 200) => Bottom-Right is (300, 250)
  // Grandchild offset: (-10, -10) => (290, 240)
  // Pivot (1,1) means its own bottom-right is at (290, 240).
  // Size: 50x50 => Top-left global_position should be (240, 190)
  scene
    .with_component::<Transform2DComponent, _, _>(gc_panel, |t| {
      assert_eq!(t.global_bounds[0..2], [240.0, 190.0]);
      assert_eq!(t.global_bounds[2..4], [50.0, 50.0]);
    })
    .unwrap();
}

#[test]
fn test_multi_micro_layer_draw_call_assignment() {
  use crate::{
    scene::{
      CameraComponent, CameraProjection, HighResTransformComponent, ReferenceFrameComponent,
      ReferenceFrameType, Scene, StaticMeshComponent, TransformComponent,
    },
    simulation::{comet::generate_uv_sphere, texture_cache::TextureCache},
  };
  use aethervk_oshal_rlib::math::vector::{vec3::Vec3f32, vec3f64::Vec3f64, vec4::Quat};
  use alloc::sync::Arc;
  use parking_lot::RwLock;

  const AU_TO_KM: f32 = 149_597_870.7;

  let tex_cache = Arc::new(RwLock::new(TextureCache::new("test_multi_micro_layer")));
  let scene = Scene::new(tex_cache);
  scene.register_all_crate_components();

  // Root entity
  let root = scene.spawn_entity("Root");
  scene
    .add_component(
      root,
      ReferenceFrameComponent {
        frame_type: ReferenceFrameType::Macro,
        scale: 1.0,
        soi_radius: f32::MAX,
        depth_layer: 0,
      },
    )
    .unwrap();

  // Camera at origin
  let camera = scene.spawn_entity("camera");
  scene.set_parent(camera, Some(root));
  scene
    .add_component(
      camera,
      HighResTransformComponent {
        position: Vec3f64::zero(),
        rotation: aethervk_oshal_rlib::math::vector::vec4f64::Quat64::from_components(
          1.0, 0.0, 0.0, 0.0,
        ),
        scale: Vec3f32::one(),
      },
    )
    .unwrap();
  scene
    .add_component(
      camera,
      CameraComponent {
        projection: CameraProjection::Perspective {
          fov: 60.0f32.to_radians(),
          aspect_ratio: 16.0 / 9.0,
          near: 1e-5,
          far: 1000.0,
        },
        focus_distance: 1.0,
      },
    )
    .unwrap();

  // 3 micro frames and their sphere children
  let sphere_mesh = Arc::new(generate_uv_sphere(2.0, 8, 8, 1.0, false));
  let layer_positions_au: [f32; 3] = [0.5, 1.0, 1.5];
  let mut sphere_ids = [root; 3]; // placeholder

  for (i, &pos_au) in layer_positions_au.iter().enumerate() {
    let layer_idx = (i + 1) as u32;

    let frame_id = scene.spawn_entity(&alloc::format!("frame_{}", layer_idx));
    scene.set_parent(frame_id, Some(root));
    scene
      .add_component(
        frame_id,
        TransformComponent {
          position: Vec3f32::from_components(pos_au, 0.0, 0.0),
          rotation: Quat::from_components(1.0, 0.0, 0.0, 0.0),
          scale: Vec3f32::one(),
        },
      )
      .unwrap();
    scene
      .add_component(
        frame_id,
        ReferenceFrameComponent {
          frame_type: ReferenceFrameType::Micro,
          scale: 1.0 / AU_TO_KM,
          soi_radius: 50.0, // 50 km SOI
          depth_layer: layer_idx,
        },
      )
      .unwrap();

    let sphere_id = scene.spawn_entity(&alloc::format!("sphere_{}", layer_idx));
    scene.set_parent(sphere_id, Some(frame_id));
    scene
      .add_component(
        sphere_id,
        TransformComponent {
          position: Vec3f32::zero(),
          rotation: Quat::from_components(1.0, 0.0, 0.0, 0.0),
          scale: Vec3f32::one(),
        },
      )
      .unwrap();
    scene
      .add_component(
        sphere_id,
        StaticMeshComponent {
          asset_path: alloc::format!("sphere_{}", layer_idx),
          mesh: Arc::clone(&sphere_mesh),
          emissive_color: [0.0; 4],
          is_visible: true,
        },
      )
      .unwrap();
    sphere_ids[i] = sphere_id;
  }

  // Assertions: each sphere's ancestor_depth_layer matches its layer index
  for (i, &sphere_id) in sphere_ids.iter().enumerate() {
    let expected_layer = (i + 1) as u32;
    let actual_layer = scene.ancestor_depth_layer(sphere_id);
    assert_eq!(
      actual_layer,
      expected_layer,
      "sphere_{} should be in depth_layer {} but got {}",
      i + 1,
      expected_layer,
      actual_layer
    );
  }

  // Camera and root are in the macro layer (0)
  assert_eq!(
    scene.ancestor_depth_layer(camera),
    0,
    "camera should be in macro layer 0"
  );
  assert_eq!(
    scene.ancestor_depth_layer(root),
    0,
    "root should be in macro layer 0"
  );
}

/// CometOrbiting regression (no_trajectories.rdc): with the camera parented to a body inside a
/// micro frame (global scale 1/AU_TO_KM), a root-level trajectory through the body must keep
/// scale 1 and land a few km from the camera, i.e. inside a 6 km ortho view.
#[test]
fn test_trajectory_rte_with_camera_parented_in_micro_frame() {
  use super::{rte_relative_to_camera, trajectory_rte};
  use crate::{
    scene::{
      HighResTransformComponent, ReferenceFrameComponent, ReferenceFrameType, Scene,
      TransformComponent, trajectory::TrajectoryComponent,
    },
    simulation::texture_cache::TextureCache,
    simulation_api::reposition::{AU_TO_KM, compute_macro_and_residual},
  };
  use aethervk_oshal_rlib::math::vector::{Vector3, vec3::Vec3f32, vec3f64::DVec3, vec4::Quat};
  use alloc::sync::Arc;
  use parking_lot::RwLock;

  let scene = Scene::new(Arc::new(RwLock::new(TextureCache::new(
    "test_traj_rte_parented",
  ))));
  scene.register_all_crate_components();
  let root = scene.spawn_entity("Root");
  scene
    .add_component(
      root,
      ReferenceFrameComponent {
        frame_type: ReferenceFrameType::Macro,
        scale: 1.0,
        soi_radius: f32::MAX,
        depth_layer: 0,
      },
    )
    .unwrap();

  // comet at ~1.5 AU: frame (AU) + body residual (km), as reposition does
  let comet_km = DVec3::from_components(1.5 * AU_TO_KM, 0.3 * AU_TO_KM, 1234.5);
  let (frame_au, residual_km) = compute_macro_and_residual(comet_km);
  let frame = scene.spawn_entity("Comet_subtree");
  scene.set_parent(frame, Some(root));
  let mut ft = TransformComponent::default();
  ft.position = frame_au;
  scene.add_component(frame, ft).unwrap();
  scene
    .add_component(
      frame,
      ReferenceFrameComponent {
        frame_type: ReferenceFrameType::Micro,
        scale: (1.0 / AU_TO_KM) as f32,
        soi_radius: 50.0,
        depth_layer: 1,
      },
    )
    .unwrap();
  let body = scene.spawn_entity("Comet_body");
  scene.set_parent(body, Some(frame));
  let mut bt = TransformComponent::default();
  bt.position = residual_km;
  scene.add_component(body, bt).unwrap();

  // camera 6 km above the nucleus, parented to it (CometOrbiting)
  let camera = scene.spawn_entity("camera");
  scene.set_parent(camera, Some(body));
  scene
    .add_component(
      camera,
      HighResTransformComponent {
        position: DVec3::from_components(0.0, 0.0, 6.0),
        rotation: aethervk_oshal_rlib::math::vector::vec4f64::Quat64::from_components(
          1.0, 0.0, 0.0, 0.0,
        ),
        scale: Vec3f32::one(),
      },
    )
    .unwrap();
  let cam = scene.global_transform_f64(camera).unwrap();
  assert!(
    cam.scale.x() < 1e-6,
    "premise: the parented camera inherits the frame scale"
  );

  // root-level trajectory: a straight 0.2 AU segment through the nucleus along +x
  let traj_entity = scene.spawn_entity("Comet_orbit");
  scene.set_parent(traj_entity, Some(root));
  scene.add_component(traj_entity, TransformComponent::default()).unwrap();
  let c = comet_km * (1.0 / AU_TO_KM);
  let p = |dx: f64| [c.x() + dx, c.y(), c.z()];
  let traj = TrajectoryComponent::from_f64(
    alloc::vec![p(-0.1), p(-0.0333), p(0.0333), p(0.1)],
    [1.0; 4],
    2.0,
    0,
    32,
  );

  let g = scene.global_transform_f64(traj_entity).unwrap();
  let rte = rte_relative_to_camera(g.position, g.rotation, g.scale, &cam);
  assert_eq!(
    rte.scale,
    Vec3f32::one(),
    "macro objects must not be scaled by the camera scale"
  );

  // 6 km ortho view: the refined, camera-relative track passes right under the camera
  let view = trajectory_rte::ViewScale {
    ortho_half_extent: 6.0 / AU_TO_KM,
    tan_half_fov: 0.0,
  };
  let rel = trajectory_rte::camera_relative_control_points(&traj, &[], &rte, view);
  let closest_km = rel
    .chunks_exact(4)
    .flat_map(|seg| seg.iter())
    .map(|q| (q[1] as f64).hypot(q[2] as f64) * AU_TO_KM)
    .fold(f64::MAX, f64::min);
  assert!(
    (closest_km - 6.0).abs() < 0.05,
    "track should pass 6 km below the camera: {closest_km}"
  );
}
