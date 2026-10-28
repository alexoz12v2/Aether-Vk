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

/// The Earth observer's light time moves the comet's label and its marked lines by the same
/// offset as its micro layer, and nothing else; without the view component nothing moves.
#[test]
fn light_time_overlay_displaces_only_the_comet_label_and_lines() {
  use super::LightTimeOverlay;
  use crate::simulation_api::earth_observer::{LightTimeViewComponent, mark_light_time_track};
  use aethervk_oshal_rlib::math::vector::Vector3;

  let scene = Scene::new(Arc::new(RwLock::new(TextureCache::new("test_light_time_overlay"))));
  scene.register_all_crate_components();
  let camera = scene.spawn_entity("camera");
  let comet = scene.spawn_entity("Comet_body");
  let orbit = scene.spawn_entity("Comet_orbit");
  let effective = scene.spawn_entity("effective_comet_trajectory");
  let earth_orbit = scene.spawn_entity("Earth_orbit");
  let other_comet = scene.spawn_entity("Other_body");
  mark_light_time_track(&scene, orbit, comet);
  mark_light_time_track(&scene, effective, comet);
  mark_light_time_track(&scene, earth_orbit, other_comet);

  let idle = LightTimeOverlay::of(&scene, camera);
  assert!(idle.label_offset(comet).is_none() && idle.track_offset(orbit).is_none());

  let offset_au = [1.5e-4, -2.0e-4, 3.0e-5];
  scene.register_component::<LightTimeViewComponent>(&[]);
  scene
    .add_component(camera, LightTimeViewComponent { comet, delay_s: 1650.0, offset_au })
    .unwrap();
  let lt = LightTimeOverlay::of(&scene, camera);
  for o in [lt.label_offset(comet), lt.track_offset(orbit), lt.track_offset(effective)] {
    let o = o.expect("the comet's overlays follow the light time");
    assert_eq!([o.x(), o.y(), o.z()], offset_au);
  }
  assert!(lt.label_offset(other_comet).is_none(), "another body's label stays geometric");
  assert!(lt.track_offset(earth_orbit).is_none(), "another comet's line stays geometric");
  assert!(lt.track_offset(comet).is_none(), "an unmarked entity is not a line");
}

/// Jet markers: a site facing the Sun (the heliocentric origin) is lit, the antipodal one is
/// not, and turning the nucleus half a revolution swaps them (the marker follows the spin).
#[test]
fn jet_site_lit_follows_the_sun_and_the_nucleus_rotation() {
  use super::jet_site_lit;
  use aethervk_oshal_rlib::math::quaternion::Quaternion;

  let scene = Scene::new(Arc::new(RwLock::new(TextureCache::new("test_jet_site_lit"))));
  scene.register_all_crate_components();
  let at = |position: Vec3f32, rotation: Quat| TransformComponent { position, rotation, scale: Vec3f32::one() };
  let nucleus = scene.spawn_entity("Comet_body");
  scene
    .add_component(nucleus, at(Vec3f32::from_components(1.0, 0.0, 0.0), Quat::identity()))
    .unwrap();
  let site = |name: &str, x: f32| {
    let jet = scene.spawn_entity(name);
    scene.set_parent(jet, Some(nucleus));
    scene.add_component(jet, at(Vec3f32::from_components(x, 0.0, 0.0), Quat::identity())).unwrap();
    jet
  };
  let sunward = site("jet_sunward", -1e-6);
  let antisolar = site("jet_antisolar", 1e-6);
  assert_eq!(jet_site_lit(&scene, sunward), Some(true));
  assert_eq!(jet_site_lit(&scene, antisolar), Some(false));
  assert_eq!(jet_site_lit(&scene, nucleus), None, "no parent: not a jet site");

  scene
    .with_component_mut(nucleus, |t: &mut TransformComponent| {
      t.rotation = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), core::f32::consts::PI);
    })
    .unwrap();
  assert_eq!(jet_site_lit(&scene, sunward), Some(false));
  assert_eq!(jet_site_lit(&scene, antisolar), Some(true));
}

/// The jet line: three quarters of the gizmo's body axes (which reach 1.5 × the gizmo radius),
/// along the site direction (local +Z of an orthonormal frame).
#[test]
fn jet_line_frame_points_local_z_along_the_site_and_is_orthonormal() {
  use super::jet_line_frame;
  use aethervk_oshal_rlib::math::vector::Vector4;
  assert!((crate::scene::JET_LINE_LENGTH_OVER_GIZMO_RADIUS - 1.125).abs() < 1e-6);

  for site in [[3.0, 4.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, -5.0], [1.0, -2.0, 0.5]] {
    let m = jet_line_frame(site).expect("a non-zero site has a frame");
    let n = (site[0] * site[0] + site[1] * site[1] + site[2] * site[2]).sqrt();
    let col = |c: &aethervk_oshal_rlib::math::vector::vec4::Vec4f32| [c.x() as f64, c.y() as f64, c.z() as f64];
    let (x, y, z) = (col(&m.x), col(&m.y), col(&m.z));
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    for i in 0..3 {
      assert!((z[i] - site[i] / n).abs() < 1e-6, "local Z is the site direction: {site:?}");
    }
    for (a, b) in [(x, y), (y, z), (z, x)] {
      assert!(dot(a, b).abs() < 1e-6, "orthogonal axes for {site:?}");
    }
    for a in [x, y, z] {
      assert!((dot(a, a) - 1.0).abs() < 1e-6, "unit axes for {site:?}");
    }
    assert_eq!(m.w.w(), 1.0);
    assert_eq!((m.w.x(), m.w.y(), m.w.z()), (0.0, 0.0, 0.0), "the line starts at the nucleus centre");
  }
  assert!(jet_line_frame([0.0, 0.0, 0.0]).is_none());
}

/// The angle arcs: longitude on the equator from the prime meridian, latitude on the jet's
/// meridian from the equator; both at 0.8 × the gizmo radius, ending at the jet's angles.
#[test]
fn jet_angle_arcs_measure_latitude_and_longitude() {
  use super::{JET_ANGLE_ARC_OVER_GIZMO_RADIUS, jet_angle_arcs};
  let (lat, lon) = (30f64.to_radians(), 45f64.to_radians());
  let site = [lat.cos() * lon.cos() * 7.0, lat.cos() * lon.sin() * 7.0, lat.sin() * 7.0];
  let gizmo_r = 10.0;
  let a = jet_angle_arcs(site, gizmo_r).expect("arcs");
  assert!((a.lat_rad - lat).abs() < 1e-12 && (a.lon_rad - lon).abs() < 1e-12);
  let r = gizmo_r * JET_ANGLE_ARC_OVER_GIZMO_RADIUS;
  let close = |p: [f64; 3], q: [f64; 3]| (0..3).all(|i| (p[i] - q[i]).abs() < 1e-9);
  let at = |phi: f64, lam: f64| [r * phi.cos() * lam.cos(), r * phi.cos() * lam.sin(), r * phi.sin()];

  let [lon_arc, lat_arc] = &a.arcs;
  assert!(close(lon_arc.points[0], [r, 0.0, 0.0]), "longitude starts on the prime meridian");
  assert!(close(*lon_arc.points.last().unwrap(), at(0.0, lon)));
  assert!(lon_arc.points.iter().all(|p| p[2].abs() < 1e-9), "on the equator");
  assert!(close(lat_arc.points[0], at(0.0, lon)), "latitude starts on the equator, at the jet's meridian");
  assert!(close(*lat_arc.points.last().unwrap(), at(lat, lon)), "and ends on the jet's direction");
  for p in lon_arc.points.iter().chain(&lat_arc.points) {
    assert!(((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - r).abs() < 1e-9, "on the arc radius");
  }
  assert!(lon_arc.points.len() >= 3 && lat_arc.points.len() >= 3);
  assert_eq!(lon_arc.label, "lon 45.0°");
  assert_eq!(lat_arc.label, "lat 30.0°");
  assert!(close(lon_arc.label_at, at(0.0, lon * 0.5)), "labels at the arc midpoints");
  assert!(close(lat_arc.label_at, at(lat * 0.5, lon)));

  // southern and western sites: signed angles
  let (lat, lon) = (-20f64.to_radians(), -100f64.to_radians());
  let a = jet_angle_arcs([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], gizmo_r).unwrap();
  assert_eq!(a.arcs[0].label, "lon -100.0°");
  assert_eq!(a.arcs[1].label, "lat -20.0°");
  assert!(jet_angle_arcs([0.0; 3], gizmo_r).is_none());
}

/// An orthographic camera shows what lies behind its eye unless `CAMERA_HIDE_BEHIND` is set:
/// the macro layer mirrors its slab and object slabs keep a negative near; with the bit the near
/// is clamped at the eye. Perspective is never changed.
#[test]
fn orthographic_depth_shows_behind_the_eye_unless_hidden() {
  use super::{fit_depth_slab, macro_layer_depth};
  assert_eq!(macro_layer_depth(1e-6, 50.0, true, false), (-50.0, 50.0));
  assert_eq!(macro_layer_depth(1e-6, 50.0, true, true), (1e-6, 50.0));
  assert_eq!(macro_layer_depth(1e-6, 50.0, false, false), (1e-6, 50.0));

  // a 10 km sphere 4 km behind the eye (depth −4): visible only when shown
  let (n, f) = fit_depth_slab(-4.0, 10.0, true, false);
  assert!(n < -14.0 && f > 6.0, "{n} {f}");
  let (n, f) = fit_depth_slab(-4.0, 10.0, true, true);
  assert_eq!(n, 0.0);
  assert!(f > 6.0);
  // wholly behind and hidden: an empty slab, skipped by the commit
  let (n, f) = fit_depth_slab(-40.0, 10.0, true, true);
  assert!(f <= n, "{n} {f}");
  // perspective: always in front, whatever the bit
  assert_eq!(fit_depth_slab(30.0, 10.0, false, false), fit_depth_slab(30.0, 10.0, false, true));
  assert!(fit_depth_slab(30.0, 10.0, false, false).0 > 0.0);
}

/// The comet's screen position is the indicator's projection, light-time displaced: dead ahead is
/// the viewport centre, a sideways offset lands right of it, outside the field is projectable but
/// off screen, and behind a perspective eye or without an observer comet there is none.
#[test]
fn comet_screen_position_projects_the_light_time_displaced_comet() {
  use super::comet_screen_position;
  use crate::{
    scene::{CameraComponent, HighResTransformComponent, Scene, TransformComponent},
    simulation::texture_cache::TextureCache,
    simulation_api::earth_observer::LightTimeViewComponent,
  };
  use aethervk_oshal_rlib::math::vector::{vec3::Vec3f32, vec3f64::DVec3, vec4f64::Quat64};

  let scene = Scene::new(Arc::new(RwLock::new(TextureCache::new("test_comet_screen_pos"))));
  scene.register_all_crate_components();
  let comet = scene.spawn_entity("comet");
  let mut ct = TransformComponent::default();
  ct.position = Vec3f32::from_components(0.0, -1.0, 0.0); // 1 AU along the camera forward (−Y)
  scene.add_component(comet, ct).unwrap();
  let camera = scene.spawn_entity("camera");
  scene
    .add_component(
      camera,
      HighResTransformComponent {
        position: DVec3::from_components(0.0, 0.0, 0.0),
        rotation: Quat64::from_components(0.0, 0.0, 0.0, 1.0),
        scale: Vec3f32::one(),
      },
    )
    .unwrap();
  // 2:1 viewport, 10° vertical field
  let fov = 10f32.to_radians();
  scene.add_component(camera, CameraComponent::new_persp(fov, 2.0, 1e-6, 10.0)).unwrap();
  let extent = [800, 400];
  assert_eq!(comet_screen_position(&scene, camera, extent), None, "no observer comet");

  let mut lt = LightTimeViewComponent { comet, delay_s: 0.0, offset_au: [0.0, 0.0, 0.0] };
  scene.add_component(camera, lt).unwrap();
  let p = comet_screen_position(&scene, camera, extent).unwrap();
  assert!(p.on_screen);
  assert!((p.x_px - 400.0).abs() < 1e-6 && (p.y_px - 200.0).abs() < 1e-6, "{p:?}");

  // displaced along screen right (world −X for this identity camera, as the indicator draws it)
  // by a quarter of the half-width: 100 px right of centre
  let half_w = (fov as f64 / 2.0).tan() * 2.0;
  lt.offset_au = [-0.25 * half_w, 0.0, 0.0];
  scene.with_component_mut(camera, |l: &mut LightTimeViewComponent| *l = lt).unwrap();
  let p = comet_screen_position(&scene, camera, extent).unwrap();
  assert!(p.on_screen && (p.x_px - 500.0).abs() < 0.05 && (p.y_px - 200.0).abs() < 0.05, "{p:?}");

  // outside the field: projectable, off screen
  lt.offset_au = [3.0 * half_w, 0.0, 0.0];
  scene.with_component_mut(camera, |l: &mut LightTimeViewComponent| *l = lt).unwrap();
  assert!(!comet_screen_position(&scene, camera, extent).unwrap().on_screen);

  // behind the eye
  lt.offset_au = [0.0, 2.0, 0.0];
  scene.with_component_mut(camera, |l: &mut LightTimeViewComponent| *l = lt).unwrap();
  assert_eq!(comet_screen_position(&scene, camera, extent), None);
}
