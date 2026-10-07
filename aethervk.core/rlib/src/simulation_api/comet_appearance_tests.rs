use super::*;
use crate::{
  scene::SphereGizmoComponent,
  simulation::{
    asset_library::{AssetLibrary, TextureChannel},
    texture_cache::TextureCache,
  },
};
use aethervk_oshal_rlib::math::{
  matrix::{Matrix4, mat4::Mat4x4f32},
  vector::{Vector4, vec4::Vec4f32},
};
use parking_lot::RwLock;
use std::path::PathBuf;

struct Fixture {
  scene: Scene,
  body: EntityId,
  visual: EntityId,
  library: AssetLibrary,
  dir: PathBuf,
}

impl Drop for Fixture {
  fn drop(&mut self) {
    self.library.clear();
    let _ = std::fs::remove_dir_all(&self.dir);
  }
}

fn fixture(tag: &str) -> Fixture {
  use aethervk_oshal_rlib::math::matrix::SquareMatrix;
  static N: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
  let n = N.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!("avk_appearance_{tag}_{}_{n}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();

  let scene = Scene::new(alloc::sync::Arc::new(RwLock::new(TextureCache::new(
    "test_comet_appearance",
  ))));
  scene.register_all_crate_components();
  let root = scene.spawn_entity("Root");
  let body = scene.spawn_entity("Comet_body");
  scene.set_parent(body, Some(root));
  scene.add_component(body, TransformComponent::default()).unwrap();
  scene
    .add_component(
      body,
      SphereGizmoComponent {
        radius: 4.0,
        subdivisions: 4.0,
        local_frame: Mat4x4f32::identity(),
        is_visible: true,
      },
    )
    .unwrap();
  let visual = spawn_comet_visual(&scene, body);

  let mut library = AssetLibrary::new();
  library.set_cache_dir(dir.join("cache").to_str().unwrap()).unwrap();
  Fixture {
    scene,
    body,
    visual,
    library,
    dir,
  }
}

/// Writes an OBJ of a UV sphere of `radius` centred at `center` and imports it.
fn import_offset_sphere(f: &mut Fixture, center: [f32; 3], radius: f32) -> AssetId {
  let sphere = generate_uv_sphere(radius, 12, 12, 1.0, false);
  let mut obj = String::new();
  for v in &sphere.vertices {
    obj += &format!(
      "v {} {} {}\n",
      v.position[0] + center[0],
      v.position[1] + center[1],
      v.position[2] + center[2]
    );
  }
  for t in sphere.indices.chunks_exact(3) {
    obj += &format!("f {} {} {}\n", t[0] + 1, t[1] + 1, t[2] + 1);
  }
  let path = f.dir.join(format!("sphere_{}_{}.obj", center[0], radius));
  std::fs::write(&path, obj).unwrap();
  f.library.import_file(path.to_str().unwrap()).unwrap().mesh.unwrap()
}

fn import_png(f: &mut Fixture, name: &str, rgba: [u8; 4]) -> AssetId {
  let img = image::RgbaImage::from_pixel(4, 4, image::Rgba(rgba));
  let path = f.dir.join(name);
  img.save(&path).unwrap();
  f.library.import_file(path.to_str().unwrap()).unwrap().textures[0]
}

fn mesh_of(f: &Fixture) -> StaticMeshComponent {
  f.scene.with_component(f.visual, |m: &StaticMeshComponent| m.clone()).unwrap()
}

fn transform_of(f: &Fixture) -> TransformComponent {
  f.scene.with_component(f.visual, |t: &TransformComponent| *t).unwrap()
}

/// Positions of the displayed mesh in the body frame, using the renderer's matrix path.
fn body_frame_positions(f: &Fixture) -> Vec<[f32; 3]> {
  use aethervk_oshal_rlib::math::matrix::MatrixVectorMul;
  let m: Mat4x4f32 = transform_of(f).to_mat4();
  mesh_of(f)
    .mesh
    .vertices
    .iter()
    .map(|v| {
      let p = m.mul_vector(Vec4f32::from_components(
        v.position[0],
        v.position[1],
        v.position[2],
        1.0,
      ));
      [p[0], p[1], p[2]]
    })
    .collect()
}

/// Axis-aligned bounding box centre (a vertex centroid would be biased by the UV sphere's
/// duplicated seam/pole vertices) and the max distance of any point from it.
fn centroid_and_max_dist(points: &[[f32; 3]]) -> ([f32; 3], f32) {
  let mut lo = [f32::MAX; 3];
  let mut hi = [f32::MIN; 3];
  for p in points {
    for i in 0..3 {
      lo[i] = lo[i].min(p[i]);
      hi[i] = hi[i].max(p[i]);
    }
  }
  let c = [
    (lo[0] + hi[0]) * 0.5,
    (lo[1] + hi[1]) * 0.5,
    (lo[2] + hi[2]) * 0.5,
  ];
  let r = points
    .iter()
    .map(|p| ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt())
    .fold(0.0f32, f32::max);
  (c, r)
}

fn custom(mesh: AssetId) -> CometAppearanceWiring {
  CometAppearanceWiring {
    mode: CometDisplayMode::Custom,
    mesh: Some(mesh),
    textures: [None; 4],
  }
}

#[test]
fn default_visual_is_the_legacy_sphere_on_a_child_entity() {
  let f = fixture("default");
  assert_eq!(f.scene.get_parent(f.visual), Some(f.body));
  assert!(
    f.scene.with_component(f.body, |_: &StaticMeshComponent| ()).is_none(),
    "the body itself must not render a mesh anymore"
  );
  let mesh = mesh_of(&f);
  assert_eq!(mesh.asset_path, DEFAULT_COMET_ASSET_PATH);
  assert!(!mesh.is_visible);
  let legacy = generate_uv_sphere(2.0, 16, 16, 1.0, false);
  assert_eq!(mesh.mesh.indices, legacy.indices);
  for (a, b) in mesh.mesh.vertices.iter().zip(&legacy.vertices) {
    assert_eq!(a.position, b.position);
    assert_eq!(a.normal, b.normal);
  }
  let t = transform_of(&f);
  assert_eq!(t.position, Vec3f32::zero());
  assert_eq!(t.scale, Vec3f32::from_components(1.0, 1.0, 1.0));
  assert_eq!(nucleus_radius_km(&f.scene, f.body), 2.0);
}

#[test]
fn custom_mesh_is_recentred_and_scaled_to_nucleus_radius() {
  let mut f = fixture("custom");
  let mesh = import_offset_sphere(&mut f, [50.0, -20.0, 7.0], 10.0);
  apply_comet_appearance(
    &f.scene,
    f.visual,
    2.0,
    custom(mesh),
    Default::default(),
    &f.library,
    false,
  )
  .unwrap();

  let points = body_frame_positions(&f);
  let (c, r) = centroid_and_max_dist(&points);
  assert!(
    c.iter().all(|x| x.abs() < 1e-3),
    "recentred on the body origin: {c:?}"
  );
  assert!(
    (r - 2.0).abs() < 0.02,
    "bounding radius {r} should equal the nucleus radius"
  );
  assert!(mesh_of(&f).asset_path.starts_with("asset:"));
}

#[test]
fn radius_change_rescales_custom_mesh_without_touching_vertices() {
  let mut f = fixture("rescale");
  let mesh = import_offset_sphere(&mut f, [0.0, 0.0, 0.0], 3.0);
  apply_comet_appearance(
    &f.scene,
    f.visual,
    2.0,
    custom(mesh),
    Default::default(),
    &f.library,
    false,
  )
  .unwrap();
  let before = mesh_of(&f);
  assert!(rescale_custom_visual(&f.scene, f.visual, 7.5));
  let after = mesh_of(&f);
  assert_eq!(
    before.mesh.id, after.mesh.id,
    "no GPU resource churn on radius change"
  );
  let (_, r) = centroid_and_max_dist(&body_frame_positions(&f));
  assert!((r - 7.5).abs() < 0.05, "radius {r}");
}

#[test]
fn rescale_is_a_noop_in_default_mode() {
  let f = fixture("rescale_default");
  assert!(!rescale_custom_visual(&f.scene, f.visual, 9.0));
  assert_eq!(
    transform_of(&f).scale,
    Vec3f32::from_components(1.0, 1.0, 1.0)
  );
}

#[test]
fn offset_translation_is_in_radii_and_rotation_is_applied() {
  let mut f = fixture("offset");
  let mesh = import_offset_sphere(&mut f, [0.0, 0.0, 0.0], 1.0);
  let offset = CometVisualOffset {
    yaw_pitch_roll_deg: [90.0, 0.0, 0.0],
    translation_radii: [0.5, 0.0, 0.0],
  };
  apply_comet_appearance(
    &f.scene,
    f.visual,
    4.0,
    custom(mesh),
    offset,
    &f.library,
    false,
  )
  .unwrap();
  let (c, r) = centroid_and_max_dist(&body_frame_positions(&f));
  assert!(
    (c[0] - 2.0).abs() < 1e-3 && c[1].abs() < 1e-3,
    "0.5 R = 2 km along +X: {c:?}"
  );
  assert!((r - 4.0).abs() < 0.05);

  // yaw 90° maps +X to +Y
  let q = offset.rotation();
  let x = q.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
  assert!(x.x().abs() < 1e-5 && (x.y() - 1.0).abs() < 1e-5, "{x:?}");
}

#[test]
fn offset_rotation_is_applied_around_the_recentred_mesh() {
  let mut f = fixture("offset_rot");
  // off-centre source: rotation must still keep the mesh centred on the body
  let mesh = import_offset_sphere(&mut f, [10.0, 0.0, 0.0], 1.0);
  let offset = CometVisualOffset {
    yaw_pitch_roll_deg: [30.0, 45.0, 60.0],
    translation_radii: [0.0; 3],
  };
  apply_comet_appearance(
    &f.scene,
    f.visual,
    2.0,
    custom(mesh),
    offset,
    &f.library,
    false,
  )
  .unwrap();
  let (c, _) = centroid_and_max_dist(&body_frame_positions(&f));
  assert!(c.iter().all(|x| x.abs() < 1e-3), "{c:?}");
}

#[test]
fn wiring_is_locked_while_running_but_offset_is_not() {
  let mut f = fixture("lock");
  let mesh = import_offset_sphere(&mut f, [0.0; 3], 1.0);
  let err = apply_comet_appearance(
    &f.scene,
    f.visual,
    2.0,
    custom(mesh),
    Default::default(),
    &f.library,
    true,
  );
  assert_eq!(err, Err(AppearanceError::SimulationRunning));
  assert_eq!(
    mesh_of(&f).asset_path,
    DEFAULT_COMET_ASSET_PATH,
    "nothing changed"
  );

  // paused: wire it
  apply_comet_appearance(
    &f.scene,
    f.visual,
    2.0,
    custom(mesh),
    Default::default(),
    &f.library,
    false,
  )
  .unwrap();
  let id = mesh_of(&f).mesh.id;
  // running: offset-only change is fine and does not rebuild the mesh
  let offset = CometVisualOffset {
    yaw_pitch_roll_deg: [0.0, 0.0, 10.0],
    translation_radii: [0.0, 0.1, 0.0],
  };
  apply_comet_appearance(
    &f.scene,
    f.visual,
    2.0,
    custom(mesh),
    offset,
    &f.library,
    true,
  )
  .unwrap();
  assert_eq!(mesh_of(&f).mesh.id, id);
  let stored = f.scene.with_component(f.visual, |a: &CometAppearanceComponent| *a).unwrap();
  assert_eq!(stored.offset, offset);
}

#[test]
fn unknown_assets_are_rejected_without_side_effects() {
  let mut f = fixture("unknown");
  let mesh = import_offset_sphere(&mut f, [0.0; 3], 1.0);
  assert_eq!(
    apply_comet_appearance(
      &f.scene,
      f.visual,
      2.0,
      custom(9999),
      Default::default(),
      &f.library,
      false
    ),
    Err(AppearanceError::UnknownMesh)
  );
  let mut w = custom(mesh);
  w.textures[TextureChannel::Albedo as usize] = Some(mesh); // a mesh id is not a texture
  assert_eq!(
    apply_comet_appearance(
      &f.scene,
      f.visual,
      2.0,
      w,
      Default::default(),
      &f.library,
      false
    ),
    Err(AppearanceError::UnknownTexture)
  );
  assert_eq!(mesh_of(&f).asset_path, DEFAULT_COMET_ASSET_PATH);
  assert_eq!(
    f.scene.with_component(f.visual, |a: &CometAppearanceComponent| *a).unwrap(),
    CometAppearanceComponent::default()
  );
}

#[test]
fn wired_textures_are_zero_copy_views_of_the_library() {
  let mut f = fixture("textures");
  let mesh = import_offset_sphere(&mut f, [0.0; 3], 1.0);
  let albedo = import_png(&mut f, "albedo.png", [200, 100, 50, 255]);
  let normal = import_png(&mut f, "normal.png", [128, 128, 255, 255]);
  let mut w = custom(mesh);
  w.textures[TextureChannel::Albedo as usize] = Some(albedo);
  w.textures[TextureChannel::Normal as usize] = Some(normal);
  apply_comet_appearance(
    &f.scene,
    f.visual,
    2.0,
    w,
    Default::default(),
    &f.library,
    false,
  )
  .unwrap();

  let m = mesh_of(&f);
  let a = m.mesh.albedo_map.as_ref().unwrap();
  let lib_a = f.library.texture(albedo).unwrap().texture();
  assert_eq!(
    a.data.as_ptr(),
    lib_a.data.as_ptr(),
    "same mapped bytes, no copy"
  );
  assert!(m.mesh.normal_map.is_some());
  assert!(m.mesh.roughness_map.is_none() && m.mesh.ao_map.is_none());

  // rewiring a channel produces a new GPU cache key
  let id = m.mesh.id;
  w.textures[TextureChannel::Normal as usize] = None;
  apply_comet_appearance(
    &f.scene,
    f.visual,
    2.0,
    w,
    Default::default(),
    &f.library,
    false,
  )
  .unwrap();
  assert_ne!(mesh_of(&f).mesh.id, id);
  assert!(mesh_of(&f).mesh.normal_map.is_none());
}

#[test]
fn switching_back_to_default_restores_the_sphere_at_current_radius() {
  let mut f = fixture("back");
  let mesh = import_offset_sphere(&mut f, [3.0, 0.0, 0.0], 2.0);
  let _ = f
    .scene
    .with_component_mut(f.visual, |m: &mut StaticMeshComponent| m.is_visible = true);
  apply_comet_appearance(
    &f.scene,
    f.visual,
    5.0,
    custom(mesh),
    Default::default(),
    &f.library,
    false,
  )
  .unwrap();
  apply_comet_appearance(
    &f.scene,
    f.visual,
    5.0,
    CometAppearanceWiring::default(),
    Default::default(),
    &f.library,
    false,
  )
  .unwrap();
  let m = mesh_of(&f);
  assert_eq!(m.asset_path, DEFAULT_COMET_ASSET_PATH);
  assert!(
    m.is_visible,
    "visibility (comet committed) survives appearance changes"
  );
  let expected = generate_uv_sphere(5.0, 16, 16, 1.0, false);
  for (a, b) in m.mesh.vertices.iter().zip(&expected.vertices) {
    assert_eq!(a.position, b.position);
  }
  assert_eq!(transform_of(&f), TransformComponent::default());
}

#[test]
fn custom_mode_without_mesh_keeps_the_default_sphere() {
  let f = fixture("custom_nomesh");
  let w = CometAppearanceWiring {
    mode: CometDisplayMode::Custom,
    mesh: None,
    textures: [None; 4],
  };
  apply_comet_appearance(
    &f.scene,
    f.visual,
    2.0,
    w,
    Default::default(),
    &f.library,
    false,
  )
  .unwrap();
  assert_eq!(mesh_of(&f).asset_path, DEFAULT_COMET_ASSET_PATH);
  assert_eq!(transform_of(&f), TransformComponent::default());
}

#[test]
fn visual_entity_is_excluded_from_scene_dumps() {
  let f = fixture("dump");
  let entities = crate::simulation_api::scene_dump::serialize_scene(&f.scene);
  let visual_ffi = f.visual.as_ffi();
  assert!(entities.iter().all(|e| e.ffi_id != visual_ffi));
  assert!(entities.iter().any(|e| e.ffi_id == f.body.as_ffi()));
}

/// The renderer places micro-frame meshes with `ancestor_depth_layer` +
/// `get_relative_transform_f64(mesh, frame)`: the visual child must land in the comet's layer
/// and compose the (almanac-driven) body transform with the appearance offset.
#[test]
fn renderer_transform_path_composes_body_and_visual_offset() {
  use crate::scene::{ReferenceFrameComponent, ReferenceFrameType};
  use aethervk_oshal_rlib::math::{matrix::MatrixVectorMul, vector::vec3f64::Vec3f64};

  let mut f = fixture("render_path");
  let frame = f.scene.spawn_entity("Comet_subtree");
  f.scene
    .add_component(
      frame,
      ReferenceFrameComponent {
        frame_type: ReferenceFrameType::Micro,
        scale: 1.0 / 149_597_870.7,
        soi_radius: 1.0,
        depth_layer: 1,
      },
    )
    .unwrap();
  f.scene.add_component(frame, TransformComponent::default()).unwrap();
  f.scene.set_parent(f.body, Some(frame));
  // body as the reposition step would leave it: km residual + spin
  let body_rot = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), 90f32.to_radians());
  let _ = f.scene.with_component_mut(f.body, |t: &mut TransformComponent| {
    t.position = Vec3f32::from_components(100.0, 0.0, 0.0);
    t.rotation = body_rot;
  });

  let mesh = import_offset_sphere(&mut f, [5.0, 5.0, 0.0], 4.0);
  let offset = CometVisualOffset {
    yaw_pitch_roll_deg: [0.0; 3],
    translation_radii: [1.0, 0.0, 0.0],
  };
  apply_comet_appearance(
    &f.scene,
    f.visual,
    2.0,
    custom(mesh),
    offset,
    &f.library,
    false,
  )
  .unwrap();

  assert_eq!(f.scene.ancestor_depth_layer(f.visual), 1);
  let rel = f.scene.get_relative_transform_f64(f.visual, frame).unwrap();
  // scale: R / bounding radius
  let s = rel.scale.x();
  let br = f.library.mesh(mesh).unwrap().bounding_radius;
  assert!((s - 2.0 / br).abs() < 1e-5, "scale {s}");
  // bounding-sphere centre → body position + body_rot · (1 R along +X) = (100, 2, 0)
  let c = f.library.mesh(mesh).unwrap().bounding_center;
  let m = rel.to_mat4_f64();
  let p = m.mul_vector(
    aethervk_oshal_rlib::math::vector::vec4f64::Vec4f64::from_components(
      c[0] as f64,
      c[1] as f64,
      c[2] as f64,
      1.0,
    ),
  );
  let expected = Vec3f64::from_components(100.0, 2.0, 0.0);
  assert!(
    (p[0] - expected.x()).abs() < 1e-3 && (p[1] - expected.y()).abs() < 1e-3 && p[2].abs() < 1e-3,
    "centre at ({}, {}, {})",
    p[0],
    p[1],
    p[2]
  );
}

#[test]
fn unwiring_the_displayed_mesh_ejects_to_the_sphere_and_keeps_placement() {
  let mut f = fixture("unwire_mesh");
  let mesh = import_offset_sphere(&mut f, [0.0; 3], 1.0);
  let albedo = import_png(&mut f, "a.png", [10, 20, 30, 255]);
  let mut w = custom(mesh);
  w.textures[TextureChannel::Albedo as usize] = Some(albedo);
  let offset = CometVisualOffset {
    yaw_pitch_roll_deg: [5.0, 0.0, 0.0],
    translation_radii: [0.2, 0.0, 0.0],
  };
  apply_comet_appearance(&f.scene, f.visual, 3.0, w, offset, &f.library, false).unwrap();

  // unrelated asset: nothing happens
  let other = import_png(&mut f, "b.png", [1, 1, 1, 255]);
  assert_eq!(
    unwire_asset(&f.scene, f.visual, 3.0, other, &f.library),
    Ok(false)
  );

  assert_eq!(
    unwire_asset(&f.scene, f.visual, 3.0, mesh, &f.library),
    Ok(true)
  );
  let a = f.scene.with_component(f.visual, |a: &CometAppearanceComponent| *a).unwrap();
  assert_eq!(a.wiring, CometAppearanceWiring::default());
  assert_eq!(a.offset, offset, "placement survives the eject");
  assert_eq!(mesh_of(&f).asset_path, DEFAULT_COMET_ASSET_PATH);
  assert_eq!(transform_of(&f), TransformComponent::default());
  let expected = generate_uv_sphere(3.0, 16, 16, 1.0, false);
  assert_eq!(
    mesh_of(&f).mesh.vertices[5].position,
    expected.vertices[5].position
  );
}

#[test]
fn unwiring_a_texture_only_clears_its_channel() {
  let mut f = fixture("unwire_tex");
  let mesh = import_offset_sphere(&mut f, [0.0; 3], 1.0);
  let albedo = import_png(&mut f, "a.png", [10, 20, 30, 255]);
  let mut w = custom(mesh);
  w.textures[TextureChannel::Albedo as usize] = Some(albedo);
  w.textures[TextureChannel::Ao as usize] = Some(albedo);
  apply_comet_appearance(
    &f.scene,
    f.visual,
    2.0,
    w,
    Default::default(),
    &f.library,
    false,
  )
  .unwrap();

  assert_eq!(
    unwire_asset(&f.scene, f.visual, 2.0, albedo, &f.library),
    Ok(true)
  );
  let a = f.scene.with_component(f.visual, |a: &CometAppearanceComponent| *a).unwrap();
  assert_eq!(a.wiring.mesh, Some(mesh));
  assert_eq!(a.wiring.textures, [None; 4]);
  assert!(mesh_of(&f).mesh.albedo_map.is_none());
  assert!(mesh_of(&f).asset_path.starts_with("asset:"));
}
