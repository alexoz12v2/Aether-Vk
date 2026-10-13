//! CPU tests of the Earth visual (`earth_visual`): spawn, texture alignment, per-camera render
//! layers, pole points, label placement, scene-dump exclusion. GPU tests: `test_earth_render.rs`.

use super::*;
use crate::{
  scene::{
    CameraRenderMaskComponent, HiddenComponent, RENDER_LAYER_DEFAULT, ReferenceFrameComponent,
    ReferenceFrameType, SphereGizmoComponent,
  },
  simulation::{
    comet::{Texture, TexelFormat},
    texture_cache::TextureCache,
  },
  simulation_api::comet_appearance::spawn_comet_visual,
};
use aethervk_oshal_rlib::math::{
  matrix::{SquareMatrix, mat4::Mat4x4f32},
  quaternion::Quaternion,
  vector::{Vector3, vec3::Vec3f32, vec3f64::Vec3f64},
};
use parking_lot::RwLock;

const AU_TO_KM: f32 = 149_597_870.7;

fn asset_dir() -> std::string::String {
  format!("{}/../../assets", env!("CARGO_MANIFEST_DIR"))
}

fn label_font() -> (Arc<FontAtlas>, u64) {
  let atlas =
    FontAtlas::from_path(&format!("{}/fonts/JetBrainsMono-Regular.ttf", asset_dir()), 64.0)
      .expect("font");
  let hash = atlas.hash_metadata();
  (Arc::new(atlas), hash)
}

struct Fixture {
  scene: Scene,
  root: EntityId,
  earth_subtree: EntityId,
  earth_body: EntityId,
  comet_visual: EntityId,
  earth: EarthVisualEntities,
}

/// Root → Earth_subtree (micro frame, AU) → Earth_body (km) → Earth_visual, plus a comet body with
/// its default visual and an `earth_indicator`, as `create_empty_scene` builds them.
fn fixture(with_font: bool) -> Fixture {
  let scene = Scene::new(Arc::new(RwLock::new(TextureCache::new("test_earth_visual"))));
  scene.register_all_crate_components();
  let root = scene.spawn_entity("Root");
  let _ = scene.add_component(root, TransformComponent::default());

  let earth_subtree = scene.spawn_entity("Earth_subtree");
  scene.set_parent(earth_subtree, Some(root));
  let _ = scene.add_component(earth_subtree, TransformComponent::default());
  let _ = scene.add_component(
    earth_subtree,
    ReferenceFrameComponent {
      frame_type: ReferenceFrameType::Micro,
      scale: 1.0 / AU_TO_KM,
      soi_radius: 1.0,
      depth_layer: 1,
    },
  );
  let earth_body = scene.spawn_entity("Earth_body");
  scene.set_parent(earth_body, Some(earth_subtree));
  let _ = scene.add_component(earth_body, TransformComponent::default());

  let comet_body = scene.spawn_entity("Comet_body");
  scene.set_parent(comet_body, Some(root));
  let _ = scene.add_component(comet_body, TransformComponent::default());
  let _ = scene.add_component(
    comet_body,
    SphereGizmoComponent {
      radius: 4.0,
      subdivisions: 4.0,
      local_frame: Mat4x4f32::identity(),
      is_visible: true,
    },
  );
  let comet_visual = spawn_comet_visual(&scene, comet_body);

  let dir = asset_dir();
  let font = with_font.then(label_font);
  let earth = spawn_earth_visual(&scene, root, earth_body, Some(&dir), font.as_ref());
  Fixture {
    scene,
    root,
    earth_subtree,
    earth_body,
    comet_visual,
    earth,
  }
}

fn mesh(scene: &Scene, e: EntityId) -> StaticMeshComponent {
  scene.with_component(e, |m: &StaticMeshComponent| m.clone()).unwrap()
}

/// Body-frame direction of a mesh-local vector under the visual's `Rz(180°)`.
fn to_body(v: [f32; 3]) -> [f64; 3] {
  let r = visual_rotation().rotate_vector(Vec3f32::from_components(v[0], v[1], v[2]));
  [r.x() as f64, r.y() as f64, r.z() as f64]
}

fn lat_lon_deg(d: [f64; 3]) -> (f64, f64) {
  let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
  ((d[2] / n).asin().to_degrees(), d[1].atan2(d[0]).to_degrees())
}

/// Bilinear-free nearest sample of an RGBA8 texture at `uv`.
fn sample(t: &Texture, uv: [f64; 2]) -> [u8; 4] {
  assert_eq!(t.format, TexelFormat::R8G8B8A8_UNORM);
  let x = ((uv[0].rem_euclid(1.0)) * t.width as f64).floor().min(t.width as f64 - 1.0) as usize;
  let y = (uv[1].clamp(0.0, 1.0) * t.height as f64).floor().min(t.height as f64 - 1.0) as usize;
  let i = (y * t.width as usize + x) * 4;
  [t.data[i], t.data[i + 1], t.data[i + 2], t.data[i + 3]]
}

/// The map column/row a geographic point must sample: equirectangular, u = 0 at 180° W.
fn expected_uv(lat_deg: f64, lon_deg: f64) -> [f64; 2] {
  [((lon_deg + 180.0) / 360.0).rem_euclid(1.0), (90.0 - lat_deg) / 180.0]
}

/// Mean of a `(2k+1)²` texel block around `uv` (robust to a single odd texel).
fn mean_rgb(t: &Texture, uv: [f64; 2], k: i32) -> [f64; 3] {
  let mut acc = [0.0; 3];
  let mut n = 0.0;
  for dy in -k..=k {
    for dx in -k..=k {
      let p = sample(
        t,
        [uv[0] + dx as f64 / t.width as f64, uv[1] + dy as f64 / t.height as f64],
      );
      for c in 0..3 {
        acc[c] += p[c] as f64;
      }
      n += 1.0;
    }
  }
  acc.map(|a| a / n)
}

// ── A-1 own mesh ──────────────────────────────────────────────────────────────────────────────

#[test]
fn earth_visual_is_a_child_of_the_body_with_its_own_mesh() {
  let f = fixture(true);
  assert_eq!(f.scene.get_parent(f.earth.visual), Some(f.earth_body));
  let earth = mesh(&f.scene, f.earth.visual);
  let comet = mesh(&f.scene, f.comet_visual);
  assert_ne!(earth.mesh.id, comet.mesh.id, "the Earth must not share the comet's mesh id");
  assert!(!Arc::ptr_eq(&earth.mesh, &comet.mesh));
  assert_eq!(earth.asset_path, EARTH_ASSET_PATH);
  // equatorial radius baked in, 2° tessellation
  let r_max = earth
    .mesh
    .vertices
    .iter()
    .map(|v| {
      let [x, y, z] = v.position;
      (x * x + y * y + z * z).sqrt()
    })
    .fold(0.0f32, f32::max);
  assert!((r_max - PlanetRadii::EARTH).abs() < 1e-2, "radius {r_max}");
  assert_eq!(
    earth.mesh.vertices.len() as u32,
    (EARTH_LAT_SEGMENTS + 1) * (EARTH_LON_SEGMENTS + 1)
  );
  let albedo = earth.mesh.albedo_map.as_ref().expect("Earth.jpg must load as the albedo map");
  assert_eq!((albedo.width, albedo.height), (1440, 720));
  assert!(earth.mesh.normal_map.is_none() && earth.mesh.roughness_map.is_none());
}

#[test]
fn comet_radius_changes_never_touch_the_earth_mesh() {
  // what `LogicCommand::UpdateCometNucleusRadius` does to the comet visual
  let f = fixture(false);
  let before = mesh(&f.scene, f.earth.visual);
  let before_positions: std::vec::Vec<[f32; 3]> =
    before.mesh.vertices.iter().map(|v| v.position).collect();
  for r in [0.5f32, 7.0, 25.0] {
    let _ = f.scene.with_component_mut(f.comet_visual, |m: &mut StaticMeshComponent| {
      let mut copy = (*m.mesh).clone();
      crate::simulation::comet::update_uv_sphere_radius_in_place(&mut copy, r);
      m.mesh = Arc::new(copy);
    });
  }
  let after = mesh(&f.scene, f.earth.visual);
  assert!(Arc::ptr_eq(&before.mesh, &after.mesh));
  let after_positions: std::vec::Vec<[f32; 3]> =
    after.mesh.vertices.iter().map(|v| v.position).collect();
  assert_eq!(before_positions, after_positions);
}

// ── A-4 texture alignment ─────────────────────────────────────────────────────────────────────

/// Proof in the module docs; here it is checked on every vertex of the globe: the body-frame
/// direction of each vertex (after the visual's `Rz(180°)`) has the latitude/longitude whose map
/// texel its uv addresses. Seam vertices (u = 0 and u = 1 at the same place) and the poles (any
/// longitude) are compared modulo 1 and on v only.
#[test]
fn every_globe_vertex_samples_its_own_geographic_texel() {
  let f = fixture(false);
  let m = mesh(&f.scene, f.earth.visual);
  for v in &m.mesh.vertices {
    let (lat, lon) = lat_lon_deg(to_body(v.normal));
    let want = expected_uv(lat, lon);
    let du = (v.uv[0] as f64 - want[0]).rem_euclid(1.0);
    let du = du.min(1.0 - du);
    let dv = (v.uv[1] as f64 - want[1]).abs();
    assert!(dv < 1e-5, "v of lat {lat:.3}: {} vs {}", v.uv[1], want[1]);
    if lat.abs() < 89.9 {
      assert!(du < 1e-5, "u of (lat {lat:.3}, lon {lon:.3}): {} vs {}", v.uv[0], want[0]);
    }
  }
}

#[test]
fn reference_points_map_to_their_texels() {
  // (lat 0, lon 0) → (0.5, 0.5); (45 N, 90 E) → (0.75, 0.25); (0, 180) → u 0 ≡ 1
  assert_eq!(expected_uv(0.0, 0.0), [0.5, 0.5]);
  assert_eq!(expected_uv(45.0, 90.0), [0.75, 0.25]);
  assert_eq!(expected_uv(0.0, 180.0)[0], 0.0);
  // and the globe vertex nearest each direction carries that uv
  let f = fixture(false);
  let m = mesh(&f.scene, f.earth.visual);
  for (lat, lon, want) in [(0.0, 0.0, [0.5, 0.5]), (45.0, 90.0, [0.75, 0.25])] {
    let v = m
      .mesh
      .vertices
      .iter()
      .min_by(|a, b| {
        let d = |v: &crate::simulation::comet::Vertex| {
          let (la, lo) = lat_lon_deg(to_body(v.normal));
          (la - lat).powi(2) + (lo - lon).powi(2)
        };
        d(a).partial_cmp(&d(b)).unwrap()
      })
      .unwrap();
    // within half a grid step (2° = 1/180 in u, 1/90 in v) of the exact texel
    let (du, dv) = ((v.uv[0] as f64 - want[0]).abs(), (v.uv[1] as f64 - want[1]).abs());
    assert!(du <= 0.5 / 180.0 + 1e-6 && dv <= 0.5 / 90.0 + 1e-6, "({lat}, {lon}): {:?}", v.uv);
  }
}

#[test]
fn earth_map_content_matches_geography() {
  let f = fixture(false);
  let m = mesh(&f.scene, f.earth.visual);
  let t = m.mesh.albedo_map.as_ref().unwrap();
  // Sahara (23° N, 13° E): sand, red over blue
  let sahara = mean_rgb(t, expected_uv(23.0, 13.0), 3);
  assert!(sahara[0] > sahara[2] + 20.0, "Sahara {sahara:?}");
  // central Pacific (0°, 160° W): ocean, blue over red
  let pacific = mean_rgb(t, expected_uv(0.0, -160.0), 3);
  assert!(pacific[2] > pacific[0] + 20.0, "Pacific {pacific:?}");
  // Gulf of Guinea (0°, 0°): ocean
  let guinea = mean_rgb(t, expected_uv(0.0, 0.0), 2);
  assert!(guinea[2] > guinea[0], "Gulf of Guinea {guinea:?}");
  // a mirrored map (east/west swapped) would put the Pacific sample on Africa (0°, 160° E is
  // the Pacific too, so check the Sahara's mirror instead: 23° N, 13° W is the Atlantic coast)
  let mirror = mean_rgb(t, expected_uv(23.0, -13.0 - 20.0), 3); // 23 N, 33 W: open Atlantic
  assert!(mirror[2] > mirror[0], "Atlantic {mirror:?}");
}

// ── A-2 per-camera render layers ──────────────────────────────────────────────────────────────

#[test]
fn render_layers_hide_the_earth_from_non_upzenith_cameras_only() {
  let f = fixture(true);
  let indicator = f.earth.indicator.expect("indicator needs the font");
  let cam_default = f.scene.spawn_entity("cam_default");
  let cam_up = f.scene.spawn_entity("cam_up");
  let _ = f.scene.add_component(
    cam_up,
    CameraRenderMaskComponent(RENDER_LAYER_DEFAULT | RENDER_LAYER_UPZENITH_ONLY),
  );
  let cam_explicit = f.scene.spawn_entity("cam_explicit");
  let _ = f.scene.add_component(cam_explicit, CameraRenderMaskComponent(RENDER_LAYER_DEFAULT));

  let mut masked = masked_roots(&f.scene, cam_default);
  masked.sort();
  let mut want = alloc::vec![f.earth.visual, indicator];
  want.sort();
  assert_eq!(masked, want);
  assert_eq!(masked_roots(&f.scene, cam_explicit).len(), 2);
  assert!(masked_roots(&f.scene, cam_up).is_empty());
  assert_eq!(camera_mask(&f.scene, cam_default), RENDER_LAYER_DEFAULT);

  // nothing else is layered: the bodies, frames and the comet stay drawn for every camera
  for e in [f.root, f.earth_subtree, f.earth_body, f.comet_visual] {
    assert!(!masked.contains(&e));
  }
  // and the layer mechanism is independent of the scene-wide HiddenComponent
  assert!(f.scene.with_component(f.earth.visual, |_: &HiddenComponent| ()).is_none());
}

#[test]
fn the_axis_and_labels_live_on_the_layered_visual() {
  let f = fixture(true);
  let axis = f
    .scene
    .with_component(f.earth.visual, |a: &PolarAxisGizmoComponent| *a)
    .expect("axis");
  assert_eq!(axis.length_factor, AXIS_LENGTH_FACTOR);
  assert_eq!(axis.radius_km, PlanetRadii::EARTH);
  assert!(axis.dash_count >= 2.0);
  assert!(f.scene.with_component(f.earth.visual, |_: &PoleLabelsComponent| ()).is_some());
  assert_eq!(
    f.scene.with_component(f.earth.visual, |l: &RenderLayerComponent| l.0),
    Some(RENDER_LAYER_UPZENITH_ONLY)
  );
  // without a font there are no labels and no indicator, the globe and axis still exist
  let g = fixture(false);
  assert!(g.earth.indicator.is_none());
  assert!(g.scene.with_component(g.earth.visual, |_: &PoleLabelsComponent| ()).is_none());
  assert!(g.scene.with_component(g.earth.visual, |_: &PolarAxisGizmoComponent| ()).is_some());
}

// ── A-5 pole points ───────────────────────────────────────────────────────────────────────────

#[test]
fn pole_points_follow_the_body_rotation_and_frame_scale() {
  let f = fixture(false);
  // subtree at (0.3, −0.9, 0.1) AU, body 5000 km off the frame origin, tilted 23.44° about X
  let subtree_pos = Vec3f32::from_components(0.3, -0.9, 0.1);
  let _ = f.scene.with_component_mut(f.earth_subtree, |t: &mut TransformComponent| {
    t.position = subtree_pos
  });
  let tilt = 23.44f32.to_radians();
  let q = Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), tilt);
  let body_km = Vec3f32::from_components(5000.0, 0.0, 0.0);
  let _ = f.scene.with_component_mut(f.earth_body, |t: &mut TransformComponent| {
    t.position = body_km;
    t.rotation = q;
  });

  let [c, n, s] =
    pole_points_global(&f.scene, f.earth.visual, PlanetRadii::EARTH, AXIS_LENGTH_FACTOR).unwrap();
  let au = 1.0 / AU_TO_KM as f64;
  let want_c = Vec3f64::from_components(
    subtree_pos.x() as f64 + 5000.0 * au,
    subtree_pos.y() as f64,
    subtree_pos.z() as f64,
  );
  let h_au = PlanetRadii::EARTH as f64 * AXIS_LENGTH_FACTOR as f64 * au;
  // Rx(tilt)·Z = (0, −sin, cos)
  let dir = [0.0, -(tilt as f64).sin(), (tilt as f64).cos()];
  let close = |a: Vec3f64, b: [f64; 3], tol: f64| {
    (a.x() - b[0]).abs() < tol && (a.y() - b[1]).abs() < tol && (a.z() - b[2]).abs() < tol
  };
  assert!(close(c, [want_c.x(), want_c.y(), want_c.z()], 1e-12));
  let tol = h_au * 1e-6;
  assert!(close(
    n,
    [want_c.x() + h_au * dir[0], want_c.y() + h_au * dir[1], want_c.z() + h_au * dir[2]],
    tol
  ));
  assert!(close(
    s,
    [want_c.x() - h_au * dir[0], want_c.y() - h_au * dir[1], want_c.z() - h_au * dir[2]],
    tol
  ));
}

/// The "N" end is the Earth's true north pole: with `Earth_body` posed by `AlmanacPlanet`
/// (ITRF93 → ecliptic J2000 from `earth_latest_high_prec.bpc`), the north point's direction from
/// the centre equals ANISE's ITRF93 Z axis in ecliptic J2000, at two epochs a year apart.
#[test]
fn north_label_anchor_is_the_itrf93_pole() {
  use crate::simulation::almanac::{AlmanacPackedData, SUN_ECLIPJ2000};
  let mut almanac = AlmanacPackedData::default();
  almanac.load_almanac("../../assets/earth_latest_high_prec.bpc").unwrap();
  let planet = crate::scene::AlmanacPlanet::new(399);
  let f = fixture(false);
  for epoch in [
    hifitime::Epoch::from_gregorian_utc_at_midnight(2025, 1, 1),
    hifitime::Epoch::from_gregorian_utc_at_midnight(2026, 1, 1),
  ] {
    let q = planet.rotation_at(epoch, &almanac, None);
    let _ = f.scene.with_component_mut(f.earth_body, |t: &mut TransformComponent| t.rotation = q);
    let [c, n, _] =
      pole_points_global(&f.scene, f.earth.visual, PlanetRadii::EARTH, AXIS_LENGTH_FACTOR)
        .unwrap();
    let d = n - c;
    let len = (d.x() * d.x() + d.y() * d.y() + d.z() * d.z()).sqrt();
    let got = [d.x() / len, d.y() / len, d.z() / len];
    let dcm = almanac
      .almanac
      .rotate(anise::constants::frames::EARTH_ITRF93, SUN_ECLIPJ2000, epoch)
      .unwrap();
    let z = dcm.rot_mat * anise::math::Vector3::new(0.0, 0.0, 1.0);
    let ang = (got[0] * z[0] + got[1] * z[1] + got[2] * z[2]).clamp(-1.0, 1.0).acos();
    // the body rotation is an f32 quaternion: ~1e-7 rad
    assert!(ang < 2e-6, "N anchor {ang:e} rad off the ITRF93 pole at {epoch}");
    // the ecliptic north pole is ~23.44° from the Earth's
    let obliquity = z[2].clamp(-1.0, 1.0).acos().to_degrees();
    assert!((obliquity - 23.44).abs() < 0.1, "obliquity {obliquity}");
  }
}

// ── A-13 label placement ──────────────────────────────────────────────────────────────────────

#[test]
fn labels_sit_beyond_the_axis_ends() {
  // axis pointing straight up on screen (screen y grows downwards)
  let tl = label_top_left([100.0, 100.0], [100.0, 40.0], 10.0, 16.0, 4.0);
  // centred horizontally on the end, fully above it with the gap
  assert!((tl[0] + 5.0 - 100.0).abs() < 1e-4);
  assert!((tl[1] + 16.0 - (40.0 - 4.0)).abs() < 1e-4, "{tl:?}");
  // pointing right: the label starts right of the end
  let tl = label_top_left([0.0, 0.0], [50.0, 0.0], 10.0, 16.0, 4.0);
  assert!((tl[0] - 54.0).abs() < 1e-4 && (tl[1] + 8.0).abs() < 1e-4, "{tl:?}");
  // seen end-on: centred on the end
  let tl = label_top_left([10.0, 10.0], [10.0, 10.0], 10.0, 16.0, 4.0);
  assert_eq!(tl, [5.0, 2.0]);
}

// ── A-7 scene dump ────────────────────────────────────────────────────────────────────────────

#[test]
fn scene_dumps_skip_the_earth_visual() {
  let f = fixture(true);
  let dumped = crate::simulation_api::scene_dump::serialize_scene(&f.scene);
  let ids: std::vec::Vec<u64> = dumped.iter().map(|e| e.ffi_id).collect();
  let ffi = |e: EntityId| slotmap::Key::data(&e).as_ffi();
  assert!(!ids.contains(&ffi(f.earth.visual)), "Earth_visual must not be dumped");
  assert!(ids.contains(&ffi(f.earth_body)), "Earth_body is still dumped");
  // restoring that dump keeps exactly one globe under the body
  let mut scene = f.scene;
  let cache = crate::scene::AssetCache::<crate::simulation::comet::Comet>::new();
  let _ = crate::simulation_api::scene_dump::deserialize_scene(&mut scene, &dumped, &cache);
  let globes = scene.query1_res::<EarthVisualComponent, _, _>(|id, _| Some(id));
  assert_eq!(globes.len(), 1);
  assert_eq!(scene.get_parent(globes[0].0), Some(f.earth_body));
}

// ── A-6 colour space of the upload ────────────────────────────────────────────────────────────

#[test]
fn albedo_formats_have_srgb_twins() {
  use ash::vk::Format;
  for (f, want) in [
    (TexelFormat::R8_UNORM, Format::R8_SRGB),
    (TexelFormat::R8G8_UNORM, Format::R8G8_SRGB),
    (TexelFormat::R8G8B8_UNORM, Format::R8G8B8_SRGB),
    (TexelFormat::R8G8B8A8_UNORM, Format::R8G8B8A8_SRGB),
    (TexelFormat::BC7_UNORM_BLOCK, Format::BC7_SRGB_BLOCK),
    (TexelFormat::ETC2_R8G8B8_UNORM_BLOCK, Format::ETC2_R8G8B8_SRGB_BLOCK),
    (TexelFormat::ASTC_4x4_UNORM_BLOCK, Format::ASTC_4X4_SRGB_BLOCK),
  ] {
    assert_eq!(f.to_vk_format_srgb(), want, "{f:?}");
    assert_ne!(f.to_vk_format(), want, "{f:?} linear upload must stay UNORM");
  }
  assert_eq!(TexelFormat::Undefined.to_vk_format_srgb(), Format::UNDEFINED);
}
