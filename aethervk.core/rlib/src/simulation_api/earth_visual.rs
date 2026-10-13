//! The Earth as a body you can see: a textured globe, a dashed polar axis marked "N" / "S" and an
//! "Earth" indicator, all drawn only by cameras in the UpZenith mode
//! ([`RENDER_LAYER_UPZENITH_ONLY`], see [`crate::scene::CameraRenderMaskComponent`]).
//!
//! Everything hangs off `Earth_visual`, a child of `Earth_body`, so it inherits the body's
//! rotation (`AlmanacPlanet`: ITRF93 → ecliptic J2000, body +Z = north pole, +X = Greenwich).
//!
//! The globe has its **own** procedural sphere: the comet's default sphere is resized in place
//! (`update_uv_sphere_radius_in_place` + `enqueue_mesh_position_update(mesh_id)`), so sharing it
//! would make every comet radius change resize the Earth too.
//!
//! Texture alignment. `generate_uv_sphere` puts `u = φ / 2π` with `φ` measured from mesh +X towards
//! +Y and `v = θ / π` from the +Z pole. `Earth.jpg` is equirectangular with `u = 0` at 180° W and
//! `v = 0` at the north pole. The visual is rotated by `Rz(180°)` relative to the body, so a mesh
//! vertex at `φ` lies at body longitude `λ = φ + 180°` and samples
//! `u = φ / 360° = (λ − 180°) / 360° ≡ (λ + 180°) / 360° (mod 1)`, which is the map's column for `λ`.
//! `Rz(180°)` leaves ±Z alone, so the axis and the labels need no correction.

use crate::{
  scene::{
    Component, EntityId, RENDER_LAYER_UPZENITH_ONLY, RenderLayerComponent, Scene,
    StaticMeshComponent, TransformComponent,
    referential_indicator::ReferentialIndicatorComponent,
    text::FontAtlas,
  },
  simulation::{
    comet::{Comet, generate_uv_sphere, load_texture_from_file},
    constants::PlanetRadii,
  },
};
use aethervk_oshal_rlib::math::vector::vec4::Quat;
use alloc::sync::Arc;

/// `StaticMeshComponent::asset_path` of the Earth globe.
pub const EARTH_ASSET_PATH: &str = "__earth__";
/// `Earth.jpg` relative to the asset directory.
pub const EARTH_ALBEDO_RELATIVE_PATH: &str = "planets/textures/Earth.jpg";
/// Globe tessellation: 2° per latitude and longitude step.
pub const EARTH_LAT_SEGMENTS: u32 = 90;
pub const EARTH_LON_SEGMENTS: u32 = 180;
/// The polar axis spans `±AXIS_LENGTH_FACTOR · R` from the centre, the labels sit at its ends.
pub const AXIS_LENGTH_FACTOR: f32 = 1.5;
/// Dash periods (dash + gap) along the whole axis. The shader lays `AXIS_DASH_COUNT + ½` periods
/// so that both ends (the poles) fall on a dash: `AXIS_DASH_COUNT + 1` dashes.
pub const AXIS_DASH_COUNT: f32 = 24.0;

/// Marks `Earth_visual`. Like the comet visual it is aesthetic: never written into scene dumps
/// (respawned with the scene).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EarthVisualComponent;
impl Component for EarthVisualComponent {}

/// A dashed line along the entity's local Z axis from `−length_factor · radius_km` to
/// `+length_factor · radius_km`. Rendered by the sphere-gizmo pipeline in its "polar axis" style.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolarAxisGizmoComponent {
  pub radius_km: f32,
  pub length_factor: f32,
  pub dash_count: f32,
}
impl Component for PolarAxisGizmoComponent {}

/// Screen-space "N" / "S" labels at the ends of the entity's local ±Z axis
/// (`±length_factor · radius_km`), pushed outward along the projected axis.
#[derive(Clone, Debug)]
pub struct PoleLabelsComponent {
  pub radius_km: f32,
  pub length_factor: f32,
  pub text_color: [f32; 4],
  pub font_atlas: Arc<FontAtlas>,
  pub font_hash: u64,
}
impl Component for PoleLabelsComponent {}

/// Entities spawned by [`spawn_earth_visual`].
#[derive(Clone, Copy, Debug)]
pub struct EarthVisualEntities {
  pub visual: EntityId,
  pub indicator: Option<EntityId>,
}

/// Registers the module's components (called from `Scene::register_all_crate_components`).
pub fn register_components(scene: &Scene) {
  let transform = [core::any::TypeId::of::<TransformComponent>()];
  scene.register_component::<EarthVisualComponent>(&transform);
  scene.register_component::<PolarAxisGizmoComponent>(&transform);
  scene.register_component::<PoleLabelsComponent>(&transform);
}

/// Rotation of the visual relative to `Earth_body` (see the module docs): `Rz(180°)`, exact.
pub fn visual_rotation() -> Quat {
  // xyzw: sin(90°)·Z, cos(90°)
  Quat::from_components(0.0, 0.0, 1.0, 0.0)
}

/// The globe mesh, radius = Earth's equatorial radius in km, albedo = `Earth.jpg` (when found).
pub fn earth_mesh(asset_dir: Option<&str>) -> Comet {
  let mut mesh = generate_uv_sphere(
    PlanetRadii::EARTH,
    EARTH_LAT_SEGMENTS,
    EARTH_LON_SEGMENTS,
    1.0,
    false,
  );
  if let Some(dir) = asset_dir {
    let path = alloc::format!("{}/{}", dir, EARTH_ALBEDO_RELATIVE_PATH);
    match load_texture_from_file(&path) {
      Ok(t) => mesh.albedo_map = Some(t),
      Err(e) => aethervk_oshal_rlib::log!("Earth albedo '{}' not loaded: {:?}", path, e),
    }
  }
  mesh
}

/// Spawns `Earth_visual` under `body` (globe, axis, labels) and `earth_indicator` under `root`.
/// All of them are in [`RENDER_LAYER_UPZENITH_ONLY`]. Labels and indicator need `label_font`.
pub fn spawn_earth_visual(
  scene: &Scene,
  root: EntityId,
  body: EntityId,
  asset_dir: Option<&str>,
  label_font: Option<&(Arc<FontAtlas>, u64)>,
) -> EarthVisualEntities {
  let radius_km = PlanetRadii::EARTH;
  let visual = scene.spawn_entity("Earth_visual");
  scene.set_parent(visual, Some(body));
  let _ = scene.add_component(
    visual,
    TransformComponent {
      rotation: visual_rotation(),
      ..TransformComponent::default()
    },
  );
  let _ = scene.add_component(visual, EarthVisualComponent);
  let _ = scene.add_component(visual, RenderLayerComponent(RENDER_LAYER_UPZENITH_ONLY));
  let _ = scene.add_component(
    visual,
    StaticMeshComponent {
      asset_path: alloc::string::String::from(EARTH_ASSET_PATH),
      mesh: Arc::new(earth_mesh(asset_dir)),
      emissive_color: [0.0, 0.0, 0.0, 0.0],
      is_visible: true,
    },
  );
  let _ = scene.add_component(
    visual,
    PolarAxisGizmoComponent {
      radius_km,
      length_factor: AXIS_LENGTH_FACTOR,
      dash_count: AXIS_DASH_COUNT,
    },
  );

  let indicator = label_font.map(|(atlas, font_hash)| {
    let _ = scene.add_component(
      visual,
      PoleLabelsComponent {
        radius_km,
        length_factor: AXIS_LENGTH_FACTOR,
        text_color: [0.85, 0.9, 1.0, 1.0],
        font_atlas: Arc::clone(atlas),
        font_hash: *font_hash,
      },
    );

    let e = scene.spawn_entity("earth_indicator");
    scene.set_parent(e, Some(root));
    let _ = scene.add_component(e, TransformComponent::default());
    let _ = scene.add_component(
      e,
      ReferentialIndicatorComponent {
        target_entity: body,
        label: alloc::string::String::from("Earth"),
        text_color: [0.55, 0.8, 1.0, 1.0],
        desired_label_distance_km: 2_000_000.0,
        font_atlas: Arc::clone(atlas),
        font_hash: *font_hash,
      },
    );
    let _ = scene.add_component(e, RenderLayerComponent(RENDER_LAYER_UPZENITH_ONLY));
    e
  });

  EarthVisualEntities { visual, indicator }
}

/// Roots of the subtrees `camera` must not draw: entities whose [`RenderLayerComponent`] shares no
/// bit with the camera's [`crate::scene::CameraRenderMaskComponent`] (default
/// [`crate::scene::RENDER_LAYER_DEFAULT`]).
pub fn masked_roots(scene: &Scene, camera: EntityId) -> alloc::vec::Vec<EntityId> {
  let mask = camera_mask(scene, camera);
  scene
    .query1_res::<RenderLayerComponent, _, _>(|id, l| (l.0 & mask == 0).then_some(id))
    .into_iter()
    .map(|(id, _)| id)
    .collect()
}

/// The render layers `camera` draws.
pub fn camera_mask(scene: &Scene, camera: EntityId) -> u32 {
  scene
    .with_component(camera, |m: &crate::scene::CameraRenderMaskComponent| m.0)
    .unwrap_or(crate::scene::RENDER_LAYER_DEFAULT)
}

/// Body-local (km) ends of the polar axis, north first: `(0, 0, ±length_factor · radius_km)` in
/// the entity's frame.
pub fn pole_points_local(radius_km: f32, length_factor: f32) -> [[f64; 3]; 2] {
  let h = radius_km as f64 * length_factor as f64;
  [[0.0, 0.0, h], [0.0, 0.0, -h]]
}

/// Global (heliocentric AU, the frame `global_transform_f64` works in) positions of `entity`'s
/// origin and of its pole points `(0, 0, ±length_factor · radius_km)`: `[centre, north, south]`.
pub fn pole_points_global(
  scene: &Scene,
  entity: EntityId,
  radius_km: f32,
  length_factor: f32,
) -> Option<[aethervk_oshal_rlib::math::vector::vec3f64::DVec3; 3]> {
  use aethervk_oshal_rlib::math::{
    quaternion::Quaternion,
    vector::{Vector3, vec3f64::Vec3f64},
  };
  let g = scene.global_transform_f64(entity)?;
  let to_global = |p: [f64; 3]| {
    g.position
      + g.rotation.rotate_vector(Vec3f64::from_components(
        g.scale.x() as f64 * p[0],
        g.scale.y() as f64 * p[1],
        g.scale.z() as f64 * p[2],
      ))
  };
  let [n, s] = pole_points_local(radius_km, length_factor);
  Some([g.position, to_global(n), to_global(s)])
}

/// Places a label of `text_width_px` × `text_height_px` beyond `end_px`, outward along the
/// screen direction `centre_px → end_px` (`gap_px` clear of the end). Returns the label's top-left
/// corner (screen y grows downwards). When the axis is seen end-on the label is centred on the end.
pub fn label_top_left(
  centre_px: [f32; 2],
  end_px: [f32; 2],
  text_width_px: f32,
  text_height_px: f32,
  gap_px: f32,
) -> [f32; 2] {
  let (dx, dy) = (end_px[0] - centre_px[0], end_px[1] - centre_px[1]);
  let len = (dx * dx + dy * dy).sqrt();
  let (ux, uy) = if len > 1e-3 { (dx / len, dy / len) } else { (0.0, 0.0) };
  // distance from the label centre to its box edge along (ux, uy)
  let half_extent = (ux.abs() * text_width_px + uy.abs() * text_height_px) * 0.5;
  let cx = end_px[0] + ux * (gap_px + half_extent);
  let cy = end_px[1] + uy * (gap_px + half_extent);
  [cx - text_width_px * 0.5, cy - text_height_px * 0.5]
}

#[cfg(test)]
#[path = "earth_visual_tests.rs"]
mod tests;
