//! Comet appearance: what the nucleus *looks like*, decoupled from what it *is*.
//!
//! The comet body entity (`Comet_body`) carries physics/ephemeris state (Transform driven by
//! the almanac, `BodyRotationalModel`, `SphereGizmoComponent`, `CometMarkerComponent`). Its
//! rendering lives on a dedicated child, `Comet_visual`, tagged with [`CometVisualComponent`]
//! and holding the [`StaticMeshComponent`] plus a [`CometAppearanceComponent`]:
//!
//! - [`CometDisplayMode::Default`]: the procedural UV sphere, radius baked into the vertices
//!   (identity visual transform) — exactly the pre-existing behaviour;
//! - [`CometDisplayMode::Custom`]: an imported mesh from the
//!   [`AssetLibrary`](crate::simulation::asset_library::AssetLibrary) with up to four wired
//!   texture channels. The mesh is recentred on its bounding sphere and uniformly scaled so the
//!   bounding sphere radius equals the nucleus radius; a user rotation and a translation (in
//!   nucleus radii) are applied in the body frame on top.
//!
//! Appearance is aesthetic only: it is not part of the scene-dump compatibility hash and the
//! visual entity is excluded from scene dumps, so a snapshot restore never rolls it back.

use alloc::sync::Arc;

use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{Vector, Vector3, vec3::Vec3f32, vec4::Quat},
};

use crate::{
  scene::{Component, EntityId, Scene, StaticMeshComponent, TransformComponent},
  simulation::{
    asset_library::{AssetId, AssetLibrary, TextureChannel},
    comet::{Comet, generate_uv_sphere},
  },
  types::{EngineError, EngineResult},
};

/// `StaticMeshComponent::asset_path` of the procedural default nucleus.
pub const DEFAULT_COMET_ASSET_PATH: &str = "__default_comet__";
/// Tessellation of the procedural default nucleus.
pub const DEFAULT_SPHERE_SEGMENTS: u32 = 16;
/// Nucleus radius used before the user configures one.
pub const DEFAULT_NUCLEUS_RADIUS_KM: f32 = 2.0;

/// Marker for the child entity rendering the comet nucleus. Such an entity is not a jet (it
/// survives `CleanupComet`) and is skipped by scene dumps.
#[derive(Debug, Clone, Copy, Default)]
pub struct CometVisualComponent;

impl Component for CometVisualComponent {}

/// How the comet nucleus is displayed.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CometDisplayMode {
  #[default]
  Default = 0,
  Custom = 1,
}

impl CometDisplayMode {
  pub fn from_u32(v: u32) -> Option<Self> {
    match v {
      0 => Some(Self::Default),
      1 => Some(Self::Custom),
      _ => None,
    }
  }
}

/// Mesh + texture wiring of the nucleus. Changing any of these rebuilds GPU resources, hence
/// they may only change while the simulation is paused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CometAppearanceWiring {
  pub mode: CometDisplayMode,
  pub mesh: Option<AssetId>,
  /// Indexed by [`TextureChannel`].
  pub textures: [Option<AssetId>; 4],
}

/// Placement of a custom mesh in the comet body frame. Cheap to change at any time.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CometVisualOffset {
  /// Intrinsic Z (yaw) → Y (pitch) → X (roll) rotation, degrees.
  pub yaw_pitch_roll_deg: [f32; 3],
  /// Translation in units of nucleus radius.
  pub translation_radii: [f32; 3],
}

impl CometVisualOffset {
  pub fn rotation(&self) -> Quat {
    let [yaw, pitch, roll] = self.yaw_pitch_roll_deg;
    let qz = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), yaw.to_radians());
    let qy = Quat::from_axis_angle(Vec3f32::from_components(0.0, 1.0, 0.0), pitch.to_radians());
    let qx = Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), roll.to_radians());
    qz * qy * qx
  }
}

/// Current appearance, stored on the visual entity.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CometAppearanceComponent {
  pub wiring: CometAppearanceWiring,
  pub offset: CometVisualOffset,
  /// Bounding sphere of the displayed custom mesh (mesh space); unused in default mode.
  pub bounding_center: [f32; 3],
  pub bounding_radius: f32,
}

impl Component for CometAppearanceComponent {}

impl CometAppearanceComponent {
  /// Whether an imported mesh is what is actually displayed.
  pub fn shows_custom_mesh(&self) -> bool {
    self.wiring.mode == CometDisplayMode::Custom && self.wiring.mesh.is_some()
  }
}

/// Builds the procedural default nucleus at `radius_km` (positions = unit normal × radius).
pub fn default_comet_mesh(radius_km: f32) -> StaticMeshComponent {
  StaticMeshComponent {
    asset_path: alloc::string::String::from(DEFAULT_COMET_ASSET_PATH),
    mesh: Arc::new(generate_uv_sphere(
      radius_km,
      DEFAULT_SPHERE_SEGMENTS,
      DEFAULT_SPHERE_SEGMENTS,
      1.0,
      false,
    )),
    emissive_color: [0.0, 0.0, 0.0, 0.0],
    is_visible: false,
  }
}

/// Spawns `Comet_visual` under `body` with the default sphere at the default radius.
pub fn spawn_comet_visual(scene: &Scene, body: EntityId) -> EntityId {
  let visual = scene.spawn_entity("Comet_visual");
  scene.set_parent(visual, Some(body));
  let _ = scene.add_component(visual, TransformComponent::default());
  let _ = scene.add_component(visual, CometVisualComponent);
  let _ = scene.add_component(visual, CometAppearanceComponent::default());
  let _ = scene.add_component(visual, default_comet_mesh(DEFAULT_NUCLEUS_RADIUS_KM));
  visual
}

/// Nucleus radius as configured on the body (the sphere gizmo is kept at 2× radius).
pub fn nucleus_radius_km(scene: &Scene, body: EntityId) -> f32 {
  scene
    .with_component(body, |g: &crate::scene::SphereGizmoComponent| {
      g.radius * 0.5
    })
    .filter(|r| *r > 0.0)
    .unwrap_or(DEFAULT_NUCLEUS_RADIUS_KM)
}

/// Visual transform (relative to the body) placing a mesh with bounding sphere
/// (`center`, `bounding_radius`) so that the sphere has radius `radius_km`, is centred on the
/// body origin moved by `offset.translation_radii · radius_km`, and is rotated by the offset.
///
/// A vertex `v` ends up at `q·(s·(v − c)) + t·R = s·q·v + (−s·q·c + t·R)`.
pub fn custom_visual_transform(
  center: [f32; 3],
  bounding_radius: f32,
  radius_km: f32,
  offset: &CometVisualOffset,
) -> TransformComponent {
  let s = if bounding_radius > 0.0 {
    radius_km / bounding_radius
  } else {
    1.0
  };
  let q = offset.rotation();
  let c = Vec3f32::from_components(center[0], center[1], center[2]);
  let t = offset.translation_radii;
  let position = q.rotate_vector(c * (-s)) + Vec3f32::from_components(t[0], t[1], t[2]) * radius_km;
  TransformComponent {
    position,
    rotation: q,
    scale: Vec3f32::from_components(s, s, s),
  }
}

/// Reasons [`apply_comet_appearance`] can refuse a change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppearanceError {
  /// Mode / mesh / texture wiring changes require a paused simulation.
  SimulationRunning,
  UnknownMesh,
  UnknownTexture,
  NoVisualEntity,
}

impl From<AppearanceError> for EngineError {
  fn from(e: AppearanceError) -> Self {
    EngineError::InvalidOperation(match e {
      AppearanceError::SimulationRunning => {
        "CometAppearance: mesh/texture wiring can only change while paused"
      }
      AppearanceError::UnknownMesh => "CometAppearance: unknown mesh asset",
      AppearanceError::UnknownTexture => "CometAppearance: unknown texture asset",
      AppearanceError::NoVisualEntity => "CometAppearance: comet has no visual entity",
    })
  }
}

/// Applies a new wiring and/or offset to `visual`.
///
/// - The mesh is rebuilt only when the wiring changes (and only then is `simulation_running`
///   an error); offset-only changes just rewrite the visual transform.
/// - A rebuilt custom mesh gets a fresh `Comet::id`, so the renderer creates new GPU resources
///   and its stale-mesh sweep frees the previous ones.
pub fn apply_comet_appearance(
  scene: &Scene,
  visual: EntityId,
  radius_km: f32,
  wiring: CometAppearanceWiring,
  offset: CometVisualOffset,
  library: &AssetLibrary,
  simulation_running: bool,
) -> Result<(), AppearanceError> {
  let current = scene
    .with_component(visual, |a: &CometAppearanceComponent| *a)
    .ok_or(AppearanceError::NoVisualEntity)?;

  let wiring_changed = current.wiring != wiring;
  if wiring_changed && simulation_running {
    return Err(AppearanceError::SimulationRunning);
  }

  // Validate before mutating anything.
  if wiring.mode == CometDisplayMode::Custom {
    if let Some(mesh) = wiring.mesh {
      library.mesh(mesh).ok_or(AppearanceError::UnknownMesh)?;
    }
    for id in wiring.textures.iter().flatten() {
      library.texture(*id).ok_or(AppearanceError::UnknownTexture)?;
    }
  }

  let mut next = CometAppearanceComponent {
    wiring,
    offset,
    bounding_center: current.bounding_center,
    bounding_radius: current.bounding_radius,
  };

  if wiring_changed {
    let is_visible = scene
      .with_component(visual, |m: &StaticMeshComponent| m.is_visible)
      .unwrap_or(false);
    let new_mesh = match (wiring.mode, wiring.mesh) {
      (CometDisplayMode::Custom, Some(mesh_id)) => {
        let asset = library.mesh(mesh_id).ok_or(AppearanceError::UnknownMesh)?;
        next.bounding_center = asset.bounding_center;
        next.bounding_radius = asset.bounding_radius;
        let mut comet: Comet =
          library.materialize_mesh(mesh_id).ok_or(AppearanceError::UnknownMesh)?;
        let tex = |c: TextureChannel| {
          wiring.textures[c as usize].and_then(|id| library.texture(id).map(|t| t.texture()))
        };
        comet.albedo_map = tex(TextureChannel::Albedo);
        comet.normal_map = tex(TextureChannel::Normal);
        comet.roughness_map = tex(TextureChannel::Roughness);
        comet.ao_map = tex(TextureChannel::Ao);
        StaticMeshComponent {
          asset_path: alloc::format!("asset:{}", asset.key),
          mesh: Arc::new(comet),
          emissive_color: [0.0, 0.0, 0.0, 0.0],
          is_visible,
        }
      }
      _ => StaticMeshComponent {
        is_visible,
        ..default_comet_mesh(radius_km)
      },
    };
    let _ = scene.with_component_mut(visual, |m: &mut StaticMeshComponent| *m = new_mesh);
  }

  let transform = if next.shows_custom_mesh() {
    custom_visual_transform(
      next.bounding_center,
      next.bounding_radius,
      radius_km,
      &offset,
    )
  } else {
    TransformComponent::default()
  };
  let _ = scene.with_component_mut(visual, |t: &mut TransformComponent| *t = transform);
  let _ = scene.with_component_mut(visual, |a: &mut CometAppearanceComponent| *a = next);
  Ok(())
}

/// Stops displaying asset `asset_id` before it is unloaded: a wired mesh ejects the comet to
/// the procedural sphere (default wiring), a wired texture has its channel(s) cleared. The
/// placement offset is kept. Returns whether the appearance changed.
pub fn unwire_asset(
  scene: &Scene,
  visual: EntityId,
  radius_km: f32,
  asset_id: AssetId,
  library: &AssetLibrary,
) -> Result<bool, AppearanceError> {
  let current = scene
    .with_component(visual, |a: &CometAppearanceComponent| *a)
    .ok_or(AppearanceError::NoVisualEntity)?;
  let mut wiring = current.wiring;
  if wiring.mesh == Some(asset_id) {
    wiring = CometAppearanceWiring::default();
  }
  for slot in wiring.textures.iter_mut() {
    if *slot == Some(asset_id) {
      *slot = None;
    }
  }
  if wiring == current.wiring {
    return Ok(false);
  }
  // only reached while paused (RemoveAsset is refused while playing)
  apply_comet_appearance(
    scene,
    visual,
    radius_km,
    wiring,
    current.offset,
    library,
    false,
  )?;
  Ok(true)
}

/// Recomputes the custom-mesh transform after a nucleus radius change. Returns `true` when the
/// visual shows a custom mesh (in which case the default-sphere vertex rescale must be skipped).
pub fn rescale_custom_visual(scene: &Scene, visual: EntityId, radius_km: f32) -> bool {
  let Some(appearance) = scene.with_component(visual, |a: &CometAppearanceComponent| *a) else {
    return false;
  };
  if !appearance.shows_custom_mesh() {
    return false;
  }
  let transform = custom_visual_transform(
    appearance.bounding_center,
    appearance.bounding_radius,
    radius_km,
    &appearance.offset,
  );
  let _ = scene.with_component_mut(visual, |t: &mut TransformComponent| *t = transform);
  true
}

#[cfg(test)]
#[path = "comet_appearance_tests.rs"]
mod tests;
