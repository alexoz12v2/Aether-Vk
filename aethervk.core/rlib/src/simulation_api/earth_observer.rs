//! Earth observer camera, posed **inside the logic-thread commit** that moves the Earth and the
//! comet (same scene write), so every rendered frame aims consistently.
//!
//! Formerly C# (`CameraService.SnapCameraToEarth`) re-aimed after the commit, via the transform
//! callback → main thread → `transformStaticCamera`: frames rendered in between showed a stale aim on
//! a camera turning with the Earth, and the callbacks' key order could aim at the previous tick's
//! comet. At a telescope field (arcseconds) any such lag loses the comet (`comet_tracking.rdc`: ~0.75°
//! off). C# now only sets this state ([`EarthObserverState`], on `SceneContext`) on user actions.
//!
//! The math is an f64 port of the C# helpers (`LookAtOriginFrom`, `EngineQuatFromBasis`,
//! `StripRoll`, `EarthObserverOrient`, `EarthObserverLookAt`, `IsBelowHorizon`). Engine camera
//! convention: forward = local −Y, basis columns (right, backward, up). Positions are world (root)
//! AU, heliocentric ecliptic.
//!
//! **Site.** The observer stands at a geodetic latitude / longitude / height on the WGS84 ellipsoid
//! ([`super::astrometry::wgs84_surface_point_bf`], Earth body-fixed = ITRF93, the frame of the
//! Earth rotation the almanac commits); the horizon is the ellipsoid's tangent plane (geodetic
//! zenith, [`EarthObserverState::zenith_bf`]).
//!
//! **Light time.** With a comet set, every pose also computes where the telescope *sees* the
//! comet: at the retarded epoch `t_r = t − Δ/c` ([`comet_light_time`], Δ topocentric, two
//! iterations of [`super::astrometry::retarded_epoch`]). Tracking aims at that astrometric
//! position, and the pose leaves a [`LightTimeViewComponent`] on the camera for the renderer: the
//! comet's micro layer is drawn displaced by `x(t_r) − x(t)` and its dust is evaluated at
//! `t − delay` (`gpu::scene_conversion`). The committed comet transform and the simulation clock
//! are untouched (planets and the Sun stay geometric). The displacement is a two-body back
//! propagation of the committed comet over the delay from its heliocentric velocity
//! ([`EarthObserverState::comet_vel_km_s`], refreshed from the SPK state at every commit and on
//! `set_earth_observer`): < 1 m from the SPK over an hour. The nucleus' spin is not retarded (it
//! is ≪ 1 px from the Earth). No aberration (see `astrometry`'s module doc).
use crate::scene::{EntityId, Scene};
use aethervk_oshal_rlib::math::vector::{vec3f64::Vec3f64, vec4f64::Quat64};

type V3 = [f64; 3];

/// mean Earth radius (AU), mirror of C# `CameraService.EarthRadiusAu` (the UI's spherical pose).
/// The native observer stands on the WGS84 ellipsoid instead
/// ([`EarthObserverState::surface_radius_au`]).
pub const EARTH_RADIUS_AU: f64 = 4.26e-5;
type Q = [f64; 4]; // x, y, z, w

/// Mirror of the C# `EarthObserverOrientationMode` (same integer values over FFI).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum EarthObserverMode {
  /// look direction fixed in the inertial (ecliptic) frame
  Free = 0,
  /// aimed at the comet when entered, then turns with the Earth
  CometLockIn = 1,
  /// re-aims at the comet every commit, holding while it is below the horizon
  CometTracking = 2,
  SunLockIn = 3,
  SunTracking = 4,
}

impl EarthObserverMode {
  pub fn from_u32(v: u32) -> Option<Self> {
    Some(match v {
      0 => Self::Free,
      1 => Self::CometLockIn,
      2 => Self::CometTracking,
      3 => Self::SunLockIn,
      4 => Self::SunTracking,
      _ => return None,
    })
  }
  pub fn is_tracking(self) -> bool {
    matches!(self, Self::CometTracking | Self::SunTracking)
  }
  pub fn is_lock_in(self) -> bool {
    matches!(self, Self::CometLockIn | Self::SunLockIn)
  }
  pub fn targets_comet(self) -> bool {
    matches!(self, Self::CometLockIn | Self::CometTracking)
  }
}

/// Earth observer posed natively each commit (see the module doc).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarthObserverState {
  pub mode: EarthObserverMode,
  pub camera: EntityId,
  pub earth: EntityId,
  /// comet body (comet submodes); the Sun is the origin
  pub comet: Option<EntityId>,
  /// surface point in the Earth body-fixed frame (AU): WGS84 at `lat_deg`, `lon_deg`, `elev_m`
  pub surface_bf_au: V3,
  /// geodetic latitude (deg)
  pub lat_deg: f64,
  /// longitude (deg, east positive)
  pub lon_deg: f64,
  /// height above the WGS84 ellipsoid (m)
  pub elev_m: f64,
  /// unit geodetic zenith (ellipsoid normal), Earth body-fixed: the horizon plane
  pub zenith_bf: V3,
  /// Free: inertial world look rotation; lock-in: body-fixed look (world = earth_rot · look)
  pub look: Q,
  /// last applied rotation (kept when the target is straight below, at the nadir)
  pub last: Q,
  /// diagnostics of the last pose: view-axis error to the target (rad; 0 when aimed, the elevation
  /// below the horizon when tracking faces the horizon) and the target elevation (rad)
  pub last_error: f64,
  pub last_elevation: f64,
  /// heliocentric velocity of the comet (km/s, world axes) at the last commit: the light-time
  /// displacement's input (`None`: no light time, the comet is drawn geometric)
  pub comet_vel_km_s: Option<V3>,
  /// light time of the last pose (`None` without a comet or its velocity)
  pub light_time: Option<LightTime>,
}

impl EarthObserverState {
  /// An observer at geodetic `lat_deg` / `lon_deg` / `elev_m` (WGS84). `last`: the camera's
  /// current rotation (held while a target is unknown).
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    mode: EarthObserverMode,
    camera: EntityId,
    earth: EntityId,
    comet: Option<EntityId>,
    lat_deg: f64,
    lon_deg: f64,
    elev_m: f64,
    look: Q,
    last: Q,
  ) -> Self {
    Self {
      mode,
      camera,
      earth,
      comet,
      surface_bf_au: super::astrometry::wgs84_surface_point_bf(lat_deg, lon_deg, elev_m),
      lat_deg,
      lon_deg,
      elev_m,
      zenith_bf: super::astrometry::wgs84_zenith_bf(lat_deg, lon_deg),
      look,
      last,
      last_error: 0.0,
      last_elevation: 0.0,
      comet_vel_km_s: None,
      light_time: None,
    }
  }

  /// distance of the site from the Earth's centre (AU): the invariant `|camera − earth|`
  pub fn surface_radius_au(&self) -> f64 {
    dot(self.surface_bf_au, self.surface_bf_au).sqrt()
  }

  /// Takes the comet's heliocentric velocity from the SPK state the logic thread just stepped
  /// (`CartesianStateComet::helio_state_km`), when the comet is in the cache.
  pub fn refresh_comet_velocity(
    &mut self,
    scene_id: u64,
    cache: &dashmap::DashMap<
      crate::simulation_api::structs::SceneEntityId,
      crate::simulation_api::structs::CartesianState,
    >,
  ) {
    let Some(comet) = self.comet else {
      return;
    };
    let state = cache
      .get(&crate::simulation_api::structs::SceneEntityId::new(
        scene_id, comet,
      ))
      .and_then(|c| c.comet_state.as_ref().and_then(|b| b.helio_state_km));
    if let Some((_, v)) = state {
      self.comet_vel_km_s = Some(v.into());
    }
  }
}

/// Light time of the comet seen by the observer (see the module doc).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LightTime {
  /// `t − t_r` (s)
  pub delay_s: f64,
  /// `x(t_r) − x(t)`: where the comet is seen minus where it is (world AU)
  pub offset_au: V3,
}

/// Light time of a comet at `comet_au` (world, heliocentric AU) moving at `comet_vel_km_s`, seen
/// from `observer_au` at the same instant: `t_r` from two iterations of
/// [`super::astrometry::retarded_epoch`] on the comet's two-body path
/// ([`super::astrometry::two_body_propagate`]).
pub fn comet_light_time(observer_au: V3, comet_au: V3, comet_vel_km_s: V3) -> LightTime {
  use super::astrometry::{AU_KM, retarded_epoch, two_body_propagate};
  let (r_km, o_km) = (scale(comet_au, AU_KM), scale(observer_au, AU_KM));
  let at = |dt: f64| two_body_propagate(r_km, comet_vel_km_s, dt).0;
  let t_r = retarded_epoch(0.0, |dt| {
    let d = sub(at(dt), o_km);
    dot(d, d).sqrt()
  });
  LightTime {
    delay_s: -t_r,
    offset_au: scale(sub(at(t_r), r_km), 1.0 / AU_KM),
  }
}

/// Left on the Earth observer's camera by every pose with a light time ([`apply`]); read by the
/// renderer (`gpu::scene_conversion`): the comet's micro layer is drawn at `x + offset_au` and its
/// dust is evaluated at the snapshot time minus `delay_s`. Removed when the observer is cleared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightTimeViewComponent {
  /// the comet body whose micro layer is displaced
  pub comet: EntityId,
  pub delay_s: f64,
  pub offset_au: V3,
}

impl crate::scene::Component for LightTimeViewComponent {}

/// Marks a root-level overlay drawn for `comet` (its orbit, the path it actually followed): with a
/// [`LightTimeViewComponent`] for that comet the renderer translates it by `offset_au` like the
/// comet's micro layer, so the line passes through the nucleus the telescope sees. A translation is
/// exact at the comet and first order along the curve (`offset ≈ −v·τ` varies with the velocity).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightTimeTrackComponent {
  pub comet: EntityId,
}

impl crate::scene::Component for LightTimeTrackComponent {}

/// Adds a [`LightTimeTrackComponent`] for `comet` to `entity` (registered on first use, as
/// [`LightTimeViewComponent`]).
pub fn mark_light_time_track(scene: &Scene, entity: EntityId, comet: EntityId) {
  let track = LightTimeTrackComponent { comet };
  if let Err(crate::scene::AddComponentError::ComponentNotRegistered) =
    scene.add_component(entity, track)
  {
    scene.register_component::<LightTimeTrackComponent>(&[]);
    let _ = scene.add_component(entity, track);
  }
}

/// Removes the light-time view from `camera` (observer cleared or moved to another camera).
pub fn clear_light_time_view(scene: &Scene, camera: EntityId) {
  let _ = scene.remove_component::<LightTimeViewComponent>(camera);
}

// ─── vector / quaternion helpers (f64) ──────────────────────────────────────

fn dot(a: V3, b: V3) -> f64 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
  [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}
fn scale(a: V3, s: f64) -> V3 {
  [a[0] * s, a[1] * s, a[2] * s]
}
fn add(a: V3, b: V3) -> V3 {
  [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn normalize(a: V3) -> V3 {
  let l = dot(a, a).sqrt();
  if l > 0.0 { scale(a, 1.0 / l) } else { a }
}
/// q·v·q* (C# `Vector3d.Transform`)
pub fn rotate(q: Q, v: V3) -> V3 {
  let u = [q[0], q[1], q[2]];
  let t = scale(cross(u, v), 2.0);
  add(add(v, scale(t, q[3])), cross(u, t))
}
/// Hamilton product a·b (C# `Quaterniond` `*`)
pub fn qmul(a: Q, b: Q) -> Q {
  [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ]
}
pub fn qnormalize(q: Q) -> Q {
  let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
  if l > 0.0 {
    [q[0] / l, q[1] / l, q[2] / l, q[3] / l]
  } else {
    [0.0, 0.0, 0.0, 1.0]
  }
}
pub fn qinverse(q: Q) -> Q {
  let n = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  [-q[0] / n, -q[1] / n, -q[2] / n, q[3] / n]
}

/// C# `EngineQuatFromBasis` (port of `Quat::from_mat4`): columns right, backward, up.
pub fn quat_from_basis(right: V3, backward: V3, up: V3) -> Q {
  let (m00, m01, m02) = (right[0], backward[0], up[0]);
  let (m10, m11, m12) = (right[1], backward[1], up[1]);
  let (m20, m21, m22) = (right[2], backward[2], up[2]);
  let trace = m00 + m11 + m22;
  if trace > 0.0 {
    let s = (trace + 1.0).sqrt() * 2.0;
    [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]
  } else if m00 > m11 && m00 > m22 {
    let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
    [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
  } else if m11 > m22 {
    let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
    [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]
  } else {
    let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
    [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]
  }
}

/// C# `LookAtOriginFrom(pos, upHint)`: rotation whose forward (−Y) points from `pos` to the origin.
pub fn look_at_origin_from(pos: V3, up_hint: Option<V3>) -> Q {
  let fwd = normalize(scale(pos, -1.0));
  let fallback = |f: V3| {
    if f[2].abs() < 0.99 {
      [0.0, 0.0, 1.0]
    } else {
      [0.0, -1.0, 0.0]
    }
  };
  let up_hint = match up_hint {
    Some(h) if dot(fwd, h).abs() <= 0.99 => h,
    _ => fallback(fwd),
  };
  let right = normalize(cross(up_hint, fwd));
  let up = cross(fwd, right);
  quat_from_basis(right, scale(fwd, -1.0), up)
}

/// C# `StripRoll`: same forward, zero roll about `up_hint`.
pub fn strip_roll(q: Q, up_hint: Option<V3>) -> Q {
  let fwd = rotate(q, [0.0, -1.0, 0.0]);
  if dot(fwd, fwd) < 1e-20 {
    return q;
  }
  look_at_origin_from(scale(fwd, -1.0), up_hint)
}

fn pole(lat_deg: f64) -> V3 {
  if lat_deg >= 0.0 {
    [0.0, 0.0, 1.0]
  } else {
    [0.0, 0.0, -1.0]
  }
}

/// C# `EarthObserverOrient`: right vector in the ecliptic plane, up towards the latitude's pole.
pub fn orient(q: Q, lat_deg: f64) -> Q {
  strip_roll(q, Some(pole(lat_deg)))
}

/// C# `EarthObserverLookAt`: forward along `direction`.
pub fn look_at(direction: V3, lat_deg: f64) -> Q {
  look_at_origin_from(scale(normalize(direction), -1.0), Some(pole(lat_deg)))
}

/// C# `IsBelowHorizon`
pub fn below_horizon(direction: V3, zenith: V3) -> bool {
  dot(direction, zenith) < 0.0
}

/// World position (AU) of the site `surface_bf_au` (Earth body-fixed) on an Earth at `earth_pos`
/// turned by `earth_rot` (body-fixed → world): the camera position of [`pose`].
pub fn observer_position(earth_pos: V3, earth_rot: Q, surface_bf_au: V3) -> V3 {
  add(earth_pos, rotate(earth_rot, surface_bf_au))
}

/// Camera pose for the observer: `(world position, world rotation)`, from the Earth's world
/// position / rotation and the target's world position (`None`: no target known). Mirrors
/// `CameraService.SnapCameraToEarth`, with the geodetic zenith.
pub fn pose(s: &EarthObserverState, earth_pos: V3, earth_rot: Q, target: Option<V3>) -> (V3, Q) {
  let zenith = normalize(rotate(earth_rot, s.zenith_bf));
  let cam_pos = observer_position(earth_pos, earth_rot, s.surface_bf_au);
  let rot = if s.mode.is_tracking() {
    match target {
      Some(t) => {
        let to = sub(t, cam_pos);
        if below_horizon(to, zenith) {
          // the Earth is in the way: face the horizon under the target's azimuth (where it is,
          // and where it will rise), not whatever direction the camera had
          let h = sub(to, scale(zenith, dot(to, zenith)));
          if dot(h, h) > 1e-30 {
            look_at(h, s.lat_deg)
          } else {
            s.last
          }
        } else {
          look_at(to, s.lat_deg)
        }
      }
      None => s.last,
    }
  } else if s.mode.is_lock_in() {
    qnormalize(qmul(earth_rot, s.look))
  } else {
    s.look
  };
  (cam_pos, orient(rot, s.lat_deg))
}

fn v3(v: Vec3f64) -> V3 {
  v.into()
}
fn q(q: Quat64) -> Q {
  [q[0], q[1], q[2], q[3]]
}

/// Poses the observer camera from the scene's current Earth / target transforms (call inside the
/// scene write that committed them), computes the comet's light time and leaves it on the camera
/// ([`LightTimeViewComponent`]). Returns the applied pose (rotation stored as `last`), `None` when
/// an entity is missing.
pub fn apply(scene: &Scene, s: &mut EarthObserverState) -> Option<(V3, Q)> {
  let earth = scene.global_transform_f64(s.earth)?;
  // the Earth's rotation is committed in f32: renormalize (|q|² − 1 ~ 1e-8 scales the surface
  // offset by ~0.1 m otherwise)
  let earth_rot = qnormalize(q(earth.rotation));
  let site = observer_position(v3(earth.position), earth_rot, s.surface_bf_au);
  let comet_geometric = s.comet.and_then(|c| scene.global_transform_f64(c)).map(|t| v3(t.position));
  s.light_time = match (comet_geometric, s.comet_vel_km_s) {
    (Some(c), Some(v)) => Some(comet_light_time(site, c, v)),
    _ => None,
  };
  let target = if s.mode.targets_comet() {
    // astrometric: where the telescope sees the comet
    comet_geometric.map(|c| add(c, s.light_time.map_or([0.0; 3], |l| l.offset_au)))
  } else {
    Some([0.0, 0.0, 0.0])
  };
  let (pos, rot) = pose(s, v3(earth.position), earth_rot, target);
  scene
    .set_global_transform_f64(
      s.camera,
      Vec3f64::from_array(pos),
      Quat64::from_components(rot[0], rot[1], rot[2], rot[3]),
    )
    .ok()?;
  let _ = scene.remove_component::<crate::scene::animation::TransformAnimationComponent>(s.camera);
  s.last = rot;
  match (s.light_time, s.comet) {
    (Some(l), Some(comet)) => {
      let view = LightTimeViewComponent {
        comet,
        delay_s: l.delay_s,
        offset_au: l.offset_au,
      };
      if scene
        .with_component_mut(s.camera, |c: &mut LightTimeViewComponent| *c = view)
        .is_none()
      {
        // not in `register_all_crate_components`: registered on first use (an unregistered type
        // is refused with `ComponentNotRegistered`)
        if let Err(crate::scene::AddComponentError::ComponentNotRegistered) =
          scene.add_component(s.camera, view)
        {
          scene.register_component::<LightTimeViewComponent>(&[]);
          let _ = scene.add_component(s.camera, view);
        }
      }
    }
    _ => clear_light_time_view(scene, s.camera),
  }
  if let Some(t) = target {
    let to = sub(t, pos);
    let zenith = normalize(rotate(earth_rot, s.zenith_bf));
    let fwd = rotate(rot, [0.0, -1.0, 0.0]);
    let c = cross(fwd, to);
    s.last_error = dot(c, c).sqrt().atan2(dot(fwd, to));
    s.last_elevation = (dot(normalize(to), zenith)).clamp(-1.0, 1.0).asin();
  }
  Some((pos, rot))
}

#[cfg(test)]
mod tests {
  use super::*;

  /// atan2(|a×b|, a·b): accurate near 0 (acos bottoms out at ~1.5e-8 rad)
  fn angle(a: V3, b: V3) -> f64 {
    let c = cross(a, b);
    dot(c, c).sqrt().atan2(dot(a, b))
  }
  fn forward(q: Q) -> V3 {
    rotate(q, [0.0, -1.0, 0.0])
  }

  /// `look_at` points the engine forward (−Y) along the direction, right in the ecliptic plane.
  #[test]
  fn look_at_points_forward_and_levels_to_the_ecliptic() {
    for d in [
      [1.0, 2.0, 0.3],
      [-0.2, 0.1, -0.9],
      [0.0, 1.0, 0.0],
      [3.0, -1.0, 0.01],
    ] {
      for lat in [45.0, -30.0] {
        let r = look_at(d, lat);
        assert!(angle(forward(r), d) < 1e-12, "{d:?}");
        let right = rotate(r, [1.0, 0.0, 0.0]);
        assert!(right[2].abs() < 1e-12, "right in the ecliptic: {right:?}");
        let up = rotate(r, [0.0, 0.0, 1.0]);
        assert!(up[2] * pole(lat)[2] >= -1e-12, "up towards the pole");
        // orient is idempotent on a levelled rotation
        let o = orient(r, lat);
        assert!(angle(forward(o), d) < 1e-12);
      }
    }
    // hand-checked C# case: fwd = (0,0,−1), hint (1,0,0) → q = (0.5,0.5,0.5,0.5)
    let q = look_at_origin_from([0.0, 0.0, 1.0], Some([1.0, 0.0, 0.0]));
    for (a, b) in q.iter().zip([0.5, 0.5, 0.5, 0.5]) {
      assert!((a - b).abs() < 1e-12, "{q:?}");
    }
  }

  fn state(mode: EarthObserverMode) -> EarthObserverState {
    EarthObserverState::new(
      mode,
      EntityId::default(),
      EntityId::default(),
      None,
      40.0,
      10.0,
      350.0,
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
    )
  }

  fn spin(t: f64) -> Q {
    // Earth spin about an obliquity-tilted axis
    let ob = 23.44f64.to_radians();
    let axis = [0.0, -ob.sin(), ob.cos()];
    let h = 0.5 * 7.292e-5 * t;
    [
      axis[0] * h.sin(),
      axis[1] * h.sin(),
      axis[2] * h.sin(),
      h.cos(),
    ]
  }

  /// Tracking: at every pose the view axis goes through the target (sub-nanoradian), whatever the
  /// Earth's rotation, as long as it is above the horizon; below it, the last aim is held.
  #[test]
  fn tracking_aims_exactly_at_the_target_every_pose() {
    let mut s = state(EarthObserverMode::CometTracking);
    let earth = [0.98, 0.17, 0.0];
    let comet0 = [1.3, -0.9, 0.05];
    let mut aimed = 0;
    for k in 0..200 {
      let t = k as f64 * 431.0; // ~1 day in 200 commits
      let comet = add(comet0, scale([1.0e-4, 2.0e-4, 0.0], k as f64));
      let (pos, rot) = pose(&s, earth, spin(t), Some(comet));
      let to = sub(comet, pos);
      let zenith = normalize(rotate(spin(t), s.zenith_bf));
      if below_horizon(to, zenith) {
        // faces the horizon under the target's azimuth
        let h = sub(to, scale(zenith, dot(to, zenith)));
        assert!(
          angle(forward(rot), h) < 1e-9,
          "commit {k}: not facing the target's azimuth"
        );
        assert!(
          dot(forward(rot), zenith).abs() < 1e-9,
          "commit {k}: not level with the horizon"
        );
      } else {
        aimed += 1;
        assert!(
          angle(forward(rot), to) < 1e-9,
          "commit {k}: {} rad off",
          angle(forward(rot), to)
        );
      }
      s.last = rot;
    }
    assert!(
      aimed > 50,
      "the target must be above the horizon part of the day ({aimed})"
    );
  }

  /// Lock-in turns with the Earth: the look direction is fixed in the body frame.
  #[test]
  fn lock_in_turns_with_the_earth() {
    let mut s = state(EarthObserverMode::CometLockIn);
    let world_look = look_at([1.0, 0.5, 0.2], s.lat_deg);
    s.look = qnormalize(qmul(qinverse(spin(0.0)), world_look));
    let (_, r0) = pose(&s, [1.0, 0.0, 0.0], spin(0.0), None);
    assert!(angle(forward(r0), forward(world_look)) < 1e-12);
    let (_, r1) = pose(&s, [1.0, 0.0, 0.0], spin(3600.0), None);
    let expect = rotate(
      spin(3600.0),
      rotate(qinverse(spin(0.0)), forward(world_look)),
    );
    assert!(angle(forward(r1), expect) < 1e-9);
    assert!(angle(forward(r1), forward(r0)) > 1e-3, "it turned");
  }

  /// Free look keeps the inertial direction whatever the Earth does; the position follows the
  /// surface point.
  #[test]
  fn free_look_is_inertial() {
    let mut s = state(EarthObserverMode::Free);
    s.look = look_at([0.3, 1.0, -0.1], s.lat_deg);
    let (p0, r0) = pose(&s, [1.0, 0.0, 0.0], spin(0.0), None);
    let (p1, r1) = pose(&s, [1.0, 0.0, 0.0], spin(5000.0), None);
    assert!(angle(forward(r0), forward(r1)) < 1e-12);
    // on the WGS84 ellipsoid at 40°, 350 m up: N(40°)·… ≈ 6369.3 km from the centre
    let r = dot(sub(p0, [1.0, 0.0, 0.0]), sub(p0, [1.0, 0.0, 0.0])).sqrt();
    assert!((r - s.surface_radius_au()).abs() < 1e-15);
    assert!(
      (r * 149_597_870.7 - 6369.69).abs() < 0.1,
      "{} km",
      r * 149_597_870.7
    );
    assert!(
      dot(sub(p0, p1), sub(p0, p1)) > 0.0,
      "the surface point moved"
    );
  }
}

#[cfg(test)]
mod scene_tests {
  use super::*;
  use crate::{
    scene::{
      CameraComponent, CameraProjection, HighResTransformComponent, ReferenceFrameComponent,
      ReferenceFrameType, TransformComponent,
    },
    simulation::texture_cache::TextureCache,
  };
  use aethervk_oshal_rlib::math::{
    quaternion::Quaternion as _,
    vector::{Vector3 as _, vec3::Vec3f32, vec4::Quat},
  };

  const AU_TO_KM: f64 = 149_597_870.7;

  /// root (macro) → Earth subtree frame (micro, km) → Earth body (f32 rotation) → camera; comet as
  /// a root child. The app's hierarchy.
  fn scene() -> (Scene, EntityId, EntityId, EntityId) {
    let scene = Scene::new(alloc::sync::Arc::new(parking_lot::RwLock::new(
      TextureCache::new("earth_observer"),
    )));
    scene.register_all_crate_components();
    let root = scene.spawn_entity("root");
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
    let frame = scene.spawn_entity("earth_subtree");
    scene.set_parent(frame, Some(root));
    scene
      .add_component(
        frame,
        TransformComponent {
          position: Vec3f32::from_components(0.98, 0.17, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::one(),
        },
      )
      .unwrap();
    scene
      .add_component(
        frame,
        ReferenceFrameComponent {
          frame_type: ReferenceFrameType::Micro,
          scale: (1.0 / AU_TO_KM) as f32,
          soi_radius: 1.0,
          depth_layer: 1,
        },
      )
      .unwrap();
    let earth = scene.spawn_entity("earth_body");
    scene.set_parent(earth, Some(frame));
    scene.add_component(earth, TransformComponent::default()).unwrap();
    let cam = scene.spawn_entity("camera");
    scene.set_parent(cam, Some(earth));
    scene.add_component(cam, HighResTransformComponent::default()).unwrap();
    scene
      .add_component(
        cam,
        CameraComponent {
          projection: CameraProjection::Perspective {
            fov: 1e-6,
            aspect_ratio: 16.0 / 9.0,
            near: 1e-9,
            far: 10.0,
          },
          focus_distance: 1.0,
        },
      )
      .unwrap();
    let comet = scene.spawn_entity("comet");
    scene.set_parent(comet, Some(root));
    scene.add_component(comet, HighResTransformComponent::default()).unwrap();
    (scene, earth, cam, comet)
  }

  fn spin_f32(t: f64) -> Quat {
    let ob = 23.44f64.to_radians();
    let h = 0.5 * 7.292e-5 * t;
    let s = h.sin();
    Quat::from_components(
      0.0,
      (-ob.sin() * s) as f32,
      (ob.cos() * s) as f32,
      h.cos() as f32,
    )
  }

  /// Every commit (Earth turned by up to 1 day of spin, comet moved), the camera read back from the
  /// scene graph looks at the comet within 1e-9 rad (a telescope field is ~1e-6 rad), and stands on
  /// the Earth's surface: no frame depends on C# re-aiming.
  #[test]
  fn earth_observer_tracking_is_consistent_in_every_frame() {
    let (scene, earth, cam, comet) = scene();
    let mut s = EarthObserverState::new(
      EarthObserverMode::CometTracking,
      cam,
      earth,
      Some(comet),
      40.0,
      10.0,
      0.0,
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
    );
    let mut aimed = 0;
    for k in 0..240 {
      let t = k as f64 * 360.0; // 1 day/s at 60 Hz ≈ 1440 s per commit; 6 min is plenty and denser
      // commit: Earth spins (f32 rotation like the app), the comet moves ~13 km/s
      scene
        .with_component_mut(earth, |tc: &mut TransformComponent| {
          tc.rotation = spin_f32(t)
        })
        .unwrap();
      let comet_pos = Vec3f64::from_array([1.3 + 13.0 * t / AU_TO_KM, -0.9, 0.05]);
      scene
        .with_component_mut(comet, |h: &mut HighResTransformComponent| {
          h.position = comet_pos
        })
        .unwrap();
      let (pos, _) = apply(&scene, &mut s).expect("posed");
      let g = scene.global_transform_f64(cam).unwrap();
      let e = scene.global_transform_f64(earth).unwrap();
      let gp: V3 = g.position.into();
      let ep: V3 = e.position.into();
      let r = sub(gp, ep);
      // metres matter nothing to the aim at 1 AU (1 m ≈ 7e-12 rad); the scene graph composes with
      // the raw f32 Earth rotation, ~1e-8 off unit length
      assert!(
        (dot(r, r).sqrt() / s.surface_radius_au() - 1.0).abs() < 1e-7,
        "commit {k}: camera off the surface: {} vs {} AU, pos round trip {}",
        dot(r, r).sqrt(),
        s.surface_radius_au(),
        dot(sub(gp, pos), sub(gp, pos)).sqrt()
      );
      assert!(
        dot(sub(gp, pos), sub(gp, pos)).sqrt() < 1e-11,
        "commit {k}: position round trip"
      );
      let to: V3 = sub(comet_pos.into(), gp);
      let zenith = normalize(rotate(
        qnormalize([e.rotation[0], e.rotation[1], e.rotation[2], e.rotation[3]]),
        s.zenith_bf,
      ));
      if !below_horizon(to, zenith) {
        let fwd = rotate(
          [g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3]],
          [0.0, -1.0, 0.0],
        );
        let c = cross(fwd, to);
        let ang = dot(c, c).sqrt().atan2(dot(fwd, to));
        assert!(ang < 1e-9, "commit {k}: comet {ang} rad off the view axis");
        aimed += 1;
      }
    }
    assert!(aimed > 50, "{aimed}");

    // light time: with the comet's velocity the pose aims at x(t_r), leaves the view component on
    // the camera, and the displacement is −v·delay to first order
    let v = [20.0, -12.0, 1.5];
    s.comet_vel_km_s = Some(v);
    // a comet above the horizon of the last pose
    let (pos, _) = apply(&scene, &mut s).expect("posed");
    let lt = s.light_time.expect("light time with a comet velocity");
    let comet_pos: V3 = scene.global_transform_f64(comet).unwrap().position.into();
    let d_km = dot(sub(comet_pos, pos), sub(comet_pos, pos)).sqrt() * AU_TO_KM;
    let retarded = add(comet_pos, lt.offset_au);
    let d_r_km = dot(sub(retarded, pos), sub(retarded, pos)).sqrt() * AU_TO_KM;
    assert!(
      (lt.delay_s - d_r_km / super::super::astrometry::C_KM_S).abs() < 1e-4,
      "delay {} s vs Δ/c {} s",
      lt.delay_s,
      d_r_km / super::super::astrometry::C_KM_S
    );
    assert!((lt.delay_s - d_km / super::super::astrometry::C_KM_S).abs() < 1.0);
    let first_order = scale(v, -lt.delay_s / AU_TO_KM);
    assert!(
      dot(
        sub(lt.offset_au, first_order),
        sub(lt.offset_au, first_order)
      )
      .sqrt()
        * AU_TO_KM
        < 1.0,
      "offset {:?} vs −v·τ {:?}",
      lt.offset_au,
      first_order
    );
    let view = scene
      .with_component(cam, |c: &LightTimeViewComponent| *c)
      .expect("light-time view on the camera");
    assert_eq!(view.comet, comet);
    assert_eq!(view.delay_s, lt.delay_s);
    if s.last_elevation > 0.0 {
      let g = scene.global_transform_f64(cam).unwrap();
      let fwd = rotate(
        [g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3]],
        [0.0, -1.0, 0.0],
      );
      let to = sub(retarded, v3(g.position));
      let c = cross(fwd, to);
      assert!(
        dot(c, c).sqrt().atan2(dot(fwd, to)) < 1e-9,
        "aims at x(t_r)"
      );
    }
    // without a velocity: geometric, and the view component is gone
    s.comet_vel_km_s = None;
    apply(&scene, &mut s).expect("posed");
    assert!(s.light_time.is_none());
    assert!(scene.with_component(cam, |_: &LightTimeViewComponent| ()).is_none());
  }
}
