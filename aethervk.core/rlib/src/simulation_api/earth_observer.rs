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
//! **Position.** Geocentric by default: the camera sits at the Earth's centre and follows only its
//! translation (the Earth's rotation never turns it, there is no horizon). **Topocentric** (optional,
//! [`EarthObserverState::topocentric`]): the observer stands at a geodetic latitude / longitude /
//! height on the WGS84 ellipsoid ([`super::astrometry::wgs84_surface_point_bf`], Earth body-fixed =
//! ITRF93, the frame of the Earth rotation the almanac commits), carried by the Earth's rotation
//! (parallax), with the ellipsoid's tangent plane as horizon (geodetic zenith,
//! [`EarthObserverState::zenith_bf`]). The orientation never turns with the Earth in either case:
//! a tracked target is looked at even while it is below the site's horizon (the horizon is reported
//! in the diagnostics only; facing the horizon instead made the view swing with the Earth's spin).
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

/// Mirror of the C# `EarthObserverOrientationMode` (same integer values over FFI; 1 and 3 were the
/// retired lock-in modes, read as `Free`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum EarthObserverMode {
  /// look direction fixed in the inertial (ecliptic) frame
  Free = 0,
  /// re-aims at the comet every commit: the comet stays at the centre of the view, below the
  /// site's horizon included
  CometTracking = 2,
  SunTracking = 4,
}

impl EarthObserverMode {
  pub fn from_u32(v: u32) -> Option<Self> {
    Some(match v {
      // 1 / 3: the retired comet / Sun lock-in (the look the caller passes is inertial)
      0 | 1 | 3 => Self::Free,
      2 => Self::CometTracking,
      4 => Self::SunTracking,
      _ => return None,
    })
  }
  pub fn is_tracking(self) -> bool {
    matches!(self, Self::CometTracking | Self::SunTracking)
  }
  pub fn targets_comet(self) -> bool {
    matches!(self, Self::CometTracking)
  }
}

/// Which way is up on the observer's screen (C# `ObserverUp`, same integer values over FFI).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum ObserverUp {
  /// right vector in the ecliptic plane, up towards the ecliptic pole on the latitude's side
  /// (`pole(lat)`): the original view, an explicit choice (value 0 over FFI / in saves)
  Ecliptic = 0,
  /// up towards the J2000 celestial north pole (`(0, sin ε, cos ε)` in the ecliptic world frame),
  /// whatever the latitude or the Earth's rotation: the astronomical image convention (north up,
  /// east left) and the default ("From Earth, celestial north up")
  #[default]
  CelestialNorth = 1,
}

impl ObserverUp {
  /// 0 reads as `Ecliptic`, 1 as `CelestialNorth`; unknown values read as the default
  /// (`CelestialNorth`)
  pub fn from_u32(v: u32) -> Self {
    match v {
      0 => Self::Ecliptic,
      _ => Self::CelestialNorth,
    }
  }
}

/// Earth observer posed natively each commit (see the module doc).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarthObserverState {
  pub mode: EarthObserverMode,
  /// screen-up convention of every pose ([`ObserverUp`]; `new` takes `Ecliptic`)
  pub up: ObserverUp,
  pub camera: EntityId,
  pub earth: EntityId,
  /// comet body (comet submodes); the Sun is the origin
  pub comet: Option<EntityId>,
  /// false: at the Earth's centre (no horizon); true: at the surface site below
  pub topocentric: bool,
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
  /// Free: inertial world look rotation
  pub look: Q,
  /// last applied rotation (kept when the target is straight below, at the nadir)
  pub last: Q,
  /// diagnostics of the last pose: view-axis error to the target (rad; 0 when aimed) and the
  /// target elevation (rad; negative: below the site's horizon, still looked at)
  pub last_error: f64,
  pub last_elevation: f64,
  /// heliocentric velocity of the comet (km/s, world axes) at the last commit: the light-time
  /// displacement's input (`None`: no light time, the comet is drawn geometric)
  pub comet_vel_km_s: Option<V3>,
  /// light time of the last pose (`None` without a comet or its velocity)
  pub light_time: Option<LightTime>,
}

impl EarthObserverState {
  /// A geocentric observer, or (`topocentric`) one at geodetic `lat_deg` / `lon_deg` / `elev_m`
  /// (WGS84). `last`: the camera's current rotation (held while a target is unknown).
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    mode: EarthObserverMode,
    camera: EntityId,
    earth: EntityId,
    comet: Option<EntityId>,
    topocentric: bool,
    lat_deg: f64,
    lon_deg: f64,
    elev_m: f64,
    look: Q,
    last: Q,
  ) -> Self {
    Self {
      mode,
      up: ObserverUp::default(),
      camera,
      earth,
      comet,
      topocentric,
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

  /// distance of the site from the Earth's centre (AU)
  pub fn surface_radius_au(&self) -> f64 {
    dot(self.surface_bf_au, self.surface_bf_au).sqrt()
  }

  /// the camera's offset from the Earth's centre in the body-fixed frame: the site, or zero
  /// (geocentric). `|camera − earth|` is its length.
  pub fn offset_bf_au(&self) -> V3 {
    if self.topocentric { self.surface_bf_au } else { [0.0; 3] }
  }

  /// latitude whose pole the view's up leans to: the site's, the ecliptic north geocentric
  pub fn up_lat_deg(&self) -> f64 {
    if self.topocentric { self.lat_deg } else { 0.0 }
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

/// J2000 celestial north pole in the ecliptic world frame: `R1(ε)·ẑ = (0, sin ε, cos ε)`, with the
/// engine's one obliquity ([`super::astrometry::OBLIQUITY_J2000_RAD`], = C# `SkyMath.ObliquityRad`).
pub fn celestial_north() -> V3 {
  let (s, c) = super::astrometry::OBLIQUITY_J2000_RAD.sin_cos();
  [0.0, s, c]
}

/// Up hint of the [`ObserverUp`] convention: the latitude's ecliptic pole (`Ecliptic`, exactly the
/// former `pole(lat)`), or the celestial north pole whatever the latitude (`CelestialNorth`).
pub fn up_hint(up: ObserverUp, lat_deg: f64) -> V3 {
  match up {
    ObserverUp::Ecliptic => pole(lat_deg),
    ObserverUp::CelestialNorth => celestial_north(),
  }
}

/// C# `EarthObserverOrient`: right vector in the ecliptic plane, up towards the latitude's pole.
pub fn orient(q: Q, lat_deg: f64) -> Q {
  orient_up(q, ObserverUp::Ecliptic, lat_deg)
}

/// [`orient`] with the `up` convention: zero roll about [`up_hint`].
pub fn orient_up(q: Q, up: ObserverUp, lat_deg: f64) -> Q {
  strip_roll(q, Some(up_hint(up, lat_deg)))
}

/// C# `EarthObserverLookAt`: forward along `direction`.
pub fn look_at(direction: V3, lat_deg: f64) -> Q {
  look_at_up(direction, ObserverUp::Ecliptic, lat_deg)
}

/// [`look_at`] with the `up` convention: forward along `direction`, screen up towards
/// [`up_hint`] (`CelestialNorth`: the projected celestial north is exactly screen up, so east is
/// screen left).
pub fn look_at_up(direction: V3, up: ObserverUp, lat_deg: f64) -> Q {
  look_at_origin_from(scale(normalize(direction), -1.0), Some(up_hint(up, lat_deg)))
}

/// C# `IsBelowHorizon`
pub fn below_horizon(direction: V3, zenith: V3) -> bool {
  dot(direction, zenith) < 0.0
}

/// World position (AU) of the point `offset_bf_au` (Earth body-fixed) on an Earth at `earth_pos`
/// turned by `earth_rot` (body-fixed → world): the camera position of [`pose`]
/// ([`EarthObserverState::offset_bf_au`]: the geocentre unless topocentric).
pub fn observer_position(earth_pos: V3, earth_rot: Q, offset_bf_au: V3) -> V3 {
  add(earth_pos, rotate(earth_rot, offset_bf_au))
}

/// Camera pose for the observer: `(world position, world rotation)`, from the Earth's world
/// position / rotation and the target's world position (`None`: no target known). Mirrors
/// `CameraService.SnapCameraToEarth`. The Earth's rotation moves only a topocentric site (diurnal
/// parallax), never the look: a tracked target is looked at from the site even while it is below
/// the site's horizon (the Earth is drawn in the way; the elevation is in the diagnostics), and
/// the up hint ([`up_hint`]) is inertial, so nothing of the orientation turns with the Earth.
pub fn pose(s: &EarthObserverState, earth_pos: V3, earth_rot: Q, target: Option<V3>) -> (V3, Q) {
  let cam_pos = observer_position(earth_pos, earth_rot, s.offset_bf_au());
  let up_lat = s.up_lat_deg();
  let rot = if s.mode.is_tracking() {
    match target {
      Some(t) => look_at_up(sub(t, cam_pos), s.up, up_lat),
      None => s.last,
    }
  } else {
    s.look
  };
  (cam_pos, orient_up(rot, s.up, up_lat))
}

/// Nucleus view "as seen from Earth": a camera `distance_au` from the comet on the Earth → comet
/// line (on the Earth's side), looking at the comet along that line with the projected J2000
/// celestial north as screen up (the Earth observer's default up, [`ObserverUp::CelestialNorth`]):
/// the nucleus shows the face and the orientation an Earth telescope sees, north up / east left,
/// without the 1 AU of parallax-free distance. `(world position, world rotation)`; `None` when
/// the Earth and the comet coincide.
pub fn nucleus_view_from_earth(earth_pos: V3, comet_pos: V3, distance_au: f64) -> Option<(V3, Q)> {
  let line = sub(comet_pos, earth_pos);
  if dot(line, line) <= 1e-30 {
    return None;
  }
  let dir = normalize(line);
  let pos = sub(comet_pos, scale(dir, distance_au));
  Some((pos, look_at_up(dir, ObserverUp::CelestialNorth, 0.0)))
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
  let site = observer_position(v3(earth.position), earth_rot, s.offset_bf_au());
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
    (s.last_error, s.last_elevation) = aim_error(s, pos, rot, earth_rot, t);
  }
  Some((pos, rot))
}

/// `(view-axis error to target, target elevation)` (rad) of a camera at `pos` turned by `rot`:
/// `atan2(|fwd × to|, fwd · to)` (accurate near 0, unlike `acos`), and the elevation above the
/// site's horizon (geocentric: no horizon, the target is always "up", π/2).
fn aim_error(s: &EarthObserverState, pos: V3, rot: Q, earth_rot: Q, target: V3) -> (f64, f64) {
  let to = sub(target, pos);
  let fwd = rotate(rot, [0.0, -1.0, 0.0]);
  let c = cross(fwd, to);
  let error = dot(c, c).sqrt().atan2(dot(fwd, to));
  let elevation = if s.topocentric {
    let zenith = normalize(rotate(earth_rot, s.zenith_bf));
    (dot(normalize(to), zenith)).clamp(-1.0, 1.0).asin()
  } else {
    core::f64::consts::FRAC_PI_2
  };
  (error, elevation)
}

/// The aim of the observer camera **as it is now** in `scene`: `(view-axis error, target
/// elevation)` (rad) against the target [`apply`] aims at (the astrometric comet: its committed
/// position plus the last pose's light-time offset; the Sun at the origin). Unlike
/// [`EarthObserverState::last_error`], which [`apply`] measured on its own pose, this sees any later
/// write to the camera (a stale UI animation, a restored snapshot) instead of reporting a pose the
/// screen no longer shows. `None` when an entity is missing, or in `Free` (no target).
pub fn current_aim(scene: &Scene, s: &EarthObserverState) -> Option<(f64, f64)> {
  let target = if s.mode.targets_comet() {
    let c = v3(scene.global_transform_f64(s.comet?)?.position);
    add(c, s.light_time.map_or([0.0; 3], |l| l.offset_au))
  } else if s.mode.is_tracking() {
    [0.0, 0.0, 0.0]
  } else {
    return None;
  };
  let cam = scene.global_transform_f64(s.camera)?;
  let earth = scene.global_transform_f64(s.earth)?;
  let (pos, rot) = (v3(cam.position), qnormalize(q(cam.rotation)));
  Some(aim_error(s, pos, rot, qnormalize(q(earth.rotation)), target))
}

/// Heliocentric velocity (km/s, world = SUN_ECLIPJ2000 axes) of the SPK body `naif` at `epoch`, the
/// light-time input ([`EarthObserverState::comet_vel_km_s`]) of a pose outside a commit (the commit
/// takes it from the SPK state it just stepped). `None` when the almanac does not cover it.
pub fn comet_velocity_at(
  almanac: &crate::simulation::almanac::AlmanacPackedData,
  naif: i32,
  epoch: hifitime::Epoch,
) -> Option<V3> {
  let frame = crate::simulation::almanac::SUN_ECLIPJ2000;
  almanac
    .get_cartesian_state(naif, frame.orientation_id, frame.ephemeris_id, epoch, false)
    .ok()
    .map(|s| [s.velocity_km_s[0], s.velocity_km_s[1], s.velocity_km_s[2]])
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

  /// `CelestialNorth`: for targets all over the sky (RA/Dec → equatorial → ecliptic world), the
  /// view axis is on the target, the celestial north projected on the image plane is exactly screen
  /// up (zero roll), and east (increasing RA) is screen left; the latitude's sign changes nothing.
  /// Engine screen axes as `gpu::frame::camera_screen_axes`: right = local −X, up = local +Z.
  #[test]
  fn celestial_north_up_puts_north_up_and_east_left() {
    use super::super::astrometry::{dir_of_ra_dec, equatorial_to_ecliptic_j2000};
    let north = celestial_north();
    // the pole is the equatorial +Z
    assert!(angle(north, equatorial_to_ecliptic_j2000([0.0, 0.0, 1.0])) < 1e-15);
    for (ra, dec) in [
      (0.0, 0.0),
      (45.0, 30.0),
      (120.0, -50.0),
      (200.0, 10.0),
      (285.0, 70.0),
      (270.0, -20.0),
      (90.0, 66.5),
      (350.0, -75.0),
    ] {
      let (ra, dec) = (f64::to_radians(ra), f64::to_radians(dec));
      let d = equatorial_to_ecliptic_j2000(dir_of_ra_dec(ra, dec));
      // east: dα > 0 at fixed Dec, unit (−sin α, cos α, 0) on the equatorial axes
      let east = equatorial_to_ecliptic_j2000([-ra.sin(), ra.cos(), 0.0]);
      let d_east = equatorial_to_ecliptic_j2000(dir_of_ra_dec(ra + 1e-4, dec));
      for lat in [45.0, -30.0, 0.0] {
        let r = look_at_up(d, ObserverUp::CelestialNorth, lat);
        assert!(angle(forward(r), d) < 1e-12, "({ra}, {dec}): off target");
        let right = rotate(r, [-1.0, 0.0, 0.0]);
        let up = rotate(r, [0.0, 0.0, 1.0]);
        let n_proj = normalize(sub(north, scale(d, dot(north, d))));
        assert!(dot(up, north) > 0.0, "({ra}, {dec}, {lat}): north not up");
        assert!(
          angle(up, n_proj) < 1e-12,
          "({ra}, {dec}, {lat}): roll error {} rad",
          angle(up, n_proj)
        );
        assert!(dot(right, n_proj).abs() < 1e-12);
        assert!(dot(east, right) < 0.0, "({ra}, {dec}, {lat}): east not left");
        assert!(angle(east, scale(right, -1.0)) < 1e-12, "east exactly screen left");
        assert!(dot(sub(d_east, d), right) < 0.0, "increasing RA moves left");
        // orient keeps a north-up rotation (idempotent), and repairs a rolled one
        let o = orient_up(r, ObserverUp::CelestialNorth, lat);
        assert!(angle(forward(o), d) < 1e-12 && angle(rotate(o, [0.0, 0.0, 1.0]), n_proj) < 1e-12);
        let rolled = qmul(r, [0.0, (0.3f64).sin(), 0.0, (0.3f64).cos()]);
        let o = orient_up(rolled, ObserverUp::CelestialNorth, lat);
        assert!(angle(rotate(o, [0.0, 0.0, 1.0]), n_proj) < 1e-12, "roll stripped");
      }
    }
    // the posed observer (tracking, geocentric and topocentric) uses the same convention
    let earth = [0.98, 0.17, 0.0];
    let comet = [1.3, -0.9, 0.05];
    for topocentric in [false, true] {
      let mut s = topo(EarthObserverMode::CometTracking, topocentric);
      s.up = ObserverUp::CelestialNorth;
      let (p, r) = pose(&s, earth, spin(0.0), Some(comet));
      let to = normalize(sub(comet, p));
      let n_proj = normalize(sub(north, scale(to, dot(north, to))));
      assert!(angle(rotate(r, [0.0, 0.0, 1.0]), n_proj) < 1e-12, "pose: north up");
    }
    assert_eq!(ObserverUp::from_u32(0), ObserverUp::Ecliptic);
    assert_eq!(ObserverUp::from_u32(1), ObserverUp::CelestialNorth);
    assert_eq!(ObserverUp::from_u32(7), ObserverUp::CelestialNorth, "unknown reads as the default");
    assert_eq!(ObserverUp::default(), ObserverUp::CelestialNorth);
    assert_eq!(topo(EarthObserverMode::Free, true).up, ObserverUp::CelestialNorth);
  }

  /// `Ecliptic` (the explicit choice, value 0) is bit for bit the former `pole(lat)` formula in
  /// `look_at`, `orient` and `pose`.
  #[test]
  fn ecliptic_up_is_unchanged() {
    let earth = [0.98, 0.17, 0.0];
    for d in [
      [1.0, 2.0, 0.3],
      [-0.2, 0.1, -0.9],
      [0.0, 1.0, 0.0],
      [3.0, -1.0, 0.01],
      [0.0, 0.0, 1.0],
    ] {
      for lat in [45.0, -30.0, 0.0] {
        let old = look_at_origin_from(scale(normalize(d), -1.0), Some(pole(lat)));
        assert_eq!(look_at(d, lat), old);
        assert_eq!(look_at_up(d, ObserverUp::Ecliptic, lat), old);
        let q = qnormalize([0.1, -0.4, 0.7, 0.5]);
        assert_eq!(orient(q, lat), strip_roll(q, Some(pole(lat))));
        assert_eq!(orient_up(q, ObserverUp::Ecliptic, lat), strip_roll(q, Some(pole(lat))));
      }
      for topocentric in [false, true] {
        let mut s = topo(EarthObserverMode::CometTracking, topocentric);
        s.up = ObserverUp::Ecliptic;
        let target = add(earth, d);
        let (p, r) = pose(&s, earth, spin(1234.0), Some(target));
        let to = sub(target, p);
        let lat = s.up_lat_deg();
        let old = strip_roll(
          look_at_origin_from(scale(normalize(to), -1.0), Some(pole(lat))),
          Some(pole(lat)),
        );
        assert_eq!(r, old, "pose d {d:?} topocentric {topocentric}");
      }
    }
  }

  fn state(mode: EarthObserverMode) -> EarthObserverState {
    topo(mode, true)
  }

  fn topo(mode: EarthObserverMode, topocentric: bool) -> EarthObserverState {
    EarthObserverState::new(
      mode,
      EntityId::default(),
      EntityId::default(),
      None,
      topocentric,
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
  /// Earth's rotation, below the site's horizon included (the view never swings to the horizon).
  #[test]
  fn tracking_aims_exactly_at_the_target_every_pose() {
    let mut s = state(EarthObserverMode::CometTracking);
    let earth = [0.98, 0.17, 0.0];
    let comet0 = [1.3, -0.9, 0.05];
    let (mut above, mut below) = (0, 0);
    for k in 0..200 {
      let t = k as f64 * 431.0; // ~1 day in 200 commits
      let comet = add(comet0, scale([1.0e-4, 2.0e-4, 0.0], k as f64));
      let (pos, rot) = pose(&s, earth, spin(t), Some(comet));
      let to = sub(comet, pos);
      let zenith = normalize(rotate(spin(t), s.zenith_bf));
      if below_horizon(to, zenith) {
        below += 1;
      } else {
        above += 1;
      }
      assert!(
        angle(forward(rot), to) < 1e-9,
        "commit {k}: {} rad off",
        angle(forward(rot), to)
      );
      s.last = rot;
    }
    assert!(above > 50 && below > 50, "the day must have both ({above} / {below})");
  }

  /// The reviewer's "the camera rotates with the Earth": over a day of 1 h steps the topocentric
  /// view direction stays on the comet within the diurnal parallax (Earth radius / Δ), and the
  /// screen up stays within 0.1° of the projected celestial north (the default up), the comet below
  /// the horizon included. Nothing of the orientation follows the Earth's spin.
  #[test]
  fn view_and_up_do_not_turn_with_the_earth_over_a_day() {
    let earth = [0.98, 0.17, 0.0];
    let comet = [1.3, -0.9, 0.05];
    let north = celestial_north();
    for lat in [40.0, -33.0, 0.0] {
      let mut s = EarthObserverState::new(
        EarthObserverMode::CometTracking,
        EntityId::default(),
        EntityId::default(),
        None,
        true,
        lat,
        10.0,
        350.0,
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
      );
      assert_eq!(s.up, ObserverUp::CelestialNorth, "default up");
      let geo_dir = normalize(sub(comet, earth));
      let delta = dot(sub(comet, earth), sub(comet, earth)).sqrt();
      let parallax = s.surface_radius_au() / delta;
      let (mut above, mut below) = (0, 0);
      for h in 0..=24 {
        let t = h as f64 * 3600.0;
        let (pos, rot) = pose(&s, earth, spin(t), Some(comet));
        let to = sub(comet, pos);
        let zenith = normalize(rotate(spin(t), s.zenith_bf));
        if below_horizon(to, zenith) {
          below += 1;
        } else {
          above += 1;
        }
        // on the comet from the site, and within the parallax of the geocentric direction
        assert!(angle(forward(rot), to) < 1e-9, "lat {lat} h {h}: off the comet");
        assert!(
          angle(forward(rot), geo_dir) <= parallax * 1.001,
          "lat {lat} h {h}: {} rad from the geocentric direction, parallax {parallax}",
          angle(forward(rot), geo_dir)
        );
        let n_proj = normalize(sub(north, scale(forward(rot), dot(north, forward(rot)))));
        let up = rotate(rot, [0.0, 0.0, 1.0]);
        assert!(
          angle(up, n_proj) < 0.1f64.to_radians(),
          "lat {lat} h {h}: up {} rad from projected celestial north",
          angle(up, n_proj)
        );
        s.last = rot;
      }
      assert!(above > 0 && below > 0, "lat {lat}: both sides of the horizon ({above} / {below})");
    }
  }

  /// The "as seen from Earth" nucleus framing: the camera is `d` from the comet on the Earth →
  /// comet line, looks along it, north up, east left; it is what a geocentric Earth observer sees
  /// (same view direction and up), at close range.
  #[test]
  fn nucleus_view_from_earth_matches_the_geocentric_observer() {
    let earth = [0.98, 0.17, 0.0];
    let comet = [1.3, -0.9, 0.05];
    let d = 2.0e-7; // ~30 km
    // comet − (comet − d·u) loses ~1e-16·|comet|/d ≈ 1e-9 rad, and acos near 0 resolves ~1e-8 rad
    const ANGLE_EPS: f64 = 1e-6;
    let (pos, rot) = nucleus_view_from_earth(earth, comet, d).expect("a view");
    let to = sub(comet, pos);
    assert!((dot(to, to).sqrt() - d).abs() < 1e-8 * d, "at distance d");
    assert!(angle(to, sub(comet, earth)) < ANGLE_EPS, "on the Earth → comet line");
    assert!(angle(forward(rot), to) < ANGLE_EPS, "looking at the comet");
    let track = topo(EarthObserverMode::CometTracking, false);
    assert_eq!(track.up, ObserverUp::CelestialNorth);
    let (_, geo) = pose(&track, earth, spin(4321.0), Some(comet));
    assert!(angle(forward(rot), forward(geo)) < ANGLE_EPS, "same view direction as the observer");
    assert!(
      angle(rotate(rot, [0.0, 0.0, 1.0]), rotate(geo, [0.0, 0.0, 1.0])) < ANGLE_EPS,
      "same up as the observer"
    );
    let north = celestial_north();
    let n_proj = normalize(sub(north, scale(forward(rot), dot(north, forward(rot)))));
    assert!(angle(rotate(rot, [0.0, 0.0, 1.0]), n_proj) < ANGLE_EPS, "celestial north up");
    assert!(nucleus_view_from_earth(earth, earth, d).is_none());
  }

  /// A target below the site's horizon is still looked at (the Earth is drawn in the way), the
  /// elevation diagnostic says so.
  #[test]
  fn target_below_the_horizon_is_still_looked_at() {
    let earth = [1.0, 0.0, 0.0];
    // site at (0°N, 0°E): on the Earth's +X side at spin 0; a target towards −X is below it
    let s = EarthObserverState::new(
      EarthObserverMode::CometTracking,
      EntityId::default(),
      EntityId::default(),
      None,
      true,
      0.0,
      0.0,
      0.0,
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
    );
    let target = [-1.0, 0.0, 0.2];
    let (pos, rot) = pose(&s, earth, [0.0, 0.0, 0.0, 1.0], Some(target));
    let to = sub(target, pos);
    let zenith = normalize(rotate([0.0, 0.0, 0.0, 1.0], s.zenith_bf));
    assert!(below_horizon(to, zenith), "the test target must be below the horizon");
    assert!(angle(forward(rot), to) < 1e-12, "looked at: {} rad off", angle(forward(rot), to));
    let (err, elev) = aim_error(&s, pos, rot, [0.0, 0.0, 0.0, 1.0], target);
    assert!(err < 1e-12 && elev < 0.0, "err {err} elev {elev}");
  }

  /// Geocentric (the default): the camera is the Earth's centre whatever its rotation, and neither
  /// the free look nor the tracking aim turns with the Earth; tracking has no horizon (the comet
  /// is centred all day, the target "behind" the Earth included).
  #[test]
  fn geocentric_camera_ignores_earth_rotation() {
    let earth = [0.98, 0.17, 0.0];
    let mut free = topo(EarthObserverMode::Free, false);
    free.look = look_at([0.3, 1.0, -0.1], 0.0);
    let mut track = topo(EarthObserverMode::CometTracking, false);
    // the ecliptic up (explicit): the roll assertions below are about the ecliptic pole
    free.up = ObserverUp::Ecliptic;
    track.up = ObserverUp::Ecliptic;
    let comet = [1.3, -0.9, 0.05];
    for k in 0..48 {
      let t = k as f64 * 1800.0; // a day, every half hour
      let (p, r) = pose(&free, earth, spin(t), None);
      assert_eq!(p, earth, "geocentre");
      assert!(angle(forward(r), forward(free.look)) < 1e-12, "free look turned at {t} s");
      let (p, r) = pose(&track, earth, spin(t), Some(comet));
      assert_eq!(p, earth);
      assert!(angle(forward(r), sub(comet, earth)) < 1e-12, "comet off centre at {t} s");
      // up towards the ecliptic north, right in the ecliptic: the frame does not roll either
      assert!(rotate(r, [1.0, 0.0, 0.0])[2].abs() < 1e-12);
      assert!(rotate(r, [0.0, 0.0, 1.0])[2] >= 0.0);
      track.last = r;
    }
  }

  /// Topocentric: the site turns with the Earth (diurnal parallax of the Earth radius on a nearby
  /// target), the look still does not.
  #[test]
  fn topocentric_site_moves_with_the_earth_and_shows_parallax() {
    let earth = [1.0, 0.0, 0.0];
    let mut free = topo(EarthObserverMode::Free, true);
    free.look = look_at([0.0, 1.0, 0.0], free.lat_deg);
    let (p0, r0) = pose(&free, earth, spin(0.0), None);
    let (p1, r1) = pose(&free, earth, spin(6.0 * 3600.0), None);
    assert!(angle(forward(r0), forward(r1)) < 1e-12, "the look stays inertial");
    let d = |a: V3, b: V3| dot(sub(a, b), sub(a, b)).sqrt();
    assert!((d(p0, earth) - free.surface_radius_au()).abs() < 1e-15);
    assert!(d(p0, p1) > 0.5 * free.surface_radius_au(), "a quarter day moves the site");
    // a target 0.1 AU away: the geocentric and topocentric directions differ by ~R/Δ
    let target = [1.0, 0.1, 0.0];
    let track_geo = topo(EarthObserverMode::CometTracking, false);
    let track_topo = topo(EarthObserverMode::CometTracking, true);
    let (_, g) = pose(&track_geo, earth, spin(0.0), Some(target));
    let (pt, t) = pose(&track_topo, earth, spin(0.0), Some(target));
    let parallax = angle(forward(g), forward(t));
    assert!(parallax > 1e-5 && parallax < track_topo.surface_radius_au() / 0.1 * 1.01, "{parallax}");
    assert!(angle(forward(t), sub(target, pt)) < 1e-12, "topocentric aims from the site");
  }

  /// The retired lock-in values read as Free over FFI (old callers, old saves).
  #[test]
  fn retired_lock_in_values_read_as_free() {
    assert_eq!(EarthObserverMode::from_u32(1), Some(EarthObserverMode::Free));
    assert_eq!(EarthObserverMode::from_u32(3), Some(EarthObserverMode::Free));
    assert_eq!(EarthObserverMode::from_u32(2), Some(EarthObserverMode::CometTracking));
    assert_eq!(EarthObserverMode::from_u32(4), Some(EarthObserverMode::SunTracking));
    assert_eq!(EarthObserverMode::from_u32(5), None);
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
      true,
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
      let fwd = rotate(
        [g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3]],
        [0.0, -1.0, 0.0],
      );
      let c = cross(fwd, to);
      let ang = dot(c, c).sqrt().atan2(dot(fwd, to));
      assert!(ang < 1e-9, "commit {k}: comet {ang} rad off the view axis");
      aimed += 1;
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
    {
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
