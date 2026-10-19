//! Temporal reuse by per-cluster cadence classes (plan "temporal reuse", step 3; CPU mirror).
//!
//! Every live cluster is drawn **exactly once per frame**: either in the *fresh* pass (the frame's
//! own view, current moments) or inside one *shown class image* — a pyramid built progressively
//! over `N` frames at a frozen view and reprojected each frame through the camera rotation
//! ([`reproject_add`]). A cluster may sit in a class whose image goes stale by at most `2N`
//! frames only when its on-screen rate allows it ([`cadence_rate`]: `δ · 2N ≤ ε`).
//!
//! The constants, the configuration by the device's memory, the view in world terms with its
//! reprojection map, the rate rule, the eligibility rule of the speck shell, the reprojection,
//! and the host state machine ([`DustCadenceState::begin`] / [`DustCadenceState::finish`]: the
//! lists, cycles, swaps, aborts and resets).

use aethervk_oshal_rlib::math::FloatLike;

use super::specks::DustSpecksPushConstants;
use super::splat::{
  DustMoments, DustSplatPushConstants, PYR_LEVEL_MASK, PYR_TAU_MAX, PYR_UNIT, PYRAMID_HEADER_WORDS,
  PYRAMID_TEXEL_WORDS, PyramidLayout, SIZE_EDGES, SplitView, dust_hash32, polyline_extent_px, project_point,
  segment_carry, splat_tier_list, streak_pred,
};
use super::{DustFlowUniform, DustRenderCluster, render_live};

// ─── Constants ───────────────────────────────────────────────────────────────

/// The cadence classes: a class-`N` image is built over `N` frames and shown for the next `N`
/// (staleness ≤ `2N`); the shortest is twice the readback latency (the list lengths arrive
/// [`DUST_CADENCE_READBACK`] frames after their classify, one cycle before they are used).
pub const DUST_CADENCE_CLASSES: [u32; 3] = [8, 16, 64];
/// the most classes a configuration holds (the owner byte's class id is two bits)
pub const DUST_CADENCE_MAX_CLASSES: usize = 3;
/// Interleaved halves of every class (by slot parity), their swaps offset by `N / 2`: half a
/// class advances every `N / 2` frames instead of the whole image jumping `N` frames at once
pub const DUST_CADENCE_PHASES: u32 = 2;
/// The on-screen tolerance ε (px) of this track (the step-5 budget loop starts at 4 px)
pub const DUST_CADENCE_EPS_DEFAULT: f32 = 1.0;
/// a build / ahead-built list is aborted when its members' rate may exceed this many times the
/// class bound `ε / 2N`
pub const DUST_CADENCE_ABORT_FACTOR: f32 = 4.0;
/// frames from a submit to its readback (the lists' lengths)
pub const DUST_CADENCE_READBACK: u32 = 4;
/// frames of eye history the window term looks back over (`2 · N_max`)
pub const DUST_CADENCE_HISTORY: usize = 128;
/// a cluster whose nearest point comes within `3σ_max` of this many shell radii of the eye may be
/// walked by the speck pass (the shell can grow between classify and display): never in a class
pub const DUST_CADENCE_SHELL_MARGIN: f32 = 2.0;
/// orthographic zoom ratio between a frozen view and the current one beyond which the images are
/// dropped (the resampling's resolution)
pub const DUST_CADENCE_ORTHO_ZOOM_MAX: f32 = 1.25;
/// the near point of the rate moves this many `σ_max` towards the eye from the nearest edge: 0
/// (round 3: the 3σ shift explained none of the rate-rule test's outlier, 1.87 against 1.95, and
/// lowered its median ratio 0.969 → 0.953; `cadence_rate_near(.., 3)` stays the test's control)
pub const DUST_CADENCE_NEAR_SIGMAS: f32 = 0.0;
/// the age floor (s) of the growth terms `Δt / age`
pub const DUST_CADENCE_AGE_MIN_S: f32 = 1.0;
/// a simulated step more than this many times the previous frame's (or any positive step after a
/// non-positive one, or a negative step) resets every class ([`CadenceReset::Time`])
pub const DUST_CADENCE_TIME_JUMP_FACTOR: f64 = 4.0;
/// the default class-image border margin (a fraction of the viewport per side; 0 = no margin, the
/// per-cluster footprint rule)
pub const DUST_CADENCE_MARGIN_DEFAULT: f32 = 0.2;
/// the margin's granularity (px): levels ≤ 6 of an image stay texel-aligned with the frame's
pub const DUST_CADENCE_MARGIN_ALIGN: u32 = 64;
/// the guard (px) the motion budget and the coverage check keep inside the image's edge
pub const DUST_CADENCE_MARGIN_GUARD: f32 = 32.0;
/// the floor (m) of `σ_min` in the enclosing-cloud term
pub const DUST_CADENCE_SIGMA_MIN_M: f32 = 1.0e-3;
const _: () = assert!(DUST_CADENCE_CLASSES[0] >= 2 * DUST_CADENCE_READBACK);
const _: () = assert!(DUST_CADENCE_HISTORY >= 2 * DUST_CADENCE_CLASSES[2] as usize);

// ─── Owner byte ──────────────────────────────────────────────────────────────

/// The owner byte of a slot (one per ring slot per view): bits 0–1 the class id (0 = none,
/// `k + 1` = class index `k`), then the states [`OWN_SHOWN`], [`OWN_BUILD`], [`OWN_PENDING`] and
/// [`OWN_IN_IMAGE`]. The phase of a slot is its parity (`slot & 1` with two phases), so the lists
/// of the two phases of one class are disjoint by construction.
///
/// **Lists.** Per (tier, class `k`, phase `p`) three slot lists: *shown* (the slots the shown
/// image holds), *build* (the image being built this cycle) and *pending* (classified one cycle
/// ahead: the host must know its length, read back [`DUST_CADENCE_READBACK`] frames late, before
/// the cycle that builds it). An entry `e` of list `L(k, state)` is **valid** iff
/// `owner[e].class == k` and `owner[e]` has that state bit; every use (draw, promotion, swap)
/// skips invalid entries, so a retired or re-emitted slot's stale entries are never drawn or
/// promoted — the retire / emit hook only clears the owner bytes of the slots that left or entered
/// the live range (CPU mirror: from the tier ranges; GPU: `cmd_fill_buffer` over the same ranges).
/// A head moving back, a re-emission at an unchanged range, a capacity change or a tier change is
/// a reset (every list, byte and image cleared).
///
/// **Invariant** (after the swaps and classifies of every frame), for every slot `s`:
/// - I1: `s` carries at most one class id `k`, and valid entries of `s` exist only in the lists
///   of class `k` and phase `parity(s)`;
/// - I2: the state bits of `s` are exactly the valid lists holding it;
/// - I3: `s` is SHOWN only if the shown image of `(k, parity(s))` holds exactly one draw of the
///   cluster now in `s` (IN_IMAGE is set when this life is drawn into the build image, the byte is
///   cleared when the life ends, and SHOWN is set only at a swap, from a valid IN_IMAGE build
///   entry);
/// - I4: fresh = live ∧ ¬SHOWN.
///
/// *Base:* a reset empties every list and zeroes every byte. *Step:* a classify of `(k, p)` takes
/// only slots whose class is 0 or `k` (I1) and sets PENDING; the swap moves valid build entries
/// with IN_IMAGE to SHOWN and clears the old shown entries' SHOWN; the promotion moves valid
/// pending entries to BUILD; an abort clears BUILD / PENDING / IN_IMAGE; the class id is dropped
/// when no state bit is left; a list is built by one scan (no duplicate) and the build cursor
/// visits each entry once per cycle (one draw per life per image). None of these sets a bit of a
/// different class. *Hence* every live cluster is drawn exactly once per frame — fresh when not
/// SHOWN, else once, in the one image of its class and phase — nothing twice, nothing missed; a
/// dead life lingers in its image as a ghost for at most `2N` frames, and a reborn slot (its byte
/// cleared) is fresh until a build of its own takes it.
pub const OWN_CLASS_MASK: u8 = 0b11;
/// in the shown list of its class: drawn by the shown image, not by the fresh pass
pub const OWN_SHOWN: u8 = 1 << 2;
/// in the build list of its class (the image being built this cycle)
pub const OWN_BUILD: u8 = 1 << 3;
/// in the ahead-built (pending) list of its class
pub const OWN_PENDING: u8 = 1 << 4;
/// drawn into the build image this cycle (its whole footprint inside the frozen frame)
pub const OWN_IN_IMAGE: u8 = 1 << 5;

/// the class index (`0..`) of an owner byte, `None` when unowned
pub fn owner_class(b: u8) -> Option<usize> {
  match b & OWN_CLASS_MASK {
    0 => None,
    c => Some(c as usize - 1),
  }
}

/// an owner byte of class index `k` with the state bits `states`
pub fn owner_byte(k: usize, states: u8) -> u8 {
  debug_assert!(k < DUST_CADENCE_MAX_CLASSES);
  (k as u8 + 1) | (states & !OWN_CLASS_MASK)
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Which classes and phases a view runs ([`Self::for_device_local`]: the `dust_vqueue_records`
/// memory rule) and its tolerance.
#[derive(Debug, Clone, PartialEq)]
pub struct CadenceConfig {
  /// the class lengths `N` (frames), ascending, at most [`DUST_CADENCE_MAX_CLASSES`]
  pub classes: alloc::vec::Vec<u32>,
  /// 1 or 2 ([`DUST_CADENCE_PHASES`])
  pub phases: u32,
  /// ε (px)
  pub eps_px: f32,
  /// the class images' border margin, a fraction of the viewport per side ([`Self::margin_px`]);
  /// 0 = no margin: an image holds only clusters wholly inside its frozen frame (the footprint
  /// rule, D3)
  pub margin_frac: f32,
}

/// One class image's geometry: the margin (px per side) and the image's pyramid layout
/// (`W + 2 mx` × `H + 2 my`).
#[derive(Debug, Clone, PartialEq)]
pub struct ImageGeom {
  pub margin: [u32; 2],
  pub layout: PyramidLayout,
}

impl ImageGeom {
  pub fn has_margin(&self) -> bool {
    self.margin != [0, 0]
  }
}

impl CadenceConfig {
  /// The full configuration: [`DUST_CADENCE_CLASSES`] × [`DUST_CADENCE_PHASES`], ε default, the
  /// default margin.
  pub fn full() -> Self {
    Self {
      classes: DUST_CADENCE_CLASSES.to_vec(),
      phases: DUST_CADENCE_PHASES,
      eps_px: DUST_CADENCE_EPS_DEFAULT,
      margin_frac: DUST_CADENCE_MARGIN_DEFAULT,
    }
  }

  /// By the device-local memory, as `Device::dust_vqueue_records`: ≥ 4 GiB every class in two
  /// phases at M = 0.2 (12 class images of 1792 × 1184 at 1280×800, ~679 MB); 2–4 GiB one class
  /// of 16 frames in one phase at M = 0.1 (8 folds into fresh, 64 into 16; 2 images); below 2 GiB
  /// none (today's path).
  pub fn for_device_local(bytes: u64) -> Option<Self> {
    const GIB: u64 = 1 << 30;
    if bytes >= 4 * GIB {
      Some(Self::full())
    } else if bytes >= 2 * GIB {
      Some(Self {
        classes: alloc::vec![16],
        phases: 1,
        eps_px: DUST_CADENCE_EPS_DEFAULT,
        margin_frac: 0.1,
      })
    } else {
      None
    }
  }

  /// The margin per side (px) for a `w × h` viewport: `⌈M · W / 64⌉ · 64`, `⌈M · H / 64⌉ · 64`
  /// (0 when M ≤ 0). The 64-px granularity keeps the image's levels ≤ 6 texel-aligned with the
  /// frame's (an unchanged view reprojects exactly on them).
  pub fn margin_px(&self, w: u32, h: u32) -> [u32; 2] {
    if !(self.margin_frac > 0.0) {
      return [0, 0];
    }
    let a = DUST_CADENCE_MARGIN_ALIGN as f32;
    let m = |n: u32| {
      let q = self.margin_frac * n as f32 / a;
      let f = <f32 as FloatLike>::floor(q);
      (if q > f { f + 1.0 } else { f }) as u32 * DUST_CADENCE_MARGIN_ALIGN
    };
    [m(w), m(h)]
  }

  /// The class images' geometry for a `w × h` viewport: the viewport plus the margins, with **the
  /// frame's level count** ([`PyramidLayout::new_capped`]; the image's coarsest level may then be
  /// 2 × 1). A build draws by its image's level count (the top-level clamp, the volume bias, the
  /// gate level) and [`reproject_add`] reads only the frame's levels: with one level more
  /// (480 × 300 → 736 × 428, 11 levels against 10) the volume bias took the eye-inside clouds to
  /// the image's level 10, which nothing read, and the lab path lost ~96 % of the shown images'
  /// light (Σ ratio 0.9854 at frame 47; round 15, `cadence_image_keeps_the_frame_levels`).
  pub fn image_geom(&self, w: u32, h: u32) -> ImageGeom {
    let margin = self.margin_px(w, h);
    let levels = PyramidLayout::new(w, h).level_count();
    ImageGeom {
      margin,
      layout: PyramidLayout::new_capped(w + 2 * margin[0], h + 2 * margin[1], levels),
    }
  }

  /// [`Self::for_device_local`] with the knobs: `AETHERVK_DUST_CADENCE=0` off,
  /// `AETHERVK_DUST_CADENCE_PHASES=1|2`, `AETHERVK_DUST_CADENCE_EPS=<px>`.
  pub fn from_env(device_local_bytes: u64) -> Option<Self> {
    let var = aethervk_oshal_rlib::os::env::var;
    if var("AETHERVK_DUST_CADENCE").is_some_and(|v| v.trim() == "0") {
      return None;
    }
    let mut c = Self::for_device_local(device_local_bytes)?;
    if let Some(p) = var("AETHERVK_DUST_CADENCE_PHASES").and_then(|v| v.trim().parse::<u32>().ok()) {
      c.phases = p.clamp(1, 2);
    }
    if let Some(e) = var("AETHERVK_DUST_CADENCE_EPS").and_then(|v| v.trim().parse::<f32>().ok())
      && e > 0.0
      && e.is_finite()
    {
      c.eps_px = e;
    }
    // `AETHERVK_DUST_CADENCE_MARGIN=<fraction>` (0 = the footprint rule)
    if let Some(mf) = var("AETHERVK_DUST_CADENCE_MARGIN").and_then(|v| v.trim().parse::<f32>().ok())
      && mf >= 0.0
      && mf.is_finite()
    {
      c.margin_frac = mf.min(1.0);
    }
    Some(c)
  }

  /// the class images of this configuration (shown + build per class and phase)
  pub fn images(&self) -> usize {
    2 * self.classes.len() * self.phases as usize
  }

  /// bytes of the class images for the viewport `layout` (header + levels of the images' own
  /// layout with the margin: no measure grid, no queues)
  pub fn image_bytes(&self, layout: &PyramidLayout) -> u64 {
    let g = self.image_geom(layout.width, layout.height);
    self.images() as u64 * class_image_words(&g.layout) as u64 * 4
  }
}

/// words of one class image: the header and the levels (the measure grid is the frame's)
pub fn class_image_words(layout: &PyramidLayout) -> u32 {
  layout.measure.0
}

// ─── The view in world terms ─────────────────────────────────────────────────

/// A camera in world terms (a frozen view is one of these, kept in f64): the camera's view matrix
/// (`CameraRenderData::view_f64`, column-major; its translation column is ignored, the eye is
/// [`Self::eye_m`]), the projection (column-major), the heliocentric eye (m), the layer's units
/// per metre and the viewport. [`Self::mvp`] composes `Proj · ViewRot · Translate(anchor − eye) ·
/// Scale(units)` in f64 exactly as `DustDrawCall::mvp_params`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CadenceView {
  pub view_rot: [f64; 16],
  pub proj: [f64; 16],
  pub eye_m: [f64; 3],
  pub units_per_m: f64,
  pub width: u32,
  pub height: u32,
}

fn mul4(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
  let mut c = [0.0f64; 16];
  for col in 0..4 {
    for row in 0..4 {
      let mut s = 0.0;
      for k in 0..4 {
        s += a[k * 4 + row] * b[col * 4 + k];
      }
      c[col * 4 + row] = s;
    }
  }
  c
}

type M3 = [[f64; 3]; 3];

fn mul3(a: &M3, b: &M3) -> M3 {
  let mut c = [[0.0f64; 3]; 3];
  for (r, cr) in c.iter_mut().enumerate() {
    for (k, ck) in cr.iter_mut().enumerate() {
      *ck = a[r][0] * b[0][k] + a[r][1] * b[1][k] + a[r][2] * b[2][k];
    }
  }
  c
}

fn det3(m: &M3) -> f64 {
  m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn inv3(m: &M3) -> Option<M3> {
  let d = det3(m);
  if !(d.abs() > 0.0) || !d.is_finite() {
    return None;
  }
  let i = 1.0 / d;
  Some([
    [
      (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * i,
      (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * i,
      (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * i,
    ],
    [
      (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * i,
      (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * i,
      (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * i,
    ],
    [
      (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * i,
      (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * i,
      (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * i,
    ],
  ])
}

/// the identity map (current px → frozen px)
pub const MAP_IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

impl CadenceView {
  /// `Proj · ViewRot` (the rotation only, f64)
  pub fn view_proj(&self) -> [f64; 16] {
    let mut r = self.view_rot;
    r[12] = 0.0;
    r[13] = 0.0;
    r[14] = 0.0;
    r[15] = 1.0;
    mul4(&self.proj, &r)
  }

  /// particle-system local metres → clip for a system anchored at `anchor_m` (heliocentric m)
  pub fn mvp(&self, anchor_m: [f64; 3]) -> [f32; 16] {
    let u = self.units_per_m;
    let mut model = [0.0f64; 16];
    model[0] = u;
    model[5] = u;
    model[10] = u;
    for i in 0..3 {
      model[12 + i] = (anchor_m[i] - self.eye_m[i]) * u;
    }
    model[15] = 1.0;
    let m = mul4(&self.view_proj(), &model);
    let mut out = [0.0f32; 16];
    for (o, v) in out.iter_mut().zip(m) {
      *o = v as f32;
    }
    out
  }

  /// the speck pass's eye-relative mvp: `Proj · ViewRot · Scale(units)` without the translation
  /// (`DustDrawCall::mvp_params_eye_relative`)
  pub fn mvp_eye_relative(&self) -> [f32; 16] {
    let u = self.units_per_m;
    let mut model = [0.0f64; 16];
    model[0] = u;
    model[5] = u;
    model[10] = u;
    model[15] = 1.0;
    let m = mul4(&self.view_proj(), &model);
    let mut out = [0.0f32; 16];
    for (o, v) in out.iter_mut().zip(m) {
      *o = v as f32;
    }
    out
  }

  /// The class image's view of this (frozen viewport) view: the clip x, y rows scaled by
  /// `W / W'`, `H / H'` and the image's size `W' × H'` — the viewport's pixel scale with the
  /// margins centred (image px = viewport px + margin). Without a margin, this view itself.
  pub fn image_view(&self, geom: &ImageGeom) -> CadenceView {
    if !geom.has_margin() {
      return *self;
    }
    let (wi, hi) = (geom.layout.width, geom.layout.height);
    let (sx, sy) = (self.width as f64 / wi as f64, self.height as f64 / hi as f64);
    let mut proj = self.proj;
    for c in 0..4 {
      proj[c * 4] *= sx;
      proj[c * 4 + 1] *= sy;
    }
    CadenceView { proj, width: wi, height: hi, ..*self }
  }

  /// the image view's mvp for a system anchored at `anchor_m` ([`Self::image_view`])
  pub fn mvp_image(&self, anchor_m: [f64; 3], geom: &ImageGeom) -> [f32; 16] {
    self.image_view(geom).mvp(anchor_m)
  }

  /// The map from this view's pixels to the class image of `frozen` (a frozen viewport view):
  /// `T(mx, my) · map_from(frozen)` (the image's px are the frozen viewport's plus the margin;
  /// an unchanged view maps by the exact translation)
  pub fn map_to_image(&self, frozen: &CadenceView, geom: &ImageGeom) -> Option<[f32; 9]> {
    let mut h = self.map_from(frozen)?;
    if geom.has_margin() {
      let (mx, my) = (geom.margin[0] as f32, geom.margin[1] as f32);
      // rows x and y gain margin · row z
      for c in 0..3 {
        h[c] += mx * h[6 + c];
        h[3 + c] += my * h[6 + c];
      }
    }
    Some(h)
  }

  /// the eye in the local metres of a system anchored at `anchor_m`
  pub fn eye_local(&self, anchor_m: [f64; 3]) -> [f32; 3] {
    [
      (self.eye_m[0] - anchor_m[0]) as f32,
      (self.eye_m[1] - anchor_m[1]) as f32,
      (self.eye_m[2] - anchor_m[2]) as f32,
    ]
  }

  /// a perspective projection (the clip `w` row depends on the position)
  pub fn perspective(&self) -> bool {
    let vp = self.view_proj();
    vp[3] != 0.0 || vp[7] != 0.0 || vp[11] != 0.0
  }

  /// the 3×3 direction map of the `x`, `y`, `w` clip rows, then pixels:
  /// `S · [rows x, y, w of (Proj · ViewRot)]`, `S` = ndc → px
  fn pixel_direction_map(&self) -> M3 {
    let vp = self.view_proj();
    let row = |r: usize| [vp[r], vp[4 + r], vp[8 + r]];
    let (x, y, w) = (row(0), row(1), row(3));
    let (hw, hh) = (0.5 * self.width as f64, 0.5 * self.height as f64);
    let mut m = [[0.0f64; 3]; 3];
    for c in 0..3 {
      m[0][c] = hw * x[c] + hw * w[c];
      m[1][c] = hh * y[c] + hh * w[c];
      m[2][c] = w[c];
    }
    m
  }

  /// The map `H` (3×3, row-major, f32) from this view's pixel coordinates to `frozen`'s:
  /// `x_s = (H · (x, y, 1))_xy / (H · (x, y, 1))_z`. Perspective: through the directions
  /// (rotation and projection only; the translation parallax between the two eyes is what the
  /// rate rule bounds), `H = S M_s (S M)⁻¹`. Orthographic: the 2-D affine map between the two
  /// image planes (pan, zoom, roll), `None` when the view directions differ (any other rotation:
  /// a reset). `None` for different viewports or projection kinds. Bit-equal views map by the
  /// exact identity.
  pub fn map_from(&self, frozen: &CadenceView) -> Option<[f32; 9]> {
    if self.width != frozen.width || self.height != frozen.height {
      return None;
    }
    if self == frozen {
      return Some(MAP_IDENTITY);
    }
    let persp = self.perspective();
    if persp != frozen.perspective() {
      return None;
    }
    // perspective: the map is through the directions only (the eyes and units do not enter it)
    if persp && self.view_rot == frozen.view_rot && self.proj == frozen.proj {
      return Some(MAP_IDENTITY);
    }
    let h: M3 = if persp {
      let m = self.pixel_direction_map();
      let ms = frozen.pixel_direction_map();
      mul3(&ms, &inv3(&m)?)
    } else {
      // x = B p' + c with p' = p − eye_now (heliocentric m), B = S₂ · A_xy · u (2 × 3)
      let affine = |v: &CadenceView| -> ([[f64; 3]; 2], [f64; 2]) {
        let vp = v.view_proj();
        let (hw, hh) = (0.5 * v.width as f64, 0.5 * v.height as f64);
        let u = v.units_per_m;
        let mut b = [[0.0f64; 3]; 2];
        for c in 0..3 {
          b[0][c] = hw * vp[c * 4] * u;
          b[1][c] = hh * vp[c * 4 + 1] * u;
        }
        // the eye offset of `v` from the current eye
        let e = [self.eye_m[0] - v.eye_m[0], self.eye_m[1] - v.eye_m[1], self.eye_m[2] - v.eye_m[2]];
        let c = [
          hw * vp[12] + hw + b[0][0] * e[0] + b[0][1] * e[1] + b[0][2] * e[2],
          hh * vp[13] + hh + b[1][0] * e[0] + b[1][1] * e[1] + b[1][2] * e[2],
        ];
        (b, c)
      };
      let (b, c) = affine(self);
      let (bs, cs) = affine(frozen);
      // G = B_s Bᵀ (B Bᵀ)⁻¹
      let dot = |x: &[f64; 3], y: &[f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
      let gram = [[dot(&b[0], &b[0]), dot(&b[0], &b[1])], [dot(&b[1], &b[0]), dot(&b[1], &b[1])]];
      let gd = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0];
      if !(gd.abs() > 0.0) {
        return None;
      }
      let gi = [[gram[1][1] / gd, -gram[0][1] / gd], [-gram[1][0] / gd, gram[0][0] / gd]];
      let mut g = [[0.0f64; 2]; 2];
      for r in 0..2 {
        let p = [dot(&bs[r], &b[0]), dot(&bs[r], &b[1])];
        for k in 0..2 {
          g[r][k] = p[0] * gi[0][k] + p[1] * gi[1][k];
        }
      }
      // the same view direction: B_s = G B (else a rotation out of the image plane)
      let mut err = 0.0f64;
      let mut scale = 0.0f64;
      for r in 0..2 {
        for k in 0..3 {
          let v = g[r][0] * b[0][k] + g[r][1] * b[1][k];
          err = err.max((v - bs[r][k]).abs());
          scale = scale.max(bs[r][k].abs());
        }
      }
      if !(err <= 1e-6 * scale) {
        return None;
      }
      [
        [g[0][0], g[0][1], cs[0] - (g[0][0] * c[0] + g[0][1] * c[1])],
        [g[1][0], g[1][1], cs[1] - (g[1][0] * c[0] + g[1][1] * c[1])],
        [0.0, 0.0, 1.0],
      ]
    };
    let mut out = [0.0f32; 9];
    for r in 0..3 {
      for c in 0..3 {
        out[r * 3 + c] = h[r][c] as f32;
      }
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
  }
}

// ─── The rate rule ───────────────────────────────────────────────────────────

/// The per-frame inputs of [`cadence_rate`] (host, f64 differenced then f32).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateInputs {
  /// simulated seconds per frame `Δt = t_now(f) − t_now(f−1)` (s)
  pub dt_s: f32,
  /// the eye's heliocentric step this frame `Δeye = eye_m(f) − eye_m(f−1)` (m; particle-system
  /// axes = root axes)
  pub d_eye_m: [f32; 3],
  /// the window term per class `a_w(N_k)` (m): `max` over the last `2N_k` frames of
  /// `|Δe(f′) − Δe(f)|`, `Δe` the eye's step relative to the nucleus
  /// (`DustDrawCall::nucleus_m`), the camera's own change of motion over the window
  pub a_w_m: [f32; DUST_CADENCE_MAX_CLASSES],
  /// ε (px)
  pub eps_px: f32,
  /// `max(width, height)` (px): the enclosing term's scale of a relative change
  pub frame_px: f32,
  /// the by-count fallback's shape bound θ (0: off, the rule without it): class `k` also needs
  /// `2N_k · Δt ≤ θ · age`, the age floored at [`DUST_CADENCE_AGE_MIN_S`]
  /// ([`DUST_CADENCE_FALLBACK_SHAPE`], [`BudgetKnobs::shape_frac`])
  pub shape_frac: f32,
}

/// What [`cadence_rate`] says of one cluster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterRate {
  /// δ_k (px per frame) per class, the window term of class `k` included
  pub delta_px: [f32; DUST_CADENCE_MAX_CLASSES],
  /// the rate's sensitivity to a change of the eye's step (px per m): the Jacobian's Frobenius
  /// norm at the near point, or `frame_px / σ_min` for a cloud enclosing the eye (the abort
  /// test's `g_max` is its maximum over a list)
  pub g_px_per_m: f32,
  pub enclosing: bool,
  /// 0 = fresh, `k + 1` = class index `k` (the largest with `δ_k · 2N_k ≤ ε` and, with a shape
  /// bound θ, `2N_k · Δt ≤ θ · age`)
  pub class: u8,
}

/// `(λ_min, λ_max)` of a symmetric 3×3 (xx, xy, xz, yy, yz, zz), closed form (trigonometric)
pub fn sym3_eigen_range(c: [f32; 6]) -> (f32, f32) {
  let (a, b, d, e, f, g) = (c[0], c[1], c[2], c[3], c[4], c[5]);
  // [[a b d] [b e f] [d f g]]
  let p1 = b * b + d * d + f * f;
  if !(p1 > 0.0) {
    let lo = a.min(e).min(g);
    let hi = a.max(e).max(g);
    return (lo, hi);
  }
  let q = (a + e + g) / 3.0;
  let p2 = (a - q) * (a - q) + (e - q) * (e - q) + (g - q) * (g - q) + 2.0 * p1;
  let p = <f32 as FloatLike>::sqrt(p2 / 6.0);
  if !(p > 0.0) {
    return (q, q);
  }
  let ip = 1.0 / p;
  let (ba, be, bg, bb, bd, bf) = ((a - q) * ip, (e - q) * ip, (g - q) * ip, b * ip, d * ip, f * ip);
  let det = ba * (be * bg - bf * bf) - bb * (bb * bg - bf * bd) + bd * (bb * bf - be * bd);
  let r = (0.5 * det).clamp(-1.0, 1.0);
  let phi = <f32 as FloatLike>::acos(r) / 3.0;
  let hi = q + 2.0 * p * <f32 as FloatLike>::cos(phi);
  let lo = q + 2.0 * p * <f32 as FloatLike>::cos(phi + 2.0 * core::f32::consts::PI / 3.0);
  (lo.min(hi), hi.max(lo))
}

fn fabs(x: f32) -> f32 {
  if x < 0.0 { -x } else { x }
}
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn len3(a: [f32; 3]) -> f32 {
  <f32 as FloatLike>::sqrt(dot3(a, a))
}

/// The on-screen rate of cluster `m` per frame (px) under `mvp` / `eye_local`, per class (plan
/// item 2, vector form):
///
/// - the near point: the polyline edge of the smallest depth along the view, moved
///   [`DUST_CADENCE_NEAR_SIGMAS`] `· σ_max` towards the eye (0 today; at most 0.9 of its
///   distance; the mean when orthographic); `s = v·Δt − Δeye` (m) with `v` the mean's heliocentric velocity
///   ([`DustMoments::helio_velocity`]; local axes = root axes) — orthographic `s = v·Δt` (the
///   reprojection's affine map takes the pan);
/// - `d1 = |px(near + s) − px(near)|` (px; ∞ when either is behind the camera);
/// - `ext = max(polyline extent, time segment)` (px), the time segment only from `pred` (the
///   caller passes the predecessor only when both were propagated at the same time);
///   `d2 = ext · Δt / age` (px · s / s), `d3 = |ω| · Δt · ext` (rad/s · s · px);
/// - a cloud enclosing the eye (perspective, and its polyline reaches behind the camera or
///   `|mean − eye|² < 9 λ_max(Σ)`): `d4 = (|s| / σ_min + Δt / age) · frame_px` (m / m · px),
///   `σ_min = √λ_min(Σ)` floored at [`DUST_CADENCE_SIGMA_MIN_M`] — instead of d1–d3 when the
///   polyline reaches behind the camera, added to them otherwise;
/// - `δ_k = d1 + d2 + d3 + d4 + a_w(N_k) · g`, `g` (px/m) the Jacobian's Frobenius norm at the
///   near point, `frame_px / σ_min` when enclosing (a cloud around the eye whose smallest axis
///   is below [`DUST_CADENCE_SIGMA_MIN_M`] is fresh);
/// - the class: the largest `k` with `δ_k · 2N_k ≤ ε` and, with the by-count fallback's shape
///   bound θ = [`RateInputs::shape_frac`] `> 0`, `2N_k · Δt ≤ θ · age` (the age as in d2), else
///   fresh. A cluster without flux or with a non-finite rate is fresh. θ = 0 (off the fallback):
///   the shape bound passes every class, the rule is the ε rule alone.
#[allow(clippy::too_many_arguments)]
pub fn cadence_rate(
  m: &DustMoments,
  pred: Option<&DustMoments>,
  mvp: &[f32; 16],
  eye_local: [f32; 3],
  width: u32,
  height: u32,
  classes: &[u32],
  inp: &RateInputs,
) -> ClusterRate {
  cadence_rate_near(m, pred, mvp, eye_local, width, height, classes, inp, DUST_CADENCE_NEAR_SIGMAS)
}

/// [`cadence_rate`] with the near point moved `near_sigmas · σ_max` towards the eye (0: the
/// nearest edge itself, the round-1 rule; the rate-rule test prints both)
#[allow(clippy::too_many_arguments)]
pub fn cadence_rate_near(
  m: &DustMoments,
  pred: Option<&DustMoments>,
  mvp: &[f32; 16],
  eye_local: [f32; 3],
  width: u32,
  height: u32,
  classes: &[u32],
  inp: &RateInputs,
  near_sigmas: f32,
) -> ClusterRate {
  let nc = classes.len().min(DUST_CADENCE_MAX_CLASSES);
  let fresh = ClusterRate {
    delta_px: [f32::INFINITY; DUST_CADENCE_MAX_CLASSES],
    g_px_per_m: f32::INFINITY,
    enclosing: false,
    class: 0,
  };
  if !(m.flux() > 0.0) {
    return fresh;
  }
  let view = SplitView::of(mvp, eye_local);
  let px = |q: [f32; 3]| project_point(q, mvp, width, height);
  // the near point
  let mut near = m.mean();
  if view.perspective {
    let mut best = f32::INFINITY;
    for j in 0..SIZE_EDGES as usize {
      let e = m.edge(j);
      let d = dot3(sub3(e, eye_local), view.fwd);
      if d < best {
        best = d;
        near = e;
      }
    }
  }
  // the Gaussian spread puts light nearer to the eye than the nearest edge, by up to 3σ_max along
  // the line of sight, and nearer light moves more: the near point moves that far towards the
  // eye (never past a tenth of its distance; the enclosing test's 3σ convention)
  let (lam_lo, lam_hi) = sym3_eigen_range(m.cov_v());
  if view.perspective && near_sigmas > 0.0 {
    let to_eye = sub3(eye_local, near);
    let r = len3(to_eye);
    if r > 0.0 {
      let l = (near_sigmas * <f32 as FloatLike>::sqrt(lam_hi.max(0.0))).min(0.9 * r);
      near = [near[0] + to_eye[0] / r * l, near[1] + to_eye[1] / r * l, near[2] + to_eye[2] / r * l];
    }
  }
  let v = m.helio_velocity();
  let dt = inp.dt_s;
  let s = if view.perspective {
    [v[0] * dt - inp.d_eye_m[0], v[1] * dt - inp.d_eye_m[1], v[2] * dt - inp.d_eye_m[2]]
  } else {
    [v[0] * dt, v[1] * dt, v[2] * dt]
  };
  let sn = len3(s);
  let d1 = match (px(near), px([near[0] + s[0], near[1] + s[1], near[2] + s[2]])) {
    (Some(a), Some(b)) => <f32 as FloatLike>::sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])),
    _ => f32::INFINITY,
  };
  // the Jacobian at the near point (px per m along each axis; mirror of project_packet_lit's)
  let mm = |r: usize, c: usize| mvp[c * 4 + r];
  let row = |r: usize| [mm(r, 0), mm(r, 1), mm(r, 2)];
  let (r0, r1, r3) = (row(0), row(1), row(3));
  let w = dot3(r3, near) + mm(3, 3);
  let mut g = if w > 0.0 {
    let nx = (dot3(r0, near) + mm(0, 3)) / w;
    let ny = (dot3(r1, near) + mm(1, 3)) / w;
    let kx = 0.5 * width as f32 / w;
    let ky = 0.5 * height as f32 / w;
    let jx = [(r0[0] - r3[0] * nx) * kx, (r0[1] - r3[1] * nx) * kx, (r0[2] - r3[2] * nx) * kx];
    let jy = [(r1[0] - r3[0] * ny) * ky, (r1[1] - r3[1] * ny) * ky, (r1[2] - r3[2] * ny) * ky];
    <f32 as FloatLike>::sqrt(dot3(jx, jx) + dot3(jy, jy))
  } else {
    f32::INFINITY
  };
  let ext_poly = polyline_extent_px(m, &px);
  let tseg = super::splat::time_segment_px(m, pred, &px);
  let ext = ext_poly.map(|e| tseg.map_or(e, |t| e.max(t)));
  let age = m.age().max(DUST_CADENCE_AGE_MIN_S);
  let sigma_min = <f32 as FloatLike>::sqrt(lam_lo.max(0.0)).max(DUST_CADENCE_SIGMA_MIN_M);
  let dm = sub3(m.mean(), eye_local);
  let enclosing = view.perspective && (ext.is_none() || dot3(dm, dm) < 9.0 * lam_hi.max(0.0));
  let d4 = if enclosing {
    (sn / sigma_min + dt / age) * inp.frame_px
  } else {
    0.0
  };
  if enclosing {
    // a cloud around the eye changes by its relative shift, not by a parallax: its sensitivity is
    // frame_px / σ_min with σ_min its own smallest axis (the geometric Jacobian of a near edge
    // just in front of the eye plane is meaningless here, 8e8 px/m in round 6); a cloud with no
    // measurable smallest axis cannot be bounded and stays fresh
    if !(lam_lo > DUST_CADENCE_SIGMA_MIN_M * DUST_CADENCE_SIGMA_MIN_M) {
      return fresh;
    }
    g = inp.frame_px / sigma_min;
  }
  let base = match ext {
    Some(e) => {
      let d2 = e * dt / age;
      let d3 = fabs(m.spin_rate()) * dt * e;
      d1 + d2 + d3 + d4
    }
    None => d4,
  };
  let mut out = ClusterRate {
    delta_px: [f32::INFINITY; DUST_CADENCE_MAX_CLASSES],
    g_px_per_m: g,
    enclosing,
    class: 0,
  };
  // the by-count fallback's shape bound (θ = 0: off, every class passes it); the GPU mirror
  // (dust_cad_rate) evaluates the same two f32 products and the same comparison
  let shape_on = inp.shape_frac > 0.0;
  let th_age = inp.shape_frac * age;
  for k in 0..nc {
    let dk = base + inp.a_w_m[k] * g;
    out.delta_px[k] = if dk.is_finite() { dk } else { f32::INFINITY };
    let shape_ok = !shape_on || 2.0 * classes[k] as f32 * dt <= th_age;
    if dk.is_finite() && dk * 2.0 * classes[k] as f32 <= inp.eps_px && shape_ok {
      out.class = k as u8 + 1;
    }
  }
  out
}

/// Whether the speck pass may walk cluster `m` before its class image is shown (the eligibility
/// rule of the speck shell): its nearest point (the mean or an edge) comes within
/// `3σ_max + DUST_CADENCE_SHELL_MARGIN · d_max` of the eye. Such a cluster stays fresh, so its
/// continuum and its specks are always drawn in the same frame with the same hand-over. `d_max ≤ 0`
/// (no shell): never.
pub fn shell_capable(m: &DustMoments, eye_local: [f32; 3], d_max: f32) -> bool {
  if !(d_max > 0.0) {
    return false;
  }
  let mut d = len3(sub3(m.mean(), eye_local));
  for j in 0..SIZE_EDGES as usize {
    d = d.min(len3(sub3(m.edge(j), eye_local)));
  }
  let (_, lam_hi) = sym3_eigen_range(m.cov_v());
  let sigma_max = <f32 as FloatLike>::sqrt(lam_hi.max(0.0));
  d - 3.0 * sigma_max < DUST_CADENCE_SHELL_MARGIN * d_max
}

// ─── Reprojection ────────────────────────────────────────────────────────────

/// The factor a reprojected texel's words take besides the unit rescale. **Production is
/// [`ReprojectFactor::Unit`]**, decided by `reprojection_is_exact_for_a_rotation` (2026-10-17:
/// Σ ratio 0.9984–0.99984 at yaw 1–15°, pitch + roll and the orthographic roll + pan + zoom;
/// `SolidAngle` lost 1–17 % of the light); `SolidAngle` stays as that test's printed control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReprojectFactor {
  /// none: the counts per px² are the per-ray optical depth (`amp ∝ √det(J Jᵀ)`, px² per m²
  /// across the ray), a function of the direction alone
  #[default]
  Unit,
  /// `Ω_new / Ω_s = |det ∂x_s/∂x| = det H / h_z³`: the frozen pixels mapped into one new pixel
  /// (flux conserving)
  SolidAngle,
}

/// One shown class image as [`reproject_add`] reads it.
#[derive(Debug, Clone, Copy)]
pub struct ShownImage<'a> {
  /// the image's words (header + levels of its own layout)
  pub words: &'a [u32],
  /// current px → the image's px ([`CadenceView::map_to_image`])
  pub map: [f32; 9],
  /// the fixed-point unit the image was drawn with
  pub unit: f32,
  /// the image's layout (`None`: the frame's, no margin)
  pub layout: Option<&'a PyramidLayout>,
}

/// Whether a class image still covers the viewport: the four viewport corners mapped by `map`
/// (current px → image px) lie inside the image's guarded interior `[G, W' − G] × [G, H' − G]`
/// ([`DUST_CADENCE_MARGIN_GUARD`]); behind the eye = not covered
pub fn image_covers(map: &[f32; 9], view_w: u32, view_h: u32, image: &PyramidLayout) -> bool {
  let g = DUST_CADENCE_MARGIN_GUARD;
  let (wi, hi) = (image.width as f32, image.height as f32);
  for (x, y) in [(0.0f32, 0.0f32), (view_w as f32, 0.0), (0.0, view_h as f32), (view_w as f32, view_h as f32)] {
    let hz = map[6] * x + map[7] * y + map[8];
    if !(hz > 0.0) {
      return false;
    }
    let (xs, ys) = ((map[0] * x + map[1] * y + map[2]) / hz, (map[3] * x + map[4] * y + map[5]) / hz);
    if !(xs >= g && ys >= g && xs <= wi - g && ys <= hi - g) {
      return false;
    }
  }
  true
}

/// The per-axis maximum displacement (px) of the four viewport corners under `map` (viewport px →
/// viewport px of another view; ∞ behind the eye): the motion budget's measure
pub fn corner_motion(map: &[f32; 9], view_w: u32, view_h: u32) -> [f32; 2] {
  let mut d = [0.0f32; 2];
  for (x, y) in [(0.0f32, 0.0f32), (view_w as f32, 0.0), (0.0, view_h as f32), (view_w as f32, view_h as f32)] {
    let hz = map[6] * x + map[7] * y + map[8];
    if !(hz > 0.0) {
      return [f32::INFINITY; 2];
    }
    let (xs, ys) = ((map[0] * x + map[1] * y + map[2]) / hz, (map[3] * x + map[4] * y + map[5]) / hz);
    d[0] = d[0].max(fabs(xs - x));
    d[1] = d[1].max(fabs(ys - y));
  }
  d
}

/// `⌊v + u⌋` with `u = segment_carry(seed, key) + ½ ∈ [0, 1)` (unbiased for faint texels),
/// saturating; exact for an integer `v` (the fraction is 0 and `u < 1`)
pub fn dither_round(v: f32, seed: u32, key: u32) -> u32 {
  if !(v > 0.0) {
    return 0;
  }
  if v >= 4_294_967_040.0 {
    return u32::MAX;
  }
  let fl = <f32 as FloatLike>::floor(v);
  let frac = v - fl;
  let u = segment_carry(seed, key) + 0.5;
  (fl as u32).saturating_add((frac + u >= 1.0) as u32)
}

/// Reproject-add (plan item 5; mirror of `dust_reproject.comp`, next track): for every frame
/// texel `(ℓ, tx, ty)` of a level any image holds, the texel centre `((t + ½) · 2^ℓ)` px is
/// mapped by each image's `H` to the image's pixels (skipped when `h_z ≤ 0`), its four bilinear
/// taps on the image's level `ℓ` (the composite's weights and its edge clamp; a pre-image outside
/// the frozen level's grid `[0, w·2^ℓ] × [0, h·2^ℓ]` takes nothing from that image) are summed over the images for the count and the three colour words, each word
/// × `u_s / u_now` (× [`ReprojectFactor::SolidAngle`]'s factor in that variant), rounded once by
/// [`dither_round`] (seed `seed`, key the frame word's index), and **stored** (one writer per
/// texel: the frame is cleared and nothing is splatted yet); the depth word is the minimum
/// non-zero depth of the weighted taps with counts. The frame header folds every image's
/// `PYR_TAU_MAX` by maximum and `PYR_LEVEL_MASK` by or (the unit follows the whole image). The
/// measure grid is not written (it is computed from the finished pyramid). Only frozen build
/// images are ever passed (never a reprojected one: no compounding).
pub fn reproject_add(
  layout: &PyramidLayout,
  frame: &mut [u32],
  images: &[ShownImage<'_>],
  unit_now: f32,
  seed: u32,
  factor: ReprojectFactor,
) {
  if images.is_empty() || !(unit_now > 0.0) {
    return;
  }
  let hn = PYRAMID_HEADER_WORDS as usize;
  let tw = PYRAMID_TEXEL_WORDS as usize;
  let mut any_mask = 0u32;
  // per image: mask, unit ratio, det H
  let prep: alloc::vec::Vec<(u32, f32, f32)> = images
    .iter()
    .map(|im| {
      let mask = if im.words.len() >= hn { im.words[PYR_LEVEL_MASK as usize] } else { 0 };
      any_mask |= mask;
      let h = &im.map;
      let det = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6])
        + h[2] * (h[3] * h[7] - h[4] * h[6]);
      let ratio = if im.unit > 0.0 { im.unit / unit_now } else { 0.0 };
      (mask, ratio, det)
    })
    .collect();
  for (l, &(off, w, hgt)) in layout.levels.iter().enumerate() {
    if any_mask & (1 << l) == 0 {
      continue;
    }
    let s = (1u32 << l) as f32;
    for ty in 0..hgt {
      for tx in 0..w {
        let (cx, cy) = ((tx as f32 + 0.5) * s, (ty as f32 + 0.5) * s);
        let mut v = [0.0f32; 4];
        let mut depth = 0u32;
        let mut hit = false;
        for (im, &(mask, ratio, det)) in images.iter().zip(&prep) {
          if mask & (1 << l) == 0 || !(ratio > 0.0) {
            continue;
          }
          // the image's own level ℓ (its layout with the margin; the frame's without)
          let (ioff, iw, ih) = match im.layout {
            Some(lay) => match lay.levels.get(l) {
              Some(&lv) => lv,
              None => continue,
            },
            None => (off, w, hgt),
          };
          let h = &im.map;
          let hx = h[0] * cx + h[1] * cy + h[2];
          let hy = h[3] * cx + h[4] * cy + h[5];
          let hz = h[6] * cx + h[7] * cy + h[8];
          if !(hz > 0.0) {
            continue;
          }
          let (xs, ys) = (hx / hz, hy / hz);
          // only inside the frozen level's grid (the frame, up to its last partial texel): beyond
          // it the image holds nothing (a cluster off its frame at the build is not in it);
          // inside, the taps clamp at the grid's edge exactly as the composite's do
          // (`composite_sample`), so a coarse level's texel 4–8 px inside the edge reads what the
          // frozen image displays there
          if !(xs >= 0.0 && ys >= 0.0 && xs <= iw as f32 * s && ys <= ih as f32 * s) {
            continue;
          }
          let k = match factor {
            ReprojectFactor::Unit => ratio,
            ReprojectFactor::SolidAngle => ratio * fabs(det / (hz * hz * hz)),
          };
          let fx = xs / s - 0.5;
          let fy = ys / s - 0.5;
          let (x0, y0) = (<f32 as FloatLike>::floor(fx), <f32 as FloatLike>::floor(fy));
          if !(x0.is_finite() && y0.is_finite()) {
            continue;
          }
          let (ax, ay) = (fx - x0, fy - y0);
          for (dx, dy, wgt) in [
            (0i64, 0i64, (1.0 - ax) * (1.0 - ay)),
            (1, 0, ax * (1.0 - ay)),
            (0, 1, (1.0 - ax) * ay),
            (1, 1, ax * ay),
          ] {
            if !(wgt > 0.0) {
              continue;
            }
            let sx = (x0 as i64 + dx).clamp(0, iw as i64 - 1);
            let sy = (y0 as i64 + dy).clamp(0, ih as i64 - 1);
            let i = (ioff + (sy as u32 * iw + sx as u32) * PYRAMID_TEXEL_WORDS) as usize;
            let c = im.words[i];
            if c == 0 {
              continue;
            }
            hit = true;
            let wk = wgt * k;
            for q in 0..4 {
              v[q] += wk * im.words[i + q] as f32;
            }
            let d = im.words[i + 4];
            if d != 0 && (depth == 0 || d < depth) {
              depth = d;
            }
          }
        }
        if !hit {
          continue;
        }
        let i = (off + (ty * w + tx) * PYRAMID_TEXEL_WORDS) as usize;
        for q in 0..4 {
          frame[i + q] = dither_round(v[q], seed, (i + q) as u32);
        }
        frame[i + tw - 1] = depth;
      }
    }
  }
  // the header folds
  for (im, &(mask, _, _)) in images.iter().zip(&prep) {
    let t = im.words[PYR_TAU_MAX as usize];
    if f32::from_bits(t) > f32::from_bits(frame[PYR_TAU_MAX as usize]) {
      frame[PYR_TAU_MAX as usize] = t;
    }
    frame[PYR_LEVEL_MASK as usize] |= mask;
  }
}

// ─── The host state machine ──────────────────────────────────────────────────

/// Why a frame reset every class (everything fresh that frame; lists, owner bytes and images
/// cleared; the cycles restart at frame 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CadenceReset {
  /// no previous frame
  First,
  /// `DUST_VIEW_FLOW` or `DUST_VIEW_TRACERS` set: the cadence is off while they are
  Off,
  Viewport,
  /// perspective: the projection's x / y / w rows changed (fov, aspect), or the projection kind
  Projection,
  /// orthographic: a rotation other than a roll
  OrthoDirection,
  /// orthographic: a zoom beyond [`DUST_CADENCE_ORTHO_ZOOM_MAX`] against a frozen view
  OrthoZoom,
  /// the simulation speed changed
  Speed,
  /// a jump in simulated time: the step `Δt` went negative, turned positive after a non-positive
  /// one (a resume, a seek from a still series), or grew beyond
  /// [`DUST_CADENCE_TIME_JUMP_FACTOR`] × the previous frame's (the class images were built for the
  /// old pace: kept, they would show a day-old ghost after a day's step)
  Time,
  /// the view flags changed
  Flags,
  /// the non-view draw inputs changed (colours, exposure, albedo, TTL, LOD, size law, knobs, ε)
  Fingerprint,
  /// a tier appeared or vanished
  TierSet,
  /// a tier's head moved back (a rewind, a restore) — the tier's key
  Range(u64),
  /// `emitted` changed with nothing appended and no rewritten slot named (`TierMeta::rewritten_from`)
  Reemit(u64),
  Capacity(u64),
}

/// What a frame is for the cadence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CadenceMode {
  /// a reused frame: nothing advances (builds pause)
  Paused,
  /// the cadence is off (flow marks or tracers): everything fresh
  Off,
  /// the first frame: everything fresh, no cycle yet
  First,
  Run,
}

/// The frame's view-level inputs (host). `nucleus_m` is required (the window term's reference:
/// `DustDrawCall::nucleus_m`; a test passing zero says so).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMeta {
  pub view: CadenceView,
  pub t_now_s: f64,
  pub speed: f64,
  pub nucleus_m: [f64; 3],
  pub flags: u32,
  pub fingerprint: u64,
  pub reused: bool,
  /// the fixed-point unit this frame is drawn with
  pub unit_now: f32,
}

/// One tier at the frame's begin (before its propagate): its range and render words.
#[derive(Debug, Clone, Copy)]
pub struct TierMeta<'a> {
  pub key: u64,
  pub first_slot: u32,
  pub capacity: u32,
  pub live_count: u32,
  /// the newest emission the range includes (`DustDrawState::compute_wait`)
  pub emitted: u64,
  /// the compact offset from which the slots hold content written or rewritten since the
  /// previous frame: the first live batch whose ready value exceeds the previous frame's
  /// `emitted` (the old live count less the retired for a pure append; the provisional open
  /// window, re-emitted longer every tick, starts there); `live_count` when nothing is new
  pub rewritten_from: u32,
  pub anchor_m: [f64; 3],
  /// compact, ring order (offset `o` ↔ slot `(first_slot + o) & (capacity − 1)`)
  pub render: &'a [DustRenderCluster],
}

/// One tier at the frame's draw (after the propagate of [`CadencePlan::propagate`]), same order as
/// the [`TierMeta`]s.
pub struct TierDraw<'a> {
  pub moments: &'a mut [DustMoments],
  pub render: &'a [DustRenderCluster],
  /// the current view's push constants (mvp, eye, `inv_unit = 1 / unit_now`)
  pub pc: DustSplatPushConstants,
  pub flow: DustFlowUniform,
  /// the speck pass's push constants (`None`: no specks this frame)
  pub specks_pc: Option<DustSpecksPushConstants>,
  /// the propagate generation this frame's propagate wrote into the moments it evaluated
  /// (`DustMoments::propagate_stamp`; 0 on the CPU device path): with the tier's anchor it keys
  /// the anchor history that shifts a stale slot to the current anchor
  pub stamp: u32,
}

/// `m` with its positions (mean, the 17 edges) shifted by `d` (m): a stale moments record, local
/// to the anchor of its propagate, re-expressed against the current anchor (`local = helio −
/// anchor`, so `local_now = local_then + (anchor_then − anchor_now)`). The delta is the host's f64
/// difference of the two anchors cast once to f32 per generation, added in f32 (both mirrors: the
/// GPU classify adds the same f32 words); covariances and velocities are unchanged
pub fn shifted_moments(m: &DustMoments, d: [f32; 3]) -> DustMoments {
  let mut out = *m;
  for i in 0..3 {
    out.mean_flux[i] = m.mean_flux[i] + d[i];
    for j in 0..SIZE_EDGES as usize {
      out.edges[j][i] = m.edges[j][i] + d[i];
    }
  }
  out
}

/// the per-generation f32 delta of two anchors (host, f64 difference cast once)
pub fn anchor_delta_f32(anchor_then: [f64; 3], anchor_now: [f64; 3]) -> [f32; 3] {
  [
    (anchor_then[0] - anchor_now[0]) as f32,
    (anchor_then[1] - anchor_now[1]) as f32,
    (anchor_then[2] - anchor_now[2]) as f32,
  ]
}

/// An aborted list ([`DustCadenceState::begin`], F1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CadenceAbort {
  pub k: usize,
  pub p: usize,
  /// 0 = the shown image (dropped: its members fresh), 1 = the build list (its image cleared),
  /// 2 = the pending list
  pub list: u8,
  /// [`ABORT_RATE`], [`ABORT_BUDGET`], [`ABORT_PREDICTION`] or [`ABORT_COVERAGE`]
  pub reason: u8,
  pub entries: u32,
  /// `|Δe − Δe_c| · g_max` (px) against the bound `(ABORT − 1) · ε / 2N`
  pub excess_px: f32,
  pub bound_px: f32,
  /// the list's largest member sensitivity (px per m) that fired
  pub g_max: f32,
  /// the shown image kept after the abort: its entries and the frames until its swap (the ghost)
  pub shown_entries: u32,
  pub ghost_frames_left: u32,
}

/// an abort by the rate rule (`|Δe − Δe_c| · g_max` beyond the class bound)
pub const ABORT_RATE: u8 = 0;
/// the class's motion budget is exceeded (`2N · U(N) > m − G`)
pub const ABORT_BUDGET: u8 = 1;
/// a build's predicted displacement at the end of its display exceeds `m − G`
pub const ABORT_PREDICTION: u8 = 2;
/// a shown image no longer covers the viewport ([`image_covers`])
pub const ABORT_COVERAGE: u8 = 3;

/// What [`DustCadenceState::begin`] decided: the mode, a reset, the aborts and swaps, and the
/// propagate list per tier (compact offsets: fresh ∪ build chunks ∪ their predecessors).
#[derive(Debug, Clone, Default)]
pub struct CadencePlan {
  pub mode: Option<CadenceMode>,
  pub reset: Option<CadenceReset>,
  pub aborts: alloc::vec::Vec<CadenceAbort>,
  pub swaps: alloc::vec::Vec<(usize, usize)>,
  pub propagate: alloc::vec::Vec<alloc::vec::Vec<u32>>,
}

/// One build entry drawn into its class image this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CadenceDrawn {
  pub tier: usize,
  pub k: usize,
  pub p: usize,
  pub offset: u32,
  /// `Σ counts · 4^ℓ` it added to the image (only with [`DustCadenceState::trace_light`]; else 0)
  pub light: f64,
}

/// The coverage of one (class, phase) at its cycle start: the build that just ended (now shown) —
/// its entries D3 rejected and drew — and the share of the frame's light the shown images hold:
/// the light they reproject into this frame over the frame's (the images' margin band, outside
/// the viewport, does not count).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CadenceCycle {
  pub k: usize,
  pub p: usize,
  pub d3_rejected: u32,
  pub drawn: u32,
  pub shown_light_share: f64,
}

/// What a frame did ([`DustCadenceState::finish`]).
#[derive(Debug, Clone, Default)]
pub struct CadenceReport {
  pub mode: Option<CadenceMode>,
  pub reset: Option<CadenceReset>,
  pub aborts: alloc::vec::Vec<CadenceAbort>,
  pub swaps: alloc::vec::Vec<(usize, usize)>,
  /// per tier: the compact offsets drawn fresh
  pub fresh: alloc::vec::Vec<alloc::vec::Vec<u32>>,
  pub drawn_build: alloc::vec::Vec<CadenceDrawn>,
  /// `(k, p, entries)` of each classify of this frame
  pub classified: alloc::vec::Vec<(usize, usize, u32)>,
  /// the admissible-class histogram of the classifies (0 = fresh, k + 1)
  pub class_hist: [u32; DUST_CADENCE_MAX_CLASSES + 1],
  /// build entries refused by the footprint rule (D3: not wholly inside the frozen frame)
  pub d3_rejected: u32,
  /// classify candidates refused by the speck-shell rule (summed over the frame's classifies)
  pub shell_forced: u32,
  /// shown images reprojected
  pub images_shown: u32,
  pub specks: alloc::vec::Vec<super::specks::SpeckPassStats>,
  /// per build chunk drawn: `(k, p, |world-frozen eye − nucleus|, |frozen eye − nucleus|,
  /// |current eye − nucleus|)` in m (the world-terms eye drifts with the comet's orbit; the
  /// nucleus-relative one stays with the camera)
  pub frozen_eye: alloc::vec::Vec<(usize, usize, f64, f64, f64)>,
  /// the coverage at each cycle start of this frame
  pub cycles: alloc::vec::Vec<CadenceCycle>,
  /// per class: admitted by the motion budget this frame
  pub admitted: alloc::vec::Vec<bool>,
  /// this frame's simulated step (s; the time reset's input, telemetry)
  pub dt_s: f64,
  /// per class, this frame's first build entry D3 rejected and first it accepted:
  /// `(k, accepted, footprint_report)`
  pub d3_samples: alloc::vec::Vec<(usize, bool, alloc::string::String)>,
  /// classify candidates refused by the classify-time footprint rule (D3 at the current view)
  pub d3_at_classify: u32,
  /// stale classify candidates whose stamp the anchor history no longer holds (classified at
  /// their stale local position)
  pub unshifted_stale: u32,
  /// up to 64 stale classify candidates per frame: `(tier, offset, stamp, the local mean used)`
  pub classified_at: alloc::vec::Vec<(usize, u32, u32, [f64; 3])>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PrevFrame {
  view: CadenceView,
  eye_m: [f64; 3],
  eye_rel: [f64; 3],
  t: f64,
  /// the simulated step of that frame (none on the first frame)
  dt: Option<f64>,
  speed: f64,
  flags: u32,
  fingerprint: u64,
}

/// begin → finish hand-over
#[derive(Debug, Clone, Default)]
struct PendingFrame {
  mode: Option<CadenceMode>,
  reset: Option<CadenceReset>,
  aborts: alloc::vec::Vec<CadenceAbort>,
  swaps: alloc::vec::Vec<(usize, usize)>,
  /// per tier: (k, p, slots) chunks
  chunks: alloc::vec::Vec<alloc::vec::Vec<(usize, usize, alloc::vec::Vec<u32>)>>,
  fresh: alloc::vec::Vec<alloc::vec::Vec<u32>>,
  anchors: alloc::vec::Vec<[f64; 3]>,
  keys: alloc::vec::Vec<u64>,
  inputs: Option<RateInputs>,
  /// this frame's simulated step (s; 0 on the first frame)
  dt: f64,
  de: [f64; 3],
  nucleus: [f64; 3],
  admitted: alloc::vec::Vec<bool>,
  propagate: alloc::vec::Vec<alloc::vec::Vec<u32>>,
  cycles: alloc::vec::Vec<CadenceCycle>,
}

/// What a classify recorded of one list (the abort test): the frame, the eye-step vector `Δe`
/// relative to the nucleus at that frame and the largest member sensitivity `g_max` (px/m).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ListMeta {
  pub classified_frame: u64,
  pub de_c: [f64; 3],
  pub g_max: f32,
}

/// One tier's cadence bookkeeping: its tracked live range, the owner bytes of its sub-ring, the
/// lists `lists[k][p][state]` (state 0 shown, 1 build, 2 pending) of sub-ring slots, and per
/// (class, phase) the build cursor, its step `⌈L / N⌉` and the lists' metadata.
#[derive(Debug, Clone, Default)]
pub struct TierCadence {
  pub first_slot: u32,
  pub live_count: u32,
  pub capacity: u32,
  pub emitted: u64,
  pub owner: alloc::vec::Vec<u8>,
  pub lists: alloc::vec::Vec<alloc::vec::Vec<[alloc::vec::Vec<u32>; 3]>>,
  pub cursor: alloc::vec::Vec<alloc::vec::Vec<u32>>,
  pub step: alloc::vec::Vec<alloc::vec::Vec<u32>>,
  pub meta: alloc::vec::Vec<alloc::vec::Vec<[ListMeta; 3]>>,
}

impl TierCadence {
  fn new(t: &TierMeta<'_>, classes: usize, phases: usize) -> Self {
    Self {
      first_slot: t.first_slot,
      live_count: t.live_count,
      capacity: t.capacity,
      emitted: t.emitted,
      owner: alloc::vec![0u8; t.capacity as usize],
      lists: (0..classes).map(|_| (0..phases).map(|_| Default::default()).collect()).collect(),
      cursor: alloc::vec![alloc::vec![0; phases]; classes],
      step: alloc::vec![alloc::vec![0; phases]; classes],
      meta: alloc::vec![alloc::vec![[ListMeta::default(); 3]; phases]; classes],
    }
  }
  fn mask(&self) -> u32 {
    self.capacity.wrapping_sub(1)
  }
  fn slot(&self, o: u32) -> u32 {
    self.first_slot.wrapping_add(o) & self.mask()
  }
  fn offset(&self, slot: u32) -> u32 {
    slot.wrapping_sub(self.first_slot) & self.mask()
  }
  fn valid(&self, slot: u32, k: usize, bit: u8) -> bool {
    let b = self.owner[slot as usize];
    (b & OWN_CLASS_MASK) as usize == k + 1 && b & bit != 0
  }
}

/// One (class, phase) pair's images: the shown and the build pyramid, their frozen views and
/// units, the generation (bumped at each swap; the reprojection's dither seed). **A frozen view is
/// kept relative to the nucleus**: its `eye_m` holds `eye − nucleus` at the freeze and every use
/// recomposes it against the current nucleus ([`frozen_at`]), so a camera that follows the comet
/// (which moves ~1 800 km per 60-s frame on its orbit) keeps its frozen view on the dust; the
/// rotation and the projection are kept as they were.
#[derive(Debug, Clone, Default)]
pub struct ClassImage {
  pub shown: alloc::vec::Vec<u32>,
  pub build: alloc::vec::Vec<u32>,
  pub shown_view: Option<CadenceView>,
  pub build_view: Option<CadenceView>,
  pub shown_unit: f32,
  pub build_unit: f32,
  pub generation: u32,
  /// the drawn frame the shown image's build started (its age: the ghost bound)
  pub shown_started: u64,
  pub build_started: u64,
  /// the heliocentric eye at the freeze (diagnostics: the frozen view had it been kept in world
  /// terms)
  pub shown_eye_helio: [f64; 3],
  pub build_eye_helio: [f64; 3],
  /// the nucleus at the freeze
  pub shown_nucleus: [f64; 3],
  pub build_nucleus: [f64; 3],
  /// the build's entries the footprint rule (D3) rejected and those it drew this cycle (the
  /// coverage report at the swap)
  pub build_d3: u32,
  pub build_drawn: u32,
}

impl ClassImage {
  /// the build image's frozen view at the current nucleus (the freeze's own eye, bit for bit,
  /// while the nucleus has not moved)
  pub fn build_at(&self, nucleus_m: [f64; 3]) -> Option<CadenceView> {
    let fv = self.build_view.as_ref()?;
    Some(if nucleus_m == self.build_nucleus { CadenceView { eye_m: self.build_eye_helio, ..*fv } } else { frozen_at(fv, nucleus_m) })
  }
  /// the shown image's frozen view at the current nucleus
  pub fn shown_at(&self, nucleus_m: [f64; 3]) -> Option<CadenceView> {
    let fv = self.shown_view.as_ref()?;
    Some(if nucleus_m == self.shown_nucleus { CadenceView { eye_m: self.shown_eye_helio, ..*fv } } else { frozen_at(fv, nucleus_m) })
  }
}

/// A frozen view (its `eye_m` relative to the nucleus at the freeze) recomposed against the
/// current nucleus: the world-terms view the build draws and the reprojection map use.
pub fn frozen_at(fv: &CadenceView, nucleus_m: [f64; 3]) -> CadenceView {
  CadenceView {
    eye_m: [fv.eye_m[0] + nucleus_m[0], fv.eye_m[1] + nucleus_m[1], fv.eye_m[2] + nucleus_m[2]],
    ..*fv
  }
}

/// The view's cadence state: [`Self::begin`] at the frame's begin (resets, ranges, aborts, swaps,
/// promotions, the propagate list), [`Self::finish`] after the propagate (classify, the build
/// chunks, the reprojection, the speck pass, the fresh splat).
#[derive(Debug, Clone)]
pub struct DustCadenceState {
  pub cfg: CadenceConfig,
  /// the rate rule's shape bound θ ([`RateInputs::shape_frac`]; 0: off, the default; the budget
  /// loop's fallback sets [`DUST_CADENCE_FALLBACK_SHAPE`] through [`BudgetKnobs::shape_frac`])
  pub shape_frac: f32,
  /// drawn frames since the last reset (builds pause on reused frames)
  pub drawn_frame: u64,
  /// by tier key; the frame's tiers come in key order of their [`TierMeta`]s
  pub tiers: alloc::collections::BTreeMap<u64, TierCadence>,
  /// `images[k * phases + p]`
  pub images: alloc::vec::Vec<ClassImage>,
  /// the last [`DUST_CADENCE_HISTORY`] eye steps relative to the nucleus (m), newest last
  pub de_hist: alloc::collections::VecDeque<[f64; 3]>,
  /// tests: measure the light each build entry adds ([`CadenceDrawn::light`]; draws them one by one)
  pub trace_light: bool,
  /// CPU threads of the fresh pass and the build chunks (`splat_tier_list_parallel`, std builds;
  /// word-identical to one thread with the in-frame gates off); 1 = serial
  pub threads: usize,
  /// the class images' geometry (from the viewport and the margin; recomputed on a reset)
  pub geom: Option<ImageGeom>,
  /// the last [`DUST_CADENCE_HISTORY`] per-axis viewport corner motions (px per frame)
  pub motion_hist: alloc::collections::VecDeque<[f32; 2]>,
  /// per class: admitted by the motion budget this frame
  pub admitted: alloc::vec::Vec<bool>,
  /// per tier key: `(stamp, anchor)` of the last [`DUST_CADENCE_ANCHOR_HISTORY`] propagates
  pub anchor_hist: alloc::collections::BTreeMap<u64, alloc::collections::VecDeque<(u32, [f64; 3])>>,
  prev: Option<PrevFrame>,
  pending: PendingFrame,
}

/// propagate generations whose anchor a tier keeps (a shown slot is at most `3N` frames stale)
pub const DUST_CADENCE_ANCHOR_HISTORY: usize = 4 * DUST_CADENCE_HISTORY;

fn fnv(h: &mut u64, v: u64) {
  for b in v.to_le_bytes() {
    *h = (*h ^ b as u64).wrapping_mul(0x100_0000_01b3);
  }
}

fn drop_empty(b: u8) -> u8 {
  if b & !OWN_CLASS_MASK == 0 { 0 } else { b }
}

/// A pyramid's light in counts: `Σ counts` over the levels in its `PYR_LEVEL_MASK` (a texel's count
/// is `τ · area / unit` already, so every level weighs the same per count, as the composite reads
/// it: `τ = count · unit / 4^ℓ` per px² over `4^ℓ` px²). Times the unit: Σ τ · px². (Until
/// 2026-10-18 this summed `count · 4^ℓ`, over-weighting level ℓ by `4^ℓ`: at 1280×800 the 1×1
/// level-11 texel holding ~3 % of a frame's light carried 99.9 % of that sum.)
pub fn pyramid_light(layout: &PyramidLayout, pyr: &[u32]) -> f64 {
  let mask = pyr[PYR_LEVEL_MASK as usize];
  let mut t = 0.0f64;
  for (l, &(off, w, h)) in layout.levels.iter().enumerate() {
    if mask & (1 << l) == 0 {
      continue;
    }
    for i in 0..(w * h) {
      t += pyr[(off + i * PYRAMID_TEXEL_WORDS) as usize] as f64;
    }
  }
  t
}

/// The traced list draw ([`DustCadenceState::trace_light`], tests): `list` drawn into `pyramid`
/// and each entry's light, `pyramid_light` after its draw − before, in list order. Serially: one
/// entry after another into `pyramid`. On `threads` threads (std): entries handed out one at a
/// time; each thread draws its entries one after another into its own partial pyramid
/// (`pyramid`'s header with the counters zeroed, as `splat_tier_list_parallel`) and takes each
/// entry's light as its partial's light after − before; the partials merge into `pyramid` by
/// `merge_pyramid`. **Identical to the serial form with the in-frame gates off** (the words: the
/// order-free merge of `splat_tier_list_parallel`; the lights: a pyramid's light is an exact f64
/// sum of integer counts over the levels of its mask, and a level holds counts only once its
/// mask bit is set, so after − before is the entry's own counts in either pyramid, barring a
/// saturated word): `cadence_traced_build_draw_matches_serial`.
#[allow(clippy::too_many_arguments)]
pub fn draw_list_traced(
  threads: usize,
  moments: &[DustMoments],
  render: &[DustRenderCluster],
  pc: &DustSplatPushConstants,
  flow: &DustFlowUniform,
  layout: &PyramidLayout,
  pyramid: &mut [u32],
  list: &[u32],
) -> alloc::vec::Vec<f64> {
  #[cfg(feature = "std")]
  if threads > 1 && list.len() > 1 {
    use super::splat::{
      PYR_EWA_TEXELS, PYR_SPLIT_CHILDREN, PYR_SPLIT_DROPPED_FLUX, PYR_VOLUME_EVALS, merge_pyramid,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    let n = list.len();
    let hn = PYRAMID_HEADER_WORDS as usize;
    let mut header = pyramid[..hn].to_vec();
    for wd in [PYR_TAU_MAX, PYR_LEVEL_MASK, PYR_SPLIT_CHILDREN, PYR_VOLUME_EVALS, PYR_EWA_TEXELS, PYR_SPLIT_DROPPED_FLUX] {
      header[wd as usize] = 0;
    }
    let (next, len) = (AtomicUsize::new(0), pyramid.len());
    let parts: alloc::vec::Vec<(alloc::vec::Vec<u32>, alloc::vec::Vec<(usize, f64)>)> = std::thread::scope(|s| {
      let handles: alloc::vec::Vec<_> = (0..threads.min(n))
        .map(|_| {
          let (header, next) = (&header, &next);
          s.spawn(move || {
            let mut p = alloc::vec![0u32; len];
            p[..hn].copy_from_slice(header);
            let mut lights = alloc::vec::Vec::new();
            let mut before = pyramid_light(layout, &p);
            loop {
              let i = next.fetch_add(1, Ordering::Relaxed);
              if i >= n {
                break;
              }
              splat_tier_list(moments, render, pc, flow, 0, layout, &mut p, &list[i..i + 1]);
              let after = pyramid_light(layout, &p);
              lights.push((i, after - before));
              before = after;
            }
            (p, lights)
          })
        })
        .collect();
      handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out = alloc::vec![0.0f64; n];
    for (p, lights) in &parts {
      merge_pyramid(pyramid, p);
      for &(i, l) in lights {
        out[i] = l;
      }
    }
    return out;
  }
  let _ = threads;
  list
    .iter()
    .map(|&o| {
      let before = pyramid_light(layout, pyramid);
      splat_tier_list(moments, render, pc, flow, 0, layout, pyramid, &[o]);
      pyramid_light(layout, pyramid) - before
    })
    .collect()
}

/// The Jacobian's Frobenius norm (px per m) of `mvp` at local point `p` (∞ behind the eye)
fn jacobian_px_per_m(mvp: &[f32; 16], p: [f32; 3], width: u32, height: u32) -> f32 {
  let mm = |r: usize, c: usize| mvp[c * 4 + r];
  let row = |r: usize| [mm(r, 0), mm(r, 1), mm(r, 2)];
  let (r0, r1, r3) = (row(0), row(1), row(3));
  let w = dot3(r3, p) + mm(3, 3);
  if !(w > 0.0) {
    return f32::INFINITY;
  }
  let nx = (dot3(r0, p) + mm(0, 3)) / w;
  let ny = (dot3(r1, p) + mm(1, 3)) / w;
  let kx = 0.5 * width as f32 / w;
  let ky = 0.5 * height as f32 / w;
  let jx = [(r0[0] - r3[0] * nx) * kx, (r0[1] - r3[1] * nx) * kx, (r0[2] - r3[2] * nx) * kx];
  let jy = [(r1[0] - r3[0] * ny) * ky, (r1[1] - r3[1] * ny) * ky, (r1[2] - r3[2] * ny) * ky];
  <f32 as FloatLike>::sqrt(dot3(jx, jx) + dot3(jy, jy))
}

/// The footprint rule (D3): cluster `m` (with its predecessor's mean, the time segment's end) is
/// wholly inside the frame of `mvp`: its mean, its 17 edges and the predecessor's mean project in
/// front of the eye, each with a margin of `3 σ_max · g` px (`σ_max = √λ_max(Σ)`, the larger of
/// the two clusters' for the predecessor's point; `g` the Jacobian's norm there) inside
/// `[0, W] × [0, H]`. A build entry failing it is not drawn into the image and stays fresh.
pub fn footprint_inside(m: &DustMoments, pred: Option<&DustMoments>, mvp: &[f32; 16], width: u32, height: u32) -> bool {
  let sig = |c: &DustMoments| <f32 as FloatLike>::sqrt(sym3_eigen_range(c.cov_v()).1.max(0.0));
  let s_m = sig(m);
  let (wf, hf) = (width as f32, height as f32);
  let inside = |p: [f32; 3], s: f32| -> bool {
    let Some(q) = project_point(p, mvp, width, height) else { return false };
    let r = 3.0 * s * jacobian_px_per_m(mvp, p, width, height);
    r.is_finite() && q[0] - r >= 0.0 && q[1] - r >= 0.0 && q[0] + r <= wf && q[1] + r <= hf
  };
  if !inside(m.mean(), s_m) {
    return false;
  }
  for j in 0..SIZE_EDGES as usize {
    if !inside(m.edge(j), s_m) {
      return false;
    }
  }
  match pred {
    Some(p) => inside(p.mean(), s_m.max(sig(p))),
    None => true,
  }
}

/// [`footprint_inside`]'s decision spelled out (diagnostics): the mean's px and clip `w` under
/// `mvp`, the px bounding box of the 17 edges and of the predecessor's mean, σ_max (m), `g` (px per
/// m) at the mean, the margin (px), and the first sub-test that fails (`none` when inside).
pub fn footprint_report(m: &DustMoments, pred: Option<&DustMoments>, mvp: &[f32; 16], width: u32, height: u32) -> alloc::string::String {
  let sig = |c: &DustMoments| <f32 as FloatLike>::sqrt(sym3_eigen_range(c.cov_v()).1.max(0.0));
  let s_m = sig(m);
  let (wf, hf) = (width as f32, height as f32);
  let wclip = |p: [f32; 3]| mvp[3] * p[0] + mvp[7] * p[1] + mvp[11] * p[2] + mvp[15];
  let mut fail: Option<alloc::string::String> = None;
  let mut check = |name: alloc::string::String, p: [f32; 3], s: f32| {
    if fail.is_some() {
      return;
    }
    match project_point(p, mvp, width, height) {
      None => fail = Some(alloc::format!("{name}: behind the eye (w {:.3e})", wclip(p))),
      Some(q) => {
        let r = 3.0 * s * jacobian_px_per_m(mvp, p, width, height);
        if !(r.is_finite() && q[0] - r >= 0.0 && q[1] - r >= 0.0 && q[0] + r <= wf && q[1] + r <= hf) {
          fail = Some(alloc::format!("{name}: at ({:.1}, {:.1}) px with margin {r:.2} px outside [0, {wf}] × [0, {hf}]", q[0], q[1]));
        }
      }
    }
  };
  check("the mean".into(), m.mean(), s_m);
  for j in 0..SIZE_EDGES as usize {
    check(alloc::format!("edge {j}"), m.edge(j), s_m);
  }
  if let Some(p) = pred {
    check("the predecessor's mean".into(), p.mean(), s_m.max(sig(p)));
  }
  let mut lo = [f32::INFINITY; 2];
  let mut hi = [f32::NEG_INFINITY; 2];
  let mut behind = 0;
  for j in 0..SIZE_EDGES as usize {
    match project_point(m.edge(j), mvp, width, height) {
      Some(q) => {
        for k in 0..2 {
          lo[k] = lo[k].min(q[k]);
          hi[k] = hi[k].max(q[k]);
        }
      }
      None => behind += 1,
    }
  }
  let mpx = project_point(m.mean(), mvp, width, height);
  let ppx = pred.and_then(|p| project_point(p.mean(), mvp, width, height));
  let g = jacobian_px_per_m(mvp, m.mean(), width, height);
  alloc::format!(
    "age {:.0} s; mean px {mpx:?} w {:.4e}; edges px x [{:.1}, {:.1}] y [{:.1}, {:.1}] ({behind} behind); predecessor's mean px {ppx:?}; σ_max {s_m:.1} m, g {g:.4e} px/m, margin {:.2} px; fails: {}",
    m.age(),
    wclip(m.mean()),
    lo[0],
    hi[0],
    lo[1],
    hi[1],
    3.0 * s_m * g,
    fail.unwrap_or_else(|| "none".into())
  )
}

impl DustCadenceState {
  pub fn new(cfg: CadenceConfig) -> Self {
    let n = cfg.classes.len() * cfg.phases as usize;
    Self {
      cfg,
      shape_frac: 0.0,
      drawn_frame: 0,
      tiers: alloc::collections::BTreeMap::new(),
      images: alloc::vec![ClassImage::default(); n],
      de_hist: alloc::collections::VecDeque::with_capacity(DUST_CADENCE_HISTORY),
      trace_light: false,
      threads: 1,
      prev: None,
      anchor_hist: alloc::collections::BTreeMap::new(),
      geom: None,
      motion_hist: alloc::collections::VecDeque::with_capacity(DUST_CADENCE_HISTORY),
      admitted: alloc::vec::Vec::new(),
      pending: PendingFrame::default(),
    }
  }

  fn nc(&self) -> usize {
    self.cfg.classes.len().min(DUST_CADENCE_MAX_CLASSES)
  }
  fn np(&self) -> usize {
    self.cfg.phases.clamp(1, 2) as usize
  }
  fn offset_of(&self, k: usize, p: usize) -> u64 {
    if self.np() == 2 { (p as u64) * (self.cfg.classes[k] as u64 / 2) } else { 0 }
  }
  /// whether (k, p)'s cycle starts at this drawn frame
  fn starts(&self, k: usize, p: usize) -> bool {
    let n = self.cfg.classes[k] as u64;
    (self.drawn_frame % n + n - self.offset_of(k, p) % n) % n == 0
  }
  /// drawn frames until (k, p)'s next swap (1..=N)
  pub fn frames_to_swap(&self, k: usize, p: usize) -> u32 {
    let n = self.cfg.classes[k] as u64;
    let r = (self.drawn_frame % n + n - self.offset_of(k, p) % n) % n;
    if r == 0 { n as u32 } else { (n - r) as u32 }
  }

  /// `a_w(N_k)` per class (m) from the eye-step history: `max |Δe(f′) − Δe(f)|` over the last
  /// `2N_k` frames, `Δe(f)` the newest
  pub fn window_terms(&self) -> [f32; DUST_CADENCE_MAX_CLASSES] {
    let mut out = [0.0f32; DUST_CADENCE_MAX_CLASSES];
    let Some(&cur) = self.de_hist.back() else { return out };
    for (k, &n) in self.cfg.classes.iter().take(DUST_CADENCE_MAX_CLASSES).enumerate() {
      let mut a = 0.0f64;
      for d in self.de_hist.iter().rev().take(2 * n as usize) {
        let e = [d[0] - cur[0], d[1] - cur[1], d[2] - cur[2]];
        a = a.max((e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt());
      }
      // fewer than 2N frames of history: the deviation over the span L − 1 extrapolated to
      // 2N − 1 (a constant rate of change of the eye's step); one frame: unknown, not admitted
      let span = self.de_hist.len().min(2 * n as usize);
      if span < 2 * n as usize {
        a = if span <= 1 { f64::INFINITY } else { a * (2 * n as usize - 1) as f64 / (span - 1) as f64 };
      }
      out[k] = a as f32;
    }
    out
  }

  /// `U(N_k)` per class (px per axis): the maximum corner motion over the last `2N_k` frames
  pub fn motion_window(&self, k: usize) -> [f32; 2] {
    let n = self.cfg.classes[k] as usize;
    let mut u = [0.0f32; 2];
    for d in self.motion_hist.iter().rev().take(2 * n) {
      u[0] = u[0].max(d[0]);
      u[1] = u[1].max(d[1]);
    }
    u
  }

  /// the motion budget per axis (px): `m − G` (∞ without a margin: the budget is off)
  pub fn motion_budget(&self) -> [f32; 2] {
    match self.geom.as_ref() {
      Some(g) if g.has_margin() => [
        g.margin[0] as f32 - DUST_CADENCE_MARGIN_GUARD,
        g.margin[1] as f32 - DUST_CADENCE_MARGIN_GUARD,
      ],
      _ => [f32::INFINITY; 2],
    }
  }

  /// whether class `k` fits the motion budget: `2N · U(N) ≤ m − G` on both axes
  pub fn class_admitted(&self, k: usize) -> bool {
    let b = self.motion_budget();
    if !b[0].is_finite() {
      return true;
    }
    let n = self.cfg.classes[k] as f32;
    let u = self.motion_window(k);
    2.0 * n * u[0] <= b[0] && 2.0 * n * u[1] <= b[1]
  }

  /// records this frame's eye step relative to the nucleus (m)
  pub fn push_eye_step(&mut self, de: [f64; 3]) {
    if self.de_hist.len() == DUST_CADENCE_HISTORY {
      self.de_hist.pop_front();
    }
    self.de_hist.push_back(de);
  }

  /// Every list, owner byte and image cleared; the cycles restart at drawn frame 0.
  pub fn clear_all(&mut self) {
    for t in self.tiers.values_mut() {
      t.owner.iter_mut().for_each(|b| *b = 0);
      for kl in t.lists.iter_mut() {
        for pl in kl.iter_mut() {
          for l in pl.iter_mut() {
            l.clear();
          }
        }
      }
      for v in t.cursor.iter_mut().chain(t.step.iter_mut()) {
        v.iter_mut().for_each(|c| *c = 0);
      }
      for v in t.meta.iter_mut() {
        v.iter_mut().for_each(|m| *m = [ListMeta::default(); 3]);
      }
    }
    for im in self.images.iter_mut() {
      let g = im.generation;
      *im = ClassImage { generation: g, ..ClassImage::default() };
    }
    self.drawn_frame = 0;
  }

  fn has_content(&self) -> bool {
    self.tiers.values().any(|t| t.owner.iter().any(|&b| b != 0))
      || self.images.iter().any(|im| im.shown_view.is_some() || im.build_view.is_some())
  }

  fn reset_reason(&self, f: &FrameMeta, tiers: &[TierMeta<'_>]) -> Option<CadenceReset> {
    let prev = self.prev.as_ref()?;
    let (v, pv) = (&f.view, &prev.view);
    if v.width != pv.width || v.height != pv.height {
      return Some(CadenceReset::Viewport);
    }
    let persp = v.perspective();
    if persp != pv.perspective() {
      return Some(CadenceReset::Projection);
    }
    if persp {
      for c in 0..4 {
        for r in [0usize, 1, 3] {
          if v.proj[c * 4 + r].to_bits() != pv.proj[c * 4 + r].to_bits() {
            return Some(CadenceReset::Projection);
          }
        }
      }
    } else {
      for im in &self.images {
        for fv in [im.shown_at(f.nucleus_m), im.build_at(f.nucleus_m)].into_iter().flatten() {
          let Some(h) = v.map_from(&fv) else { return Some(CadenceReset::OrthoDirection) };
          let det = (h[0] * h[4] - h[1] * h[3]).abs();
          let z = <f32 as FloatLike>::sqrt(det);
          if !(z <= DUST_CADENCE_ORTHO_ZOOM_MAX && z >= 1.0 / DUST_CADENCE_ORTHO_ZOOM_MAX) {
            return Some(CadenceReset::OrthoZoom);
          }
        }
      }
    }
    if f.speed.to_bits() != prev.speed.to_bits() {
      return Some(CadenceReset::Speed);
    }
    // the time reset: a negative step, a positive one after a non-positive one, or one beyond the
    // jump factor × the previous step
    let dt = f.t_now_s - prev.t;
    if dt < 0.0 {
      return Some(CadenceReset::Time);
    }
    if let Some(q) = prev.dt
      && dt > 0.0
      && (q <= 0.0 || dt > DUST_CADENCE_TIME_JUMP_FACTOR * q)
    {
      return Some(CadenceReset::Time);
    }
    if f.flags != prev.flags {
      return Some(CadenceReset::Flags);
    }
    if f.fingerprint != prev.fingerprint {
      return Some(CadenceReset::Fingerprint);
    }
    if tiers.len() != self.tiers.len() || tiers.iter().any(|t| !self.tiers.contains_key(&t.key)) {
      return Some(CadenceReset::TierSet);
    }
    for t in tiers {
      let old = &self.tiers[&t.key];
      if t.capacity != old.capacity {
        return Some(CadenceReset::Capacity(t.key));
      }
      let a = t.first_slot.wrapping_sub(old.first_slot) & old.mask();
      let appended = t.live_count as i64 + a as i64 - old.live_count as i64;
      if a <= old.live_count && appended < 0 {
        return Some(CadenceReset::Range(t.key));
      }
      // the emission changed, yet nothing was appended and the host names no rewritten slot:
      // an inconsistent input (a re-emission it cannot place)
      if appended == 0 && t.emitted != old.emitted && t.rewritten_from >= t.live_count {
        return Some(CadenceReset::Reemit(t.key));
      }
    }
    None
  }

  /// the retire / emit hook: the owner bytes of the slots that left (retired at the tail),
  /// entered (emitted at the head) or were rewritten (`TierMeta::rewritten_from`) are cleared (GPU: `cmd_fill_buffer` over
  /// the same ranges); the stale list entries become invalid
  fn track_ranges(&mut self, tiers: &[TierMeta<'_>]) {
    for t in tiers {
      let old = self.tiers.get_mut(&t.key).expect("tier");
      let mask = old.mask();
      let cap = old.capacity;
      let a = t.first_slot.wrapping_sub(old.first_slot) & mask;
      let appended = (t.live_count as i64 + a as i64 - old.live_count as i64).max(0) as u64;
      if a > old.live_count || appended >= cap as u64 {
        old.owner.iter_mut().for_each(|b| *b = 0);
      } else {
        for i in 0..a {
          old.owner[(old.first_slot.wrapping_add(i) & mask) as usize] = 0;
        }
        let end = old.first_slot.wrapping_add(old.live_count);
        for i in 0..appended as u32 {
          old.owner[(end.wrapping_add(i) & mask) as usize] = 0;
        }
        // the rewritten slots (the provisional window re-emitted, a re-emission the host placed):
        // new content in old slots, fresh until a build of their own
        for o in t.rewritten_from.min(t.live_count)..t.live_count {
          old.owner[(t.first_slot.wrapping_add(o) & mask) as usize] = 0;
        }
      }
      old.first_slot = t.first_slot;
      old.live_count = t.live_count;
      old.emitted = t.emitted;
    }
  }

  /// The frame's begin (F0–F2a): reused frames return [`CadenceMode::Paused`]; the resets; the
  /// range hook; the eye-step history; the aborts; the swaps and promotions of the (class, phase)
  /// pairs whose cycle starts now; the build chunks; the fresh set; the propagate list.
  pub fn begin(&mut self, f: &FrameMeta, tiers: &[TierMeta<'_>], layout: &PyramidLayout) -> CadencePlan {
    if f.reused {
      self.pending = PendingFrame { mode: Some(CadenceMode::Paused), ..Default::default() };
      return CadencePlan { mode: Some(CadenceMode::Paused), ..Default::default() };
    }
    let (nc, np) = (self.nc(), self.np());
    let off = f.flags & (super::DUST_VIEW_FLOW | super::DUST_VIEW_TRACERS) != 0;
    let first = self.prev.is_none();
    let mut reset = if first { Some(CadenceReset::First) } else { self.reset_reason(f, tiers) };
    if off && reset.is_none() && self.has_content() {
      reset = Some(CadenceReset::Off);
    }
    if reset.is_some() || off {
      self.clear_all();
    }
    // the tiers: (re)created on a reset or a new key, else their ranges tracked
    if reset.is_some() || off {
      self.tiers.clear();
      for t in tiers {
        self.tiers.insert(t.key, TierCadence::new(t, nc, np));
      }
    } else {
      self.track_ranges(tiers);
    }
    // the eye steps
    let eye_rel = [
      f.view.eye_m[0] - f.nucleus_m[0],
      f.view.eye_m[1] - f.nucleus_m[1],
      f.view.eye_m[2] - f.nucleus_m[2],
    ];
    let (dt, d_eye, de) = match &self.prev {
      Some(p) => (
        f.t_now_s - p.t,
        [f.view.eye_m[0] - p.eye_m[0], f.view.eye_m[1] - p.eye_m[1], f.view.eye_m[2] - p.eye_m[2]],
        [eye_rel[0] - p.eye_rel[0], eye_rel[1] - p.eye_rel[1], eye_rel[2] - p.eye_rel[2]],
      ),
      None => (0.0, [0.0; 3], [0.0; 3]),
    };
    if !first {
      self.push_eye_step(de);
    }
    // the class images' geometry
    if self.geom.as_ref().is_none_or(|g| g.layout.width != layout.width + 2 * g.margin[0] || g.layout.height != layout.height + 2 * g.margin[1] || reset.is_some()) {
      self.geom = Some(self.cfg.image_geom(layout.width, layout.height));
    }
    // the motion budget: this frame's corner motion against the previous view, the admit flags
    if let Some(pv) = self.prev.as_ref()
      && !first
    {
      let u = f.view.map_from(&pv.view).map_or([f32::INFINITY; 2], |m| corner_motion(&m, f.view.width, f.view.height));
      if self.motion_hist.len() == DUST_CADENCE_HISTORY {
        self.motion_hist.pop_front();
      }
      self.motion_hist.push_back(u);
    }
    self.admitted = (0..self.nc()).map(|k| self.class_admitted(k)).collect();
    self.prev = Some(PrevFrame {
      view: f.view,
      eye_m: f.view.eye_m,
      eye_rel,
      t: f.t_now_s,
      dt: (!first).then_some(dt),
      speed: f.speed,
      flags: f.flags,
      fingerprint: f.fingerprint,
    });
    let mode = if off {
      CadenceMode::Off
    } else if first {
      CadenceMode::First
    } else {
      CadenceMode::Run
    };
    let all = |t: &TierMeta<'_>| (0..t.live_count.min(t.render.len() as u32)).collect::<alloc::vec::Vec<u32>>();
    let mut pend = PendingFrame {
      mode: Some(mode),
      reset,
      anchors: tiers.iter().map(|t| t.anchor_m).collect(),
      nucleus: f.nucleus_m,
      admitted: self.admitted.clone(),
      keys: tiers.iter().map(|t| t.key).collect(),
      de,
      dt,
      ..Default::default()
    };
    if mode != CadenceMode::Run {
      pend.fresh = tiers.iter().map(all).collect();
      pend.chunks = tiers.iter().map(|_| alloc::vec::Vec::new()).collect();
      let plan = CadencePlan { mode: Some(mode), reset, propagate: pend.fresh.clone(), ..Default::default() };
      self.pending = pend;
      return plan;
    }
    pend.inputs = Some(RateInputs {
      dt_s: dt as f32,
      d_eye_m: [d_eye[0] as f32, d_eye[1] as f32, d_eye[2] as f32],
      a_w_m: self.window_terms(),
      eps_px: self.cfg.eps_px,
      frame_px: f.view.width.max(f.view.height) as f32,
      shape_frac: self.shape_frac,
    });
    // F1: aborts
    if reset.is_none() {
      for k in 0..nc {
        let n = self.cfg.classes[k] as f32;
        let bound = (DUST_CADENCE_ABORT_FACTOR - 1.0) * self.cfg.eps_px / (2.0 * n);
        for p in 0..np {
          for list in [1usize, 2] {
            let (mut entries, mut g_max, mut de_c) = (0u32, 0.0f32, [0.0f64; 3]);
            for t in self.tiers.values() {
              if !t.lists[k][p][list].is_empty() {
                entries += t.lists[k][p][list].len() as u32;
                g_max = g_max.max(t.meta[k][p][list].g_max);
                de_c = t.meta[k][p][list].de_c;
              }
            }
            if entries == 0 {
              continue;
            }
            let dd = [de[0] - de_c[0], de[1] - de_c[1], de[2] - de_c[2]];
            let excess = ((dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]).sqrt() as f32) * g_max;
            // the margin's rules: the class over its motion budget; a build whose displacement at
            // the end of its display is predicted beyond the budget
            let budget = self.motion_budget();
            let predicted = if list == 1 && budget[0].is_finite() {
              let im = &self.images[k * np + p];
              match im.build_at(f.nucleus_m).and_then(|fv| f.view.map_from(&fv)) {
                Some(mp) => {
                  let d = corner_motion(&mp, f.view.width, f.view.height);
                  let u = self.motion_window(k);
                  let rest = (self.frames_to_swap(k, p) + self.cfg.classes[k]) as f32;
                  d[0] + rest * u[0] > budget[0] || d[1] + rest * u[1] > budget[1]
                }
                None => true,
              }
            } else {
              false
            };
            let reason = if excess > bound {
              ABORT_RATE
            } else if !self.admitted[k] {
              ABORT_BUDGET
            } else if predicted {
              ABORT_PREDICTION
            } else {
              continue;
            };
            let bit = if list == 1 { OWN_BUILD | OWN_IN_IMAGE } else { OWN_PENDING };
            let mut shown_entries = 0u32;
            for t in self.tiers.values_mut() {
              let l = core::mem::take(&mut t.lists[k][p][list]);
              for &s in &l {
                if t.valid(s, k, if list == 1 { OWN_BUILD } else { OWN_PENDING }) {
                  let b = t.owner[s as usize] & !bit;
                  t.owner[s as usize] = drop_empty(b);
                }
              }
              t.meta[k][p][list] = ListMeta::default();
              if list == 1 {
                t.cursor[k][p] = 0;
                t.step[k][p] = 0;
              }
              shown_entries += t.lists[k][p][0].iter().filter(|&&s| t.valid(s, k, OWN_SHOWN)).count() as u32;
            }
            if list == 1 {
              let im = &mut self.images[k * np + p];
              // the words go, the frozen view stays (an aborted build is an empty one to the host)
              im.build.clear();
              im.build_d3 = 0;
              im.build_drawn = 0;
            }
            pend.aborts.push(CadenceAbort {
              k,
              p,
              list: list as u8,
              reason,
              entries,
              excess_px: excess,
              bound_px: bound,
              g_max,
              shown_entries,
              ghost_frames_left: self.frames_to_swap(k, p),
            });
          }
        }
      }
    }
    // the coverage of the shown images (with a margin): an image whose viewport corners leave its
    // guarded interior is dropped now, its members fresh (ghost 0)
    if reset.is_none()
      && let Some(geom) = self.geom.clone()
      && geom.has_margin()
    {
      for k in 0..nc {
        for p in 0..np {
          let covered = {
            let im = &self.images[k * np + p];
            if im.shown.is_empty() {
              continue;
            }
            im.shown_at(f.nucleus_m)
              .and_then(|sv| f.view.map_to_image(&sv, &geom))
              .is_some_and(|mp| image_covers(&mp, f.view.width, f.view.height, &geom.layout))
          };
          if covered {
            continue;
          }
          let mut entries = 0u32;
          for t in self.tiers.values_mut() {
            let l = core::mem::take(&mut t.lists[k][p][0]);
            for &s in &l {
              if t.valid(s, k, OWN_SHOWN) {
                t.owner[s as usize] = drop_empty(t.owner[s as usize] & !OWN_SHOWN);
                entries += 1;
              }
            }
          }
          self.images[k * np + p].shown.clear();
          pend.aborts.push(CadenceAbort {
            k,
            p,
            list: 0,
            reason: ABORT_COVERAGE,
            entries,
            excess_px: 0.0,
            bound_px: 0.0,
            g_max: 0.0,
            shown_entries: 0,
            ghost_frames_left: 0,
          });
        }
      }
    }
    // F2a: swaps and promotions
    for k in 0..nc {
      let n = self.cfg.classes[k];
      for p in 0..np {
        if !self.starts(k, p) {
          continue;
        }
        let mut any_build = false;
        for t in self.tiers.values_mut() {
          // the swap: the old shown entries leave; the valid build entries drawn into the image
          // become shown
          let old = core::mem::take(&mut t.lists[k][p][0]);
          for &s in &old {
            if t.valid(s, k, OWN_SHOWN) {
              t.owner[s as usize] = drop_empty(t.owner[s as usize] & !OWN_SHOWN);
            }
          }
          let build = core::mem::take(&mut t.lists[k][p][1]);
          let mut shown = alloc::vec::Vec::with_capacity(build.len());
          for &s in &build {
            if t.valid(s, k, OWN_BUILD) {
              let b = t.owner[s as usize];
              let mut nb = b & !(OWN_BUILD | OWN_IN_IMAGE);
              if b & OWN_IN_IMAGE != 0 {
                nb |= OWN_SHOWN;
                shown.push(s);
              }
              t.owner[s as usize] = drop_empty(nb);
            }
          }
          t.lists[k][p][0] = shown;
          // the promotion: the valid pending entries become the build
          let pending = core::mem::take(&mut t.lists[k][p][2]);
          let mut nb_list = alloc::vec::Vec::with_capacity(pending.len());
          for &s in &pending {
            if t.valid(s, k, OWN_PENDING) {
              t.owner[s as usize] = (t.owner[s as usize] & !OWN_PENDING) | OWN_BUILD;
              nb_list.push(s);
            }
          }
          any_build |= !nb_list.is_empty();
          t.step[k][p] = (nb_list.len() as u32).div_ceil(n);
          t.cursor[k][p] = 0;
          t.lists[k][p][1] = nb_list;
          t.meta[k][p][1] = t.meta[k][p][2];
          t.meta[k][p][2] = ListMeta::default();
        }
        let started = self.drawn_frame;
        let im = &mut self.images[k * np + p];
        im.shown = core::mem::take(&mut im.build);
        im.shown_view = im.build_view.take();
        im.shown_eye_helio = im.build_eye_helio;
        im.shown_nucleus = im.build_nucleus;
        im.shown_unit = im.build_unit;
        im.shown_started = im.build_started;
        pend.cycles.push(CadenceCycle { k, p, d3_rejected: im.build_d3, drawn: im.build_drawn, shown_light_share: 0.0 });
        im.build_d3 = 0;
        im.build_drawn = 0;
        im.generation = im.generation.wrapping_add(1);
        // the frozen view at every promotion (an empty build too: the GPU host freezes it
        // regardless of the list's length); the image's words only when something will be drawn
        if any_build {
          let il = &self.geom.as_ref().expect("the image geometry").layout;
          let mut b = alloc::vec![0u32; il.total_words as usize];
          b[..PYRAMID_HEADER_WORDS as usize].copy_from_slice(&il.header());
          b[PYR_UNIT as usize] = f.unit_now.to_bits();
          im.build = b;
        }
        {
          im.build_view = Some(CadenceView {
            eye_m: [
              f.view.eye_m[0] - f.nucleus_m[0],
              f.view.eye_m[1] - f.nucleus_m[1],
              f.view.eye_m[2] - f.nucleus_m[2],
            ],
            ..f.view
          });
          im.build_eye_helio = f.view.eye_m;
          im.build_nucleus = f.nucleus_m;
          im.build_unit = f.unit_now;
          im.build_started = started;
        }
        pend.swaps.push((k, p));
      }
    }
    // the build chunks, the fresh set, the propagate list (offsets)
    for t in tiers {
      let tc = &self.tiers[&t.key];
      let live = t.live_count.min(t.render.len() as u32);
      let mut chunks = alloc::vec::Vec::new();
      let mut prop: alloc::vec::Vec<u32> = alloc::vec::Vec::new();
      for k in 0..nc {
        for p in 0..np {
          let l = &tc.lists[k][p][1];
          if l.is_empty() {
            continue;
          }
          let lo = (tc.cursor[k][p] as usize).min(l.len());
          let hi = (lo + tc.step[k][p] as usize).min(l.len());
          let c: alloc::vec::Vec<u32> = l[lo..hi].to_vec();
          for &s in &c {
            if tc.valid(s, k, OWN_BUILD) {
              let o = tc.offset(s);
              if o < live {
                prop.push(o);
              }
            }
          }
          chunks.push((k, p, c));
        }
      }
      let fresh: alloc::vec::Vec<u32> =
        (0..live).filter(|&o| tc.owner[tc.slot(o) as usize] & OWN_SHOWN == 0).collect();
      prop.extend_from_slice(&fresh);
      let mut preds: alloc::vec::Vec<u32> =
        prop.iter().filter_map(|&o| streak_pred(t.render, o as usize).map(|q| q as u32)).collect();
      prop.append(&mut preds);
      prop.sort_unstable();
      prop.dedup();
      pend.chunks.push(chunks);
      pend.fresh.push(fresh);
      pend.propagate.push(prop);
    }
    for t in self.tiers.values_mut() {
      for k in 0..nc {
        for p in 0..np {
          let len = t.lists[k][p][1].len() as u32;
          t.cursor[k][p] = (t.cursor[k][p] + t.step[k][p]).min(len);
        }
      }
    }
    let propagate = core::mem::take(&mut pend.propagate);
    let plan = CadencePlan { mode: Some(mode), reset, aborts: pend.aborts.clone(), swaps: pend.swaps.clone(), propagate };
    self.pending = pend;
    plan
  }

  /// The frame's draw (F2b–F7), after the propagate of the plan's list: the classifies of the
  /// pairs whose cycle started (`frame`'s header gives the speck shell), the build chunks into
  /// their images at the frozen views (D3, the mask lane cleared, the header counters routed
  /// through the frame's), the reprojection of every shown image into `frame` (cleared, header
  /// written), the speck pass (n_eye over every live cluster, walks only where not SHOWN), the
  /// fresh splat. `tiers` in the order of [`Self::begin`]'s.
  pub fn finish(&mut self, tiers: &mut [TierDraw<'_>], layout: &PyramidLayout, frame: &mut [u32]) -> CadenceReport {
    let pend = core::mem::take(&mut self.pending);
    let mode = pend.mode;
    let mut rep = CadenceReport {
      mode,
      reset: pend.reset,
      aborts: pend.aborts.clone(),
      swaps: pend.swaps.clone(),
      fresh: pend.fresh.clone(),
      ..Default::default()
    };
    if mode.is_none() || mode == Some(CadenceMode::Paused) {
      return rep;
    }
    assert_eq!(tiers.len(), pend.keys.len(), "finish: the tiers of begin");
    let np = self.np();
    let (w, h) = (layout.width, layout.height);
    let d_max = super::specks::SpeckHeader::read(frame).d_max;
    let geom = self.geom.clone().unwrap_or_else(|| self.cfg.image_geom(w, h));
    let footprint_rule = !geom.has_margin();
    rep.admitted = pend.admitted.clone();
    rep.dt_s = pend.dt;
    // the anchor of this frame's propagate generation, per tier
    for (ti, td) in tiers.iter().enumerate() {
      let h = self.anchor_hist.entry(pend.keys[ti]).or_default();
      if h.back().is_none_or(|&(s, a)| s != td.stamp || a != pend.anchors[ti]) {
        if h.len() == DUST_CADENCE_ANCHOR_HISTORY {
          h.pop_front();
        }
        h.push_back((td.stamp, pend.anchors[ti]));
      }
    }
    // F2b: classify the pairs whose cycle started
    if mode == Some(CadenceMode::Run) {
      let inputs = pend.inputs.expect("rate inputs");
      let classes = self.cfg.classes.clone();
      for &(k, p) in &pend.swaps {
        // a class over its motion budget is not classified until the motion over 2N frames fits
        if !pend.admitted.get(k).copied().unwrap_or(true) {
          rep.classified.push((k, p, 0));
          continue;
        }
        let mut total = 0u32;
        let mut g_max = 0.0f32;
        for (ti, td) in tiers.iter().enumerate() {
          let key = pend.keys[ti];
          let parity2 = np == 2;
          let tc = self.tiers.get_mut(&key).expect("tier");
          let live = tc.live_count.min(td.render.len() as u32).min(td.moments.len() as u32);
          let eye = [td.pc.eye_local[0], td.pc.eye_local[1], td.pc.eye_local[2]];
          let mut list = alloc::vec::Vec::new();
          let mut tg = 0.0f32;
          for o in 0..live {
            let s = tc.slot(o);
            if parity2 && (s & 1) as usize != p {
              continue;
            }
            let b = tc.owner[s as usize];
            let cls = (b & OWN_CLASS_MASK) as usize;
            if (cls != 0 && cls != k + 1) || b & OWN_PENDING != 0 {
              continue;
            }
            let m0 = &td.moments[o as usize];
            if !(m0.flux() > 0.0) || !render_live(td.render[o as usize].age_id_dbeta_flux[1]) {
              continue;
            }
            // a stale (not propagated this frame) slot: its positions are local to the anchor of
            // its propagate; shifted to the current anchor by the delta of the two anchors
            let stamp = m0.propagate_stamp();
            let anchor_now = pend.anchors[ti];
            let delta = if stamp == td.stamp {
              Some([0.0f32; 3])
            } else {
              self
                .anchor_hist
                .get(&key)
                .and_then(|h| h.iter().rev().find(|x| x.0 == stamp))
                .map(|&(_, a)| anchor_delta_f32(a, anchor_now))
            };
            let m_shift;
            let m: &DustMoments = match delta {
              Some(d) if d != [0.0; 3] => {
                m_shift = shifted_moments(m0, d);
                &m_shift
              }
              Some(_) => m0,
              None => {
                rep.unshifted_stale += 1;
                m0
              }
            };
            if stamp != td.stamp && rep.classified_at.len() < 64 {
              let mm = m.mean();
              rep.classified_at.push((ti, o, stamp, [mm[0] as f64, mm[1] as f64, mm[2] as f64]));
            }
            if shell_capable(m, eye, d_max) {
              rep.shell_forced += 1;
              continue;
            }
            // the predecessor only at the same stamp, so the same shift
            let pred_shift;
            let pred = match streak_pred(td.render, o as usize).map(|q| &td.moments[q]).filter(|q| q.propagate_stamp() == stamp) {
              Some(q) => match delta {
                Some(d) if d != [0.0; 3] => {
                  pred_shift = shifted_moments(q, d);
                  Some(&pred_shift)
                }
                _ => Some(q),
              },
              None => None,
            };
            let r = cadence_rate(m, pred, &td.pc.mvp, eye, w, h, &classes, &inputs);
            rep.class_hist[(r.class as usize).min(DUST_CADENCE_MAX_CLASSES)] += 1;
            // the classify-time footprint rule (D3 at the current view): an entry the build would
            // reject stays fresh instead of filling the list
            // the class it goes to: the largest admitted by the motion budget among those the rate
            // admits (a class of N admits every smaller class's bound: δ_k' · 2N_k' ≤ δ_k · 2N_k)
            let target = (0..r.class as usize).rev().find(|&kk| pend.admitted.get(kk).copied().unwrap_or(true));
            if footprint_rule && target == Some(k) && !footprint_inside(m, pred, &td.pc.mvp, w, h) {
              rep.d3_at_classify += 1;
              continue;
            }
            if target == Some(k) {
              tc.owner[s as usize] = owner_byte(k, (b & !OWN_CLASS_MASK) | OWN_PENDING);
              list.push(s);
              if r.g_px_per_m.is_finite() {
                tg = tg.max(r.g_px_per_m);
              }
            }
          }
          total += list.len() as u32;
          g_max = g_max.max(tg);
          tc.lists[k][p][2] = list;
          tc.meta[k][p][2] = ListMeta { classified_frame: self.drawn_frame, de_c: pend.de, g_max: tg };
        }
        let _ = g_max;
        rep.classified.push((k, p, total));
      }
    }
    // F3: the build chunks
    let hn = PYRAMID_HEADER_WORDS as usize;
    // the image's own header words: its geometry (0–2, the level table 16–57), PYR_TAU_MAX,
    // PYR_LEVEL_MASK, PYR_UNIT and word 58 (the GPU's owner address)
    let own = |wd: usize| -> bool {
      wd <= 2 || (16..=58).contains(&wd) || wd == PYR_TAU_MAX as usize || wd == PYR_LEVEL_MASK as usize || wd == PYR_UNIT as usize
    };
    for (ti, td) in tiers.iter_mut().enumerate() {
      let key = pend.keys[ti];
      let anchor = pend.anchors[ti];
      for (k, p, slots) in &pend.chunks[ti] {
        let (k, p) = (*k, *p);
        let im = &mut self.images[k * np + p];
        let Some(fv) = im.build_at(pend.nucleus) else { continue };
        if im.build.is_empty() {
          continue;
        }
        {
          let n = pend.nucleus;
          let dist = |e: [f64; 3]| ((e[0] - n[0]).powi(2) + (e[1] - n[1]).powi(2) + (e[2] - n[2]).powi(2)).sqrt();
          let cur = self.prev.as_ref().map_or(f64::NAN, |pv| dist(pv.view.eye_m));
          rep.frozen_eye.push((k, p, dist(im.build_eye_helio), dist(fv.eye_m), cur));
        }
        let tc = self.tiers.get_mut(&key).expect("tier");
        let live = tc.live_count.min(td.render.len() as u32).min(td.moments.len() as u32);
        let mvp_s = fv.mvp_image(anchor, &geom);
        let e = fv.eye_local(anchor);
        let mut pc = td.pc;
        pc.mvp = mvp_s;
        pc.eye_local = [e[0], e[1], e[2], td.pc.eye_local[3]];
        pc.inv_unit = if im.build_unit > 0.0 { 1.0 / im.build_unit } else { 0.0 };
        let mut batch = alloc::vec::Vec::new();
        for &s in slots {
          if !tc.valid(s, k, OWN_BUILD) {
            continue;
          }
          let o = tc.offset(s);
          if o >= live || !render_live(td.render[o as usize].age_id_dbeta_flux[1]) || !(td.moments[o as usize].flux() > 0.0) {
            continue;
          }
          // the continuum whole: an owned cluster is never walked for specks
          td.moments[o as usize].edges[15][3] = 0.0;
          let pred = streak_pred(td.render, o as usize).map(|q| &td.moments[q]);
          // the footprint rule (D3) only without a margin; with one, every entry is drawn,
          // clipped at the image's own bounds
          let inside = !footprint_rule || footprint_inside(&td.moments[o as usize], pred, &mvp_s, w, h);
          if footprint_rule && !rep.d3_samples.iter().any(|x| x.0 == k && x.1 == inside) {
            rep.d3_samples.push((k, inside, footprint_report(&td.moments[o as usize], pred, &mvp_s, w, h)));
          }
          if !inside {
            rep.d3_rejected += 1;
            im.build_d3 += 1;
            continue;
          }
          im.build_drawn += 1;
          tc.owner[s as usize] |= OWN_IN_IMAGE;
          batch.push(o);
        }
        if batch.is_empty() {
          continue;
        }
        // the header routing: the frame's words in, the image's own three kept, the counters out
        for wd in 0..hn {
          if !own(wd) {
            im.build[wd] = frame[wd];
          }
        }
        let il = &geom.layout;
        if self.trace_light {
          let lights = draw_list_traced(self.threads, td.moments, td.render, &pc, &td.flow, il, &mut im.build, &batch);
          rep.drawn_build.extend(batch.iter().zip(lights).map(|(&o, light)| CadenceDrawn { tier: ti, k, p, offset: o, light }));
        } else {
          Self::draw_list(self.threads, td.moments, td.render, &pc, &td.flow, il, &mut im.build, &batch);
          rep.drawn_build.extend(batch.iter().map(|&o| CadenceDrawn { tier: ti, k, p, offset: o, light: 0.0 }));
        }
        for wd in 0..hn {
          if !own(wd) {
            frame[wd] = im.build[wd];
          }
        }
      }
    }
    // F4: the reprojection of the shown images (its light alone, the coverage share's numerator)
    let mut reprojected_light = 0.0f64;
    {
      let view = self.prev.as_ref().expect("a begun frame").view;
      let unit_now = if tiers.is_empty() || !(tiers[0].pc.inv_unit > 0.0) { 0.0 } else { 1.0 / tiers[0].pc.inv_unit };
      let mut seed = 0xC4DE_5EEDu32;
      let mut shown: alloc::vec::Vec<ShownImage<'_>> = alloc::vec::Vec::new();
      for im in &self.images {
        let Some(sv) = im.shown_at(pend.nucleus) else { continue };
        if im.shown.is_empty() {
          continue;
        }
        let Some(map) = view.map_to_image(&sv, &geom) else { continue };
        seed ^= dust_hash32(im.generation);
        shown.push(ShownImage {
          words: &im.shown,
          map,
          unit: im.shown_unit,
          layout: geom.has_margin().then_some(&geom.layout),
        });
      }
      rep.images_shown = shown.len() as u32;
      if !shown.is_empty() {
        reproject_add(layout, frame, &shown, unit_now, dust_hash32(seed), ReprojectFactor::Unit);
        if !pend.cycles.is_empty() {
          reprojected_light = pyramid_light(layout, frame) * unit_now as f64;
        }
      }
    }
    // F5: the speck pass (n_eye over every live cluster; walks only where not SHOWN)
    for (ti, td) in tiers.iter_mut().enumerate() {
      let Some(spc) = td.specks_pc else { continue };
      let tc = &self.tiers[&pend.keys[ti]];
      let walk = |o: usize| tc.owner[tc.slot(o as u32) as usize] & OWN_SHOWN == 0;
      let st = super::specks::specks_tier_walking(td.moments, td.render, &spc, &td.flow, layout, frame, &walk);
      rep.specks.push(st);
    }
    // F6: the fresh splat
    for (ti, td) in tiers.iter().enumerate() {
      Self::draw_list(self.threads, td.moments, td.render, &td.pc, &td.flow, layout, frame, &pend.fresh[ti]);
    }
    // the coverage at the cycle starts: the light the shown images put into this frame (the
    // frame right after the reprojection) against the frame's whole light
    if !pend.cycles.is_empty() {
      let unit_now = if tiers.is_empty() || !(tiers[0].pc.inv_unit > 0.0) { 0.0 } else { 1.0 / tiers[0].pc.inv_unit as f64 };
      let total = pyramid_light(layout, frame) * unit_now;
      for c in &pend.cycles {
        rep.cycles.push(CadenceCycle { shown_light_share: reprojected_light / total.max(1e-300), ..*c });
      }
    }
    if mode == Some(CadenceMode::Run) {
      self.drawn_frame += 1;
    }
    rep
  }

  /// a list draw on `threads` threads (std) or serially
  #[allow(clippy::too_many_arguments)]
  fn draw_list(
    threads: usize,
    moments: &[DustMoments],
    render: &[DustRenderCluster],
    pc: &DustSplatPushConstants,
    flow: &DustFlowUniform,
    layout: &PyramidLayout,
    pyramid: &mut [u32],
    list: &[u32],
  ) {
    #[cfg(feature = "std")]
    if threads > 1 {
      super::splat::splat_tier_list_parallel(moments, render, pc, flow, 0, layout, pyramid, list, threads);
      return;
    }
    let _ = threads;
    splat_tier_list(moments, render, pc, flow, 0, layout, pyramid, list);
  }

  /// D1's check (tests call it every frame): every state bit of every stored owner byte is backed
  /// by a membership of the slot in the list of that class, phase and state; IN_IMAGE implies
  /// BUILD; a class id goes with at least one state bit and back; no valid entry appears twice in
  /// a list; every valid entry sits in the list of its parity.
  pub fn check_owner_bytes(&self) -> Result<(), alloc::string::String> {
    let np = self.np();
    let bits = [OWN_SHOWN, OWN_BUILD, OWN_PENDING];
    for (key, t) in &self.tiers {
      let mut rec = alloc::vec![0u8; t.owner.len()];
      for (k, kl) in t.lists.iter().enumerate() {
        for (p, pl) in kl.iter().enumerate() {
          for (st, l) in pl.iter().enumerate() {
            for &s in l {
              if !t.valid(s, k, bits[st]) {
                continue;
              }
              if np == 2 && (s & 1) as usize != p {
                return Err(alloc::format!("tier {key}: slot {s} valid in the phase-{p} list of class {k}"));
              }
              if rec[s as usize] & bits[st] != 0 {
                return Err(alloc::format!("tier {key}: slot {s} twice in list ({k}, {p}, {st})"));
              }
              rec[s as usize] |= bits[st];
            }
          }
        }
      }
      for (s, &b) in t.owner.iter().enumerate() {
        let states = b & (OWN_SHOWN | OWN_BUILD | OWN_PENDING);
        if states & !rec[s] != 0 {
          return Err(alloc::format!("tier {key}: slot {s} byte {b:#04x} has a state no list backs ({:#04x})", rec[s]));
        }
        if b & OWN_IN_IMAGE != 0 && b & OWN_BUILD == 0 {
          return Err(alloc::format!("tier {key}: slot {s} byte {b:#04x}: IN_IMAGE without BUILD"));
        }
        if (b & OWN_CLASS_MASK == 0) != (b & !OWN_CLASS_MASK == 0) {
          return Err(alloc::format!("tier {key}: slot {s} byte {b:#04x}: a class id without a state or back"));
        }
      }
    }
    Ok(())
  }

  /// A digest of the whole state (the reused-frame test: nothing advances)
  pub fn state_digest(&self) -> u64 {
    let mut hsh = 0xcbf2_9ce4_8422_2325u64;
    fnv(&mut hsh, self.drawn_frame);
    for (key, t) in &self.tiers {
      fnv(&mut hsh, *key);
      fnv(&mut hsh, t.first_slot as u64 | (t.live_count as u64) << 32);
      fnv(&mut hsh, t.emitted);
      for &b in &t.owner {
        fnv(&mut hsh, b as u64);
      }
      for kl in &t.lists {
        for pl in kl {
          for l in pl {
            fnv(&mut hsh, l.len() as u64);
            for &s in l {
              fnv(&mut hsh, s as u64);
            }
          }
        }
      }
      for v in t.cursor.iter().chain(t.step.iter()) {
        for &c in v {
          fnv(&mut hsh, c as u64);
        }
      }
    }
    for im in &self.images {
      fnv(&mut hsh, im.generation as u64);
      for &wd in im.shown.iter().chain(im.build.iter()) {
        fnv(&mut hsh, wd as u64);
      }
    }
    fnv(&mut hsh, self.de_hist.len() as u64);
    hsh
  }
}

// ─── The budget loop (step 5: option B on top of A) ─────────────────────────

/// ε where the budget loop starts (px); step 3's tests keep [`DUST_CADENCE_EPS_DEFAULT`]
pub const DUST_CADENCE_EPS_LOOP_START: f32 = 4.0;
/// the ε ladder's ceiling (px): the last ε that is an on-screen tolerance
pub const DUST_CADENCE_EPS_MAX: f32 = 16.0;
/// the by-count fallback's ε ceiling (px): every finite rate is admitted (the fallback's ε is a
/// count knob: the fresh set is the top-K by δ outside the protected set of
/// [`DUST_CADENCE_FALLBACK_SHAPE`], the staleness bound is `2N` frames)
pub const DUST_CADENCE_EPS_FALLBACK_MAX: f32 = 1.0e6;
/// the by-count fallback's shape bound θ ([`RateInputs::shape_frac`]): under the fallback a
/// cluster is held in class `k` only while `2N_k · Δt ≤ θ · age`. ε as a count knob admits every
/// finite on-screen rate, and the rate (px per frame) is smallest where the image is brightest:
/// the young dust at the nucleus (its 2 m/s ejection against the old dust's radiation drift with
/// a following camera; mid-depth of the coma with a still one) — at 1 d/s the top 10 % by δ held
/// none of the youngest 1 % or of the 1 % nearest the nucleus (`fallback_keeps_the_nucleus_fresh`).
/// A cluster's extent grows ∝ its age, so over its `2N_k` stale frames it reshapes by
/// `2N_k Δt / age` of itself: θ bounds that relative change whatever ε, and every cluster younger
/// than `2N₀ Δt / θ` is fresh (25.6 h at Δt 1 440 s and N₀ 8: ~3.6 % of tier 0's 30 days). The
/// fresh set is that protected set P plus the rate's top by δ; the loop's ε search counts both
/// in the measured fresh count ([`BudgetSample::fresh`]), so K is spent on the rate's top
/// `K − |P|`. A P over K alone leaves ε at its ceiling, then the N raises (which keep N₀ and so P)
/// and the verdict: P is the fallback's floor, `|P| ≈ live₀ · 2N₀ Δt / (θ · TTL)`.
pub const DUST_CADENCE_FALLBACK_SHAPE: f32 = 0.25;
/// the ladder's step of a knob
pub const DUST_BUDGET_STEP: f32 = 1.25;
/// the frame budget the loop holds (ms; `AETHERVK_DUST_BUDGET_MS`): the app's 8 ms at 1280×800
pub const DUST_BUDGET_MS_DEFAULT: f64 = 8.0;
/// the EWA and volume feedback budgets are halved up to this many times (the in-frame hard gates
/// stay at `DUST_FRAME_HARD_FACTOR` × the default budgets)
pub const DUST_BUDGET_SHIFT_MAX: u32 = 3;
/// the fallback's raises of the longest class (× 2 each: 64 → 128 → 256)
pub const DUST_BUDGET_N_LEVELS: u32 = 2;
/// measured frames at one setting before the loop decides
pub const DUST_BUDGET_SAMPLES: usize = 3;
/// the share of the budget the fallback sizes the fresh set for (K)
pub const DUST_BUDGET_HEADROOM: f64 = 0.9;
/// the specks' share of the budget (their shell-radius cap is the specks' time knob)
pub const DUST_BUDGET_SPECKS_SHARE: f64 = 0.25;

/// One rung of the budget loop's ladder (a pinned knob's rungs are left out)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetRung {
  /// ε = `min(START · 1.25^i, MAX)` (rung `Eps(0)` is the rest state)
  Eps(u32),
  /// lod_px = `min(DEFAULT · 1.25^i, MAX)`
  Lod(u32),
  /// the EWA and volume feedback budgets `>> i`
  Budget(u32),
}

/// The knobs a [`BudgetLoop`] holds this frame
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetKnobs {
  pub eps_px: f32,
  pub lod_px: f32,
  /// the EWA and volume feedback budgets are `>> budget_shift`
  pub budget_shift: u32,
  /// the by-count fallback is on (ε is then a count knob)
  pub fallback: bool,
  /// the longest class is `× 2^n_level`
  pub n_level: u32,
  /// every knob at its ceiling and still over the budget (the verdict state)
  pub saturated: bool,
}

impl BudgetKnobs {
  /// The cadence's shape bound θ under these knobs ([`RateInputs::shape_frac`]):
  /// [`DUST_CADENCE_FALLBACK_SHAPE`] under the by-count fallback, else 0 (off: the ε rule alone)
  pub fn shape_frac(&self) -> f32 {
    if self.fallback { DUST_CADENCE_FALLBACK_SHAPE } else { 0.0 }
  }
}

/// Pinned knobs (`AETHERVK_DUST_CADENCE_EPS`, `AETHERVK_DUST_LOD_PX`, `AETHERVK_DUST_BUDGET_SHIFT`)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BudgetPins {
  pub eps_px: Option<f32>,
  pub lod_px: Option<f32>,
  pub budget_shift: Option<u32>,
}

impl BudgetPins {
  pub fn from_env() -> Self {
    let var = aethervk_oshal_rlib::os::env::var;
    let f = |n: &str| var(n).and_then(|v| v.trim().parse::<f32>().ok()).filter(|v| v.is_finite() && *v >= 0.0);
    Self {
      eps_px: f("AETHERVK_DUST_CADENCE_EPS").filter(|v| *v > 0.0),
      lod_px: f("AETHERVK_DUST_LOD_PX"),
      budget_shift: var("AETHERVK_DUST_BUDGET_SHIFT").and_then(|v| v.trim().parse::<u32>().ok()).map(|s| s.min(DUST_BUDGET_SHIFT_MAX)),
    }
  }
}

/// One measured, drawn, cadenced frame (the loop never sees a reused or a full-draw frame)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetSample {
  /// the view's frame number the timings are of
  pub frame: u64,
  pub total_ms: f64,
  /// the fresh splat (`DustGpuTimings::splat_only_ms`)
  pub fresh_ms: f64,
  /// the clusters drawn fresh (the cadence's newest state)
  pub fresh: u32,
}

impl BudgetSample {
  /// The fallback's K: the budget left after the frame's other costs over the measured cost per
  /// fresh cluster; `None` without a measured cost
  pub fn k_target(&self, budget_ms: f64) -> Option<u32> {
    if !(self.fresh > 0 && self.fresh_ms > 0.0 && self.total_ms.is_finite()) {
      return None;
    }
    let per = self.fresh_ms / self.fresh as f64;
    let avail = DUST_BUDGET_HEADROOM * budget_ms - (self.total_ms - self.fresh_ms);
    // (a float → u32 cast truncates toward zero and saturates: the floor of a positive quotient)
    Some(if avail > 0.0 { (avail / per) as u32 } else { 0 })
  }
}

/// `x · 1.25^i`, capped
fn budget_step_pow(x: f32, i: u32, cap: f32) -> f32 {
  let mut v = x;
  for _ in 0..i {
    v *= DUST_BUDGET_STEP;
  }
  v.min(cap)
}

/// The ε rungs: `Eps(0)` .. the first index whose value reaches [`DUST_CADENCE_EPS_MAX`]
fn budget_eps_rungs() -> u32 {
  let mut i = 0;
  while budget_step_pow(DUST_CADENCE_EPS_LOOP_START, i, DUST_CADENCE_EPS_MAX) < DUST_CADENCE_EPS_MAX {
    i += 1;
  }
  i
}

/// The lod rungs: `Lod(1)` .. the first reaching `DUST_LOD_PX_MAX`
fn budget_lod_rungs() -> u32 {
  let mut i = 0;
  while budget_step_pow(super::DUST_LOD_PX_DEFAULT, i, super::DUST_LOD_PX_MAX) < super::DUST_LOD_PX_MAX {
    i += 1;
  }
  i
}

/// The frame-time loop (plan item 7): a ladder ε → lod → budgets, then the by-count fallback
/// (ε as a count knob, then the longest class's N), moved up when [`DUST_BUDGET_SAMPLES`]
/// consecutive samples at one setting are all over the budget, down (reverse order) when all are
/// under half of it. Host only: the device feeds it samples ([`Self::observe`]) and applies
/// [`Self::knobs`].
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetLoop {
  pub budget_ms: f64,
  pub pins: BudgetPins,
  pub ladder: alloc::vec::Vec<BudgetRung>,
  /// the ladder index
  pub rung: usize,
  /// the fallback's ε and N level (`None`: on the ladder)
  pub fallback: Option<(f32, u32)>,
  pub saturated: bool,
  /// the ratios `total / budget` measured at this setting (newest last)
  pub ratios: alloc::vec::Vec<f64>,
  /// samples of frames before this one were recorded before the last move
  pub since: u64,
  /// an exponential average of the ratio (weight 1/4; telemetry)
  pub ema: f64,
  /// the last sample's K (the fallback's arithmetic; telemetry)
  pub k_last: Option<u32>,
  pub moves: u32,
  /// the specks' shell radius cap (local m; `None`: uncapped)
  pub shell_cap_m: Option<f32>,
  /// the cadence's class lengths before the fallback raised any (the device keeps them here)
  pub base_classes: Option<alloc::vec::Vec<u32>>,
}

impl BudgetLoop {
  pub fn new(budget_ms: f64, pins: BudgetPins) -> Self {
    let mut ladder = alloc::vec![BudgetRung::Eps(0)];
    if pins.eps_px.is_none() {
      ladder.extend((1..=budget_eps_rungs()).map(BudgetRung::Eps));
    }
    if pins.lod_px.is_none() {
      ladder.extend((1..=budget_lod_rungs()).map(BudgetRung::Lod));
    }
    if pins.budget_shift.is_none() {
      ladder.extend((1..=DUST_BUDGET_SHIFT_MAX).map(BudgetRung::Budget));
    }
    Self {
      budget_ms: if budget_ms > 0.0 && budget_ms.is_finite() { budget_ms } else { DUST_BUDGET_MS_DEFAULT },
      pins,
      ladder,
      rung: 0,
      fallback: None,
      saturated: false,
      ratios: alloc::vec::Vec::new(),
      since: 0,
      ema: f64::NAN,
      k_last: None,
      moves: 0,
      shell_cap_m: None,
      base_classes: None,
    }
  }

  /// `AETHERVK_DUST_BUDGET_LOOP=0`: `None` (the loop off); else the env's budget and pins
  pub fn from_env() -> Option<Self> {
    let var = aethervk_oshal_rlib::os::env::var;
    if var("AETHERVK_DUST_BUDGET_LOOP").is_some_and(|v| v.trim() == "0") {
      return None;
    }
    let ms = var("AETHERVK_DUST_BUDGET_MS").and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(DUST_BUDGET_MS_DEFAULT);
    Some(Self::new(ms, BudgetPins::from_env()))
  }

  /// The knobs of the current setting
  pub fn knobs(&self) -> BudgetKnobs {
    let (mut ei, mut li, mut bi) = (0u32, 0u32, 0u32);
    for r in &self.ladder[..=self.rung] {
      match *r {
        BudgetRung::Eps(i) => ei = i,
        BudgetRung::Lod(i) => li = i,
        BudgetRung::Budget(i) => bi = i,
      }
    }
    let ladder_eps = self.pins.eps_px.unwrap_or_else(|| budget_step_pow(DUST_CADENCE_EPS_LOOP_START, ei, DUST_CADENCE_EPS_MAX));
    let (eps_px, n_level) = match self.fallback {
      Some((e, n)) => (self.pins.eps_px.unwrap_or(e), n),
      None => (ladder_eps, 0),
    };
    BudgetKnobs {
      eps_px,
      lod_px: self.pins.lod_px.unwrap_or_else(|| budget_step_pow(super::DUST_LOD_PX_DEFAULT, li, super::DUST_LOD_PX_MAX)),
      budget_shift: self.pins.budget_shift.unwrap_or(bi),
      fallback: self.fallback.is_some(),
      n_level,
      saturated: self.saturated,
    }
  }

  /// The class lengths of `base` under the fallback's N level (the longest `× 2^n`)
  pub fn classes(&self, base: &[u32]) -> alloc::vec::Vec<u32> {
    let mut c = base.to_vec();
    if let Some(l) = c.last_mut() {
      *l <<= self.knobs().n_level;
    }
    c
  }

  /// Folds one sample taken at view frame `now` (the frame being begun: a move applies to it);
  /// returns whether the knobs moved
  pub fn observe(&mut self, s: &BudgetSample, now: u64) -> bool {
    if s.frame < self.since || !(s.total_ms.is_finite() && s.total_ms >= 0.0) {
      return false;
    }
    let r = s.total_ms / self.budget_ms;
    self.ema = if self.ema.is_nan() { r } else { self.ema + (r - self.ema) * 0.25 };
    self.k_last = s.k_target(self.budget_ms);
    self.ratios.push(r);
    if self.ratios.len() > DUST_BUDGET_SAMPLES {
      self.ratios.remove(0);
    }
    if self.ratios.len() < DUST_BUDGET_SAMPLES {
      return false;
    }
    let lo = self.ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = self.ratios.iter().copied().fold(0.0, f64::max);
    let moved = if lo > 1.0 {
      self.up(lo, s)
    } else if hi < 0.5 {
      self.down()
    } else {
      false
    };
    if moved {
      self.ratios.clear();
      self.since = now;
      self.moves += 1;
    }
    moved
  }

  fn up(&mut self, lo: f64, s: &BudgetSample) -> bool {
    let top = self.ladder.len() - 1;
    let Some((eps, n)) = self.fallback else {
      if self.rung < top {
        let steps = if lo < 2.0 { 1 } else if lo < 8.0 { 2 } else { 3 };
        self.rung = (self.rung + steps).min(top);
      } else {
        let e = self.knobs().eps_px;
        self.fallback = Some(((e * DUST_BUDGET_STEP).min(DUST_CADENCE_EPS_FALLBACK_MAX), 0));
      }
      return true;
    };
    let k = s.k_target(self.budget_ms);
    let fresh_over = k.is_none_or(|k| s.fresh > k);
    if fresh_over && self.pins.eps_px.is_none() && eps < DUST_CADENCE_EPS_FALLBACK_MAX {
      let f = k.map_or(4.0, |k| s.fresh as f32 / k.max(1) as f32).clamp(DUST_BUDGET_STEP, 16.0);
      self.fallback = Some(((eps * f).min(DUST_CADENCE_EPS_FALLBACK_MAX), n));
      return true;
    }
    if n < DUST_BUDGET_N_LEVELS {
      self.fallback = Some((eps, n + 1));
      return true;
    }
    self.saturated = true;
    false
  }

  fn down(&mut self) -> bool {
    self.saturated = false;
    if let Some((eps, n)) = self.fallback {
      if n > 0 {
        self.fallback = Some((eps, n - 1));
      } else if self.pins.eps_px.is_none() && eps / DUST_BUDGET_STEP > DUST_CADENCE_EPS_MAX {
        self.fallback = Some((eps / DUST_BUDGET_STEP, 0));
      } else {
        self.fallback = None;
      }
      return true;
    }
    if self.rung > 0 {
      self.rung -= 1;
      return true;
    }
    false
  }

  /// The specks' shell cap from one frame's speck time and the measured (uncapped) radius:
  /// over [`DUST_BUDGET_SPECKS_SHARE`] of the budget → 0.8 × the radius in use; under half of
  /// it → × 1.25, removed once at the measured radius. Returns the cap.
  pub fn observe_specks(&mut self, specks_ms: f64, measured_m: f32) -> Option<f32> {
    if !(specks_ms.is_finite() && measured_m > 0.0) {
      return self.shell_cap_m;
    }
    let share = specks_ms / self.budget_ms;
    let used = self.shell_cap_m.map_or(measured_m, |c| c.min(measured_m));
    if share > DUST_BUDGET_SPECKS_SHARE {
      self.shell_cap_m = Some(0.8 * used);
    } else if share < 0.5 * DUST_BUDGET_SPECKS_SHARE {
      if let Some(c) = self.shell_cap_m {
        let up = c * DUST_BUDGET_STEP;
        self.shell_cap_m = if up >= measured_m { None } else { Some(up) };
      }
    }
    self.shell_cap_m
  }

  /// the staleness bound (frames) of the classes `base` under the current N level: `2 · N_max`
  pub fn staleness_frames(&self, base: &[u32]) -> u32 {
    2 * self.classes(base).last().copied().unwrap_or(0)
  }

  /// the loop's state for the telemetry: `"saturated"`, `"settling"` (fewer samples than
  /// [`DUST_BUDGET_SAMPLES`] at this setting) or `"hold"`
  pub fn state_name(&self) -> &'static str {
    if self.saturated {
      "saturated"
    } else if self.ratios.len() < DUST_BUDGET_SAMPLES {
      "settling"
    } else {
      "hold"
    }
  }
}
