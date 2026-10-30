//! The GPU mirror of the cadence classes (temporal reuse).
//!
//! The CPU mirror [`crate::scene::dust::cadence`] is the specification: the same decisions in the
//! same order. The per-slot work (the owner bytes, the lists, the cursors and steps, the aborts'
//! test and effects, the propagate bitmask, the chunk pre-pass) runs on the GPU
//! (`dust_classify.comp`, `dust_compact.comp`, layouts in `dust_cadence.glsl`); the host keeps
//! what the CPU's [`crate::scene::dust::cadence::DustCadenceState::begin`] decides from host data
//! alone (the mode, the resets, the range hooks, the eye steps and window terms, the cycle
//! schedule, the frozen views, the units and generations) and the upper bounds it records with
//! (the classified lengths read back [`DUST_READBACK_SLOTS`] frames later, before their promotion:
//! `N ≥ 8`).
//!
//! Per frame: [`Device::cmd_dust_cadence_begin`] (fills, params, CONTROL, TRANSITION, the swap
//! lists' compaction, the propagate marks) → [`Device::cmd_dust_cadence_propagate`] per tier →
//! [`Device::cmd_dust_cadence_classify`] per tier (CLASSIFY, the pending lists' compaction) →
//! [`Device::cmd_dust_cadence_chunks`] per tier (the chunk pre-pass; the draws are I3) →
//! [`Device::cmd_dust_cadence_end`] (the state's readback copy).
//!
//! Limitations (recorded): a view's cadence is exact only when one view draws a system per frame
//! (the moments are per system); `CadenceReset::Off` is reported from the host's knowledge (any
//! classify since the reset) where the CPU reads its owner bytes — the clear happens either way.

use super::*;
use crate::scene::dust::cadence::{
  ABORT_BUDGET, ABORT_PREDICTION, CadenceConfig, CadenceMode, CadenceReset, CadenceView, DUST_CADENCE_ABORT_FACTOR,
  DUST_CADENCE_ANCHOR_HISTORY, DUST_CADENCE_HISTORY, DUST_CADENCE_MARGIN_GUARD, DUST_CADENCE_MAX_CLASSES,
  DUST_CADENCE_ORTHO_ZOOM_MAX, FrameMeta, ImageGeom, anchor_delta_f32, corner_motion, frozen_at, image_covers,
};

// ─── GPU layouts (mirror of dust_cadence.glsl) ──────────────────────────────

pub const CADP_DT: usize = 0;
pub const CADP_DEYE: usize = 1;
pub const CADP_AW: usize = 4;
pub const CADP_EPS: usize = 7;
pub const CADP_FRAME_PX: usize = 8;
pub const CADP_W: usize = 9;
pub const CADP_H: usize = 10;
pub const CADP_NC: usize = 11;
pub const CADP_NP: usize = 12;
pub const CADP_N: usize = 13;
pub const CADP_NSWAP: usize = 16;
pub const CADP_SWAP0: usize = 17;
pub const CADP_FLAGS: usize = 23;
pub const CADP_DEBUG: usize = 24;
pub const CADP_DMAX: usize = 26;
/// f32 the rate rule's shape bound θ (`RateInputs::shape_frac`; 0: off)
pub const CADP_SHAPE: usize = 27;
pub const CADP_STATE: usize = 28;
pub const CADP_NT: usize = 30;
pub const CADP_ADMIT: usize = 31;
pub const CADP_DD: usize = 32;
pub const CADP_BOUND: usize = 44;
pub const CADP_FORCE: usize = 48;
pub const CADP_COVER: usize = 60;
pub const CADP_FOOT: usize = 61;
pub const CADP_ANCH: usize = 62;
pub const CADP_TIER0: usize = 64;
pub const CADP_TIER_WORDS: usize = 8;
pub const CADP_STEPU: usize = 128;
pub const CADP_CLEAR0: usize = 192;
pub const CADP_CLEAR_WORDS: usize = 8;
pub const CADP_WORDS: usize = 256;
pub const CAD_MAX_TIERS: usize = 8;
pub const CADF_ADVANCE: u32 = 1;
pub const CADF_ABORTS: u32 = 2;
pub const CADF_COVERAGE: u32 = 4;
/// the classify retains a member whose target is a slower class (`CadenceConfig::retain`)
pub const CADF_RETAIN: u32 = 8;
/// the handover (`CadenceConfig::handover_on`): incoming memberships of slower classes, aligned
/// phases
pub const CADF_HANDOVER: u32 = 16;
/// per (k * 2 + p): drawn frames to the pair's next swap (`DustCadenceState::frames_to_swap`; the
/// handover's rule a)
pub const CADP_FTS: usize = 176;
/// the anchor tables: per tier slot, words 0 n, 1 this frame's stamp, then n × (stamp, Δ xyz)
pub const CAD_ANCH_TIER_WORDS: usize = 2304;
pub const CAD_ANCH_ENTRY0: usize = 4;
const _: () = assert!(CAD_ANCH_ENTRY0 + 4 * DUST_CADENCE_ANCHOR_HISTORY <= CAD_ANCH_TIER_WORDS);

pub const CADS_PAIR0: usize = 16;
pub const CADS_PAIR_WORDS: usize = 16;
pub const CADS_TP0: usize = 128;
pub const CADS_TP_WORDS: usize = 16;
pub const CADS_HIST: usize = 1024;
pub const CADS_SHELL_FORCED: usize = 1028;
pub const CADS_PROF: usize = 1536;
pub const CADS_WORDS: usize = 2048;
const _: () = assert!(CADS_TP0 + CADS_TP_WORDS * CAD_MAX_TIERS * 6 <= CADS_HIST);
const _: () = assert!(CADS_PAIR0 + CADS_PAIR_WORDS * 6 <= CADS_TP0);
const _: () = assert!(CADP_STEPU + CAD_MAX_TIERS * 6 <= CADP_FTS);
const _: () = assert!(CADP_FTS + 6 <= CADP_CLEAR0);
const _: () = assert!(CADP_CLEAR0 + CADP_CLEAR_WORDS * CAD_MAX_TIERS <= CADP_WORDS);
const _: () = assert!(CADP_TIER0 + CADP_TIER_WORDS * CAD_MAX_TIERS <= CADP_STEPU);

/// the state word of pair `(k, p)`
pub fn cads_pair(k: usize, p: usize) -> usize {
  CADS_PAIR0 + CADS_PAIR_WORDS * (k * 2 + p)
}
/// the state word of tier slot `t`, pair `(k, p)`
pub fn cads_tp(t: usize, k: usize, p: usize) -> usize {
  CADS_TP0 + CADS_TP_WORDS * ((t * 3 + k) * 2 + p)
}

/// dust_classify.comp's modes
pub const CADM_CLASSIFY: u32 = 0;
pub const CADM_TRANSITION: u32 = 1;
pub const CADM_MARK_FRESH: u32 = 2;
pub const CADM_MARK_CHUNK: u32 = 3;
pub const CADM_CHUNK: u32 = 4;
pub const CADM_CLEAR: u32 = 5;
pub const CADM_DEBUG: u32 = 6;
/// dust_classify.comp's stage modes (`DUST_CAD_STAGES`): the slot's evaluation, each stage
/// its own dispatch writing the slot's row (`CadenceSysLayout::rows0`, dust_cadence.glsl's
/// `DUST_CADR_*`), a barrier between, in the order PREP (CLASSIFY) or PREP_DEBUG (DEBUG), LAM, SHELL,
/// EXTENT, NEAR, JAC, STEP, RATE, FOOT; then CLASSIFY or DEBUG reads the row. CHUNK_FOOT before
/// CHUNK (the footprint rule into the chunk's draw-list words)
pub const CADM_PREP: u32 = 7;
pub const CADM_PREP_DEBUG: u32 = 8;
pub const CADM_LAM: u32 = 9;
pub const CADM_SHELL: u32 = 10;
pub const CADM_EXTENT: u32 = 11;
pub const CADM_NEAR: u32 = 12;
pub const CADM_JAC: u32 = 13;
pub const CADM_STEP: u32 = 14;
pub const CADM_RATE: u32 = 15;
pub const CADM_FOOT: u32 = 16;
pub const CADM_CHUNK_FOOT: u32 = 17;
/// the stages after PREP / PREP_DEBUG, in dispatch order
pub const CAD_EVAL_STAGES: [u32; 8] = [CADM_LAM, CADM_SHELL, CADM_EXTENT, CADM_NEAR, CADM_JAC, CADM_STEP, CADM_RATE, CADM_FOOT];
/// words of a slot's row (dust_cadence.glsl's `DUST_CADR_WORDS`)
pub const CADR_WORDS: u32 = 12;
/// `pc` with its mode byte (`tier_mode` bits 8-15) replaced by `mode`
fn cad_pc_mode(pc: &DustClassifyPushConstants, mode: u32) -> DustClassifyPushConstants {
  DustClassifyPushConstants { tier_mode: (pc.tier_mode & !0xFF00) | mode << 8, ..*pc }
}
/// words per debug record (class, δ0, δ1, δ2, g, flags: 1 live and lit, 2 shell-capable, 4
/// enclosing, slot, 0)
pub const CAD_DEBUG_WORDS: usize = 8;
/// dust_compact.comp's modes
pub const CMPM_COUNT: u32 = 0;
pub const CMPM_PREFIX: u32 = 1;
pub const CMPM_SCATTER: u32 = 2;
pub const CMPM_CONTROL: u32 = 3;
pub const CMP_BLOCK: u32 = 1024;

/// `dust_classify.comp`'s push constants
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustClassifyPushConstants {
  pub moments: u64,
  pub render: u64,
  pub sys: u64,
  pub params: u64,
  pub mvp: [f32; 16],
  pub eye_local: [f32; 4],
  pub first_slot: u32,
  pub live_count: u32,
  pub ring_mask: u32,
  /// tier | mode << 8 | (k | p << 4) << 16
  pub tier_mode: u32,
}
const _: () = assert!(core::mem::size_of::<DustClassifyPushConstants>() == 128);

/// `dust_compact.comp`'s push constants
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustCompactPushConstants {
  pub sys: u64,
  pub params: u64,
  pub first_slot: u32,
  pub live_count: u32,
  pub ring_mask: u32,
  /// tier | mode << 8 | set << 16
  pub tier_mode: u32,
}
const _: () = assert!(core::mem::size_of::<DustCompactPushConstants>() == 32);

/// The per-system cadence buffer's words (mirror of `dust_cad_*` in dust_cadence.glsl)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CadenceSysLayout {
  /// the system ring's capacity R (slots; a multiple of 32)
  pub ring: u32,
  pub nc: u32,
  pub np: u32,
}

impl CadenceSysLayout {
  pub fn pmask0(&self) -> u32 {
    self.ring / 4
  }
  pub fn lists0(&self) -> u32 {
    self.ring / 4 + self.ring / 32
  }
  pub fn list(&self, rb: u32, cap: u32, k: u32, p: u32, s: u32) -> u32 {
    self.lists0() + 3 * self.nc * rb + ((k * self.np + p) * 3 + s) * (cap / self.np)
  }
  pub fn draw0(&self) -> u32 {
    self.lists0() + 3 * self.nc * self.ring
  }
  pub fn draw(&self, rb: u32, cap: u32, k: u32, p: u32) -> u32 {
    self.draw0() + self.nc * rb + (k * self.np + p) * (cap / self.np)
  }
  pub fn blocks0(&self) -> u32 {
    self.draw0() + self.nc * self.ring
  }
  /// the incoming bytes (the handover; byte ring base + slot, as the owner bytes)
  pub fn incoming0(&self) -> u32 {
    self.blocks0() + 6 * self.nc * self.np * (self.ring / CMP_BLOCK + CAD_MAX_TIERS as u32)
  }
  /// the classify stages' rows (`dust_cad_rows0`: `CADR_WORDS` words per ring index,
  /// rewritten by PREP / PREP_DEBUG each classify; never cleared, never read back)
  pub fn rows0(&self) -> u32 {
    self.incoming0() + self.ring / 4
  }
  pub fn words(&self) -> u32 {
    self.rows0() + self.ring * CADR_WORDS
  }
}

/// One tier of the frame, as the cadence needs it before its propagate (host data only; mirror of
/// [`crate::scene::dust::cadence::TierMeta`] with the GPU's buffers named by system and ring base)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CadenceTierInput {
  /// stable over the tier's life ([`crate::scene::dust::cadence::TierMeta::key`])
  pub key: u64,
  pub system: u64,
  pub ring_base: u32,
  pub first_slot: u32,
  pub capacity: u32,
  pub live_count: u32,
  pub emitted: u64,
  pub rewritten_from: u32,
  /// the tier's render anchor (heliocentric m; `DustDrawState::anchor_m`): the stale slots' shift
  pub anchor_m: [f64; 3],
}

/// What [`Device::cmd_dust_cadence_begin`] decided (the CPU's `CadencePlan` without the
/// GPU-decided parts: the aborts and the propagate set are read back or downloaded)
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CadenceGpuPlan {
  pub mode: Option<CadenceMode>,
  pub reset: Option<CadenceReset>,
  pub swaps: alloc::vec::Vec<(usize, usize)>,
  /// per tier key: the chunk's draw-list bound per (k, p) (`step_upper[k][p]`)
  pub step_upper: alloc::collections::BTreeMap<u64, [[u32; 2]; DUST_CADENCE_MAX_CLASSES]>,
  /// the pairs this frame's build draws are for (their frozen views and units)
  pub builds: alloc::vec::Vec<CadenceBuild>,
  /// the class images' geometry (the build draws' `mvp_image`)
  pub geom: Option<ImageGeom>,
}

/// One tier's GPU cadence data (a test download)
#[derive(Debug, Clone, Default)]
pub struct CadenceTierSnapshot {
  pub key: u64,
  /// the tier's slot in the state (`cads_tp`)
  pub t: usize,
  /// the owner bytes of the tier's sub-ring, slot indexed
  pub owner: alloc::vec::Vec<u8>,
  /// the incoming bytes (the handover), slot indexed
  pub incoming: alloc::vec::Vec<u8>,
  /// `lists[k][p][s]` (s: 0 shown, 1 build, 2 pending), their lengths from the state
  pub lists: alloc::vec::Vec<alloc::vec::Vec<[alloc::vec::Vec<u32>; 3]>>,
  /// the slots of the propagate bitmask, ascending
  pub propagate: alloc::vec::Vec<u32>,
}

/// The view's GPU cadence data (a test download)
#[derive(Debug, Clone, Default)]
pub struct CadenceGpuSnapshot {
  pub state: alloc::vec::Vec<u32>,
  pub tiers: alloc::vec::Vec<CadenceTierSnapshot>,
}

// ─── The host state ─────────────────────────────────────────────────────────

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

#[derive(Debug, Clone, Copy, PartialEq)]
struct TierHost {
  key: u64,
  system: u64,
  ring_base: u32,
  /// the tier's slot in the state and params (0..CAD_MAX_TIERS), assigned at the reset
  t: usize,
  first_slot: u32,
  live_count: u32,
  capacity: u32,
  emitted: u64,
  anchor_m: [f64; 3],
}

/// one (class, phase) pair: the frozen views (kept at every swap: the superset rule, Q4), the
/// units, the generation, the lists' Δe at their classify, the chunk bounds
#[derive(Debug, Clone, Default)]
struct PairHost {
  shown_view: Option<CadenceView>,
  build_view: Option<CadenceView>,
  shown_unit: f32,
  build_unit: f32,
  generation: u32,
  shown_eye_helio: [f64; 3],
  build_eye_helio: [f64; 3],
  shown_nucleus: [f64; 3],
  build_nucleus: [f64; 3],
  /// Δe at the classify of the build list `[0]` and of the pending list `[1]`
  de_c: [[f64; 3]; 2],
  /// per tier key: the build cycle's draw-list bound
  step_upper: alloc::collections::BTreeMap<u64, u32>,
  /// per tier key: the newest classified pending length read back
  known_pending: alloc::collections::BTreeMap<u64, u32>,
  /// which of the pair's two images is the build one (the other is shown); flipped at each swap
  role: u8,
}

impl PairHost {
  /// the build view at the current nucleus (`ClassImage::build_at`)
  fn build_at(&self, nucleus_m: [f64; 3]) -> Option<CadenceView> {
    let fv = self.build_view.as_ref()?;
    Some(if nucleus_m == self.build_nucleus { CadenceView { eye_m: self.build_eye_helio, ..*fv } } else { frozen_at(fv, nucleus_m) })
  }
  /// the shown view at the current nucleus (`ClassImage::shown_at`)
  fn shown_at(&self, nucleus_m: [f64; 3]) -> Option<CadenceView> {
    let fv = self.shown_view.as_ref()?;
    Some(if nucleus_m == self.shown_nucleus { CadenceView { eye_m: self.shown_eye_helio, ..*fv } } else { frozen_at(fv, nucleus_m) })
  }
}

struct SysBuf {
  buf: DustBuffer,
  layout: CadenceSysLayout,
}

/// what one readback slot's frame classified: (tier key, slot t, k, p)
#[derive(Debug, Clone, Default)]
struct SlotMeta {
  classified: alloc::vec::Vec<(u64, usize, usize, usize)>,
  /// the frame's mode, reset, swaps, admissions and live clusters (the telemetry)
  info: Option<SlotInfo>,
}

#[derive(Debug, Clone)]
struct FrameRec {
  mode: CadenceMode,
  swaps: alloc::vec::Vec<(usize, usize)>,
  nucleus: [f64; 3],
  /// the unit this frame is drawn with
  unit_now: f32,
}

/// The cadence of one view on the GPU (host half). Created by [`Device::dust_cadence_enable`].
pub struct DustCadenceGpu {
  pub cfg: CadenceConfig,
  /// the rate rule's shape bound θ (params word [`CADP_SHAPE`]; `DustCadenceState::shape_frac`):
  /// 0 off the budget loop's fallback
  pub shape_frac: f32,
  /// drawn frames since the last reset
  pub drawn_frame: u64,
  prev: Option<PrevFrame>,
  de_hist: alloc::collections::VecDeque<[f64; 3]>,
  tiers: alloc::collections::BTreeMap<u64, TierHost>,
  pairs: alloc::vec::Vec<PairHost>,
  systems: alloc::collections::BTreeMap<u64, SysBuf>,
  state: DustBuffer,
  params: DustBuffer,
  readback: DustBuffer,
  readback_mapped: *mut u8,
  copies: u64,
  slot_meta: [Option<SlotMeta>; DUST_READBACK_SLOTS],
  frame: Option<FrameRec>,
  advance_next: bool,
  content: bool,
  /// the newest state read back (telemetry, the report)
  pub last_state: Option<alloc::vec::Vec<u32>>,
  /// the class images' geometry (the config's margin; recomputed on a reset or a size change)
  pub geom: Option<ImageGeom>,
  /// the last [`DUST_CADENCE_HISTORY`] per-axis viewport corner motions (px per frame)
  motion_hist: alloc::collections::VecDeque<[f32; 2]>,
  /// per class: admitted by the motion budget this frame
  pub admitted: alloc::vec::Vec<bool>,
  /// per tier key: `(stamp, anchor)` of the last [`DUST_CADENCE_ANCHOR_HISTORY`] propagates
  anchor_hist: alloc::collections::BTreeMap<u64, alloc::collections::VecDeque<(u32, [f64; 3])>>,
  /// the anchor tables the classify reads (`CAD_ANCH_TIER_WORDS` per tier slot)
  anchors: DustBuffer,
  /// the reprojection table (`RPT_WORDS`)
  rpt: DustBuffer,
  /// a frame recorded its end, its readback copy is the frame end's
  copy_pending: bool,
  /// the newest readback's frame info (the telemetry)
  last_info: Option<SlotInfo>,
}

// SAFETY: `readback_mapped` points into a persistently mapped VMA allocation owned by this struct
unsafe impl Send for DustCadenceGpu {}
unsafe impl Sync for DustCadenceGpu {}

fn sqrt32(x: f32) -> f32 {
  <f32 as aethervk_oshal_rlib::math::FloatLike>::sqrt(x)
}

/// `(built, shown)` per class: the list entries a frame drew into its build images (per tier slot
/// and pair, `min(step, build length − cursor)`, the cursor before the frame's advance) and the
/// entries of the shown lists, from a state readback (`CADS_TP*` words). The feedback's full-frame
/// estimate scales each class's builds' counters by its `shown / built` (`dust::full_frame_estimate`).
pub fn cads_work_members(
  st: &[u32],
  nt: usize,
  nc: usize,
  np: usize,
) -> ([u32; DUST_CADENCE_MAX_CLASSES], [u32; DUST_CADENCE_MAX_CLASSES]) {
  let (mut built, mut shown) = ([0u32; DUST_CADENCE_MAX_CLASSES], [0u32; DUST_CADENCE_MAX_CLASSES]);
  for t in 0..nt {
    for k in 0..nc.min(DUST_CADENCE_MAX_CLASSES) {
      for p in 0..np {
        let tp = cads_tp(t, k, p);
        if tp + 4 >= st.len() {
          continue;
        }
        let (len, cursor, step) = (st[tp + 1], st[tp + 3], st[tp + 4]);
        built[k] = built[k].saturating_add(step.min(len.saturating_sub(cursor)));
        shown[k] = shown[k].saturating_add(st[tp]);
      }
    }
  }
  (built, shown)
}

impl DustCadenceGpu {
  /// [`cads_work_members`] of the newest state readback (`None` before one)
  pub fn work_members(&self) -> Option<([u32; DUST_CADENCE_MAX_CLASSES], [u32; DUST_CADENCE_MAX_CLASSES])> {
    let st = self.last_state.as_ref()?;
    let nt = self.tiers.values().map(|t| t.t + 1).max().unwrap_or(0);
    Some(cads_work_members(st, nt, self.nc(), self.np()))
  }

  fn nc(&self) -> usize {
    self.cfg.classes.len().min(DUST_CADENCE_MAX_CLASSES)
  }
  fn np(&self) -> usize {
    self.cfg.phases.clamp(1, 2) as usize
  }
  /// `DustCadenceState::offset_of` (aligned under the handover)
  fn offset_of(&self, k: usize, p: usize) -> u64 {
    if self.np() != 2 {
      0
    } else if self.cfg.handover_on() {
      (p as u64) * (self.cfg.classes.iter().copied().min().unwrap_or(2) as u64 / 2)
    } else {
      (p as u64) * (self.cfg.classes[k] as u64 / 2)
    }
  }
  /// whether (k, p)'s cycle starts at this drawn frame (`DustCadenceState::starts`)
  fn starts(&self, k: usize, p: usize) -> bool {
    let n = self.cfg.classes[k] as u64;
    (self.drawn_frame % n + n - self.offset_of(k, p) % n) % n == 0
  }
  /// `DustCadenceState::window_terms`
  fn window_terms(&self) -> [f32; DUST_CADENCE_MAX_CLASSES] {
    let mut out = [0.0f32; DUST_CADENCE_MAX_CLASSES];
    let Some(&cur) = self.de_hist.back() else { return out };
    for (k, &n) in self.cfg.classes.iter().take(DUST_CADENCE_MAX_CLASSES).enumerate() {
      let mut a = 0.0f64;
      for d in self.de_hist.iter().rev().take(2 * n as usize) {
        let e = [d[0] - cur[0], d[1] - cur[1], d[2] - cur[2]];
        a = a.max((e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt());
      }
      // fewer than 2N frames of history: the deviation over the span L − 1 extrapolated
      // to 2N − 1; one frame: unknown, not admitted
      let span = self.de_hist.len().min(2 * n as usize);
      if span < 2 * n as usize {
        a = if span <= 1 { f64::INFINITY } else { a * (2 * n as usize - 1) as f64 / (span - 1) as f64 };
      }
      out[k] = a as f32;
    }
    out
  }
  /// drawn frames until (k, p)'s next swap (1..=N; `DustCadenceState::frames_to_swap`)
  fn frames_to_swap(&self, k: usize, p: usize) -> u32 {
    let n = self.cfg.classes[k] as u64;
    let r = (self.drawn_frame % n + n - self.offset_of(k, p) % n) % n;
    if r == 0 { n as u32 } else { (n - r) as u32 }
  }
  /// `U(N_k)` (`DustCadenceState::motion_window`)
  fn motion_window(&self, k: usize) -> [f32; 2] {
    let n = self.cfg.classes[k] as usize;
    let mut u = [0.0f32; 2];
    for d in self.motion_hist.iter().rev().take(2 * n) {
      u[0] = u[0].max(d[0]);
      u[1] = u[1].max(d[1]);
    }
    u
  }
  /// `m − G` per axis, ∞ without a margin (`DustCadenceState::motion_budget`)
  fn motion_budget(&self) -> [f32; 2] {
    match self.geom.as_ref() {
      Some(g) if g.has_margin() => [
        g.margin[0] as f32 - DUST_CADENCE_MARGIN_GUARD,
        g.margin[1] as f32 - DUST_CADENCE_MARGIN_GUARD,
      ],
      _ => [f32::INFINITY; 2],
    }
  }
  /// `DustCadenceState::class_admitted`
  fn class_admitted(&self, k: usize) -> bool {
    let b = self.motion_budget();
    if !b[0].is_finite() {
      return true;
    }
    let n = self.cfg.classes[k] as f32;
    let u = self.motion_window(k);
    2.0 * n * u[0] <= b[0] && 2.0 * n * u[1] <= b[1]
  }
  fn push_eye_step(&mut self, de: [f64; 3]) {
    if self.de_hist.len() == DUST_CADENCE_HISTORY {
      self.de_hist.pop_front();
    }
    self.de_hist.push_back(de);
  }
  /// `DustCadenceState::clear_all` (host half; the caller fills the GPU's)
  fn clear_host(&mut self) {
    for p in self.pairs.iter_mut() {
      let g = p.generation;
      *p = PairHost { generation: g, ..PairHost::default() };
    }
    self.drawn_frame = 0;
    self.content = false;
  }
  /// `DustCadenceState::reset_reason`, with the frozen views of every pair kept since its swap
  /// (empty images included: the superset rule both mirrors use)
  fn reset_reason(&self, f: &FrameMeta, tiers: &[CadenceTierInput]) -> Option<CadenceReset> {
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
      for pr in &self.pairs {
        for fv in [pr.shown_at(f.nucleus_m), pr.build_at(f.nucleus_m)].into_iter().flatten() {
          let Some(h) = v.map_from(&fv) else { return Some(CadenceReset::OrthoDirection) };
          let det = (h[0] * h[4] - h[1] * h[3]).abs();
          let z = sqrt32(det);
          if !(z <= DUST_CADENCE_ORTHO_ZOOM_MAX && z >= 1.0 / DUST_CADENCE_ORTHO_ZOOM_MAX) {
            return Some(CadenceReset::OrthoZoom);
          }
        }
      }
    }
    if f.speed.to_bits() != prev.speed.to_bits() {
      return Some(CadenceReset::Speed);
    }
    // the time reset (`DustCadenceState::reset_reason`): a negative step, a positive one after
    // a non-positive one, or one beyond the jump factor × the previous step
    let dt = f.t_now_s - prev.t;
    if dt < 0.0 {
      return Some(CadenceReset::Time);
    }
    if let Some(q) = prev.dt
      && dt > 0.0
      && (q <= 0.0 || dt > crate::scene::dust::cadence::DUST_CADENCE_TIME_JUMP_FACTOR * q)
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
      let a = t.first_slot.wrapping_sub(old.first_slot) & old.capacity.wrapping_sub(1);
      let appended = t.live_count as i64 + a as i64 - old.live_count as i64;
      if a <= old.live_count && appended < 0 {
        return Some(CadenceReset::Range(t.key));
      }
      if appended == 0 && t.emitted != old.emitted && t.rewritten_from >= t.live_count {
        return Some(CadenceReset::Reemit(t.key));
      }
    }
    None
  }
  /// `DustCadenceState::track_ranges`: per tier the owner ranges `(first slot, count)` to clear
  fn track_ranges(&mut self, tiers: &[CadenceTierInput]) -> alloc::vec::Vec<(u64, [(u32, u32); 3])> {
    let mut out = alloc::vec::Vec::new();
    for t in tiers {
      let old = self.tiers.get_mut(&t.key).expect("tier");
      let mask = old.capacity.wrapping_sub(1);
      let cap = old.capacity;
      let a = t.first_slot.wrapping_sub(old.first_slot) & mask;
      let appended = (t.live_count as i64 + a as i64 - old.live_count as i64).max(0) as u64;
      let mut r = [(0u32, 0u32); 3];
      if a > old.live_count || appended >= cap as u64 {
        r[0] = (0, cap);
      } else {
        r[0] = (old.first_slot, a);
        r[1] = (old.first_slot.wrapping_add(old.live_count) & mask, appended as u32);
        let from = t.rewritten_from.min(t.live_count);
        r[2] = (t.first_slot.wrapping_add(from) & mask, t.live_count - from);
      }
      old.first_slot = t.first_slot;
      old.live_count = t.live_count;
      old.emitted = t.emitted;
      out.push((t.key, r));
    }
    out
  }
}

impl Device {
  /// Turns `pe`'s GPU cadence on (`Some`, the configuration) or off (`None`); the next frame resets.
  pub fn dust_cadence_enable(&self, pe: gpu::PresentationEngineHandle, cfg: Option<CadenceConfig>) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let allocator = res.allocator.allocator.as_allocator_view();
    if let Some((_, old)) = mgr.cadence.remove(&pe) {
      let timeline = res.timeline_manager.get_next_submit_value();
      for b in old.into_inner().into_buffers() {
        res.discard_pool.discard_buffer(allocator.as_allocator_view(), b.buffer, b.alloc, timeline);
      }
    }
    let Some(cfg) = cfg else { return Ok(()) };
    if crate::gpu_backends::vulkan::physics::is_cpu_particles_mode() {
      return Ok(());
    }
    let (state, _) = self.dust_create_buffer(
      &allocator,
      CADS_WORDS as u64 * 4,
      DustMem::Device,
      false,
      vk::BufferUsageFlags::TRANSFER_DST,
      &alloc::format!("DustCadenceState_{}", pe.0),
    )?;
    let (params, _) = self.dust_create_buffer(
      &allocator,
      CADP_WORDS as u64 * 4,
      DustMem::Device,
      false,
      vk::BufferUsageFlags::TRANSFER_DST,
      &alloc::format!("DustCadenceParams_{}", pe.0),
    )?;
    let (readback, readback_mapped) = self.dust_create_buffer(
      &allocator,
      (DUST_READBACK_SLOTS * CADS_WORDS) as u64 * 4,
      DustMem::HostRead,
      false,
      vk::BufferUsageFlags::TRANSFER_DST,
      &alloc::format!("DustCadenceReadback_{}", pe.0),
    )?;
    let (anchors, _) = self.dust_create_buffer(
      &allocator,
      (CAD_MAX_TIERS * CAD_ANCH_TIER_WORDS) as u64 * 4,
      DustMem::Device,
      false,
      vk::BufferUsageFlags::TRANSFER_DST,
      &alloc::format!("DustCadenceAnchors_{}", pe.0),
    )?;
    let (rpt, _) = self.dust_create_buffer(
      &allocator,
      RPT_WORDS as u64 * 4,
      DustMem::Device,
      false,
      vk::BufferUsageFlags::TRANSFER_DST,
      &alloc::format!("DustCadenceReproject_{}", pe.0),
    )?;
    let n = cfg.classes.len() * cfg.phases as usize;
    mgr.cadence.insert(
      pe,
      spin::Mutex::new(DustCadenceGpu {
        cfg,
        shape_frac: 0.0,
        drawn_frame: 0,
        prev: None,
        de_hist: alloc::collections::VecDeque::with_capacity(DUST_CADENCE_HISTORY),
        tiers: alloc::collections::BTreeMap::new(),
        pairs: alloc::vec![PairHost::default(); n],
        systems: alloc::collections::BTreeMap::new(),
        state,
        params,
        readback,
        readback_mapped,
        copies: 0,
        slot_meta: [const { None }; DUST_READBACK_SLOTS],
        frame: None,
        advance_next: false,
        content: false,
        last_state: None,
        geom: None,
        motion_hist: alloc::collections::VecDeque::with_capacity(DUST_CADENCE_HISTORY),
        admitted: alloc::vec::Vec::new(),
        anchor_hist: alloc::collections::BTreeMap::new(),
        anchors,
        rpt,
        copy_pending: false,
        last_info: None,
      }),
    );
    Ok(())
  }

  /// Whether `pe` runs the GPU cadence
  pub fn dust_cadence_active(&self, pe: gpu::PresentationEngineHandle) -> bool {
    let res = self.res.read();
    res.dust_manager.as_ref().is_some_and(|m| m.cadence.contains_key(&pe))
  }

  fn cad_classify_dispatch(&self, cmd: vk::CommandBuffer, pc: &DustClassifyPushConstants, threads: u32) {
    if threads == 0 {
      return;
    }
    let pipeline = self.kernels.pipelines.dust_classify;
    self.kernels.pipelines.assert_pc_size(pipeline, core::mem::size_of_val(pc));
    unsafe {
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
      self.device.cmd_push_constants(
        cmd,
        self.kernels.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(pc),
      );
      self.device.cmd_dispatch(cmd, threads.div_ceil(64), 1, 1);
    }
    self.cmd_dust_pre_splat_barrier(cmd);
  }

  fn cad_compact_dispatch(&self, cmd: vk::CommandBuffer, pc: &DustCompactPushConstants, groups: u32) {
    if groups == 0 {
      return;
    }
    let pipeline = self.kernels.pipelines.dust_compact;
    self.kernels.pipelines.assert_pc_size(pipeline, core::mem::size_of_val(pc));
    unsafe {
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
      self.device.cmd_push_constants(
        cmd,
        self.kernels.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(pc),
      );
      self.device.cmd_dispatch(cmd, groups, 1, 1);
    }
    self.cmd_dust_pre_splat_barrier(cmd);
  }

  /// The three compaction passes of one tier over list set `set` (0: the swapped pairs' shown and
  /// build lists; 1: their pending lists)
  fn cad_compact_tier(&self, cmd: vk::CommandBuffer, sys: u64, params: u64, t: &TierHost, set: u32) {
    let pc = |mode: u32| DustCompactPushConstants {
      sys,
      params,
      first_slot: t.first_slot,
      live_count: t.live_count,
      ring_mask: t.capacity.wrapping_sub(1),
      tier_mode: t.t as u32 | mode << 8 | set << 16,
    };
    let blocks = t.live_count.div_ceil(CMP_BLOCK);
    if blocks == 0 {
      // empty lists: the prefix still writes their lengths and the finalize
      self.cad_compact_dispatch(cmd, &pc(CMPM_PREFIX), 1);
      return;
    }
    self.cad_compact_dispatch(cmd, &pc(CMPM_COUNT), blocks);
    self.cad_compact_dispatch(cmd, &pc(CMPM_PREFIX), 1);
    self.cad_compact_dispatch(cmd, &pc(CMPM_SCATTER), blocks);
  }

  /// The render / moments addresses of a tier (the propagate's)
  fn cad_tier_buffers(&self, mgr: &DustManager, system: u64, ring_base: u32) -> GpuResult<(u64, u64)> {
    let sys = mgr.systems.get(&system).ok_or(gpu_err!("dust system {} not found", system))?;
    let (Some(render), Some(moments)) = (sys.render.as_ref(), sys.moments.as_ref()) else {
      return Err(gpu_err!("dust system {} has no GPU buffers", system));
    };
    Ok((
      render.address + ring_base as u64 * core::mem::size_of::<DustRenderCluster>() as u64,
      moments.address + ring_base as u64 * core::mem::size_of::<DustMoments>() as u64,
    ))
  }

  /// The frame's begin (the CPU's `begin`, F0–F2a, and the propagate marks): see the module doc.
  /// `tiers` in the frame's order; `unit_now` is in `f`; `d_max` the frame's speck shell radius
  /// (`DustViewParams::d_max_m`, 0 = none). A reused frame (`f.reused`) records nothing.
  pub fn cmd_dust_cadence_begin(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
    f: &FrameMeta,
    tiers: &[CadenceTierInput],
    d_max: f32,
  ) -> GpuResult<CadenceGpuPlan> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let allocator = res.allocator.allocator.as_allocator_view();
    let entry = mgr.cadence.get(&pe).ok_or(gpu_err!("dust cadence of PE {} absent", pe.0))?;
    let mut c = entry.lock();
    // a frame whose end was recorded without a frame end (no readback copy): its slot's meta goes
    if c.copy_pending {
      let s = (c.copies % DUST_READBACK_SLOTS as u64) as usize;
      c.slot_meta[s] = None;
      c.copy_pending = false;
    }
    // the readback of DUST_READBACK_SLOTS copies ago: the classified pending lengths
    let slot = (c.copies % DUST_READBACK_SLOTS as u64) as usize;
    if let Some(meta) = c.slot_meta[slot].take() {
      allocator.invalidate_allocation(&c.readback.alloc, (slot * CADS_WORDS * 4) as u64, (CADS_WORDS * 4) as u64)?;
      let words: alloc::vec::Vec<u32> = unsafe {
        core::slice::from_raw_parts(c.readback_mapped.add(slot * CADS_WORDS * 4).cast::<u32>(), CADS_WORDS)
      }
      .to_vec();
      for &(key, t, k, p) in &meta.classified {
        let n = words[cads_tp(t, k, p) + 2];
        let np = c.np();
        c.pairs[k * np + p].known_pending.insert(key, n);
      }
      c.last_state = Some(words);
      c.last_info = meta.info.clone();
    }
    if f.reused {
      c.frame = Some(FrameRec { mode: CadenceMode::Paused, swaps: alloc::vec::Vec::new(), nucleus: f.nucleus_m, unit_now: f.unit_now });
      return Ok(CadenceGpuPlan { mode: Some(CadenceMode::Paused), ..Default::default() });
    }
    if tiers.len() > CAD_MAX_TIERS {
      return Err(gpu_err!("dust cadence: {} tiers (at most {})", tiers.len(), CAD_MAX_TIERS));
    }
    let (nc, np) = (c.nc(), c.np());
    let off = f.flags & (crate::scene::dust::DUST_VIEW_FLOW | crate::scene::dust::DUST_VIEW_TRACERS) != 0;
    let first = c.prev.is_none();
    let mut reset = if first { Some(CadenceReset::First) } else { c.reset_reason(f, tiers) };
    if off && reset.is_none() && c.content {
      reset = Some(CadenceReset::Off);
    }
    let clear = reset.is_some() || off;
    // the systems' cadence buffers (a new system or a changed ring: a new buffer, all cleared)
    for t in tiers {
      let ring = mgr.systems.get(&t.system).ok_or(gpu_err!("dust system {} not found", t.system))?.capacity;
      let layout = CadenceSysLayout { ring, nc: nc as u32, np: np as u32 };
      if c.systems.get(&t.system).is_some_and(|s| s.layout == layout) {
        continue;
      }
      assert!(ring % 32 == 0 && t.ring_base % 32 == 0, "dust cadence: ring {ring} / base {} not multiples of 32", t.ring_base);
      if let Some(old) = c.systems.remove(&t.system) {
        let timeline = res.timeline_manager.get_next_submit_value();
        res.discard_pool.discard_buffer(allocator.as_allocator_view(), old.buf.buffer, old.buf.alloc, timeline);
      }
      let (buf, _) = self.dust_create_buffer(
        &allocator,
        layout.words() as u64 * 4,
        DustMem::Device,
        false,
        vk::BufferUsageFlags::TRANSFER_DST,
        &alloc::format!("DustCadenceSys_{}_{}", pe.0, t.system),
      )?;
      unsafe { self.device.cmd_fill_buffer(cmd, buf.buffer, 0, vk::WHOLE_SIZE, 0) };
      c.systems.insert(t.system, SysBuf { buf, layout });
    }
    if clear {
      c.clear_host();
    }
    let clears = if clear {
      c.tiers.clear();
      for (i, t) in tiers.iter().enumerate() {
        c.tiers.insert(
          t.key,
          TierHost {
            key: t.key,
            system: t.system,
            ring_base: t.ring_base,
            t: i,
            first_slot: t.first_slot,
            live_count: t.live_count,
            capacity: t.capacity,
            emitted: t.emitted,
            anchor_m: t.anchor_m,
          },
        );
      }
      alloc::vec::Vec::new()
    } else {
      c.track_ranges(tiers)
    };
    // the eye steps
    let eye_rel = [f.view.eye_m[0] - f.nucleus_m[0], f.view.eye_m[1] - f.nucleus_m[1], f.view.eye_m[2] - f.nucleus_m[2]];
    let (dt, d_eye, de) = match &c.prev {
      Some(p) => (
        f.t_now_s - p.t,
        [f.view.eye_m[0] - p.eye_m[0], f.view.eye_m[1] - p.eye_m[1], f.view.eye_m[2] - p.eye_m[2]],
        [eye_rel[0] - p.eye_rel[0], eye_rel[1] - p.eye_rel[1], eye_rel[2] - p.eye_rel[2]],
      ),
      None => (0.0, [0.0; 3], [0.0; 3]),
    };
    if !first {
      c.push_eye_step(de);
    }
    // the class images' geometry
    if c.geom.as_ref().is_none_or(|g| {
      g.layout.width != f.view.width + 2 * g.margin[0] || g.layout.height != f.view.height + 2 * g.margin[1] || reset.is_some()
    }) {
      c.geom = Some(c.cfg.image_geom(f.view.width, f.view.height));
    }
    // the motion budget: this frame's corner motion against the previous view, the admit flags
    let motion = if first {
      None
    } else {
      c.prev
        .as_ref()
        .map(|pv| f.view.map_from(&pv.view).map_or([f32::INFINITY; 2], |m| corner_motion(&m, f.view.width, f.view.height)))
    };
    if let Some(u) = motion {
      if c.motion_hist.len() == DUST_CADENCE_HISTORY {
        c.motion_hist.pop_front();
      }
      c.motion_hist.push_back(u);
    }
    let admitted: alloc::vec::Vec<bool> = (0..c.nc()).map(|k| c.class_admitted(k)).collect();
    c.admitted = admitted;
    // `AETHERVK_DUST_CADENCE_LOG=1`: the motion budget's inputs per frame (diagnostics)
    if aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_CADENCE_LOG").is_some_and(|v| v.trim() == "1") {
      aethervk_oshal_rlib::log!(
        "[Dust cadence] motion: last {:?}, budget {:?}, windows {:?}, admitted {:?}, margin {:?}, eye step {:?} m, dt {} s",
        c.motion_hist.back(),
        c.motion_budget(),
        (0..c.nc()).map(|k| c.motion_window(k)).collect::<alloc::vec::Vec<_>>(),
        c.admitted,
        c.geom.as_ref().map(|g| g.margin),
        de,
        dt
      );
    }
    c.prev = Some(PrevFrame {
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
    let run = mode == CadenceMode::Run;
    // the params
    let mut prm = alloc::vec![0u32; CADP_WORDS];
    let fbits = |x: f32| x.to_bits();
    prm[CADP_DT] = fbits(dt as f32);
    for i in 0..3 {
      prm[CADP_DEYE + i] = fbits(d_eye[i] as f32);
    }
    let aw = c.window_terms();
    for k in 0..DUST_CADENCE_MAX_CLASSES {
      prm[CADP_AW + k] = fbits(aw[k]);
    }
    prm[CADP_EPS] = fbits(c.cfg.eps_px);
    prm[CADP_SHAPE] = fbits(crate::scene::dust::cadence::admission_shape(c.shape_frac, c.cfg.light_theta()));
    prm[CADP_FRAME_PX] = fbits(f.view.width.max(f.view.height) as f32);
    prm[CADP_W] = fbits(f.view.width as f32);
    prm[CADP_H] = fbits(f.view.height as f32);
    prm[CADP_NC] = nc as u32;
    prm[CADP_NP] = np as u32;
    for k in 0..nc {
      prm[CADP_N + k] = c.cfg.classes[k];
    }
    prm[CADP_DMAX] = fbits(d_max);
    prm[CADP_STATE] = c.state.address as u32;
    prm[CADP_STATE + 1] = (c.state.address >> 32) as u32;
    prm[CADP_NT] = tiers.len() as u32;
    // the abort inputs, before this frame's swaps (the CPU's F1 reads the lists' Δe_c as they are)
    for k in 0..nc {
      let n = c.cfg.classes[k] as f32;
      prm[CADP_BOUND + k] = fbits((DUST_CADENCE_ABORT_FACTOR - 1.0) * c.cfg.eps_px / (2.0 * n));
      for p in 0..np {
        for list in 0..2 {
          let dc = c.pairs[k * np + p].de_c[list];
          let dd = [de[0] - dc[0], de[1] - dc[1], de[2] - dc[2]];
          prm[CADP_DD + (k * 2 + p) * 2 + list] = fbits((dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]).sqrt() as f32);
        }
      }
    }
    let mut flags = 0u32;
    if run && c.advance_next {
      flags |= CADF_ADVANCE;
    }
    if run && reset.is_none() {
      flags |= CADF_ABORTS;
    }
    // the host's abort reasons (F1, when the rate does not fire): the class over its motion
    // budget; a build whose displacement at the end of its display is predicted beyond it
    if run && reset.is_none() {
      let budget = c.motion_budget();
      for k in 0..nc {
        for p in 0..np {
          let predicted = budget[0].is_finite() && {
            match c.pairs[k * np + p].build_at(f.nucleus_m).and_then(|fv| f.view.map_from(&fv)) {
              Some(mp) => {
                let d = corner_motion(&mp, f.view.width, f.view.height);
                let u = c.motion_window(k);
                let rest = (c.frames_to_swap(k, p) + c.cfg.classes[k]) as f32;
                d[0] + rest * u[0] > budget[0] || d[1] + rest * u[1] > budget[1]
              }
              None => true,
            }
          };
          for list in 0..2 {
            let r = if !c.admitted[k] {
              ABORT_BUDGET as u32
            } else if list == 0 && predicted {
              ABORT_PREDICTION as u32
            } else {
              0
            };
            prm[CADP_FORCE + (k * 2 + p) * 2 + list] = r;
          }
        }
      }
      // the coverage of the shown images (with a margin)
      if let Some(geom) = c.geom.clone()
        && geom.has_margin()
      {
        flags |= CADF_COVERAGE;
        let mut cover = 0u32;
        for k in 0..nc {
          for p in 0..np {
            let covered = c.pairs[k * np + p]
              .shown_at(f.nucleus_m)
              .and_then(|sv| f.view.map_to_image(&sv, &geom))
              .is_some_and(|mp| image_covers(&mp, f.view.width, f.view.height, &geom.layout));
            if !covered {
              cover |= 1 << (k * 2 + p);
            }
          }
        }
        prm[CADP_COVER] = cover;
      }
    }
    if c.cfg.retain {
      flags |= CADF_RETAIN;
    }
    if c.cfg.handover_on() {
      flags |= CADF_HANDOVER;
    }
    for k in 0..nc {
      for p in 0..np {
        prm[CADP_FTS + k * 2 + p] = c.frames_to_swap(k, p);
      }
    }
    prm[CADP_FLAGS] = flags;
    prm[CADP_ADMIT] = c.admitted.iter().enumerate().map(|(k, &a)| (a as u32) << k).sum();
    prm[CADP_FOOT] = c.geom.as_ref().is_none_or(|g| !g.has_margin()) as u32;
    prm[CADP_ANCH] = c.anchors.address as u32;
    prm[CADP_ANCH + 1] = (c.anchors.address >> 32) as u32;
    // F2a's host half: the swaps (the CPU's k, then p order), the frozen views, the bounds
    let mut swaps = alloc::vec::Vec::new();
    // the pairs whose new build image is cleared (k, p, its role)
    let mut refills: alloc::vec::Vec<(usize, usize, u8)> = alloc::vec::Vec::new();
    if run {
      for k in 0..nc {
        let n = c.cfg.classes[k];
        for p in 0..np {
          if !c.starts(k, p) {
            continue;
          }
          let tier_list: alloc::vec::Vec<(u64, u32)> = c.tiers.values().map(|t| (t.key, t.capacity)).collect();
          let pr = &mut c.pairs[k * np + p];
          pr.shown_view = pr.build_view.take();
          pr.shown_eye_helio = pr.build_eye_helio;
          pr.shown_nucleus = pr.build_nucleus;
          pr.shown_unit = pr.build_unit;
          pr.generation = pr.generation.wrapping_add(1);
          pr.build_view = Some(CadenceView { eye_m: eye_rel, ..f.view });
          pr.build_eye_helio = f.view.eye_m;
          pr.build_nucleus = f.nucleus_m;
          pr.build_unit = f.unit_now;
          // the old build image is the shown one now; the other becomes the build
          pr.role ^= 1;
          refills.push((k, p, pr.role));
          pr.de_c[0] = pr.de_c[1];
          // the classify of this frame (finish) takes the pending list's Δe
          pr.de_c[1] = de;
          for (key, cap) in tier_list {
            let len = pr.known_pending.get(&key).copied().unwrap_or(cap / np as u32);
            pr.step_upper.insert(key, len.div_ceil(n));
          }
          pr.known_pending.clear();
          swaps.push((k, p));
        }
      }
      if !swaps.is_empty() {
        c.content = true;
      }
    }
    prm[CADP_NSWAP] = swaps.len() as u32;
    for (j, &(k, p)) in swaps.iter().enumerate() {
      prm[CADP_SWAP0 + j] = k as u32 | (p as u32) << 8;
    }
    let mut step_upper = alloc::collections::BTreeMap::new();
    for t in tiers {
      let th = c.tiers[&t.key];
      let ring = c.systems[&t.system].layout.ring;
      let tb = CADP_TIER0 + CADP_TIER_WORDS * th.t;
      prm[tb] = t.ring_base;
      prm[tb + 1] = t.capacity;
      prm[tb + 2] = t.live_count;
      prm[tb + 3] = t.first_slot;
      prm[tb + 4] = ring;
      let mut su = [[0u32; 2]; DUST_CADENCE_MAX_CLASSES];
      for k in 0..nc {
        for p in 0..np {
          let v = if run { c.pairs[k * np + p].step_upper.get(&t.key).copied().unwrap_or(0) } else { 0 };
          su[k][p] = v;
          prm[CADP_STEPU + (th.t * 3 + k) * 2 + p] = v;
        }
      }
      step_upper.insert(t.key, su);
    }
    // the build draws of this frame: the pairs with a chunk bound on some tier
    let mut builds = alloc::vec::Vec::new();
    if run {
      for k in 0..nc {
        for p in 0..np {
          if step_upper.values().any(|su| su[k][p] > 0)
            && let Some(view) = c.pairs[k * np + p].build_at(f.nucleus_m)
          {
            builds.push(CadenceBuild { k, p, view, unit: c.pairs[k * np + p].build_unit });
          }
        }
      }
    }
    for (key, r) in &clears {
      let th = c.tiers[key];
      let cb = CADP_CLEAR0 + CADP_CLEAR_WORDS * th.t;
      for (i, &(s0, n)) in r.iter().enumerate() {
        prm[cb + 2 * i] = s0;
        prm[cb + 2 * i + 1] = n;
      }
    }
    // ── the GPU work ──
    // the previous frame's passes and readback copy → this frame's fills and updates
    self.cmd_dust_pre_propagate_barrier(cmd);
    unsafe {
      if clear {
        for s in c.systems.values() {
          self.device.cmd_fill_buffer(cmd, s.buf.buffer, 0, s.layout.pmask0() as u64 * 4, 0);
          self.device.cmd_fill_buffer(cmd, s.buf.buffer, s.layout.incoming0() as u64 * 4, (s.layout.ring / 4) as u64 * 4, 0);
        }
        self.device.cmd_fill_buffer(cmd, c.state.buffer, 0, vk::WHOLE_SIZE, 0);
      }
      for s in c.systems.values() {
        self.device.cmd_fill_buffer(cmd, s.buf.buffer, s.layout.pmask0() as u64 * 4, (s.layout.ring / 32) as u64 * 4, 0);
      }
      // the frame's counters and the telemetry's words
      self.device.cmd_fill_buffer(
        cmd,
        c.state.buffer,
        CADS_FRAME_WORDS.start as u64 * 4,
        (CADS_FRAME_WORDS.end - CADS_FRAME_WORDS.start) as u64 * 4,
        0,
      );
      self.device.cmd_update_buffer(cmd, c.params.buffer, 0, bytemuck::cast_slice(&prm));
    }
    // the new build images: their levels zero, their headers (their own geometry, the unit they
    // are drawn with, the queue word relative to themselves)
    if !refills.is_empty() {
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      if view.image_words == c.image_span(&view.layout) && view.image_words > 0 {
        let pyramid = view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?;
        let geom = c.image_geom(&view.layout);
        for &(k, p, role) in &refills {
          let iw = c.image_word(&view.layout, k, p, role);
          let mut header = geom.layout.header();
          header[dust::PYR_UNIT as usize] = f.unit_now.to_bits();
          header[dust::PYR_QUEUE_BASE as usize] = view.queue_word() - iw;
          unsafe {
            self.device.cmd_update_buffer(cmd, pyramid.buffer, iw as u64 * 4, bytemuck::cast_slice(&header));
            self.device.cmd_fill_buffer(
              cmd,
              pyramid.buffer,
              (iw + dust::PYRAMID_HEADER_WORDS) as u64 * 4,
              (image_words(&geom) - dust::PYRAMID_HEADER_WORDS) as u64 * 4,
              0,
            );
          }
        }
      }
    }
    self.cmd_dust_pre_splat_barrier(cmd);
    let params = c.params.address;
    let pc_of = |t: &TierHost, sys: u64, render: u64, moments: u64, mode: u32, pair: u32| DustClassifyPushConstants {
      moments,
      render,
      sys,
      params,
      mvp: [0.0; 16],
      eye_local: [0.0; 4],
      first_slot: t.first_slot,
      live_count: t.live_count,
      ring_mask: t.capacity.wrapping_sub(1),
      tier_mode: t.t as u32 | mode << 8 | pair << 16,
    };
    // the range hook's clears (track_ranges)
    for (key, r) in &clears {
      let th = c.tiers[key];
      let n: u32 = r.iter().map(|x| x.1).sum();
      if n == 0 {
        continue;
      }
      let sys = c.systems[&th.system].buf.address;
      let (render, moments) = self.cad_tier_buffers(mgr, th.system, th.ring_base)?;
      self.cad_classify_dispatch(cmd, &pc_of(&th, sys, render, moments, CADM_CLEAR, 0), n);
    }
    for t in tiers {
      if let Some(th) = c.tiers.get_mut(&t.key) {
        th.anchor_m = t.anchor_m;
      }
    }
    let order: alloc::vec::Vec<TierHost> = tiers.iter().map(|t| c.tiers[&t.key]).collect();
    if run {
      // F1 + F2a's scalars (CONTROL), the bytes (TRANSITION), the swapped lists (compaction)
      let sys0 = c.systems.values().next().map_or(0, |s| s.buf.address);
      self.cad_compact_dispatch(
        cmd,
        &DustCompactPushConstants { sys: sys0, params, first_slot: 0, live_count: 0, ring_mask: 0, tier_mode: CMPM_CONTROL << 8 },
        1,
      );
      for th in &order {
        let sys = c.systems[&th.system].buf.address;
        let (render, moments) = self.cad_tier_buffers(mgr, th.system, th.ring_base)?;
        self.cad_classify_dispatch(cmd, &pc_of(th, sys, render, moments, CADM_TRANSITION, 0), th.live_count);
        if !swaps.is_empty() {
          self.cad_compact_tier(cmd, sys, params, th, 0);
        }
      }
    }
    // the propagate set: fresh ∪ the chunks ∪ their predecessors (pre-propagate render words)
    for th in &order {
      let sys = c.systems[&th.system].buf.address;
      let (render, moments) = self.cad_tier_buffers(mgr, th.system, th.ring_base)?;
      self.cad_classify_dispatch(cmd, &pc_of(th, sys, render, moments, CADM_MARK_FRESH, 0), th.live_count);
      if run {
        let su = &step_upper[&th.key];
        for k in 0..nc {
          for p in 0..np {
            if su[k][p] > 0 {
              let pair = k as u32 | (p as u32) << 4;
              self.cad_classify_dispatch(cmd, &pc_of(th, sys, render, moments, CADM_MARK_CHUNK, pair), su[k][p]);
            }
          }
        }
      }
    }
    {
      let slot = (c.copies % DUST_READBACK_SLOTS as u64) as usize;
      let live: u32 = tiers.iter().map(|t| t.live_count).sum();
      let admitted = c.admitted.clone();
      c.slot_meta[slot].get_or_insert_with(SlotMeta::default).info = Some(SlotInfo { mode: Some(mode), reset, swaps: swaps.clone(), admitted, live });
    }
    c.frame = Some(FrameRec { mode, swaps: swaps.clone(), nucleus: f.nucleus_m, unit_now: f.unit_now });
    Ok(CadenceGpuPlan { mode: Some(mode), reset, swaps, step_upper, builds, geom: c.geom.clone() })
  }

  /// One tier's propagate under the cadence: only the slots of the frame's propagate bitmask
  /// ([`Self::cmd_dust_cadence_begin`]), and of those only the ones not stamped with this
  /// generation (the stamp skip); the dispatch-level cache is bypassed (a cached skip would keep
  /// slots that became fresh at a swap at an older time).
  #[allow(clippy::too_many_arguments)]
  pub fn cmd_dust_cadence_propagate(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
    id: u64,
    ring_base: u32,
    capacity: u32,
    first_slot: u32,
    live_count: u32,
    frame: &DustFrame,
    emitted: u64,
  ) -> GpuResult<(u64, u64)> {
    let mask = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let entry = mgr.cadence.get(&pe).ok_or(gpu_err!("dust cadence of PE {} absent", pe.0))?;
      let c = entry.lock();
      let s = c.systems.get(&id).ok_or(gpu_err!("dust cadence: system {id} not begun"))?;
      s.buf.address + (s.layout.pmask0() + ring_base / 32) as u64 * 4
    };
    self
      .propagate(cmd, id, ring_base, capacity, first_slot, live_count, frame, Some(emitted), mask)
      .map(|(a, _)| a)
  }

  /// One tier's classify (the CPU's F2b for the pairs whose cycle started) and the compaction of
  /// their pending lists, after the propagate; `mvp` / `eye_local` the current view's (the splat's)
  #[allow(clippy::too_many_arguments)]
  pub fn cmd_dust_cadence_classify(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
    key: u64,
    render: u64,
    moments: u64,
    mvp: [f32; 16],
    eye_local: [f32; 4],
  ) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let entry = mgr.cadence.get(&pe).ok_or(gpu_err!("dust cadence of PE {} absent", pe.0))?;
    let mut c = entry.lock();
    let Some(fr) = c.frame.clone() else { return Err(gpu_err!("dust cadence: classify before begin")) };
    if fr.mode == CadenceMode::Paused {
      return Ok(());
    }
    let th = *c.tiers.get(&key).ok_or(gpu_err!("dust cadence: tier {key} not begun"))?;
    // the anchor of this frame's propagate generation (cadence::finish's start, every drawn mode),
    // then the tier's anchor table against the current anchor (`anchor_delta_f32` per entry)
    let stamp = mgr
      .systems
      .get(&th.system)
      .and_then(|s| s.propagate_gen.lock().get(&th.ring_base).map(|e| e.1))
      .unwrap_or(0);
    let table = {
      let h = c.anchor_hist.entry(key).or_default();
      if h.back().is_none_or(|&(s0, a)| s0 != stamp || a != th.anchor_m) {
        if h.len() == DUST_CADENCE_ANCHOR_HISTORY {
          h.pop_front();
        }
        h.push_back((stamp, th.anchor_m));
      }
      let mut e: alloc::vec::Vec<(u32, [f32; 3])> = h.iter().map(|&(s0, a)| (s0, anchor_delta_f32(a, th.anchor_m))).collect();
      // (the CPU finds the newest entry of a stamp; stamps grow, a duplicate keeps the newest)
      e.sort_by_key(|x| x.0);
      e.dedup_by(|b, a| {
        if a.0 == b.0 {
          *a = *b;
          true
        } else {
          false
        }
      });
      let mut w = alloc::vec![0u32; CAD_ANCH_ENTRY0 + 4 * e.len()];
      w[0] = e.len() as u32;
      w[1] = stamp;
      for (i, &(s0, d)) in e.iter().enumerate() {
        w[CAD_ANCH_ENTRY0 + 4 * i] = s0;
        for q in 0..3 {
          w[CAD_ANCH_ENTRY0 + 4 * i + 1 + q] = d[q].to_bits();
        }
      }
      w
    };
    if fr.mode != CadenceMode::Run || fr.swaps.is_empty() {
      return Ok(());
    }
    self.cmd_dust_pre_propagate_barrier(cmd);
    unsafe {
      self.device.cmd_update_buffer(cmd, c.anchors.buffer, (th.t * CAD_ANCH_TIER_WORDS) as u64 * 4, bytemuck::cast_slice(&table));
    }
    self.cmd_dust_pre_splat_barrier(cmd);
    let sys = c.systems[&th.system].buf.address;
    let params = c.params.address;
    let pc = DustClassifyPushConstants {
      moments,
      render,
      sys,
      params,
      mvp,
      eye_local,
      first_slot: th.first_slot,
      live_count: th.live_count,
      ring_mask: th.capacity.wrapping_sub(1),
      tier_mode: th.t as u32 | CADM_CLASSIFY << 8,
    };
    // the slot's evaluation into its row (the stage modes, a barrier after each), then
    // CLASSIFY reads the row
    self.cad_classify_dispatch(cmd, &cad_pc_mode(&pc, CADM_PREP), th.live_count);
    for mode in CAD_EVAL_STAGES {
      self.cad_classify_dispatch(cmd, &cad_pc_mode(&pc, mode), th.live_count);
    }
    self.cad_classify_dispatch(cmd, &pc, th.live_count);
    self.cad_compact_tier(cmd, sys, params, &th, 1);
    // the readback of this frame names its classified lists
    let slot = (c.copies % DUST_READBACK_SLOTS as u64) as usize;
    let meta = c.slot_meta[slot].get_or_insert_with(SlotMeta::default);
    for &(k, p) in &fr.swaps {
      meta.classified.push((key, th.t, k, p));
    }
    Ok(())
  }

  /// One tier's chunk pre-pass (the CPU's build-chunk step before its draws: the draw lists, the
  /// walk-mask lanes cleared, the footprint rule, IN_IMAGE), after the classify; the draws are
  /// [`Self::cmd_dust_cadence_builds`]'s
  pub fn cmd_dust_cadence_chunks(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
    key: u64,
    render: u64,
    moments: u64,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let entry = mgr.cadence.get(&pe).ok_or(gpu_err!("dust cadence of PE {} absent", pe.0))?;
    let c = entry.lock();
    let Some(fr) = c.frame.as_ref() else { return Err(gpu_err!("dust cadence: chunks before begin")) };
    if fr.mode != CadenceMode::Run {
      return Ok(());
    }
    let th = *c.tiers.get(&key).ok_or(gpu_err!("dust cadence: tier {key} not begun"))?;
    let sys = c.systems[&th.system].buf.address;
    let (nc, np) = (c.nc(), c.np());
    let nucleus = fr.nucleus;
    for k in 0..nc {
      for p in 0..np {
        let su = c.pairs[k * np + p].step_upper.get(&key).copied().unwrap_or(0);
        if su == 0 {
          continue;
        }
        // the frozen image view's mvp (the footprint rule's, without a margin)
        let mvp = match (c.pairs[k * np + p].build_at(nucleus), c.geom.as_ref()) {
          (Some(fv), Some(g)) => fv.mvp_image(th.anchor_m, g),
          (Some(fv), None) => fv.mvp(th.anchor_m),
          _ => [0.0; 16],
        };
        let pc = DustClassifyPushConstants {
          moments,
          render,
          sys,
          params: c.params.address,
          mvp,
          eye_local: [0.0; 4],
          first_slot: th.first_slot,
          live_count: th.live_count,
          ring_mask: th.capacity.wrapping_sub(1),
          tier_mode: th.t as u32 | CADM_CHUNK << 8 | (k as u32 | (p as u32) << 4) << 16,
        };
        // the footprint rule into the draw-list words first (CHUNK reads them)
        self.cad_classify_dispatch(cmd, &cad_pc_mode(&pc, CADM_CHUNK_FOOT), su);
        self.cad_classify_dispatch(cmd, &pc, su);
      }
    }
    Ok(())
  }

  /// The frame's end: the state's readback copy; the drawn frame and the deferred advance
  pub fn cmd_dust_cadence_end(&self, cmd: vk::CommandBuffer, pe: gpu::PresentationEngineHandle) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let entry = mgr.cadence.get(&pe).ok_or(gpu_err!("dust cadence of PE {} absent", pe.0))?;
    let mut c = entry.lock();
    let Some(fr) = c.frame.take() else { return Ok(()) };
    if fr.mode == CadenceMode::Paused {
      return Ok(());
    }
    // the state's readback copy is the frame end's (`cad_frame_end`: after the frame's queue, so
    // the telemetry's frame light is the finished frame's)
    c.copy_pending = true;
    // (the command buffer is the frame end's)
    let _ = cmd;
    let run = fr.mode == CadenceMode::Run;
    if run {
      c.drawn_frame += 1;
    }
    c.advance_next = run;
    Ok(())
  }

  /// Test hook: the view's GPU cadence state, owner bytes, lists and propagate set (synchronous)
  pub fn download_dust_cadence(&self, pe: gpu::PresentationEngineHandle) -> GpuResult<CadenceGpuSnapshot> {
    struct T {
      key: u64,
      t: usize,
      rb: u32,
      cap: u32,
      buffer: vk::Buffer,
      layout: CadenceSysLayout,
    }
    let (state_buf, ts, nc, np) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let entry = mgr.cadence.get(&pe).ok_or(gpu_err!("dust cadence of PE {} absent", pe.0))?;
      let c = entry.lock();
      let ts: alloc::vec::Vec<T> = c
        .tiers
        .values()
        .map(|t| {
          let s = &c.systems[&t.system];
          T { key: t.key, t: t.t, rb: t.ring_base, cap: t.capacity, buffer: s.buf.buffer, layout: s.layout }
        })
        .collect();
      (c.state.buffer, ts, c.nc(), c.np())
    };
    let state: alloc::vec::Vec<u32> = pod_vec(&self.dust_download_bytes(state_buf, 0, (CADS_WORDS * 4) as u64)?);
    let mut out = CadenceGpuSnapshot { state, tiers: alloc::vec::Vec::new() };
    for t in ts {
      // (up to the classify rows: scratch, not state)
      let all: alloc::vec::Vec<u32> = pod_vec(&self.dust_download_bytes(t.buffer, 0, t.layout.rows0() as u64 * 4)?);
      let bytes: &[u8] = bytemuck::cast_slice(&all[..(t.layout.ring / 4) as usize]);
      let owner = bytes[t.rb as usize..(t.rb + t.cap) as usize].to_vec();
      let i0 = t.layout.incoming0() as usize;
      let ibytes: &[u8] = bytemuck::cast_slice(&all[i0..i0 + (t.layout.ring / 4) as usize]);
      let incoming = ibytes[t.rb as usize..(t.rb + t.cap) as usize].to_vec();
      let mut lists = alloc::vec::Vec::new();
      for k in 0..nc {
        let mut kl = alloc::vec::Vec::new();
        for p in 0..np {
          let tp = cads_tp(t.t, k, p);
          let l: [alloc::vec::Vec<u32>; 3] = core::array::from_fn(|s| {
            let at = t.layout.list(t.rb, t.cap, k as u32, p as u32, s as u32) as usize;
            let n = (out.state[tp + s] as usize).min((t.cap / np as u32) as usize);
            all[at..at + n].to_vec()
          });
          kl.push(l);
        }
        lists.push(kl);
      }
      let pm = t.layout.pmask0() as usize;
      let propagate = (0..t.cap)
        .filter(|&s| {
          let i = (t.rb + s) as usize;
          all[pm + (i >> 5)] & (1 << (i & 31)) != 0
        })
        .collect();
      out.tiers.push(CadenceTierSnapshot { key: t.key, t: t.t, owner, incoming, lists, propagate });
    }
    Ok(out)
  }

  /// Test hook: `dust_classify.comp`'s rate of every live slot of one tier under the given inputs
  /// (the DEBUG mode: nothing else is written), `CAD_DEBUG_WORDS` words per range offset. `inp`
  /// and `classes` as [`crate::scene::dust::cadence::cadence_rate`] takes them; `d_max` the shell.
  #[allow(clippy::too_many_arguments)]
  pub fn dust_cadence_classify_debug(
    &self,
    render: u64,
    moments: u64,
    first_slot: u32,
    capacity: u32,
    live_count: u32,
    mvp: [f32; 16],
    eye_local: [f32; 4],
    width: u32,
    height: u32,
    classes: &[u32],
    inp: &crate::scene::dust::cadence::RateInputs,
    d_max: f32,
  ) -> GpuResult<alloc::vec::Vec<[u32; CAD_DEBUG_WORDS]>> {
    let (params, dbg, sys) = {
      let res = self.res.read();
      let allocator = res.allocator.allocator.as_allocator_view();
      let (params, _) = self.dust_create_buffer(&allocator, CADP_WORDS as u64 * 4, DustMem::Device, false, vk::BufferUsageFlags::TRANSFER_DST, "DustCadenceDebugParams")?;
      let (dbg, _) = self.dust_create_buffer(
        &allocator,
        (live_count.max(1) as usize * CAD_DEBUG_WORDS * 4) as u64,
        DustMem::Device,
        false,
        vk::BufferUsageFlags::TRANSFER_DST,
        "DustCadenceDebug",
      )?;
      // the stage modes' rows (a scratch system buffer: PREP_DEBUG reads no owner byte)
      let nc = classes.len().min(DUST_CADENCE_MAX_CLASSES) as u32;
      let sys_words = CadenceSysLayout { ring: capacity, nc, np: 1 }.words();
      let (sys, _) = self.dust_create_buffer(&allocator, sys_words as u64 * 4, DustMem::Device, false, vk::BufferUsageFlags::TRANSFER_DST, "DustCadenceDebugSys")?;
      (params, dbg, sys)
    };
    let mut prm = alloc::vec![0u32; CADP_WORDS];
    prm[CADP_DT] = inp.dt_s.to_bits();
    for i in 0..3 {
      prm[CADP_DEYE + i] = inp.d_eye_m[i].to_bits();
    }
    for k in 0..DUST_CADENCE_MAX_CLASSES {
      prm[CADP_AW + k] = inp.a_w_m[k].to_bits();
    }
    prm[CADP_EPS] = inp.eps_px.to_bits();
    prm[CADP_SHAPE] = inp.shape_frac.to_bits();
    prm[CADP_FRAME_PX] = inp.frame_px.to_bits();
    prm[CADP_W] = (width as f32).to_bits();
    prm[CADP_H] = (height as f32).to_bits();
    let nc = classes.len().min(DUST_CADENCE_MAX_CLASSES);
    prm[CADP_NC] = nc as u32;
    prm[CADP_NP] = 1;
    for k in 0..nc {
      prm[CADP_N + k] = classes[k];
    }
    prm[CADP_DEBUG] = dbg.address as u32;
    prm[CADP_DEBUG + 1] = (dbg.address >> 32) as u32;
    prm[CADP_DMAX] = d_max.to_bits();
    prm[CADP_TIER0 + 1] = capacity;
    prm[CADP_TIER0 + 2] = live_count;
    prm[CADP_TIER0 + 3] = first_slot;
    // the system ring R of the rows (dust_cad_rows0; the scratch's layout ring)
    prm[CADP_TIER0 + 4] = capacity;
    let pc = DustClassifyPushConstants {
      moments,
      render,
      sys: sys.address,
      params: params.address,
      mvp,
      eye_local,
      first_slot,
      live_count,
      ring_mask: capacity.wrapping_sub(1),
      tier_mode: CADM_DEBUG << 8,
    };
    let mut cmds = self.run_transient_commands(|cmd| {
      self.cmd_dust_pre_propagate_barrier(cmd);
      unsafe {
        self.device.cmd_update_buffer(cmd, params.buffer, 0, bytemuck::cast_slice(&prm));
        self.device.cmd_fill_buffer(cmd, sys.buffer, 0, vk::WHOLE_SIZE, 0);
      }
      self.cmd_dust_pre_splat_barrier(cmd);
      // the stage modes into the rows, then DEBUG reads them
      self.cad_classify_dispatch(cmd, &cad_pc_mode(&pc, CADM_PREP_DEBUG), live_count);
      for mode in CAD_EVAL_STAGES {
        self.cad_classify_dispatch(cmd, &cad_pc_mode(&pc, mode), live_count);
      }
      self.cad_classify_dispatch(cmd, &pc, live_count);
      Ok(())
    })?;
    cmds.cleanup(&self.device);
    let words: alloc::vec::Vec<u32> = pod_vec(&self.dust_download_bytes(dbg.buffer, 0, (live_count.max(1) as usize * CAD_DEBUG_WORDS * 4) as u64)?);
    let res = self.res.read();
    let allocator = res.allocator.allocator.as_allocator_view();
    for b in [params, dbg, sys] {
      let mut a = b.alloc;
      unsafe { allocator.destroy_buffer(b.buffer, &mut a) };
    }
    Ok(words.chunks_exact(CAD_DEBUG_WORDS).take(live_count as usize).map(|c| core::array::from_fn(|i| c[i])).collect())
  }
}

impl DustCadenceGpu {
  /// every buffer this cadence owns (for the destruction)
  pub(super) fn into_buffers(self) -> alloc::vec::Vec<DustBuffer> {
    let mut v = alloc::vec![self.state, self.params, self.readback, self.anchors, self.rpt];
    v.extend(self.systems.into_values().map(|s| s.buf));
    v
  }
}

// ─── The class images' geometry and the reprojection (I2) ───────────────────

/// words of one class image of `geom` (header + levels of its own layout: no measure grid, no
/// queues; `cadence::class_image_words` of the image's layout). The geometry itself is the CPU
/// mirror's (`CadenceConfig::image_geom`, `margin_px`).
pub fn image_words(geom: &ImageGeom) -> u32 {
  crate::scene::dust::cadence::class_image_words(&geom.layout)
}

pub const RPT_N: usize = 0;
pub const RPT_UNIT_NOW: usize = 1;
pub const RPT_STATE: usize = 4;
pub const RPT_ENTRY0: usize = 16;
pub const RPT_ENTRY_WORDS: usize = 16;
pub const RPT_MAX_IMAGES: usize = 2 * DUST_CADENCE_MAX_CLASSES * 2;
pub const RPT_WORDS: usize = RPT_ENTRY0 + RPT_ENTRY_WORDS * RPT_MAX_IMAGES;
/// the CPU's initial reprojection seed (`DustCadenceState::finish`, F4)
pub const RPT_SEED0: u32 = 0xC4DE_5EED;

/// One shown image as `dust_reproject.comp` reads it (the CPU's `ShownImage` with its address,
/// generation and existence)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReprojectImage {
  /// device address of the image's header (its own level table)
  pub address: u64,
  /// current viewport px → the image's px (`CadenceView::map_from` against the image's frozen
  /// projection, the margin included)
  pub map: [f32; 9],
  /// `false` when the host has no map (the CPU skips the image)
  pub map_valid: bool,
  /// the fixed-point unit the image was drawn with
  pub unit: f32,
  pub generation: u32,
  /// the state word saying the image exists (the GPU's swaps), `None` = it does
  pub exists_word: Option<u32>,
}

/// `dust_reproject.comp`'s push constants
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustReprojectPushConstants {
  pub frame: u64,
  pub table: u64,
  pub texels: u32,
  pub groups_x: u32,
  pub pad: [u32; 2],
}
const _: () = assert!(core::mem::size_of::<DustReprojectPushConstants>() == 32);

/// The reprojection table (`dust_reproject.comp`'s `DUST_RPT_*`): the images in the CPU's order,
/// each ratio `u_s / u_now` computed as the CPU's `reproject_add` does
pub fn reproject_table(images: &[ReprojectImage], unit_now: f32, state: u64) -> alloc::vec::Vec<u32> {
  assert!(images.len() <= RPT_MAX_IMAGES, "{} images", images.len());
  let mut t = alloc::vec![0u32; RPT_WORDS];
  t[RPT_N] = images.len() as u32;
  t[RPT_UNIT_NOW] = unit_now.to_bits();
  t[RPT_STATE] = state as u32;
  t[RPT_STATE + 1] = (state >> 32) as u32;
  for (e, im) in images.iter().enumerate() {
    let at = RPT_ENTRY0 + RPT_ENTRY_WORDS * e;
    t[at] = im.address as u32;
    t[at + 1] = (im.address >> 32) as u32;
    let ratio = if im.unit > 0.0 { im.unit / unit_now } else { 0.0 };
    t[at + 2] = ratio.to_bits();
    t[at + 3] = im.generation;
    t[at + 4] = im.map_valid as u32 | (im.exists_word.is_some() as u32) << 1;
    t[at + 5] = im.exists_word.unwrap_or(0);
    for i in 0..9 {
      t[at + 6 + i] = im.map[i].to_bits();
    }
  }
  t
}

/// the CPU's reprojection seed for the images it reprojects (`finish`, F4)
pub fn reproject_seed(generations: impl IntoIterator<Item = u32>) -> u32 {
  let mut s = RPT_SEED0;
  for g in generations {
    s ^= dust::dust_hash32(g);
  }
  dust::dust_hash32(s)
}

impl Device {
  /// Records the reprojection of `table`'s images into the frame pyramid at `frame` (layout
  /// `layout`): the table update, the dispatch (a thread per frame texel over every level), the
  /// barriers. After the frame's clear, before any splat write.
  pub fn cmd_dust_reproject(
    &self,
    cmd: vk::CommandBuffer,
    table_buffer: vk::Buffer,
    table_address: u64,
    frame: u64,
    layout: &PyramidLayout,
    table: &[u32],
  ) {
    let texels: u32 = layout.levels.iter().map(|&(_, w, h)| w * h).sum();
    let groups = texels.div_ceil(64).max(1);
    let gx = groups.min(DUST_DISPATCH_MAX_X);
    let gy = groups.div_ceil(gx);
    let pc = DustReprojectPushConstants { frame, table: table_address, texels, groups_x: gx, pad: [0; 2] };
    let pipeline = self.kernels.pipelines.dust_reproject;
    self.kernels.pipelines.assert_pc_size(pipeline, core::mem::size_of_val(&pc));
    self.cmd_dust_pre_propagate_barrier(cmd);
    unsafe { self.device.cmd_update_buffer(cmd, table_buffer, 0, bytemuck::cast_slice(table)) };
    self.cmd_dust_pre_splat_barrier(cmd);
    unsafe {
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
      self.device.cmd_push_constants(
        cmd,
        self.kernels.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(&pc),
      );
      self.device.cmd_dispatch(cmd, gx, gy, 1);
    }
    self.cmd_dust_pre_splat_barrier(cmd);
  }

  /// Test hook: the GPU reprojection of `images` (each its words in its own layout, its map, unit
  /// and generation) into a frame of `layout` whose header is `header` (texels zero), synchronous;
  /// returns the frame's words
  pub fn dust_reproject_test(
    &self,
    layout: &PyramidLayout,
    header: &[u32],
    images: &[(&[u32], [f32; 9], f32, u32)],
    unit_now: f32,
  ) -> GpuResult<alloc::vec::Vec<u32>> {
    let (frame, bufs, table, tb) = {
      let res = self.res.read();
      let allocator = res.allocator.allocator.as_allocator_view();
      let upload = |words: &[u32], name: &str| -> GpuResult<DustBuffer> {
        let (b, mapped) = self.dust_create_buffer(
          &allocator,
          words.len().max(1) as u64 * 4,
          DustMem::HostWrite,
          false,
          vk::BufferUsageFlags::TRANSFER_DST,
          name,
        )?;
        unsafe { core::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), mapped, words.len() * 4) };
        allocator.flush_allocation(&b.alloc, 0, words.len() as u64 * 4)?;
        Ok(b)
      };
      let mut frame_words = alloc::vec![0u32; layout.total_words as usize];
      frame_words[..header.len()].copy_from_slice(header);
      let frame = upload(&frame_words, "DustReprojectTestFrame")?;
      let mut bufs = alloc::vec::Vec::new();
      let mut ri = alloc::vec::Vec::new();
      for (k, &(words, map, unit, generation)) in images.iter().enumerate() {
        let b = upload(words, &alloc::format!("DustReprojectTestImage{k}"))?;
        ri.push(ReprojectImage { address: b.address, map, map_valid: true, unit, generation, exists_word: None });
        bufs.push(b);
      }
      let table = reproject_table(&ri, unit_now, 0);
      let (tb, _) = self.dust_create_buffer(
        &allocator,
        RPT_WORDS as u64 * 4,
        DustMem::Device,
        false,
        vk::BufferUsageFlags::TRANSFER_DST,
        "DustReprojectTestTable",
      )?;
      (frame, bufs, table, tb)
    };
    let mut cmds = self.run_transient_commands(|cmd| {
      self.cmd_dust_reproject(cmd, tb.buffer, tb.address, frame.address, layout, &table);
      Ok(())
    })?;
    cmds.cleanup(&self.device);
    let out: alloc::vec::Vec<u32> = pod_vec(&self.dust_download_bytes(frame.buffer, 0, layout.total_words as u64 * 4)?);
    let res = self.res.read();
    let allocator = res.allocator.allocator.as_allocator_view();
    for b in bufs.into_iter().chain([frame, tb]) {
      let mut a = b.alloc;
      unsafe { allocator.destroy_buffer(b.buffer, &mut a) };
    }
    Ok(out)
  }
}

// ─── The class images' draws (I3) ───────────────────────────────────────────

/// One build pair the frame draws: its frozen view at the current nucleus (`ClassImage::build_at`)
/// and the unit its image is drawn with
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CadenceBuild {
  pub k: usize,
  pub p: usize,
  pub view: CadenceView,
  pub unit: f32,
}

/// One (pair, tier) build draw: the tier's buffers and the frozen push constants (the current
/// view's with the frozen mvp / eye and `inv_unit = 1 / unit`, as the CPU's F3)
#[derive(Debug, Clone, Copy)]
pub struct CadenceBuildDraw {
  pub k: usize,
  pub p: usize,
  pub key: u64,
  pub system: u64,
  pub ring_base: u32,
  pub first_slot: u32,
  pub capacity: u32,
  pub render: u64,
  pub moments: u64,
  pub pc: DustSplatPushConstants,
}

/// the header words a class image routes through the frame's (copy-in before its draws, copy-out
/// after): everything but its geometry (levels, size, level table), its own τ_max, level mask and
/// unit, and its queue word
pub const CAD_ROUTED_WORDS: [(u32, u32); 3] = [(4, 7), (8, 15), (59, dust::PYRAMID_HEADER_WORDS)];

impl DustCadenceGpu {
  /// the class images' geometry for the view's layout (the config's margin: the CPU mirror's
  /// `CadenceConfig::image_geom`)
  pub fn image_geom(&self, layout: &PyramidLayout) -> ImageGeom {
    self.cfg.image_geom(layout.width, layout.height)
  }
  /// words between two class images in the view buffer (256-byte aligned)
  pub fn image_stride(&self, layout: &PyramidLayout) -> u32 {
    image_words(&self.image_geom(layout)).next_multiple_of(64)
  }
  /// the class images' span in the view buffer
  pub fn image_span(&self, layout: &PyramidLayout) -> u32 {
    self.cfg.images() as u32 * self.image_stride(layout)
  }
  /// the word of image (k, p, role) in the view buffer (after the frame pyramid)
  pub fn image_word(&self, layout: &PyramidLayout, k: usize, p: usize, role: u8) -> u32 {
    layout.total_words + ((k * self.np() + p) * 2 + role as usize) as u32 * self.image_stride(layout)
  }
  /// the owner bytes of system `system`'s tier at `ring_base` (byte `slot` of the address)
  pub fn owner_address(&self, system: u64, ring_base: u32) -> Option<u64> {
    self.systems.get(&system).map(|s| s.buf.address + ring_base as u64)
  }
}

impl Device {
  /// the owner bytes the view's fresh splat and speck walks filter by (0: no cadence)
  pub(super) fn cad_owner_address(&self, mgr: &DustManager, pe: gpu::PresentationEngineHandle, id: u64, ring_base: u32) -> u64 {
    mgr.cadence.get(&pe).and_then(|c| c.lock().owner_address(id, ring_base)).unwrap_or(0)
  }

  /// Test hook: the light tolerance τ of `pe`'s cadence from its next frame
  /// (`CadenceConfig::light_tol`; the CPU mirror's `cfg.light_tol` must be set alike)
  pub fn dust_cadence_set_light_tol(&self, pe: gpu::PresentationEngineHandle, tau: f32) {
    let res = self.res.read();
    if let Some(entry) = res.dust_manager.as_ref().and_then(|m| m.cadence.get(&pe)) {
      entry.lock().cfg.light_tol = tau;
    }
  }

  /// Test hook: the in-frame gates of `pe`'s next frames (`DUST_FRAME_GATES_OFF_BIT`; the
  /// identity tests run with them off)
  pub fn dust_set_frame_gates(&self, pe: gpu::PresentationEngineHandle, on: bool) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      view.host.lock().frame_gates_off = !on;
    }
  }

  /// The EWA queue's records drawn into `target` (a pyramid of the view buffer whose queue word
  /// points at the view's queue), then the queue emptied: its dispatch (0, 1, 1) and append
  /// counter 0 — never word 4 (the capacity override) or the profiler's words 8–127
  fn cad_queue_drain(&self, cmd: vk::CommandBuffer, buffer: vk::Buffer, queue_word: u32, target: u64) {
    let qpc = dust::DustMeasurePushConstants { pyramid: target, pad: [0; 2] };
    let pipeline = match self.kernels.pipelines.dust_draw_queue_prof {
      Some(p) if crate::gpu_backends::vulkan::physics::dust_profile_requested() => p,
      _ => self.kernels.pipelines.dust_draw_queue,
    };
    // the mono draw family's queue draw (the monolithic blob; the same dispatch)
    let pipeline = if self.dust_draw_staged() {
      pipeline
    } else {
      match self.kernels.pipelines.dust_draw_queue_mono_prof {
        Some(p) if crate::gpu_backends::vulkan::physics::dust_profile_requested() => p,
        _ => self.kernels.pipelines.dust_draw_queue_mono.unwrap_or(pipeline),
      }
    };
    self.kernels.pipelines.assert_pc_size(pipeline, core::mem::size_of_val(&qpc));
    let to_queue = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::COMPUTE_SHADER)
      .dst_access_mask(
        vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
      );
    let to_reset = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::DRAW_INDIRECT)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::INDIRECT_COMMAND_READ)
      .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
    unsafe {
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_queue)));
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
      self.device.cmd_push_constants(cmd, self.kernels.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&qpc));
      self.device.cmd_dispatch_indirect(cmd, buffer, queue_word as u64 * 4);
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_reset)));
      self.device.cmd_update_buffer(cmd, buffer, queue_word as u64 * 4, bytemuck::cast_slice(&[0u32, 1, 1, 0]));
    }
    self.cmd_dust_pre_splat_barrier(cmd);
  }

  /// header copies between the frame pyramid (word 0 of the view buffer) and a class image
  fn cad_header_copy(&self, cmd: vk::CommandBuffer, buffer: vk::Buffer, image_word: u32, into_image: bool) {
    let regions: alloc::vec::Vec<vk::BufferCopy> = CAD_ROUTED_WORDS
      .iter()
      .map(|&(a, b)| {
        let (src, dst) = if into_image { (a as u64 * 4, (image_word + a) as u64 * 4) } else { ((image_word + a) as u64 * 4, a as u64 * 4) };
        vk::BufferCopy::default().src_offset(src).dst_offset(dst).size((b - a) as u64 * 4)
      })
      .collect();
    let to_copy = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::TRANSFER_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE);
    unsafe {
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_copy)));
      self.device.cmd_copy_buffer(cmd, buffer, buffer, &regions);
    }
    self.cmd_dust_pre_splat_barrier(cmd);
  }

  /// The frame's build draws (the CPU's F3 draws, after [`Self::cmd_dust_cadence_chunks`] of every
  /// tier): per pair, in the draws' order, the frame's header words into the image, each tier's
  /// chunk drawn from its draw list into the image at its frozen view, the EWA queue drained into
  /// the image, the counters back into the frame's header
  pub fn cmd_dust_cadence_builds(&self, cmd: vk::CommandBuffer, pe: gpu::PresentationEngineHandle, draws: &[CadenceBuildDraw]) -> GpuResult<()> {
    struct Job {
      k: usize,
      p: usize,
      image_word: u32,
      list: u64,
      n: u32,
      d: CadenceBuildDraw,
    }
    let (buffer, base, queue_word, jobs) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let entry = mgr.cadence.get(&pe).ok_or(gpu_err!("dust cadence of PE {} absent", pe.0))?;
      let c = entry.lock();
      if !matches!(c.frame.as_ref(), Some(f) if f.mode == CadenceMode::Run) {
        return Ok(());
      }
      if view.image_words != c.image_span(&view.layout) {
        return Err(gpu_err!("dust cadence: the view of PE {} has no class images yet", pe.0));
      }
      let pyramid = view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?;
      let np = c.np();
      let mut jobs = alloc::vec::Vec::new();
      for d in draws {
        let pr = &c.pairs[d.k * np + d.p];
        let n = pr.step_upper.get(&d.key).copied().unwrap_or(0);
        let s = c.systems.get(&d.system).ok_or(gpu_err!("dust cadence: system {} not begun", d.system))?;
        jobs.push(Job {
          k: d.k,
          p: d.p,
          image_word: c.image_word(&view.layout, d.k, d.p, pr.role),
          list: s.buf.address + s.layout.draw(d.ring_base, d.capacity, d.k as u32, d.p as u32) as u64 * 4,
          n,
          d: *d,
        });
      }
      (pyramid.buffer, pyramid.address, view.queue_word(), jobs)
    };
    let mut i = 0;
    while i < jobs.len() {
      let (k, p, iw) = (jobs[i].k, jobs[i].p, jobs[i].image_word);
      let target = base + iw as u64 * 4;
      self.cad_header_copy(cmd, buffer, iw, true);
      while i < jobs.len() && jobs[i].k == k && jobs[i].p == p {
        let j = &jobs[i];
        if j.n > 0 {
          self.dust_splat_impl(
            cmd, pe, j.d.system, j.d.ring_base, j.d.first_slot, j.d.capacity, j.d.render, j.d.moments, j.d.pc, Some(target), j.list, j.n, Some(0),
          )?;
        }
        i += 1;
      }
      self.cad_queue_drain(cmd, buffer, queue_word, target);
      self.cad_header_copy(cmd, buffer, iw, false);
      self.cmd_dust_timestamp(cmd, pe, DUST_TS_PAIR0 + (k * 2 + p) as u32);
    }
    Ok(())
  }

  /// Test hook: a tier's splat over a list of ring slots (`list` a device address of `n` entries,
  /// ~0 = a hole) into the frame pyramid, no owner filter (`index_lists_draw_the_same_pyramid`)
  #[allow(clippy::too_many_arguments)]
  pub fn cmd_dust_splat_list(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
    id: u64,
    ring_base: u32,
    first_slot: u32,
    capacity: u32,
    render: u64,
    moments: u64,
    pc: DustSplatPushConstants,
    list: u64,
    n: u32,
  ) -> GpuResult<()> {
    self.dust_splat_impl(cmd, pe, id, ring_base, first_slot, capacity, render, moments, pc, None, list, n, Some(0))
  }

  /// Test hook: a host-visible device buffer holding `words` (its address in `.address`)
  pub fn dust_test_upload(&self, words: &[u32]) -> GpuResult<DustBuffer> {
    let res = self.res.read();
    let allocator = res.allocator.allocator.as_allocator_view();
    let (b, mapped) = self.dust_create_buffer(&allocator, words.len().max(1) as u64 * 4, DustMem::HostWrite, false, vk::BufferUsageFlags::TRANSFER_DST, "DustTestUpload")?;
    unsafe { core::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), mapped, words.len() * 4) };
    allocator.flush_allocation(&b.alloc, 0, words.len().max(1) as u64 * 4)?;
    Ok(b)
  }

  /// Test hook: frees a [`Self::dust_test_upload`] buffer (after the device is idle)
  pub fn dust_test_free(&self, b: DustBuffer) {
    let res = self.res.read();
    let allocator = res.allocator.allocator.as_allocator_view();
    let mut a = b.alloc;
    unsafe { allocator.destroy_buffer(b.buffer, &mut a) };
  }

  /// Test hook: every class image of `pe` as `(k, p, shown words, build words)` (header + levels
  /// of the image's own layout)
  #[allow(clippy::type_complexity)]
  pub fn download_dust_cadence_images(
    &self,
    pe: gpu::PresentationEngineHandle,
  ) -> GpuResult<alloc::vec::Vec<(usize, usize, alloc::vec::Vec<u32>, alloc::vec::Vec<u32>)>> {
    let (buffer, words, places) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let entry = mgr.cadence.get(&pe).ok_or(gpu_err!("dust cadence of PE {} absent", pe.0))?;
      let c = entry.lock();
      let pyramid = view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?;
      let (nc, np) = (c.nc(), c.np());
      let mut places = alloc::vec::Vec::new();
      for k in 0..nc {
        for p in 0..np {
          let role = c.pairs[k * np + p].role;
          places.push((k, p, c.image_word(&view.layout, k, p, role ^ 1), c.image_word(&view.layout, k, p, role)));
        }
      }
      (pyramid.buffer, image_words(&c.image_geom(&view.layout)), places)
    };
    let mut out = alloc::vec::Vec::new();
    for (k, p, shown, build) in places {
      let s: alloc::vec::Vec<u32> = pod_vec(&self.dust_download_bytes(buffer, shown as u64 * 4, words as u64 * 4)?);
      let b: alloc::vec::Vec<u32> = pod_vec(&self.dust_download_bytes(buffer, build as u64 * 4, words as u64 * 4)?);
      out.push((k, p, s, b));
    }
    Ok(out)
  }
}

// ─── The frame loop's pieces (I4) ───────────────────────────────────────────

impl Device {
  /// The device-local memory (bytes) the memory rules read (`dust_vqueue_records`,
  /// `CadenceConfig::for_device_local`)
  pub fn dust_device_local_bytes(&self) -> u64 {
    let props = unsafe { self.instance.instance.get_physical_device_memory_properties(self.query_result.physical_device) };
    props.memory_heaps[..props.memory_heap_count as usize]
      .iter()
      .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
      .map(|h| h.size)
      .sum()
  }

  /// The frame loop's switch: the scene's temporal reuse tolerance `reuse_px` (ε px, 0 = off)
  /// turns the view's cadence on with ε = `reuse_px` (no budget loop: the image is what the setting
  /// promises); `AETHERVK_DUST_CADENCE=1` turns it on whatever the setting (the diagnostic path,
  /// with the budget loop), `AETHERVK_DUST_CADENCE=0` off. A view without a cadence gets one by the
  /// memory rule (`CadenceConfig::from_env`: the classes, phases, ε and margin knobs;
  /// `AETHERVK_DUST_CADENCE_EPS` overrides the setting's ε). Returns whether the view runs the cadence.
  pub fn dust_cadence_auto(&self, pe: gpu::PresentationEngineHandle, reuse_px: u32) -> bool {
    let var = aethervk_oshal_rlib::os::env::var;
    let env = var("AETHERVK_DUST_CADENCE");
    let want = match env.as_deref().map(str::trim) {
      Some("1") => true,
      Some("0") => false,
      _ => reuse_px > 0,
    };
    // the setting's ε (the env knob and the budget loop's own ε win)
    let eps = (reuse_px > 0 && var("AETHERVK_DUST_CADENCE_EPS").is_none()).then_some(reuse_px as f32);
    let active = self.dust_cadence_active(pe);
    if want == active {
      if active && let Some(e) = eps {
        let res = self.res.read();
        if let Some(entry) = res.dust_manager.as_ref().and_then(|m| m.cadence.get(&pe)) {
          entry.lock().cfg.eps_px = e;
        }
      }
      return active;
    }
    let cfg = if want {
      CadenceConfig::from_env(self.dust_device_local_bytes()).map(|mut c| {
        if let Some(e) = eps {
          c.eps_px = e;
        }
        c
      })
    } else {
      None
    };
    if let Err(e) = self.dust_cadence_enable(pe, cfg) {
      aethervk_oshal_rlib::log!("[Dust cadence] view {}: {e}", pe.0);
      return false;
    }
    self.dust_cadence_active(pe)
  }

  /// The emission value tier `key` had at `pe`'s last cadence frame (the host's
  /// `rewritten_from` needs it), `None` before any
  pub fn dust_cadence_tier_emitted(&self, pe: gpu::PresentationEngineHandle, key: u64) -> Option<u64> {
    let res = self.res.read();
    let c = res.dust_manager.as_ref()?.cadence.get(&pe)?;
    let e = c.lock().tiers.get(&key).map(|t| t.emitted);
    e
  }

  /// The reprojection of `pe`'s shown class images into its frame pyramid (the CPU's F4): the
  /// table from the host (per pair in the CPU's order: the shown image's address by its role, the
  /// map `map_to_image` from this frame's view to its frozen view, its unit and generation; its
  /// existence the GPU's state word), then `dust_reproject.comp`. After the builds, before the
  /// speck pass and the fresh splat.
  pub fn cmd_dust_cadence_reproject(&self, cmd: vk::CommandBuffer, pe: gpu::PresentationEngineHandle) -> GpuResult<()> {
    let (rpt_buffer, rpt_address, frame, layout, table) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let entry = mgr.cadence.get(&pe).ok_or(gpu_err!("dust cadence of PE {} absent", pe.0))?;
      let c = entry.lock();
      let Some(fr) = c.frame.as_ref() else { return Ok(()) };
      if fr.mode != CadenceMode::Run || view.image_words != c.image_span(&view.layout) || view.image_words == 0 {
        return Ok(());
      }
      let (Some(pv), Some(geom)) = (c.prev.as_ref(), c.geom.as_ref()) else { return Ok(()) };
      let pyramid = view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?;
      let (nc, np) = (c.nc(), c.np());
      let mut images = alloc::vec::Vec::new();
      for k in 0..nc {
        for p in 0..np {
          let pr = &c.pairs[k * np + p];
          let map = pr.shown_at(fr.nucleus).and_then(|sv| pv.view.map_to_image(&sv, geom));
          images.push(ReprojectImage {
            address: pyramid.address + c.image_word(&view.layout, k, p, pr.role ^ 1) as u64 * 4,
            map: map.unwrap_or(crate::scene::dust::cadence::MAP_IDENTITY),
            map_valid: map.is_some(),
            unit: pr.shown_unit,
            generation: pr.generation,
            exists_word: Some(cads_pair(k, p) as u32),
          });
        }
      }
      let table = reproject_table(&images, fr.unit_now, c.state.address);
      (c.rpt.buffer, c.rpt.address, pyramid.address, view.layout.clone(), table)
    };
    self.cmd_dust_reproject(cmd, rpt_buffer, rpt_address, frame, &layout, &table);
    Ok(())
  }

  /// Test hook: a digest of `pe`'s cadence (host state and the GPU's state words and owner bytes;
  /// the reused-frame test: nothing advances)
  pub fn dust_cadence_digest(&self, pe: gpu::PresentationEngineHandle) -> GpuResult<u64> {
    let snap = self.download_dust_cadence(pe)?;
    let drawn = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let c = mgr.cadence.get(&pe).ok_or(gpu_err!("dust cadence of PE {} absent", pe.0))?;
      let d = c.lock().drawn_frame;
      d
    };
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |v: u64| {
      for b in v.to_le_bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
      }
    };
    eat(drawn);
    // (the frame's own counters excluded: words 1024.. are per-frame diagnostics)
    for &w in &snap.state[..CADS_HIST] {
      eat(w as u64);
    }
    for t in &snap.tiers {
      for &b in t.owner.iter().chain(t.incoming.iter()) {
        eat(b as u64);
      }
    }
    Ok(h)
  }
}

// ─── The telemetry (I5) ─────────────────────────────────────────────────────

pub const CADS_FRESH: usize = 1029;
/// the handover's collapses and dropped collapses this frame (2 words)
pub const CADS_HANDOVER: usize = 1030;
pub const CADS_SHOWN_PAIR0: usize = 1032;
pub const CADS_SHOWN_LIGHT: usize = 1040;
pub const CADS_FRAME_LIGHT: usize = 1042;
/// the per-frame words the host zeroes each drawn frame (the histogram, the shell count, the
/// telemetry)
pub const CADS_FRAME_WORDS: core::ops::Range<usize> = 1024..1048;

/// What the cadence of a view did, [`DUST_READBACK_SLOTS`] frames ago (its readback) — the lab's
/// `dust cadence:` line and `stats.jsonl` ([`Device::dust_cadence_telemetry`])
#[derive(Debug, Clone, Default)]
pub struct CadenceTelemetry {
  pub eps_px: f32,
  pub classes: alloc::vec::Vec<u32>,
  pub phases: u32,
  pub margin_frac: f32,
  pub margin_px: [u32; 2],
  pub mode: Option<CadenceMode>,
  pub reset: Option<CadenceReset>,
  pub swaps: alloc::vec::Vec<(usize, usize)>,
  /// per class: within the motion budget
  pub admitted: alloc::vec::Vec<bool>,
  /// the live clusters, those drawn fresh, and their share
  pub live: u32,
  pub fresh: u32,
  pub fresh_fraction: f32,
  /// per pair `(k, p, [shown, build, pending])`: the lists' lengths over the tiers
  pub lists: alloc::vec::Vec<(usize, usize, [u32; 3])>,
  /// per pair: the shown list's entries still SHOWN (the others are ghosts: dead lives)
  pub shown_valid: alloc::vec::Vec<u32>,
  /// shown entries no longer valid, and their share of the shown entries (by entries, not light)
  pub ghost_entries: u32,
  pub ghost_share: f32,
  /// the shown images that exist
  pub images_shown: u32,
  /// aborted lists by reason (`cadence::ABORT_RATE`, `_BUDGET`, `_PREDICTION`) and the coverage drops
  pub aborts: [u32; 4],
  /// entries refused by the footprint rule at the build and at the classify (no margin only), stale slots without
  /// an anchor, classify candidates refused by the shell rule, the admissible-class histogram
  pub d3_rejected: u32,
  pub d3_at_classify: u32,
  pub unshifted_stale: u32,
  pub shell_forced: u32,
  pub class_hist: [u32; 4],
  /// the light the reprojection wrote over the frame's (counts · 4^ℓ; 0 without a frame)
  pub shown_light_share: f64,
  /// the newest GPU timings of the view (`cadence_ms`, `classify_ms`, `builds_ms`, `pair_build_ms`,
  /// `reproject_ms`, `splat_only_ms` = the fresh splat): a readback of its own, also a few frames
  /// late
  pub timings: Option<DustGpuTimings>,
}

/// what a frame was for the cadence, kept with its readback slot
#[derive(Debug, Clone, Default)]
struct SlotInfo {
  mode: Option<CadenceMode>,
  reset: Option<CadenceReset>,
  swaps: alloc::vec::Vec<(usize, usize)>,
  admitted: alloc::vec::Vec<bool>,
  live: u32,
}

impl Device {
  /// The frame end's cadence work (from [`Device::cmd_dust_frame_end`], after its packet queue's
  /// draw: the frame is finished): the frame's light for the telemetry, then the state's readback
  /// copy (the frame recorded [`Device::cmd_dust_cadence_end`])
  pub(super) fn cad_frame_end(&self, cmd: vk::CommandBuffer, mgr: &DustManager, pe: gpu::PresentationEngineHandle, frame: u64, layout: &PyramidLayout) {
    let Some(entry) = mgr.cadence.get(&pe) else { return };
    let mut c = entry.lock();
    if !c.copy_pending {
      return;
    }
    c.copy_pending = false;
    let telemetry = !aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_CADENCE_TELEMETRY").is_some_and(|v| v.trim() == "0");
    if telemetry {
      let mut head = [0u32; RPT_ENTRY0];
      head[RPT_STATE] = c.state.address as u32;
      head[RPT_STATE + 1] = (c.state.address >> 32) as u32;
      let texels: u32 = layout.levels.iter().map(|&(_, w, h)| w * h).sum();
      let groups = texels.div_ceil(64).max(1);
      let gx = groups.min(DUST_DISPATCH_MAX_X);
      let gy = groups.div_ceil(gx);
      let pc = DustReprojectPushConstants { frame, table: c.rpt.address, texels, groups_x: gx, pad: [1, 0] };
      let pipeline = self.kernels.pipelines.dust_reproject;
      self.cmd_dust_pre_propagate_barrier(cmd);
      unsafe { self.device.cmd_update_buffer(cmd, c.rpt.buffer, 0, bytemuck::cast_slice(&head)) };
      self.cmd_dust_pre_splat_barrier(cmd);
      unsafe {
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        self.device.cmd_push_constants(cmd, self.kernels.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc));
        self.device.cmd_dispatch(cmd, gx, gy, 1);
      }
    }
    let slot = (c.copies % DUST_READBACK_SLOTS as u64) as usize;
    let to_copy = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
    let to_host = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::HOST)
      .dst_access_mask(vk::AccessFlags2::HOST_READ);
    unsafe {
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_copy)));
      self.device.cmd_copy_buffer(
        cmd,
        c.state.buffer,
        c.readback.buffer,
        &[vk::BufferCopy::default().src_offset(0).dst_offset((slot * CADS_WORDS * 4) as u64).size((CADS_WORDS * 4) as u64)],
      );
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_host)));
    }
    c.slot_meta[slot].get_or_insert_with(SlotMeta::default);
    c.copies += 1;
  }

  /// The cadence telemetry of `pe` (its newest readback; `None` without a cadence or before the
  /// first readback)
  pub fn dust_cadence_telemetry(&self, pe: gpu::PresentationEngineHandle) -> Option<CadenceTelemetry> {
    let timings = self.dust_gpu_timings(pe);
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref()?;
    let entry = mgr.cadence.get(&pe)?;
    let c = entry.lock();
    let st = c.last_state.as_ref()?;
    let info = c.last_info.clone().unwrap_or_default();
    let (nc, np) = (c.nc(), c.np());
    let nt = c.tiers.values().map(|t| t.t + 1).max().unwrap_or(0);
    let mut tel = CadenceTelemetry {
      eps_px: c.cfg.eps_px,
      classes: c.cfg.classes.clone(),
      phases: c.cfg.phases,
      margin_frac: c.cfg.margin_frac,
      margin_px: c.geom.as_ref().map_or([0, 0], |g| g.margin),
      mode: info.mode,
      reset: info.reset,
      swaps: info.swaps.clone(),
      admitted: info.admitted.clone(),
      live: info.live,
      fresh: st[CADS_FRESH],
      shell_forced: st[CADS_SHELL_FORCED],
      class_hist: [st[CADS_HIST], st[CADS_HIST + 1], st[CADS_HIST + 2], st[CADS_HIST + 3]],
      timings,
      ..Default::default()
    };
    tel.fresh_fraction = if info.live > 0 { tel.fresh as f32 / info.live as f32 } else { 0.0 };
    let (mut shown_total, mut valid_total) = (0u32, 0u32);
    for k in 0..nc {
      for p in 0..np {
        let mut l = [0u32; 3];
        for t in 0..nt {
          let tp = cads_tp(t, k, p);
          for s in 0..3 {
            l[s] += st[tp + s];
          }
        }
        let pr = cads_pair(k, p);
        let flags = st[pr + 2];
        for list in [1usize, 2] {
          if flags & list as u32 != 0 {
            let r = (st[pr + 8 + list] as usize).min(2);
            tel.aborts[r] += 1;
          }
        }
        if st[pr + 11] != 0 {
          tel.aborts[3] += 1;
        }
        tel.images_shown += (st[pr] != 0) as u32;
        let valid = st[CADS_SHOWN_PAIR0 + k * 2 + p];
        shown_total += l[0];
        valid_total += valid.min(l[0]);
        tel.shown_valid.push(valid);
        tel.lists.push((k, p, l));
      }
    }
    for t in 0..nt {
      for k in 0..nc {
        for p in 0..np {
          tel.d3_rejected += st[cads_tp(t, k, p) + 9];
        }
      }
      tel.d3_at_classify += st[cads_tp(t, 0, 0) + 10];
      tel.unshifted_stale += st[cads_tp(t, 0, 0) + 11];
    }
    tel.ghost_entries = shown_total - valid_total;
    tel.ghost_share = if shown_total > 0 { tel.ghost_entries as f32 / shown_total as f32 } else { 0.0 };
    let u64w = |w: usize| st[w] as u64 | (st[w + 1] as u64) << 32;
    let (shown, frame) = (u64w(CADS_SHOWN_LIGHT), u64w(CADS_FRAME_LIGHT));
    tel.shown_light_share = if frame > 0 { shown as f64 / frame as f64 } else { 0.0 };
    Some(tel)
  }
}

#[cfg(test)]
#[path = "test_cadence.rs"]
mod tests;
