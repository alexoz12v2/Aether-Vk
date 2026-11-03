//! Play quality (handoff P3): while the simulation **runs** (not paused, not seeking) the dust
//! is drawn with coarser knobs so a frame costs ≥ 5× less and the dust updates at display rate
//! (the reviewer's "the animation doesn't work": at 1 h/s the dust frame rate was 0.7–6 fps, the
//! nucleus turned at display rate, so the dust read as immobile). Paused, the knobs pass through
//! **byte-identical**: the first frame after a pause is today's image (the knobs enter the reuse
//! hash and the cadence fingerprint, `gpu::frame`, so that frame redraws from scratch).
//!
//! One place for the mapping ([`PlayQuality::apply`]); `AETHERVK_DUST_PLAY_QUALITY=0` disables it
//! ([`PlayQuality::from_env`] → `None`), `AETHERVK_DUST_PLAY_QUALITY=v,h,p` overrides the three
//! play values (volume detail, haze detail, trajectory detail).
//!
//! The preview (the "Playback preview quality" setting, [`PlayPreview`]) picks the play values:
//! Off draws the scene's knobs while playing too, Smooth / Balanced / Quality coarsen them less
//! and less. Any preview turns the temporal reuse off while it lasts: its class images are
//! rebuilt over 8–64 frames at a frozen view, so a playing coma read as one still picture laid
//! over the comet. The preview lasts while the simulation runs and for [`PREVIEW_HOLD_US`] after
//! a pause or a seek (scrubbing stays in preview); the full render then starts and refines.

use super::{
  DUST_HAZE_DETAIL_MIN, DUST_PROPAGATE_DETAIL_MIN, DUST_VOLUME_DETAIL_MIN, clamp_haze_detail,
  clamp_propagate_detail, clamp_volume_detail,
};

/// The scene's dust quality knobs the render thread stamps on a frame
/// (`SceneContext::dust_volume_detail` and friends)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DustKnobs {
  /// "Volume detail" (pieces of a volumetric packet, 1..=16)
  pub volume_detail: u32,
  /// "Haze detail" (top-level sub-packets, 1..=16)
  pub haze_detail: u32,
  /// "Trajectory detail" (solved polyline edges, 17 / 9 / 5)
  pub propagate_detail: u32,
  /// "Temporal reuse tolerance" (ε px, 0 = off)
  pub reuse_px: u32,
}

/// The play-time values of the knobs (the paused values are the scene's own)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlayQuality {
  /// volume detail while playing (default 2: a volumetric packet drawn with 1/8 of its pieces)
  pub volume_detail: u32,
  /// haze detail while playing (default 4: a quarter of the top-level sub-packets)
  pub haze_detail: u32,
  /// trajectory detail while playing (default 5: 9 Kepler solves per cluster instead of 21)
  pub propagate_detail: u32,
}

impl Default for PlayQuality {
  fn default() -> Self {
    Self {
      volume_detail: 2,
      haze_detail: 4,
      propagate_detail: DUST_PROPAGATE_DETAIL_MIN,
    }
  }
}

/// The "Playback preview quality" setting: how much the dust is coarsened while playing
/// (`SceneContext::dust_play_preview`; 0 Off, 1 Smooth, 2 Balanced, 3 Quality)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayPreview {
  /// the scene's knobs while playing too (the full render, however slow)
  Off,
  /// the coarsest: volume 1, haze 2, trajectory 5
  Smooth,
  /// the default: volume 2, haze 4, trajectory 5 ([`PlayQuality::default`])
  Balanced,
  /// volume 4, haze 8, trajectory 9
  Quality,
}

/// The default preview level ([`PlayPreview::Balanced`])
pub const PLAY_PREVIEW_DEFAULT: u32 = 2;

/// How long the preview outlasts a pause or a seek before the full render starts (µs)
pub const PREVIEW_HOLD_US: i64 = 2_000_000;

/// How long after a paused full frame drew the full render counts as still in progress (µs): a
/// refinement step waits for the previous frame's readback (up to ~0.9 s on a GTX 1070 at Low)
/// before it draws the next one
pub const FULL_RENDER_BUSY_US: i64 = 1_500_000;

/// What the dust view is doing, for the viewport's badge (`SceneContext::dust_render_phase`)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DustRenderPhase {
  /// the full render is settled (or nothing is drawing)
  Settled,
  /// the simulation runs: the preview
  Playing,
  /// paused or seeked, still in the preview: the full render starts when the hold ends; the
  /// fraction of the hold elapsed, in [0, 1]
  Hold(f32),
  /// the full render (and its refinement) is drawing: frames of up to a second each
  Rendering,
}

/// The dust phase at `now_us`: `running` the simulation, `preview_on` a preview level other than
/// Off, `hold_until_us` the end of the hold ([`PREVIEW_HOLD_US`] after a pause or a seek,
/// `i64::MIN` none), `full_draw_us` when a paused full frame last drew (`i64::MIN` never)
pub fn dust_render_phase(
  now_us: i64,
  running: bool,
  preview_on: bool,
  hold_until_us: i64,
  full_draw_us: i64,
) -> DustRenderPhase {
  if preview_on && running {
    return DustRenderPhase::Playing;
  }
  if preview_on && now_us < hold_until_us {
    let left = (hold_until_us - now_us).min(PREVIEW_HOLD_US) as f32;
    return DustRenderPhase::Hold(1.0 - left / PREVIEW_HOLD_US as f32);
  }
  if !running && full_draw_us != i64::MIN && now_us.saturating_sub(full_draw_us) < FULL_RENDER_BUSY_US {
    return DustRenderPhase::Rendering;
  }
  DustRenderPhase::Settled
}

impl PlayPreview {
  /// The setting's value (out of range → [`PLAY_PREVIEW_DEFAULT`])
  pub fn from_level(level: u32) -> Self {
    match level {
      0 => Self::Off,
      1 => Self::Smooth,
      3 => Self::Quality,
      _ => Self::Balanced,
    }
  }

  /// The play values of this level (`None` for Off)
  pub fn quality(self) -> Option<PlayQuality> {
    match self {
      Self::Off => None,
      Self::Smooth => Some(PlayQuality {
        volume_detail: 1,
        haze_detail: 2,
        propagate_detail: DUST_PROPAGATE_DETAIL_MIN,
      }),
      Self::Balanced => Some(PlayQuality::default()),
      Self::Quality => Some(PlayQuality { volume_detail: 4, haze_detail: 8, propagate_detail: 9 }),
    }
  }

  /// The preview's dust frame budget (ms of GPU time) and the largest cluster stride
  /// [`PlayStride`] may reach to meet it; `None` when Off (no stride)
  pub fn stride_budget(self) -> Option<StrideBudget> {
    match self {
      Self::Off => None,
      Self::Smooth => Some(StrideBudget { budget_ms: 16.0, max_stride: 64 }),
      Self::Balanced => Some(StrideBudget { budget_ms: 33.0, max_stride: 32 }),
      Self::Quality => Some(StrideBudget { budget_ms: 66.0, max_stride: 8 }),
    }
  }

  /// `AETHERVK_DUST_PLAY_PREVIEW=0..3` (the setting's initial value; the setting overrides it).
  /// `AETHERVK_DUST_PLAY_QUALITY=0` (the older switch) means Off.
  pub fn default_level() -> u32 {
    if PlayQuality::from_env().is_none() {
      return 0;
    }
    aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_PLAY_PREVIEW")
      .and_then(|v| v.trim().parse::<u32>().ok())
      .filter(|l| *l <= 3)
      .unwrap_or(PLAY_PREVIEW_DEFAULT)
  }
}

impl PlayQuality {
  /// `AETHERVK_DUST_PLAY_QUALITY`: unset or `1` → the default; `0` → `None` (off: the scene's
  /// knobs whatever the clock does); `v,h,p` → those play values (clamped)
  pub fn from_env() -> Option<Self> {
    let v = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_PLAY_QUALITY");
    Self::parse(v.as_deref())
  }

  /// [`Self::from_env`] on a given value (tests)
  pub fn parse(v: Option<&str>) -> Option<Self> {
    let v = v.map(str::trim).unwrap_or("");
    match v {
      "" | "1" => Some(Self::default()),
      "0" => None,
      _ => {
        let mut it = v.split(',').map(|s| s.trim().parse::<u32>().ok());
        let d = Self::default();
        let vol = it.next().flatten().unwrap_or(d.volume_detail);
        let haze = it.next().flatten().unwrap_or(d.haze_detail);
        let prop = it.next().flatten().unwrap_or(d.propagate_detail);
        Some(Self {
          volume_detail: clamp_volume_detail(vol.max(DUST_VOLUME_DETAIL_MIN)),
          haze_detail: clamp_haze_detail(haze.max(DUST_HAZE_DETAIL_MIN)),
          propagate_detail: clamp_propagate_detail(prop),
        })
      }
    }
  }

  /// The knobs a frame draws with: `running` (playing, or inside the preview hold) → the play
  /// values, never above the scene's own (a user who set a coarser detail keeps it), and no
  /// temporal reuse (a reused class image is a still picture of the dust); not running → `knobs`
  /// **unchanged** (every paused frame, the byte gate's included, draws with the scene's knobs).
  pub fn apply(&self, running: bool, knobs: DustKnobs) -> DustKnobs {
    if !running {
      return knobs;
    }
    DustKnobs {
      volume_detail: clamp_volume_detail(self.volume_detail.min(knobs.volume_detail)),
      haze_detail: clamp_haze_detail(self.haze_detail.min(knobs.haze_detail)),
      propagate_detail: clamp_propagate_detail(self.propagate_detail.min(knobs.propagate_detail)),
      reuse_px: 0,
    }
  }

  /// the log line's content: `[Dust] play quality on (volume 2, haze 4, trajectory 5)`
  pub fn describe(&self) -> alloc::string::String {
    alloc::format!(
      "volume {}, haze {}, trajectory {}",
      self.volume_detail, self.haze_detail, self.propagate_detail
    )
  }
}

/// The feedback biases a frame is pinned to while the simulation **runs** (handoff P3b). The
/// adaptive feedback (`DustViewHost::absorb_at`) walks the volume bias up 3 levels per readback
/// and never cuts the ray-integral evaluations below the per-packet cap until the bias is
/// top-only (`volume_top_only`: the volume and haze details act only there); inside 100 km of
/// 67P that is 4–5 readbacks of a 2.4 s frame each. While playing the frame starts on the
/// plateau instead: split depth [`Self::split_depth`], the volume bias at the pyramid's top-only
/// level (`levels − 1`, or [`Self::volume_bias`] when set) and the EWA bias
/// [`Self::ewa_bias`]. The paused feedback state is saved at the transition and restored at the
/// pause (`DustViewHost::play_pin_enter` / `play_pin_leave`): a paused frame never sees the pin.
///
/// `AETHERVK_DUST_PLAY_FEEDBACK`: unset or `1` → the default; `0` → `None` (no pin: the
/// adaptive feedback alone); `d,e[,v]` → split depth `d`, EWA bias `e` and (optional) a fixed
/// volume bias `v` instead of the top-only level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlayFeedback {
  /// the packet split's depth while playing (default 0: no split pass)
  pub split_depth: u32,
  /// the EWA bias while playing (default 3: 1/64 of the bias-0 texels; the feedback's own
  /// ceiling is [`super::DUST_EWA_BIAS_FEEDBACK_MAX`])
  pub ewa_bias: u32,
  /// a fixed volume bias (`None`: the top-only level of the view's pyramid)
  pub volume_bias: Option<u32>,
}

/// What [`PlayFeedback::pin`] resolves to for one view: the three header biases
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinnedFeedback {
  pub split_depth: u32,
  pub volume_bias: u32,
  pub ewa_bias: u32,
}

impl Default for PlayFeedback {
  fn default() -> Self {
    Self { split_depth: 0, ewa_bias: 3, volume_bias: None }
  }
}

impl PlayFeedback {
  pub fn from_env() -> Option<Self> {
    let v = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_PLAY_FEEDBACK");
    Self::parse(v.as_deref())
  }

  /// [`Self::from_env`] on a given value (tests)
  pub fn parse(v: Option<&str>) -> Option<Self> {
    let v = v.map(str::trim).unwrap_or("");
    match v {
      "" | "1" => Some(Self::default()),
      "0" => None,
      _ => {
        let mut it = v.split(',').map(|s| s.trim().parse::<u32>().ok());
        let d = Self::default();
        let split_depth = it.next().flatten().unwrap_or(d.split_depth).min(super::DUST_SPLIT_DEPTH_MAX);
        let ewa_bias = it.next().flatten().unwrap_or(d.ewa_bias).min(super::DUST_EWA_BIAS_FEEDBACK_MAX);
        let volume_bias = it.next().flatten().map(|b| b.min(super::DUST_VOLUME_BIAS_MAX));
        Some(Self { split_depth, ewa_bias, volume_bias })
      }
    }
  }

  /// The biases for a view whose pyramid has `levels` levels: the volume bias at the top-only
  /// level `levels − 1` (0 for a view without a pyramid yet) unless fixed
  pub fn pin(&self, levels: u32) -> PinnedFeedback {
    PinnedFeedback {
      split_depth: self.split_depth,
      volume_bias: self.volume_bias.unwrap_or(levels.saturating_sub(1)).min(super::DUST_VOLUME_BIAS_MAX),
      ewa_bias: self.ewa_bias,
    }
  }
}

/// What [`PlayStride`] aims at: the dust frame's GPU time and the stride's ceiling
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrideBudget {
  pub budget_ms: f64,
  pub max_stride: u32,
}

/// Readbacks under half the budget in a row before the stride comes down
pub const STRIDE_LOWER_AFTER: u32 = 16;

/// The preview's cluster stride `S` (a power of two): while previewing, the splat draws only the
/// live clusters whose ring slot is a multiple of `S`, each `S ×` brighter
/// (`Device::cmd_dust_splat`), so the frame's splat work drops about `S ×` while the image keeps
/// its brightness on average. A slot keeps its cluster for life, so the drawn set is the same
/// from frame to frame (no flicker), and a halved stride keeps every cluster drawn before. A
/// visual approximation of the preview only: the full render (paused) draws every cluster.
///
/// Fed with the GPU timing readbacks: a frame over 1.25 × the budget raises `S` at once to the
/// power of two the measured time asks for (`S · t / budget`); [`STRIDE_LOWER_AFTER`] readbacks
/// under half the budget lower it the same way. Readbacks of frames recorded before a change are
/// ignored (they measured the old stride).
///
/// Every play session ([`Self::play`] .. [`Self::end_session`]) starts at the preset's cap
/// ([`session_start_stride`]), so the first preview frame after Play does not draw every cluster
/// (a frame of several hundred ms that stalls the logic thread); the controller lowers it from
/// there. No stride is remembered across sessions: a low stride a light session settled on made
/// the next, heavier session's first frames slow. Outside a session the stride is 1: the paused
/// full render draws every cluster.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayStride {
  stride: u32,
  /// the frame recorded last when the stride changed: readbacks up to it are stale
  changed_at: u64,
  last_seen: u64,
  low_run: u32,
  /// a play session is running ([`Self::play`] since the last [`Self::end_session`])
  active: bool,
}

/// The stride a play session's first frame draws with: the budget's cap (a power of two), for
/// every session (no measurement of this session yet: the cheapest frame, so Play never stalls
/// on a full-density frame; the controller lowers it after [`STRIDE_LOWER_AFTER`] fast readbacks
/// per halving)
pub fn session_start_stride(b: StrideBudget) -> u32 {
  b.max_stride.max(1).next_power_of_two()
}

impl PlayStride {
  /// the stride in effect (1: every cluster)
  pub fn stride(&self) -> u32 {
    self.stride.max(1)
  }

  /// back to every cluster, no session running
  pub fn reset(&mut self) {
    *self = Self::default();
  }

  /// the play session ended (paused, or the preview off): back to every cluster; the next
  /// session starts at the cap again
  pub fn end_session(&mut self) {
    *self = Self::default();
  }

  /// A preview frame `now_frame` is about to be recorded under budget `b`: the first frame of a
  /// session starts at [`session_start_stride`] (readbacks up to `now_frame` measured the paused
  /// render and are ignored); later frames keep the stride under the cap (a preset changed while
  /// playing). Returns the new stride when it changed.
  pub fn play(&mut self, now_frame: u64, b: StrideBudget) -> Option<u32> {
    let max = b.max_stride.max(1).next_power_of_two();
    let next = if self.active {
      if self.stride() <= max {
        return None;
      }
      max
    } else {
      self.active = true;
      session_start_stride(b)
    };
    let changed = next != self.stride();
    self.stride = next;
    self.changed_at = now_frame;
    self.low_run = 0;
    changed.then_some(next)
  }

  /// A readback: frame `frame` took `total_ms`; `now_frame` the view's frame being recorded.
  /// Returns the new stride when it changed.
  pub fn absorb(&mut self, frame: u64, total_ms: f64, now_frame: u64, b: StrideBudget) -> Option<u32> {
    if frame <= self.changed_at || frame <= self.last_seen || !(total_ms > 0.0) || !total_ms.is_finite() {
      return None;
    }
    self.last_seen = frame;
    let s = self.stride();
    let max = b.max_stride.max(1).next_power_of_two();
    let wanted = |aim: f64| -> u32 {
      let x = (s as f64 * total_ms / aim).ceil().clamp(1.0, max as f64) as u32;
      x.next_power_of_two().min(max)
    };
    let next = if total_ms > 1.25 * b.budget_ms {
      self.low_run = 0;
      wanted(b.budget_ms).max(s)
    } else if total_ms < 0.5 * b.budget_ms && s > 1 {
      self.low_run += 1;
      if self.low_run < STRIDE_LOWER_AFTER {
        return None;
      }
      self.low_run = 0;
      // aim at 80 % of the budget, and never below half the stride at once
      wanted(0.8 * b.budget_ms).clamp((s / 2).max(1), s)
    } else {
      self.low_run = 0;
      s
    };
    if next == s {
      return None;
    }
    self.stride = next;
    self.changed_at = now_frame;
    Some(next)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn render_phase_follows_play_hold_and_full_render() {
    use DustRenderPhase::*;
    let none = i64::MIN;
    assert_eq!(dust_render_phase(10, true, true, none, none), Playing);
    // the hold fills from 0 to 1 over PREVIEW_HOLD_US
    let until = 1_000 + PREVIEW_HOLD_US;
    assert_eq!(dust_render_phase(1_000, false, true, until, none), Hold(0.0));
    assert_eq!(dust_render_phase(1_000 + PREVIEW_HOLD_US / 2, false, true, until, none), Hold(0.5));
    // the hold's end: the full render, busy while frames keep drawing
    let drew = until + 10;
    assert_eq!(dust_render_phase(until, false, true, until, none), Settled);
    assert_eq!(dust_render_phase(drew + 1_000_000, false, true, until, drew), Rendering);
    assert_eq!(dust_render_phase(drew + FULL_RENDER_BUSY_US, false, true, until, drew), Settled);
    // preview Off: no hold, the full render still reported, never "playing"
    assert_eq!(dust_render_phase(1_000, false, false, until, none), Settled);
    assert_eq!(dust_render_phase(drew + 5, false, false, until, drew), Rendering);
    assert_eq!(dust_render_phase(drew + 5, true, false, until, drew), Settled);
  }

  /// the play pin: top-only volume bias from the level count, the env forms, the clamps
  #[test]
  fn play_feedback_pin_and_env() {
    let d = PlayFeedback::default();
    assert_eq!(d.pin(11), PinnedFeedback { split_depth: 0, volume_bias: 10, ewa_bias: 3 });
    assert_eq!(d.pin(0).volume_bias, 0, "no pyramid yet: no coarsening to pin");
    assert_eq!(PlayFeedback::parse(None), Some(d));
    assert_eq!(PlayFeedback::parse(Some(" 1")), Some(d));
    assert_eq!(PlayFeedback::parse(Some("0")), None);
    let e = PlayFeedback::parse(Some("2,4,7")).unwrap();
    assert_eq!(e, PlayFeedback { split_depth: 2, ewa_bias: 4, volume_bias: Some(7) });
    assert_eq!(e.pin(11), PinnedFeedback { split_depth: 2, volume_bias: 7, ewa_bias: 4 });
    // clamps: the EWA bias never above the feedback's ceiling, the volume bias never above its max
    let c = PlayFeedback::parse(Some("9,9,99")).unwrap();
    assert_eq!(c.split_depth, crate::scene::dust::DUST_SPLIT_DEPTH_MAX);
    assert_eq!(c.ewa_bias, crate::scene::dust::DUST_EWA_BIAS_FEEDBACK_MAX);
    assert_eq!(c.volume_bias, Some(crate::scene::dust::DUST_VOLUME_BIAS_MAX));
  }
  use crate::scene::dust::{
    DUST_HAZE_DETAIL_DEFAULT, DUST_PROPAGATE_DETAIL_DEFAULT, DUST_VOLUME_DETAIL_DEFAULT,
  };

  fn scene_knobs() -> DustKnobs {
    DustKnobs {
      volume_detail: DUST_VOLUME_DETAIL_DEFAULT,
      haze_detail: DUST_HAZE_DETAIL_DEFAULT,
      propagate_detail: DUST_PROPAGATE_DETAIL_DEFAULT,
      reuse_px: 16,
    }
  }

  /// paused: the knobs pass through byte-identical (the byte gate runs paused); running: the
  /// play values, each at most the scene's own
  #[test]
  fn play_quality_mapping() {
    let pq = PlayQuality::default();
    let k = scene_knobs();
    assert_eq!(pq.apply(false, k), k, "paused: identical knobs");
    let odd = DustKnobs {
      volume_detail: 4,
      haze_detail: 2,
      propagate_detail: 9,
      reuse_px: 0,
    };
    assert_eq!(pq.apply(false, odd), odd, "paused: identical knobs (user's own)");
    let play = pq.apply(true, k);
    assert_eq!(
      play,
      DustKnobs {
        volume_detail: 2,
        haze_detail: 4,
        propagate_detail: 5,
        reuse_px: 0
      }
    );
    // never above the scene's own setting
    let play_odd = pq.apply(true, odd);
    assert_eq!(play_odd.volume_detail, 2);
    assert_eq!(play_odd.haze_detail, 2);
    assert_eq!(play_odd.propagate_detail, 5);
    assert_eq!(play_odd.reuse_px, 0);
  }

  /// the preview levels: Off has no play values, the others coarsen less and less; out of range
  /// is the default
  #[test]
  fn play_preview_levels() {
    assert_eq!(PlayPreview::from_level(0).quality(), None);
    assert_eq!(PlayPreview::from_level(2).quality(), Some(PlayQuality::default()));
    assert_eq!(PlayPreview::from_level(9), PlayPreview::from_level(PLAY_PREVIEW_DEFAULT));
    assert_eq!(PlayPreview::Off.stride_budget(), None);
    assert_eq!(PlayPreview::Balanced.stride_budget().unwrap().budget_ms, 33.0);
    let smooth = PlayPreview::Smooth.quality().unwrap();
    let quality = PlayPreview::Quality.quality().unwrap();
    let balanced = PlayQuality::default();
    assert!(smooth.volume_detail <= balanced.volume_detail && balanced.volume_detail <= quality.volume_detail);
    assert!(smooth.haze_detail <= balanced.haze_detail && balanced.haze_detail <= quality.haze_detail);
    // every preview: no temporal reuse while it lasts
    assert_eq!(smooth.apply(true, scene_knobs()).reuse_px, 0);
    assert_eq!(quality.apply(true, scene_knobs()).reuse_px, 0);
  }

  #[test]
  fn play_quality_env() {
    assert_eq!(PlayQuality::parse(None), Some(PlayQuality::default()));
    assert_eq!(PlayQuality::parse(Some("1")), Some(PlayQuality::default()));
    assert_eq!(PlayQuality::parse(Some(" 0 ")), None);
    assert_eq!(
      PlayQuality::parse(Some("8,8,9")),
      Some(PlayQuality {
        volume_detail: 8,
        haze_detail: 8,
        propagate_detail: 9
      })
    );
    // a partial / malformed override keeps the defaults for the rest
    assert_eq!(
      PlayQuality::parse(Some("4")),
      Some(PlayQuality {
        volume_detail: 4,
        ..PlayQuality::default()
      })
    );
  }

  /// The stride climbs at once to the power of two the measured frame asks for, ignores the
  /// readbacks of frames recorded before the change, holds inside the band, and comes down by
  /// halves only after a run of fast readbacks
  #[test]
  fn play_stride_follows_the_budget() {
    let b = StrideBudget { budget_ms: 33.0, max_stride: 32 };
    let mut p = PlayStride::default();
    assert_eq!(p.stride(), 1);
    // 670 ms at S = 1: 670 / 33 = 20.3 → 32 (the cap)
    assert_eq!(p.absorb(10, 670.0, 14, b), Some(32));
    // frames up to 14 measured the old stride: ignored
    assert_eq!(p.absorb(12, 670.0, 16, b), None);
    assert_eq!(p.absorb(14, 670.0, 16, b), None);
    assert_eq!(p.stride(), 32);
    // inside the band [0.5, 1.25] × budget: held
    assert_eq!(p.absorb(15, 30.0, 19, b), None);
    // fast frames: lowered only after STRIDE_LOWER_AFTER of them, by at most half
    let mut f = 16;
    let mut changed = None;
    for _ in 0..STRIDE_LOWER_AFTER {
      changed = p.absorb(f, 4.0, f + 4, b);
      f += 1;
    }
    assert_eq!(changed, Some(16));
    // never above the cap, never below 1; reset back to every cluster
    let mut q = PlayStride::default();
    assert_eq!(q.absorb(1, 1e6, 2, b), Some(32));
    q.reset();
    assert_eq!(q.stride(), 1);
    // a non-finite or zero time is no reading
    assert_eq!(q.absorb(5, f64::NAN, 6, b), None);
    assert_eq!(q.absorb(6, 0.0, 7, b), None);
  }

  /// Every play session's first frame draws at the preset's cap, whatever stride the previous
  /// session ended on (a low stride a light session settled on made the next, heavier session's
  /// first frames slow); the paused frames in between draw every cluster
  #[test]
  fn play_stride_every_session_starts_at_the_cap() {
    let b = StrideBudget { budget_ms: 33.0, max_stride: 32 };
    let mut p = PlayStride::default();
    let mut f = 10;
    assert_eq!(p.play(f, b), Some(32));
    // lowered to 4 by runs of fast readbacks from the cap
    for want in [16, 8, 4] {
      let mut changed = None;
      for _ in 0..STRIDE_LOWER_AFTER {
        f += 1;
        changed = p.absorb(f, 2.0, f + 4, b);
      }
      assert_eq!(changed, Some(want));
      f += 4;
    }
    assert_eq!(p.stride(), 4);
    p.end_session();
    assert_eq!(p.stride(), 1, "paused: every cluster");
    assert_eq!(p, PlayStride::default(), "nothing remembered");
    // paused frames change nothing
    p.end_session();
    p.end_session();
    assert_eq!(p.stride(), 1);
    f += 20;
    assert_eq!(p.play(f, b), Some(32), "the next session's first frame: the cap again");
    assert_eq!(p.stride(), 32);
    // readbacks of frames up to the session's start measured the paused render: ignored
    assert_eq!(p.absorb(f, 2_000.0, f + 1, b), None);
    assert_eq!(p.absorb(f - 3, 2_000.0, f + 1, b), None);
    assert_eq!(p.stride(), 32);
    // later frames: no change, and the controller adapts as before
    assert_eq!(p.play(f + 1, b), None);
    let mut changed = None;
    for g in f + 1..f + 1 + STRIDE_LOWER_AFTER as u64 {
      changed = p.absorb(g, 4.0, g + 4, b);
    }
    assert_eq!(changed, Some(16));
  }

  /// The first session ever (or after `reset`) starts at the preset's cap: no measurement yet,
  /// so the cheapest frame, never a full-density frame on Play; the controller then lowers it
  #[test]
  fn play_stride_first_session_starts_at_the_cap() {
    for (b, cap) in [
      (StrideBudget { budget_ms: 33.0, max_stride: 32 }, 32),
      (StrideBudget { budget_ms: 16.0, max_stride: 64 }, 64),
      (StrideBudget { budget_ms: 66.0, max_stride: 8 }, 8),
      (StrideBudget { budget_ms: 33.0, max_stride: 0 }, 1),
    ] {
      assert_eq!(session_start_stride(b), cap);
      let mut p = PlayStride::default();
      assert_eq!(p.stride(), 1);
      // paused frames before the first session change nothing
      p.end_session();
      assert_eq!(p, PlayStride::default());
      assert_eq!(p.play(3, b), (cap > 1).then_some(cap));
      assert_eq!(p.stride(), cap);
      // reset forgets the earlier sessions: the next starts as the first
      p.end_session();
      p.reset();
      assert_eq!(p, PlayStride::default());
      assert_eq!(p.play(9, b), (cap > 1).then_some(cap));
    }
    // the controller lowers the start after STRIDE_LOWER_AFTER fast readbacks
    let b = StrideBudget { budget_ms: 33.0, max_stride: 32 };
    let mut p = PlayStride::default();
    p.play(100, b);
    let mut changed = None;
    for f in 101..101 + STRIDE_LOWER_AFTER as u64 {
      changed = p.absorb(f, 4.0, f + 4, b);
    }
    assert_eq!(changed, Some(16));
  }

  /// A preset change clamps: a session starts at its own preset's cap, and a preset lowered
  /// while playing brings the stride under the cap on the next frame
  #[test]
  fn play_stride_preset_change_clamps() {
    let smooth = StrideBudget { budget_ms: 16.0, max_stride: 64 };
    let quality = StrideBudget { budget_ms: 66.0, max_stride: 8 };
    let balanced = StrideBudget { budget_ms: 33.0, max_stride: 32 };
    assert_eq!(session_start_stride(quality), 8);
    assert_eq!(session_start_stride(balanced), 32);
    assert_eq!(session_start_stride(smooth), 64);
    assert_eq!(session_start_stride(StrideBudget { budget_ms: 33.0, max_stride: 5 }), 8, "a power of two");
    // a Smooth session at 64, then Play under Quality: 8, not 64
    let mut p = PlayStride::default();
    assert_eq!(p.play(1, smooth), Some(64));
    p.end_session();
    assert_eq!(p.play(10, quality), Some(8));
    // a session under a higher cap starts at that cap, not at the previous session's stride
    p.end_session();
    assert_eq!(p.play(20, smooth), Some(64));
    // the preset changed while playing: clamped at once, and readbacks up to the change ignored
    let mut q = PlayStride::default();
    assert_eq!(q.play(1, smooth), Some(64));
    assert_eq!(q.play(2, smooth), None);
    assert_eq!(q.play(3, quality), Some(8));
    assert_eq!(q.stride(), 8);
    assert_eq!(q.absorb(3, 2_000.0, 4, quality), None);
    assert_eq!(q.play(4, balanced), None, "under the new cap: kept");
    q.end_session();
    assert_eq!(q.play(9, balanced), Some(32));
  }
}
