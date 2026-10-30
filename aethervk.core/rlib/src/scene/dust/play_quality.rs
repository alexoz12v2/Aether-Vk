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

use super::{
  DUST_HAZE_DETAIL_MIN, DUST_PROPAGATE_DETAIL_MIN, DUST_VOLUME_DETAIL_MIN, clamp_haze_detail,
  clamp_propagate_detail, clamp_reuse_px, clamp_volume_detail,
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

  /// The knobs a frame draws with: `running` → the play values, never above the scene's own
  /// (a user who set a coarser detail keeps it); not running → `knobs` **unchanged** (every
  /// paused frame, the byte gate's included, draws with the scene's knobs).
  pub fn apply(&self, running: bool, knobs: DustKnobs) -> DustKnobs {
    if !running {
      return knobs;
    }
    DustKnobs {
      volume_detail: clamp_volume_detail(self.volume_detail.min(knobs.volume_detail)),
      haze_detail: clamp_haze_detail(self.haze_detail.min(knobs.haze_detail)),
      propagate_detail: clamp_propagate_detail(self.propagate_detail.min(knobs.propagate_detail)),
      reuse_px: clamp_reuse_px(knobs.reuse_px),
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

#[cfg(test)]
mod tests {
  use super::*;

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
        reuse_px: 16
      }
    );
    // never above the scene's own setting
    let play_odd = pq.apply(true, odd);
    assert_eq!(play_odd.volume_detail, 2);
    assert_eq!(play_odd.haze_detail, 2);
    assert_eq!(play_odd.propagate_detail, 5);
    assert_eq!(play_odd.reuse_px, 0);
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
}
