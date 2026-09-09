use crate::scene::{Component, EntityId, text::FontAtlas};
use alloc::{string::String, sync::Arc};
use core::sync::atomic::{AtomicU64, Ordering};

/// A HUD indicator that finds and labels the best visible point on a
/// `TrajectoryComponent` curve, with temporal smoothing to prevent jitter.
#[derive(Debug, Clone)]
pub struct TrajectoryIndicatorComponent {
  /// Entity carrying the `TrajectoryComponent` to sample.
  pub target_entity: EntityId,
  pub label: String,
  pub text_color: [f32; 4],
  pub font_atlas: Arc<FontAtlas>,
  pub font_hash: u64,
  /// Smoothed Bézier parameter `t` (segment_idx + local_t) stored as f64 bits.
  /// `u64::MAX` = uninitialized (snap on first frame).
  /// Wrapped in `Arc` so `Clone` copies the handle, not the atom — both the
  /// scene owner and the render thread see the same value.
  pub current_t_bits: Arc<AtomicU64>,
}

impl TrajectoryIndicatorComponent {
  pub fn new(
    target_entity: EntityId,
    label: String,
    text_color: [f32; 4],
    font_atlas: Arc<FontAtlas>,
    font_hash: u64,
  ) -> Self {
    Self {
      target_entity,
      label,
      text_color,
      font_atlas,
      font_hash,
      current_t_bits: Arc::new(AtomicU64::new(u64::MAX)),
    }
  }

  pub fn get_current_t(&self) -> Option<f64> {
    let bits = self.current_t_bits.load(Ordering::Relaxed);
    if bits == u64::MAX {
      None
    } else {
      Some(f64::from_bits(bits))
    }
  }

  pub fn set_current_t(&self, t: f64) {
    self.current_t_bits.store(t.to_bits(), Ordering::Relaxed);
  }
}

impl Component for TrajectoryIndicatorComponent {}
