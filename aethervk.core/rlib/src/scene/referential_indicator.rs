use crate::scene::{Component, EntityId, text::FontAtlas};
use alloc::{string::String, sync::Arc};

/// A HUD indicator whose 3D target is derived each frame from a live entity's
/// global transform. Position is never stored — it is re-queried via `compute_rte`
/// on every render frame, so the label tracks moving bodies exactly.
#[derive(Debug, Clone)]
pub struct ReferentialIndicatorComponent {
  pub target_entity: EntityId,
  pub label: String,
  pub text_color: [f32; 4],
  pub desired_label_distance_km: f64,
  pub font_atlas: Arc<FontAtlas>,
  pub font_hash: u64,
}

impl Component for ReferentialIndicatorComponent {}
