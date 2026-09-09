//! indicator module — IndicatorComponent

use crate::scene::Component;
use aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64;
use alloc::{string::String, sync::Arc};

/// A fixed-position HUD indicator that renders a labelled leader-line pointing
/// to a world-space location.
///
/// On each render frame `scene_conversion` reads all visible `IndicatorComponent`
/// entities, frustum-culls them, runs the heuristic anti-overlap layout pass and
/// emits the resulting line quads + text glyphs into the existing `UiElementGpu` /
/// `TextGlyphGpu` batches. No new shader or archetype is required.
///
/// The component deliberately carries no GPU types. The font atlas `Arc` is
/// lightweight (same pattern as `ScreenSpaceTextComponent`) and is simply shared
/// by reference during the render pass.
///
/// # Units
/// All positions are in **kilometres** in the heliocentric ecliptic J2000 frame
/// (same as `HighResTransformComponent.position`).
#[derive(Debug, Clone)]
pub struct IndicatorComponent {
  /// World-space target position in kilometres.
  pub global_position_km: Vec3f64,

  /// Label text displayed next to the indicator.
  pub label: String,

  /// RGBA text/line colour, each channel in `[0.0, 1.0]`.
  pub text_color: [f32; 4],

  /// Desired distance between the projected target pixel and the near edge of the
  /// text box, in kilometres. Converted to screen pixels by the layout pass using
  /// the current projection scale.
  ///
  /// The layout pass may move the box further away to avoid overlaps with other
  /// indicators or screen bounds.
  pub desired_label_distance_km: f64,

  /// Font atlas used to rasterise the label.  Shared by `Arc` — no extra cost.
  pub font_atlas: Arc<crate::scene::text::FontAtlas>,

  /// Stable hash of the font atlas metadata, used as a GPU descriptor cache key.
  pub font_hash: u64,
}

impl Component for IndicatorComponent {}
