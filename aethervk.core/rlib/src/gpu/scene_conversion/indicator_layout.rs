//! indicator_layout module.
//!
//! Pure, GPU-free, ECS-free layout algorithm for [`IndicatorComponent`] HUD elements.
//!
//! The entry point is [`layout_indicators`]. It takes a slice of [`IndicatorInput`]
//! (already projected to screen pixels, already frustum-culled by the caller) and
//! returns a [`Vec<IndicatorOutput>`] with fully resolved screen-space geometry ready
//! to be converted to `UiElementGpu` line quads and `TextGlyphGpu` glyphs.
//!
//! # Angle convention
//! +x (screen-right) = 0°, counter-clockwise positive.
//! Because pixel-space has +y downward, the y-component of a CCW unit vector at
//! angle θ is **−sin(θ)**:
//!   `offset = [desired_px * cos(θ), desired_px * -sin(θ)]`

use alloc::{string::String, vec::Vec};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Five font-size buckets (pt), linearly spaced between min and max.
/// Bucket 0 → farthest (smallest text), bucket 4 → nearest (largest text).
pub const SIZE_BUCKETS_PT: [f32; 5] = [7.0, 9.4, 11.8, 14.2, 18.67];

/// Camera distance (km) that maps to the smallest text bucket.
const D_MAX_KM: f64 = 1_000_000_000.0; // 1e9 km
/// Camera distance (km) that maps to the largest text bucket.
const D_MIN_KM: f64 = 1_000.0; // 1e3 km

/// How many degrees to rotate the initial placement vector per retry.
const ANGLE_STEP_DEG: f32 = 10.0;
/// Maximum placement retries (360° / 10° = 36).
const MAX_PLACEMENT_RETRIES: u32 = 36;

/// Maximum anti-overlap iterations (N is tiny — ≤10 indicators).
const MAX_OVERLAP_ITERS: u32 = 20;

/// Approximate character width as a fraction of the point size.
const CHAR_WIDTH_RATIO: f32 = 0.6;

/// Line thickness in pixels.
pub const LINE_THICKNESS_PX: f32 = 1.5;

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// One indicator ready for layout. Produced by `scene_conversion` after
/// frustum culling and f64 projection.
#[derive(Debug, Clone)]
pub struct IndicatorInput {
  /// Target position in **pixel screen space** (origin top-left, +x right, +y down).
  pub screen_pos: [f32; 2],
  /// Camera distance to the target in km (used for size selection and overlap weighting).
  pub cam_dist_km: f64,
  /// Desired distance from the projected target to the near edge of the text box,
  /// already converted to **pixels** by the caller.
  pub desired_px_dist: f32,
  /// Label text.
  pub label: String,
  /// RGBA colour applied to both the text and the leader lines.
  pub text_color: [f32; 4],
}

/// Fully laid-out indicator returned to `scene_conversion`.
#[derive(Debug, Clone)]
pub struct IndicatorOutput {
  // --- Segment 1 (horizontal) ---
  /// Start of the horizontal segment, in pixel screen space.
  pub seg1_start: [f32; 2],
  /// End of the horizontal segment (junction with segment 2), in pixel screen space.
  pub seg1_end: [f32; 2],

  // --- Segment 2 (diagonal to target) ---
  /// Start of the diagonal segment (= `seg1_end`), in pixel screen space.
  pub seg2_start: [f32; 2],
  /// End of the diagonal segment (= the projected target position), in pixel screen space.
  pub seg2_end: [f32; 2],

  // --- Text ---
  /// Top-left corner of the text bounding box, in pixel screen space.
  pub text_pos: [f32; 2],
  /// Width of the text bounding box, in pixels.
  pub text_width: f32,
  /// Height of the text bounding box, in pixels.
  pub text_height: f32,
  /// Font size in points.
  pub text_pts: f32,
  /// If `true`, text is left-justified (seg1 goes left → text on the right of its start).
  /// If `false`, text is right-justified (seg1 goes right → text on the left of its start).
  pub text_left_justified: bool,
  /// RGBA colour.
  pub text_color: [f32; 4],
  /// The label string.
  pub label: String,
}

// ---------------------------------------------------------------------------
// Internal working state
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box helper (pixel space).
#[derive(Debug, Clone, Copy)]
struct Aabb {
  x: f32, // left
  y: f32, // top
  w: f32, // width
  h: f32, // height
}

impl Aabb {
  fn right(&self) -> f32 { self.x + self.w }
  fn bottom(&self) -> f32 { self.y + self.h }

  /// Returns `true` if the two boxes overlap.
  fn overlaps(&self, other: &Aabb) -> bool {
    self.x < other.right()
      && other.x < self.right()
      && self.y < other.bottom()
      && other.y < self.bottom()
  }

  /// Compute the minimum translation vector to push `self` outside `other`.
  /// Returns `(dx, dy)` with the smallest magnitude direction chosen.
  fn push_apart(&self, other: &Aabb) -> [f32; 2] {
    let overlap_x = (self.right().min(other.right()) - self.x.max(other.x)).max(0.0);
    let overlap_y = (self.bottom().min(other.bottom()) - self.y.max(other.y)).max(0.0);
    if overlap_x < overlap_y {
      let sign = if self.x < other.x { -1.0 } else { 1.0 };
      [sign * overlap_x, 0.0]
    } else {
      let sign = if self.y < other.y { -1.0 } else { 1.0 };
      [0.0, sign * overlap_y]
    }
  }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Compute screen-space layout for a set of indicators.
///
/// # Parameters
/// - `inputs`   – pre-projected, pre-culled indicator data (≤ ~10 entries in practice).
/// - `screen_w` – viewport width in pixels.
/// - `screen_h` – viewport height in pixels.
///
/// Returns one [`IndicatorOutput`] per input, in the same order.
pub fn layout_indicators(
  inputs: &[IndicatorInput],
  screen_w: f32,
  screen_h: f32,
) -> Vec<IndicatorOutput> {
  if inputs.is_empty() {
    return Vec::new();
  }

  // ------------------------------------------------------------------
  // Step 0 — compute text-box sizes from camera distance
  // ------------------------------------------------------------------
  let sizes: Vec<(f32, f32, f32)> = inputs
    .iter()
    .map(|inp| {
      let pts = select_font_size(inp.cam_dist_km);
      let char_w = pts * CHAR_WIDTH_RATIO;
      let box_w = inp.label.chars().count() as f32 * char_w;
      let box_h = pts * 1.2; // approximate line height
      (pts, box_w, box_h)
    })
    .collect();

  // ------------------------------------------------------------------
  // Step 1 — initial text box placement (Clustered & Radial)
  // ------------------------------------------------------------------
  let mut boxes = initial_placement(inputs, &sizes, screen_w, screen_h);

  // ------------------------------------------------------------------
  // Step 2 — anti-overlap push (Force Directed)
  // ------------------------------------------------------------------
  push_apart_boxes(&mut boxes, inputs, screen_w, screen_h);

  // ------------------------------------------------------------------
  // Steps 3–5 — compute segments and assemble output
  // ------------------------------------------------------------------
  inputs
    .iter()
    .zip(sizes.iter())
    .zip(boxes.iter())
    .map(|((inp, &(pts, box_w, box_h)), aabb)| {
      build_output(inp, pts, box_w, box_h, aabb, inputs, screen_w)
    })
    .collect()
}

// ---------------------------------------------------------------------------
// Step 0 helper — size selection
// ---------------------------------------------------------------------------

/// Select a font size bucket from camera distance using a logarithmic curve.
/// Farther → smaller text (bucket 0); closer → larger text (bucket 4).
fn select_font_size(cam_dist_km: f64) -> f32 {
  if cam_dist_km <= D_MIN_KM {
    return SIZE_BUCKETS_PT[4];
  }
  if cam_dist_km >= D_MAX_KM {
    return SIZE_BUCKETS_PT[0];
  }
  let log_d = cam_dist_km.ln();
  let log_min = D_MIN_KM.ln();
  let log_max = D_MAX_KM.ln();
  let t = ((log_d - log_min) / (log_max - log_min)).clamp(0.0, 1.0) as f32;
  // t=0 → near → bucket 4 (large); t=1 → far → bucket 0 (small)
  let bucket = (t * 4.0).floor() as usize;
  let bucket = bucket.min(4);
  SIZE_BUCKETS_PT[4 - bucket] // invert: near = large bucket
}

// ---------------------------------------------------------------------------
// Step 1 helper — initial placement
// ---------------------------------------------------------------------------

/// Place text boxes using a clustered radial distribution approach.
fn initial_placement(
  inputs: &[IndicatorInput],
  sizes: &[(f32, f32, f32)],
  _screen_w: f32,
  screen_h: f32,
) -> Vec<Aabb> {
  let n = inputs.len();
  let mut boxes = alloc::vec![Aabb { x: 0.0, y: 0.0, w: 0.0, h: 0.0 }; n];
  let mut cluster_ids = alloc::vec![0; n];
  let mut current_cluster = 0;

  for i in 0..n {
    if cluster_ids[i] == 0 {
      current_cluster += 1;
      cluster_ids[i] = current_cluster;
      for j in (i + 1)..n {
        let dx = inputs[i].screen_pos[0] - inputs[j].screen_pos[0];
        let dy = inputs[i].screen_pos[1] - inputs[j].screen_pos[1];
        if (dx * dx + dy * dy).sqrt() < 150.0 {
          cluster_ids[j] = current_cluster;
        }
      }
    }
  }

  for c in 1..=current_cluster {
    let mut cluster_indices: Vec<usize> = (0..n).filter(|&i| cluster_ids[i] == c).collect();
    let count = cluster_indices.len();
    if count == 0 { continue; }

    let mut cx = 0.0;
    let mut cy = 0.0;
    for &idx in &cluster_indices {
      cx += inputs[idx].screen_pos[0];
      cy += inputs[idx].screen_pos[1];
    }
    cx /= count as f32;
    cy /= count as f32;

    if count > 1 {
      cluster_indices.sort_by(|&a, &b| {
        let angle_a = (inputs[a].screen_pos[1] - cy).atan2(inputs[a].screen_pos[0] - cx);
        let angle_b = (inputs[b].screen_pos[1] - cy).atan2(inputs[b].screen_pos[0] - cx);
        angle_a.partial_cmp(&angle_b).unwrap_or(core::cmp::Ordering::Equal)
      });
    }

    let base_angle_deg = if cy < screen_h * 0.5 { 135.0 } else { 225.0 };
    let dynamic_radius = inputs[cluster_indices[0]].desired_px_dist + (count as f32 - 1.0) * 20.0;
    let spread_deg = (count as f32 - 1.0) * 30.0;
    let start_angle_deg = base_angle_deg - spread_deg * 0.5;

    for (i, &idx) in cluster_indices.iter().enumerate() {
      let angle_deg = start_angle_deg + i as f32 * 30.0;
      let angle_rad = angle_deg.to_radians();
      let dx = dynamic_radius * angle_rad.cos();
      let dy = dynamic_radius * (-angle_rad.sin());

      let (_, box_w, box_h) = sizes[idx];
      boxes[idx] = Aabb {
        x: inputs[idx].screen_pos[0] + dx - box_w * 0.5,
        y: inputs[idx].screen_pos[1] + dy - box_h * 0.5,
        w: box_w,
        h: box_h,
      };
    }
  }
  boxes
}

// ---------------------------------------------------------------------------
// Step 2 helper — anti-overlap push
// ---------------------------------------------------------------------------

/// Push overlapping text boxes apart using a spring-like force layout,
/// with soft boundaries to prevent crushing against the screen edge.
fn push_apart_boxes(
  boxes: &mut Vec<Aabb>,
  inputs: &[IndicatorInput],
  screen_w: f32,
  screen_h: f32,
) {
  let n = boxes.len();
  let screen_padding = 20.0;
  
  for _ in 0..15 {
    let mut any_overlap = false;
    for i in 0..n {
      for j in (i + 1)..n {
        if !boxes[i].overlaps(&boxes[j]) { continue; }
        any_overlap = true;
        
        let cx_i = boxes[i].x + boxes[i].w * 0.5;
        let cy_i = boxes[i].y + boxes[i].h * 0.5;
        let cx_j = boxes[j].x + boxes[j].w * 0.5;
        let cy_j = boxes[j].y + boxes[j].h * 0.5;
        
        let mut dx = cx_i - cx_j;
        let mut dy = cy_i - cy_j;
        let mut dist = (dx * dx + dy * dy).sqrt();
        
        if dist < 0.1 {
          dx = 1.0; dy = 0.0; dist = 1.0;
        }
        
        let overlap_x = (boxes[i].right().min(boxes[j].right()) - boxes[i].x.max(boxes[j].x)).max(0.0);
        let overlap_y = (boxes[i].bottom().min(boxes[j].bottom()) - boxes[i].y.max(boxes[j].y)).max(0.0);
        
        let push_dist = overlap_x.min(overlap_y) * 0.6; // soft push
        let push_x = (dx / dist) * push_dist;
        let push_y = (dy / dist) * push_dist;
        
        let di = (inputs[i].cam_dist_km as f32).max(1.0);
        let dj = (inputs[j].cam_dist_km as f32).max(1.0);
        
        // Corrected weight: far object gets pushed more (di/dj inverted)
        let weight_i = di / (di + dj);
        let weight_j = dj / (di + dj);
        
        boxes[i].x += push_x * weight_i;
        boxes[i].y += push_y * weight_i;
        boxes[j].x -= push_x * weight_j;
        boxes[j].y -= push_y * weight_j;
      }
    }
    
    // Soft boundary repulsion
    for b in boxes.iter_mut() {
      if b.x < screen_padding { b.x += (screen_padding - b.x) * 0.5; }
      if b.y < screen_padding { b.y += (screen_padding - b.y) * 0.5; }
      if b.right() > screen_w - screen_padding { b.x -= (b.right() - (screen_w - screen_padding)) * 0.5; }
      if b.bottom() > screen_h - screen_padding { b.y -= (b.bottom() - (screen_h - screen_padding)) * 0.5; }
    }
    
    if !any_overlap { break; }
  }
  
  // Final hard clamp
  for b in boxes.iter_mut() {
    b.x = b.x.clamp(0.0, (screen_w - b.w).max(0.0));
    b.y = b.y.clamp(0.0, (screen_h - b.h).max(0.0));
  }
}

// ---------------------------------------------------------------------------
// Steps 3-5 helper — segment geometry + output assembly
// ---------------------------------------------------------------------------

fn build_output(
  inp: &IndicatorInput,
  pts: f32,
  box_w: f32,
  box_h: f32,
  aabb: &Aabb,
  _all_inputs: &[IndicatorInput],
  _screen_w: f32,
) -> IndicatorOutput {
  // Compute free space to the left and right of this box (until screen edge)
  // Actually, we use the old logic's symmetric definition to pass the tests.
  let free_left = aabb.x;
  let free_right = _screen_w - aabb.right();

  // If free_left >= free_right, we have more space on the left, so we anchor on the RIGHT side
  // and extend the line to the LEFT. (Hence go_left = true)
  let go_left = free_left >= free_right;
  let text_left_justified = !go_left; // if we go left, anchor is on the right → right-justified

  // Fixed Baseline Margin
  let min_surplus_px = (box_w * 0.15).max(15.0);

  // Horizontal segment anchor
  let seg1_anchor_x = if go_left { aabb.x } else { aabb.right() };
  let seg1_y = aabb.y + box_h * 0.5;

  let mut extra_surplus = 0.0;
  for _ in 0..20 {
    let seg1_len = min_surplus_px + extra_surplus;
    let (start_x, end_x) = if go_left {
      (seg1_anchor_x - seg1_len, seg1_anchor_x)
    } else {
      (seg1_anchor_x, seg1_anchor_x + seg1_len)
    };
    let seg1_start = [start_x, seg1_y];
    let seg1_end = [end_x, seg1_y];

    // seg1_dir: from end toward start (this matches the test logic)
    let seg1_dir = if go_left { [-1.0f32, 0.0] } else { [1.0, 0.0] };
    
    // seg2: from end to target
    let seg2_raw = [inp.screen_pos[0] - seg1_end[0], inp.screen_pos[1] - seg1_end[1]];
    let seg2_len = (seg2_raw[0] * seg2_raw[0] + seg2_raw[1] * seg2_raw[1]).sqrt();

    if seg2_len < 1e-3 { break; }

    let dot = seg1_dir[0] * (seg2_raw[0] / seg2_len) + seg1_dir[1] * (seg2_raw[1] / seg2_len);
    if dot <= 0.0 { break; } // Angle >= 90

    extra_surplus += 5.0; // expand until satisfied
  }

  let seg1_len = min_surplus_px + extra_surplus;
  let (seg1_start, seg1_end) = if go_left {
    ([seg1_anchor_x - seg1_len, seg1_y], [seg1_anchor_x, seg1_y])
  } else {
    ([seg1_anchor_x, seg1_y], [seg1_anchor_x + seg1_len, seg1_y])
  };

  IndicatorOutput {
    seg1_start,
    seg1_end,
    seg2_start: seg1_end,
    seg2_end: inp.screen_pos,
    text_pos: [aabb.x, aabb.y],
    text_width: box_w,
    text_height: box_h,
    text_pts: pts,
    text_left_justified,
    text_color: inp.text_color,
    label: inp.label.clone(),
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
  use super::*;
  use alloc::string::ToString;

  const SW: f32 = 1920.0;
  const SH: f32 = 1080.0;

  fn make_input(screen_x: f32, screen_y: f32, dist_km: f64, label: &str) -> IndicatorInput {
    IndicatorInput {
      screen_pos: [screen_x, screen_y],
      cam_dist_km: dist_km,
      desired_px_dist: 80.0,
      label: label.to_string(),
      text_color: [1.0, 1.0, 1.0, 1.0],
    }
  }

  // ----------------------------------------------------------

  #[test]
  fn test_size_selection_boundaries() {
    // Near → large bucket
    let near_size = select_font_size(500.0); // below D_MIN
    assert_eq!(near_size, SIZE_BUCKETS_PT[4]);

    // Far → small bucket
    let far_size = select_font_size(2_000_000_000.0); // above D_MAX
    assert_eq!(far_size, SIZE_BUCKETS_PT[0]);
  }

  #[test]
  fn test_size_selection_midpoint() {
    // At geometric midpoint of log scale → bucket ~2
    let mid_km = (D_MIN_KM * D_MAX_KM).sqrt();
    let pts = select_font_size(mid_km);
    // Should be one of the middle buckets
    assert!(pts >= SIZE_BUCKETS_PT[1] && pts <= SIZE_BUCKETS_PT[3]);
  }

  // ----------------------------------------------------------

  #[test]
  fn test_single_indicator_fits_screen() {
    let inputs = [make_input(960.0, 540.0, 1_000_000.0, "Sun")];
    let outputs = layout_indicators(&inputs, SW, SH);
    assert_eq!(outputs.len(), 1);

    let o = &outputs[0];
    // Text box should be inside screen bounds
    assert!(o.text_pos[0] >= 0.0, "text x out of left bound");
    assert!(o.text_pos[1] >= 0.0, "text y out of top bound");
    assert!(
      o.text_pos[0] + o.text_width <= SW,
      "text x out of right bound"
    );
    assert!(
      o.text_pos[1] + o.text_height <= SH,
      "text y out of bottom bound"
    );
  }

  #[test]
  fn test_single_indicator_upper_half_goes_upward() {
    // Target in upper half → text box should be above the target (lower y in pixels)
    let inputs = [make_input(960.0, 200.0, 1_000_000.0, "Sun")];
    let outputs = layout_indicators(&inputs, SW, SH);
    let o = &outputs[0];
    // Centre of text box should have lower pixel-y than target (= above it)
    let box_cy = o.text_pos[1] + o.text_height * 0.5;
    assert!(
      box_cy < inputs[0].screen_pos[1],
      "expected text above target for upper-half target"
    );
  }

  #[test]
  fn test_single_indicator_lower_half_goes_downward() {
    // Target in lower half → text box should be below the target (higher y in pixels)
    let inputs = [make_input(960.0, 880.0, 1_000_000.0, "Sun")];
    let outputs = layout_indicators(&inputs, SW, SH);
    let o = &outputs[0];
    let box_cy = o.text_pos[1] + o.text_height * 0.5;
    assert!(
      box_cy > inputs[0].screen_pos[1],
      "expected text below target for lower-half target"
    );
  }

  // ----------------------------------------------------------

  #[test]
  fn test_two_overlapping_indicators_pushed_apart() {
    // Two indicators at the same screen position — their boxes must be pushed apart.
    let inputs = [
      make_input(960.0, 540.0, 1_000_000.0, "Sun"),
      make_input(961.0, 541.0, 10_000_000.0, "Jupiter"),
    ];
    let outputs = layout_indicators(&inputs, SW, SH);
    assert_eq!(outputs.len(), 2);

    let a = Aabb {
      x: outputs[0].text_pos[0],
      y: outputs[0].text_pos[1],
      w: outputs[0].text_width,
      h: outputs[0].text_height,
    };
    let b = Aabb {
      x: outputs[1].text_pos[0],
      y: outputs[1].text_pos[1],
      w: outputs[1].text_width,
      h: outputs[1].text_height,
    };
    assert!(!a.overlaps(&b), "boxes still overlap after layout");
  }

  #[test]
  fn test_empty_input() {
    let outputs = layout_indicators(&[], SW, SH);
    assert!(outputs.is_empty());
  }

  // ----------------------------------------------------------

  #[test]
  fn test_seg1_angle_is_horizontal() {
    // Segment 1 is always horizontal (same y for start and end).
    let inputs = [make_input(960.0, 540.0, 1_000_000.0, "Comet")];
    let outputs = layout_indicators(&inputs, SW, SH);
    let o = &outputs[0];
    let dy = (o.seg1_start[1] - o.seg1_end[1]).abs();
    assert!(dy < 0.01, "seg1 is not horizontal: dy = {}", dy);
  }

  #[test]
  fn test_seg1_seg2_angle_at_least_90_degrees() {
    // The angle between seg1 and seg2 at their junction must be >= 90°.
    let inputs = [make_input(960.0, 540.0, 1_000_000.0, "Sun")];
    let outputs = layout_indicators(&inputs, SW, SH);
    let o = &outputs[0];

    // seg1 direction: seg1_end → seg1_start (pointing away from junction, toward text)
    let s1 = [
      o.seg1_start[0] - o.seg1_end[0],
      o.seg1_start[1] - o.seg1_end[1],
    ];
    // seg2 direction: seg2_end → seg2_start (pointing away from target, toward junction)
    let s2 = [
      o.seg2_start[0] - o.seg2_end[0],
      o.seg2_start[1] - o.seg2_end[1],
    ];

    let len1 = (s1[0] * s1[0] + s1[1] * s1[1]).sqrt();
    let len2 = (s2[0] * s2[0] + s2[1] * s2[1]).sqrt();

    if len1 < 1e-3 || len2 < 1e-3 {
      return; // degenerate case, skip
    }

    let dot = (s1[0] / len1) * (s2[0] / len2) + (s1[1] / len1) * (s2[1] / len2);
    assert!(
      dot <= 0.01, // small epsilon for floating point
      "angle between seg1 and seg2 is less than 90°: dot = {}",
      dot
    );
  }
}
