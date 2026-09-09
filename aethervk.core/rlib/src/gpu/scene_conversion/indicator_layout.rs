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
  // Step 1 — initial text box placement
  // ------------------------------------------------------------------
  let mut boxes: Vec<Aabb> = inputs
    .iter()
    .enumerate()
    .zip(sizes.iter())
    .map(|((idx, inp), &(_, box_w, box_h))| {
      place_text_box(inp, idx, box_w, box_h, screen_w, screen_h)
    })
    .collect();

  // ------------------------------------------------------------------
  // Step 2 — anti-overlap push
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

/// Place a text box starting from the projected target with the given desired
/// offset distance. Each indicator gets an additional angle offset of `idx * 20°`
/// so that coincident targets spread out immediately without relying solely on the
/// push-apart pass. Tries CCW rotations in 10° increments until the box fits
/// inside the screen, or all 36 attempts are exhausted.
fn place_text_box(
  inp: &IndicatorInput,
  idx: usize,
  box_w: f32,
  box_h: f32,
  screen_w: f32,
  screen_h: f32,
) -> Aabb {
  // Initial angle: upper half → 135° (up-left), lower half → 225° (down-left).
  // Each indicator is additionally staggered by 20° × idx to prevent coincident
  // targets from producing overlapping initial placements.
  let base_angle_deg: f32 = if inp.screen_pos[1] < screen_h * 0.5 {
    135.0
  } else {
    225.0
  };
  let initial_angle_deg = base_angle_deg + idx as f32 * 20.0;


  let mut best = None::<Aabb>;

  for step in 0..MAX_PLACEMENT_RETRIES {
    let angle_deg = initial_angle_deg + step as f32 * ANGLE_STEP_DEG;
    let angle_rad = angle_deg.to_radians();
    let dx = inp.desired_px_dist * angle_rad.cos();
    let dy = inp.desired_px_dist * (-angle_rad.sin()); // flip because pixel-y is down

    // Centre of the text box
    let cx = inp.screen_pos[0] + dx;
    let cy = inp.screen_pos[1] + dy;
    let candidate = Aabb {
      x: cx - box_w * 0.5,
      y: cy - box_h * 0.5,
      w: box_w,
      h: box_h,
    };

    if fits_screen(&candidate, screen_w, screen_h) {
      return candidate;
    }

    // Keep track of last tried in case none fits
    if best.is_none() {
      best = Some(candidate);
    }
  }

  // Fallback: clamp the best candidate inside screen bounds
  let mut aabb = best.unwrap_or(Aabb {
    x: inp.screen_pos[0] - box_w * 0.5,
    y: inp.screen_pos[1] - box_h * 0.5,
    w: box_w,
    h: box_h,
  });
  clamp_to_screen(&mut aabb, screen_w, screen_h);
  aabb
}

fn fits_screen(aabb: &Aabb, screen_w: f32, screen_h: f32) -> bool {
  aabb.x >= 0.0 && aabb.y >= 0.0 && aabb.right() <= screen_w && aabb.bottom() <= screen_h
}

fn clamp_to_screen(aabb: &mut Aabb, screen_w: f32, screen_h: f32) {
  aabb.x = aabb.x.clamp(0.0, (screen_w - aabb.w).max(0.0));
  aabb.y = aabb.y.clamp(0.0, (screen_h - aabb.h).max(0.0));
}

// ---------------------------------------------------------------------------
// Step 2 helper — anti-overlap push
// ---------------------------------------------------------------------------

/// Push overlapping text boxes apart.
///
/// The push amount for each box is weighted by:
/// - Distance (closer → pushed less) — 60% weight
/// - Label length (longer → pushed less) — 40% weight
///
/// Both boxes are clamped to screen bounds after each push.
fn push_apart_boxes(
  boxes: &mut Vec<Aabb>,
  inputs: &[IndicatorInput],
  screen_w: f32,
  screen_h: f32,
) {
  let n = boxes.len();
  for _ in 0..MAX_OVERLAP_ITERS {
    let mut any_overlap = false;

    for i in 0..n {
      for j in (i + 1)..n {
        let bi = boxes[i];
        let bj = boxes[j];
        if !bi.overlaps(&bj) {
          continue;
        }
        any_overlap = true;

        let push = bi.push_apart(&bj);

        // Weight: distance factor (closer → smaller denominator → less push for i, more for j)
        // We invert: near objects are pushed LESS.
        // w_i = weight that goes to i (push applied to i is inversely proportional to distance_j
        // divided by total, but we want near → less push, so weight for the push applied to i
        // is proportional to dist_j / (dist_i + dist_j): far object pushes more).
        let di = (inputs[i].cam_dist_km as f32).max(1.0);
        let dj = (inputs[j].cam_dist_km as f32).max(1.0);
        let li = inputs[i].label.len() as f32 + 1.0;
        let lj = inputs[j].label.len() as f32 + 1.0;

        // Distance contribution (60%): push_i ∝ dj, push_j ∝ di
        let dist_weight_i = dj / (di + dj);
        let dist_weight_j = di / (di + dj);

        // Length contribution (40%): longer label → pushed less
        let len_weight_i = lj / (li + lj);
        let len_weight_j = li / (li + lj);

        let weight_i = 0.6 * dist_weight_i + 0.4 * len_weight_i;
        let weight_j = 0.6 * dist_weight_j + 0.4 * len_weight_j;

        boxes[i].x -= push[0] * weight_i;
        boxes[i].y -= push[1] * weight_i;
        boxes[j].x += push[0] * weight_j;
        boxes[j].y += push[1] * weight_j;

        clamp_to_screen(&mut boxes[i], screen_w, screen_h);
        clamp_to_screen(&mut boxes[j], screen_w, screen_h);
      }
    }

    if !any_overlap {
      break;
    }
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
  all_inputs: &[IndicatorInput],
  screen_w: f32,
) -> IndicatorOutput {
  // ---- Step 3: choose horizontal segment direction --------------------------------
  //
  // Compute free space to the left and right of this box (until screen edge
  // or the nearest other indicator box edge — simplified: use screen edges only,
  // since boxes are few and the cost of a proper gap scan is unnecessary).
  let free_left = aabb.x;
  let free_right = screen_w - aabb.right();

  let go_left = free_left >= free_right;
  let text_left_justified = !go_left; // if we go left, anchor is on the right → right-justified

  let free_space = if go_left { free_left } else { free_right };

  // ---- Step 4: segment-1 length with angle-check adjustment ---------------------
  //
  // surplus ∈ [0, 2]; total seg1 length = box_w * (1 + surplus)
  // min length = box_w (0% surplus), max = 3 × box_w (200% surplus).
  let surplus_ratio_base =
    (free_space / box_w.max(1.0)).clamp(0.0, 2.0);

  // Horizontal segment midpoint on the side we chose
  let seg1_anchor_x = if go_left { aabb.x } else { aabb.right() };
  // Vertical midpoint of the text box
  let seg1_y = aabb.y + box_h * 0.5;

  let final_surplus = find_valid_surplus(
    inp,
    seg1_anchor_x,
    seg1_y,
    box_w,
    surplus_ratio_base,
    go_left,
  );

  let seg1_len = box_w * (1.0 + final_surplus);
  let (seg1_start, seg1_end) = if go_left {
    // goes from (anchor - len) to anchor
    (
      [seg1_anchor_x - seg1_len, seg1_y],
      [seg1_anchor_x, seg1_y],
    )
  } else {
    // goes from anchor to (anchor + len)
    (
      [seg1_anchor_x, seg1_y],
      [seg1_anchor_x + seg1_len, seg1_y],
    )
  };

  // ---- Step 5: segment-2 (diagonal to target) -----------------------------------
  let seg2_start = seg1_end;
  let seg2_end = inp.screen_pos;

  // ---- Assemble text position ---------------------------------------------------
  // Text box top-left: for left-justified text (go_right), x = seg1_start;
  // for right-justified text (go_left), x = seg1_start (which is seg1_end - len, i.e. the left edge)
  // Actually the aabb already holds the correct top-left position from Step 1/2.
  let text_pos = [aabb.x, aabb.y];

  IndicatorOutput {
    seg1_start,
    seg1_end,
    seg2_start,
    seg2_end,
    text_pos,
    text_width: box_w,
    text_height: box_h,
    text_pts: pts,
    text_left_justified,
    text_color: inp.text_color,
    label: inp.label.clone(),
  }
}

/// Find a surplus factor for seg1 such that the angle between seg1 and seg2
/// in the half-space of seg1's direction is ≥ 90°.
///
/// If the base surplus already satisfies the constraint, it is returned unchanged.
/// Otherwise, the surplus is decayed by 1/1.2 per iteration (exponential decay)
/// until the constraint is satisfied or surplus reaches 0.
fn find_valid_surplus(
  inp: &IndicatorInput,
  seg1_anchor_x: f32,
  seg1_y: f32,
  box_w: f32,
  base_surplus: f32,
  go_left: bool,
) -> f32 {
  let target = inp.screen_pos;
  let mut surplus = base_surplus;
  let min_surplus: f32 = 0.0;
  let decay: f32 = 1.0 / 1.2;

  for _ in 0..30 {
    let seg1_len = box_w * (1.0 + surplus);
    let end_x = if go_left {
      seg1_anchor_x - seg1_len
    } else {
      seg1_anchor_x + seg1_len
    };
    let seg1_end = [end_x, seg1_y];

    // seg1 direction: from seg1_end toward the text side (horizontal ±x)
    let seg1_dir = if go_left { [-1.0f32, 0.0] } else { [1.0, 0.0] };

    // seg2 direction: from seg1_end to target
    let seg2_raw = [target[0] - seg1_end[0], target[1] - seg1_end[1]];
    let seg2_len = (seg2_raw[0] * seg2_raw[0] + seg2_raw[1] * seg2_raw[1]).sqrt();
    if seg2_len < 1e-3 {
      break; // degenerate — target is on top of seg1_end
    }
    let seg2_dir = [seg2_raw[0] / seg2_len, seg2_raw[1] / seg2_len];

    // Dot product: cos of angle between seg1 and seg2
    let dot = seg1_dir[0] * seg2_dir[0] + seg1_dir[1] * seg2_dir[1];

    // We want angle ≥ 90°, i.e. dot ≤ 0
    if dot <= 0.0 {
      return surplus; // constraint satisfied
    }

    // Decay surplus
    surplus = (surplus * decay).max(min_surplus);
    if surplus < 1e-3 {
      return 0.0;
    }
  }

  surplus
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
