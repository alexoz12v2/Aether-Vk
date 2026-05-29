use crate::scene::{Component, EntityId, Scene};
use alloc::sync::Arc;

/// Defines the layout, stacking, and hierarchy positioning of a 2D entity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2DComponent {
  // --- Local Inputs (Set by you) ---
  pub local_position: [f32; 2], // X, Y relative to parent
  pub size: [f32; 2],           // Width and Height
  pub scale: [f32; 2],
  pub rotation: f32,      // Radians
  pub local_z_index: i32, // Explicit sorting among siblings

  // --- Layout System Inputs ---
  pub pivot: [f32; 2],      // Normalized [0..1], default [0, 0] (top-left)
  pub anchor_min: [f32; 2], // Normalized [0..1], default [0, 0]
  pub anchor_max: [f32; 2], // Normalized [0..1], default [0, 0]
  pub is_dirty: bool,       // Set to true when local transform changes

  // --- Global Computed (Updated by your Layout System) ---
  pub global_bounds: [f32; 4], // [abs_x, abs_y, abs_width, abs_height]
  pub global_clip: [f32; 4],   // CSS Overflow clipping rect [min_x, min_y, max_x, max_y]
  pub global_depth: u32,       // Hierarchy depth (used for Vulkan painter's algorithm)
}

impl Default for Transform2DComponent {
  fn default() -> Self {
    Self {
      local_position: [0.0, 0.0],
      size: [0.0, 0.0],
      scale: [1.0, 1.0],
      rotation: 0.0,
      local_z_index: 0,
      pivot: [0.0, 0.0],
      anchor_min: [0.0, 0.0],
      anchor_max: [0.0, 0.0],
      is_dirty: true,
      global_bounds: [0.0, 0.0, 0.0, 0.0],
      global_clip: [-9999.0, -9999.0, 9999.0, 9999.0],
      global_depth: 0,
    }
  }
}

impl Component for Transform2DComponent {}

#[derive(Debug, Clone, PartialEq)]
pub struct UiComponent {
  pub color_start: [f32; 4],
  pub color_end: [f32; 4],
  pub color_border: [f32; 4],
  pub color_shadow: [f32; 4],

  pub border_radius: [f32; 4], // TL, TR, BR, BL
  pub shadow_params: [f32; 4], // offset_x, offset_y, blur_radius, spread_radius

  pub gradient_dir: [f32; 2],
  pub border_width: f32,
  pub texture_id: u32, // 0xFFFFFFFF for None

  pub opacity: f32,

  /// If true, applies `overflow: hidden` by intersecting bounds
  /// with the parent's `global_clip` during the layout pass.
  pub is_clipping_mask: bool,
}

impl Default for UiComponent {
  fn default() -> Self {
    Self {
      color_start: [1.0, 1.0, 1.0, 1.0],
      color_end: [1.0, 1.0, 1.0, 1.0],
      color_border: [0.0; 4],
      color_shadow: [0.0; 4],
      border_radius: [0.0; 4],
      shadow_params: [0.0; 4],
      gradient_dir: [0.0, 1.0],
      border_width: 0.0,
      texture_id: 0xFFFFFFFF,
      opacity: 1.0,
      is_clipping_mask: false,
    }
  }
}

impl Component for UiComponent {}

#[derive(Clone, Debug)]
pub struct ScreenSpaceTextComponent {
  pub text: alloc::string::String,
  pub font_atlas: alloc::sync::Arc<crate::scene::text::FontAtlas>,
  pub font_hash: u64,
  pub color: [f32; 4],
  pub points: f32,
  pub use_new_path: bool,
}

impl Component for ScreenSpaceTextComponent {}

/// A simple abstraction to build UI components efficiently
pub struct UiBuilder<'a> {
  scene: &'a Scene,
}

impl<'a> UiBuilder<'a> {
  pub fn new(scene: &'a Scene) -> Self {
    Self { scene }
  }

  pub fn build_panel(
    &self,
    name: &str,
    pos: [f32; 2],
    size: [f32; 2],
    color: [f32; 4],
  ) -> EntityId {
    let e = self.scene.spawn_entity(name);
    let mut t2d = Transform2DComponent::default();
    t2d.local_position = pos;
    t2d.size = size;

    let mut ui = UiComponent::default();
    ui.color_start = color;
    ui.color_end = color;

    let _ = self.scene.add_component(e, t2d);
    let _ = self.scene.add_component(e, ui);
    e
  }

  pub fn build_text(
    &self,
    name: &str,
    text: &str,
    pos: [f32; 2],
    font_atlas: Arc<crate::scene::text::FontAtlas>,
    font_hash: u64,
    color: [f32; 4],
    points: f32,
  ) -> EntityId {
    let e = self.scene.spawn_entity(name);
    let mut t2d = Transform2DComponent::default();
    t2d.local_position = pos;
    t2d.global_depth = 2;

    let text_comp = ScreenSpaceTextComponent {
      text: text.into(),
      font_atlas,
      font_hash,
      color,
      points,
      use_new_path: true, // Default to true for new UI builder
    };

    let _ = self.scene.add_component(e, t2d);
    let _ = self.scene.add_component(e, text_comp);
    e
  }
}

/// Computes global bounds and clipping masks recursively for all UI elements
pub fn update_ui_layouts(scene: &Scene, window_size: [f32; 2]) {
  // Find all UI elements
  let all_ui = scene.query1_res(|id, _t2d: &Transform2DComponent| Some(id));

  // Find UI roots (entities with UI but no parent with UI)
  let mut ui_roots = alloc::vec::Vec::new();
  for (id, _) in &all_ui {
    let parent = scene.get_parent(*id);
    if let Some(p) = parent {
      let parent_has_ui: bool = scene.has_component::<Transform2DComponent>(p).into();
      if !parent_has_ui {
        ui_roots.push(*id);
      }
    } else {
      ui_roots.push(*id);
    }
  }

  for root_id in ui_roots {
    let initial_bounds = [0.0, 0.0, window_size[0], window_size[1]];
    let initial_clip = [-9999.0, -9999.0, 9999.0, 9999.0];
    update_ui_node(scene, root_id, initial_bounds, initial_clip, false);
  }
}

fn update_ui_node(
  scene: &Scene,
  entity_id: EntityId,
  parent_bounds: [f32; 4],
  parent_clip: [f32; 4],
  mut force_dirty: bool,
) {
  let is_clipping: bool = scene
    .with_component::<UiComponent, _, _>(entity_id, |ui| ui.is_clipping_mask)
    .unwrap_or(false);

  let mut new_bounds = parent_bounds;
  let mut new_clip = parent_clip;
  let mut should_continue = false;

  scene.with_component_mut::<Transform2DComponent, _, _>(entity_id, |t2d| {
    if t2d.is_dirty || force_dirty {
      let p_bounds = parent_bounds;

      // Anchor pos
      let anchor_x = p_bounds[0] + (p_bounds[2] * t2d.anchor_min[0]);
      let anchor_y = p_bounds[1] + (p_bounds[3] * t2d.anchor_min[1]);

      // Pivot offset
      let pivot_offset_x = t2d.size[0] * t2d.pivot[0];
      let pivot_offset_y = t2d.size[1] * t2d.pivot[1];

      t2d.global_bounds[0] = anchor_x + t2d.local_position[0] - pivot_offset_x;
      t2d.global_bounds[1] = anchor_y + t2d.local_position[1] - pivot_offset_y;
      t2d.global_bounds[2] = t2d.size[0] * t2d.scale[0];
      t2d.global_bounds[3] = t2d.size[1] * t2d.scale[1];

      // Clip masking (intersection with parent clip)
      let mut my_clip = parent_clip;

      if is_clipping {
        let min_x = if my_clip[0] > t2d.global_bounds[0] {
          my_clip[0]
        } else {
          t2d.global_bounds[0]
        };
        let min_y = if my_clip[1] > t2d.global_bounds[1] {
          my_clip[1]
        } else {
          t2d.global_bounds[1]
        };
        let max_x = if my_clip[2] < t2d.global_bounds[0] + t2d.global_bounds[2] {
          my_clip[2]
        } else {
          t2d.global_bounds[0] + t2d.global_bounds[2]
        };
        let max_y = if my_clip[3] < t2d.global_bounds[1] + t2d.global_bounds[3] {
          my_clip[3]
        } else {
          t2d.global_bounds[1] + t2d.global_bounds[3]
        };
        my_clip = [min_x, min_y, max_x, max_y];
      }

      t2d.global_clip = my_clip;
      t2d.is_dirty = false;

      new_bounds = t2d.global_bounds;
      new_clip = my_clip;
      force_dirty = true;
    } else {
      new_bounds = t2d.global_bounds;
      new_clip = t2d.global_clip;
    }
    should_continue = true;
  });

  if should_continue {
    if let Some(children) = scene.get_children(entity_id) {
      for child in children {
        let child_has_ui: bool = scene.has_component::<Transform2DComponent>(child).into();
        if child_has_ui {
          update_ui_node(scene, child, new_bounds, new_clip, force_dirty);
        }
      }
    }
  }
}
