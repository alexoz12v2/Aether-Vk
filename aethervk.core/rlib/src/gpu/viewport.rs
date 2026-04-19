use super::{Rect2D, Viewport};
use alloc::boxed::Box;

#[derive(Clone)]
pub enum DrawingProgram {
  Viewport3D {
    camera_entity: Option<crate::scene::EntityId>,
  },
  Gui {
    // Identifier to a specific UI Canvas
    ui_canvas_id: u64,
  },
}

pub struct ViewportNode {
  pub viewport: Viewport,
  pub scissor: Rect2D,
  pub program: DrawingProgram,
  pub children: Option<Box<[ViewportNode; 4]>>,
}

pub struct ViewportQuadTree {
  pub root: ViewportNode,
}
