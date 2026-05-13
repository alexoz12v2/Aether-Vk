use crate::scene::{Component, EntityId, Scene};

#[derive(Clone)]
pub struct UpdateComponent {
  pub entities: [Option<EntityId>; 4],
  pub arbitrary_data: [f64; 4],
  pub callback: fn(EntityId, &Scene, &mut [Option<EntityId>; 4], &mut [f64; 4], f32),
}

impl core::fmt::Debug for UpdateComponent {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("UpdateComponent")
      .field("entities", &self.entities)
      .field("arbitrary_data", &self.arbitrary_data)
      .field("callback", &"<fn>")
      .finish()
  }
}

impl Component for UpdateComponent {}
