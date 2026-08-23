//! interaction module.

use crate::{
  scene::{AddComponentError, EntityId, HiddenComponent, Scene},
  types::{EngineError, EngineResult},
};
use alloc::vec::Vec;

pub trait SceneInteractionExt {
  fn hide_entity(&self, entity: EntityId) -> EngineResult<()>;
  fn show_entity(&self, entity: EntityId) -> EngineResult<()>;
}

impl SceneInteractionExt for Scene {
  fn hide_entity(&self, entity: EntityId) -> EngineResult<()> {
    if self.with_component::<HiddenComponent, _, _>(entity, |_| ()).is_none() {
      self
        .add_component(entity, HiddenComponent {})
        .map_err(|e| <AddComponentError as Into<EngineError>>::into(e.into()))?;
    }
    Ok(())
  }

  fn show_entity(&self, entity: EntityId) -> EngineResult<()> {
    let _ = self.remove_component::<HiddenComponent>(entity);
    Ok(())
  }
}

#[cfg(test)]
mod test_interaction;
