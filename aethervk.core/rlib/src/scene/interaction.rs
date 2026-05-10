//! interaction module.

use crate::scene::{
  AddComponentError, EntityId, FollowingComponent, HiddenComponent, Scene, SelectedComponent,
};
use crate::types::{EngineError, EngineResult};
use alloc::vec::Vec;

// TODO parallel with option pool (all trait functions)

/// TODO: Document this item
pub trait SceneInteractionExt {
  fn hide_entity(&self, entity: EntityId) -> EngineResult<()>;
  fn show_entity(&self, entity: EntityId) -> EngineResult<()>;
  fn select_entity(
    &self,
    entity: EntityId,
    pool: Option<&aethervk_oshal_rlib::os::pool::ThreadPool>,
  ) -> EngineResult<()>;
  fn unselect_entity(&self, entity: EntityId) -> EngineResult<()>;
  fn follow_entity(
    &self,
    entity: EntityId,
    pool: Option<&aethervk_oshal_rlib::os::pool::ThreadPool>,
  ) -> EngineResult<()>;
  fn unfollow_entity(&self, entity: EntityId) -> EngineResult<()>;
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

  fn select_entity(
    &self,
    entity: EntityId,
    pool: Option<&aethervk_oshal_rlib::os::pool::ThreadPool>,
  ) -> EngineResult<()> {
    let mut to_remove = Vec::new();
    self.query1::<SelectedComponent, _>(|id, _| {
      if id != entity {
        to_remove.push(id);
      }
    });
    for id in to_remove {
      let _ = self.remove_component::<SelectedComponent>(id);
    }
    if self.with_component::<SelectedComponent, _, _>(entity, |_| ()).is_none() {
      self
        .add_component(entity, SelectedComponent {})
        .map_err(|e| <AddComponentError as Into<EngineError>>::into(e.into()))?;
    }
    Ok(())
  }

  fn unselect_entity(&self, entity: EntityId) -> EngineResult<()> {
    let _ = self.remove_component::<SelectedComponent>(entity);
    Ok(())
  }

  fn follow_entity(
    &self,
    entity: EntityId,
    pool: Option<&aethervk_oshal_rlib::os::pool::ThreadPool>,
  ) -> EngineResult<()> {
    let mut to_remove = Vec::new();
    if self.should_parallelize() && pool.is_some() {
      let results = self.query1_res_par::<FollowingComponent, _, _>(pool.unwrap(), |id, _| {
        if id != entity { Some(id) } else { None }
      });
      for (id, _) in results {
        to_remove.push(id);
      }
    } else {
      let results = self
        .query1_res::<FollowingComponent, _, _>(|id, _| if id != entity { Some(id) } else { None });
      for (id, _) in results {
        to_remove.push(id);
      }
    }
    for id in to_remove {
      let _ = self.remove_component::<FollowingComponent>(id);
    }
    if self.with_component::<FollowingComponent, _, _>(entity, |_| ()).is_none() {
      self
        .add_component(entity, FollowingComponent {})
        .map_err(|e| <AddComponentError as Into<EngineError>>::into(e.into()))?;
    }
    Ok(())
  }

  fn unfollow_entity(&self, entity: EntityId) -> EngineResult<()> {
    let _ = self.remove_component::<FollowingComponent>(entity);
    Ok(())
  }
}
