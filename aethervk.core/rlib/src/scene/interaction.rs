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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::scene::{HasComponentResultEnum, Scene, TransformComponent};
  use alloc::sync::Arc;
  use spin::RwLock;

  fn setup_scene() -> Scene {
    let texture_cache = Arc::new(RwLock::new(
      crate::simulation::texture_cache::TextureCache::new("test"),
    ));
    let scene = Scene::new(texture_cache);
    scene.register_component::<TransformComponent>(&[]);
    scene.register_component::<HiddenComponent>(&[]);
    scene.register_component::<SelectedComponent>(&[]);
    scene.register_component::<FollowingComponent>(&[]);
    scene
  }

  #[test]
  fn test_hide_show_entity() {
    let scene = setup_scene();
    let e1 = scene.spawn_entity("e1");

    assert_eq!(
      scene.has_component::<HiddenComponent>(e1),
      HasComponentResultEnum::ComponentNotFound
    );

    scene.hide_entity(e1).unwrap();
    assert_eq!(
      scene.has_component::<HiddenComponent>(e1),
      HasComponentResultEnum::EntityHasComponent
    );

    // Hiding again should not fail
    scene.hide_entity(e1).unwrap();

    scene.show_entity(e1).unwrap();
    assert_eq!(
      scene.has_component::<HiddenComponent>(e1),
      HasComponentResultEnum::ComponentNotFound
    );
  }

  #[test]
  fn test_select_entity() {
    let scene = setup_scene();
    let e1 = scene.spawn_entity("e1");
    let e2 = scene.spawn_entity("e2");

    scene.select_entity(e1, None).unwrap();
    assert_eq!(
      scene.has_component::<SelectedComponent>(e1),
      HasComponentResultEnum::EntityHasComponent
    );

    // Selecting e2 should unselect e1
    scene.select_entity(e2, None).unwrap();
    assert_eq!(
      scene.has_component::<SelectedComponent>(e1),
      HasComponentResultEnum::ComponentNotFound
    );
    assert_eq!(
      scene.has_component::<SelectedComponent>(e2),
      HasComponentResultEnum::EntityHasComponent
    );

    scene.unselect_entity(e2).unwrap();
    assert_eq!(
      scene.has_component::<SelectedComponent>(e2),
      HasComponentResultEnum::ComponentNotFound
    );
  }

  #[test]
  fn test_follow_entity() {
    let scene = setup_scene();
    let e1 = scene.spawn_entity("e1");
    let e2 = scene.spawn_entity("e2");

    scene.follow_entity(e1, None).unwrap();
    assert_eq!(
      scene.has_component::<FollowingComponent>(e1),
      HasComponentResultEnum::EntityHasComponent
    );

    // Following e2 should unfollow e1
    scene.follow_entity(e2, None).unwrap();
    assert_eq!(
      scene.has_component::<FollowingComponent>(e1),
      HasComponentResultEnum::ComponentNotFound
    );
    assert_eq!(
      scene.has_component::<FollowingComponent>(e2),
      HasComponentResultEnum::EntityHasComponent
    );

    scene.unfollow_entity(e2).unwrap();
    assert_eq!(
      scene.has_component::<FollowingComponent>(e2),
      HasComponentResultEnum::ComponentNotFound
    );
  }
}
