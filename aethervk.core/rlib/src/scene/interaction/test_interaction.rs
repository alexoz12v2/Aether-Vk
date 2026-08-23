use super::*;
use crate::scene::{HasComponentResultEnum, Scene, TransformComponent};
use alloc::sync::Arc;
use parking_lot::RwLock;

fn setup_scene() -> Scene {
  let texture_cache = Arc::new(RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("test"),
  ));
  let scene = Scene::new(texture_cache);
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<HiddenComponent>(&[]);
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
