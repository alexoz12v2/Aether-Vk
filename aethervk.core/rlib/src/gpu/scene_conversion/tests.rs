use super::*;
use crate::{
  scene::{
    CameraComponent, Scene, TransformComponent,
    ui::{Transform2DComponent, UiComponent},
  },
  simulation::texture_cache::TextureCache,
};
use alloc::sync::Arc;
use parking_lot::RwLock;

#[test]
fn test_ui_layout_relative_placement() {
  let tex_cache = Arc::new(RwLock::new(TextureCache::new("test_tex_cache_ui")));
  let scene = Scene::new(tex_cache);
  scene.register_all_crate_components();

  // 1. Root Background Panel
  let bg_entity = scene.spawn_entity("Background");
  let mut bg_t2d = Transform2DComponent::default();
  bg_t2d.local_position = [0.0, 0.0];
  bg_t2d.size = [1000.0, 1000.0];
  scene.add_component(bg_entity, bg_t2d).unwrap();
  scene.add_component(bg_entity, UiComponent::default()).unwrap();

  // 2. Child Panel
  let child_panel = scene.spawn_entity("Child");
  scene.set_parent(child_panel, Some(bg_entity));
  let mut child_t2d = Transform2DComponent::default();
  child_t2d.local_position = [100.0, 50.0];
  child_t2d.size = [200.0, 200.0];
  scene.add_component(child_panel, child_t2d).unwrap();
  scene.add_component(child_panel, UiComponent::default()).unwrap();

  // 3. Grandchild Panel (Anchored to Bottom-Right of Child)
  let gc_panel = scene.spawn_entity("GrandChild");
  scene.set_parent(gc_panel, Some(child_panel));
  let mut gc_t2d = Transform2DComponent::default();
  gc_t2d.anchor_min = [1.0, 1.0];
  gc_t2d.pivot = [1.0, 1.0];
  gc_t2d.local_position = [-10.0, -10.0]; // 10px padding from right-bottom corner
  gc_t2d.size = [50.0, 50.0];
  scene.add_component(gc_panel, gc_t2d).unwrap();
  scene.add_component(gc_panel, UiComponent::default()).unwrap();

  // Run layout pass directly
  crate::scene::ui::update_ui_layouts(&scene, [1000.0, 1000.0]);

  // Verify background
  scene
    .with_component::<Transform2DComponent, _, _>(bg_entity, |t| {
      assert_eq!(t.global_bounds[0..2], [0.0, 0.0]);
      assert_eq!(t.global_bounds[2..4], [1000.0, 1000.0]);
    })
    .unwrap();

  // Verify child
  scene
    .with_component::<Transform2DComponent, _, _>(child_panel, |t| {
      assert_eq!(t.global_bounds[0..2], [100.0, 50.0]);
      assert_eq!(t.global_bounds[2..4], [200.0, 200.0]);
    })
    .unwrap();

  // Verify grandchild
  // Parent pos: (100, 50), size: (200, 200) => Bottom-Right is (300, 250)
  // Grandchild offset: (-10, -10) => (290, 240)
  // Pivot (1,1) means its own bottom-right is at (290, 240).
  // Size: 50x50 => Top-left global_position should be (240, 190)
  scene
    .with_component::<Transform2DComponent, _, _>(gc_panel, |t| {
      assert_eq!(t.global_bounds[0..2], [240.0, 190.0]);
      assert_eq!(t.global_bounds[2..4], [50.0, 50.0]);
    })
    .unwrap();
}
