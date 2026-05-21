//! models_api module.

use super::*;
use crate::{
  scene::{PhysicalMeshComponent, TransformComponent},
  simulation_api::SimulationContext,
};
use alloc::{sync::Arc, vec::Vec};
use core::ffi::{CStr, c_char};
use oshal::math::vector::{vec3::Vec3f32, vec4::Quat};

impl SimulationContext {
  /// TODO: Document this item
  pub fn unload_model(&self, model_id: u64) {
    if self.scenes.write().model_registry.remove(&model_id).is_some() {
      emit_breadcrumb(1, &alloc::format!("Unloaded model {}", model_id));
    }
  }

  /// TODO: Document this item
  pub fn get_almanac_loaded_files(&self, count: *mut u32) -> *mut *mut c_char {
    let logic = self.logic_state.read();
    if !count.is_null() {
      unsafe {
        *count = logic.almanac_data.file_names.len() as u32;
      }
    }

    let mut ptrs: Vec<*mut c_char> = logic
      .almanac_data
      .file_names
      .iter()
      .filter_map(|s| alloc::ffi::CString::new(s.as_str()).ok())
      .map(|c| c.into_raw())
      .collect();
    let ptr = ptrs.as_mut_ptr();
    core::mem::forget(ptrs);
    ptr
  }

  /// TODO: Document this item
  pub fn spawn_procedural_sphere(
    &self,
    scene_id: u64,
    name: *const c_char,
    radius: f32,
    mass: f32,
  ) -> Result<u64, EngineError> {
    let name_str = if name.is_null() {
      "ProceduralSphere"
    } else {
      unsafe { CStr::from_ptr(name).to_str().unwrap_or("ProceduralSphere") }
    };

    let sphere = crate::simulation::comet::generate_uv_sphere(radius, 32, 32, mass);
    let scenes = self.scenes.write();
    let scene_ctx_lock =
      scenes.get(&scene_id).ok_or(EngineError::InvalidOperation("scene not found"))?;
    let mut active_scene = scene_ctx_lock.write();
    let entity_id = active_scene.scene.spawn_entity(name_str);

    let _ = active_scene.scene.add_component(
      entity_id,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    );

    let _ = active_scene.scene.add_component(
      entity_id,
      PhysicalMeshComponent {
        asset_path: alloc::string::String::new(),
        mesh: Arc::from(sphere),
        emissive_intensity: 0.0,
        emissive_color: [0.0, 0.0, 0.0],
        use_new_path: false,
        paint_display_mode: 0,
        sphere_center: [0.0, 0.0, 0.0],
        sphere_radius: 1.0,
        grid_color: [0.0, 0.0, 0.0],
        grid_density: 1.0,
      },
    );

    let root_entity = active_scene.root_entity;
    active_scene.scene.set_parent(entity_id, Some(root_entity));

    Ok(active_scene.register_entity(entity_id))
  }
}
