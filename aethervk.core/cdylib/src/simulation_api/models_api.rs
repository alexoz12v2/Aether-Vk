use super::*;
use crate::simulation_api::SimulationContext;
use alloc::{vec::Vec, string::String, sync::Arc, format};
use core::ffi::{c_char, CStr};

impl SimulationContext {
  pub fn load_almanac_file(&mut self, path: *const c_char) -> Result<bool, EngineError> {
    if path.is_null() {
      return Err(EngineError::InvalidNullArgument);
    }
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };

    {
      let logic = self.logic_state.read();
      if logic.almanac_data.file_names.iter().any(|f| f == path_str) {
        return Ok(true); 
      }
    }

    emit_breadcrumb(0, &format!("Loading almanac file: {}", path_str));

    let mut path_buf = oshal::os::fs::PathBuf::new();
    path_buf.push(path_str);

    if let Ok(data) = oshal::os::fs::read(path_buf.as_ref()) {
      let mut logic = self.logic_state.write();
      logic.almanac_data.data.push(data);
      if let Some(last_data) = logic.almanac_data.data.last() {
        let bytes = bytes::BytesMut::from(last_data.as_slice());
        logic.almanac_data.file_names.push(String::from(path_str));
        if let Ok(new_almanac) = logic
          .almanac_data
          .almanac
          .clone()
          .load_from_bytes(bytes, path_str)
        {
          logic.almanac_data.almanac = new_almanac;
          emit_breadcrumb(1, &format!("Successfully loaded: {}", path_str));
          return Ok(true);
        } else {
          logic.almanac_data.data.pop();
          logic.almanac_data.file_names.pop();
          emit_breadcrumb(3, &format!("Failed to parse: {}", path_str));
        }
      }
    } else {
      emit_breadcrumb(3, &format!("Failed to read file: {}", path_str));
    }
    Ok(false)
  }

  pub fn import_model(&mut self, path: *const c_char) -> Result<u64, EngineError> {
    if path.is_null() {
      return Err(EngineError::InvalidNullArgument);
    }
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };

    emit_breadcrumb(0, &format!("Trying to load GLTF from path: {}", path_str));

    if let Ok(mesh) = simulation::comet::load_comet_from_gltf(&path_str, false) {
      emit_breadcrumb(1, &format!("Generating BVH for path: {}", path_str));
      let model_id = self.next_model_id;
      self.next_model_id += 1;

      // Add to cache
      self.mesh_cache.insert(String::from(path_str), mesh);

      // Add to registry
      self.model_registry.insert(model_id, String::from(path_str));
      return Ok(model_id);
    }

    emit_breadcrumb(3, &format!("Failed to load GLTF from path: {}", path_str));
    Ok(0)
  }

  pub fn unload_model(&mut self, model_id: u64) {
    if self.model_registry.remove(&model_id).is_some() {
      emit_breadcrumb(1, &format!("Unloaded model {}", model_id));
    }
  }

  pub fn spawn_model_instance(&mut self, model_id: u64, name: *const c_char) -> Result<u64, EngineError> {
    let path_opt = self.model_registry.get(&model_id).cloned();
    if let Some(path_str) = path_opt {
      let name_str = if name.is_null() {
        "ModelInstance"
      } else {
        unsafe { CStr::from_ptr(name).to_str().unwrap_or("ModelInstance") }
      };

      let mesh_arc = if let Some(cached_mesh) = self.mesh_cache.get(&path_str) {
        cached_mesh
      } else {
        if let Ok(loaded_mesh) = simulation::comet::load_comet_from_gltf(&path_str, false) {
          self.mesh_cache.insert(path_str.clone(), loaded_mesh)
        } else {
          return Ok(0); 
        }
      };

      let mut active = self.active_scene_mut().ok_or(EngineError::InvalidNullArgument)?;
      let entity_id = active.scene.spawn_entity(name_str);

      let _ = active.scene.add_component(
        entity_id,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      );

      let _ = active.scene.add_component(
        entity_id,
        PhysicalMeshComponent {
          asset_path: path_str.clone(),
          mesh: mesh_arc,
          emissive_intensity: 0.0,
          emissive_color: [0.0, 0.0, 0.0],
        },
      );

      let root_entity = active.root_entity;
      active.scene.set_parent(entity_id, Some(root_entity));
      return Ok(active.register_entity(entity_id));
    }
    Ok(0)
  }

  pub fn get_almanac_loaded_files(&mut self, count: *mut u32) -> *mut *mut c_char {
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

  pub fn spawn_procedural_sphere(&mut self, name: *const c_char, radius: f32) -> Result<u64, EngineError> {
    let name_str = if name.is_null() {
      "ProceduralSphere"
    } else {
      unsafe { CStr::from_ptr(name).to_str().unwrap_or("ProceduralSphere") }
    };

    let sphere = simulation::comet::generate_uv_sphere(radius, 32, 32);
    let mut active_scene = self.active_scene_mut().ok_or(EngineError::InvalidNullArgument)?;
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
      },
    );

    let root_entity = active_scene.root_entity;
    active_scene.scene.set_parent(entity_id, Some(root_entity));

    Ok(active_scene.register_entity(entity_id))
  }
}
