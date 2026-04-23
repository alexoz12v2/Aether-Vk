use super::*;
use crate::simulation_api::SimulationContext;
use alloc::{vec::Vec, sync::Arc};
use core::ffi::{c_char, CStr};

impl SimulationContext {
  pub fn add_transform_component(
    &mut self,
    entity: u64,
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    rot_w: f32,
    rot_x: f32,
    rot_y: f32,
    rot_z: f32,
    scale_x: f32,
    scale_y: f32,
    scale_z: f32,
  ) -> Result<(), EngineError> {
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      let _ = active_scene.scene.add_component(
        entity_id,
        TransformComponent {
          position: Vec3f32::from_components(pos_x, pos_y, pos_z),
          rotation: Quat::from_components(rot_x, rot_y, rot_z, rot_w),
          scale: Vec3f32::from_components(scale_x, scale_y, scale_z),
        },
      );
      Ok(())
    } else {
      Err(EngineError::Gpu(GpuError::InvalidState))
    }
  }

  pub fn set_transform_component(
    &mut self,
    entity: u64,
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    rot_w: f32,
    rot_x: f32,
    rot_y: f32,
    rot_z: f32,
    scale_x: f32,
    scale_y: f32,
    scale_z: f32,
  ) -> Result<(), EngineError> {
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      active_scene
        .scene
        .with_component_mut(entity_id, |c: &mut TransformComponent| {
          c.position = Vec3f32::from_components(pos_x, pos_y, pos_z);
          c.rotation = Quat::from_components(rot_x, rot_y, rot_z, rot_w);
          c.scale = Vec3f32::from_components(scale_x, scale_y, scale_z);
        });
      Ok(())
    } else {
      Err(EngineError::Gpu(GpuError::InvalidState))
    }
  }

  pub fn get_transform_component(
    &mut self,
    entity: u64,
    pos_x: *mut f32,
    pos_y: *mut f32,
    pos_z: *mut f32,
    rot_w: *mut f32,
    rot_x: *mut f32,
    rot_y: *mut f32,
    rot_z: *mut f32,
    scale_x: *mut f32,
    scale_y: *mut f32,
    scale_z: *mut f32,
  ) -> Result<bool, EngineError> {
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      if let Some(transform) = active_scene.scene.global_transform(entity_id) {
        unsafe {
          if !pos_x.is_null() { *pos_x = transform.position.x(); }
          if !pos_y.is_null() { *pos_y = transform.position.y(); }
          if !pos_z.is_null() { *pos_z = transform.position.z(); }
          if !rot_w.is_null() { *rot_w = transform.rotation.scalar_part(); }
          let v = transform.rotation.vector_part();
          if !rot_x.is_null() { *rot_x = v.x(); }
          if !rot_y.is_null() { *rot_y = v.y(); }
          if !rot_z.is_null() { *rot_z = v.z(); }
          if !scale_x.is_null() { *scale_x = transform.scale.x(); }
          if !scale_y.is_null() { *scale_y = transform.scale.y(); }
          if !scale_z.is_null() { *scale_z = transform.scale.z(); }
        }
        return Ok(true);
      }
    }
    Ok(false)
  }

  pub fn set_bvh_node_visibility(&mut self, entity: u64, node_index: u32, is_visible: bool) -> Result<(), EngineError> {
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      let mut bvh_len = 0;
      let _ = active_scene.scene.with_component(
        entity_id,
        |mesh: &aethervk_core_rlib::scene::PhysicalMeshComponent| {
          if let Some(bvh) = &mesh.mesh.bvh {
            bvh_len = bvh.nodes.len();
          }
        },
      );

      if (node_index as usize) < bvh_len {
        let mut dbg_opt = None;
        let _ = active_scene.scene.with_component(
          entity_id,
          |dbg: &aethervk_core_rlib::scene::BvhDebugComponent| {
            dbg_opt = Some(dbg.node_render_states.clone());
          },
        );

        let mut states = match dbg_opt {
          Some(s) => s,
          None => {
            let mut s = Vec::with_capacity(bvh_len);
            s.resize(bvh_len, false);
            s
          }
        };

        if (node_index as usize) < states.len() {
          states[node_index as usize] = is_visible;

          let _ = active_scene.scene.add_component(
            entity_id,
            aethervk_core_rlib::scene::BvhDebugComponent {
              node_render_states: states,
            },
          );
        }
      }
      Ok(())
    } else {
      Err(EngineError::Gpu(GpuError::InvalidState))
    }
  }

  pub fn add_camera_component(
    &mut self,
    entity: u64,
    fov: f32,
    aspect_ratio: f32,
    near_plane: f32,
    far_plane: f32,
  ) -> Result<(), EngineError> {
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      let _ = active_scene.scene.add_component(
        entity_id,
        aethervk_core_rlib::scene::CameraComponent {
          projection: Mat4x4f32::perspective_vk(
            fov.to_radians(),
            aspect_ratio,
            near_plane,
            far_plane,
          ),
          near_plane,
          far_plane,
        },
      );
      Ok(())
    } else {
      Err(EngineError::Gpu(GpuError::InvalidState))
    }
  }

  pub fn set_camera_component(
    &mut self,
    entity: u64,
    is_orthographic: bool,
    fov: f32,
    aspect_ratio: f32,
    near_plane: f32,
    far_plane: f32,
  ) -> Result<(), EngineError> {
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      active_scene
        .scene
        .with_component_mut(entity_id, |c: &mut CameraComponent| {
          if is_orthographic {
            // TODO: implement ortho projection if needed
            c.projection = Mat4x4f32::perspective_vk(fov.to_radians(), aspect_ratio, near_plane, far_plane);
          } else {
            c.projection =
              Mat4x4f32::perspective_vk(fov.to_radians(), aspect_ratio, near_plane, far_plane);
          }
          c.near_plane = near_plane;
          c.far_plane = far_plane;
        });
      Ok(())
    } else {
      Err(EngineError::Gpu(GpuError::InvalidState))
    }
  }

  pub fn get_camera_component(&mut self, entity: u64, proj_out: *mut f32) -> Result<bool, EngineError> {
    if proj_out.is_null() {
      return Err(EngineError::InvalidNullArgument);
    }
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      let mut found = false;
      let _ = active_scene
        .scene
        .with_component(entity_id, |c: &CameraComponent| {
          let p: [f32; 16] = c.projection.into();
          unsafe {
            for i in 0..16 {
              *proj_out.add(i) = p[i];
            }
          }
          found = true;
        });
      return Ok(found);
    }
    Ok(false)
  }

  pub fn add_physical_mesh_component(
    &mut self,
    entity: u64,
    gltf_path: *const c_char,
    emissive_intensity: f32,
    emissive_color_r: f32,
    emissive_color_g: f32,
    emissive_color_b: f32,
  ) -> Result<bool, EngineError> {
    if gltf_path.is_null() {
      return Err(EngineError::InvalidNullArgument);
    }
    let path_str = unsafe { CStr::from_ptr(gltf_path).to_str().unwrap_or("") };
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      if let Ok(mesh) = simulation::comet::load_comet_from_gltf(path_str, false) {
        let mesh_arc = Arc::from(mesh);
        let _ = active_scene.scene.add_component(
          entity_id,
          PhysicalMeshComponent {
            asset_path: alloc::string::String::from(path_str),
            mesh: mesh_arc,
            emissive_intensity,
            emissive_color: [emissive_color_r, emissive_color_g, emissive_color_b],
          },
        );
        return Ok(true);
      }
    }
    Ok(false)
  }

  pub fn add_sky_component(&mut self, entity: u64) -> Result<(), EngineError> {
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      let _ = active_scene.scene.add_component(entity_id, SkyComponent {});
      Ok(())
    } else {
      Err(EngineError::Gpu(GpuError::InvalidState))
    }
  }

  pub fn add_cursor_component(&mut self, entity: u64) -> Result<(), EngineError> {
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      let _ = active_scene
        .scene
        .add_component(entity_id, CursorComponent {});
      Ok(())
    } else {
      Err(EngineError::Gpu(GpuError::InvalidState))
    }
  }

  pub fn add_sun_component(
    &mut self,
    entity: u64,
    resolution_x: u32,
    resolution_y: u32,
    resolution_z: u32,
  ) -> Result<(), EngineError> {
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      let _ = active_scene.scene.add_component(
        entity_id,
        SunComponent {
          resolution: (resolution_x, resolution_y, resolution_z),
        },
      );
      Ok(())
    } else {
      Err(EngineError::Gpu(GpuError::InvalidState))
    }
  }

  pub fn add_grid_component(&mut self, entity: u64) -> Result<(), EngineError> {
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      let _ = active_scene
        .scene
        .add_component(entity_id, GridComponent {});
      Ok(())
    } else {
      Err(EngineError::Gpu(GpuError::InvalidState))
    }
  }

  pub fn add_measurement_component(
    &mut self,
    entity: u64,
    p1_x: f32,
    p1_y: f32,
    p1_z: f32,
    p2_x: f32,
    p2_y: f32,
    p2_z: f32,
  ) -> Result<(), EngineError> {
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      let _ = active_scene.scene.add_component(
        entity_id,
        aethervk_core_rlib::scene::MeasurementComponent {
          pos1: Vec3f32::from_components(p1_x, p1_y, p1_z),
          pos2: Vec3f32::from_components(p2_x, p2_y, p2_z),
          points: 1.0,
        },
      );
      Ok(())
    } else {
      Err(EngineError::Gpu(GpuError::InvalidState))
    }
  }

  pub fn add_image_billboard_component(
    &mut self,
    entity: u64,
    is_screen_space: bool,
    width: f32,
    height: f32,
  ) -> Result<(), EngineError> {
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      let billboard_type = if is_screen_space {
        aethervk_core_rlib::scene::BillboardType::ScreenSpace {
          pct_width: width,
          pct_height: height,
        }
      } else {
        aethervk_core_rlib::scene::BillboardType::WorldSpace { width, height }
      };
      let _ = active_scene.scene.add_component(
        entity_id,
        aethervk_core_rlib::scene::ImageBillboardComponent {
          texture_id: 0,
          billboard_type,
        },
      );
      Ok(())
    } else {
      Err(EngineError::Gpu(GpuError::InvalidState))
    }
  }

  pub fn set_markers(
    &mut self,
    entity: u64,
    count: u32,
    px: *const f32,
    py: *const f32,
    pz: *const f32,
    cr: *const f32,
    cg: *const f32,
    cb: *const f32,
    sizes: *const f32,
  ) -> Result<(), EngineError> {
    let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(entity) {
      let mut markers = alloc::vec::Vec::new();

      for i in 0..count as isize {
        unsafe {
          markers.push(aethervk_core_rlib::scene::Marker {
            local_pos: [*px.offset(i), *py.offset(i), *pz.offset(i)],
            color: [*cr.offset(i), *cg.offset(i), *cb.offset(i)],
            size: *sizes.offset(i),
          });
        }
      }

      let mut found = false;
      let _ = active_scene.scene.with_component_mut(
        entity_id,
        |m: &mut aethervk_core_rlib::scene::MarkersComponent| {
          m.markers = markers.clone();
          found = true;
        },
      );

      if !found {
        let _ = active_scene.scene.add_component(
          entity_id,
          aethervk_core_rlib::scene::MarkersComponent { markers },
        );
      }
      Ok(())
    } else {
      Err(EngineError::Gpu(GpuError::InvalidState))
    }
  }
}
