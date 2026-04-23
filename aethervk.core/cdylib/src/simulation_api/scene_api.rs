use super::*;
use crate::simulation_api::SimulationContext;
use alloc::{vec::Vec, string::String, boxed::Box, sync::Arc, format, collections::BTreeMap};
use core::ffi::{c_char, CStr};

impl SimulationContext {
  pub fn raycast_ndc(
    &mut self,
    ndc_x: f32,
    ndc_y: f32,
    out_hit_entity: *mut u64,
    out_px: *mut f32,
    out_py: *mut f32,
    out_pz: *mut f32,
  ) -> Result<bool, EngineError> {
    let mut view_proj_inv = Mat4x4f32::identity();

    let mut view = Mat4x4f32::identity();
    let active = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    active.scene.with_component(
      active.active_camera_entity,
      |c: &TransformComponent| {
        view =
          <Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::from_columns(
            aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(1.0, 0.0, 0.0, 0.0),
            aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, -1.0, 0.0),
            aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, -1.0, 0.0, 0.0),
            aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, 0.0, 1.0),
          ) * <Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::from_quat_custom_frame(
            c.rotation.conjugate(),
          ) * <Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::translation(
            c.position * -1.0,
          );
      },
    );

    active.scene.with_component(
      active.active_camera_entity,
      |cam: &CameraComponent| {
        let proj = cam.projection;
        let view_proj = proj * view;
        view_proj_inv = view_proj.inverse().unwrap_or(Mat4x4f32::identity());
      },
    );

    let ndc_near =
      aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(ndc_x, ndc_y, 0.0, 1.0);
    let ndc_far =
      aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(ndc_x, ndc_y, 1.0, 1.0);

    let mut world_near = view_proj_inv.mul_vector(ndc_near);
    let mut world_far = view_proj_inv.mul_vector(ndc_far);

    if world_near.w() != 0.0 {
      world_near = world_near / world_near.w();
    }
    if world_far.w() != 0.0 {
      world_far = world_far / world_far.w();
    }

    let ro = Vec3f32::from_components(world_near.x(), world_near.y(), world_near.z());
    let target = Vec3f32::from_components(world_far.x(), world_far.y(), world_far.z());
    
    let delta = target - ro;
    let len_sq = delta.dot(delta);
    if len_sq < 1e-6 {
      return Ok(false);
    }
    let rd = delta.normalize();

    drop(active);

    self.raycast(
      ro.x(),
      ro.y(),
      ro.z(),
      rd.x(),
      rd.y(),
      rd.z(),
      out_hit_entity,
      out_px,
      out_py,
      out_pz,
    )
  }

  pub fn raycast(
    &mut self,
    ro_x: f32,
    ro_y: f32,
    ro_z: f32,
    rd_x: f32,
    rd_y: f32,
    rd_z: f32,
    out_hit_entity: *mut u64,
    out_px: *mut f32,
    out_py: *mut f32,
    out_pz: *mut f32,
  ) -> Result<bool, EngineError> {
    let ro = Vec3f32::from_components(ro_x, ro_y, ro_z);
    let dir = Vec3f32::from_components(rd_x, rd_y, rd_z);
    let len_sq = dir.dot(dir);
    if len_sq < 1e-6 {
      return Ok(false);
    }
    let rd = dir.normalize();

    let mut closest_t = core::f32::MAX;
    let mut hit_point = Vec3f32::from_components(0.0, 0.0, 0.0);
    let mut hit_entity = None;

    let ray = aethervk_core_rlib::math::collision::intersection::Ray {
      origin: ro,
      direction: rd,
      length: core::f32::MAX,
    };

    let mut hit_instances = alloc::vec::Vec::new();
    {
      let active_scene = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
      let ps = active_scene.physics_scene.read();
      for node in ps.world_bvh.nodes.iter() {
        let hits_instance = match &node.bound {
          aethervk_core_rlib::math::collision::linear_bvh::LinearBound::AABB(aabb) => {
            aethervk_core_rlib::math::collision::intersection::intersect_ray_aabb(&ray, aabb)
          }
          aethervk_core_rlib::math::collision::linear_bvh::LinearBound::OBB(obb) => {
            aethervk_core_rlib::math::collision::intersection::intersect_ray_obb::<
              f32,
              Vec3f32,
              aethervk_oshal_rlib::math::matrix::mat3::Mat3f32,
            >(&ray, obb)
          }
        };

        if hits_instance {
          if let Some(&entity) = ps.entity_mappings.get(node.left_child_or_primitive_offset as usize) {
            hit_instances.push(entity);
          }
        }
      }
    }

    let active = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    active.scene.query2::<PhysicalMeshComponent, TransformComponent, _>(|entity, mesh, transform| {
        if !hit_instances.contains(&entity) {
          return;
        }

        if let Some(bvh) = &mesh.mesh.bvh {
          let model_matrix = Mat4x4f32::translation(transform.position)
            * <Mat4x4f32 as Matrix4>::from_quat_custom_frame(transform.rotation)
            * Mat4x4f32::from_scale(transform.scale);

          let inv_model = model_matrix.inverse().unwrap_or(Mat4x4f32::identity());

          let local_ro = inv_model.mul_vector(
            aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(
              ro.x(),
              ro.y(),
              ro.z(),
              1.0,
            ),
          );
          let local_rd = inv_model.mul_vector(
            aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(
              rd.x(),
              rd.y(),
              rd.z(),
              0.0,
            ),
          );

          let local_ro = Vec3f32::from_components(local_ro.x(), local_ro.y(), local_ro.z());
          let local_rd_vec = Vec3f32::from_components(local_rd.x(), local_rd.y(), local_rd.z());
          if local_rd_vec.dot(local_rd_vec) < 1e-6 {
            return;
          }
          let local_rd = local_rd_vec.normalize();

          let local_ray = aethervk_core_rlib::math::collision::intersection::Ray {
            origin: local_ro,
            direction: local_rd,
            length: core::f32::MAX,
          };

          let mut stack = alloc::vec::Vec::new();
          if !bvh.nodes.is_empty() {
            stack.push(0);
          }

          while let Some(node_idx) = stack.pop() {
            let local_node = &bvh.nodes[node_idx];

            let hit_local_node = match &local_node.bound {
              aethervk_core_rlib::math::collision::linear_bvh::LinearBound::AABB(aabb) => {
                aethervk_core_rlib::math::collision::intersection::intersect_ray_aabb(
                  &local_ray, aabb,
                )
              }
              aethervk_core_rlib::math::collision::linear_bvh::LinearBound::OBB(obb) => {
                aethervk_core_rlib::math::collision::intersection::intersect_ray_obb::<
                  f32,
                  Vec3f32,
                  Mat3f32,
                >(&local_ray, &obb)
              }
            };

            if hit_local_node {
              if local_node.primitive_count > 0 {
                let prim_start = local_node.left_child_or_primitive_offset as usize;
                let prim_end = prim_start + local_node.primitive_count as usize;
                for j in prim_start..prim_end {
                  let tri_idx = bvh.primitives[j];
                  let v0 = mesh.mesh.vertices[mesh.mesh.indices[tri_idx * 3] as usize].position;
                  let v1 = mesh.mesh.vertices[mesh.mesh.indices[tri_idx * 3 + 1] as usize].position;
                  let v2 = mesh.mesh.vertices[mesh.mesh.indices[tri_idx * 3 + 2] as usize].position;

                  let v0 = Vec3f32::from_components(v0[0], v0[1], v0[2]);
                  let v1 = Vec3f32::from_components(v1[0], v1[1], v1[2]);
                  let v2 = Vec3f32::from_components(v2[0], v2[1], v2[2]);

                  let edge1 = v1 - v0;
                  let edge2 = v2 - v0;
                  let h = local_rd.cross(edge2);
                  let a = edge1.dot(h);

                  if a > -1e-6 && a < 1e-6 {
                    continue;
                  }

                  let f = 1.0 / a;
                  let s = local_ro - v0;
                  let u = f * s.dot(h);
                  if u < 0.0 || u > 1.0 {
                    continue;
                  }

                  let q = s.cross(edge1);
                  let v = f * local_rd.dot(q);
                  if v < 0.0 || u + v > 1.0 {
                    continue;
                  }

                  let t = f * edge2.dot(q);
                  if t > 1e-5 && t < closest_t {
                    closest_t = t;
                    let local_hit = local_ro + local_rd * t;

                    let global_hit = model_matrix.mul_vector(
                      aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(
                        local_hit.x(),
                        local_hit.y(),
                        local_hit.z(),
                        1.0,
                      ),
                    );
                    hit_point =
                      Vec3f32::from_components(global_hit.x(), global_hit.y(), global_hit.z());
                    hit_entity = Some(entity);
                  }
                }
              } else {
                if local_node.right_child_offset != u32::MAX {
                  stack.push(local_node.right_child_offset as usize);
                }
                if local_node.left_child_or_primitive_offset != u32::MAX {
                  stack.push(local_node.left_child_or_primitive_offset as usize);
                }
              }
            }
          }
        }
      });

    if let Some(entity) = hit_entity {
      // Find the external u64 ID
      let mut external_id = 0;
      for (ext_id, internal_id) in &active.entity_map {
        if *internal_id == entity {
          external_id = *ext_id;
          break;
        }
      }

      unsafe {
        if !out_hit_entity.is_null() { *out_hit_entity = external_id; }
        if !out_px.is_null() { *out_px = hit_point.x(); }
        if !out_py.is_null() { *out_py = hit_point.y(); }
        if !out_pz.is_null() { *out_pz = hit_point.z(); }
      }
      return Ok(true);
    }

    Ok(false)
  }

  pub fn spawn_entity(&mut self, name: *const c_char) -> Result<u64, EngineError> {
    let name_str = if name.is_null() {
      "Entity"
    } else {
      unsafe { CStr::from_ptr(name).to_str().unwrap_or("Entity") }
    };
    let mut active = self.active_scene_mut().ok_or(EngineError::InvalidNullArgument)?;
    let id = active.scene.spawn_entity(name_str);
    Ok(active.register_entity(id))
  }

  pub fn remove_entity(&mut self, entity: u64) -> Result<bool, EngineError> {
    let mut active = self.active_scene_mut().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active.get_entity(entity) {
      active.scene.remove_entity(entity_id);
      active.entity_map.remove(&entity);
      Ok(true)
    } else {
      Ok(false)
    }
  }

  pub fn set_parent(&mut self, entity: u64, parent: u64) -> Result<(), EngineError> {
    let mut active = self.active_scene_mut().ok_or(EngineError::InvalidNullArgument)?;
    let entity_id = active.get_entity(entity).ok_or(EngineError::Gpu(GpuError::InvalidState))?;
    let parent_opt = if parent == 0 {
      None
    } else {
      active.get_entity(parent)
    };
    active.scene.set_parent(entity_id, parent_opt);
    Ok(())
  }

  pub fn get_bvh_nodes(&mut self, entity: u64, count: *mut u32) -> Result<*mut FfiBvhNode, EngineError> {
    let active = self.active_scene().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active.get_entity(entity) {
      let mut ffi_nodes = Vec::new();

      active.scene.with_component(entity_id, |mesh: &PhysicalMeshComponent| {
          if let Some(bvh) = &mesh.mesh.bvh {
            for node in &bvh.nodes {
              let mut ffi_node = FfiBvhNode {
                node_type: NodeType::AABB,
                min_x: 0.0, min_y: 0.0, min_z: 0.0,
                max_x: 0.0, max_y: 0.0, max_z: 0.0,
                center_x: 0.0, center_y: 0.0, center_z: 0.0,
                extents_x: 0.0, extents_y: 0.0, extents_z: 0.0,
                left_child: node.left_child_or_primitive_offset,
                right_child: node.right_child_offset,
                primitive_count: node.primitive_count,
              };

              match &node.bound {
                aethervk_core_rlib::math::collision::linear_bvh::LinearBound::AABB(aabb) => {
                  ffi_node.node_type = NodeType::AABB;
                  ffi_node.min_x = aabb.min::<Vec3f32>().x();
                  ffi_node.min_y = aabb.min::<Vec3f32>().y();
                  ffi_node.min_z = aabb.min::<Vec3f32>().z();
                  ffi_node.max_x = aabb.max::<Vec3f32>().x();
                  ffi_node.max_y = aabb.max::<Vec3f32>().y();
                  ffi_node.max_z = aabb.max::<Vec3f32>().z();
                }
                aethervk_core_rlib::math::collision::linear_bvh::LinearBound::OBB(obb) => {
                  ffi_node.node_type = NodeType::OBB;
                  let t: Vec3f32 = obb.translation();
                  let ext: Vec3f32 = obb.half_extent();
                  ffi_node.center_x = t.x();
                  ffi_node.center_y = t.y();
                  ffi_node.center_z = t.z();
                  ffi_node.extents_x = ext.x();
                  ffi_node.extents_y = ext.y();
                  ffi_node.extents_z = ext.z();
                }
              }
              ffi_nodes.push(ffi_node);
            }
          }
        });

      if !count.is_null() {
        unsafe { *count = ffi_nodes.len() as u32; }
      }

      if ffi_nodes.is_empty() {
        return Ok(core::ptr::null_mut());
      }

      let ptr = ffi_nodes.as_mut_ptr();
      core::mem::forget(ffi_nodes);
      return Ok(ptr);
    }

    if !count.is_null() {
      unsafe { *count = 0; }
    }
    Ok(core::ptr::null_mut())
  }

  pub fn free_bvh_nodes(ptr: *mut FfiBvhNode, count: u32) {
    if !ptr.is_null() {
      let _ = unsafe { Vec::from_raw_parts(ptr, count as usize, count as usize) };
    }
  }

  pub fn get_entity_count(&mut self) -> u32 {
    self.active_scene().map(|s| s.entity_map.len() as u32).unwrap_or(0)
  }

  pub fn get_entity_ids(&mut self, out_ids: *mut u64, max_count: u32) {
    if out_ids.is_null() { return; }
    if let Some(scene) = self.active_scene() {
      for (i, &id) in scene.entity_map.keys().enumerate().take(max_count as usize) {
        unsafe { *out_ids.add(i) = id; }
      }
    }
  }

  pub fn get_entity_name(&mut self, entity: u64, out_name: *mut c_char, max_len: u32) -> bool {
    if out_name.is_null() || max_len == 0 { return false; }
    if let Some(scene_ctx) = self.active_scene() {
      if let Some(internal_id) = scene_ctx.get_entity(entity) {
        if let Some(name) = scene_ctx.scene.get_name(internal_id) {
          let bytes = name.as_bytes();
          let copy_len = core::cmp::min(bytes.len(), (max_len - 1) as usize);
          let dest = unsafe { core::slice::from_raw_parts_mut(out_name as *mut u8, max_len as usize) };
          dest[..copy_len].copy_from_slice(&bytes[..copy_len]);
          dest[copy_len] = 0;
          return true;
        }
      }
    }
    false
  }

  pub fn get_entity_parent(&mut self, entity: u64) -> u64 {
    if let Some(scene_ctx) = self.active_scene() {
      if let Some(internal_id) = scene_ctx.get_entity(entity) {
        if let Some(parent_id) = scene_ctx.scene.get_parent(internal_id) {
          for (ext_id, int_id) in &scene_ctx.entity_map {
            if *int_id == parent_id { return *ext_id; }
          }
        }
      }
    }
    0
  }

  pub fn create_default_scene(&mut self) -> Result<u64, EngineError> {
    let scene = Scene::new();
    scene.register_component::<TransformComponent>(&[]);
    scene.register_component::<PhysicalMeshComponent>(&[TypeId::of::<TransformComponent>()]);
    scene.register_component::<CameraComponent>(&[TypeId::of::<TransformComponent>()]);
    scene.register_component::<CursorComponent>(&[TypeId::of::<TransformComponent>()]);
    scene.register_component::<SunComponent>(&[TypeId::of::<TransformComponent>()]);
    scene.register_component::<SkyComponent>(&[]);
    scene.register_component::<GridComponent>(&[]);
    scene.register_component::<aethervk_core_rlib::scene::MarkersComponent>(&[TypeId::of::<TransformComponent>()]);
    scene.register_component::<aethervk_core_rlib::scene::SelectedComponent>(&[]);
    scene.register_component::<aethervk_core_rlib::scene::FollowingComponent>(&[]);
    scene.register_component::<aethervk_core_rlib::scene::HiddenComponent>(&[]);
    scene.register_component::<aethervk_core_rlib::scene::BvhDebugComponent>(&[]);
    scene.register_component::<aethervk_core_rlib::scene::MeasurementComponent>(&[]);

    let root_entity = scene.spawn_entity("root");
    let _ = scene.add_component(
      root_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    );

    let camera_entity = scene.spawn_entity("camera");
    let _ = scene.add_component(
      camera_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, -400.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    );
    let _ = scene.add_component(
      camera_entity,
      CameraComponent {
        projection: Mat4x4f32::perspective_vk(45.0f32.to_radians(), 800.0 / 600.0, 0.1, 10000.0),
        near_plane: 0.1,
        far_plane: 10000.0,
      },
    );
    scene.set_parent(camera_entity, Some(root_entity));

    let sky_entity = scene.spawn_entity("sky");
    let _ = scene.add_component(sky_entity, SkyComponent {});
    scene.set_parent(sky_entity, Some(root_entity));

    let cursor_entity = scene.spawn_entity("cursor");
    let _ = scene.add_component(
      cursor_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    );
    let _ = scene.add_component(cursor_entity, CursorComponent {});
    scene.set_parent(cursor_entity, Some(root_entity));

    let sun_entity = scene.spawn_entity("sun");
    let _ = scene.add_component(
      sun_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    );
    let _ = scene.add_component(
      sun_entity,
      SunComponent {
        resolution: (128, 128, 128),
      },
    );
    scene.set_parent(sun_entity, Some(root_entity));

    let sun_core_entity = scene.spawn_entity("sun_core");
    let sun_sphere = {
      let res = self.thread_pool.spawn_tasklet(|| simulation::comet::generate_uv_sphere(0.45 * 0.95, 64, 64));
      match res {
        Ok(handle) => handle.wait(),
        Err(_) => return Err(EngineError::Gpu(GpuError::InvalidState)),
      }
    };
    let _ = scene.add_component(
      sun_core_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    );
    let _ = scene.add_component(
      sun_core_entity,
      PhysicalMeshComponent {
        asset_path: alloc::string::String::new(),
        mesh: Arc::from(sun_sphere),
        emissive_intensity: 0.9,
        emissive_color: [1.0, 0.35, 0.02],
      },
    );
    scene.set_parent(sun_core_entity, Some(sun_entity));

    let grid_entity = scene.spawn_entity("grid");
    let _ = scene.add_component(grid_entity, GridComponent {});
    scene.set_parent(grid_entity, Some(root_entity));

    let physics_scene = Arc::new(RwLock::new(
      aethervk_core_rlib::physics::physics_scene::PhysicsScene::build_from_scene(&scene),
    ));

    let scene_ctx = Arc::new(RwLock::new(SceneContext {
      scene: Arc::new(scene),
      entity_map: BTreeMap::new(),
      next_entity_id: 1,
      root_entity,
      active_camera_entity: camera_entity,
      cursor_entity,
      sun_entity,
      grid_entity,
      outlines_enabled: Arc::new(AtomicBool::new(false)),
      physics_scene,
    }));

    {
      let mut write_scene_ctx = scene_ctx.write();
      write_scene_ctx.register_entity(root_entity);
      write_scene_ctx.register_entity(camera_entity);
      write_scene_ctx.register_entity(cursor_entity);
      write_scene_ctx.register_entity(sun_entity);
      write_scene_ctx.register_entity(sun_core_entity);
      write_scene_ctx.register_entity(grid_entity);
      write_scene_ctx.register_entity(sky_entity);
    }

    let new_id = self.next_scene_id;
    self.next_scene_id += 1;
    self.scenes.insert(new_id, scene_ctx);
    self.active_scene_id = new_id;
    Ok(new_id)
  }

  pub fn set_entity_visibility(&mut self, entity: u64, visible: bool) -> Result<(), EngineError> {
    let mut active = self.active_scene_mut().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(id) = active.get_entity(entity) {
      if visible {
        let _ = active.scene.remove_component::<aethervk_core_rlib::scene::HiddenComponent>(id);
      } else {
        let _ = active.scene.add_component(id, aethervk_core_rlib::scene::HiddenComponent {});
      }
    }
    Ok(())
  }

  pub fn set_entity_selected(&mut self, entity: u64, selected: bool) -> Result<(), EngineError> {
    let mut active = self.active_scene_mut().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(id) = active.get_entity(entity) {
      if selected {
        let _ = active.scene.add_component(id, aethervk_core_rlib::scene::SelectedComponent {});
      } else {
        let _ = active.scene.remove_component::<aethervk_core_rlib::scene::SelectedComponent>(id);
      }
    }
    Ok(())
  }

  pub fn set_entity_following(&mut self, entity: u64, following: bool) -> Result<(), EngineError> {
    let mut active = self.active_scene_mut().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(id) = active.get_entity(entity) {
      if following {
        let _ = active.scene.add_component(id, aethervk_core_rlib::scene::FollowingComponent {});
      } else {
        let _ = active.scene.remove_component::<aethervk_core_rlib::scene::FollowingComponent>(id);
      }
    }
    Ok(())
  }
}
