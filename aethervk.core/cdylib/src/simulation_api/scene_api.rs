use crate::{expect_scene, expect_scene_and_entity};
use crate::simulation_api::SimulationContext;
use aethervk_core_rlib::physics::physics_scene::math::{closest_intersection, PhysicsSceneMathExt};
use aethervk_core_rlib::scene::{AddComponentError, BvhDebugComponent, CameraComponent, CursorComponent, EntityId, FollowingComponent, GridComponent, HiddenComponent, MarkersComponent, MeasurementComponent, PhysicalMeshComponent, Scene, SelectedComponent, SkyComponent, SunComponent, TransformComponent};
use aethervk_core_rlib::types::{EngineError, EngineResult};
use aethervk_oshal_rlib::math::vector::vec4::{Quat, Vec4f32};
use alloc::{string::String, sync::Arc, vec::Vec};
use core::any::TypeId;
use core::cmp::min;
use core::ffi::c_char;
use spin::{RwLock, RwLockReadGuard};
use aethervk_core_rlib::simulation;
use aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32;
use aethervk_oshal_rlib::math::matrix::{Matrix4, MatrixVectorMul, SquareMatrix};
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::{Vector, Vector3, Vector4};
use crate::structs::{FfiBvhNode, FfiNodeType, SceneContext};

// TODO add scene validate

impl SimulationContext {
  pub fn raycast_ndc(
    &mut self,
    scene_id: u64,
    ndc_x: f32,
    ndc_y: f32,
    out_hit_entity: *mut u64,
    out_px: *mut f32,
    out_py: *mut f32,
    out_pz: *mut f32,
  ) -> EngineResult<bool> {
    let mut view_proj_inv = Mat4x4f32::identity();

    let mut view = Mat4x4f32::identity();
    let active = expect_scene!(self.get_scene(scene_id), "scene_api:raycast_ndc");
    if active.active_camera_entity.is_none() {
      return Err(EngineError::InvalidOperation(
        "scene_api:raycast_ndc | scene doesn't have active_camera_entity",
      ));
    }
    let active_camera_entity = unsafe { active.active_camera_entity.unwrap_unchecked() };
    active
      .scene
      .with_component(active_camera_entity, |c: &TransformComponent| {
        view = Mat4x4f32::from_columns(
          Vec4f32::from_components(1.0, 0.0, 0.0, 0.0),
          Vec4f32::from_components(0.0, 0.0, -1.0, 0.0),
          Vec4f32::from_components(0.0, -1.0, 0.0, 0.0),
          Vec4f32::from_components(0.0, 0.0, 0.0, 1.0),
        ) * Mat4x4f32::from_quat_custom_frame(c.rotation.conjugate())
          * Mat4x4f32::translation(c.position * -1.0);
      })
      .ok_or(EngineError::InvalidOperation(
        "scene_api:raycast_ndc | active_camera_entity doesn't have transform component",
      ))?;

    active
      .scene
      .with_component(active_camera_entity, |cam: &CameraComponent| {
        let proj = cam.projection;
        let view_proj = proj * view;
        view_proj_inv = view_proj.inverse().unwrap_or(Mat4x4f32::identity());
      })
      .ok_or(EngineError::InvalidOperation(
        "scene_api:raycast_ndc | scene doesn't have camera component",
      ))?;

    let ndc_near = Vec4f32::from_components(ndc_x, ndc_y, 0.0, 1.0);
    let ndc_far = Vec4f32::from_components(ndc_x, ndc_y, 1.0, 1.0);

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

    self.raycast_internal(active, ro, rd, out_hit_entity, out_px, out_py, out_pz)
  }

  pub fn raycast(
    &mut self,
    scene_id: u64,
    ro: Vec3f32,
    rd: Vec3f32,
    out_hit_entity: *mut u64,
    out_px: *mut f32,
    out_py: *mut f32,
    out_pz: *mut f32,
  ) -> EngineResult<bool> {
    let active = expect_scene!(self.get_scene(scene_id), "scene_api:raycast");
    self.raycast_internal(active, ro, rd, out_hit_entity, out_px, out_py, out_pz)
  }

  pub fn spawn_entity(&mut self, scene_id: u64, name: &str) -> EngineResult<u64> {
    let mut active = expect_scene!(self.get_scene_mut(scene_id), "scene_api:spawn_entity");
    let id = active.scene.spawn_entity(name);
    Ok(active.register_entity(id))
  }

  pub fn remove_entity(&mut self, scene_id: u64, entity: u64) -> EngineResult<()> {
    let (mut active, entity_id) = expect_scene_and_entity!(
      self.get_scene_mut(scene_id),
      entity,
      "scene_api:remove_entity"
    );
    active.scene.remove_entity(entity_id);

    if active.entity_map.remove(&entity).is_some() {
      Ok(())
    } else {
      Err(EngineError::InvalidOperation(
        "scene_api:remove_entity | entity not found",
      ))
    }
  }

  pub fn set_parent(&mut self, scene_id: u64, entity: u64, parent: u64) -> EngineResult<()> {
    let (active, entity_id) =
      expect_scene_and_entity!(self.get_scene(scene_id), entity, "scene_api:set_parent");
    let parent = if parent == 0 {
      None
    } else {
      active.get_entity(parent)
    }
    .ok_or(EngineError::InvalidOperation(
      "scene_api:set_parent | parent entity not found",
    ))?;
    active.scene.set_parent(entity_id, Some(parent));
    Ok(())
  }

  pub fn get_bvh_nodes(
    &mut self,
    scene_id: u64,
    entity: u64,
    count: *mut u32,
  ) -> EngineResult<*mut FfiBvhNode> {
    let (active, entity_id) =
      expect_scene_and_entity!(self.get_scene(scene_id), entity, "scene_api:get_bvh_nodes");
    let mut ffi_nodes = Vec::new();

    active
      .scene
      .with_component(entity_id, |mesh: &PhysicalMeshComponent| {
        if let Some(bvh) = &mesh.mesh.bvh {
          for node in &bvh.nodes {
            let mut ffi_node = FfiBvhNode::from_offsets(
              node.left_child_or_primitive_offset,
              node.right_child_offset,
              node.primitive_count,
            );

            match &node.bound {
              aethervk_core_rlib::math::collision::linear_bvh::LinearBound::AABB(aabb) => {
                ffi_node.node_type = FfiNodeType::AABB;
                ffi_node.min_x = aabb.min::<Vec3f32>().x();
                ffi_node.min_y = aabb.min::<Vec3f32>().y();
                ffi_node.min_z = aabb.min::<Vec3f32>().z();
                ffi_node.max_x = aabb.max::<Vec3f32>().x();
                ffi_node.max_y = aabb.max::<Vec3f32>().y();
                ffi_node.max_z = aabb.max::<Vec3f32>().z();
              }
              aethervk_core_rlib::math::collision::linear_bvh::LinearBound::OBB(obb) => {
                ffi_node.node_type = FfiNodeType::OBB;
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
      unsafe {
        *count = ffi_nodes.len() as u32;
      }
    }

    if ffi_nodes.is_empty() {
      return Ok(core::ptr::null_mut());
    }

    let ptr = ffi_nodes.as_mut_ptr();
    // TODO should this method marked unsafe for this?
    core::mem::forget(ffi_nodes);
    Ok(ptr)
  }

  pub fn free_bvh_nodes(ptr: *mut FfiBvhNode, count: u32) {
    if !ptr.is_null() {
      let _ = unsafe { Vec::from_raw_parts(ptr, count as usize, count as usize) };
    }
  }

  pub fn get_entity_count(&mut self, scene_id: u64) -> EngineResult<u32> {
    let scene = expect_scene!(self.get_scene(scene_id), "scene_api:get_entity_count");
    // TODO check for overflow?
    Ok(scene.entity_map.len() as u32)
  }

  /// Return number of entities copied, and number of missing ones
  pub fn get_entity_ids(&mut self, scene_id: u64, out_ids: &mut [u64]) -> EngineResult<(u32, u32)> {
    let scene = expect_scene!(self.get_scene(scene_id), "scene_api:get_entity_ids");
    let enumerate_entities = scene.entity_map.keys().enumerate();
    let entities_num = enumerate_entities.len();
    let buffer_len = out_ids.len();
    let missing = if buffer_len >= entities_num {
      0
    } else {
      (entities_num - out_ids.len()) as u32
    };
    let take_len = min(entities_num, buffer_len) as u32;
    for (i, &id) in enumerate_entities.take(take_len as usize) {
      out_ids[i] = id;
    }
    Ok((take_len, missing))
  }

  /// Returns number of missing characters in the name
  pub fn get_entity_name(
    &mut self,
    scene_id: u64,
    entity: u64,
    out_name: &mut [c_char],
  ) -> EngineResult<u32> {
    let (scene, internal_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "scene_api:get_entity_name"
    );
    // unwrap because if the entity is in there, there's the name.
    let name = scene.scene.get_name(internal_id).unwrap();
    let bytes: &[c_char] = bytemuck::cast_slice(name.as_bytes());
    let copy_len = min(bytes.len(), out_name.len());
    let missing: u32 = if out_name.len() >= bytes.len() {
      0
    } else {
      bytes.len() - out_name.len()
    } as _;
    out_name[..copy_len].copy_from_slice(&bytes[..copy_len]);
    out_name[copy_len] = 0;
    Ok(missing)
  }

  pub fn get_entity_parent(&mut self, entity_id: u64, entity: u64) -> EngineResult<u64> {
    let scene = expect_scene!(self.get_scene(entity_id), "scene_api:get_entity_parent");
    let internal_id = scene
      .get_entity(entity)
      .ok_or(EngineError::InvalidOperation(
        "scene_api:get_entity_parent child entity not found",
      ))?;
    let parent_id = scene
      .scene
      .get_parent(internal_id)
      .ok_or(EngineError::InvalidOperation(
        "scene_api:get_entity_parent parent not found",
      ))?;
    // we don't maintain an inverse mapping, so we need to find it manually
    // precondition for unwrap: if entity exists, then the simulation api has its external id.
    Ok(
      scene
        .entity_map
        .iter()
        .find(|&(_, v)| *v == parent_id)
        .map(|(ext, _)| *ext)
        .unwrap(),
    )
  }
  pub fn create_empty_scene(&mut self) -> EngineResult<u64> {
    let (scene, root_entity) = empty_scene_object()?;
    let camera_entity = scene.add_camera("camera", Self::CAMERA_START_POS, root_entity)?;
    let scene_ctx = Arc::new(RwLock::new({
      let mut ctx = SceneContext::new_empty(Arc::new(scene), root_entity)
        .with_active_camera_entity(camera_entity)?;
      ctx.register_present_entities();
      ctx
    }));
    Ok(self.scenes.insert_scene(scene_ctx))
  }

  pub const CAMERA_START_POS: Vec3f32 = Vec3f32::from_components(0.0, -400.0, 0.0);

  pub fn create_default_scene(&mut self) -> EngineResult<u64> {
    let (scene, root_entity) = empty_scene_object()?;

    let camera_entity = scene.add_camera("camera", Self::CAMERA_START_POS, root_entity)?;

    let sky_entity = scene.spawn_entity("sky");
    scene.add_component(sky_entity, SkyComponent {})?;
    scene.set_parent(sky_entity, Some(root_entity));

    let cursor_entity = scene.spawn_entity("cursor");
    scene.add_component(
      cursor_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )?;
    scene.add_component(cursor_entity, CursorComponent {})?;
    scene.set_parent(cursor_entity, Some(root_entity));

    let sun_entity = scene.spawn_entity("sun");
    scene.add_component(
      sun_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )?;
    scene.add_component(
      sun_entity,
      SunComponent {
        resolution: (128, 128, 128),
      },
    )?;
    scene.set_parent(sun_entity, Some(root_entity));

    let sun_sphere = simulation::comet::generate_uv_sphere(0.45 * 0.95, 64, 64);
    scene.add_component(
      sun_entity,
      PhysicalMeshComponent {
        asset_path: String::new(),
        mesh: Arc::from(sun_sphere),
        emissive_intensity: 0.9,
        emissive_color: [1.0, 0.35, 0.02],
      },
    )?;

    let grid_entity = scene.spawn_entity("grid");
    scene.add_component(grid_entity, GridComponent {})?;
    scene.set_parent(grid_entity, Some(root_entity));

    let scene_ctx = Arc::new(RwLock::new(
      SceneContext::new_empty(Arc::new(scene), root_entity)
        .with_active_camera_entity(camera_entity)
        .and_then(|s| s.with_cursor_entity(cursor_entity))
        .and_then(|s| s.with_sun_entity(sun_entity))
        .and_then(|s| s.with_grid_entity(grid_entity))
        .and_then(|s| s.with_sky_entity(sky_entity))
        .map(|mut s| {
          s.register_present_entities();
          s
        })
        .map(|s| s.with_physics_scene())?,
    ));

    Ok(self.scenes.insert_scene(scene_ctx))
  }

  pub fn set_entity_visibility(
    &mut self,
    scene_id: u64,
    entity: u64,
    visible: bool,
  ) -> EngineResult<()> {
    let (scene, id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "scene_api:set_entity_visibility"
    );
    if visible {
      scene
        .scene
        .remove_component::<aethervk_core_rlib::scene::HiddenComponent>(id)
        .map_err(|e| EngineError::InvalidOperation(e))?;
    } else {
      scene
        .scene
        .add_component(id, aethervk_core_rlib::scene::HiddenComponent {})
        .map_err(|e| <AddComponentError as Into<EngineError>>::into(e.into()))?;
    }
    Ok(())
  }

  pub fn set_entity_selected(
    &mut self,
    scene_id: u64,
    entity: u64,
    selected: bool,
  ) -> EngineResult<()> {
    let (scene, id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "scene_api:set_entity_selected"
    );
    if selected {
      scene
        .scene
        .add_component(id, aethervk_core_rlib::scene::SelectedComponent {})
        .map_err(|e| <AddComponentError as Into<EngineError>>::into(e.into()))?;
    } else {
      scene
        .scene
        .remove_component::<aethervk_core_rlib::scene::SelectedComponent>(id)
        .map_err(|e| EngineError::InvalidOperation(e))?;
    }
    Ok(())
  }

  pub fn set_entity_following(
    &mut self,
    scene_id: u64,
    entity: u64,
    following: bool,
  ) -> EngineResult<()> {
    let (active, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "scene_api:set_entity_following"
    );
    if following {
      active
        .scene
        .add_component(entity_id, aethervk_core_rlib::scene::FollowingComponent {})?;
    } else {
      active
        .scene
        .remove_component::<aethervk_core_rlib::scene::FollowingComponent>(entity_id)
        .map_err(|s| EngineError::InvalidOperation(s))?;
    }
    Ok(())
  }

  // ------------------------- INTERNAL --------------------------------------
  fn raycast_internal(
    &self,
    scene: RwLockReadGuard<SceneContext>,
    ro: Vec3f32,
    dir: Vec3f32,
    out_hit_entity: *mut u64,
    out_px: *mut f32,
    out_py: *mut f32,
    out_pz: *mut f32,
  ) -> EngineResult<bool> {
    if scene.physics_scene.is_none() {
      return Err(EngineError::InvalidOperation(
        "scene_api:raycast_internal | scene doesn't have a physics_scene up to date",
      ));
    }
    let ps = unsafe { scene.physics_scene.as_ref().unwrap_unchecked() }.read();
    let len_sq = dir.dot(dir);
    if len_sq < 1e-6 {
      return Ok(false);
    }
    let rd = dir.normalize();

    let ray = aethervk_core_rlib::math::collision::intersection::Ray {
      origin: ro,
      direction: rd,
      length: f32::MAX,
    };

    let hit_instances = ps.intersect_world_bvh_math(&ray);

    // TODO write in scene.rs a method (extension method) which lets the query_res methods
    // TODO inside the thread pool
    let intersections: Vec<((f32, Vec3f32), EntityId)> = scene
      .scene
      .query2_res::<PhysicalMeshComponent, TransformComponent, _, (f32, Vec3f32)>(
        |entity, mesh, transform| {
          if !hit_instances.contains(&entity) || mesh.mesh.bvh.is_none() {
            return None;
          }
          let model_matrix = Mat4x4f32::translation(transform.position)
            * <Mat4x4f32 as Matrix4>::from_quat_custom_frame(transform.rotation)
            * Mat4x4f32::from_scale(transform.scale);
          return ps.intersect_mesh_bvh_math(ro, rd, model_matrix, mesh, ray.length);
        },
      );

    if let Some((_, hit_point, hit_entity)) = closest_intersection(&intersections) {
      // unwrap because if the entity exists, then simulation context has a mapping for it as per precondition
      let external_id = scene
        .entity_map
        .iter()
        .find(|&(_, v)| *v == hit_entity)
        .map(|(ext, _)| *ext)
        .unwrap();

      unsafe {
        if !out_hit_entity.is_null() {
          *out_hit_entity = external_id;
        }
        if !out_px.is_null() {
          *out_px = hit_point.x();
        }
        if !out_py.is_null() {
          *out_py = hit_point.y();
        }
        if !out_pz.is_null() {
          *out_pz = hit_point.z();
        }
      }
      return Ok(true);
    }

    Ok(false)
  }
}

// TODO probably to move in scene.rs in rlib
pub(crate) fn empty_scene_object() -> EngineResult<(Scene, EntityId)> {
  let scene = Scene::new();
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<PhysicalMeshComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<CameraComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<CursorComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<SunComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<SkyComponent>(&[]);
  scene.register_component::<GridComponent>(&[]);
  scene.register_component::<MarkersComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<SelectedComponent>(&[]);
  scene.register_component::<FollowingComponent>(&[]);
  scene.register_component::<HiddenComponent>(&[]);
  scene.register_component::<BvhDebugComponent>(&[]);
  scene.register_component::<MeasurementComponent>(&[]);

  let root_entity = scene.spawn_entity("root");
  scene
    .add_component(
      root_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))?;

  Ok((scene, root_entity))
}
