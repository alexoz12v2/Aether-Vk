//! components_api module.

use super::*;
use crate::{
  expect_scene, expect_scene_and_entity,
  scene::{AddComponentError, CameraProjection, Marker},
  simulation_api::SimulationContext,
};
use alloc::{sync::Arc, vec::Vec};
use oshal::os::fs;

impl SimulationContext {
  /// TODO: Document this item
  pub fn add_transform_component(
    &self,
    scene_id: u64,
    entity: u64,
    position: Vec3f32,
    rotation: Quat,
    scale: Vec3f32,
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:add_transform_component"
    );
    scene
      .write()
      .scene
      .add_component(
        entity_id,
        TransformComponent {
          position,
          rotation,
          scale,
        },
      )
      .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))
  }

  /// TODO: Document this item
  pub fn set_transform_component(
    &self,
    scene_id: u64,
    entity: u64,
    position: Vec3f32,
    rotation: Quat,
    scale: Vec3f32,
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:set_transform_component"
    );
    scene
      .write()
      .scene
      .with_component_mut(entity_id, |c: &mut TransformComponent| {
        c.position = position;
        c.rotation = rotation;
        c.scale = scale;
      })
      .ok_or(EngineError::InvalidOperation(
        "components_api:set_transform_component couldn't find transform component",
      ))
  }

  /// TODO: Document this item
  pub fn get_transform_component(
    &self,
    scene_id: u64,
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
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:get_transform_component"
    );
    let transform =
      scene.read().scene.global_transform(entity_id).ok_or(EngineError::InvalidOperation(
        "component_api:get_transform_component couldn't compute global transform",
      ))?;
    unsafe {
      if !pos_x.is_null() {
        *pos_x = transform.position.x();
      }
      if !pos_y.is_null() {
        *pos_y = transform.position.y();
      }
      if !pos_z.is_null() {
        *pos_z = transform.position.z();
      }
      if !rot_w.is_null() {
        *rot_w = transform.rotation.scalar_part();
      }
      let v = transform.rotation.vector_part();
      if !rot_x.is_null() {
        *rot_x = v.x();
      }
      if !rot_y.is_null() {
        *rot_y = v.y();
      }
      if !rot_z.is_null() {
        *rot_z = v.z();
      }
      if !scale_x.is_null() {
        *scale_x = transform.scale.x();
      }
      if !scale_y.is_null() {
        *scale_y = transform.scale.y();
      }
      if !scale_z.is_null() {
        *scale_z = transform.scale.z();
      }
    }
    Ok(())
  }

  /// TODO: Document this item
  pub fn set_bvh_node_visibility(
    &self,
    scene_id: u64,
    entity: u64,
    node_index: u32,
    is_visible: bool,
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:set_bvh_node_visibility"
    );
    let mut bvh_len = 0;
    scene
      .read()
      .scene
      .with_component(entity_id, |mesh: &PhysicalMeshComponent| {
        if let Some(bvh) = &mesh.mesh.bvh {
          bvh_len = bvh.nodes.len();
        }
      })
      .ok_or(EngineError::InvalidOperation(
        "component_api:set_bvh_node_visibility entity doesn't have PhysicalMeshComponent",
      ))?;

    if (node_index as usize) < bvh_len {
      let mut dbg_opt = None;
      let _ =
        scene.read().scene.with_component(entity_id, |dbg: &crate::scene::BvhDebugComponent| {
          dbg_opt = Some(dbg.node_render_states.clone());
        });

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

        let res = scene.write().scene.add_component(
          entity_id,
          crate::scene::BvhDebugComponent {
            node_render_states: states,
            use_new_path: true, // TODO test first
          },
        );
        if let Err(err) = res {
          match err {
            AddComponentError::EntityNotFound
            | AddComponentError::ComponentNotRegistered
            | AddComponentError::DependencyNotSatisfied { .. } => {
              // TODO should we return error or continue? Maybe accumulate errors and return error if at least one?
              oshal::log!("components_api:set_bvh_node_visibility failed, {}", err);
            }
            AddComponentError::ComponentAlreadyExists => {}
          }
        }
      }
    }
    Ok(())
  }

  /// TODO: Document this item
  pub fn add_camera_component(
    &self,
    scene_id: u64,
    entity: u64,
    params: CameraParams,
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:add_camera_component"
    );
    scene
      .write()
      .scene
      .add_component(
        entity_id,
        CameraComponent {
          projection: params.into(),
        },
      )
      .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))
  }

  /// TODO: Document this item
  pub fn set_camera_component(
    &self,
    scene_id: u64,
    entity: u64,
    params: CameraParams,
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:set_camera_component"
    );
    scene
      .write()
      .scene
      .with_component_mut(entity_id, |c: &mut CameraComponent| {
        *c = CameraComponent {
          projection: params.into(),
        };
      })
      .ok_or(EngineError::InvalidOperation(
        "components_api:set_camera_component couldn't find camera component",
      ))
  }

  /// TODO: Document this item
  pub fn get_camera_component(&self, scene_id: u64, entity: u64) -> EngineResult<CameraParams> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:get_camera_component"
    );
    scene
      .read()
      .scene
      .with_component(entity_id, |c: &CameraComponent| match &c.projection {
        crate::scene::CameraProjection::Perspective {
          fov,
          aspect_ratio,
          near,
          far,
        } => CameraParams::Perspective(PerspectiveCameraParams {
          fov: *fov,
          aspect_ratio: *aspect_ratio,
          near_plane: *near,
          far_plane: *far,
        }),
        crate::scene::CameraProjection::Orthographic {
          left,
          right,
          bottom,
          top,
          near,
          far,
        } => CameraParams::Orthographic(OrthographicCameraParams {
          left: *left,
          right: *right,
          bottom: *bottom,
          top: *top,
          near: *near,
          far: *far,
        }),
      })
      .ok_or(EngineError::InvalidOperation(
        "component_api:get_camera_component couldn't find camera component",
      ))
  }

  /// TODO: Document this item
  pub fn add_physical_mesh_component(
    &self,
    scene_id: u64,
    entity: u64,
    gltf_path: &fs::Path,
    emissive_intensity: f32,
    emissive_color: [f32; 3],
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:add_physical_mesh_component"
    );
    let path_str = gltf_path.to_str_unified().unwrap().to_string();
    let mesh = simulation::comet::load_comet_from_gltf(&path_str, false, None)?;
    let mesh_arc = Arc::from(mesh);
    scene
      .write()
      .scene
      .add_component(
        entity_id,
        // TODO first test this extensively, once stable transition to new rendering
        PhysicalMeshComponent {
          asset_path: path_str,
          mesh: mesh_arc,
          emissive_intensity,
          emissive_color,
          use_new_path: false,
          paint_display_mode: 0,
          sphere_center: [0.0, 0.0, 0.0],
          sphere_radius: 1.0,
          grid_color: [0.0, 0.0, 0.0],
          grid_density: 1.0,
        },
      )
      .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))
  }

  /// TODO: Document this item
  pub fn add_sky_component(&self, scene_id: u64, entity: u64) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:add_sky_component"
    );
    scene
      .write()
      .scene
      .add_component(entity_id, SkyComponent {})
      .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))?;

    if let Some(tx) = self.threads.render_thread.tx_opt() {
      let _ = tx.try_send(crate::simulation_api::structs::RenderCommand::GenerateSky);
    }

    Ok(())
  }

  /// TODO: Document this item
  pub fn add_cursor_component(&self, scene_id: u64, entity: u64) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:add_cursor_component"
    );
    scene
      .write()
      .scene
      .add_component(entity_id, CursorComponent {})
      .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))
  }

  /// TODO: Document this item
  pub fn add_sun_component(
    &self,
    scene_id: u64,
    entity: u64,
    resolution: (u32, u32, u32),
    radius: f32,
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:add_sun_component"
    );
    scene
      .write()
      .scene
      .add_component(entity_id, SunComponent { resolution, radius })
      .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))
  }

  /// TODO: Document this item
  pub fn add_grid_component(&self, scene_id: u64, entity: u64) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:add_grid_component"
    );
    scene
      .write()
      .scene
      .add_component(entity_id, GridComponent {})
      .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))
  }

  /// TODO: Document this item
  pub fn add_measurement_component(
    &self,
    scene_id: u64,
    entity: u64,
    pos1: Vec3f32,
    pos2: Vec3f32,
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:add_measurement_component"
    );
    scene
      .write()
      .scene
      .add_component(
        entity_id,
        crate::scene::MeasurementComponent {
          significant_digits: 2,
          pos1,
          pos2,
          points: 1.0,
        },
      )
      .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))
  }

  /// TODO: Document this item
  pub fn add_image_billboard_component(
    &self,
    scene_id: u64,
    entity: u64,
    is_screen_space: bool,
    width: f32,
    height: f32,
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:add_image_billboard"
    );
    let billboard_type = if is_screen_space {
      crate::scene::BillboardType::ScreenSpace {
        pct_width: width,
        pct_height: height,
      }
    } else {
      crate::scene::BillboardType::WorldSpace { width, height }
    };
    scene
      .write()
      .scene
      .add_component(
        entity_id,
        crate::scene::ImageBillboardComponent {
          texture_id: 0,
          billboard_type,
        },
      )
      .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))
  }

  /// TODO: Document this item
  pub fn set_markers<T: Into<Marker> + Copy>(
    &self,
    scene_id: u64,
    entity: u64,
    markers: &[T],
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:set_markers"
    );
    let markers: Vec<crate::scene::Marker> = markers.iter().map(|m| (*m).into()).collect();
    let mut found = false;
    let _ = scene.write().scene.with_component_mut(
      entity_id,
      |m: &mut crate::scene::MarkersComponent| {
        // TODO can I avoid copying? Probably if I wrap into an mut Option
        m.markers = markers.clone();
        found = true;
      },
    );

    if !found {
      scene
        .write()
        .scene
        .add_component(entity_id, crate::scene::MarkersComponent { markers })
        .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))?
    }
    Ok(())
  }

  /// TODO: Document this item
  pub fn set_particle_emitter_circles_component(
    &self,
    scene_id: u64,
    entity: u64,
    circles: alloc::vec::Vec<crate::scene::EmissionCircle>,
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:set_particle_emitter_circles_component"
    );
    // First, grab the Comet mesh if it exists
    let mut mesh_opt = None;
    let _ = scene.read().scene.with_component(entity_id, |c: &crate::scene::PhysicalMeshComponent| {
      mesh_opt = Some(c.mesh.clone());
    });

    let mut final_circles = circles.clone();
    if let Some(mesh) = mesh_opt {
      use aethervk_oshal_rlib::math::vector::{Vector, Vector3, vec3::Vec3f32};
      for c in final_circles.iter_mut() {
        let r = 1.0;
        let dir_z = c.latitude_rad.sin();
        let dir_x = c.latitude_rad.cos() * c.longitude_rad.cos();
        let dir_y = c.latitude_rad.cos() * c.longitude_rad.sin();
        let ray_dir = Vec3f32::from_components(dir_x, dir_y, dir_z).normalize();
        let ray_orig = Vec3f32::from_components(0.0, 0.0, 0.0);

        let mut closest_t = f32::MAX;
        let mut hit_point = None;
        let mut hit_normal = None;

        let num_triangles = mesh.indices.len() / 3;
        for i in 0..num_triangles {
          let i0 = mesh.indices[i * 3] as usize;
          let i1 = mesh.indices[i * 3 + 1] as usize;
          let i2 = mesh.indices[i * 3 + 2] as usize;
          let v0 = Vec3f32::from_array(mesh.vertices[i0].position);
          let v1 = Vec3f32::from_array(mesh.vertices[i1].position);
          let v2 = Vec3f32::from_array(mesh.vertices[i2].position);

          let edge1 = v1 - v0;
          let edge2 = v2 - v0;
          let h = ray_dir.cross(edge2);
          let a = edge1.dot(h);
          if a.abs() < 1e-6 { continue; }
          let f = 1.0 / a;
          let s = ray_orig - v0;
          let u = f * s.dot(h);
          if u < 0.0 || u > 1.0 { continue; }
          let q = s.cross(edge1);
          let v = f * ray_dir.dot(q);
          if v < 0.0 || u + v > 1.0 { continue; }
          let t = f * edge2.dot(q);

          if t > 1e-6 && t < closest_t {
            closest_t = t;
            hit_point = Some(ray_orig + ray_dir * t);
            let n = edge1.cross(edge2).normalize();
            let n = if n.dot(ray_dir) > 0.0 { n * -1.0 } else { n };
            hit_normal = Some(n);
          }
        }

        if let (Some(p), Some(n)) = (hit_point, hit_normal) {
          c.cached_point = Some([p.x(), p.y(), p.z()]);
          c.cached_normal = Some([n.x(), n.y(), n.z()]);
        } else {
          // Fallback if raycast fails (e.g. from origin didn't hit anything, ray originated outside or mesh has holes)
          c.cached_point = Some([ray_dir.x(), ray_dir.y(), ray_dir.z()]);
          c.cached_normal = Some([ray_dir.x(), ray_dir.y(), ray_dir.z()]);
        }
      }
    }

    let mut old_children = alloc::vec::Vec::new();
    let _ = scene.read().scene.with_component(entity_id, |c: &crate::scene::ParticleEmitterCirclesComponent| {
      old_children = c.child_entities.clone();
    });
    for child in old_children {
      let _ = self.remove_entity(scene_id, child);
    }

    use aethervk_oshal_rlib::math::quaternion::Quaternion;
    use crate::scene::{TransformComponent, particles::{ParticleSystemComponent, ParticleEmitterComponent, GaussianParams}};
    
    let mut new_children = alloc::vec::Vec::new();
    for (i, c) in final_circles.iter().enumerate() {
      let child_ext_id = self.spawn_entity(scene_id, &alloc::format!("jet_{}", i)).unwrap();
      
      // We need the internal entity_id for ECS operations
      let scene_data = self.scenes.read();
      let (active, child_internal) = expect_scene_and_entity!(
        scene_data.get_scene(scene_id),
        child_ext_id,
        "set_particle_emitter_circles_component"
      );

      active.write().scene.set_parent(child_internal, Some(entity_id));

      let pos = c.cached_point.unwrap_or([0.0, 0.0, 0.0]);
      // let norm = c.cached_normal.unwrap_or([1.0, 0.0, 0.0]);
      let _ = active.write().scene.add_component(child_internal, TransformComponent {
        position: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(pos),
        rotation: Quat::identity(),
        scale: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(1.0, 1.0, 1.0),
      });

      let _ = active.write().scene.add_component(child_internal, ParticleSystemComponent::new(1000));
      
      let _ = active.write().scene.add_component(child_internal, ParticleEmitterComponent {
        uv_distribution: crate::math::distribution::Distribution2D::new(&[1.0], 1, 1),
        delta: 0,
        max_particles: 1000,
        velocity_intensity: GaussianParams { mean: c.mean_velocity, std_dev: c.mean_velocity * 0.2, min: c.mean_velocity * 0.1, max: c.mean_velocity * 2.0 },
        emission_count: GaussianParams { mean: c.particles_per_tick as f32, std_dev: (c.particles_per_tick as f32) * 0.2, min: 1.0, max: (c.particles_per_tick as f32) * 5.0 },
        particle_radius: c.circle_radius_frac,
        density: c.mass,
        lifetime: c.ttl as i64,
        color: c.color,
        beta: 1.0,
        use_particle2: true,
      });

      new_children.push(child_ext_id);
    }

    let mut found = false;
    let _ = scene.write().scene.with_component_mut(
      entity_id,
      |c: &mut crate::scene::ParticleEmitterCirclesComponent| {
        c.circles = final_circles.clone();
        c.child_entities = new_children.clone();
        found = true;
      },
    );

    if !found {
      scene
        .write()
        .scene
        .add_component(entity_id, crate::scene::ParticleEmitterCirclesComponent { circles: final_circles, child_entities: new_children })
        .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))?
    }
    Ok(())
  }

  /// TODO: Document this item
  pub fn get_particle_emitter_circles_component(
    &self,
    scene_id: u64,
    entity: u64,
  ) -> EngineResult<alloc::vec::Vec<crate::scene::EmissionCircle>> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:get_particle_emitter_circles_component"
    );
    scene
      .read()
      .scene
      .with_component(entity_id, |c: &crate::scene::ParticleEmitterCirclesComponent| c.circles.clone())
      .ok_or(EngineError::InvalidOperation(
        "component_api:get_particle_emitter_circles_component couldn't find component",
      ))
  }
}

#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
pub struct PerspectiveCameraParams {
  /// Should already be in radians!
  pub fov: f32,
  pub aspect_ratio: f32,
  pub near_plane: f32,
  pub far_plane: f32,
}

#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
pub struct OrthographicCameraParams {
  pub left: f32,
  pub right: f32,
  pub bottom: f32,
  pub top: f32,
  pub near: f32,
  pub far: f32,
}

#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
pub enum CameraParams {
  Perspective(PerspectiveCameraParams),
  Orthographic(OrthographicCameraParams),
}

impl From<CameraParams> for CameraProjection {
  fn from(value: CameraParams) -> Self {
    match value {
      CameraParams::Perspective(persp) => CameraProjection::Perspective {
        fov: persp.fov,
        aspect_ratio: persp.aspect_ratio,
        near: persp.near_plane,
        far: persp.far_plane,
      },
      CameraParams::Orthographic(ortho) => CameraProjection::Orthographic {
        left: ortho.left,
        right: ortho.right,
        bottom: ortho.bottom,
        top: ortho.top,
        near: ortho.near,
        far: ortho.far,
      },
    }
  }
}

impl CameraParams {
  /// TODO: Document this item
  pub fn new_perspective(fov: f32, aspect_ratio: f32, near_plane: f32, far_plane: f32) -> Self {
    Self::Perspective(PerspectiveCameraParams {
      fov,
      aspect_ratio,
      near_plane,
      far_plane,
    })
  }

  /// TODO: Document this item
  pub fn new_orthographic(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
  ) -> Self {
    Self::Orthographic(OrthographicCameraParams {
      left,
      right,
      bottom,
      top,
      near,
      far,
    })
  }

  /// TODO: Document this item
  pub fn near(&self) -> f32 {
    match self {
      CameraParams::Perspective(PerspectiveCameraParams { near_plane, .. }) => *near_plane,
      CameraParams::Orthographic(OrthographicCameraParams { near, .. }) => *near,
    }
  }

  /// TODO: Document this item
  pub fn far(&self) -> f32 {
    match self {
      CameraParams::Perspective(PerspectiveCameraParams { far_plane, .. }) => *far_plane,
      CameraParams::Orthographic(OrthographicCameraParams { far, .. }) => *far,
    }
  }
}
