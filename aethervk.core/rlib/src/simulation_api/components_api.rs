//! components_api module.

use super::*;
use crate::{
  expect_scene, expect_scene_and_entity,
  scene::{self, AddComponentError, CameraProjection, Marker},
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
    let opt = scene.write().scene.with_component_mut(entity_id, |c: &mut TransformComponent| {
      #[cfg(test)]
      println!(
        "Inside set_transform_component | with_component_mut {:?} | position: {:?}",
        c, position
      );
      c.position = position;
      c.rotation = rotation;
      c.scale = scale;
    });

    #[cfg(test)]
    {
      scene
        .read()
        .scene
        .with_component(entity_id, |c: &TransformComponent| {
          assert_eq!(c.position, position);
        })
        .unwrap();
    }

    opt.ok_or(EngineError::InvalidOperation(
      "components_api:set_transform_component couldn't find transform component",
    ))
  }

  // TODO: Relative to its frame!
  pub fn get_transform_component2(
    &self,
    scene_id: u64,
    entity: u64,
  ) -> EngineResult<TransformComponent> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:get_transform_component"
    );
    let transform =
      scene.read().scene.global_transform(entity_id).ok_or(EngineError::InvalidOperation(
        "component_api:get_transform_component couldn't compute global transform",
      ))?;
    Ok(transform)
  }

  /// TODO: Relative to its frame!
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
          use_new_path: true,
          paint_display_mode: 0,
          sphere_center: [0.0, 0.0, 0.0],
          sphere_radius: 1.0,
          grid_color: [0.0, 0.0, 0.0],
          grid_density: 1.0,
        },
      )
      .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))
  }

  pub fn add_component_to_entity<C: scene::Component>(
    &self,
    scene_id: u64,
    entity: u64,
    c: C,
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:add_component_to_entity"
    );
    scene
      .write()
      .scene
      .add_component(entity_id, c)
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
    let mut scene_guard = scene.write();
    scene_guard
      .scene
      .add_component(entity_id, SunComponent { resolution, radius })
      .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))?;

    scene_guard
      .scene
      .add_component(
        entity_id,
        crate::scene::ForceEmitterComponent::Gravity {
          mu: 1.3271244e11_f32,
          beta: 0.0,
        },
      )
      .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))?;

    Ok(())
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
    let mut final_circles = circles.clone();
    use aethervk_oshal_rlib::math::vector::{Vector, Vector3, vec3::Vec3f32};
    {
      let lock = scene.read();
      let tlas_guard = lock.static_tlas.read();
      if !tlas_guard.is_empty() {
        for c in final_circles.iter_mut() {
          let dir_z = c.latitude_rad.sin();
          let dir_x = c.latitude_rad.cos() * c.longitude_rad.cos();
          let dir_y = c.latitude_rad.cos() * c.longitude_rad.sin();
          let ray_dir = Vec3f32::from_components(dir_x, dir_y, dir_z).normalize();

          let ray_orig = if let Some(t) = lock.scene.global_transform(entity_id) {
            Vec3f32::from_components(t.position[0], t.position[1], t.position[2])
          } else {
            Vec3f32::zero()
          };

          if let Some((hit_entity, _, hit_point, hit_normal)) =
            crate::math::collision::linear_bvh::raycast_scene(&lock, &tlas_guard, ray_orig, ray_dir)
          {
            if hit_entity == entity_id {
              c.cached_point = Some([hit_point.x(), hit_point.y(), hit_point.z()]);
              c.cached_normal = Some([hit_normal.x(), hit_normal.y(), hit_normal.z()]);
            } else {
              c.cached_point = Some([hit_point.x(), hit_point.y(), hit_point.z()]);
              c.cached_normal = Some([hit_normal.x(), hit_normal.y(), hit_normal.z()]);
            }
          } else {
            c.cached_point = Some([
              ray_orig.x() + ray_dir.x(),
              ray_orig.y() + ray_dir.y(),
              ray_orig.z() + ray_dir.z(),
            ]);
            c.cached_normal = Some([ray_dir.x(), ray_dir.y(), ray_dir.z()]);
          }
        }
      } else {
        // Fallback to local mesh raycast or simple sphere if TLAS isn't built yet
        // For now, if TLAS is empty, we just fallback to the direction
        for c in final_circles.iter_mut() {
          let dir_z = c.latitude_rad.sin();
          let dir_x = c.latitude_rad.cos() * c.longitude_rad.cos();
          let dir_y = c.latitude_rad.cos() * c.longitude_rad.sin();
          let ray_dir = Vec3f32::from_components(dir_x, dir_y, dir_z).normalize();

          let ray_orig = if let Some(t) = lock.scene.global_transform(entity_id) {
            Vec3f32::from_components(t.position[0], t.position[1], t.position[2])
          } else {
            Vec3f32::zero()
          };
          c.cached_point = Some([
            ray_orig.x() + ray_dir.x(),
            ray_orig.y() + ray_dir.y(),
            ray_orig.z() + ray_dir.z(),
          ]);
          c.cached_normal = Some([ray_dir.x(), ray_dir.y(), ray_dir.z()]);
        }
      }
    }

    let mut old_children = alloc::vec::Vec::new();
    let _ = scene.read().scene.with_component(
      entity_id,
      |c: &crate::scene::ParticleEmitterCirclesComponent| {
        old_children = c.child_entities.clone();
      },
    );
    for child in old_children {
      let _ = self.remove_entity(scene_id, child);
    }

    use crate::scene::{
      TransformComponent,
      particles::{GaussianParams, ParticleEmitterComponent, ParticleSystemComponent},
    };
    use aethervk_oshal_rlib::math::quaternion::Quaternion;

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
      let _ = active.write().scene.add_component(
        child_internal,
        TransformComponent {
          position: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(pos),
          rotation: Quat::identity(),
          scale: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      );

      let _ =
        active.write().scene.add_component(child_internal, ParticleSystemComponent::new(1000));

      let _ = active.write().scene.add_component(
        child_internal,
        ParticleEmitterComponent {
          uv_distribution: crate::math::distribution::Distribution2D::new(&[1.0], 1, 1),
          delta: 0,
          max_particles: 1000,
          velocity_intensity: GaussianParams {
            mean: c.mean_velocity,
            std_dev: c.mean_velocity * 0.2,
            min: c.mean_velocity * 0.1,
            max: c.mean_velocity * 2.0,
          },
          emission_count: GaussianParams {
            mean: c.particles_per_tick as f32,
            std_dev: (c.particles_per_tick as f32) * 0.2,
            min: 1.0,
            max: (c.particles_per_tick as f32) * 5.0,
          },
          particle_radius: c.circle_radius_frac,
          density: c.mass,
          lifetime: c.ttl as i64,
          color: c.color,
          beta: 1.0,
          use_particle2: true,
        },
      );

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
        .add_component(
          entity_id,
          crate::scene::ParticleEmitterCirclesComponent {
            circles: final_circles,
            child_entities: new_children,
          },
        )
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
      .with_component(
        entity_id,
        |c: &crate::scene::ParticleEmitterCirclesComponent| c.circles.clone(),
      )
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
