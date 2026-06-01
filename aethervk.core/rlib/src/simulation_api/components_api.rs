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

  /// Adds a HighResTransformComponent (f64 position) to an entity.
  pub fn add_highres_transform_component(
    &self,
    scene_id: u64,
    entity: u64,
    pos_x: f64,
    pos_y: f64,
    pos_z: f64,
    rot_w: f32,
    rot_x: f32,
    rot_y: f32,
    rot_z: f32,
    scale_x: f32,
    scale_y: f32,
    scale_z: f32,
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:add_highres_transform_component"
    );
    use aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64;
    scene
      .write()
      .scene
      .add_component(
        entity_id,
        HighResTransformComponent {
          position: Vec3f64::from_components(pos_x, pos_y, pos_z),
          rotation: Quat::from_components(rot_x, rot_y, rot_z, rot_w),
          scale: Vec3f32::from_components(scale_x, scale_y, scale_z),
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
    let transform = scene.read().scene.frame_relative_transform(entity_id).map(|(t, _)| t).ok_or(
      EngineError::InvalidOperation(
        "component_api:get_transform_component couldn't compute local transform",
      ),
    )?;
    Ok(transform)
  }

  #[allow(clippy::not_unsafe_ptr_arg_deref)]
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
    let transform = scene.read().scene.frame_relative_transform(entity_id).map(|(t, _)| t).ok_or(
      EngineError::InvalidOperation(
        "component_api:get_transform_component couldn't compute local transform",
      ),
    )?;
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

  /// Sets the HighResTransformComponent (f64 position) on an entity.
  pub fn set_highres_transform_component(
    &self,
    scene_id: u64,
    entity: u64,
    pos_x: f64,
    pos_y: f64,
    pos_z: f64,
    rot_w: f32,
    rot_x: f32,
    rot_y: f32,
    rot_z: f32,
    scale_x: f32,
    scale_y: f32,
    scale_z: f32,
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:set_highres_transform_component"
    );
    use aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64;
    let opt =
      scene
        .write()
        .scene
        .with_component_mut(entity_id, |c: &mut HighResTransformComponent| {
          c.position = Vec3f64::from_components(pos_x, pos_y, pos_z);
          c.rotation = Quat::from_components(rot_x, rot_y, rot_z, rot_w);
          c.scale = Vec3f32::from_components(scale_x, scale_y, scale_z);
        });
    opt.ok_or(EngineError::InvalidOperation(
      "components_api:set_highres_transform_component couldn't find component",
    ))
  }

  /// Gets the HighResTransformComponent (f64 position) from an entity.
  #[allow(clippy::not_unsafe_ptr_arg_deref)]
  pub fn get_highres_transform_component(
    &self,
    scene_id: u64,
    entity: u64,
    pos_x: *mut f64,
    pos_y: *mut f64,
    pos_z: *mut f64,
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
      "component_api:get_highres_transform_component"
    );
    let hrt = scene
      .read()
      .scene
      .with_component(entity_id, |c: &HighResTransformComponent| *c)
      .ok_or(EngineError::InvalidOperation(
        "component_api:get_highres_transform_component couldn't find component",
      ))?;
    unsafe {
      if !pos_x.is_null() {
        *pos_x = hrt.position.x();
      }
      if !pos_y.is_null() {
        *pos_y = hrt.position.y();
      }
      if !pos_z.is_null() {
        *pos_z = hrt.position.z();
      }
      if !rot_w.is_null() {
        *rot_w = hrt.rotation.scalar_part();
      }
      let v = hrt.rotation.vector_part();
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
        *scale_x = hrt.scale.x();
      }
      if !scale_y.is_null() {
        *scale_y = hrt.scale.y();
      }
      if !scale_z.is_null() {
        *scale_z = hrt.scale.z();
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
        scene
          .read()
          .scene
          .with_component(entity_id, |dbg: &crate::scene::BvhDebugComponent| {
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
    // Add HighRes first (Camera depends on it).
    // Initialize from existing TransformComponent if present, otherwise default.
    {
      let scene_read = scene.read();
      let hrt = scene_read
        .scene
        .with_component(entity_id, |c: &TransformComponent| {
          HighResTransformComponent::from_transform(c)
        })
        .unwrap_or_default();
      drop(scene_read);
      scene
        .write()
        .scene
        .add_component(entity_id, hrt)
        .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))?;
    }
    scene
      .write()
      .scene
      .add_component(
        entity_id,
        CameraComponent {
          projection: params.into(),
          focus_distance: 1.0,
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
          focus_distance: 1.0,
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

  /// Adds a ScreenSpaceBillboardComponent to an entity.
  pub fn add_screen_space_billboard_component(
    &self,
    scene_id: u64,
    entity: u64,
    image_path: &str,
    ndc_x: f32,
    ndc_y: f32,
    scale: f32,
    rotation_deg: f32,
    opacity: f32,
    z_index: i32,
    viewport_id: u64,
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:add_screen_space_billboard"
    );
    scene
      .write()
      .scene
      .add_component(
        entity_id,
        crate::scene::ScreenSpaceBillboardComponent {
          image_path: alloc::string::String::from(image_path),
          ndc_x,
          ndc_y,
          scale,
          rotation_deg,
          opacity,
          z_index,
          viewport_id,
        },
      )
      .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))
  }

  /// Updates the transform properties of a ScreenSpaceBillboardComponent.
  pub fn set_screen_space_billboard(
    &self,
    scene_id: u64,
    entity: u64,
    ndc_x: f32,
    ndc_y: f32,
    scale: f32,
    rotation_deg: f32,
    opacity: f32,
    z_index: i32,
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:set_screen_space_billboard"
    );
    scene
      .write()
      .scene
      .with_component_mut(
        entity_id,
        |c: &mut crate::scene::ScreenSpaceBillboardComponent| {
          c.ndc_x = ndc_x;
          c.ndc_y = ndc_y;
          c.scale = scale;
          c.rotation_deg = rotation_deg;
          c.opacity = opacity;
          c.z_index = z_index;
        },
      )
      .ok_or(EngineError::InvalidOperation(
        "component_api:set_screen_space_billboard couldn't find ScreenSpaceBillboardComponent",
      ))
  }

  /// Reads the ScreenSpaceBillboardComponent data into a DTO.
  pub fn get_screen_space_billboard(
    &self,
    scene_id: u64,
    entity: u64,
  ) -> EngineResult<crate::scene::ScreenSpaceBillboardDTO> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:get_screen_space_billboard"
    );
    scene
      .read()
      .scene
      .with_component(
        entity_id,
        |c: &crate::scene::ScreenSpaceBillboardComponent| crate::scene::ScreenSpaceBillboardDTO {
          ndc_x: c.ndc_x,
          ndc_y: c.ndc_y,
          scale: c.scale,
          rotation_deg: c.rotation_deg,
          opacity: c.opacity,
          z_index: c.z_index,
          viewport_id: c.viewport_id,
        },
      )
      .ok_or(EngineError::InvalidOperation(
        "component_api:get_screen_space_billboard couldn't find ScreenSpaceBillboardComponent",
      ))
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

    let mut old_children = alloc::vec::Vec::new();
    let mut found = false;
    let _ = scene.write().scene.with_component_mut(
      entity_id,
      |c: &mut crate::scene::ParticleEmitterCirclesComponent| {
        for circle in &c.circles {
          if let Some(id) = circle.child_entity {
            old_children.push(id);
          }
        }
        c.circles = final_circles.clone();
        found = true;
      },
    );

    if found {
      let mut scene_write = scene.write();
      for old_id in old_children {
        if !final_circles.iter().any(|c| c.child_entity == Some(old_id)) {
          let _ = scene_write.scene.remove_entity(old_id);
        }
      }
    } else {
      scene
        .write()
        .scene
        .add_component(
          entity_id,
          crate::scene::ParticleEmitterCirclesComponent {
            circles: final_circles,
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
