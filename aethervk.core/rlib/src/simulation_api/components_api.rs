//! components_api module.

use super::*;
use crate::scene::{AddComponentError, Marker};
use crate::simulation_api::SimulationContext;
use crate::{expect_scene, expect_scene_and_entity};
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
          projection: match params {
            CameraParams::Perspective(PerspectiveCameraParams {
              fov,
              aspect_ratio,
              near_plane,
              far_plane,
            }) => Mat4x4f32::perspective_vk(fov, aspect_ratio, near_plane, far_plane),
            CameraParams::Orthographic(OrthographicCameraParams {
              left,
              right,
              bottom,
              top,
              near,
              far,
            }) => Mat4x4f32::orthographic_vk(left, right, bottom, top, near, far),
          },
          near_plane: params.near(),
          far_plane: params.far(),
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
        match params {
          CameraParams::Perspective(PerspectiveCameraParams {
            fov,
            aspect_ratio,
            near_plane,
            far_plane,
          }) => {
            c.projection = Mat4x4f32::perspective_vk(fov, aspect_ratio, near_plane, far_plane);
          }
          CameraParams::Orthographic(OrthographicCameraParams {
            left,
            right,
            bottom,
            top,
            near,
            far,
          }) => {
            c.projection = Mat4x4f32::orthographic_vk(left, right, bottom, top, near, far);
          }
        }
        c.near_plane = params.near();
        c.far_plane = params.far();
      })
      .ok_or(EngineError::InvalidOperation(
        "components_api:set_camera_component couldn't find camera component",
      ))
  }

  /// TODO: Document this item
  pub fn get_camera_component(
    &self,
    scene_id: u64,
    entity: u64,
    proj_out: &mut [f32; 16],
  ) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "component_api:get_camera_component"
    );
    scene
      .read()
      .scene
      .with_component(entity_id, |c: &CameraComponent| {
        *proj_out = c.projection.into();
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
    let mesh = simulation::comet::load_comet_from_gltf(&path_str, false)?;
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
