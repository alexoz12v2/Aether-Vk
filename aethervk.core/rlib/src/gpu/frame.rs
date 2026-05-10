//! frame module.

use crate::gpu;
use crate::gpu::{
  GpuResourceHandle, GridPushConstants, PipelineKey, PresentationEngineHandle, PushConstants,
  RenderDevice, RenderDeviceExt, SkyPushConstants, SunPushConstants, TextureFlags,
};
use crate::math::collision::linear_bvh::LinearBound;
use crate::scene::{CameraComponent, EntityId, RenderableDataRef, TransformComponent};
use crate::types::{GpuError, GpuResult};
use aethervk_oshal_rlib::math::vector::vec4::{Quat, Vec4f32};
use aethervk_oshal_rlib::math::{
  matrix::{Matrix, Matrix4, MatrixVectorMul, SquareMatrix, mat4::Mat4x4f32},
  quaternion::Quaternion,
  vector::{Vector, Vector3, Vector4, vec3::Vec3f32},
};
use alloc::vec::Vec;
// TODO move render_frame here

#[derive(Clone, Copy, PartialEq)]
/// TODO: Document this item
pub struct ResourceUploadResult {
  /// The pipeline to use for this draw call.
  pub pipeline: PipelineKey,
  pub outline_pipeline: Option<PipelineKey>,
  /// The buffer group (vertex, index, ...) to bind.
  pub buffers: GpuResourceHandle,
  pub indirect_buffer: Option<GpuResourceHandle>,
  pub texture_flags: TextureFlags,
  pub descriptor_index: Option<u32>,
}

/// Represents a single draw call with all necessary information.
#[derive(Clone)]
pub struct DrawCall {
  /// The pipeline to use for this draw call.
  pub pipeline: PipelineKey,
  pub outline_pipeline: Option<PipelineKey>,
  /// The vertex buffer to bind.
  pub buffers: GpuResourceHandle,
  /// index count
  pub index_count: u32,
  /// The model matrix of the object to draw.
  pub model_matrix: Mat4x4f32,
  pub texture_flags: TextureFlags,
  /// From `PhysicalMeshComponent`
  pub emissive_intensity: f32,
  /// From `PhysicalMeshComponent`
  pub emissive_color: [f32; 3],
  pub draw_outline: bool,
  pub outline_color: [f32; 4],
  pub use_new_path: bool,
  pub paint_display_mode: u32,
}

impl DrawCall {
  /// TODO: Document this item
  pub fn from_handles_and_matrix(
    result: ResourceUploadResult,
    index_count: u32,
    outline: Option<[f32; 4]>,
    model_matrix: Mat4x4f32,
    emissive_intensity: f32,
    emissive_color: [f32; 3],
    use_new_path: bool,
    paint_display_mode: u32,
  ) -> Self {
    Self {
      pipeline: result.pipeline,
      outline_pipeline: result.outline_pipeline,
      buffers: result.buffers,
      index_count,
      model_matrix,
      texture_flags: result.texture_flags,
      emissive_intensity,
      emissive_color,
      draw_outline: outline.is_some(),
      outline_color: outline.unwrap_or([0.0; 4]),
      use_new_path,
      paint_display_mode,
    }
  }
}

/// Represents a draw call for a cursor.
pub struct CursorDrawCall {
  pub pipeline: PipelineKey,
  pub vertex_count: u32,
  pub model_matrix: Mat4x4f32,
  pub cursor_size: f32,
}

impl CursorDrawCall {
  /// TODO: Document this item
  pub fn from_result_and_matrix(
    result: ResourceUploadResult,
    vertex_count: u32,
    model_matrix: Mat4x4f32,
    cursor_size: f32,
  ) -> Self {
    Self {
      pipeline: result.pipeline,
      vertex_count,
      model_matrix,
      cursor_size,
    }
  }
}

/// Represents a draw call for a marker.
pub struct MarkerDrawCall {
  pub pipeline: PipelineKey,
  pub vertex_count: u32,
  pub model_matrix: Mat4x4f32,
  pub local_pos: [f32; 3],
  pub size: f32,
  pub color: [f32; 3],
}

impl MarkerDrawCall {
  const VERTEX_COUNT_VK: u32 = 4;
  /// TODO: Document this item
  pub fn from_values(
    pipeline: PipelineKey,
    model_matrix: Mat4x4f32,
    local_pos: [f32; 3],
    size: f32,
    color: [f32; 3],
  ) -> Self {
    Self {
      pipeline,
      vertex_count: Self::VERTEX_COUNT_VK,
      model_matrix,
      local_pos,
      size,
      color,
    }
  }
}

/// TODO: Document this item
pub struct MeasurementDrawCall {
  pub pipeline: PipelineKey,
  pub vertex_count: u32,
  pub p1: [f32; 3],
  pub p2: [f32; 3],
  pub points: f32, // TODO handle font size (since text used by text archetype too, need a font registry class)
  pub significant_digits: u32,
}

impl MeasurementDrawCall {
  const VERTEX_COUNT_VK: u32 = 6;

  /// TODO: Document this item
  pub fn from_data_and_pipeline(
    p1: Vec3f32,
    p2: Vec3f32,
    points: f32,
    significant_digits: u32,
    pipeline_key: PipelineKey,
  ) -> Self {
    Self {
      pipeline: pipeline_key,
      vertex_count: Self::VERTEX_COUNT_VK,
      p1: [p1.x(), p1.y(), p1.z()],
      p2: [p2.x(), p2.y(), p2.z()],
      points,
      significant_digits,
    }
  }
}

/// TODO: Document this item
pub struct GizmoDrawCall {
  pub pipeline: PipelineKey,
  pub vertex_count: u32,
  pub scale: f32,
  pub buffer_index: u32,
}

impl GizmoDrawCall {
  const VERTEX_COUNT_VK: u32 = 6;
  /// TODO: Document this item
  pub fn from_values(pipeline: PipelineKey, scale: f32, buffer_index: u32) -> Self {
    Self {
      pipeline,
      vertex_count: Self::VERTEX_COUNT_VK,
      scale,
      buffer_index,
    }
  }
}

/// TODO: Document this item
pub struct BillboardDrawCall {
  pub pipeline: PipelineKey,
  pub vertex_count: u32,
  pub model_matrix: Mat4x4f32,
  pub texture_id: u64,
  pub billboard_type: crate::scene::BillboardType,
}

impl BillboardDrawCall {
  const VERTEX_COUNT_VK: u32 = 4;
  /// TODO: Document this item
  pub fn from_data(
    pipeline: PipelineKey,
    model_matrix: Mat4x4f32,
    texture_id: u64,
    billboard_type: crate::scene::BillboardType,
  ) -> Self {
    Self {
      pipeline,
      vertex_count: Self::VERTEX_COUNT_VK,
      model_matrix,
      texture_id,
      billboard_type,
    }
  }
}

/// TODO: Document this item
pub struct SunDrawCall {
  // TODO: remove. This is needed because the RenderDevice maps entity id to pipeline layout and descriptor set.
  pub entity: EntityId,
  pub pipeline: PipelineKey,
  pub model_matrix: Mat4x4f32,
  /// camera position in local space of the sun
  pub local_camera_pos: Vec3f32,
  pub vertex_count: u32,
  pub radius: f32,
}

impl SunDrawCall {
  const VERTEX_COUNT_TRIANGLE_STRIP_VK: u32 = 14;

  /// Result is meant to be logged and converted to a None. Shouldn't stop rendering
  /// TODO: remove entity
  pub fn from_model_and_camera(
    model: Mat4x4f32,
    c: &CameraRenderData,
    pipeline_key: PipelineKey,
    entity: EntityId,
    radius: f32,
  ) -> GpuResult<Self> {
    let model_inv = model.inverse().ok_or(GpuError::BackendSpecific(alloc::format!(
      "SunDrawCall: Couldn't invert model matrix {:?}",
      model
    )))?;
    let local_camera_pos = Vec3f32(model_inv.mul_vector(c.pos.to_point()));
    Ok(Self {
      entity,
      pipeline: pipeline_key,
      model_matrix: model,
      local_camera_pos,
      vertex_count: Self::VERTEX_COUNT_TRIANGLE_STRIP_VK,
      radius,
    })
  }

  /// TODO: Document this item
  pub fn sun_pos(&self) -> Vec3f32 {
    Vec3f32(self.model_matrix.w)
  }
}

/// TODO: Document this item
pub struct SkyDrawCall {
  pub sky_view_proj: Mat4x4f32,
  pub pipeline: PipelineKey,
  pub inv_view_proj_mat: Mat4x4f32,
  pub vertex_count: u32,
}

impl SkyDrawCall {
  const VERTEX_COUNT_VK: u32 = 3;

  /// TODO: Document this item
  pub fn from_camera(camera_data: &CameraRenderData, pipeline_key: PipelineKey) -> GpuResult<Self> {
    let sky_view_proj = {
      let sky_view = camera_data.view.zeroed_translation();
      camera_data.proj * sky_view
    };
    let inv_view_proj_mat = camera_data.view_proj.inverse().ok_or(GpuError::InvalidState(
      "SkyDrawCall: couldn't invert view_proj matrix",
    ))?;

    Ok(Self {
      sky_view_proj,
      pipeline: pipeline_key,
      inv_view_proj_mat,
      vertex_count: Self::VERTEX_COUNT_VK,
    })
  }
}

/// TODO: Document this item
pub struct GridDrawCall {
  pub pipeline: PipelineKey,
  pub density: f32,
  pub grid_size: f32, // TODO main lines size
  pub grid_color: [f32; 3],
  pub vertex_count: u32,
}

impl GridDrawCall {
  const VERTEX_COUNT_VK: u32 = 4;
  /// TODO: Document this item
  pub fn new(pipeline: PipelineKey, density: f32, grid_size: f32, grid_color: [f32; 3]) -> Self {
    Self {
      pipeline,
      density,
      grid_size,
      grid_color,
      vertex_count: Self::VERTEX_COUNT_VK,
    }
  }
}

/// Model is not stored here cause it's shared with its physical mesh
pub struct BvhDrawCall {
  /// "weak reference" to physical mesh draw call (to take model)
  pub physical_mesh_call_index: usize,
  pub pipeline: PipelineKey,
  pub vertex_count: u32, // 24
  /// object space axes
  pub axes: [[f32; 3]; 3],
  /// object space center
  pub center: Vec3f32,
  /// half lengths along each axis
  pub extents: Vec3f32,
}

impl BvhDrawCall {
  const VERTEX_COUNT_VK: u32 = 24;

  /// TODO: Document this item
  pub fn new(
    bound: &LinearBound<f32>,
    pipeline_key: PipelineKey,
    physical_mesh_call_index: usize,
  ) -> Self {
    let (center, extents, ax, ay, az) = match bound {
      LinearBound::AABB(aabb) => {
        let center = aabb.center();
        let he = aabb.half_extents();
        (
          center,
          he,
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0],
        )
      }
      LinearBound::OBB(obb) => {
        let center: Vec3f32 = obb.center();
        let he: Vec3f32 = obb.half_extents();
        let axes: [Vec3f32; 3] = obb.axes();
        (
          center,
          he,
          [axes[0].x(), axes[0].y(), axes[0].z()],
          [axes[1].x(), axes[1].y(), axes[1].z()],
          [axes[2].x(), axes[2].y(), axes[2].z()],
        )
      }
    };
    Self {
      physical_mesh_call_index,
      pipeline: pipeline_key,
      vertex_count: Self::VERTEX_COUNT_VK,
      axes: [ax, ay, az],
      center,
      extents,
    }
  }

  /// TODO: Document this item
  pub fn to_push_constants(
    &self,
    mesh_draw_calls: &[DrawCall],
    camera_data: &CameraRenderData,
  ) -> Option<super::BvhPushConstants> {
    let center_arr: [f32; 3] = self.center.into();
    let extents_arr: [f32; 3] = self.extents.into();
    let ax = self.axes[0];
    let ay = self.axes[1];
    let az = self.axes[2];
    let model = &mesh_draw_calls.get(self.physical_mesh_call_index)?.model_matrix;
    let view_proj = &camera_data.view_proj;
    let mvp_mat = *view_proj * *model;
    Some(super::BvhPushConstants {
      mvp_arr: mvp_mat.into(),
      center_type: [center_arr[0], center_arr[1], center_arr[2], 1.0],
      extents_arr: [extents_arr[0], extents_arr[1], extents_arr[2], 0.0],
      axes_x: [ax[0], ax[1], ax[2], 0.0],
      axes_y: [ay[0], ay[1], ay[2], 0.0],
      axes_z: [az[0], az[1], az[2], 0.0],
    })
  }
}

/// TODO: Document this item
pub struct CameraRenderData {
  pub pos: Vec3f32,
  pub rot: Quat,
  pub view: Mat4x4f32,
  pub proj: Mat4x4f32,
  pub view_proj: Mat4x4f32,
  pub up: [f32; 3],
  pub right: [f32; 3],
  pub near: f32,
  pub far: f32,
}

impl CameraRenderData {
  /// Note: camera component should have been updated with presentation engine data
  pub fn new(transform: &TransformComponent, camera: &CameraComponent) -> Self {
    // Extract camera's local axes in world space
    let right = transform.rotation.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
    let up = transform.rotation.rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));
    let forward = transform.rotation.rotate_vector(Vec3f32::from_components(0.0, -1.0, 0.0));

    let view = Mat4x4f32::look_at_axes(right, forward, up, transform.position);
    let view_proj = camera.projection * view;

    Self {
      pos: transform.position,
      rot: transform.rotation,
      view,
      proj: camera.projection,
      view_proj,
      up: [up.x(), up.y(), up.z()],
      right: [right.x(), right.y(), right.z()],
      far: camera.far_plane,
      near: camera.near_plane,
    }
  }
}

/// TODO: Document this item
pub struct ParticleDrawCall {
  pub pipeline: PipelineKey,
  pub system_particle_offset: u32,
  pub system_indirect_offset: u32,
  pub config: crate::scene::particles::ParticleEmitterConfig,
  pub particles: alloc::sync::Weak<spin::RwLock<Vec<crate::scene::particles::ParticleData>>>,
}

/// TODO: Document this item
pub struct Particle2DrawCall {
  pub pipeline: PipelineKey,
  pub system_particle_offset: u32,
  pub system_indirect_offset: u32,
  pub config: crate::scene::particles::ParticleEmitterConfig,
  pub particles: alloc::sync::Weak<spin::RwLock<Vec<crate::scene::particles::ParticleData>>>,
}

#[derive(Clone)]
/// TODO: Document this item
pub struct TrajectoryBatchCall {
  pub pipeline: PipelineKey,
  pub total_vertices: u32, // (MAX_STEPS + 1) * 2
  pub total_segments: u32, // TOTAL_SEGMENTS_ACROSS_ALL_TRAJECTORIES
  pub map_ptr: u64,
  pub traj_ptr: u64,
}

/// TODO: Document this item
pub struct RenderScene {
  pub time_readings: aethervk_oshal_rlib::os::time::TimeReadings,
  pub camera_data: CameraRenderData,
  pub window_extent: [u32; 2],

  pub draw_calls: Vec<DrawCall>,
  pub marker_calls: Vec<MarkerDrawCall>,
  pub measurement_calls: Vec<MeasurementDrawCall>,
  pub billboard_calls: Vec<BillboardDrawCall>,
  pub bvh_draw_calls: Vec<BvhDrawCall>,
  pub gizmo_calls: Vec<GizmoDrawCall>,
  pub particle_calls: Vec<ParticleDrawCall>,
  pub particle2_calls: Vec<Particle2DrawCall>,

  pub cursor_call: Option<CursorDrawCall>,
  pub sun_call: Option<SunDrawCall>,
  pub sky_call: Option<SkyDrawCall>,
  pub grid_call: Option<GridDrawCall>,
  pub trajectory_call: Option<TrajectoryBatchCall>,
}

impl RenderScene {
  const START_VEC_CAPACITY: usize = 32;
  /// TODO: Document this item
  pub fn new(
    camera: (TransformComponent, CameraComponent),
    time_readings: aethervk_oshal_rlib::os::time::TimeReadings,
    window_extent: [u32; 2],
  ) -> Self {
    Self {
      window_extent,
      time_readings,
      draw_calls: Vec::with_capacity(Self::START_VEC_CAPACITY),
      cursor_call: None,
      marker_calls: Vec::with_capacity(Self::START_VEC_CAPACITY),
      measurement_calls: Vec::with_capacity(Self::START_VEC_CAPACITY),
      billboard_calls: Vec::with_capacity(Self::START_VEC_CAPACITY),
      bvh_draw_calls: Vec::with_capacity(Self::START_VEC_CAPACITY),
      gizmo_calls: Vec::with_capacity(Self::START_VEC_CAPACITY),
      particle_calls: Vec::with_capacity(Self::START_VEC_CAPACITY),
      particle2_calls: Vec::with_capacity(Self::START_VEC_CAPACITY),
      camera_data: CameraRenderData::new(&camera.0, &camera.1),
      sun_call: None,
      sky_call: None,
      grid_call: None,
      trajectory_call: None,
    }
  }

  /// Registers a renderable entity to be drawn in this frame.
  pub fn add_renderable(
    &mut self,
    cmd_buffer: gpu::CommandBufferHandle,
    device: &dyn RenderDevice,
    entity_id: EntityId,
    model_matrix: Mat4x4f32,
    renderable: RenderableDataRef,
    presentation_engine_handle: PresentationEngineHandle,
    debug_name: &str,
    draw_outline: bool,
    outline_color: [f32; 4],
  ) -> GpuResult<()> {
    match renderable {
      RenderableDataRef::ImageBillboard(component) => {
        let res: ResourceUploadResult =
          match device.get_billboard_resources(presentation_engine_handle) {
            Ok(r) => r,
            Err(_) => device.create_billboard_resources(cmd_buffer, presentation_engine_handle)?,
          };
        self.billboard_calls.push(BillboardDrawCall {
          pipeline: res.pipeline,
          vertex_count: 4,
          model_matrix,
          texture_id: component.texture_id,
          billboard_type: component.billboard_type,
        });
      }
      RenderableDataRef::PhysicalMesh(component) => {
        let asset_hash = component.mesh.id;
        let res: ResourceUploadResult = if component.use_new_path {
          match device.get_physical_mesh2_resources(asset_hash, presentation_engine_handle) {
            Ok(r) => r,
            Err(_) => device.create_physical_mesh2_resources(
              cmd_buffer,
              asset_hash,
              &component,
              presentation_engine_handle,
              debug_name,
            )?,
          }
        } else {
          match device.get_physical_mesh_resources(asset_hash, presentation_engine_handle) {
            Ok(r) => r,
            Err(_) => device.create_physical_mesh_resources(
              cmd_buffer,
              asset_hash,
              &component,
              presentation_engine_handle,
              debug_name,
            )?,
          }
        };
        let index_count = component.mesh.indices.len() as u32;
        let dc = DrawCall::from_handles_and_matrix(
          res,
          index_count,
          if draw_outline {
            Some(outline_color)
          } else {
            None
          },
          model_matrix,
          component.emissive_intensity,
          component.emissive_color,
          component.use_new_path,
          component.paint_display_mode,
        );
        self.draw_calls.push(dc);
      }
      RenderableDataRef::Cursor(_) => {
        if self.cursor_call.is_some() {
          return Err(GpuError::InvalidState(
            "[Vulkan] RenderScene:add_renderable cursor call already present",
          ));
        }
        let res: ResourceUploadResult =
          match device.get_cursor_resources(presentation_engine_handle) {
            Ok(r) => r,
            Err(_) => device.create_cursor_resources(cmd_buffer, presentation_engine_handle)?,
          };
        self.cursor_call = Some(CursorDrawCall::from_result_and_matrix(
          res,
          4,
          model_matrix,
          0.05,
        ));
      }
      RenderableDataRef::Markers(component) => {
        let res: ResourceUploadResult =
          match device.get_marker_resources(presentation_engine_handle) {
            Ok(r) => r,
            Err(_) => device.create_marker_resources(cmd_buffer, presentation_engine_handle)?,
          };
        for marker in &component.markers {
          self.marker_calls.push(MarkerDrawCall {
            pipeline: res.pipeline,
            vertex_count: 4,
            model_matrix,
            local_pos: marker.local_pos,
            size: marker.size,
            color: marker.color,
          });
        }
      }
      RenderableDataRef::Measurement(component) => {
        let res: ResourceUploadResult = match device
          .get_measurement_resources(presentation_engine_handle)
        {
          Ok(r) => r,
          Err(_) => device.create_measurement_resources(cmd_buffer, presentation_engine_handle)?,
        };
        self.measurement_calls.push(MeasurementDrawCall {
          pipeline: res.pipeline,
          vertex_count: 6,
          p1: [component.pos1.x(), component.pos1.y(), component.pos1.z()],
          p2: [component.pos2.x(), component.pos2.y(), component.pos2.z()],
          points: 12.0,
          significant_digits: 2, // fallback
        });
      }
      RenderableDataRef::ParticleSystem(component) => {
        let count = component.particles.read().len() as u32;
        if count == 0 {
          return Ok(());
        }
        if component.config.use_particle2 {
          let particle_pipeline = device.get_particle2_pipeline_key(presentation_engine_handle)?;
          self.particle2_calls.push(Particle2DrawCall {
            pipeline: particle_pipeline,
            system_particle_offset: 0,
            system_indirect_offset: 0,
            config: component.config.clone(),
            particles: alloc::sync::Arc::downgrade(&component.particles),
          });
        } else {
          let particle_pipeline = device.get_particle_pipeline_key(presentation_engine_handle)?;
          self.particle_calls.push(ParticleDrawCall {
            pipeline: particle_pipeline,
            system_particle_offset: 0,
            system_indirect_offset: 0,
            config: component.config.clone(),
            particles: alloc::sync::Arc::downgrade(&component.particles),
          });
        }
      }
      RenderableDataRef::Gizmo(_) => {} // Handled elsewhere
    }
    Ok(())
  }
}

/// TODO: Document this item
pub fn do_draw_cursor(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &CursorDrawCall,
  window_extent: [u32; 2],
) -> GpuResult<()> {
  // 2. Bind pipeline
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;

  let push_constants = crate::gpu::CursorPushConstants {
    view: camera.view.into(),
    view_proj: camera.view_proj.into(),
    model: draw_call.model_matrix.into(),
    cursor_size: draw_call.cursor_size,
    _padding: 0.0,
    window_extent: [window_extent[0] as f32, window_extent[1] as f32],
  };

  device.push_cursor_constants(cmd_buffer, &push_constants)?;
  device.draw(cmd_buffer, draw_call.vertex_count)?;

  Ok(())
}

/// TODO: Document this item
pub fn do_draw_marker(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &MarkerDrawCall,
) -> GpuResult<()> {
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;

  // Matrix-Vector multiplication is fast and perfectly correct here
  let global_center = draw_call.model_matrix.mul_vector(
    aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(
      draw_call.local_pos[0],
      draw_call.local_pos[1],
      draw_call.local_pos[2],
      1.0,
    ),
  );

  let push_constants = crate::gpu::MarkerPushConstants {
    view_proj: camera.view_proj.into(),
    center_pos: [global_center.x(), global_center.y(), global_center.z()],
    size: draw_call.size,
    color: draw_call.color,
    camera_up: camera.up,       // <--- Passed directly!
    camera_right: camera.right, // <--- Passed directly!
    _pad0: 0.0,
    _pad1: 0.0,
    _pad2: 0.0,
  };
  device.push_marker_constants(cmd_buffer, &push_constants)?;
  device.draw(cmd_buffer, draw_call.vertex_count)?;

  Ok(())
}

/// TODO: Document this item
pub fn do_draw_measurement(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &MeasurementDrawCall,
) -> GpuResult<()> {
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;

  let push_constants = crate::gpu::MeasurementPushConstants {
    view_proj: camera.view_proj.into(),
    p1: draw_call.p1,
    _pad0: 0.0,
    p2: draw_call.p2,
    _pad1: 0.0,
    camera_up: camera.up,
    _pad2: 0.0,
    camera_right: camera.right,
    _pad3: 0.0,
    color: [1.0, 1.0, 1.0], // White
    _pad4: 0.0,
  };
  device.push_measurement_constants(cmd_buffer, &push_constants)?;
  device.draw(cmd_buffer, draw_call.vertex_count)?;

  Ok(())
}

/// Requires pipeline and descriptor sets to be already bound
pub fn do_draw_gizmo(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &GizmoDrawCall,
) -> GpuResult<()> {
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;

  let push_constants = crate::gpu::GizmoPushConstants {
    view_proj: camera.view_proj.into(),
    scale: draw_call.scale,
    instance_id: draw_call.buffer_index,
  };
  device.push_gizmo_constants(cmd_buffer, &push_constants)?;
  device.set_line_width(cmd_buffer, 1.0)?;
  device.draw(cmd_buffer, draw_call.vertex_count)?;

  Ok(())
}

/// TODO: Document this item
pub fn do_draw_trajectory_batch(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &TrajectoryBatchCall,
  window_extent: [f32; 2],
  handle: PresentationEngineHandle,
) -> GpuResult<()> {
  device.prepare_trajectory_archetype_for_render_and_bind_pipeline(cmd_buffer, handle)?;
  let push_constants = crate::gpu::TrajectoryPushConstants {
    map_ptr: draw_call.map_ptr,
    traj_ptr: draw_call.traj_ptr,
    view_proj: camera.view_proj.into(),
    viewport_size: window_extent,
    _pad: [0.0, 0.0],
  };
  device.push_trajectory_constants(cmd_buffer, &push_constants)?;
  device.draw_instanced(
    cmd_buffer,
    draw_call.total_vertices,
    draw_call.total_segments,
  )?;
  Ok(())
}

/// TODO: Document this item
pub fn do_draw_billboard(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &BillboardDrawCall,
) -> GpuResult<()> {
  if device.check_billboard_texture_id(draw_call.texture_id).is_err() {
    return Ok(());
  }

  // Optimization: Matrix column extraction is significantly faster than vector multiplication.
  // The translation component of a 4x4 matrix is always the final column!
  let center_pos: aethervk_oshal_rlib::math::vector::vec4::Vec4f32 =
    draw_call.model_matrix.column(3).unwrap();

  let (size, is_screen_space) = match draw_call.billboard_type {
    crate::scene::BillboardType::WorldSpace { width, height } => ([width, height], 0),
    crate::scene::BillboardType::ScreenSpace {
      pct_width,
      pct_height,
    } => ([pct_width, pct_height], 1),
  };

  let push_constants = crate::gpu::BillboardPushConstants {
    view_proj: camera.view_proj.into(),
    center_pos: [center_pos.x(), center_pos.y(), center_pos.z()],
    size,
    is_screen_space,
    texture_id: draw_call.texture_id as u32,
    camera_up: camera.up,
    camera_right: camera.right,
    _pad0: 0.0,
    _pad1: 0.0,
    _pad2: 0.0,
  };

  device.push_billboard_constants(cmd_buffer, &push_constants)?;
  device.draw(cmd_buffer, draw_call.vertex_count)?;

  Ok(())
}

/// TODO: Document this item
pub fn do_draw_call2(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  sun_pos: Vec3f32,
  sun_color: [f32; 4],
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &DrawCall,
  window_extent: [f32; 2],
) -> GpuResult<()> {
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;
  device.draw_physical_mesh2(
    cmd_buffer,
    draw_call.pipeline,
    draw_call.buffers,
    camera,
    sun_pos,
    sun_color,
    window_extent,
    draw_call,
  )?;

  if draw_call.draw_outline {
    if let Some(outline_pipeline) = draw_call.outline_pipeline {
      device.bind_pipeline(cmd_buffer, outline_pipeline)?;

      let mut outline_call = draw_call.clone();
      outline_call.emissive_intensity = -1.0;
      outline_call.emissive_color = [
        draw_call.outline_color[0],
        draw_call.outline_color[1],
        draw_call.outline_color[2],
      ];

      device.draw_physical_mesh2(
        cmd_buffer,
        outline_pipeline,
        draw_call.buffers,
        camera,
        sun_pos,
        sun_color,
        window_extent,
        &outline_call,
      )?;
    }
  }

  Ok(())
}

/// TODO: Document this item
pub fn do_draw_call(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  sun_pos: Vec3f32,
  sun_color: [f32; 4],
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &DrawCall,
) -> GpuResult<()> {
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;
  device.bind_buffers(cmd_buffer, draw_call.pipeline, draw_call.buffers)?;

  let model = draw_call.model_matrix;
  let mvp = camera.view_proj * model;
  let push_constants = PushConstants {
    model_view_proj: mvp.into(),
    model: model.into(),
    sun_pos: sun_pos.into(),
    texture_flags: draw_call.texture_flags,
    sun_color,
    camera_pos: camera.pos.into(),
    emissive_intensity: draw_call.emissive_intensity,
    emissive_color: draw_call.emissive_color,
    _unused_pad: 0,
  };
  device.push_constants_mesh(cmd_buffer, &push_constants)?;
  device.draw_indexed(cmd_buffer, draw_call.index_count)?;

  if draw_call.draw_outline {
    if let Some(outline_pipeline) = draw_call.outline_pipeline {
      device.bind_pipeline(cmd_buffer, outline_pipeline)?;
      // Note: same buffers because geometry is identical, only pipeline changes
      // but wait, bind_buffers also requires pipeline_key to identify layout in some engines
      // Let's assume it works or we use the regular pipeline key for bind_buffers
      device.bind_buffers(cmd_buffer, outline_pipeline, draw_call.buffers)?;

      let outline_push = PushConstants {
        model_view_proj: mvp.into(),
        model: model.into(),
        sun_pos: sun_pos.into(),
        texture_flags: draw_call.texture_flags,
        sun_color,
        camera_pos: camera.pos.into(),
        emissive_intensity: draw_call.outline_color[3], // using intensity for alpha? Or just packing color
        emissive_color: [
          draw_call.outline_color[0],
          draw_call.outline_color[1],
          draw_call.outline_color[2],
        ], // Emissive color abused for outline color
        _unused_pad: 0,
      };
      device.push_constants_mesh(cmd_buffer, &outline_push)?;
      device.set_line_width(cmd_buffer, 1.0)?;
      device.draw_indexed(cmd_buffer, draw_call.index_count)?;
    }
  }

  Ok(())
}

/// This is called after render device has already bound descriptor sets (cause we didn't abstract them yet)
/// TODO: Add to the push constant abstraction an abstraction for descriptor sets
pub fn do_draw_sun(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &SunDrawCall,
) -> GpuResult<()> {
  device.prepare_sun_for_render(cmd_buffer, draw_call.entity)?;
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;
  let mvp = camera.view_proj * draw_call.model_matrix;
  let push_constants = SunPushConstants {
    model_view_proj: mvp.into(),
    local_camera_pos: draw_call.local_camera_pos.into(),
    _unused: 0,
  };
  device.push_sun_constants(cmd_buffer, &push_constants)?;
  device.draw(cmd_buffer, draw_call.vertex_count)
}

/// TODO: Document this item
pub fn do_draw_sky(
  device: &dyn RenderDevice,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &SkyDrawCall,
) -> GpuResult<()> {
  let push_constants = SkyPushConstants {
    inv_view_proj: draw_call.inv_view_proj_mat.into(),
  };

  device.prepare_sky_for_render(cmd_buffer)?;
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;
  device.push_sky_constants(cmd_buffer, &push_constants)?;
  device.draw(cmd_buffer, draw_call.vertex_count)
}

/// TODO: Document this item
pub fn do_draw_particle(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &ParticleDrawCall,
) -> GpuResult<()> {
  // Bind pipeline (the descriptor set should have been bound once per frame)
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;

  let push_constants = crate::gpu::ParticlePushConstants {
    view_proj: camera.view_proj.into(),
    camera_up: camera.up,
    time: 0.0,
    camera_right: camera.right,
    seed: 0.0,
    color: draw_call.config.color,
    radius: draw_call.config.particle_radius,
  };

  device.push_constants(
    cmd_buffer,
    crate::gpu::ArchetypeId::Particle,
    &push_constants,
  )?;

  // Notice we don't pass the indirect_buffer as a GpuResourceHandle anymore,
  // we use a specific method that draws from the global mega buffer
  device.draw_particle_indirect(cmd_buffer, draw_call.system_indirect_offset)?;

  Ok(())
}

/// TODO: Document this item
pub fn do_draw_particle2(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &Particle2DrawCall,
  time: f32,
) -> GpuResult<()> {
  // Bind pipeline (the descriptor set should have been bound once per frame)
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;

  let push_constants = crate::gpu::Particle2PushConstants {
    view_proj: camera.view_proj.into(),
    camera_up: camera.up,
    time,
    camera_right: camera.right,
    seed: draw_call.system_particle_offset as f32,
    color: draw_call.config.color,
    radius: draw_call.config.particle_radius,
  };

  device.push_particle2_constants(cmd_buffer, &push_constants)?;

  // Notice we don't pass the indirect_buffer as a GpuResourceHandle anymore,
  // we use a specific method that draws from the global mega buffer
  device.draw_particle2_indirect(cmd_buffer, draw_call.system_indirect_offset)?;

  Ok(())
}

/// TODO: Document this item
pub fn do_draw_grid(
  device: &dyn RenderDevice,
  cmd_buffer: super::CommandBufferHandle,
  camera: &CameraRenderData,
  draw_call: &GridDrawCall,
) -> GpuResult<()> {
  let push_constants = GridPushConstants {
    view_proj: camera.view_proj.into(),
    camera_pos: camera.pos.into(),
    near_plane: camera.near,
    far_plane: camera.far,
    density: draw_call.density,
    _pad1: [0.0, 0.0],
    grid_color: draw_call.grid_color,
    _pad2: 0.0,
  };
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;
  device.push_grid_constants(cmd_buffer, &push_constants)?;
  device.draw(cmd_buffer, draw_call.vertex_count)
}

/// prepare_bvh_archetype_for_render_and_bind_pipeline should have been already called
pub fn do_bvh_draw_call(
  device: &dyn RenderDevice,
  cmd_buffer: super::CommandBufferHandle,
  camera: &CameraRenderData,
  draw_call: &BvhDrawCall,
  mesh_draw_calls: &[DrawCall],
) -> GpuResult<()> {
  let push_constants = draw_call.to_push_constants(mesh_draw_calls, camera).ok_or(
    GpuError::InvalidState("[Render Frame] Couldn't compute BVH push constants"),
  )?;
  device.push_bvh_constants(cmd_buffer, &push_constants)?;
  device.draw(cmd_buffer, draw_call.vertex_count)
}

// TODO: all of the do_draw_* functions should have a rollback mechanism
// TODO trait type for internal command buffer so that you get it once
/// TODO: Document this item
pub fn render_frame(
  device: &dyn RenderDevice,
  cmd_buffer: gpu::CommandBufferHandle,
  render_scene: &gpu::RenderScene,
  handle: PresentationEngineHandle,
) -> GpuResult<()> {
  // First sky and grid
  if let Some(draw_call) = &render_scene.sky_call {
    do_draw_sky(device, cmd_buffer, draw_call)?;
  }
  if let Some(draw_call) = &render_scene.grid_call {
    do_draw_grid(device, cmd_buffer, &render_scene.camera_data, draw_call)?;
  }

  let sun_pos = if let Some(draw_call) = &render_scene.sun_call {
    draw_call.sun_pos()
  } else {
    Vec3f32::zero()
  };
  // TODO setup method (binds pipeline, descriptor set, ...)
  for draw_call in &render_scene.draw_calls {
    if draw_call.use_new_path {
      do_draw_call2(
        device,
        &render_scene.camera_data,
        sun_pos,
        [1.0, 1.0, 1.0, 1.0], // TODO
        cmd_buffer,
        draw_call,
        [
          render_scene.window_extent[0] as f32,
          render_scene.window_extent[1] as f32,
        ],
      )?;
    } else {
      do_draw_call(
        device,
        &render_scene.camera_data,
        sun_pos,
        [1.0, 1.0, 1.0, 1.0], // TODO
        cmd_buffer,
        draw_call,
      )?;
    }
  }

  // End of opaque Stuff, beginning semitransparent/transparent stuff

  // Draw Sun Volume after opaque meshes so it properly blends over them instead of being overwritten
  if let Some(draw_call) = &render_scene.sun_call {
    gpu::frame::do_draw_sun(device, &render_scene.camera_data, cmd_buffer, draw_call)?;
  }

  if !render_scene.particle_calls.is_empty() {
    device.prepare_particle_archetype_for_render_and_bind_pipeline(cmd_buffer, handle)?;
    for particle_call in &render_scene.particle_calls {
      gpu::frame::do_draw_particle(device, &render_scene.camera_data, cmd_buffer, particle_call)?;
    }
  }

  if !render_scene.particle2_calls.is_empty() {
    device.prepare_particle2_archetype_for_render_and_bind_pipeline(cmd_buffer, handle)?;
    let time = (render_scene.time_readings.time as f64 / 1_000_000.0) as f32;
    for particle_call in &render_scene.particle2_calls {
      gpu::frame::do_draw_particle2(
        device,
        &render_scene.camera_data,
        cmd_buffer,
        particle_call,
        time,
      )?;
    }
  }

  if let Some(draw_call) = &render_scene.trajectory_call {
    gpu::frame::do_draw_trajectory_batch(
      device,
      &render_scene.camera_data,
      cmd_buffer,
      draw_call,
      [
        render_scene.window_extent[0] as f32,
        render_scene.window_extent[1] as f32,
      ],
      handle,
    )?;
  }

  // end of scene, begin rendering UI elements

  if let Some(cursor_call) = &render_scene.cursor_call {
    gpu::frame::do_draw_cursor(
      device,
      &render_scene.camera_data,
      cmd_buffer,
      cursor_call,
      render_scene.window_extent,
    )?;
  }

  for marker_call in &render_scene.marker_calls {
    gpu::frame::do_draw_marker(device, &render_scene.camera_data, cmd_buffer, marker_call)?;
  }

  for measurement_call in &render_scene.measurement_calls {
    gpu::frame::do_draw_measurement(
      device,
      &render_scene.camera_data,
      cmd_buffer,
      measurement_call,
    )?;
  }

  if !render_scene.gizmo_calls.is_empty() {
    // bind the descriptor set
    device.prepare_gizmo_archetype_for_render_and_bind_pipeline(cmd_buffer, handle)?;
    for gizmo_call in &render_scene.gizmo_calls {
      gpu::frame::do_draw_gizmo(device, &render_scene.camera_data, cmd_buffer, gizmo_call)?;
    }
  }

  if render_scene.billboard_calls.len() > 0 {
    device.prepare_billboard_archetype_for_render_and_bind_pipeline(cmd_buffer, handle)?;
  }
  for billboard_call in &render_scene.billboard_calls {
    gpu::frame::do_draw_billboard(
      device,
      &render_scene.camera_data,
      cmd_buffer,
      billboard_call,
    )?;
    // TODO draw associated text
  }

  if !render_scene.bvh_draw_calls.is_empty() {
    device.prepare_bvh_archetype_for_render_and_bind_pipeline(cmd_buffer, handle)?;
    for bvh_call in &render_scene.bvh_draw_calls {
      gpu::frame::do_bvh_draw_call(
        device,
        cmd_buffer,
        &render_scene.camera_data,
        bvh_call,
        &render_scene.draw_calls,
      )?
    }
  }

  Ok(())
}
