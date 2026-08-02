//! frame module.

use crate::{
  gpu,
  gpu::{
    GpuResourceHandle, GridPushConstants, PipelineKey, PresentationEngineHandle, RenderDevice,
    RenderDeviceExt, SkyPushConstants, SunPushConstants, TextureFlags, UiBatchCall,
  },
  scene::{CameraComponent, EntityId, TransformComponent},
  types::GpuResult,
};
use aethervk_oshal_rlib::{
  math::{
    matrix::{Matrix, Matrix4, MatrixVectorMul, SquareMatrix, mat4::Mat4x4f32},
    quaternion::Quaternion,
    vector::{
      Vector3, Vector4,
      vec3::Vec3f32,
      vec4::{Quat, Vec4f32},
    },
  },
  os::time::timeus_t,
};
use alloc::vec::Vec;
use function_name::named;

#[derive(Clone, Copy, PartialEq)]
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
  /// Near/far planes for the cursor's depth layer (km for micro, AU for macro)
  pub layer_near: f32,
  pub layer_far: f32,
  pub relative_cam_pos: [f32; 3],
}

impl CursorDrawCall {
  pub fn from_result_and_matrix(
    result: ResourceUploadResult,
    vertex_count: u32,
    model_matrix: Mat4x4f32,
    cursor_size: f32,
    layer_near: f32,
    layer_far: f32,
    relative_cam_pos: [f32; 3],
  ) -> Self {
    Self {
      pipeline: result.pipeline,
      vertex_count,
      model_matrix,
      cursor_size,
      layer_near,
      layer_far,
      relative_cam_pos,
    }
  }
}

pub struct BackgroundDrawCall {
  pub pipeline: PipelineKey,
  pub color_top: [f32; 4],
  pub color_bottom: [f32; 4],
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
  const VERTEX_COUNT_VK: u32 = 3; // single full-screen triangle u2014 no diagonal seam

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

pub struct GizmoDrawCall {
  pub pipeline: PipelineKey,
  pub vertex_count: u32,
  pub scale: f32,
  pub buffer_index: u32,
}

impl GizmoDrawCall {
  const VERTEX_COUNT_VK: u32 = 6;

  pub fn from_values(pipeline: PipelineKey, scale: f32, buffer_index: u32) -> Self {
    Self {
      pipeline,
      vertex_count: Self::VERTEX_COUNT_VK,
      scale,
      buffer_index,
    }
  }
}

pub struct BillboardDrawCall {
  pub pipeline: PipelineKey,
  pub vertex_count: u32,
  pub model_matrix: Mat4x4f32,
  pub texture_id: u64,
  pub billboard_type: crate::scene::BillboardType,
}

impl BillboardDrawCall {
  const VERTEX_COUNT_VK: u32 = 4; // four vertices for triangle strip

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

pub struct SunDrawCall {
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
  pub fn from_model_and_camera(
    model: Mat4x4f32,
    c: &CameraRenderData,
    pipeline_key: PipelineKey,
    entity: EntityId,
    radius: f32,
  ) -> Self {
    let model_inv = model.inverse().unwrap_or_else(|| {
      use aethervk_oshal_rlib::math::vector::vec4::Vec4f32;
      let scale_sq =
        model.x.x() * model.x.x() + model.x.y() * model.x.y() + model.x.z() * model.x.z();
      let inv_scale_sq = if scale_sq > 1e-30 {
        1.0 / scale_sq
      } else {
        0.0
      };
      let mut m = model.clone();
      m.x = Vec4f32::from_components(
        model.x.x() * inv_scale_sq,
        model.y.x() * inv_scale_sq,
        model.z.x() * inv_scale_sq,
        0.0,
      );
      m.y = Vec4f32::from_components(
        model.x.y() * inv_scale_sq,
        model.y.y() * inv_scale_sq,
        model.z.y() * inv_scale_sq,
        0.0,
      );
      m.z = Vec4f32::from_components(
        model.x.z() * inv_scale_sq,
        model.y.z() * inv_scale_sq,
        model.z.z() * inv_scale_sq,
        0.0,
      );
      let tx = model.w.x();
      let ty = model.w.y();
      let tz = model.w.z();
      let wx = -(m.x.x() * tx + m.y.x() * ty + m.z.x() * tz);
      let wy = -(m.x.y() * tx + m.y.y() * ty + m.z.y() * tz);
      let wz = -(m.x.z() * tx + m.y.z() * ty + m.z.z() * tz);
      m.w = Vec4f32::from_components(wx, wy, wz, 1.0);
      m
    });

    let local_camera_pos = Vec3f32(model_inv.mul_vector(c.pos.to_point()));
    Self {
      entity,
      pipeline: pipeline_key,
      model_matrix: model,
      local_camera_pos,
      vertex_count: Self::VERTEX_COUNT_TRIANGLE_STRIP_VK,
      radius,
    }
  }

  pub fn sun_pos(&self) -> Vec3f32 {
    Vec3f32(self.model_matrix.w)
  }
}

pub struct SkyDrawCall {
  pub sky_view_proj: Mat4x4f32,
  pub pipeline: PipelineKey,
  pub inv_view_proj_mat: Mat4x4f32,
  pub vertex_count: u32,
}

impl SkyDrawCall {
  const VERTEX_COUNT_VK: u32 = 3;

  #[named]
  pub fn from_camera(camera_data: &CameraRenderData, pipeline_key: PipelineKey) -> GpuResult<Self> {
    let sky_view_proj = {
      let sky_view = camera_data.view.zeroed_translation();
      camera_data.proj * sky_view
    };
    let inv_view_proj_mat = camera_data.view_proj.inverse().ok_or(crate::gpu_err!(
      "SkyDrawCall: couldn't invert view_proj matrix"
    ))?;

    Ok(Self {
      sky_view_proj,
      pipeline: pipeline_key,
      inv_view_proj_mat,
      vertex_count: Self::VERTEX_COUNT_VK,
    })
  }
}

pub struct GridDrawCall {
  pub pipeline: PipelineKey,
  pub density: f32,
  pub grid_size: f32,
  pub grid_color: [f32; 3],
  pub vertex_count: u32,
}

impl GridDrawCall {
  const VERTEX_COUNT_VK: u32 = 3; // single full-screen triangle u2014 no diagonal seam
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

#[derive(Debug, Clone)]
pub struct CameraRenderData {
  pub pos: Vec3f32,
  pub absolute_pos: Vec3f32,
  pub rot: Quat,
  pub view: Mat4x4f32,
  pub proj: Mat4x4f32,
  pub view_proj: Mat4x4f32,
  pub up: [f32; 3],
  pub right: [f32; 3],
  pub near: f32,
  pub far: f32,
  /// Stored projection parameters so we can rebuild the projection per depth layer.
  pub projection_params: CameraProjectionParams,
  pub window_extent: [u32; 2],
}

/// Cached projection parameters from the CameraComponent, allowing per-layer
/// projection rebuild without fragile column patching.
#[derive(Debug, Clone, Copy)]
pub enum CameraProjectionParams {
  Perspective {
    fov: f32,
    aspect_ratio: f32,
  },
  Orthographic {
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
  },
}

impl CameraRenderData {
  /// Note: camera component should have been updated with presentation engine data
  pub fn new(
    transform: &TransformComponent,
    camera: &CameraComponent,
    frame_scale: f32,
    window_extent: [u32; 2],
  ) -> Self {
    // Extract camera's local axes in world space
    let right = transform.rotation.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
    let up = transform.rotation.rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));
    let forward = transform.rotation.rotate_vector(Vec3f32::from_components(0.0, -1.0, 0.0));

    // RTE View Matrix: Camera is at the center [0,0,0], only rotating.
    let view = Mat4x4f32::look_at_axes(right, forward, up, Vec3f32::from_components(0.0, 0.0, 0.0));

    let near = camera.near_plane() * frame_scale;
    let far = camera.far_plane() * frame_scale;
    let (proj, projection_params) = match camera.projection {
      crate::scene::CameraProjection::Perspective {
        fov, aspect_ratio, ..
      } => (
        Mat4x4f32::perspective_vk_reverse_z(fov, aspect_ratio, near, far),
        CameraProjectionParams::Perspective { fov, aspect_ratio },
      ),
      crate::scene::CameraProjection::Orthographic {
        left,
        right,
        bottom,
        top,
        ..
      } => (
        Mat4x4f32::orthographic_vk_reverse_z(left, right, bottom, top, near, far),
        CameraProjectionParams::Orthographic {
          left,
          right,
          bottom,
          top,
        },
      ),
    };
    let view_proj = proj * view;

    Self {
      // In RTE rendering, the camera is strictly at the origin in the shader's "world" space.
      pos: Vec3f32::from_components(0.0, 0.0, 0.0),
      absolute_pos: transform.position,
      rot: transform.rotation,
      view,
      proj,
      view_proj,
      up: [up.x(), up.y(), up.z()],
      right: [right.x(), right.y(), right.z()],
      near,
      far,
      projection_params,
      window_extent,
    }
  }

  /// Rebuild this camera's projection and view_proj for a specific depth layer's
  /// near/far planes. The view matrix (rotation-only in RTE) is shared across all layers.
  pub fn rebuild_for_layer(&self, layer_near: f32, layer_far: f32) -> Self {
    let proj = match self.projection_params {
      CameraProjectionParams::Perspective { fov, aspect_ratio } => {
        Mat4x4f32::perspective_vk_reverse_z(fov, aspect_ratio, layer_near, layer_far)
      }
      CameraProjectionParams::Orthographic {
        left,
        right,
        bottom,
        top,
      } => Mat4x4f32::orthographic_vk_reverse_z(left, right, bottom, top, layer_near, layer_far),
    };
    let view_proj = proj * self.view;
    Self {
      pos: self.pos,
      absolute_pos: self.absolute_pos,
      rot: self.rot,
      view: self.view,
      proj,
      view_proj,
      up: self.up,
      right: self.right,
      near: layer_near,
      far: layer_far,
      projection_params: self.projection_params,
      window_extent: self.window_extent,
    }
  }
}

#[derive(Clone)]
pub struct TrajectoryBatchCall {
  pub pipeline: PipelineKey,
  pub total_vertices: u32, // (MAX_STEPS + 1) * 2
  pub total_segments: u32, // TOTAL_SEGMENTS_ACROSS_ALL_TRAJECTORIES
  pub map_ptr: u64,
  pub traj_ptr: u64,
}

#[derive(Clone)]
pub struct SphereGizmoBatchCall {
  pub pipeline: PipelineKey,
  pub total_gizmos: u32,
  pub total_vertices: u32,
  pub data_ptr: u64,
}

#[derive(Clone)]
pub struct DustDrawCall {
  pub entity_id: EntityId,
  pub rte_mat: Mat4x4f32,
  pub stream_color: [f32; 4],
  pub chunk_offset: u32,
  pub current_time: u32,
  pub max_ttl: f32, // still 300ths, but float so vertex shader doesn't need to perform conversion
  pub macro_scale: f32,
  pub micro_radius: f32,
  pub num_spots: u32,
  pub dispersion_rate: f32,
}

pub struct RenderLayer {
  pub layer_index: u32,
  pub frame_scale: f32,
  /// Camera position relative to this layer's reference frame origin,
  /// in frame-local units (e.g. km for micro). Used by the grid shader
  /// so that absolutePosXY is numerically precise instead of mixing AU + km.
  pub camera_frame_local_pos: Vec3f32,
  pub near: f32,
  pub far: f32,
  pub draw_calls: Vec<DrawCall>,
  pub billboard_calls: Vec<BillboardDrawCall>,
  pub marker_calls: Vec<MarkerDrawCall>,
  pub measurement_calls: Vec<MeasurementDrawCall>,
  pub gizmo_calls: Vec<GizmoDrawCall>,
  pub sphere_gizmo_batch_call: Option<SphereGizmoBatchCall>,
  pub trajectory_call: Option<TrajectoryBatchCall>,
  pub cursor_call: Option<CursorDrawCall>,
  pub sun_call: Option<SunDrawCall>,
  pub sky_call: Option<SkyDrawCall>,
  pub grid_call: Option<GridDrawCall>,
  pub background_call: Option<BackgroundDrawCall>,
  pub dust_calls: Vec<DustDrawCall>,
}

pub struct RenderScene {
  pub unscaled_time_us: timeus_t,
  pub unscaled_time_delta_us: timeus_t,
  pub camera_data: CameraRenderData,
  pub window_extent: [u32; 2],

  pub depth_layers: Vec<RenderLayer>,

  pub cursor_call: Option<CursorDrawCall>,
  pub ui_call: Option<UiBatchCall>,
  pub text2_call: Option<crate::gpu::Text2BatchCall>,
}

impl RenderScene {
  const START_VEC_CAPACITY: usize = 32;

  pub fn new(
    camera: (TransformComponent, CameraComponent),
    unscaled_time_us: timeus_t,
    unscaled_time_delta_us: timeus_t,
    window_extent: [u32; 2],
  ) -> Self {
    Self {
      window_extent,
      unscaled_time_us,
      unscaled_time_delta_us,
      depth_layers: Vec::with_capacity(Self::START_VEC_CAPACITY),
      camera_data: CameraRenderData::new(&camera.0, &camera.1, 1.0, window_extent),
      cursor_call: None,
      ui_call: None,
      text2_call: None,
    }
  }
}

pub fn do_draw_cursor(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  _handle: PresentationEngineHandle,
  draw_call: &CursorDrawCall,
  window_extent: [u32; 2],
) -> GpuResult<()> {
  // 2. Bind pipeline
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;

  let proj_slice: [f32; 16] = camera.proj.into();
  // In AetherVk's perspective_vk_reverse_z, the vertical focal length (-f) is
  // at column 2, row 1 (index 9) because View +Z maps to Clip -Y.
  let proj_1_1 = proj_slice[9];

  let inv_view = camera.view.inverse().unwrap_or(camera.view);
  let inv_view_slice: [f32; 16] = inv_view.into();
  // Column-major layout: col0=[0..3]=right, col1=[4..7]=backward, col2=[8..11]=up
  let right = [inv_view_slice[0], inv_view_slice[1], inv_view_slice[2]];
  let up = [inv_view_slice[8], inv_view_slice[9], inv_view_slice[10]];

  let push_constants = crate::gpu::CursorPushConstants {
    view_proj: camera.view_proj.into(),
    right_proj11: [right[0], right[1], right[2], proj_1_1],
    screen_y_win_x: [up[0], up[1], up[2], window_extent[0] as f32],
    relative_cam_pos_win_y: [
      draw_call.relative_cam_pos[0],
      draw_call.relative_cam_pos[1],
      draw_call.relative_cam_pos[2],
      window_extent[1] as f32,
    ],
  };

  device.push_cursor_constants(cmd_buffer, &push_constants)?;
  device.draw(cmd_buffer, draw_call.vertex_count)?;

  Ok(())
}

pub fn do_draw_marker(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  _handle: PresentationEngineHandle,
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
  _handle: PresentationEngineHandle,
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
    color: [1.0, 1.0, 1.0], // White
    _pad3: 0.0,
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
  _handle: PresentationEngineHandle,
  draw_call: &GizmoDrawCall,
) -> GpuResult<()> {
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;

  let push_constants = crate::gpu::GizmoPushConstants {
    view_proj: camera.view_proj.into(),
    scale: draw_call.scale,
    instance_id: draw_call.buffer_index,
    _pad: [0; 2],
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
  _handle: PresentationEngineHandle,
  draw_call: &TrajectoryBatchCall,
  window_extent: [f32; 2],
) -> GpuResult<()> {
  device.prepare_trajectory_archetype_for_render_and_bind_pipeline(cmd_buffer)?;
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
  _handle: PresentationEngineHandle, // TODO how is it possible that both screen space and world
  // space don't need this?
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

pub fn do_draw_call2(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  sun_pos: Vec3f32,
  sun_color: [f32; 4],
  cmd_buffer: super::CommandBufferHandle,
  handle: PresentationEngineHandle,
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
    handle,
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
        handle,
        &outline_call,
      )?;
    }
  }

  Ok(())
}

/// This is called after render device has already bound descriptor sets (cause we didn't abstract them yet)
pub fn do_draw_sun(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  _handle: PresentationEngineHandle,
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
  _handle: PresentationEngineHandle,
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

pub fn do_draw_grid(
  device: &dyn RenderDevice,
  cmd_buffer: super::CommandBufferHandle,
  _handle: PresentationEngineHandle,
  camera: &CameraRenderData,
  draw_call: &GridDrawCall,
) -> GpuResult<()> {
  let push_constants = GridPushConstants {
    view_proj: camera.view_proj.into(),
    camera_pos: camera.absolute_pos.into(),
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

pub fn do_draw_sphere_gizmo_batch(
  device: &dyn RenderDevice,
  camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  _handle: PresentationEngineHandle,
  draw_call: &SphereGizmoBatchCall,
) -> GpuResult<()> {
  device.prepare_sphere_gizmo_archetype_for_render_and_bind_pipeline(cmd_buffer)?;
  let push_constants = crate::gpu::SphereGizmoPushConstants {
    view_proj: camera.view_proj.into(),
    gizmo_ptr: draw_call.data_ptr,
    _pad: 0,
  };
  device.push_sphere_gizmo_constants(cmd_buffer, &push_constants)?;
  device.draw_instanced(cmd_buffer, draw_call.total_vertices, draw_call.total_gizmos)?;
  Ok(())
}

pub fn do_draw_ui_batch(
  device: &dyn RenderDevice,
  _camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  _handle: PresentationEngineHandle,
  draw_call: &crate::gpu::UiBatchCall,
  window_extent: [f32; 2],
) -> GpuResult<()> {
  device.prepare_ui_archetype_for_render_and_bind_pipeline(cmd_buffer)?;

  // Standard Vulkan 2D orthographic matrix (Top-Left = 0,0, Bottom-Right = w,h)
  let w = window_extent[0];
  let h = window_extent[1];
  #[rustfmt::skip]
  let proj = [
    [2.0 / w, 0.0, 0.0, 0.0],
    [0.0, 2.0 / h, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, -1.0, 0.0, 1.0],
  ];

  let push_constants = crate::gpu::UiPushConstants {
    elements_ptr: draw_call.elements_ptr,
    _pad0: 0,
    view_proj: proj,
  };
  device.push_ui_constants(cmd_buffer, &push_constants)?;
  device.draw_instanced(cmd_buffer, 6, draw_call.total_elements)?;
  Ok(())
}

pub fn do_draw_text2_batch(
  device: &dyn RenderDevice,
  _camera: &CameraRenderData,
  cmd_buffer: super::CommandBufferHandle,
  _handle: PresentationEngineHandle,
  draw_call: &crate::gpu::Text2BatchCall,
  window_extent: [f32; 2],
) -> GpuResult<()> {
  device.prepare_text2_archetype_for_render_and_bind_pipeline(cmd_buffer)?;

  let w = window_extent[0];
  let h = window_extent[1];
  let proj = [
    [2.0 / w, 0.0, 0.0, 0.0],
    [0.0, 2.0 / h, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [-1.0, -1.0, 0.0, 1.0],
  ];

  let push_constants = crate::gpu::Text2PushConstants {
    glyphs_ptr: draw_call.glyphs_ptr,
    _pad0: 0,
    view_proj: proj,
  };
  device.push_text2_constants(cmd_buffer, &push_constants)?;
  device.draw_instanced(cmd_buffer, 4, draw_call.total_glyphs)?;
  Ok(())
}

pub fn do_draw_background(
  device: &dyn RenderDevice,
  cmd_buffer: super::CommandBufferHandle,
  handle: PresentationEngineHandle,
  draw_call: &BackgroundDrawCall,
) -> GpuResult<()> {
  device.prepare_background_archetype_for_render_and_bind_pipeline(cmd_buffer, handle)?; // Handle not actually used for descriptor sets in this archetype
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;

  let push_constants = crate::gpu::BackgroundPushConstants {
    color_top: draw_call.color_top,
    color_bottom: draw_call.color_bottom,
  };
  device.push_background_constants(cmd_buffer, &push_constants)?;
  device.draw(cmd_buffer, 3)
}

pub fn do_draw_dust_batch(
  device: &crate::gpu_backends::vulkan::device::Device,
  cmd_buffer: gpu::CommandBufferHandle,
  handle: PresentationEngineHandle,
  camera: &CameraRenderData,
  draw_calls: &[DustDrawCall],
) -> GpuResult<()> {
  if draw_calls.is_empty() {
    return Ok(());
  }

  // 1. Fetch the pipeline key dynamically we only want to bind the pipeline once
  let pipeline_key = device.get_pipeline_key(handle, gpu::ArchetypeId::Particles)?;
  device.bind_pipeline(cmd_buffer, pipeline_key)?;

  let cmd = device.get_cmd(cmd_buffer)?;

  for call in draw_calls {
    // Large World Coordinates (LWC) Camera Relative Transformation
    // Proj * ViewRot * RelativeModel
    let view_rot_only = {
      let mut v = camera.view.clone();
      v.w = Vec4f32::from_components(0.0, 0.0, 0.0, 1.0);
      v
    };
    let mvp = camera.proj * view_rot_only * call.rte_mat;
    let mut pc = gpu::new_particles::DustPushConstants {
      global_particle_buffer: 0, // populated by Device
      particle_page_table: 0,    // populated by Device
      view_proj: mvp.into(),
      stream_color: call.stream_color,
      chunk_offset: call.chunk_offset,
      current_time: call.current_time,
      max_ttl: call.max_ttl,
      macro_scale: call.macro_scale,
      micro_radius: call.micro_radius,
      num_spots: call.num_spots,
      dispersion_rate: call.dispersion_rate,
      _pad: 0,
    };

    // 2. Feed the GPU-managed buffers intto the push constant
    // The device uses EntityId as the particle system ID
    match device.complete_graphics_particle_push_constant(call.entity_id.as_ffi(), &mut pc) {
      Ok(indirect_buffer) => {
        // 3. Issue the dispatch
        device.cmd_draw_particle_system(cmd, indirect_buffer, &pc)?;
      }
      Err(e) => {
        aethervk_oshal_rlib::log!(
          "Skipping dust draw call for {:?} due to error: '{}'",
          call.entity_id,
          e
        );
      }
    }
  }
  Ok(())
}

// TODO: all of the do_draw_* functions should have a rollback mechanism
// TODO trait type for internal command buffer so that you get it once
/// TODO: Document this item
pub fn render_frame(
  device: &dyn RenderDevice,
  cmd_buffer: gpu::CommandBufferHandle,
  handle: PresentationEngineHandle,
  render_scene: &gpu::RenderScene,
) -> GpuResult<()> {
  // ── Extract global sun position (RTE) from whichever layer has a SunDrawCall ──
  let global_sun_pos = render_scene
    .depth_layers
    .iter()
    .find_map(|l| l.sun_call.as_ref().map(|s| s.sun_pos()));

  // ── Multi-layer compositing mode (always 3 subpasses now) ─────────
  // Subpass 0: draw the macro layer (layer_index == 0)
  if let Some(macro_layer) = render_scene.depth_layers.iter().find(|l| l.layer_index == 0) {
    draw_layer_content(
      device,
      cmd_buffer,
      handle,
      render_scene,
      macro_layer,
      global_sun_pos,
    )?;
  }

  // Transition to subpass 1 (micro)
  device.next_subpass(cmd_buffer)?;
  device.set_viewport(
    cmd_buffer,
    &gpu::Viewport::from_extent(render_scene.window_extent),
  )?;
  device.set_scissor(
    cmd_buffer,
    &gpu::Rect2D::from_extent(render_scene.window_extent),
  )?;

  // Subpass 1: draw the micro layer (layer_index == 1)
  if let Some(micro_layer) = render_scene.depth_layers.iter().find(|l| l.layer_index == 1) {
    draw_layer_content(
      device,
      cmd_buffer,
      handle,
      render_scene,
      micro_layer,
      global_sun_pos,
    )?;
  }

  // Transition to subpass 2 (composite + UI)
  device.next_subpass(cmd_buffer)?;
  device.set_viewport(
    cmd_buffer,
    &gpu::Viewport::from_extent(render_scene.window_extent),
  )?;
  device.set_scissor(
    cmd_buffer,
    &gpu::Rect2D::from_extent(render_scene.window_extent),
  )?;

  // Draw fullscreen composite triangle to merge macro+micro
  let macro_layer = render_scene.depth_layers.iter().find(|l| l.layer_index == 0);
  let micro_layer = render_scene.depth_layers.iter().find(|l| l.layer_index == 1);
  let constants = gpu::CompositePushConstants {
    macro_near: macro_layer.map(|l| l.near).unwrap_or(0.1),
    macro_far: macro_layer.map(|l| l.far).unwrap_or(1000.0),
    micro_near: micro_layer.map(|l| l.near).unwrap_or(0.001),
    micro_far: micro_layer.map(|l| l.far).unwrap_or(10.0),
    macro_scale: macro_layer.map(|l| l.frame_scale).unwrap_or(1.0),
    micro_scale: micro_layer.map(|l| l.frame_scale).unwrap_or(1.0),
  };
  device.draw_composite(cmd_buffer, handle, &constants)?;

  // Draw cursor after composite (always on top of both layers)
  if let Some(cursor_call) = &render_scene.cursor_call {
    // Rebuild viewProj for the cursor's depth layer so the projection matrix
    // matches the cursor's coordinate space (km for micro, AU for macro).
    let cursor_camera = render_scene
      .camera_data
      .rebuild_for_layer(cursor_call.layer_near, cursor_call.layer_far);
    do_draw_cursor(
      device,
      &cursor_camera,
      cmd_buffer,
      handle,
      cursor_call,
      render_scene.window_extent,
    )?;
  }

  // ── UI / Text (always drawn last, in the final subpass) ────────────
  if let Some(draw_call) = &render_scene.ui_call {
    do_draw_ui_batch(
      device,
      &render_scene.camera_data,
      cmd_buffer,
      handle,
      draw_call,
      [
        render_scene.window_extent[0] as f32,
        render_scene.window_extent[1] as f32,
      ],
    )?;
  }

  if let Some(draw_call) = &render_scene.text2_call {
    do_draw_text2_batch(
      device,
      &render_scene.camera_data,
      cmd_buffer,
      handle,
      draw_call,
      [
        render_scene.window_extent[0] as f32,
        render_scene.window_extent[1] as f32,
      ],
    )?;
  }

  Ok(())
}

/// Draws the content of a single depth layer (all draw calls, gizmos, particles, etc).
/// Extracted to avoid duplicating layer drawing logic between single-subpass and compositing modes.
fn draw_layer_content(
  device: &dyn RenderDevice,
  cmd_buffer: gpu::CommandBufferHandle,
  handle: PresentationEngineHandle,
  render_scene: &gpu::RenderScene,
  layer: &RenderLayer,
  global_sun_pos: Option<Vec3f32>,
) -> GpuResult<()> {
  // Use the global sun position (shared across all layers) if available,
  // otherwise fall back to this layer's own sun or a sensible default.
  let sun_pos = layer
    .sun_call
    .as_ref()
    .map(|s| s.sun_pos())
    .or(global_sun_pos)
    .unwrap_or_else(|| Vec3f32::from_components(100.0, 100.0, 100.0));

  // For micro layers, transform sun position from AU to frame-local units.
  // sun_pos is in RTE AU coordinates; micro layer meshes are in km.
  // Dividing by frame_scale (AU/km) converts AU → km.
  let sun_pos = if layer.layer_index > 0 && layer.frame_scale > 0.0 && layer.frame_scale < 1.0 {
    use aethervk_oshal_rlib::math::vector::Vector3;
    Vec3f32::from_components(
      sun_pos.x() / layer.frame_scale,
      sun_pos.y() / layer.frame_scale,
      sun_pos.z() / layer.frame_scale,
    )
  } else {
    sun_pos
  };

  // Rebuild projection matrix for this layer's near/far planes.
  // The view matrix (rotation-only in RTE) is shared across all layers.
  let layer_camera = render_scene.camera_data.rebuild_for_layer(layer.near, layer.far);

  if let Some(draw_call) = &layer.background_call {
    do_draw_background(device, cmd_buffer, handle, draw_call)?;
  }
  if let Some(draw_call) = &layer.sky_call {
    do_draw_sky(device, cmd_buffer, handle, draw_call)?;
  }
  for draw_call in &layer.draw_calls {
    do_draw_call2(
      device,
      &layer_camera,
      sun_pos,
      [1.0, 1.0, 1.0, 1.0], // TODO
      cmd_buffer,
      handle,
      draw_call,
      [
        render_scene.window_extent[0] as f32,
        render_scene.window_extent[1] as f32,
      ],
    )?;
  }

  if let Some(draw_call) = &layer.grid_call {
    // Use camera_frame_local_pos (computed precisely during scene conversion via
    // get_relative_transform) instead of dividing absolute_pos by frame_scale.
    // This preserves f32 precision for micro layers where global→km conversion
    // would produce values too large for fract() in the grid shader.
    let grid_camera = {
      let mut c = layer_camera.clone();
      c.absolute_pos = layer.camera_frame_local_pos;
      c
    };
    do_draw_grid(device, cmd_buffer, handle, &grid_camera, draw_call)?;
  }

  if !layer.billboard_calls.is_empty() {
    device.prepare_billboard_archetype_for_render_and_bind_pipeline(cmd_buffer)?;
    for billboard_call in &layer.billboard_calls {
      do_draw_billboard(device, &layer_camera, cmd_buffer, handle, billboard_call)?;
    }
  }

  if !layer.gizmo_calls.is_empty() {
    device.prepare_gizmo_archetype_for_render_and_bind_pipeline(cmd_buffer)?;
    for draw_call in &layer.gizmo_calls {
      do_draw_gizmo(device, &layer_camera, cmd_buffer, handle, draw_call)?;
    }
  }

  if let Some(batch_call) = &layer.sphere_gizmo_batch_call {
    do_draw_sphere_gizmo_batch(device, &layer_camera, cmd_buffer, handle, batch_call)?;
  }

  for measurement_call in &layer.measurement_calls {
    do_draw_measurement(device, &layer_camera, cmd_buffer, handle, measurement_call)?;
  }

  for marker_call in &layer.marker_calls {
    do_draw_marker(device, &layer_camera, cmd_buffer, handle, marker_call)?;
  }

  if let Some(draw_call) = &layer.trajectory_call {
    do_draw_trajectory_batch(
      device,
      &layer_camera,
      cmd_buffer,
      handle,
      draw_call,
      [
        render_scene.window_extent[0] as f32,
        render_scene.window_extent[1] as f32,
      ],
    )?;
  }

  // End of opaque Stuff, beginning semitransparent/transparent stuff

  // particles here (transparency)
  if !layer.dust_calls.is_empty() {
    do_draw_dust_batch(
      device.as_any().downcast_ref().unwrap(),
      cmd_buffer,
      handle,
      &layer_camera,
      &layer.dust_calls,
    )?;
  }

  // Draw Sun Volume after opaque meshes so it properly blends over them instead of being overwritten
  if let Some(draw_call) = &layer.sun_call {
    do_draw_sun(device, &layer_camera, cmd_buffer, handle, draw_call)?;
  }

  Ok(())
}
