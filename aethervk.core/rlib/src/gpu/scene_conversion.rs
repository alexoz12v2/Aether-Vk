//! scene_conversion module.

use crate::{
  gpu,
  gpu::{RenderDevice, frame::CameraRenderData},
  math::collision::linear_bvh::LinearBound,
  scene::{
    BackgroundComponent, BillboardType, BvhDebugComponent, CameraComponent, CursorComponent,
    EntityId, FollowingComponent, GridComponent, HiddenComponent, HighResTransformComponent,
    ImageBillboardComponent, MarkersComponent, MeasurementComponent, PhysicalMeshComponent,
    ReferenceFrameComponent, SelectedComponent, SkyComponent, SunComponent, TransformComponent,
  },
  types::GpuResult,
};
use aethervk_oshal_rlib::math::{
  matrix::{Matrix4, MatrixVectorMul, mat4::Mat4x4f32},
  vector::{Vector, Vector3, Vector4, vec3::Vec3f32},
};
use alloc::{string::ToString, vec::Vec};
use function_name::named;

// TODO extensive unit testing. (with valid scenes of course scene.validate)
// TODO first step shouldn't be done in render thread? (cdylib and simulation_test)

/// TODO: Document this item
pub struct PhysicalMeshSceneData {
  entity_id: EntityId,
  mesh: PhysicalMeshComponent,
  global_transform: TransformComponent,
  outline: Option<[f32; 4]>,
  use_new_path: bool,
  paint_display_mode: u32,
  sphere_center: [f32; 3],
  sphere_radius: f32,
  grid_color: [f32; 3],
  grid_density: f32,
}

impl PhysicalMeshSceneData {
  fn new(
    entity_id: EntityId,
    mesh: PhysicalMeshComponent,
    global_transform: TransformComponent,
    outline: Option<[f32; 4]>,
    use_new_path: bool,
    paint_display_mode: u32,
  ) -> Self {
    let sphere_center = mesh.sphere_center;
    let sphere_radius = mesh.sphere_radius;
    let grid_color = mesh.grid_color;
    let grid_density = mesh.grid_density;
    Self {
      entity_id,
      mesh,
      global_transform,
      outline,
      use_new_path,
      paint_display_mode,
      sphere_center,
      sphere_radius,
      grid_color,
      grid_density,
    }
  }
}

pub struct DepthLayerData {
  pub layer_index: u32,
  pub near: f32,
  pub far: f32,
  pub meshes: Vec<PhysicalMeshSceneData>,
  pub billboards: Vec<(Mat4x4f32, u64, BillboardType)>,
  pub markers: Vec<(EntityId, TransformComponent, MarkersComponent)>,
  pub measurements: Vec<(EntityId, Vec3f32, Vec3f32, f32, u32)>,
  pub bvhs: Vec<(
    BvhDebugComponent,
    Vec<LinearBound<f32>>,
    Mat4x4f32,
    EntityId,
  )>,
  pub particles: Vec<(
    EntityId,
    alloc::sync::Weak<spin::RwLock<Vec<crate::scene::particles::ParticleData>>>,
    crate::scene::particles::ParticleEmitterComponent,
  )>,
  pub gizmos: Vec<(EntityId, Mat4x4f32, f32)>,
  pub sphere_gizmos: Vec<(EntityId, Mat4x4f32, f32, f32)>,
  pub trajectories: Vec<(
    EntityId,
    crate::scene::trajectory::TrajectoryComponent,
    Mat4x4f32,
  )>,
  /// Camera position relative to this layer's frame origin, in frame-local units.
  /// For macro layer: camera global pos in AU. For micro: cam pos relative to frame in km.
  pub camera_frame_local_pos: Vec3f32,
}

/// Data extracted from ECS Scene struct. Middleman between [`crate::scene::Scene`]
/// and [`crate::gpu::frame::RenderScene`]
pub struct RenderSceneExtraction {
  pub depth_layers: Vec<DepthLayerData>,
  pub extracted_ui: Vec<crate::gpu::UiElementGpu>,
  pub extracted_texts: Vec<(
    crate::scene::ui::Transform2DComponent,
    crate::scene::ui::ScreenSpaceTextComponent,
  )>,
  pub extracted_background: Option<([f32; 4], [f32; 4], u32, f32)>,
  pub extracted_sky: Option<(u32, f32)>,
  pub extracted_sun: Option<((Mat4x4f32, f32), u32, f32)>,
  pub extracted_grid: Option<(f32, f32, [f32; 3], u32, f32)>,
  pub camera_data: gpu::frame::CameraRenderData,
  pub window_extent: [u32; 2],
  pub cursor_transform: Option<(TransformComponent, u32, f32, [f32; 3])>,
  pub layer_frame_scales: hashbrown::HashMap<u32, f32>,
}

impl RenderSceneExtraction {
  /// Second step of scene conversion to a render scene:
  /// reorganize the extracted data into draw calls
  pub fn build_render_scene(
    self,
    device: &dyn RenderDevice,
    presentation_engine_handle: gpu::PresentationEngineHandle,
    cmd_buffer: gpu::CommandBufferHandle,
    time_readings: aethervk_oshal_rlib::os::time::TimeReadings,
    window_extent: [u32; 2],
    debug_name: &str,
  ) -> GpuResult<gpu::RenderScene> {
    let mut render_scene = gpu::RenderScene {
      time_readings,
      window_extent,
      depth_layers: Vec::with_capacity(self.depth_layers.len()),
      text_calls: Vec::with_capacity(self.extracted_texts.len()),
      camera_data: self.camera_data,
      cursor_call: None,
      ui_call: None,
      text2_call: None,
    };

    for ext_layer in self.depth_layers {
      let mut draw_calls = Vec::with_capacity(ext_layer.meshes.len());
      let mut billboard_calls = Vec::with_capacity(ext_layer.billboards.len());
      let mut marker_calls = Vec::with_capacity(ext_layer.markers.len());
      let mut measurement_calls = Vec::with_capacity(ext_layer.measurements.len());
      let mut bvh_draw_calls = Vec::with_capacity(ext_layer.bvhs.len());
      let mut bvhwire2_data = Vec::with_capacity(ext_layer.bvhs.len());
      let mut gizmo_calls = Vec::with_capacity(ext_layer.gizmos.len());
      let mut particle_calls = Vec::with_capacity(ext_layer.particles.len());
      let mut particle2_calls = Vec::with_capacity(ext_layer.particles.len());

      // Populate Meshes
      for mesh_data in ext_layer.meshes {
        let asset_hash = mesh_data.mesh.mesh.id;
        let res = if mesh_data.use_new_path {
          match device.get_physical_mesh2_resources(asset_hash, presentation_engine_handle) {
            Ok(r) => r,
            Err(_) => {
              if debug_name.contains("MeshViewer") {
                aethervk_oshal_rlib::log!(
                  "[MeshViewer Debug] Creating physical mesh2 resources for entity {:?}",
                  mesh_data.entity_id
                );
              }
              device.create_physical_mesh2_resources(
                cmd_buffer,
                asset_hash,
                &mesh_data.mesh,
                presentation_engine_handle,
                &mesh_data.mesh.asset_path,
              )?
            }
          }
        } else {
          match device.get_physical_mesh_resources(asset_hash, presentation_engine_handle) {
            Ok(r) => r,
            Err(_) => {
              if debug_name.contains("MeshViewer") {
                aethervk_oshal_rlib::log!(
                  "[MeshViewer Debug] Creating physical mesh resources for entity {:?}",
                  mesh_data.entity_id
                );
              }
              device.create_physical_mesh_resources(
                cmd_buffer,
                asset_hash,
                &mesh_data.mesh,
                presentation_engine_handle,
                &mesh_data.mesh.asset_path,
              )?
            }
          }
        };
        let dc = gpu::frame::DrawCall::from_handles_and_matrix(
          res,
          mesh_data.mesh.mesh.indices.len() as u32,
          mesh_data.outline,
          mesh_data.global_transform.to_mat4(),
          mesh_data.mesh.emissive_intensity,
          mesh_data.mesh.emissive_color,
          mesh_data.use_new_path,
          mesh_data.mesh.paint_display_mode,
          mesh_data.mesh.sphere_center,
          mesh_data.mesh.sphere_radius,
          mesh_data.mesh.grid_color,
          mesh_data.mesh.grid_density,
        );
        draw_calls.push(dc);
      }

      // Populate Billboards
      if !ext_layer.billboards.is_empty() {
        let pipeline = match device.get_billboard_resources(presentation_engine_handle) {
          Ok(r) => r.pipeline,
          Err(_) => {
            device
              .create_billboard_resources(cmd_buffer, presentation_engine_handle)?
              .pipeline
          }
        };
        for (mat, texture_id, billboard_type) in ext_layer.billboards {
          billboard_calls.push(gpu::frame::BillboardDrawCall::from_data(
            pipeline,
            mat,
            texture_id,
            billboard_type,
          ));
        }
      }

      // Populate Markers
      if !ext_layer.markers.is_empty() {
        let res = match device.get_marker_resources(presentation_engine_handle) {
          Ok(r) => r,
          Err(_) => device.create_marker_resources(cmd_buffer, presentation_engine_handle)?,
        };
        for (_id, t, markers_comp) in ext_layer.markers {
          let model_matrix = t.to_mat4();
          for marker in markers_comp.markers {
            marker_calls.push(gpu::frame::MarkerDrawCall::from_values(
              res.pipeline,
              model_matrix,
              marker.local_pos,
              marker.size,
              marker.color,
            ));
          }
        }
      }

      // Measurements
      if !ext_layer.measurements.is_empty() {
        let pipeline = match device.get_measurement_resources(presentation_engine_handle) {
          Ok(r) => r.pipeline,
          Err(_) => {
            device
              .create_measurement_resources(cmd_buffer, presentation_engine_handle)?
              .pipeline
          }
        };
        for (_id, p1, p2, points, significant_digits) in ext_layer.measurements {
          measurement_calls.push(gpu::frame::MeasurementDrawCall::from_data_and_pipeline(
            p1,
            p2,
            points,
            significant_digits,
            pipeline,
          ));
        }
      }

      // Gizmos
      if !ext_layer.gizmos.is_empty() {
        let gizmo_resources = match device.get_gizmo_resources(presentation_engine_handle) {
          Ok(r) => r,
          Err(_) => device.create_gizmo_resources(cmd_buffer, presentation_engine_handle)?,
        };
        for (entity_id, mat, scale) in ext_layer.gizmos {
          let gizmo_idx =
            device.update_gizmo_instance(entity_id, mat, presentation_engine_handle)?;
          gizmo_calls.push(gpu::frame::GizmoDrawCall::from_values(
            gizmo_resources.pipeline,
            scale,
            gizmo_idx,
          ));
        }
      }

      // Sphere Gizmos
      let mut sphere_gizmo_batch_call = None;
      if !ext_layer.sphere_gizmos.is_empty() {
        let mut sphere_gizmo_data = Vec::with_capacity(ext_layer.sphere_gizmos.len());
        for (entity_id, model, radius, subdivisions) in ext_layer.sphere_gizmos {
          let idx = device.allocate_sphere_gizmo_instance(entity_id)?;
          sphere_gizmo_data.push((
            idx,
            crate::gpu::SphereGizmoDataGpu {
              model: model.into(),
              radius,
              subdivisions,
              _pad: [0.0, 0.0],
            },
          ));
        }
        sphere_gizmo_batch_call =
          device.upload_sphere_gizmos_batch(cmd_buffer, &sphere_gizmo_data)?;
      }

      // BVH
      let mut bvhwire2_batch_call = None;
      if !ext_layer.bvhs.is_empty() {
        let bvh_pipeline = device.get_bvh_pipeline_kay(presentation_engine_handle)?;
        for (dbg_comp, nodes, global_model, _entity_id) in &ext_layer.bvhs {
          if dbg_comp.use_new_path {
            for node in nodes {
              let (center, extents, ax, ay, az, type_val) = match node {
                crate::math::collision::linear_bvh::LinearBound::AABB(aabb) => {
                  let c = aabb.center::<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>();
                  let e = aabb.half_extents::<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>();
                  (
                    [c.x(), c.y(), c.z()],
                    [e.x(), e.y(), e.z()],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    0.0_f32,
                  )
                }
                crate::math::collision::linear_bvh::LinearBound::OBB(obb) => {
                  let axes = obb.axes::<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>();
                  let c = obb.center::<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>();
                  let e = obb.half_extents::<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>();
                  (
                    [c.x(), c.y(), c.z()],
                    [e.x(), e.y(), e.z()],
                    [axes[0].x(), axes[0].y(), axes[0].z()],
                    [axes[1].x(), axes[1].y(), axes[1].z()],
                    [axes[2].x(), axes[2].y(), axes[2].z()],
                    1.0_f32,
                  )
                }
              };
              bvhwire2_data.push(crate::gpu::Bvhwire2DataGpu {
                center_type: [center[0], center[1], center[2], type_val],
                extents: [extents[0], extents[1], extents[2], 0.0],
                axes_x: [ax[0], ax[1], ax[2], 0.0],
                axes_y: [ay[0], ay[1], ay[2], 0.0],
                axes_z: [az[0], az[1], az[2], 0.0],
              });
            }
          } else {
            for node in nodes {
              match node {
                crate::math::collision::linear_bvh::LinearBound::AABB(_)
                | crate::math::collision::linear_bvh::LinearBound::OBB(_) => {
                  bvh_draw_calls.push(gpu::frame::BvhDrawCall::new(
                    node,
                    bvh_pipeline,
                    *global_model,
                  ));
                }
              }
            }
          }
        }
        bvhwire2_batch_call = device.upload_bvhwire2_batch(cmd_buffer, &bvhwire2_data)?;
      }

      // Particles
      let particle_pipeline = device.get_particle_pipeline_key(presentation_engine_handle)?;
      let particle2_pipeline = device.get_particle2_pipeline_key(presentation_engine_handle)?;
      for (_entity_id, particles, config) in ext_layer.particles {
        if config.use_particle2 {
          particle2_calls.push(gpu::frame::Particle2DrawCall {
            pipeline: particle2_pipeline,
            system_particle_offset: 0,
            system_indirect_offset: 0,
            config,
            particles,
          });
        } else {
          particle_calls.push(gpu::frame::ParticleDrawCall {
            pipeline: particle_pipeline,
            system_particle_offset: 0,
            system_indirect_offset: 0,
            config,
            particles,
          });
        }
      }

      // Trajectories
      let trajectory_call = device.upload_trajectories(cmd_buffer, &ext_layer.trajectories)?;

      render_scene.depth_layers.push(gpu::frame::RenderLayer {
        layer_index: ext_layer.layer_index,
        frame_scale: self.layer_frame_scales.get(&ext_layer.layer_index).copied().unwrap_or(1.0),
        camera_frame_local_pos: ext_layer.camera_frame_local_pos,
        near: ext_layer.near,
        far: ext_layer.far,
        draw_calls,
        billboard_calls,
        marker_calls,
        measurement_calls,
        bvh_draw_calls,
        bvhwire2_data,
        bvhwire2_batch_call,
        gizmo_calls,
        particle_calls,
        particle2_calls,
        sphere_gizmo_batch_call,
        trajectory_call,
        cursor_call: None,
        sun_call: None,
        sky_call: None,
        background_call: None,
        grid_call: None,
      });
    }

    // Cursor — drawn in composite subpass (always on top of all layers)
    if let Some((t, layer_idx, _scale, relative_cam_pos)) = self.cursor_transform {
      let res = match device.get_cursor_resources(presentation_engine_handle) {
        Ok(r) => r,
        Err(_) => device.create_cursor_resources(cmd_buffer, presentation_engine_handle)?,
      };
      // Look up near/far for the cursor's depth layer so the viewProj matches its coordinate space
      let (cursor_near, cursor_far) = render_scene
        .depth_layers
        .iter()
        .find(|l| l.layer_index == layer_idx)
        .map(|l| (l.near, l.far))
        .unwrap_or((render_scene.camera_data.near, render_scene.camera_data.far));
      render_scene.cursor_call = Some(gpu::frame::CursorDrawCall::from_result_and_matrix(
        res,
        4,
        t.to_mat4(),
        t.scale.x(),
        cursor_near,
        cursor_far,
        relative_cam_pos,
      ));
    }

    // Sun
    if let Some(((global_model, radius), layer_idx, _frame_scale)) = self.extracted_sun {
      if let Some(layer) = render_scene.depth_layers.iter_mut().find(|l| l.layer_index == layer_idx)
      {
        let pipeline = device.get_sun_pipeline_key(presentation_engine_handle)?;

        let sun_camera = render_scene.camera_data.rebuild_for_layer(layer.near, layer.far);

        layer.sun_call = Some(gpu::frame::SunDrawCall::from_model_and_camera(
          global_model,
          &sun_camera,
          pipeline,
          crate::scene::EntityId::default(),
          radius,
        )?);
      }
    }

    // Sky
    if let Some((layer_idx, scale)) = self.extracted_sky {
      if let Some(layer) = render_scene.depth_layers.iter_mut().find(|l| l.layer_index == layer_idx)
      {
        let pipeline = device.get_sky_pipeline_key(presentation_engine_handle)?;

        let sky_camera = render_scene.camera_data.rebuild_for_layer(layer.near, layer.far);

        layer.sky_call = Some(gpu::frame::SkyDrawCall::from_camera(&sky_camera, pipeline)?);
      }
    }

    // Background
    if let Some((color_top, color_bottom, layer_idx, _frame_scale)) = self.extracted_background {
      if let Some(layer) = render_scene.depth_layers.iter_mut().find(|l| l.layer_index == layer_idx)
      {
        let pipeline = device.get_background_pipeline_key(presentation_engine_handle)?;
        layer.background_call = Some(gpu::frame::BackgroundDrawCall {
          color_top,
          color_bottom,
          pipeline,
        });
      }
    }

    // Grid — create for macro layer, then clone into micro layers
    if let Some((density, grid_size, grid_color, layer_idx, _frame_scale)) = self.extracted_grid {
      let pipeline = device.get_grid_pipeline_kay(presentation_engine_handle)?;
      // Macro layer
      if let Some(layer) = render_scene.depth_layers.iter_mut().find(|l| l.layer_index == layer_idx)
      {
        layer.grid_call = Some(gpu::frame::GridDrawCall::new(
          pipeline, density, grid_size, grid_color,
        ));
      }
      // Clone grid into any micro layers that don't already have one.
      // The grid shader's LOD system (log₁₀-based) adapts automatically to the
      // micro layer's km-scale distances.
      for layer in render_scene.depth_layers.iter_mut() {
        if layer.layer_index > 0 && layer.grid_call.is_none() {
          layer.grid_call = Some(gpu::frame::GridDrawCall::new(
            pipeline, density, grid_size, grid_color,
          ));
        }
      }
    }

    // UI
    render_scene.ui_call = device.upload_ui(cmd_buffer, &self.extracted_ui)?;

    // Texts
    if !self.extracted_texts.is_empty() {
      let mut text_batch = Vec::new();

      for (t2d, text_comp) in self.extracted_texts {
        let descriptor_index = device.allocate_rasterized_font_atlas(
          cmd_buffer,
          text_comp.font_hash,
          text_comp.font_atlas.clone(),
        )?;

        if text_comp.use_new_path {
          let style = crate::scene::text::TextStyle {
            size_pt: text_comp.points,
            color: text_comp.color,
            style_flags: text_comp.style_flags,
          };

          crate::scene::text::push_text_to_batch(
            &text_comp.text,
            [t2d.global_bounds[0], t2d.global_bounds[1]],
            &style,
            &text_comp.font_atlas,
            descriptor_index,
            &mut text_batch,
          );
        } else {
          render_scene.text_calls.push(gpu::frame::TextDrawCall {
            text: text_comp.text.clone(),
            font_atlas: text_comp.font_atlas.clone(),
            font_id: (text_comp.font_hash, descriptor_index),
            start_cursor_position: [t2d.global_bounds[0], t2d.global_bounds[1]],
            desired_points: text_comp.points,
            color: text_comp.color,
          });
        }
      }

      if !text_batch.is_empty() {
        render_scene.text2_call = device.upload_text2(cmd_buffer, &text_batch)?;
      }
    }

    // ... More components here

    Ok(render_scene)
  }
}

/// TODO: Document this item
pub trait SceneConversionExt {
  /// First step of scene to render scene conversion
  /// query the ECS scene to gather rendering data
  fn convert_scene(
    &self,
    camera_entity: EntityId,
    render_outline: bool,
    pool: Option<&aethervk_oshal_rlib::os::pool::ThreadPool>,
    window_extent: [u32; 2],
  ) -> GpuResult<RenderSceneExtraction>;
}

impl SceneConversionExt for crate::scene::Scene {
  #[named]
  fn convert_scene(
    &self,
    camera_entity: EntityId,
    render_outline: bool,
    pool: Option<&aethervk_oshal_rlib::os::pool::ThreadPool>,
    window_extent: [u32; 2],
  ) -> GpuResult<RenderSceneExtraction> {
    crate::scene::ui::update_ui_layouts(self, [window_extent[0] as f32, window_extent[1] as f32]);

    const START_VEC_CAPACITY: usize = 32;
    let mut extracted_meshes: Vec<PhysicalMeshSceneData> = Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_markers: Vec<(EntityId, TransformComponent, MarkersComponent)> =
      Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_billboards: Vec<(EntityId, Mat4x4f32, u64, BillboardType)> =
      Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_measurements: Vec<(EntityId, Vec3f32, Vec3f32, f32, u32)> =
      Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_bvhs: Vec<(
      BvhDebugComponent,
      Vec<LinearBound<f32>>,
      Mat4x4f32,
      EntityId,
    )> = Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_particles: Vec<(
      EntityId,
      alloc::sync::Weak<spin::RwLock<Vec<crate::scene::particles::ParticleData>>>,
      crate::scene::particles::ParticleEmitterComponent,
    )> = Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_gizmos: Vec<(EntityId, Mat4x4f32, f32)> =
      Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_sphere_gizmos: Vec<(EntityId, Mat4x4f32, f32, f32)> =
      Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_trajectories: Vec<(
      EntityId,
      crate::scene::trajectory::TrajectoryComponent,
      Mat4x4f32,
    )> = Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_ui: Vec<crate::gpu::UiElementGpu> = Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_texts: Vec<(
      crate::scene::ui::Transform2DComponent,
      crate::scene::ui::ScreenSpaceTextComponent,
    )> = Vec::with_capacity(START_VEC_CAPACITY);

    let cursor_transform: Option<(TransformComponent, u32, f32, [f32; 3])>;
    let extracted_sky: Option<(u32, f32)>;
    let mut extracted_sun: Option<((Mat4x4f32, f32), u32, f32)>;
    let extracted_grid: Option<(f32, f32, [f32; 3], u32, f32)>;
    let extracted_background: Option<([f32; 4], [f32; 4], u32, f32)>;

    let camera_data: CameraRenderData;

    // Camera — read f64 HighResTransformComponent, downcast to f32 for GPU
    let cam_transform = self
      .with_component(camera_entity, |h: &HighResTransformComponent| {
        h.to_transform()
      })
      .or_else(|| self.global_transform(camera_entity))
      .unwrap_or_default();
    let cam_comp = self.with_component(camera_entity, |c: &CameraComponent| *c).ok_or(
      crate::gpu_invalid_arg!("[ Scene has no camera component in the specified entity"),
    )?;

    // We get the base scale of the camera's frame
    let frame_scale = self.ancestor_frame_scale(camera_entity);
    camera_data = CameraRenderData::new(&cam_transform, &cam_comp, frame_scale);

    let hidden_roots = if let Some(p) = pool {
      self.query1_res_par::<HiddenComponent, _, _>(p, |id, _| Some(id))
    } else {
      self.query1_res::<HiddenComponent, _, _>(|id, _| Some(id))
    };

    let mut hidden_set = hashbrown::HashSet::new();
    for (root_id, _) in hidden_roots {
      self.traverse_dfs_pre_order(
        root_id,
        &mut hidden_set,
        &|_, _| true,
        &mut |_, child_id, set| {
          set.insert(child_id);
          true
        },
      );
    }

    // Cursor — use f64 relative transform to prevent precision loss at extreme zoom
    cursor_transform = self
      .query1_first_res_without::<_, HiddenComponent, _, _>(|id, _c: &CursorComponent| {
        if hidden_set.contains(&id) {
          return None;
        }

        let camera_global =
          self
            .global_transform_f64(camera_entity)
            .unwrap_or_else(|| HighResTransformComponent {
              position: Default::default(),
              rotation: Default::default(),
              scale: Vec3f32::from_components(1.0, 1.0, 1.0),
            });
        let cursor_global =
          self.global_transform_f64(id).unwrap_or_else(|| HighResTransformComponent {
            position: Default::default(),
            rotation: Default::default(),
            scale: Vec3f32::from_components(1.0, 1.0, 1.0),
          });
        let rel_pos = camera_global.position - cursor_global.position;
        let relative_cam_pos = [rel_pos.x() as f32, rel_pos.y() as f32, rel_pos.z() as f32];

        self.get_relative_transform_f64(id, camera_entity).map(|hrt| {
          (
            TransformComponent {
              position: hrt.position.to_f32(),
              rotation: hrt.rotation,
              scale: hrt.scale,
            },
            self.ancestor_depth_layer(id),
            self.ancestor_frame_scale(id),
            relative_cam_pos,
          )
        })
      })
      .map(|(t, _id)| t);

    // Meshes
    let should_par = self.should_parallelize() && pool.is_some();
    if should_par {
      let results = self.query1_res_without_par::<PhysicalMeshComponent, HiddenComponent, _, _>(
        pool.unwrap(),
        |id, mesh| {
          if hidden_set.contains(&id) {
            return None;
          }
          if let Some(t) = self.get_relative_transform(id, camera_entity) {
            let mesh_clone = mesh.clone();
            let is_selected: bool = self.has_component::<SelectedComponent>(id).into();
            let is_following: bool = self.has_component::<FollowingComponent>(id).into();
            let outline = get_mesh_outline(is_selected, is_following, render_outline);
            let m = PhysicalMeshSceneData::new(
              id,
              mesh_clone,
              t,
              outline,
              mesh.use_new_path,
              mesh.paint_display_mode,
            );
            // BVH debug rendering
            let bvh = m.mesh.mesh.bvh.as_ref().map(|bvh| &bvh.nodes).and_then(|nodes| {
              let mut dbg_comp = None;
              self.with_component(id, |dbg: &BvhDebugComponent| {
                dbg_comp = Some(dbg.clone());
              });
              dbg_comp.map(|c| (nodes, c))
            });

            let bvh_data = if let Some((nodes, comp)) = bvh {
              let mut extracted = Vec::with_capacity(nodes.len());
              comp
                .node_render_states
                .iter()
                .zip(nodes.iter())
                .filter(|&(show, _)| *show)
                .for_each(|(_, node)| {
                  extracted.push(node.bound.clone());
                });
              Some((comp, extracted, t.to_mat4(), id))
            } else {
              None
            };

            Some((m, bvh_data))
          } else {
            None
          }
        },
      );
      for ((m, bvh_data), _) in results {
        if let Some(bvh) = bvh_data {
          extracted_bvhs.push(bvh);
        }
        extracted_meshes.push(m);
      }
    } else {
      self.query1_without::<_, HiddenComponent, _>(|id, mesh: &PhysicalMeshComponent| {
        if hidden_set.contains(&id) {
          return;
        }
        if let Some(t) = self.get_relative_transform(id, camera_entity) {
          let mesh_clone = mesh.clone();
          let is_selected: bool = self.has_component::<SelectedComponent>(id).into();
          let is_following: bool = self.has_component::<FollowingComponent>(id).into();
          let outline = get_mesh_outline(is_selected, is_following, render_outline);
          let m = PhysicalMeshSceneData::new(
            id,
            mesh_clone,
            t,
            outline,
            mesh.use_new_path,
            mesh.paint_display_mode,
          );
          // BVH debug rendering
          let bvh = m.mesh.mesh.bvh.as_ref().map(|bvh| &bvh.nodes).and_then(|nodes| {
            let mut dbg_comp = None;
            self.with_component(id, |dbg: &BvhDebugComponent| {
              dbg_comp = Some(dbg.clone());
            });
            dbg_comp.map(|c| (nodes, c))
          });
          if let Some((nodes, comp)) = bvh {
            let mut extracted = Vec::with_capacity(nodes.len());
            comp
              .node_render_states
              .iter()
              .zip(nodes.iter())
              .filter(|&(show, _)| *show)
              .for_each(|(_, node)| {
                extracted.push(node.bound.clone());
              });
            extracted_bvhs.push((comp, extracted, t.to_mat4(), id));
          }
          extracted_meshes.push(m);
        }
      });
    }

    // Static Meshes
    if should_par {
      let results = self
        .query1_res_without_par::<crate::scene::StaticMeshComponent, HiddenComponent, _, _>(
          pool.unwrap(),
          |id, mesh| {
            if hidden_set.contains(&id) {
              return None;
            }
            if let Some(t) = self.get_relative_transform(id, camera_entity) {
              let is_selected: bool = self.has_component::<SelectedComponent>(id).into();
              let is_following: bool = self.has_component::<FollowingComponent>(id).into();
              let outline = get_mesh_outline(is_selected, is_following, render_outline);
              let pseudo_mesh = PhysicalMeshComponent {
                asset_path: mesh.asset_path.clone(),
                mesh: mesh.mesh.clone(),
                emissive_intensity: mesh.emissive_color[3],
                emissive_color: [
                  mesh.emissive_color[0],
                  mesh.emissive_color[1],
                  mesh.emissive_color[2],
                ],
                use_new_path: true,
                paint_display_mode: 0,
                sphere_center: [0.0, 0.0, 0.0],
                sphere_radius: 1.0,
                grid_color: [0.0, 0.0, 0.0],
                grid_density: 0.0,
              };
              let m = PhysicalMeshSceneData::new(id, pseudo_mesh, t, outline, true, 0);
              Some(m)
            } else {
              None
            }
          },
        );
      for (m, _) in results {
        extracted_meshes.push(m);
      }
    } else {
      self.query1_without::<_, HiddenComponent, _>(
        |id, mesh: &crate::scene::StaticMeshComponent| {
          if hidden_set.contains(&id) {
            return;
          }
          if let Some(t) = self.get_relative_transform(id, camera_entity) {
            let is_selected: bool = self.has_component::<SelectedComponent>(id).into();
            let is_following: bool = self.has_component::<FollowingComponent>(id).into();
            let outline = get_mesh_outline(is_selected, is_following, render_outline);
            let pseudo_mesh = PhysicalMeshComponent {
              asset_path: mesh.asset_path.clone(),
              mesh: mesh.mesh.clone(),
              emissive_intensity: mesh.emissive_color[3],
              emissive_color: [
                mesh.emissive_color[0],
                mesh.emissive_color[1],
                mesh.emissive_color[2],
              ],
              use_new_path: true,
              paint_display_mode: 0,
              sphere_center: [0.0, 0.0, 0.0],
              sphere_radius: 1.0,
              grid_color: [0.0, 0.0, 0.0],
              grid_density: 0.0,
            };
            let m = PhysicalMeshSceneData::new(id, pseudo_mesh, t, outline, true, 0);
            extracted_meshes.push(m);
          }
        },
      );
    }

    // Markers
    if should_par {
      let results = self.query1_res_without_par::<MarkersComponent, HiddenComponent, _, _>(
        pool.unwrap(),
        |id, m| {
          if hidden_set.contains(&id) {
            return None;
          }
          self.get_relative_transform(id, camera_entity).map(|t| (id, t, m.clone()))
        },
      );
      for (res, _) in results {
        extracted_markers.push(res);
      }
    } else {
      self.query1_without::<_, HiddenComponent, _>(|id, m: &MarkersComponent| {
        if hidden_set.contains(&id) {
          return;
        }
        if let Some(t) = self.get_relative_transform(id, camera_entity) {
          extracted_markers.push((id, t, m.clone()));
        }
      });
    }

    // Measurements
    if should_par {
      let results = self.query1_res_without_par::<MeasurementComponent, HiddenComponent, _, _>(
        pool.unwrap(),
        |id, m| {
          if hidden_set.contains(&id) {
            return None;
          }
          self.get_relative_transform(id, camera_entity).map(|t| {
            let mat: Mat4x4f32 = t.to_mat4();
            let p1 = Vec3f32(mat.mul_vector(m.pos1.to_point()));
            let p2 = Vec3f32(mat.mul_vector(m.pos2.to_point()));
            (id, p1, p2, m.points, m.significant_digits)
          })
        },
      );
      for (res, _) in results {
        extracted_measurements.push(res);
      }
    } else {
      self.query1_without::<_, HiddenComponent, _>(|id, m: &MeasurementComponent| {
        if hidden_set.contains(&id) {
          return;
        }
        if let Some(t) = self.get_relative_transform(id, camera_entity) {
          let mat: Mat4x4f32 = t.to_mat4();
          let p1 = Vec3f32(mat.mul_vector(m.pos1.to_point()));
          let p2 = Vec3f32(mat.mul_vector(m.pos2.to_point()));
          extracted_measurements.push((id, p1, p2, m.points, m.significant_digits));
        }
      });
    }

    // Billboards
    if should_par {
      let results = self.query1_res_without_par::<ImageBillboardComponent, HiddenComponent, _, _>(
        pool.unwrap(),
        |id, i| {
          if hidden_set.contains(&id) {
            return None;
          }
          self.get_relative_transform(id, camera_entity).map(|t| {
            let mat: Mat4x4f32 = t.to_mat4();
            (id, mat, i.texture_id, i.billboard_type)
          })
        },
      );
      for (res, _) in results {
        extracted_billboards.push(res);
      }
    } else {
      self.query1_without::<_, HiddenComponent, _>(|id, i: &ImageBillboardComponent| {
        if hidden_set.contains(&id) {
          return;
        }
        if let Some(t) = self.get_relative_transform(id, camera_entity) {
          let mat: Mat4x4f32 = t.to_mat4();
          extracted_billboards.push((id, mat, i.texture_id, i.billboard_type));
        }
      });
    }

    let mut sun_entity_id: Option<EntityId> = None;
    extracted_sun = self
      .query2_first_res_without::<_, _, HiddenComponent, _, _>(
        |id, _t: &TransformComponent, s: &SunComponent| {
          if hidden_set.contains(&id) {
            return None;
          }
          self.get_relative_transform(id, camera_entity).map(|t| {
            (
              (t.to_mat4::<Mat4x4f32>(), s.radius),
              self.ancestor_depth_layer(id),
              self.ancestor_frame_scale(id),
            )
          })
        },
      )
      .map(|(r, id)| {
        sun_entity_id = Some(id);
        r
      });

    // Recompute sun model matrix if camera and sun are in different depth layers
    if let (Some(sun_data), Some(s_id)) = (&mut extracted_sun, sun_entity_id) {
      let sun_layer = sun_data.1;
      let cam_layer = self.ancestor_depth_layer(camera_entity);
      if sun_layer != cam_layer {
        // Sun and camera in different frames — recompute using global coordinates
        if let (Some(sun_global), Some(cam_g)) = (
          self.global_transform(s_id),
          self.global_transform(camera_entity),
        ) {
          use crate::scene::TransformComponent;
          use aethervk_oshal_rlib::math::vector::{Vector, Vector3};
          let safe_div = |a: f32, b: f32| -> f32 { if b.abs() < 1e-15 { 0.0 } else { a / b } };
          let corrected = TransformComponent {
            position: sun_global.position - cam_g.position,
            rotation: sun_global.rotation,
            scale: Vec3f32::from_components(
              safe_div(sun_global.scale.x(), cam_g.scale.x()),
              safe_div(sun_global.scale.y(), cam_g.scale.y()),
              safe_div(sun_global.scale.z(), cam_g.scale.z()),
            ),
          };
          sun_data.0.0 = corrected.to_mat4::<Mat4x4f32>();
        }
      }
    }

    extracted_sky = self
      .query1_first_res_without::<_, HiddenComponent, _, _>(|id, _s: &SkyComponent| {
        if hidden_set.contains(&id) {
          return None;
        }
        Some((self.ancestor_depth_layer(id), self.ancestor_frame_scale(id)))
      })
      .map(|(r, _id)| r);

    extracted_grid = self
      .query1_first_res_without::<_, HiddenComponent, _, _>(|id, _s: &GridComponent| {
        if hidden_set.contains(&id) {
          return None;
        }
        // Scale down the grid by 5000x (increase density to 500.0, cell size = 0.0004 AU)
        let density: f32 = 500.0;
        let grid_size: f32 = 1.0;
        let grid_color: [f32; 3] = [0.5, 0.5, 0.5];
        Some((
          density,
          grid_size,
          grid_color,
          self.ancestor_depth_layer(id),
          self.ancestor_frame_scale(id),
        ))
      })
      .map(|(r, _id)| r);

    // Particles
    self.query2::<crate::scene::particles::ParticleSystemComponent, crate::scene::particles::ParticleEmitterComponent, _>(
      |id, sys, config| {
        if hidden_set.contains(&id) {
          return;
        }
        extracted_particles.push((
          id,
          alloc::sync::Arc::downgrade(&sys.particles),
          config.clone(),
        ));
      },
    );

    // Gizmos — use f64 relative transform to prevent wobble from f32 cancellation
    self.query1_without::<_, HiddenComponent, _>(|id, gizmo: &crate::scene::GizmoComponent| {
      if hidden_set.contains(&id) {
        return;
      }
      if gizmo.gizmo_visible {
        if let Some(hrt) = self.get_relative_transform_f64(id, camera_entity) {
          let pos_f32 = hrt.position.to_f32();
          let t_mat = aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::translation(pos_f32)
            * aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::from_quat_custom_frame(
              hrt.rotation,
            );

          extracted_gizmos.push((id, t_mat, gizmo.gizmo_scale));
        }
      }
    });

    // SphereGizmos — use f64 relative transform to prevent wobble from f32 cancellation
    self.query1_without::<_, HiddenComponent, _>(
      |id, gizmo: &crate::scene::SphereGizmoComponent| {
        if hidden_set.contains(&id) {
          return;
        }
        if gizmo.is_visible {
          if let Some(hrt) = self.get_relative_transform_f64(id, camera_entity) {
            let t = TransformComponent {
              position: hrt.position.to_f32(),
              rotation: hrt.rotation,
              scale: hrt.scale,
            };
            let gizmo_model = t.to_mat4::<Mat4x4f32>() * gizmo.local_frame;
            extracted_sphere_gizmos.push((id, gizmo_model, gizmo.radius, gizmo.subdivisions));
          }
        }
      },
    );

    // Trajectories — use f64 relative transform to prevent precision loss
    self.query1_without::<_, HiddenComponent, _>(
      |id, traj: &crate::scene::trajectory::TrajectoryComponent| {
        if hidden_set.contains(&id) {
          return;
        }
        if let Some(hrt) = self.get_relative_transform_f64(id, camera_entity) {
          let t = TransformComponent {
            position: hrt.position.to_f32(),
            rotation: hrt.rotation,
            scale: hrt.scale,
          };
          extracted_trajectories.push((id, traj.clone(), t.to_mat4()));
        }
      },
    );

    // UI
    let mut ui_items = if should_par {
      self.query2_res_par::<crate::scene::ui::Transform2DComponent, crate::scene::ui::UiComponent, _, _>(
        pool.unwrap(),
        |id, t2d, ui| {
          if hidden_set.contains(&id) { return None; }
          Some((id, *t2d, ui.clone()))
        },
      )
    } else {
      self
        .query2_res::<crate::scene::ui::Transform2DComponent, crate::scene::ui::UiComponent, _, _>(
          |id, t2d, ui| {
            if hidden_set.contains(&id) {
              return None;
            }
            Some((id, *t2d, ui.clone()))
          },
        )
    };

    ui_items.sort_unstable_by(|a, b| {
      a.0
        .1
        .global_depth
        .cmp(&b.0.1.global_depth)
        .then(a.0.1.local_z_index.cmp(&b.0.1.local_z_index))
    });

    for ((_, t2d, ui), _) in ui_items {
      let mut flags = 0;
      if t2d.global_clip[0] > -9999.0 {
        flags |= crate::gpu::UI_FLAG_HAS_CLIP;
      }
      extracted_ui.push(crate::gpu::UiElementGpu {
        bounds: t2d.global_bounds,
        clip_rect: t2d.global_clip,
        color_start: ui.color_start,
        color_end: ui.color_end,
        color_border: ui.color_border,
        color_shadow: ui.color_shadow,
        border_radius: ui.border_radius,
        shadow_params: ui.shadow_params,
        gradient_dir: ui.gradient_dir,
        border_width: ui.border_width,
        texture_id: ui.texture_id,
        flags,
        opacity: ui.opacity,
        rotation: t2d.rotation,
        _pad: 0,
      });
    }

    // Texts
    let mut text_items = if should_par {
      self.query2_res_par::<crate::scene::ui::Transform2DComponent, crate::scene::ui::ScreenSpaceTextComponent, _, _>(
        pool.unwrap(),
        |id, t2d, txt| {
          if hidden_set.contains(&id) { return None; }
          Some((id, *t2d, txt.clone()))
        },
      )
    } else {
      self
        .query2_res::<crate::scene::ui::Transform2DComponent, crate::scene::ui::ScreenSpaceTextComponent, _, _>(
          |id, t2d, txt| {
            if hidden_set.contains(&id) {
              return None;
            }
            Some((id, *t2d, txt.clone()))
          },
        )
    };

    text_items.sort_unstable_by(|a, b| {
      a.0
        .1
        .global_depth
        .cmp(&b.0.1.global_depth)
        .then(a.0.1.local_z_index.cmp(&b.0.1.local_z_index))
    });

    for ((_, t2d, txt), _) in text_items {
      extracted_texts.push((t2d, txt));
    }

    extracted_background = self
      .query1_first_res_without::<_, HiddenComponent, _, _>(|id, b: &BackgroundComponent| {
        if hidden_set.contains(&id) {
          return None;
        }
        Some((
          b.color_top,
          b.color_bottom,
          self.ancestor_depth_layer(id),
          self.ancestor_frame_scale(id),
        ))
      })
      .map(|(r, _id)| r);

    // ... More components here

    // ── Pre-compute per-layer near/far bounds from ReferenceFrameComponents ──
    // With per-layer virtual camera, each layer's coordinates are in its own space:
    // • Macro (layer 0): root space (AU). Near/far in AU.
    // • Micro (layer N): frame-local space. Near/far in that frame's units (e.g. km).
    //
    // The macro layer always uses frame_scale=1.0 (root), not the camera's actual
    // frame_scale, because mesh model matrices are computed in global AU.
    let macro_near = cam_comp.near_plane(); // in AU (root space, scale=1.0)
    let macro_far = cam_comp.far_plane(); // in AU
    let mut layer_bounds: hashbrown::HashMap<u32, (f32, f32)> = hashbrown::HashMap::new();
    let mut layer_frame_scales: hashbrown::HashMap<u32, f32> = hashbrown::HashMap::new();
    layer_bounds.insert(0, (macro_near, macro_far));
    layer_frame_scales.insert(0, 1.0);

    self.query1_without::<ReferenceFrameComponent, HiddenComponent, _>(
      |id, frame: &ReferenceFrameComponent| {
        if frame.depth_layer > 0 {
          // For micro layers, compute bounds in the frame's LOCAL coordinate space.
          if let Some(cam_in_frame) = self.get_relative_transform(camera_entity, id) {
            use aethervk_oshal_rlib::math::vector::Vector;
            // get_relative_transform(camera, frame) gives camera pos relative to frame
            // in the frame's LOCAL coordinate space (because it divides by the frame's scale).
            let dist_local = cam_in_frame.position.length(); // in km (frame-local)

            // SOI radius is in AU (parent coordinate space). Convert to frame-local units:
            //   soi_local = soi_au / frame.scale
            let soi_local = frame.soi_radius / frame.scale;

            // safe_micro_near must scale with the camera distance, down to a tiny absolute minimum.
            // Units are in km (frame-local). 0.001 km = 1 meter.
            let safe_micro_near = (dist_local * 0.01).max(0.001); // 1% of distance, or 1 meter minimum
            let tight_near = (dist_local - soi_local).max(safe_micro_near);
            let tight_far = (dist_local + soi_local).max(tight_near + safe_micro_near);

            layer_bounds
              .entry(frame.depth_layer)
              .and_modify(|(n, f)| {
                *n = n.min(tight_near);
                *f = f.max(tight_far);
              })
              .or_insert((tight_near, tight_far));
            layer_frame_scales.entry(frame.depth_layer).or_insert(frame.scale);
          }
        }
      },
    );

    let mut layer_map: hashbrown::HashMap<u32, DepthLayerData> = hashbrown::HashMap::new();

    macro_rules! get_or_create_layer {
      ($map:expr, $layer:expr, $_scale:expr) => {
        $map.entry($layer).or_insert_with(|| {
          let (near, far) = layer_bounds.get(&$layer).copied().unwrap_or((macro_near, macro_far));
          DepthLayerData {
            layer_index: $layer,
            near,
            far,
            meshes: Vec::new(),
            billboards: Vec::new(),
            markers: Vec::new(),
            measurements: Vec::new(),
            bvhs: Vec::new(),
            particles: Vec::new(),
            gizmos: Vec::new(),
            sphere_gizmos: Vec::new(),
            trajectories: Vec::new(),
            camera_frame_local_pos: Vec3f32::from_components(0.0, 0.0, 0.0),
          }
        })
      };
    }

    for mesh in extracted_meshes {
      let layer = self.ancestor_depth_layer(mesh.entity_id);
      let scale = self.ancestor_frame_scale(mesh.entity_id);
      get_or_create_layer!(layer_map, layer, scale).meshes.push(mesh);
    }

    // ── Per-layer virtual camera: recompute mesh transforms ──────────────────
    // The initial get_relative_transform(mesh, camera) uses the camera's actual
    // scene-graph parent. When camera is inside a micro-frame but we're rendering
    // the macro layer, the cross-frame LCA math produces enormous numbers.
    //
    // Fix: for each layer, recompute every mesh's transform as if the camera were
    // "virtually" in the same coordinate space as that layer.
    //
    // • Macro (layer 0): both mesh and camera expressed in root-space (AU).
    //   RTE model = mesh_global_AU − camera_global_AU
    //
    // • Micro (layer N): both mesh and camera expressed in that frame's local space.
    //   RTE model = mesh_frame_local − camera_frame_local
    // Collect frame_entity for each depth_layer > 0
    let mut layer_frame_entities: hashbrown::HashMap<u32, EntityId> = hashbrown::HashMap::new();
    self.query1_without::<ReferenceFrameComponent, HiddenComponent, _>(
      |id, frame: &ReferenceFrameComponent| {
        if frame.depth_layer > 0 {
          layer_frame_entities.entry(frame.depth_layer).or_insert(id);
        }
      },
    );
    let camera_depth_layer = self.ancestor_depth_layer(camera_entity);

    {
      let cam_global = self.global_transform(camera_entity).unwrap_or_default();
      // f64 camera global position for precision in subtraction
      let cam_global_f64 = self.global_transform_f64(camera_entity);

      for (_layer_idx, layer_data) in layer_map.iter_mut() {
        let layer_idx = *_layer_idx;

        if layer_idx == camera_depth_layer {
          // Camera is in this layer's frame — the original get_relative_transform
          // was correct. No recomputation needed.
          // But still set the camera local pos for the grid.
          if layer_idx == 0 {
            if let Some(ref g64) = cam_global_f64 {
              layer_data.camera_frame_local_pos = g64.position.to_f32();
            } else {
              layer_data.camera_frame_local_pos = cam_global.position;
            }
          } else if let Some(&frame_entity) = layer_frame_entities.get(&layer_idx) {
            if let Some(cam_local) = self.get_relative_transform_f64(camera_entity, frame_entity) {
              layer_data.camera_frame_local_pos = cam_local.position.to_f32();
            }
          }
          continue;
        }

        if layer_idx == 0 {
          // Macro layer: recompute using global (root AU) coordinates.
          for mesh in layer_data.meshes.iter_mut() {
            if let Some(mesh_global) = self.global_transform_f64(mesh.entity_id) {
              use crate::scene::TransformComponent;
              use aethervk_oshal_rlib::math::vector::{Vector, Vector3};

              let safe_div = |a: f32, b: f32| -> f32 { if b.abs() < 1e-15 { 0.0 } else { a / b } };

              let cam_pos = cam_global_f64
                .as_ref()
                .map(|c| c.position)
                .unwrap_or_else(|| cam_global.position.to_f64());
              let diff = mesh_global.position - cam_pos;

              mesh.global_transform = TransformComponent {
                position: diff.to_f32(),
                rotation: mesh_global.rotation,
                scale: Vec3f32::from_components(
                  safe_div(mesh_global.scale.x(), cam_global.scale.x()),
                  safe_div(mesh_global.scale.y(), cam_global.scale.y()),
                  safe_div(mesh_global.scale.z(), cam_global.scale.z()),
                ),
              };
            }
          }
          if let Some(ref g64) = cam_global_f64 {
            layer_data.camera_frame_local_pos = g64.position.to_f32();
          } else {
            layer_data.camera_frame_local_pos = cam_global.position;
          }
        } else if let Some(&frame_entity) = layer_frame_entities.get(&layer_idx) {
          // Micro layer: use f64 for camera-relative position to avoid cancellation
          let cam_in_frame_f64 = self.get_relative_transform_f64(camera_entity, frame_entity);
          let cam_in_frame = self.get_relative_transform(camera_entity, frame_entity);
          if let (Some(cam_f64), Some(cam_local)) = (&cam_in_frame_f64, &cam_in_frame) {
            for mesh in layer_data.meshes.iter_mut() {
              let mesh_in_frame = self.get_relative_transform_f64(mesh.entity_id, frame_entity);
              if let Some(mesh_local) = mesh_in_frame {
                use crate::scene::TransformComponent;
                use aethervk_oshal_rlib::math::vector::{Vector, Vector3};

                let safe_div =
                  |a: f32, b: f32| -> f32 { if b.abs() < 1e-15 { 0.0 } else { a / b } };

                let diff = mesh_local.position - cam_f64.position;

                mesh.global_transform = TransformComponent {
                  position: diff.to_f32(),
                  rotation: mesh_local.rotation,
                  scale: Vec3f32::from_components(
                    safe_div(mesh_local.scale.x(), cam_global.scale.x()),
                    safe_div(mesh_local.scale.y(), cam_global.scale.y()),
                    safe_div(mesh_local.scale.z(), cam_global.scale.z()),
                  ),
                };
              }
            }
            // Use f64 camera position for grid precision
            if let Some(ref cam_f64) = cam_in_frame_f64 {
              layer_data.camera_frame_local_pos = cam_f64.position.to_f32();
            } else {
              layer_data.camera_frame_local_pos = cam_local.position;
            }
          }
        }
      }
    }

    for billboard in extracted_billboards {
      let layer = self.ancestor_depth_layer(billboard.0);
      let scale = self.ancestor_frame_scale(billboard.0);
      get_or_create_layer!(layer_map, layer, scale).billboards.push((
        billboard.1,
        billboard.2,
        billboard.3,
      ));
    }

    for marker in extracted_markers {
      let layer = self.ancestor_depth_layer(marker.0);
      let scale = self.ancestor_frame_scale(marker.0);
      get_or_create_layer!(layer_map, layer, scale).markers.push(marker);
    }

    for meas in extracted_measurements {
      let layer = self.ancestor_depth_layer(meas.0);
      let scale = self.ancestor_frame_scale(meas.0);
      get_or_create_layer!(layer_map, layer, scale).measurements.push(meas);
    }

    for bvh in extracted_bvhs {
      let layer = self.ancestor_depth_layer(bvh.3);
      let scale = self.ancestor_frame_scale(bvh.3);
      get_or_create_layer!(layer_map, layer, scale).bvhs.push(bvh);
    }

    for part in extracted_particles {
      let layer = self.ancestor_depth_layer(part.0);
      let scale = self.ancestor_frame_scale(part.0);
      get_or_create_layer!(layer_map, layer, scale).particles.push(part);
    }

    for gizmo in extracted_gizmos {
      let layer = self.ancestor_depth_layer(gizmo.0);
      let scale = self.ancestor_frame_scale(gizmo.0);
      get_or_create_layer!(layer_map, layer, scale).gizmos.push(gizmo);
    }

    for sg in extracted_sphere_gizmos {
      let layer = self.ancestor_depth_layer(sg.0);
      let scale = self.ancestor_frame_scale(sg.0);
      get_or_create_layer!(layer_map, layer, scale).sphere_gizmos.push(sg);
    }

    for traj in extracted_trajectories {
      let layer = self.ancestor_depth_layer(traj.0);
      let scale = self.ancestor_frame_scale(traj.0);
      get_or_create_layer!(layer_map, layer, scale).trajectories.push(traj);
    }

    if let Some((_, layer_idx, scale, _)) = cursor_transform {
      get_or_create_layer!(layer_map, layer_idx, scale);
    }
    if let Some((layer_idx, scale)) = extracted_sky {
      get_or_create_layer!(layer_map, layer_idx, scale);
    }
    if let Some((_, layer_idx, scale)) = extracted_sun {
      get_or_create_layer!(layer_map, layer_idx, scale);
    }
    if let Some((_, _, _, layer_idx, scale)) = extracted_grid {
      get_or_create_layer!(layer_map, layer_idx, scale);
    }
    if let Some((_, _, layer_idx, scale)) = extracted_background {
      get_or_create_layer!(layer_map, layer_idx, scale);
    }

    // ── Per-layer recomputation for non-mesh entities ────────────────────────
    // The initial extraction used get_relative_transform(entity, camera) which
    // crosses frame boundaries and produces AU-scale values when the entity is
    // in a micro-frame but the camera is not (or vice versa). Recompute using
    // frame-local coordinates, matching the mesh recomputation above.
    for (_layer_idx, layer_data) in layer_map.iter_mut() {
      let layer_idx = *_layer_idx;
      if layer_idx == camera_depth_layer {
        continue; // Same frame — initial get_relative_transform was correct
      }

      if layer_idx > 0 {
        if let Some(&frame_entity) = layer_frame_entities.get(&layer_idx) {
          let cam_in_frame_f64 = self.get_relative_transform_f64(camera_entity, frame_entity);
          let cam_in_frame = self.get_relative_transform(camera_entity, frame_entity);
          if let (Some(cam_f64), Some(_cam_local)) = (cam_in_frame_f64, cam_in_frame) {
            // Sphere gizmos: recompute model matrix in frame-local (km) coordinates using f64
            for sg in layer_data.sphere_gizmos.iter_mut() {
              let entity_id = sg.0;
              if let Some(hrt) = self.get_relative_transform_f64(entity_id, frame_entity) {
                use crate::scene::TransformComponent;
                let rte_pos = (hrt.position - cam_f64.position).to_f32();
                let rte_transform = TransformComponent {
                  position: rte_pos,
                  rotation: hrt.rotation,
                  scale: hrt.scale,
                };
                // Re-read the local_frame from the component
                let local_frame = {
                  use aethervk_oshal_rlib::math::matrix::SquareMatrix;
                  self
                    .with_component(entity_id, |g: &crate::scene::SphereGizmoComponent| {
                      g.local_frame
                    })
                    .unwrap_or(Mat4x4f32::identity())
                };
                sg.1 = rte_transform.to_mat4::<Mat4x4f32>() * local_frame;
              }
            }

            // Gizmos: recompute model matrix using f64
            for gizmo in layer_data.gizmos.iter_mut() {
              let entity_id = gizmo.0;
              if let Some(hrt) = self.get_relative_transform_f64(entity_id, frame_entity) {
                let rte_pos = (hrt.position - cam_f64.position).to_f32();
                let t_mat =
                  Mat4x4f32::translation(rte_pos) * Mat4x4f32::from_quat_custom_frame(hrt.rotation);
                gizmo.1 = t_mat;
              }
            }

            // Billboards: don't carry EntityId in the layer tuple, so we can't
            // recompute here. If billboard jitter is observed in micro layers,
            // add EntityId to the billboard layer tuple.

            // Markers: recompute transform using f64
            for marker in layer_data.markers.iter_mut() {
              let entity_id = marker.0;
              if let Some(hrt) = self.get_relative_transform_f64(entity_id, frame_entity) {
                use crate::scene::TransformComponent;
                marker.1 = TransformComponent {
                  position: (hrt.position - cam_f64.position).to_f32(),
                  rotation: hrt.rotation,
                  scale: hrt.scale,
                };
              }
            }

            // Trajectories: recompute model matrix using f64
            for traj in layer_data.trajectories.iter_mut() {
              let entity_id = traj.0;
              if let Some(hrt) = self.get_relative_transform_f64(entity_id, frame_entity) {
                use crate::scene::TransformComponent;
                let rte_transform = TransformComponent {
                  position: (hrt.position - cam_f64.position).to_f32(),
                  rotation: hrt.rotation,
                  scale: hrt.scale,
                };
                traj.2 = rte_transform.to_mat4::<Mat4x4f32>();
              }
            }
          }
        }
      }
    }

    let mut depth_layers: Vec<DepthLayerData> = layer_map.into_values().collect();
    // Sort layers farthest to nearest (layer 0 is farthest, macro)
    depth_layers.sort_by_key(|l| l.layer_index);

    // ── Debug: dump layer info every 120 frames ──
    {
      use core::sync::atomic::{AtomicU64, Ordering};
      static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);
      let frame = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
      if frame % 120 == 0 {
        aethervk_oshal_rlib::log!(
          "\x1b[36m[MULTI-SCALE] Frame {} | camera pos=({:.6},{:.6},{:.6}) frame_scale={:.2e} | macro near/far=({:.2e},{:.2e})\x1b[0m",
          frame,
          camera_data.absolute_pos.x(),
          camera_data.absolute_pos.y(),
          camera_data.absolute_pos.z(),
          frame_scale,
          macro_near,
          macro_far
        );
        for (layer_idx, (near, far)) in &layer_bounds {
          aethervk_oshal_rlib::log!(
            "\x1b[36m  layer_bounds[{}]: near={:.2e} far={:.2e}\x1b[0m",
            layer_idx,
            near,
            far
          );
        }
        for layer in &depth_layers {
          aethervk_oshal_rlib::log!(
            "\x1b[36m  DepthLayer {} -> near={:.2e} far={:.2e} meshes={} billboards={} gizmos={} sphere_gizmos={} trajectories={}\x1b[0m",
            layer.layer_index,
            layer.near,
            layer.far,
            layer.meshes.len(),
            layer.billboards.len(),
            layer.gizmos.len(),
            layer.sphere_gizmos.len(),
            layer.trajectories.len()
          );
          for mesh in &layer.meshes {
            let p = mesh.global_transform.position;
            let s = mesh.global_transform.scale;
            aethervk_oshal_rlib::log!(
              "\x1b[33m    mesh entity={:?} pos=({:.2e},{:.2e},{:.2e}) scale=({:.2e},{:.2e},{:.2e}) emissive={:.1} path={}\x1b[0m",
              mesh.entity_id,
              p.x(),
              p.y(),
              p.z(),
              s.x(),
              s.y(),
              s.z(),
              mesh.mesh.emissive_intensity,
              mesh.mesh.asset_path
            );
          }
        }
        aethervk_oshal_rlib::log!(
          "\x1b[36m  sun={} sky={} grid={} cursor={} background={}\x1b[0m",
          extracted_sun.is_some(),
          extracted_sky.is_some(),
          extracted_grid.is_some(),
          cursor_transform.is_some(),
          extracted_background.is_some()
        );
        if let Some(((model, radius), layer_idx, _)) = &extracted_sun {
          aethervk_oshal_rlib::log!(
            "\x1b[36m  sun -> layer {} radius={:.6} pos=({:.6},{:.6},{:.6})\x1b[0m",
            layer_idx,
            radius,
            model.w.x(),
            model.w.y(),
            model.w.z()
          );
        }
        if let Some((layer_idx, _)) = &extracted_sky {
          aethervk_oshal_rlib::log!("\x1b[36m  sky -> layer {}\x1b[0m", layer_idx);
        }
        if let Some((_, _, _, layer_idx, _)) = &extracted_grid {
          aethervk_oshal_rlib::log!("\x1b[36m  grid -> layer {}\x1b[0m", layer_idx);
        }
      }
    }

    Ok(RenderSceneExtraction {
      depth_layers,
      extracted_ui,
      extracted_texts,
      extracted_background,
      extracted_sky,
      extracted_sun,
      extracted_grid,
      camera_data,
      window_extent,
      cursor_transform,
      layer_frame_scales,
    })
  }
}

const fn get_mesh_outline(
  is_selected: bool,
  is_following: bool,
  outlines_enabled: bool,
) -> Option<[f32; 4]> {
  const SELECTED_FOLLOWING_OUTLINE_COLOR: [f32; 4] = [0.7, 0.5, 1.0, 1.0];
  const SELECTED_OUTLINE_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
  const FOLLOWING_OUTLINE_COLOR: [f32; 4] = [0.2, 0.5, 1.0, 1.0];
  const GENERAL_OUTLINE_COLOR: [f32; 4] = [0.2, 0.5, 1.0, 0.5];

  if is_selected && is_following {
    Some(SELECTED_FOLLOWING_OUTLINE_COLOR)
  } else if is_selected {
    Some(SELECTED_OUTLINE_COLOR)
  } else if is_following {
    Some(FOLLOWING_OUTLINE_COLOR)
  } else if outlines_enabled {
    Some(GENERAL_OUTLINE_COLOR)
  } else {
    None
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    scene::{
      CameraComponent, Scene, TransformComponent,
      ui::{Transform2DComponent, UiComponent},
    },
    simulation::texture_cache::TextureCache,
  };
  use alloc::sync::Arc;
  use spin::RwLock;

  #[test]
  fn test_ui_layout_relative_placement() {
    let tex_cache = Arc::new(RwLock::new(TextureCache::new("test_tex_cache_ui")));
    let scene = Scene::new(tex_cache);
    scene.register_all_crate_components();

    // 1. Root Background Panel
    let bg_entity = scene.spawn_entity("Background");
    let mut bg_t2d = Transform2DComponent::default();
    bg_t2d.local_position = [0.0, 0.0];
    bg_t2d.size = [1000.0, 1000.0];
    scene.add_component(bg_entity, bg_t2d).unwrap();
    scene.add_component(bg_entity, UiComponent::default()).unwrap();

    // 2. Child Panel
    let child_panel = scene.spawn_entity("Child");
    scene.set_parent(child_panel, Some(bg_entity));
    let mut child_t2d = Transform2DComponent::default();
    child_t2d.local_position = [100.0, 50.0];
    child_t2d.size = [200.0, 200.0];
    scene.add_component(child_panel, child_t2d).unwrap();
    scene.add_component(child_panel, UiComponent::default()).unwrap();

    // 3. Grandchild Panel (Anchored to Bottom-Right of Child)
    let gc_panel = scene.spawn_entity("GrandChild");
    scene.set_parent(gc_panel, Some(child_panel));
    let mut gc_t2d = Transform2DComponent::default();
    gc_t2d.anchor_min = [1.0, 1.0];
    gc_t2d.pivot = [1.0, 1.0];
    gc_t2d.local_position = [-10.0, -10.0]; // 10px padding from right-bottom corner
    gc_t2d.size = [50.0, 50.0];
    scene.add_component(gc_panel, gc_t2d).unwrap();
    scene.add_component(gc_panel, UiComponent::default()).unwrap();

    // Run layout pass directly
    crate::scene::ui::update_ui_layouts(&scene, [1000.0, 1000.0]);

    // Verify background
    scene
      .with_component::<Transform2DComponent, _, _>(bg_entity, |t| {
        assert_eq!(t.global_bounds, [0.0, 0.0, 1000.0, 1000.0]);
      })
      .unwrap();

    // Verify child
    scene
      .with_component::<Transform2DComponent, _, _>(child_panel, |t| {
        assert_eq!(t.global_bounds, [100.0, 50.0, 200.0, 200.0]);
      })
      .unwrap();

    // Verify grandchild
    scene
      .with_component::<Transform2DComponent, _, _>(gc_panel, |t| {
        // Child absolute is [100, 50, 200, 200].
        // Anchor (1.0, 1.0) gives start pos: 100 + 200 = 300 (X), 50 + 200 = 250 (Y).
        // Local pos [-10, -10] gives pos: 290, 240.
        // Pivot (1.0, 1.0) gives offset: size(50) * pivot(1.0) = 50.
        // Final pos: 290 - 50 = 240, 240 - 50 = 190.
        assert_eq!(t.global_bounds, [240.0, 190.0, 50.0, 50.0]);
      })
      .unwrap();
  }

  // ── Multi-scale rendering tests ──────────────────────────────────────

  use crate::scene::{
    GridComponent, ReferenceFrameComponent, ReferenceFrameType, SkyComponent, SphereGizmoComponent,
    SunComponent,
  };
  use aethervk_oshal_rlib::math::{
    quaternion::Quaternion,
    vector::{Vector, Vector3, vec3::Vec3f32, vec4::Quat},
  };

  /// Helper: create a scene with macro (sun+camera+sky+grid) and micro (comet) content
  /// that mirrors the spawn_comet_debug test scenario.
  fn create_multi_scale_scene() -> (Scene, crate::scene::EntityId /* camera */) {
    let tex_cache = Arc::new(RwLock::new(TextureCache::new("test_multiscale")));
    let scene = Scene::new(tex_cache);
    scene.register_all_crate_components();

    // Camera at (0.0115, 0.0115, 0.0115) AU looking at origin
    let root = scene.spawn_entity("Root");

    let camera = scene.spawn_entity("Camera");
    scene.set_parent(camera, Some(root));
    scene
      .add_component(
        camera,
        TransformComponent {
          position: Vec3f32::from_components(0.0115, 0.0115, 0.0115),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene
      .add_component(
        camera,
        CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 1e-5, 1000.0),
      )
      .unwrap();

    // Sun at origin (macroframe, layer 0)

    let sun = scene.spawn_entity("Sun");
    scene.set_parent(sun, Some(root));
    scene
      .add_component(
        sun,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene
      .add_component(
        sun,
        SunComponent {
          resolution: (64, 64, 1),
          radius: 0.00465,
        },
      )
      .unwrap();

    // Sky (macroframe)
    let sky = scene.spawn_entity("Sky");
    scene.set_parent(sky, Some(root));
    scene
      .add_component(
        sky,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene.add_component(sky, SkyComponent {}).unwrap();

    // Grid (macroframe)
    let grid = scene.spawn_entity("Grid");
    scene.set_parent(grid, Some(root));
    scene
      .add_component(
        grid,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene.add_component(grid, GridComponent {}).unwrap();

    // Reference frame at (0.01, 0, 0) AU — this is the micro frame boundary
    let frame_ref = scene.spawn_entity("FrameRef");
    scene.set_parent(frame_ref, Some(root));
    scene
      .add_component(
        frame_ref,
        TransformComponent {
          position: Vec3f32::from_components(0.01, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene
      .add_component(
        frame_ref,
        ReferenceFrameComponent {
          frame_type: ReferenceFrameType::Micro,
          scale: 6.684587e-9, // AU per km
          soi_radius: 0.005,  // 0.005 AU ≈ 0.75 million km
          depth_layer: 1,
        },
      )
      .unwrap();

    // Comet as child of frame_ref (microframe content)
    let comet = scene.spawn_entity("Comet");
    scene.set_parent(comet, Some(frame_ref));
    scene
      .add_component(
        comet,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    // A sphere gizmo makes this entity renderable, triggering micro layer creation
    scene
      .add_component(
        comet,
        SphereGizmoComponent {
          radius: 1.0,
          subdivisions: 8.0,
          local_frame: aethervk_oshal_rlib::math::matrix::SquareMatrix::identity(),
          is_visible: true,
        },
      )
      .unwrap();

    (scene, camera)
  }

  #[test]
  fn test_multi_scale_layer_separation() {
    // Verify that convert_scene separates macro and micro entities into different depth layers
    let (scene, camera) = create_multi_scale_scene();
    let result = scene.convert_scene(camera, false, None, [800, 600]).unwrap();

    // Should have 2 depth layers: macro (0) and micro (1)
    assert!(
      result.depth_layers.len() >= 2,
      "Expected at least 2 depth layers, got {}",
      result.depth_layers.len()
    );

    let macro_layer = result.depth_layers.iter().find(|l| l.layer_index == 0);
    let micro_layer = result.depth_layers.iter().find(|l| l.layer_index == 1);

    assert!(macro_layer.is_some(), "Missing macro layer (index 0)");
    assert!(micro_layer.is_some(), "Missing micro layer (index 1)");
  }

  #[test]
  fn test_multi_scale_near_far_invariant() {
    // Verify that near < far for ALL layers (the bug we fixed)
    let (scene, camera) = create_multi_scale_scene();
    let result = scene.convert_scene(camera, false, None, [800, 600]).unwrap();

    for layer in &result.depth_layers {
      assert!(
        layer.near < layer.far,
        "Layer {} violates near < far: near={}, far={}",
        layer.layer_index,
        layer.near,
        layer.far,
      );
      assert!(
        layer.near > 0.0,
        "Layer {} has non-positive near: {}",
        layer.layer_index,
        layer.near,
      );
    }
  }

  #[test]
  fn test_multi_scale_micro_tight_bounds() {
    // Verify that the micro layer's near/far are tight SOI-based bounds,
    // expressed in the micro frame's LOCAL coordinate space (km).
    let (scene, camera) = create_multi_scale_scene();
    let result = scene.convert_scene(camera, false, None, [800, 600]).unwrap();

    let micro_layer = result.depth_layers.iter().find(|l| l.layer_index == 1).unwrap();

    // Camera at (0.0115, 0.0115, 0.0115) AU, frame at (0.01, 0, 0) AU
    // Distance ≈ 0.0164 AU → in km = 0.0164 / 6.684587e-9 ≈ 2.45e6 km
    // SOI = 0.005 AU → in km = 0.005 / 6.684587e-9 ≈ 7.48e5 km
    // Expected: near ≈ 1.7e6 km, far ≈ 3.2e6 km
    assert!(
      micro_layer.far > 1e5 && micro_layer.far < 1e8,
      "Micro layer far={} should be in frame-local km range (1e5..1e8)",
      micro_layer.far,
    );
    assert!(
      micro_layer.near > 1e4,
      "Micro layer near={} is unreasonably small for frame-local SOI bounds",
      micro_layer.near,
    );
    // The depth range should be approximately 2 × SOI in km ≈ 1.5e6 km
    let range = micro_layer.far - micro_layer.near;
    assert!(
      range > 1e5 && range < 1e7,
      "Micro layer depth range {} not in expected frame-local SOI range",
      range,
    );
  }

  #[test]
  fn test_multi_scale_macro_full_range() {
    // Verify that the macro layer uses the camera's full depth range
    let (scene, camera) = create_multi_scale_scene();
    let result = scene.convert_scene(camera, false, None, [800, 600]).unwrap();

    let macro_layer = result.depth_layers.iter().find(|l| l.layer_index == 0).unwrap();

    // Camera near=1e-5, far=1000, frame_scale=1.0
    assert!(
      (macro_layer.near - 1e-5).abs() < 1e-7,
      "Macro near={} should be ~1e-5",
      macro_layer.near,
    );
    assert!(
      (macro_layer.far - 1000.0).abs() < 1.0,
      "Macro far={} should be ~1000",
      macro_layer.far,
    );
  }

  #[test]
  fn test_multi_scale_layers_sorted() {
    // Verify that layers are sorted by layer_index (macro first, micro second)
    let (scene, camera) = create_multi_scale_scene();
    let result = scene.convert_scene(camera, false, None, [800, 600]).unwrap();

    for window in result.depth_layers.windows(2) {
      assert!(
        window[0].layer_index <= window[1].layer_index,
        "Layers not sorted: {} > {}",
        window[0].layer_index,
        window[1].layer_index,
      );
    }
  }

  #[test]
  fn test_single_layer_scene() {
    // A scene with NO reference frame should produce exactly 1 layer
    let tex_cache = Arc::new(RwLock::new(TextureCache::new("test_single")));
    let scene = Scene::new(tex_cache);
    scene.register_all_crate_components();

    let camera = scene.spawn_entity("Camera");
    scene
      .add_component(
        camera,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 10.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene
      .add_component(
        camera,
        CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 0.1, 100.0),
      )
      .unwrap();

    let sun = scene.spawn_entity("Sun");
    scene
      .add_component(
        sun,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene
      .add_component(
        sun,
        SunComponent {
          resolution: (64, 64, 1),
          radius: 0.00465,
        },
      )
      .unwrap();

    let result = scene.convert_scene(camera, false, None, [800, 600]).unwrap();

    // Only macro layer (0) should exist
    assert_eq!(
      result.depth_layers.len(),
      1,
      "Single-layer scene should have exactly 1 layer"
    );
    assert_eq!(result.depth_layers[0].layer_index, 0);
    assert!(result.depth_layers[0].near < result.depth_layers[0].far);
  }

  #[test]
  fn test_multi_microframe_merged_bounds() {
    // Two micro frames with the same depth_layer should merge their SOI bounds
    let tex_cache = Arc::new(RwLock::new(TextureCache::new("test_merged")));
    let scene = Scene::new(tex_cache);
    scene.register_all_crate_components();

    let camera = scene.spawn_entity("Camera");
    scene
      .add_component(
        camera,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.05),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene
      .add_component(
        camera,
        CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 1e-5, 1000.0),
      )
      .unwrap();

    // Frame A at (0.01, 0, 0), SOI=0.003
    let frame_a = scene.spawn_entity("FrameA");
    scene
      .add_component(
        frame_a,
        TransformComponent {
          position: Vec3f32::from_components(0.01, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene
      .add_component(
        frame_a,
        ReferenceFrameComponent {
          frame_type: ReferenceFrameType::Micro,
          scale: 6.684587e-9,
          soi_radius: 0.003,
          depth_layer: 1,
        },
      )
      .unwrap();

    // Child entity in frame_a (makes micro layer renderable)
    let child_a = scene.spawn_entity("ChildA");
    scene.set_parent(child_a, Some(frame_a));
    scene
      .add_component(
        child_a,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene
      .add_component(
        child_a,
        SphereGizmoComponent {
          radius: 1.0,
          subdivisions: 8.0,
          local_frame: aethervk_oshal_rlib::math::matrix::SquareMatrix::identity(),
          is_visible: true,
        },
      )
      .unwrap();

    // Frame B at (0.02, 0, 0), SOI=0.004 — SAME depth_layer as Frame A
    let frame_b = scene.spawn_entity("FrameB");
    scene
      .add_component(
        frame_b,
        TransformComponent {
          position: Vec3f32::from_components(0.02, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene
      .add_component(
        frame_b,
        ReferenceFrameComponent {
          frame_type: ReferenceFrameType::Micro,
          scale: 6.684587e-9,
          soi_radius: 0.004,
          depth_layer: 1,
        },
      )
      .unwrap();

    // Child entity in frame_b (makes micro layer renderable)
    let child_b = scene.spawn_entity("ChildB");
    scene.set_parent(child_b, Some(frame_b));
    scene
      .add_component(
        child_b,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene
      .add_component(
        child_b,
        SphereGizmoComponent {
          radius: 1.0,
          subdivisions: 8.0,
          local_frame: aethervk_oshal_rlib::math::matrix::SquareMatrix::identity(),
          is_visible: true,
        },
      )
      .unwrap();

    let result = scene.convert_scene(camera, false, None, [800, 600]).unwrap();

    let micro_layer = result.depth_layers.iter().find(|l| l.layer_index == 1).unwrap();

    // Frame A distance from camera ≈ sqrt(0.01^2 + 0.05^2) ≈ 0.051, SOI=0.003
    //   → near_a = 0.048, far_a = 0.054
    // Frame B distance from camera ≈ sqrt(0.02^2 + 0.05^2) ≈ 0.0539, SOI=0.004
    //   → near_b = 0.0499, far_b = 0.0579
    // Merged: near = min(0.048, 0.0499) = 0.048, far = max(0.054, 0.0579) = 0.0579

    assert!(
      micro_layer.near < micro_layer.far,
      "Merged bounds violate near < far: near={}, far={}",
      micro_layer.near,
      micro_layer.far,
    );

    // The merged range should be wider than either SOI alone
    let range = micro_layer.far - micro_layer.near;
    assert!(
      range > 0.008,
      "Merged range {} should be wider than single SOI diameter",
      range,
    );
  }
}
