//! scene_conversion module.

use crate::{
  gpu,
  gpu::{RenderDevice, frame::CameraRenderData},
  math::collision::linear_bvh::LinearBound,
  scene::{
    BackgroundComponent, BillboardType, BvhDebugComponent, CameraComponent, CursorComponent,
    EntityId, FollowingComponent, GridComponent, HiddenComponent, ImageBillboardComponent,
    MarkersComponent, MeasurementComponent, PhysicalMeshComponent, SelectedComponent, SkyComponent,
    SunComponent, TransformComponent,
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

/// Data extracted from ECS Scene struct. Middleman between [`crate::scene::Scene`]
/// and [`crate::gpu::frame::RenderScene`]
pub struct RenderSceneExtraction {
  pub extracted_meshes: Vec<PhysicalMeshSceneData>,
  pub extracted_markers: Vec<(TransformComponent, MarkersComponent)>,
  pub extracted_billboards: Vec<(Mat4x4f32, u64, BillboardType)>,
  pub extracted_measurements: Vec<(Vec3f32, Vec3f32, f32, u32)>,
  pub extracted_bvhs: Vec<(
    BvhDebugComponent,
    Vec<LinearBound<f32>>,
    Mat4x4f32,
    EntityId,
  )>,
  pub extracted_particles: Vec<(
    EntityId,
    alloc::sync::Weak<spin::RwLock<Vec<crate::scene::particles::ParticleData>>>,
    crate::scene::particles::ParticleEmitterComponent,
  )>,
  pub extracted_gizmos: Vec<(EntityId, Mat4x4f32, f32)>,
  pub extracted_sphere_gizmos: Vec<(EntityId, Mat4x4f32, f32, f32)>, // entity, model, radius, subdivisions
  pub extracted_trajectories: Vec<(
    EntityId,
    crate::scene::trajectory::TrajectoryComponent,
    Mat4x4f32,
  )>,
  pub extracted_ui: Vec<crate::gpu::UiElementGpu>,
  pub extracted_texts: Vec<(
    crate::scene::ui::Transform2DComponent,
    crate::scene::ui::ScreenSpaceTextComponent,
  )>,
  pub extracted_background: Option<([f32; 4], [f32; 4])>,

  pub extracted_sky: Option<()>,
  pub extracted_sun: Option<((Mat4x4f32, f32), EntityId)>,
  pub extracted_grid: Option<(f32, f32, [f32; 3])>,
  // ... more components here
  pub camera_data: CameraRenderData,
  pub window_extent: [u32; 2],
  pub cursor_transform: Option<TransformComponent>,
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
      draw_calls: Vec::with_capacity(self.extracted_meshes.len()),
      cursor_call: None,
      marker_calls: Vec::with_capacity(self.extracted_markers.len()),
      measurement_calls: Vec::with_capacity(self.extracted_measurements.len()),
      billboard_calls: Vec::with_capacity(self.extracted_billboards.len()),
      bvh_draw_calls: Vec::with_capacity(self.extracted_bvhs.len()),
      bvhwire2_data: Vec::with_capacity(self.extracted_bvhs.len()),
      gizmo_calls: Vec::with_capacity(self.extracted_gizmos.len()),
      particle_calls: Vec::with_capacity(self.extracted_particles.len()),
      text_calls: Vec::with_capacity(self.extracted_texts.len()),
      camera_data: self.camera_data,
      sun_call: None,
      sky_call: None,
      grid_call: None,
      particle2_calls: Vec::with_capacity(self.extracted_particles.len()),
      trajectory_call: None,
      bvhwire2_batch_call: None,
      ui_call: None,
      text2_call: None,
      background_call: None,
      sphere_gizmo_batch_call: None,
    };

    // Populate Meshes
    for mesh_data in &self.extracted_meshes {
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
      render_scene.draw_calls.push(dc);
    }

    // Populate Cursor
    if let Some(t) = self.cursor_transform {
      let res = match device.get_cursor_resources(presentation_engine_handle) {
        Ok(r) => r,
        Err(_) => device.create_cursor_resources(cmd_buffer, presentation_engine_handle)?,
      };
      render_scene.cursor_call = Some(gpu::frame::CursorDrawCall::from_result_and_matrix(
        res,
        4,
        t.to_mat4(),
        t.scale.x(),
      ));
    }

    // Populate Markers
    if !self.extracted_markers.is_empty() {
      let res = match device.get_marker_resources(presentation_engine_handle) {
        Ok(r) => r,
        Err(_) => device.create_marker_resources(cmd_buffer, presentation_engine_handle)?,
      };
      for (t, markers_comp) in self.extracted_markers {
        let model_matrix = t.to_mat4();
        for marker in markers_comp.markers {
          render_scene.marker_calls.push(gpu::frame::MarkerDrawCall::from_values(
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
    if !self.extracted_measurements.is_empty() {
      let pipeline = match device.get_measurement_resources(presentation_engine_handle) {
        Ok(r) => r.pipeline,
        Err(_) => {
          device.create_measurement_resources(cmd_buffer, presentation_engine_handle)?.pipeline
        }
      };
      for (p1, p2, points, significant_digits) in self.extracted_measurements {
        render_scene.measurement_calls.push(
          gpu::frame::MeasurementDrawCall::from_data_and_pipeline(
            p1,
            p2,
            points,
            significant_digits,
            pipeline,
          ),
        );
      }
    }

    // Billboards
    if !self.extracted_billboards.is_empty() {
      let pipeline = match device.get_billboard_resources(presentation_engine_handle) {
        Ok(r) => r.pipeline,
        Err(_) => {
          device.create_billboard_resources(cmd_buffer, presentation_engine_handle)?.pipeline
        }
      };
      for (mat, texture_id, billboard_type) in self.extracted_billboards {
        render_scene.billboard_calls.push(gpu::frame::BillboardDrawCall::from_data(
          pipeline,
          mat,
          texture_id,
          billboard_type,
        ));
      }
    }

    // Gizmos
    if !self.extracted_gizmos.is_empty() {
      let gizmo_resources = match device.get_gizmo_resources(presentation_engine_handle) {
        Ok(r) => r,
        Err(_) => device.create_gizmo_resources(cmd_buffer, presentation_engine_handle)?,
      };
      for (entity_id, mat, scale) in self.extracted_gizmos {
        let gizmo_idx = device.update_gizmo_instance(entity_id, mat, presentation_engine_handle)?;
        render_scene.gizmo_calls.push(gpu::frame::GizmoDrawCall::from_values(
          gizmo_resources.pipeline,
          scale,
          gizmo_idx,
        ));
      }
    }

    // Sphere Gizmos
    if !self.extracted_sphere_gizmos.is_empty() {
      let mut sphere_gizmo_data = Vec::with_capacity(self.extracted_sphere_gizmos.len());
      for (entity_id, model, radius, subdivisions) in self.extracted_sphere_gizmos {
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
      render_scene.sphere_gizmo_batch_call =
        device.upload_sphere_gizmos_batch(cmd_buffer, &sphere_gizmo_data)?;
    }

    // BVH
    if !self.extracted_bvhs.is_empty() {
      for (dbg_comp, nodes, global_model, entity_id) in &self.extracted_bvhs {
        render_scene.add_renderable(
          cmd_buffer,
          device,
          *entity_id,
          *global_model,
          crate::scene::RenderableDataRef::BvhWireframe(dbg_comp, nodes),
          presentation_engine_handle,
          "bvh_wireframe",
          false,
          [0.0; 4],
        )?;
      }

      if !render_scene.bvhwire2_data.is_empty() {
        render_scene.bvhwire2_batch_call =
          device.upload_bvhwire2_batch(cmd_buffer, &render_scene.bvhwire2_data)?;
      }
    }

    // Sun
    if let Some(((global_model, radius), entity_id)) = self.extracted_sun {
      let pipeline = device.get_sun_pipeline_key(presentation_engine_handle)?;
      render_scene.sun_call = Some(gpu::frame::SunDrawCall::from_model_and_camera(
        global_model,
        &render_scene.camera_data,
        pipeline,
        entity_id,
        radius,
      )?);
    }

    // Sky
    if let Some(()) = self.extracted_sky {
      let pipeline = device.get_sky_pipeline_key(presentation_engine_handle)?;
      render_scene.sky_call = Some(gpu::frame::SkyDrawCall::from_camera(
        &render_scene.camera_data,
        pipeline,
      )?);
    }

    // Background
    if let Some((color_top, color_bottom)) = self.extracted_background {
      let pipeline = device.get_background_pipeline_key(presentation_engine_handle)?;
      render_scene.background_call = Some(gpu::frame::BackgroundDrawCall {
        color_top,
        color_bottom,
        pipeline,
      });
    }

    // Grid
    if let Some((density, grid_size, grid_color)) = self.extracted_grid {
      let pipeline = device.get_grid_pipeline_kay(presentation_engine_handle)?;
      render_scene.grid_call = Some(gpu::frame::GridDrawCall::new(
        pipeline, density, grid_size, grid_color,
      ));
    }

    // Particles
    let particle_pipeline = device.get_particle_pipeline_key(presentation_engine_handle)?;
    let particle2_pipeline = device.get_particle2_pipeline_key(presentation_engine_handle)?;
    for (_entity_id, particles, config) in self.extracted_particles {
      if config.use_particle2 {
        render_scene.particle2_calls.push(gpu::frame::Particle2DrawCall {
          pipeline: particle2_pipeline,
          system_particle_offset: 0,
          system_indirect_offset: 0,
          config,
          particles,
        });
      } else {
        render_scene.particle_calls.push(gpu::frame::ParticleDrawCall {
          pipeline: particle_pipeline,
          system_particle_offset: 0,
          system_indirect_offset: 0,
          config,
          particles,
        });
      }
    }

    // Trajectories
    render_scene.trajectory_call =
      device.upload_trajectories(cmd_buffer, &self.extracted_trajectories)?;

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
            style_flags: 0, // Normal by default
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
    let mut extracted_markers: Vec<(TransformComponent, MarkersComponent)> =
      Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_billboards: Vec<(Mat4x4f32, u64, BillboardType)> =
      Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_measurements: Vec<(Vec3f32, Vec3f32, f32, u32)> =
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

    let extracted_sky: Option<()>;
    let extracted_sun: Option<((Mat4x4f32, f32), EntityId)>;
    let extracted_grid: Option<(f32, f32, [f32; 3])>;
    let extracted_background: Option<([f32; 4], [f32; 4])>;
    // ... more components here

    let camera_data: CameraRenderData;
    let cursor_transform: Option<TransformComponent>;

    // Camera
    let cam_transform = self.global_transform(camera_entity).unwrap_or_default();
    let cam_comp = self.with_component(camera_entity, |c: &CameraComponent| *c).ok_or(
      crate::gpu_invalid_arg!("[ Scene has no camera component in the specified entity"),
    )?;
    camera_data = CameraRenderData::new(&cam_transform, &cam_comp);

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

    // Cursor
    cursor_transform = self
      .query1_first_res_without::<_, HiddenComponent, _, _>(|id, _c: &CursorComponent| {
        if hidden_set.contains(&id) {
          return None;
        }
        self.global_transform(id)
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
              comp.node_render_states.iter().zip(nodes.iter()).filter(|&(show, _)| *show).for_each(
                |(_, node)| {
                  extracted.push(node.bound.clone());
                },
              );
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
            comp.node_render_states.iter().zip(nodes.iter()).filter(|&(show, _)| *show).for_each(
              |(_, node)| {
                extracted.push(node.bound.clone());
              },
            );
            extracted_bvhs.push((comp, extracted, t.to_mat4(), id));
          }
          extracted_meshes.push(m);
        }
      });
    }

    // Markers
    if should_par {
      let results = self.query1_res_without_par::<MarkersComponent, HiddenComponent, _, _>(
        pool.unwrap(),
        |id, m| {
          if hidden_set.contains(&id) {
            return None;
          }
          self.get_relative_transform(id, camera_entity).map(|t| (t, m.clone()))
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
          extracted_markers.push((t, m.clone()));
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
            (p1, p2, m.points, m.significant_digits)
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
          extracted_measurements.push((p1, p2, m.points, m.significant_digits));
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
            (mat, i.texture_id, i.billboard_type)
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
          extracted_billboards.push((mat, i.texture_id, i.billboard_type));
        }
      });
    }

    extracted_sun = self.query2_first_res_without::<_, _, HiddenComponent, _, _>(
      |id, _t: &TransformComponent, s: &SunComponent| {
        if hidden_set.contains(&id) {
          return None;
        }
        self.get_relative_transform(id, camera_entity).map(|t| (t.to_mat4::<Mat4x4f32>(), s.radius))
      },
    );

    extracted_sky = self
      .query1_first_res_without::<_, HiddenComponent, _, _>(|id, _s: &SkyComponent| {
        if hidden_set.contains(&id) {
          return None;
        }
        Some(())
      })
      .map(|_| ());

    extracted_grid = self
      .query1_first_res_without::<_, HiddenComponent, _, _>(|id, _s: &GridComponent| {
        if hidden_set.contains(&id) {
          return None;
        }
        // TODO grid component should have data (adjust these values for now)
        let density: f32 = 0.1;
        let grid_size: f32 = 1.0;
        let grid_color: [f32; 3] = [0.5, 0.5, 0.5];
        Some((density, grid_size, grid_color))
      })
      .map(|(d, _)| d);

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

    // Gizmos
    self.query1_without::<_, HiddenComponent, _>(|id, gizmo: &crate::scene::GizmoComponent| {
      if hidden_set.contains(&id) {
        return;
      }
      if gizmo.gizmo_visible {
        if let Some(t) = self.get_relative_transform(id, camera_entity) {
          let t_mat = aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::translation(t.position)
            * aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::from_quat_custom_frame(
              t.rotation,
            );

          let gizmo_model = t_mat;

          extracted_gizmos.push((id, gizmo_model, gizmo.gizmo_scale));
        }
      }
    });

    // SphereGizmos
    self.query1_without::<_, HiddenComponent, _>(
      |id, gizmo: &crate::scene::SphereGizmoComponent| {
        if hidden_set.contains(&id) {
          return;
        }
        if let Some(t) = self.get_relative_transform(id, camera_entity) {
          let gizmo_model = t.to_mat4::<Mat4x4f32>() * gizmo.local_frame;
          extracted_sphere_gizmos.push((id, gizmo_model, gizmo.radius, gizmo.subdivisions));
        }
      },
    );

    // Trajectories
    self.query1_without::<_, HiddenComponent, _>(
      |id, traj: &crate::scene::trajectory::TrajectoryComponent| {
        if hidden_set.contains(&id) {
          return;
        }
        if let Some(t) = self.get_relative_transform(id, camera_entity) {
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
        Some((b.color_top, b.color_bottom))
      })
      .map(|(b, _)| b);

    // ... More components here

    Ok(RenderSceneExtraction {
      extracted_meshes,
      extracted_markers,
      extracted_billboards,
      extracted_measurements,
      extracted_bvhs,
      extracted_particles,
      extracted_gizmos,
      extracted_sphere_gizmos,
      extracted_trajectories,
      extracted_ui,
      extracted_texts,
      extracted_background,
      extracted_sky,
      extracted_sun,
      extracted_grid,
      camera_data,
      window_extent,
      cursor_transform,
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
}
