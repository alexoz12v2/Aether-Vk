use aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32;
use aethervk_oshal_rlib::math::matrix::{MatrixVectorMul, Matrix4};
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::{Vector, Vector3};
use crate::gpu;
use crate::gpu::frame::CameraRenderData;
use crate::gpu::RenderDevice;
use crate::math::collision::linear_bvh::LinearBound;
use crate::scene::{
  BillboardType, BvhDebugComponent, CameraComponent, CursorComponent, EntityId, FollowingComponent,
  GridComponent, HiddenComponent, ImageBillboardComponent, MarkersComponent, MeasurementComponent,
  PhysicalMeshComponent, SelectedComponent, SkyComponent, SunComponent, TransformComponent,
};
use crate::types::{GpuError, GpuResult};
use alloc::vec::Vec;

// TODO extensive unit testing. (with valid scenes of course scene.validate)
// TODO first step shouldn't be done in render thread? (cdylib and simulation_test)

pub struct PhysicalMeshSceneData {
  entity_id: EntityId,
  mesh: PhysicalMeshComponent,
  global_transform: TransformComponent,
  outline: Option<[f32; 4]>,
}

impl PhysicalMeshSceneData {
  fn new(
    entity_id: EntityId,
    mesh: PhysicalMeshComponent,
    global_transform: TransformComponent,
    outline: Option<[f32; 4]>,
  ) -> Self {
    Self {
      entity_id,
      mesh,
      global_transform,
      outline,
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
  pub extracted_bvhs: Vec<(Vec<LinearBound<f32>>, usize)>,
  pub extracted_particles: Vec<(
    EntityId,
    Vec<crate::scene::particles::ParticleData>,
    crate::scene::particles::ParticleEmitterConfig,
  )>,
  pub extracted_gizmos: Vec<(EntityId, Mat4x4f32, f32)>,

  pub extracted_sky: Option<()>,
  pub extracted_sun: Option<(Mat4x4f32, EntityId)>,
  pub extracted_grid: Option<(f32, f32, [f32; 3])>,
  // ... more components here
  pub camera_data: CameraRenderData,
  pub cursor_transform: Option<TransformComponent>,
}

impl RenderSceneExtraction {
  /// Second step of scene conversion to a render scene:
  /// reorganize the extracted data into draw calls
  pub fn build_render_scene(
    self,
    device: &dyn RenderDevice,
    presentation_engine_handle: gpu::PresentationEngineHandle,
  ) -> GpuResult<gpu::RenderScene> {
    let mut render_scene = gpu::RenderScene {
      draw_calls: Vec::with_capacity(self.extracted_meshes.len()),
      cursor_call: None,
      marker_calls: Vec::with_capacity(self.extracted_markers.len()),
      measurement_calls: Vec::with_capacity(self.extracted_measurements.len()),
      billboard_calls: Vec::with_capacity(self.extracted_billboards.len()),
      bvh_draw_calls: Vec::with_capacity(self.extracted_bvhs.len()),
      gizmo_calls: Vec::with_capacity(self.extracted_gizmos.len()),
      particle_calls: Vec::with_capacity(self.extracted_particles.len()),
      camera_data: self.camera_data,
      sun_call: None,
      sky_call: None,
      grid_call: None,
    };

    // Populate Meshes
    for mesh_data in &self.extracted_meshes {
      let res = device.get_or_create_physical_mesh_resources(
        mesh_data.entity_id,
        &mesh_data.mesh,
        presentation_engine_handle,
        &mesh_data.mesh.asset_path,
      )?;
      let dc = gpu::frame::DrawCall::from_handles_and_matrix(
        res,
        mesh_data.mesh.mesh.indices.len() as u32,
        mesh_data.outline,
        mesh_data.global_transform.to_mat4(),
      );
      render_scene.draw_calls.push(dc);
    }

    // Populate Cursor
    if let Some(t) = self.cursor_transform {
      let res = device.get_or_create_cursor_resources(presentation_engine_handle)?;
      render_scene.cursor_call = Some(gpu::frame::CursorDrawCall::from_result_and_matrix(
        res,
        4,
        t.to_mat4(),
        t.scale.x(),
      ));
    }

    // Populate Markers
    if !self.extracted_markers.is_empty() {
      let res = device.get_or_create_marker_resources(presentation_engine_handle)?;
      for (t, markers_comp) in self.extracted_markers {
        let model_matrix = t.to_mat4();
        for marker in markers_comp.markers {
          render_scene
            .marker_calls
            .push(gpu::frame::MarkerDrawCall::from_values(
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
      let pipeline = device
        .get_or_create_measurement_resources(presentation_engine_handle)?
        .pipeline;
      for (p1, p2, points, significant_digits) in self.extracted_measurements {
        render_scene.measurement_calls.push(
          gpu::frame::MeasurementDrawCall::from_data_and_pipeline(p1, p2, points, significant_digits, pipeline),
        );
      }
    }

    // Billboards
    if !self.extracted_billboards.is_empty() {
      let pipeline = device
        .get_or_create_billboard_resources(presentation_engine_handle)?
        .pipeline;
      for (mat, texture_id, billboard_type) in self.extracted_billboards {
        render_scene
          .billboard_calls
          .push(gpu::frame::BillboardDrawCall::from_data(
            pipeline,
            mat,
            texture_id,
            billboard_type,
          ));
      }
    }

    // BVH
    if !self.extracted_bvhs.is_empty() {
      let pipeline_key = device.get_bvh_pipeline_kay()?;
      for (nodes, mesh_index) in &self.extracted_bvhs {
        for node in nodes {
          render_scene
            .bvh_draw_calls
            .push(gpu::frame::BvhDrawCall::new(
              node,
              pipeline_key,
              *mesh_index,
            ));
        }
      }
    }

    // Sun
    if let Some((global_model, entity_id)) = self.extracted_sun {
      let pipeline = device.get_sun_pipeline_key()?;
      render_scene.sun_call = Some(gpu::frame::SunDrawCall::from_model_and_camera(
        global_model,
        &render_scene.camera_data,
        pipeline,
        entity_id,
      )?);
    }

    // Sky
    if let Some(()) = self.extracted_sky {
      let pipeline = device.get_sky_pipeline_key()?;
      render_scene.sky_call = Some(gpu::frame::SkyDrawCall::from_camera(
        &render_scene.camera_data,
        pipeline,
      )?);
    }

    // Grid
    if let Some((density, grid_size, grid_color)) = self.extracted_grid {
      let pipeline = device.get_grid_pipeline_kay()?;
      render_scene.grid_call = Some(gpu::frame::GridDrawCall::new(
        pipeline, density, grid_size, grid_color,
      ));
    }

    // ... More components here

    Ok(render_scene)
  }
}

pub trait SceneConversionExt {
  /// First step of scene to render scene conversion
  /// query the ECS scene to gather rendering data
  fn convert_scene(
    &self,
    camera_entity: EntityId,
    render_outline: bool,
  ) -> GpuResult<RenderSceneExtraction>;
}

impl SceneConversionExt for crate::scene::Scene {
  fn convert_scene(
    &self,
    camera_entity: EntityId,
    render_outline: bool,
  ) -> GpuResult<RenderSceneExtraction> {
    const START_VEC_CAPACITY: usize = 32;
    let mut extracted_meshes: Vec<PhysicalMeshSceneData> = Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_markers: Vec<(TransformComponent, MarkersComponent)> =
      Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_billboards: Vec<(Mat4x4f32, u64, BillboardType)> =
      Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_measurements: Vec<(Vec3f32, Vec3f32, f32, u32)> =
      Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_bvhs: Vec<(Vec<LinearBound<f32>>, usize)> =
      Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_particles: Vec<(
      EntityId,
      Vec<crate::scene::particles::ParticleData>,
      crate::scene::particles::ParticleEmitterConfig,
    )> = Vec::with_capacity(START_VEC_CAPACITY);
    let mut extracted_gizmos: Vec<(EntityId, Mat4x4f32, f32)> =
      Vec::with_capacity(START_VEC_CAPACITY);

    let extracted_sky: Option<()>;
    let extracted_sun: Option<(Mat4x4f32, EntityId)>;
    let extracted_grid: Option<(f32, f32, [f32; 3])>;
    // ... more components here

    let camera_data: CameraRenderData;
    let cursor_transform: Option<TransformComponent>;

    // Camera
    let cam_transform = self.global_transform(camera_entity).unwrap_or_default();
    let cam_comp = self
      .with_component(camera_entity, |c: &CameraComponent| *c)
      .ok_or(GpuError::InvalidArgument(
        "[RenderDevice] convert_scene: Scene has no camera component in the specified entity",
      ))?;
    camera_data = CameraRenderData::new(&cam_transform, &cam_comp);

    // Cursor
    cursor_transform = self
      .query1_first_res_without::<_, HiddenComponent, _, _>(|_id, _c: &CursorComponent| {
        self.global_transform(_id)
      })
      .map(|(t, id)| t);

    // Meshes
    self.query1_without::<_, HiddenComponent, _>(|id, mesh: &PhysicalMeshComponent| {
      if let Some(t) = self.global_transform(id) {
        let mesh_clone = mesh.clone();
        let is_selected: bool = self.has_component::<SelectedComponent>(id).into();
        let is_following: bool = self.has_component::<FollowingComponent>(id).into();
        let outline = get_mesh_outline(is_selected, is_following, render_outline);
        let m = PhysicalMeshSceneData::new(id, mesh_clone, t, outline);
        // BVH debug rendering
        let bvh = m
          .mesh
          .mesh
          .bvh
          .as_ref()
          .map(|bvh| &bvh.nodes)
          .and_then(|nodes| {
            let bvh_dbg_states = {
              let mut dbg_states = None;
              self.with_component(id, |dbg: &BvhDebugComponent| {
                dbg_states = Some(dbg.node_render_states.clone());
              });
              dbg_states
            };
            if let Some(bvh_dbg_states) = bvh_dbg_states {
              Some((nodes, bvh_dbg_states))
            } else {
              None
            }
          });
        if let Some((nodes, states)) = bvh {
          extracted_bvhs.push((Vec::with_capacity(nodes.len()), extracted_meshes.len()));
          let inserted_bvh = extracted_bvhs.last_mut().unwrap();
          states
            .iter()
            .zip(nodes.iter())
            .filter(|&(show, _)| *show)
            .for_each(|(_, node)| {
              inserted_bvh.0.push(node.bound.clone());
            });
        }
        extracted_meshes.push(m);
      }
    });

    // Markers
    self.query1_without::<_, HiddenComponent, _>(|id, m: &MarkersComponent| {
      if let Some(t) = self.global_transform(id) {
        extracted_markers.push((t, m.clone()));
      }
    });

    // Measurements
    self.query1_without::<_, HiddenComponent, _>(|id, m: &MeasurementComponent| {
      if let Some(t) = self.global_transform(id) {
        let mat: Mat4x4f32 = t.to_mat4();
        let p1 = Vec3f32(mat.mul_vector(m.pos1.to_point()));
        let p2 = Vec3f32(mat.mul_vector(m.pos2.to_point()));
        extracted_measurements.push((p1, p2, m.points, m.significant_digits));
      }
    });

    // Billboards
    self.query1_without::<_, HiddenComponent, _>(|id, i: &ImageBillboardComponent| {
      if let Some(t) = self.global_transform(id) {
        let mat: Mat4x4f32 = t.to_mat4();
        extracted_billboards.push((mat, i.texture_id, i.billboard_type));
      }
    });

    // Sun
    extracted_sun = self.query2_first_res_without::<_, _, HiddenComponent, _, _>(
      |id, _t: &TransformComponent, _s: &SunComponent| {
        self.global_transform(id).map(|t| t.to_mat4::<Mat4x4f32>())
      },
    );

    // Sky
    extracted_sky = self
      .query1_first_res_without::<_, HiddenComponent, _, _>(|_id, _s: &SkyComponent| Some(()))
      .map(|_| ());

    // Grid
    extracted_grid = self
      .query1_first_res_without::<_, HiddenComponent, _, _>(|_id, _s: &GridComponent| {
        // TODO grid component should have data (adjust these values for now)
        let density: f32 = 0.1;
        let grid_size: f32 = 1.0;
        let grid_color: [f32; 3] = [0.5, 0.5, 0.5];
        Some((density, grid_size, grid_color))
      })
      .map(|(d, _)| d);

    // Particles
    self.query1_without::<_, HiddenComponent, _>(
      |id, sys: &crate::scene::particles::ParticleSystemComponent| {
        extracted_particles.push((id, sys.particles.clone(), sys.config.clone()));
      },
    );

    // Gizmos
    self.query1_without::<_, HiddenComponent, _>(|id, gizmo: &crate::scene::GizmoComponent| {
      if gizmo.gizmo_visible {
        if let Some(t) = self.global_transform(id) {
          let t_mat = aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::translation(t.position)
            * aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::from_quat_custom_frame(
              t.rotation,
            );

          let gizmo_model = t_mat;

          extracted_gizmos.push((id, gizmo_model, gizmo.gizmo_scale));
        }
      }
    });

    // ... More components here

    Ok(RenderSceneExtraction {
      extracted_meshes,
      extracted_markers,
      extracted_billboards,
      extracted_measurements,
      extracted_bvhs,
      extracted_particles,
      extracted_gizmos,
      extracted_sky,
      extracted_sun,
      extracted_grid,
      camera_data,
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
