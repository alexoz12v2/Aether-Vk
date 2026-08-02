//! scene_conversion module.

use crate::{
  gpu::{self, RenderDevice, frame::*},
  gpu_backends::vulkan,
  gpu_invalid_arg,
  scene::{
    BackgroundComponent, CameraComponent, CameraProjection, CursorComponent, EntityId,
    GizmoComponent, GridComponent, HiddenComponent, HighResTransformComponent,
    ImageBillboardComponent, MarkersComponent, MeasurementComponent, ParticleSystemComponent,
    ReferenceFrameComponent, Scene, SkyComponent, SphereGizmoComponent, StaticMeshComponent,
    SunComponent, TransformComponent, text, trajectory::TrajectoryComponent, ui,
  },
  types::GpuResult,
};
use aethervk_oshal_rlib::{
  math::{
    FloatLike,
    matrix::{Matrix4, MatrixVectorMul, mat4::Mat4x4f32},
    vector::{Vector, Vector3, vec3::Vec3f32},
  },
  os::{
    pool::ThreadPool,
    time::{timeus_t, us_to_300ths_rounded},
  },
};
use function_name::named;

/// New implemnetation for ECS scene conversion into a list of draw calls
pub trait SceneConversionExt2 {
  /// Fused Step for Querying ECS scene, computing cross-frame spatial math, request GPU resources,
  /// and directly output the final Render Draw Calls
  fn build_render_scene(
    &self,
    device: &vulkan::device::Device,
    pe_handle: gpu::PresentationEngineHandle,
    cmd_buffer: gpu::CommandBufferHandle,
    camera_entity: EntityId,
    render_outline: bool,
    pool: Option<&ThreadPool>,
    window_extent: [u32; 2],
    unscaled_time_us: timeus_t,
    unscaled_time_delta_us: timeus_t,
    scaled_time_us: timeus_t,
    scaled_time_delta_us: timeus_t,
    mean_intra_grains_distance_mm: f32,
    min_cumulated_mass_g: f32,
    debug_name: &str,
  ) -> GpuResult<gpu::RenderScene>;
}

impl SceneConversionExt2 for Scene {
  #[named]
  fn build_render_scene(
    &self,
    device: &vulkan::device::Device,
    pe_handle: gpu::PresentationEngineHandle,
    cmd_buffer: gpu::CommandBufferHandle,
    camera_entity: EntityId,
    render_outline: bool,
    pool: Option<&ThreadPool>,
    window_extent: [u32; 2],
    unscaled_time_us: timeus_t,
    unscaled_time_delta_us: timeus_t,
    scaled_time_us: timeus_t,
    scaled_time_delta_us: timeus_t,
    mean_intra_grains_distance_mm: f32,
    min_cumulated_mass_g: f32,
    debug_name: &str,
  ) -> GpuResult<gpu::RenderScene> {
    // ------ 1. Precompute Camera & Hierarchy ----------------------------------------------
    ui::update_ui_layouts(self, [window_extent[0] as f32, window_extent[1] as f32]);
    let should_par = self.should_parallelize() && pool.is_some();

    let cam_global_f64 = self
      .global_transform_f64(camera_entity)
      .ok_or(gpu_invalid_arg!("invalid camera entity"))?;
    let cam_global_f32 = cam_global_f64.to_transform();

    let cam_comp =
      self
        .with_component(camera_entity, |c: &CameraComponent| *c)
        .ok_or(gpu_invalid_arg!(
          "scene has no camera compoent on the specified entity"
        ))?;

    let camera_data = CameraRenderData::new(
      &cam_global_f32,
      &cam_comp,
      self.ancestor_frame_scale(camera_entity),
      window_extent,
    );

    // Filter hidden subtrees
    let hidden_roots = if should_par {
      self.query1_res_par::<HiddenComponent, _, _>(unsafe { pool.unwrap_unchecked() }, |id, _| {
        Some(id)
      })
    } else {
      self.query1_res::<HiddenComponent, _, _>(|id, _| Some(id))
    };
    let mut hidden_set = hashbrown::HashSet::with_capacity(128);
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

    // ------ 2. O(Frames) RTE Caching & Layer Pre-Allocation -------------------------------
    // Note: we are assuming that camera is in macro frame, not micro. TODO: assertion
    let macro_near = cam_comp.near_plane();
    let macro_far = cam_comp.far_plane();

    let mut layer_bounds: hashbrown::HashMap<u32, (f32, f32)> =
      hashbrown::HashMap::with_capacity(64);
    let mut layer_frame_scales: hashbrown::HashMap<u32, f32> =
      hashbrown::HashMap::with_capacity(64);

    layer_bounds.insert(0, (macro_near, macro_far));
    layer_frame_scales.insert(0, 1.0);

    let mut layer_frame_entities: hashbrown::HashMap<u32, EntityId> =
      hashbrown::HashMap::with_capacity(64);
    let mut camera_in_frames: hashbrown::HashMap<u32, HighResTransformComponent> =
      hashbrown::HashMap::with_capacity(16);

    camera_in_frames.insert(0, cam_global_f64);

    // for each micro layer, compute transform relative to camera, and from that, bounds
    self.query1_without::<ReferenceFrameComponent, HiddenComponent, _>(
      |id, frame: &ReferenceFrameComponent| {
        debug_assert!((frame.depth_layer == 0) ^ (self.get_root().unwrap() == id));
        if frame.depth_layer > 0 {
          layer_frame_entities.insert(frame.depth_layer, id);
          if let Some(cam_in_frame_f64) = self.get_relative_transform_f64(camera_entity, id) {
            camera_in_frames.insert(frame.depth_layer, cam_in_frame_f64);

            let dist_local = cam_in_frame_f64.position.length() as f32;
            let soi_local = frame.soi_radius / frame.scale; // TODO check if useful
            let safe_micro_near = (dist_local * 0.01).max(0.001);
            let tight_near = (dist_local - soi_local).max(safe_micro_near);
            let tight_far = (dist_local + soi_local).max(tight_near + safe_micro_near);

            layer_bounds.insert(frame.depth_layer, (tight_near, tight_far));
            layer_frame_scales.insert(frame.depth_layer, frame.scale);
          }
        }
      },
    );

    let mut layer_map: hashbrown::HashMap<u32, RenderLayer> = hashbrown::HashMap::with_capacity(16);

    // Ensures layers are lazily construted into our final memory footprint format only when
    // actually used
    macro_rules! get_or_create_layer {
      ($layer_idx:expr) => {
        layer_map.entry($layer_idx).or_insert_with(|| {
          let (near, far) =
            layer_bounds.get(&$layer_idx).copied().unwrap_or((macro_near, macro_far));
          let scale = layer_frame_scales.get(&$layer_idx).copied().unwrap_or(1.0);
          RenderLayer {
            layer_index: $layer_idx,
            frame_scale: scale,
            near,
            far,
            camera_frame_local_pos: camera_in_frames
              .get(&$layer_idx)
              .map(|c| c.position.to_f32())
              .unwrap_or_default(),
            draw_calls: alloc::vec::Vec::<DrawCall>::with_capacity(16),
            billboard_calls: alloc::vec::Vec::<BillboardDrawCall>::with_capacity(16),
            marker_calls: alloc::vec::Vec::<MarkerDrawCall>::with_capacity(16),
            measurement_calls: alloc::vec::Vec::<MeasurementDrawCall>::with_capacity(16),
            gizmo_calls: alloc::vec::Vec::<GizmoDrawCall>::with_capacity(16),
            dust_calls: alloc::vec::Vec::<DustDrawCall>::with_capacity(16),
            sphere_gizmo_batch_call: None,
            trajectory_call: None,
            cursor_call: None,
            sun_call: None,
            sky_call: None,
            grid_call: None,
            background_call: None,
          }
        })
      };
    }

    // Instant O(1) mathematical `f64` Relative-To-Eye (RTE) calculation using the cache
    let compute_rte = |scene: &Scene, id: EntityId| -> Option<(u32, TransformComponent)> {
      let layer_idx = scene.ancestor_depth_layer(id);
      let cam_in_frame = camera_in_frames.get(&layer_idx)?;

      let (pos_f64, rot, obj_scale) = if layer_idx == 0 {
        let g = scene.global_transform_f64(id)?;
        (g.position, g.rotation, g.scale)
      } else {
        let frame_id = layer_frame_entities.get(&layer_idx)?;
        let l = scene.get_relative_transform_f64(id, *frame_id)?;
        (l.position, l.rotation, l.scale)
      };

      let diff = pos_f64 - cam_in_frame.position;

      Some((
        layer_idx,
        TransformComponent {
          position: diff.to_f32(),
          rotation: rot,
          scale: obj_scale / cam_global_f32.scale,
        },
      ))
    };

    let mut render_scene = gpu::RenderScene {
      unscaled_time_us,
      unscaled_time_delta_us,
      camera_data: camera_data.clone(),
      window_extent,
      depth_layers: alloc::vec::Vec::with_capacity(4),
      cursor_call: None,
      ui_call: None,
      text2_call: None,
    };

    // ------ 3. Zero-Copy GPU Upload Abstraction Macro -------------------------------------
    macro_rules! extract {
      ($Comp:ty, |$id:ident, $comp:ident| $logic:expr) => {{
        let process = |$id: EntityId, $comp: &$Comp| {
          if hidden_set.contains(&$id) {
            return None;
          }
          $logic
        };
        if should_par {
          self
            .query1_res_without_par::<$Comp, HiddenComponent, _, _>(pool.unwrap(), process)
            .into_iter()
            .map(|(r, _)| r)
            .collect::<alloc::vec::Vec<_>>()
        } else {
          let mut res = alloc::vec::Vec::with_capacity(32);
          self.query1_without::<_, HiddenComponent, _>(|$id, c: &$Comp| {
            if let Some(r) = process($id, c) {
              res.push(r);
            }
          });
          res
        }
      }};
      ($Comp1:ty, $Comp2:ty, |$id:ident, $comp1:ident, $comp2:ident| $logic:expr) => {{
        let process = |$id: EntityId, $comp1: &$Comp1, $comp2: &$Comp2| {
          if hidden_set.contains(&$id) {
            return None;
          }
          $logic
        };
        if should_par {
          self
            .query2_res_par::<$Comp1, $Comp2, _, _>(pool.unwrap(), process)
            .into_iter()
            .map(|(r, _)| r)
            .collect::<alloc::vec::Vec<_>>()
        } else {
          let mut res = alloc::vec::Vec::with_capacity(32);
          self.query2::<$Comp1, $Comp2, _>(|$id, c1, c2| {
            if let Some(r) = process($id, c1, c2) {
              res.push(r);
            }
          });
          res
        }
      }};
    }

    // ------ 4. Fused Component Extraction & GPU Draw Call Creation ------------------------
    // 1. Meshes
    let extracted_meshes = extract!(StaticMeshComponent, |id, mesh| {
      compute_rte(self, id).map(|(layer_idx, rte)| {
        // TODO reintroduce following and selected if necessary. If reintroduced, the selection and
        // following state should have been stored in the scene
        let outline = get_mesh_outline(false, false, render_outline);
        (layer_idx, id, mesh.clone(), rte, outline)
      })
    });

    for (layer_idx, _id, mesh, rte, outline) in extracted_meshes {
      let gpu_res = device.get_physical_mesh2_resources(mesh.mesh.id, pe_handle).or_else(|_| {
        device.create_physical_mesh2_resources(
          cmd_buffer,
          mesh.mesh.id,
          &mesh,
          pe_handle,
          &alloc::format!("Mesh_{}", mesh.mesh.id),
        )
      });
      if let Ok(res) = gpu_res {
        let mat = rte.to_mat4();
        let l = get_or_create_layer!(layer_idx);

        l.draw_calls.push(DrawCall::from_handles_and_matrix(
          res,
          mesh.mesh.indices.len() as u32,
          outline,
          mat,
          mesh.emissive_color[3],
          [
            mesh.emissive_color[0],
            mesh.emissive_color[1],
            mesh.emissive_color[2],
          ],
          true,
          0,
        ));
      } else {
        let err = unsafe { gpu_res.unwrap_err_unchecked() };
        aethervk_oshal_rlib::log!("GPU Upload Error: {}", err);
      }
    }

    // 2. Billboards
    let extracted_billboards = extract!(ImageBillboardComponent, |id, i| {
      compute_rte(self, id)
        .map(|(layer_idx, rte)| (layer_idx, rte.to_mat4(), i.texture_id, i.billboard_type))
    });
    if !extracted_billboards.is_empty() {
      if let Ok(pipe) = device
        .get_billboard_resources(pe_handle)
        .or_else(|_| device.create_billboard_resources(cmd_buffer, pe_handle))
        .map(|r| r.pipeline)
      {
        for (layer_idx, mat, tex, b_type) in extracted_billboards {
          get_or_create_layer!(layer_idx)
            .billboard_calls
            .push(BillboardDrawCall::from_data(pipe, mat, tex, b_type));
        }
      } else {
        aethervk_oshal_rlib::log!("GPU Errore creating/getting billboard resources");
      }
    }

    // 3. Markers (TODO remove)
    let extracted_markers = extract!(MarkersComponent, |id, m| {
      compute_rte(self, id).map(|(layer_idx, rte)| (layer_idx, rte.to_mat4(), m.clone()))
    });
    if !extracted_markers.is_empty() {
      if let Ok(pipe) = device
        .get_marker_resources(pe_handle)
        .or_else(|_| device.create_marker_resources(cmd_buffer, pe_handle))
        .map(|r| r.pipeline)
      {
        for (layer_idx, mat, m_comp) in extracted_markers {
          let layer = get_or_create_layer!(layer_idx);
          for marker in m_comp.markers {
            layer.marker_calls.push(MarkerDrawCall::from_values(
              pipe,
              mat,
              marker.local_pos,
              marker.size,
              marker.color,
            ));
          }
        }
      } else {
        aethervk_oshal_rlib::log!("GPU Errore creating/getting Markers resources");
      }
    }

    // 4. Measurements
    let extracted_meas = extract!(MeasurementComponent, |id, m| {
      compute_rte(self, id).map(|(layer_idx, rte)| {
        let mat: Mat4x4f32 = rte.to_mat4();
        let p1 = Vec3f32(mat.mul_vector(m.pos1.to_point()));
        let p2 = Vec3f32(mat.mul_vector(m.pos2.to_point()));
        (layer_idx, p1, p2, m.points, m.significant_digits)
      })
    });
    if !extracted_meas.is_empty() {
      if let Ok(pipe) = device
        .get_measurement_resources(pe_handle)
        .or_else(|_| device.create_measurement_resources(cmd_buffer, pe_handle))
        .map(|r| r.pipeline)
      {
        for (layer_idx, p1, p2, pts, sig) in extracted_meas {
          get_or_create_layer!(layer_idx).measurement_calls.push(
            MeasurementDrawCall::from_data_and_pipeline(p1, p2, pts, sig, pipe),
          );
        }
      } else {
        aethervk_oshal_rlib::log!("GPU Error creating/getting Measurement resources");
      }
    }

    // 5. Gizmos (TODO remove)
    let extracted_gizmos = extract!(GizmoComponent, |id, g| {
      if !g.gizmo_visible {
        return None;
      }
      compute_rte(self, id).map(|(layer_idx, rte)| {
        (
          layer_idx,
          id,
          Mat4x4f32::translation(rte.position) * Mat4x4f32::from_quat_custom_frame(rte.rotation),
          g.gizmo_scale, // ignore scale from transform and use gizmo scale
        )
      })
    });
    if !extracted_gizmos.is_empty() {
      if let Ok(pipe) = device
        .get_gizmo_resources(pe_handle)
        .or_else(|_| device.create_gizmo_resources(cmd_buffer, pe_handle))
        .map(|r| r.pipeline)
      {
        for (layer_idx, id, mat, scale) in extracted_gizmos {
          if let Ok(idx) = device.update_gizmo_instance(id, mat, pe_handle) {
            get_or_create_layer!(layer_idx)
              .gizmo_calls
              .push(GizmoDrawCall::from_values(pipe, scale, idx));
          }
        }
      } else {
        aethervk_oshal_rlib::log!("GPU Upload Error creating/getting gizmo resources")
      }
    }

    // 6. Sphere Gizmos (Batched - deferred upload)
    let mut sg_batch_buffers = hashbrown::HashMap::<
      u32,
      alloc::vec::Vec<(EntityId, Mat4x4f32, f32, f32)>,
    >::with_capacity(16);
    let extracted_sg = extract!(SphereGizmoComponent, |id, sg| {
      if !sg.is_visible {
        return None;
      }
      // TODO remove sg.local_frame
      compute_rte(self, id).map(|(layer_idx, rte)| {
        (
          layer_idx,
          id,
          rte.to_mat4::<Mat4x4f32>() * sg.local_frame,
          sg.radius,
          sg.subdivisions,
        )
      })
    });
    for (layer_idx, id, mat, rad, sub) in extracted_sg {
      sg_batch_buffers.entry(layer_idx).or_default().push((id, mat, rad, sub));
    }

    // 7. Trajectories (Batched - deferred upload)
    let mut traj_batch_buffers = hashbrown::HashMap::<
      u32,
      alloc::vec::Vec<(EntityId, TrajectoryComponent, Mat4x4f32)>,
    >::with_capacity(16);
    let extracted_traj = extract!(TrajectoryComponent, |id, traj| {
      // Note: traj.clone() copies the array of control points
      compute_rte(self, id).map(|(layer_idx, rte)| (layer_idx, id, traj.clone(), rte.to_mat4()))
    });
    for (layer_idx, id, traj, mat) in extracted_traj {
      traj_batch_buffers.entry(layer_idx).or_default().push((id, traj, mat));
    }

    // 9. Particles
    let current_time_scaled_300ths = us_to_300ths_rounded(scaled_time_us);
    let proj_scale = match cam_comp.projection {
      CameraProjection::Perspective { fov, .. } => {
        // Formula: (ViewportHeight / 2) / tan(FOV / 2)
        (window_extent[1] as f32 * 0.5) / <f32 as FloatLike>::tan(fov * 0.5)
      }
      CameraProjection::Orthographic { bottom, top, .. } => {
        // Formula: ViewportHeight / OrthoHeight
        window_extent[1] as f32 / (top - bottom).abs()
      }
    };

    let dust_calls = extract!(ParticleSystemComponent, |id, ps| {
      const DISPERSION_RATE_MULTIPLIER: f32 = 0.5;

      let v_exp_m_per_s = ps.emission_params.start_velocity_std * DISPERSION_RATE_MULTIPLIER;

      let ttl_300ths_f32 = us_to_300ths_rounded(ps.ttl_us) as f32;
      let cluster_params = ps
        .emission_params
        .cluster_params(mean_intra_grains_distance_mm, min_cumulated_mass_g);

      let single_grain_mass_g = {
        use core::f32::consts::PI;
        let radius_cm = (ps.emission_params.diametre_um * 0.5) * 1e-4;
        let volume_cm3 = (4.0 / 3.0) * PI * radius_cm.powi(3);
        volume_cm3 * ps.emission_params.density_gcm3
      };

      let num_spots = (cluster_params.mass_g / single_grain_mass_g) as u32;

      // calculate cluster diametre in metres (double precision)
      let grain_radius_m = (ps.emission_params.diametre_um * 0.5) * 1e-6;
      debug_assert!(grain_radius_m > f32::EPSILON);
      // calculate cluster diametre in metres (double precision)
      let cluster_diameter_m = 2.0 * cluster_params.radius_m;

      // Micro Radius in UV Space: spans 1.0 across the cluster diametre
      let micro_radius = if cluster_diameter_m > 0.0 {
        (grain_radius_m / cluster_diameter_m) as f32
      } else {
        0.0
      };

      compute_rte(self, id).map(|(layer_idx, rte)| {
        // convert from metres to correct unit of measurement based on layer
        let (cluster_diameter_units, v_exp_units_per_s) = if layer_idx == 0 {
          // Macro layer: convert metres to AU
          (
            cluster_diameter_m / 149_597_870_700.0,
            v_exp_m_per_s / 149_597_870_700.0,
          )
        } else {
          // Micro layer: convert metres to km
          (cluster_diameter_m / 1000.0, v_exp_m_per_s / 1000.0)
        };
        // Macro scale in screen space by assuming base pixel size at distance = 1.0 units
        let macro_scale = (cluster_diameter_units * proj_scale) as f32;

        // Dispersion rate (Screen space pixel expansion per 1/300th second at distance = 1.0 units)
        // Shader computes: expandedScale = macroScale + (age * dispersionRate)
        let dispersion_rate = (v_exp_units_per_s / 300.0) * proj_scale;

        (
          layer_idx,
          DustDrawCall {
            entity_id: id,
            rte_mat: rte.to_mat4(),
            stream_color: ps.draw_params.stream_color,
            chunk_offset: 0,
            current_time: current_time_scaled_300ths,
            max_ttl: ttl_300ths_f32,
            macro_scale,
            micro_radius,
            num_spots,
            dispersion_rate,
          },
        )
      })
    });
    if !dust_calls.is_empty() {
      for (layer_idx, call) in dust_calls {
        get_or_create_layer!(layer_idx).dust_calls.push(call);
      }
    }

    // ------ 5. Singletons Rendering -------------------------------------------------------
    // Cursor
    const CURSOR_VERTEX_COUNT: u32 = 4;
    if let Some((_, id)) =
      self.query1_first_res_without::<_, HiddenComponent, _, _>(|id, _c: &CursorComponent| {
        if hidden_set.contains(&id) {
          None
        } else {
          Some(())
        }
      })
    {
      if let Some((layer_idx, rte)) = compute_rte(self, id) {
        let cur_g = self.global_transform_f64(id).unwrap_or_default();
        let rel_pos = cam_global_f64.position - cur_g.position;
        if let Ok(res) = device
          .get_cursor_resources(pe_handle)
          .or_else(|_| device.create_cursor_resources(cmd_buffer, pe_handle))
        {
          let l = get_or_create_layer!(layer_idx);
          l.cursor_call = Some(CursorDrawCall::from_result_and_matrix(
            res,
            CURSOR_VERTEX_COUNT,
            rte.to_mat4(),
            rte.scale.x(),
            l.near,
            l.far,
            rel_pos.to_f32().into(),
          ));
        } else {
          aethervk_oshal_rlib::log!("GPU Error Uploading Cursor Resources");
        }
      }
    }

    // Sun
    if let Some((rad, id)) = self.query2_first_res_without::<_, _, HiddenComponent, _, _>(
      |id, _t: &TransformComponent, s: &SunComponent| {
        if hidden_set.contains(&id) {
          None
        } else {
          Some(s.radius)
        }
      },
    ) {
      if let Some((layer_idx, mut rte)) = compute_rte(self, id) {
        // Note: this should never happen cause either sun is son of root with identity or sun is
        // root
        if layer_idx != self.ancestor_depth_layer(camera_entity) {
          if let Some(sun_g) = self.global_transform_f64(id) {
            rte.position = (sun_g.position - cam_global_f64.position).to_f32();
            rte.scale = safe_div_vec3(sun_g.scale, cam_global_f64.scale);
          }
        }
        if let Ok(pipe) = device.get_sun_pipeline_key(pe_handle) {
          let l = get_or_create_layer!(layer_idx);
          let sun_cam = render_scene.camera_data.rebuild_for_layer(l.near, l.far);
          l.sun_call = Some(SunDrawCall::from_model_and_camera(
            rte.to_mat4(),
            &sun_cam,
            pipe,
            id,
            rad,
          ));
        } else {
          aethervk_oshal_rlib::log!("GPU Error While getting Sun upload");
        }
      }
    }

    // Sky
    if let Some((_, id)) =
      self.query1_first_res_without::<_, HiddenComponent, _, _>(|id, _s: &SkyComponent| {
        if hidden_set.contains(&id) {
          None
        } else {
          Some(())
        }
      })
    {
      if let Ok(pipe) = device.get_sky_pipeline_key(pe_handle) {
        // TODO sky should not be in layer but global in macro
        let l = get_or_create_layer!(self.ancestor_depth_layer(id));
        let sky_cam = render_scene.camera_data.rebuild_for_layer(l.near, l.far);
        // projection matrix inversion can fail.
        l.sky_call = SkyDrawCall::from_camera(&sky_cam, pipe).ok();
      } else {
        aethervk_oshal_rlib::log!("GPU Error getting Sky resources");
      }
    }

    // Grid (Macro layer injects downwards)
    if let Some(_) =
      self.query1_first_res_without::<_, HiddenComponent, _, _>(|id, _g: &GridComponent| {
        if hidden_set.contains(&id) {
          None
        } else {
          Some(())
        }
      })
    {
      if let Ok(pipe) = device.get_grid_pipeline_kay(pe_handle) {
        for l in layer_map.values_mut() {
          if l.layer_index == 0 || l.grid_call.is_none() {
            // TODO density, size, color for now hardcoded
            l.grid_call = Some(GridDrawCall::new(pipe, 500.0, 1.0, [0.5, 0.5, 0.5]));
          }
        }
      } else {
        aethervk_oshal_rlib::log!("GPU Error getting grid resources")
      }
    }

    // Background
    if let Some(((color_top, color_bottom), id)) = self
      .query1_first_res_without::<_, HiddenComponent, _, _>(|id, b: &BackgroundComponent| {
        if hidden_set.contains(&id) {
          None
        } else {
          Some((b.color_top, b.color_bottom))
        }
      })
    {
      if let Ok(pipeline) = device.get_background_pipeline_key(pe_handle) {
        // Note: should always be macro layer TODO assert
        get_or_create_layer!(self.ancestor_depth_layer(id)).background_call =
          Some(BackgroundDrawCall {
            pipeline,
            color_top,
            color_bottom,
          });
      } else {
        aethervk_oshal_rlib::log!("GPU Error getting background resources")
      }
    }

    // ------ 6. UI & Text ------------------------------------------------------------------
    let mut ui_items = extract!(ui::Transform2DComponent, ui::UiComponent, |id, t2d, ui| {
      Some((*t2d, ui.clone()))
    });
    ui_items.sort_unstable_by(|a, b| {
      a.0
        .global_depth
        .cmp(&b.0.global_depth)
        .then(a.0.local_z_index.cmp(&b.0.local_z_index))
    });

    let mut gpu_ui = alloc::vec::Vec::with_capacity(ui_items.len());
    for (t2d, ui) in ui_items {
      let flags = if t2d.global_clip[0] > -9999.0 {
        gpu::UI_FLAG_HAS_CLIP
      } else {
        0
      };
      gpu_ui.push(gpu::UiElementGpu {
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

    if !gpu_ui.is_empty() {
      render_scene.ui_call = device.upload_ui(cmd_buffer, &gpu_ui).ok().flatten();
    }

    let mut text_items = extract!(
      ui::Transform2DComponent,
      ui::ScreenSpaceTextComponent,
      |id, t2d, txt| Some((*t2d, txt.clone()))
    );
    text_items.sort_unstable_by(|a, b| {
      a.0
        .global_depth
        .cmp(&b.0.global_depth)
        .then(a.0.local_z_index.cmp(&b.0.local_z_index))
    });

    let mut text_batch = alloc::vec::Vec::with_capacity(32);
    for (t2d, text_comp) in text_items {
      if let Ok(descriptor_index) = device.allocate_rasterized_font_atlas(
        cmd_buffer,
        text_comp.font_hash,
        text_comp.font_atlas.clone(),
      ) {
        // TODO: remove old
        let start_pos = [t2d.global_bounds[0], t2d.global_bounds[1]];
        let style = text::TextStyle {
          size_pt: text_comp.points,
          color: text_comp.color,
          style_flags: text_comp.style_flags,
        };
        text::push_text_to_batch(
          &text_comp.text,
          start_pos,
          &style,
          &text_comp.font_atlas,
          descriptor_index,
          &mut text_batch,
        );
      } else {
        aethervk_oshal_rlib::log!(
          "Error allocating descriptor index for text {:?}",
          text_comp.font_hash
        );
      }
    }

    if !text_batch.is_empty() {
      render_scene.text2_call = device.upload_text2(cmd_buffer, &text_batch).ok().flatten();
    }

    // ------ 7. Batch Uploads & Finalization -----------------------------------------------
    let mut depth_layers: alloc::vec::Vec<RenderLayer> = layer_map
      .into_values()
      .map(|mut l| {
        if let Some(sg_list) = sg_batch_buffers.remove(&l.layer_index) {
          let sg_data: alloc::vec::Vec<_> = sg_list
            .into_iter()
            .filter_map(|(id, m, r, sub)| {
              device.allocate_sphere_gizmo_instance(id).ok().map(|idx| {
                (
                  idx,
                  gpu::SphereGizmoDataGpu {
                    model: m.into(),
                    radius: r,
                    subdivisions: sub,
                    _pad: [0.0; 2],
                  },
                )
              })
            })
            .collect();
          l.sphere_gizmo_batch_call =
            device.upload_sphere_gizmos_batch(cmd_buffer, &sg_data).ok().flatten();
        }
        if let Some(traj_list) = traj_batch_buffers.remove(&l.layer_index) {
          l.trajectory_call = device.upload_trajectories(cmd_buffer, &traj_list).ok().flatten();
        }
        l
      })
      .collect();

    depth_layers.sort_by_key(|l| l.layer_index);
    render_scene.depth_layers = depth_layers;

    // ------------ Debug: log every 120 frames ------------
    #[cfg(debug_assertions)]
    {
      use core::sync::atomic::{AtomicU64, Ordering};
      static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);
      let frame = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
      if frame % 120 == 0 {
        aethervk_oshal_rlib::log!(
          "\x1b[36m[MULTI-SCALE] Frame {} | camera pos=({:.6},{:.6},{:.6})\x1b[0m",
          frame,
          camera_data.absolute_pos.x(),
          camera_data.absolute_pos.y(),
          camera_data.absolute_pos.z()
        )
      }
    }

    Ok(render_scene)
  }
}

fn safe_div(a: f32, b: f32) -> f32 {
  if b.abs() < 1e-15 { 0.0 } else { a / b }
}

fn safe_div_vec3(a: Vec3f32, b: Vec3f32) -> Vec3f32 {
  Vec3f32::from_components(
    safe_div(a.x(), b.x()),
    safe_div(a.y(), b.y()),
    safe_div(a.z(), b.z()),
  )
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
  use parking_lot::RwLock;

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
