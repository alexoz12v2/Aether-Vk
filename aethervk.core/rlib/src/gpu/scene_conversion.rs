//! scene_conversion module.

pub mod indicator_layout;
pub mod trajectory_indicator;
pub mod trajectory_rte;

use crate::{
  gpu::{self, RenderDevice, frame::*},
  gpu_backends::vulkan,
  gpu_invalid_arg,
  scene::{
    BackgroundComponent, CameraComponent, CameraProjection, CursorComponent, EntityId,
    GizmoComponent, GridComponent, HiddenComponent, HighResTransformComponent,
    ImageBillboardComponent, IndicatorComponent, MarkersComponent, MeasurementComponent,
    ParticleSystemComponent, ReferenceFrameComponent, ReferentialIndicatorComponent, Scene,
    SkyComponent, SphereGizmoComponent, StaticMeshComponent, SunComponent,
    TrajectoryIndicatorComponent, TransformComponent, text, trajectory::TrajectoryComponent, ui,
  },
  types::GpuResult,
};
use aethervk_oshal_rlib::{
  math::{
    FloatLike,
    matrix::{Matrix4, MatrixVectorMul, mat4::Mat4x4f32, mat4f64::Mat4x4f64},
    vector::{Vector, Vector3, Vector4, vec3::Vec3f32, vec3f64::Vec3f64, vec4::Quat},
  },
  os::{pool::ThreadPool, time::timeus_t},
};
use function_name::named;

use crate::simulation_api::reposition::AU_TO_KM;

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
    sky_rotation_offset: Quat,
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
    sky_rotation_offset: Quat,
    debug_name: &str,
  ) -> GpuResult<gpu::RenderScene> {
    // ------ 1. Precompute Camera & Hierarchy ----------------------------------------------
    ui::update_ui_layouts(self, [window_extent[0] as f32, window_extent[1] as f32]);
    let should_par = self.should_parallelize() && pool.is_some();

    let cam_global_f64 = self
      .global_transform_f64(camera_entity)
      .ok_or(gpu_invalid_arg!("invalid camera entity"))?;

    let cam_comp =
      self
        .with_component(camera_entity, |c: &CameraComponent| *c)
        .ok_or(gpu_invalid_arg!(
          "scene has no camera compoent on the specified entity"
        ))?;

    // The camera projection (near/far, ortho extents) is always expressed in AU by the UI, even
    // when the camera is parented inside a micro frame (CometOrbiting / EarthPosition): never
    // scale it by the camera's ancestor frame scale (that turned 6 km ortho extents into 4e-17).
    let camera_data = CameraRenderData::new_f64(&cam_global_f64, &cam_comp, 1.0, window_extent);

    // Filter hidden subtrees
    let hidden_roots = if should_par {
      self.query1_res_par::<HiddenComponent, _, _>(unsafe { pool.unwrap_unchecked() }, |id, _| {
        Some(id)
      })
    } else {
      self.query1_res::<HiddenComponent, _, _>(|id, _| Some(id))
    };
    // Subtrees outside this camera's render layers (`RenderLayerComponent` vs
    // `CameraRenderMaskComponent`) are hidden for this camera only: the Earth globe, axis, labels
    // and indicator exist for UpZenith cameras alone.
    let masked_roots = crate::simulation_api::earth_visual::masked_roots(self, camera_entity);
    let mut hidden_set = hashbrown::HashSet::with_capacity(128);
    for root_id in hidden_roots.into_iter().map(|(id, _)| id).chain(masked_roots) {
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
    let camera_flags = self
      .with_component(camera_entity, |s: &crate::scene::CameraShadingComponent| s.0)
      .unwrap_or(0);
    let camera_exposure = self
      .with_component(camera_entity, |e: &crate::scene::CameraExposureComponent| e.0)
      .unwrap_or(1.0);
    let hide_behind = camera_flags & crate::scene::CAMERA_HIDE_BEHIND != 0;
    let is_ortho = matches!(
      camera_data.projection_params,
      CameraProjectionParams::Orthographic { .. }
    );
    let (macro_near, macro_far) = macro_layer_depth(
      cam_comp.near_plane() as f64,
      cam_comp.far_plane() as f64,
      is_ortho,
      hide_behind,
    );

    let mut layer_bounds: hashbrown::HashMap<u32, (f64, f64)> =
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

    // for each micro layer, compute transform relative to camera, and from that, bounds.
    //
    // DEADLOCK FIX: `query1_without` holds `archetypes.read()` for the duration of its
    // callback. Calling `get_relative_transform_f64` (→ `with_component` → `archetypes.read()`)
    // from inside that callback creates a re-entrant read acquisition. Under `parking_lot`'s
    // write-preferring policy, a concurrent `archetypes.write()` from the logic thread's
    // `remove_component` will block the re-entrant read while the outer read guard is still
    // live, deadlocking both threads.
    //
    // Fix: Phase 1 — collect only the data we need from each frame entity into a Vec,
    // keeping the callback free of nested scene queries so the read lock is released
    // before Phase 2.
    struct FrameEntry {
      id: EntityId,
      depth_layer: u32,
      scale: f32,
      soi_radius: f32,
    }
    let frame_entries: alloc::vec::Vec<FrameEntry> = {
      let mut entries = alloc::vec::Vec::new();
      self.query1_without::<ReferenceFrameComponent, HiddenComponent, _>(
        |id, frame: &ReferenceFrameComponent| {
          debug_assert!((frame.depth_layer == 0) == (self.get_root().unwrap() == id));
          if frame.depth_layer > 0 {
            entries.push(FrameEntry {
              id,
              depth_layer: frame.depth_layer,
              scale: frame.scale,
              soi_radius: frame.soi_radius,
            });
          }
        },
      );
      entries
    };
    // Earth observer light time (`earth_observer::LightTimeViewComponent`): the comet's micro
    // layer is drawn where the telescope sees it, `x(t_r) = x(t) + offset`, by moving that layer's
    // camera by `−offset` (the committed comet transform is the geometric one). A comet on the
    // macro layer is not displaced (it would move everything); its dust is retarded in
    // `dust_draw_calls` either way.
    let light_time = self
      .with_component(
        camera_entity,
        |l: &crate::simulation_api::earth_observer::LightTimeViewComponent| *l,
      )
      .map(|l| (self.ancestor_depth_layer(l.comet), l))
      .filter(|(layer, _)| *layer > 0);
    // the comet's root-level overlays follow the same displacement (collected before the
    // extraction closures)
    let light_time_overlay = LightTimeOverlay::of(self, camera_entity);
    // Phase 2 — `archetypes.read()` from `query1_without` is now released.
    // Safe to call `get_relative_transform_f64` which re-acquires `archetypes.read()`.
    for entry in frame_entries {
      layer_frame_entities.insert(entry.depth_layer, entry.id);
      if let Some(mut cam_in_frame_f64) = self.get_relative_transform_f64(camera_entity, entry.id) {
        if let Some((_, lt)) = light_time.filter(|(layer, _)| *layer == entry.depth_layer) {
          if let Some(frame) = self.global_transform_f64(entry.id) {
            use crate::simulation_api::earth_observer::{qinverse, qnormalize, rotate};
            let r = qnormalize([
              frame.rotation[0],
              frame.rotation[1],
              frame.rotation[2],
              frame.rotation[3],
            ]);
            // world units (AU) per local unit of the frame's children
            let k = frame.scale.x() as f64 * entry.scale as f64;
            let o = rotate(qinverse(r), lt.offset_au);
            cam_in_frame_f64.position = cam_in_frame_f64.position
              - aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
                o[0] / k,
                o[1] / k,
                o[2] / k,
              );
          }
        }
        camera_in_frames.insert(entry.depth_layer, cam_in_frame_f64);

        let dist_local = cam_in_frame_f64.position.length();
        let soi_local = (entry.soi_radius / entry.scale) as f64;
        let safe_micro_near = 0.001_f64;
        let tight_near = (dist_local - soi_local).max(safe_micro_near);
        let tight_far = (dist_local + soi_local).max(tight_near + safe_micro_near);

        layer_bounds.insert(entry.depth_layer, (tight_near, tight_far));
        layer_frame_scales.insert(entry.depth_layer, entry.scale);
      }
    }
    // Composite precision references, one per micro layer from the final per-layer maps (the
    // frame that won the layer: two frames sharing a layer must not give a track an anchor whose
    // reference the layer's meshes do not use). D_m = |camera in frame| (frame units, incl. the
    // light-time shift); o_m = −camera·frame_scale (AU, camera-relative), the same factor as the
    // composite's layerUnitAu, so |o_m| = D_m·frame_scale by construction.
    let mut layer_ref_dists: hashbrown::HashMap<u32, f64> = hashbrown::HashMap::with_capacity(8);
    let mut anchor_candidates: alloc::vec::Vec<([f64; 3], u32)> = alloc::vec::Vec::with_capacity(8);
    if crate::gpu::frame::PRECISE_COMPOSITE {
      for (&layer, cam) in camera_in_frames.iter() {
        let Some(&scale) = layer_frame_scales.get(&layer) else {
          continue; // layer 0 (the root) has no micro reference
        };
        if layer == 0 || layer >= 8 {
          continue;
        }
        let p = [cam.position.x(), cam.position.y(), cam.position.z()];
        layer_ref_dists.insert(layer, cam.position.length());
        anchor_candidates.push((trajectory_rte::anchor_origin(p, scale as f64), layer));
      }
    }

    let mut layer_map: hashbrown::HashMap<u32, RenderLayer> = hashbrown::HashMap::with_capacity(16);

    // Ensures layers are lazily construted into our final memory footprint format only when
    // actually used
    macro_rules! get_or_create_layer {
      ($layer_idx:expr) => {
        layer_map.entry($layer_idx).or_insert_with(|| {
          let (near, far) = layer_bounds
            .get(&$layer_idx)
            .copied()
            .unwrap_or((macro_near, macro_far));
          let scale = layer_frame_scales.get(&$layer_idx).copied().unwrap_or(1.0);
          RenderLayer::new(
            $layer_idx,
            scale,
            camera_in_frames
              .get(&$layer_idx)
              .map(|c| c.position.to_f32())
              .unwrap_or_default(),
            near,
            far,
          )
        })
      };
    }

    // Instant O(1) mathematical `f64` Relative-To-Eye (RTE) calculation using the cache
    let compute_rte =
      |scene: &Scene, id: EntityId| -> Option<(u32, crate::scene::HighResTransformComponent)> {
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

        Some((
          layer_idx,
          rte_relative_to_camera(pos_f64, rot, obj_scale, cam_in_frame),
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
      dust_softening: crate::scene::dust::DUST_SOFTENING_DEFAULT,
      dust_volume_detail: crate::scene::dust::DUST_VOLUME_DETAIL_DEFAULT,
      dust_haze_detail: crate::scene::dust::DUST_HAZE_DETAIL_DEFAULT,
      dust_propagate_detail: crate::scene::dust::DUST_PROPAGATE_DETAIL_DEFAULT,
      dust_reuse_px: crate::scene::dust::dust_reuse_px_default(),
      dust_feedback_gen: 0,
      dust_sim_speed: 0.0,
      dust_view_flags: 0,
      dust_exposure_ev: 0.0,
      dust_nucleus_px: None,
      dust_flow: Default::default(),
      dust_view: Default::default(),
      layer_ref_dist: core::array::from_fn(|i| layer_ref_dists.get(&(i as u32)).copied().unwrap_or(0.0)),
      mesh_shading: camera_flags,
      mesh_exposure: camera_exposure,
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

    // ------ 0. Flush pending mesh vertex buffer updates ------------------------------
    // Records vkCmdCopyBuffer + TRANSFER→VERTEX_INPUT barrier for each pending update
    // into this frame's cmd_buffer, then swaps the handles in physical_mesh2_resources.
    // Must run before any draw call that references the position buffer of an affected mesh.
    //
    // INVARIANT: All presentation engines submit to the same Vulkan graphics queue,
    // so draining the queue here (on the first PE's cmd_buffer) is safe. Subsequent
    // PE command buffers in the same vkQueueSubmit batch will see the already-swapped
    // ForwardMesh2RenderResource and the barrier is resolved for all of them.
    let _ = device.flush_pending_mesh_updates(cmd_buffer);

    // ------ 4. Fused Component Extraction & GPU Draw Call Creation ------------------------
    // 1. Meshes
    let extracted_meshes = extract!(StaticMeshComponent, |id, mesh| {
      compute_rte(self, id).map(|(layer_idx, mut rte)| {
        // Apply MeshScaleMultiplierComponent if present
        self.with_component(
          id,
          |scale_cmp: &crate::scene::MeshScaleMultiplierComponent| {
            rte.scale = Vec3f32::from_components(
              rte.scale.x() * scale_cmp.multiplier,
              rte.scale.y() * scale_cmp.multiplier,
              rte.scale.z() * scale_cmp.multiplier,
            );
          },
        );

        // TODO reintroduce following and selected if necessary. If reintroduced, the selection and
        // following state should have been stored in the scene
        let outline = get_mesh_outline(false, false, render_outline);
        (layer_idx, id, mesh.clone(), rte, outline)
      })
    });

    // Sphere gizmo extraction is done early so the depth fitting pass below can
    // include gizmo bounding spheres in the per-layer near/far computation.
    // The sg_batch_buffers push and get_or_create_layer! call still happen downstream.
    let extracted_sg = extract!(SphereGizmoComponent, |id, sg| {
      if !sg.is_visible {
        return None;
      }
      // TODO remove sg.local_frame
      compute_rte(self, id).map(|(layer_idx, rte)| {
        // sphere_gizmo.vert generates localPos in km (from data.radius in km).
        // viewProj for the micro layer is also in km.
        // The RTE scale (≈6.68e-9 AU/km) baked into rte.to_mat4() diagonal would
        // multiply every sphere vertex offset by 6.68e-9, collapsing a 50 km sphere
        // to a 334 μm point — invisible at any viewing distance.
        // Override scale to (1,1,1): preserves rotation and translation, lets km be km.
        let mut rte_for_gizmo = rte;
        rte_for_gizmo.scale = Vec3f32::from_components(1.0, 1.0, 1.0);
        (
          layer_idx,
          id,
          rte_for_gizmo.to_transform().to_mat4::<Mat4x4f32>() * sg.local_frame,
          [rte.position.x(), rte.position.y(), rte.position.z()],
          sg.radius,
          sg.subdivisions,
          gpu::SPHERE_GIZMO_STYLE_SPHERE,
          0.0f32,
          sphere_gizmo_parts_color(self, id, layer_frame_entities.get(&layer_idx).copied()),
        )
      })
    });
    // Dashed polar axes (`earth_visual::PolarAxisGizmoComponent`) ride the same instanced
    // pipeline in the polar-axis style: radius = half length of the axis, km, scale forced to 1.
    let mut extracted_sg = extracted_sg;
    extracted_sg.extend(extract!(
      crate::simulation_api::earth_visual::PolarAxisGizmoComponent,
      |id, axis| {
        compute_rte(self, id).map(|(layer_idx, rte)| {
          let mut rte_for_gizmo = rte;
          rte_for_gizmo.scale = Vec3f32::from_components(1.0, 1.0, 1.0);
          (
            layer_idx,
            id,
            rte_for_gizmo.to_transform().to_mat4::<Mat4x4f32>(),
            [rte.position.x(), rte.position.y(), rte.position.z()],
            axis.radius_km * axis.length_factor,
            0.0f32,
            gpu::SPHERE_GIZMO_STYLE_POLAR_AXIS,
            axis.dash_count,
            [0.85f32, 0.9, 1.0, 1.0],
          )
        })
      }
    ));
    // Jet direction lines (`JetDirectionLineComponent`): from the nucleus centre along the site
    // direction, 0.75 × the body axes, the jet's colour (dimmed while the site is unlit).
    extracted_sg.extend(extract!(crate::scene::JetDirectionLineComponent, |id, line| {
      if !line.is_visible {
        return None;
      }
      let nucleus = self.get_parent(id)?;
      let gizmo_radius = self.with_component(nucleus, |g: &SphereGizmoComponent| g.radius)?;
      let site = self.with_component(id, |t: &crate::scene::TransformComponent| t.position)?;
      let frame = jet_line_frame(vec3_f64(site))?;
      let mut color = line.color;
      if jet_site_lit(self, id) == Some(false) {
        for c in &mut color[..3] {
          *c *= UNLIT_JET_MARKER_DIM;
        }
      }
      compute_rte(self, nucleus).map(|(layer_idx, rte)| {
        let mut rte_for_gizmo = rte;
        rte_for_gizmo.scale = Vec3f32::from_components(1.0, 1.0, 1.0);
        (
          layer_idx,
          id,
          rte_for_gizmo.to_transform().to_mat4::<Mat4x4f32>() * frame,
          [rte.position.x(), rte.position.y(), rte.position.z()],
          gizmo_radius * crate::scene::JET_LINE_LENGTH_OVER_GIZMO_RADIUS,
          0.0f32,
          gpu::SPHERE_GIZMO_STYLE_JET_LINE,
          0.0f32,
          color,
        )
      })
    }));

    // ------ Depth fitting pass -------------------------------------------------------
    // Refits micro-layer near/far from actual draw-call geometry rather than the
    // conservative SOI sphere. Only meshes whose bounding sphere passes a frustum
    // cull test contribute to the tight bounds — objects outside the frustum (behind
    // the camera, off to the sides, etc.) must not expand the depth range.
    //
    // The SOI-sphere bounds (computed in Phase 2 above) remain as a fallback if no
    // StaticMeshComponent survives the cull test for a given micro layer.
    //
    // Bounding radius is read from vertex data and multiplied by the entity's RTE scale
    // (procedural spheres have scale 1 with the radius baked in via
    // update_uv_sphere_radius_in_place; imported meshes carry radius_km / bounding_sphere).
    //
    // Constants:
    //   DEPTH_NEAR_FLOOR — absolute minimum near plane (1 m) to handle camera-inside-mesh
    //                      (perspective only; orthographic near may be negative)
    //   DEPTH_MARGIN     — 5% padding on both sides to prevent near/far edge clipping
    {

      // Frustum plane extraction via Gribb-Hartmann from VP in f64.
      // L/R/T/B planes depend only on FOV + aspect (not near/far), so the macro-layer
      // VP is valid for culling objects in any micro layer.
      // Planes are in RTE world space (km); testing rte.position directly is correct.
      let vp = camera_data.proj_f64 * camera_data.view_f64;

      // Helper: element at (row i, col j) from column-major Mat4x4f64.
      // cols[j] = column j; element at row i uses .x()/.y()/.z()/.w().
      let vp_e = |i: usize, j: usize| -> f64 {
        use aethervk_oshal_rlib::math::vector::Vector4;
        match i {
          0 => vp.cols[j].x(),
          1 => vp.cols[j].y(),
          2 => vp.cols[j].z(),
          _ => vp.cols[j].w(),
        }
      };

      // Extract rows 0, 1, 3 from VP (rows 2 encodes depth — not needed for L/R/T/B).
      let r0 = [vp_e(0, 0), vp_e(0, 1), vp_e(0, 2), vp_e(0, 3)];
      let r1 = [vp_e(1, 0), vp_e(1, 1), vp_e(1, 2), vp_e(1, 3)];
      let r3 = [vp_e(3, 0), vp_e(3, 1), vp_e(3, 2), vp_e(3, 3)];

      // Gribb-Hartmann frustum planes (Vulkan NDC z∈[0,1]).
      // A point p is inside plane iff dot(plane.xyz, p) + plane.w ≥ 0.
      let frustum_planes: [[f64; 4]; 5] = [
        [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]], // Left
        [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]], // Right
        [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]], // Bottom
        [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]], // Top
        r3, // Front (w_clip ≥ 0 — object is in front of the camera)
      ];

      // Sphere-vs-frustum: returns false if the sphere is COMPLETELY outside any plane.
      let sphere_in_frustum = |cx: f64, cy: f64, cz: f64, r: f64| -> bool {
        for p in &frustum_planes {
          let dot = p[0] * cx + p[1] * cy + p[2] * cz + p[3];
          // Plane magnitude needed to convert homogeneous dot to world-space distance.
          let mag = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
          if dot < -r * mag {
            return false; // sphere entirely outside this half-space
          }
        }
        true
      };

      let mut per_layer_depth: hashbrown::HashMap<u32, (f64, f64)> =
        hashbrown::HashMap::with_capacity(4);

      let fit_slab = |depth: f64, r: f64| fit_depth_slab(depth, r, is_ortho, hide_behind);

      for (layer_idx, _id, mesh, rte, _outline) in &extracted_meshes {
        if *layer_idx == 0 {
          continue; // macro layer uses camera near/far from UI — do not touch
        }

        // Bounding radius from vertex positions, multiplied by the entity's RTE scale.
        // The procedural sphere has scale 1 (radius baked into positions), but GLTF/OBJ/PLY
        // comets get `scale = radius_km / bounding_sphere` and MeshScaleMultiplierComponent
        // also scales rte — ignoring it produced too-tight near/far and clipped the mesh.
        let max_scale = {
          use aethervk_oshal_rlib::math::vector::Vector3;
          (rte.scale.x().abs()).max(rte.scale.y().abs()).max(rte.scale.z().abs()) as f64
        };
        let obj_radius = mesh
          .mesh
          .vertices
          .iter()
          .map(|v| {
            let [x, y, z] = v.position;
            ((x * x + y * y + z * z) as f64).sqrt()
          })
          .fold(0.0f64, f64::max)
          * max_scale;

        // Camera-to-object-center vector in km (rte.position = obj_pos − cam_pos in km).
        let (cx, cy, cz) = {
          use aethervk_oshal_rlib::math::vector::Vector3;
          (
            rte.position.x() as f64,
            rte.position.y() as f64,
            rte.position.z() as f64,
          )
        };

        // We need the planar depth along the camera's forward axis for clipping planes.
        use aethervk_oshal_rlib::math::vector::{Vector4, vec4f64::Vec4f64};
        let view_pos = camera_data.view_f64 * Vec4f64::from_components(cx, cy, cz, 1.0);
        // Signed forward depth (view −Y is forward). `fit_slab` takes abs() for perspective.
        let obj_depth = -view_pos.y();

        // Frustum cull: skip objects whose bounding sphere lies entirely outside the view.
        let au_scale = 1.0 / AU_TO_KM as f64;
        if !sphere_in_frustum(
          cx * au_scale,
          cy * au_scale,
          cz * au_scale,
          obj_radius * au_scale,
        ) {
          continue;
        }

        let (obj_near, obj_far) = fit_slab(obj_depth, obj_radius);

        let e = per_layer_depth.entry(*layer_idx).or_insert((f64::MAX, f64::NEG_INFINITY));
        e.0 = e.0.min(obj_near);
        e.1 = e.1.max(obj_far);
      }

      // Sphere gizmos: axes extend to `radius * 1.5` km beyond the sphere surface
      // (sphere_gizmo.vert line 121), with arrowheads adding another `radius * 0.2`
      // (line 151). The full gizmo bounding sphere is therefore `radius * 1.7` km.
      // -> scratch that, it seems that the yellow one is still cut. Trying out 2.1 km
      // extracted_sg tuple: (layer_idx, id, mat, center, rad, sub)
      for (layer_idx, _id, _mat, _center, rad, _sub, style, _dash, _color) in &extracted_sg {
        if *layer_idx == 0 {
          continue;
        }

        // Gizmo center = RTE position of the parent entity. We need to recompute it
        // here since extracted_sg stores the final mat4, not the raw rte.position.
        // Mat4x4f32 columns are named .x/.y/.z/.w (column-major); .w is the
        // translation column (RTE offset in km, since scale was forced to 1).
        let (cx, cy, cz) = {
          use aethervk_oshal_rlib::math::vector::Vector4;
          (_mat.w.x() as f64, _mat.w.y() as f64, _mat.w.z() as f64)
        };

        let view_pos = camera_data.view_f64
          * aethervk_oshal_rlib::math::vector::vec4f64::Vec4f64::from_components(cx, cy, cz, 1.0);
        let obj_depth = {
          use aethervk_oshal_rlib::math::vector::Vector4;
          -view_pos.y()
        };

        // Bounding radius: full axis + arrowhead envelope; a polar axis or a jet line ends at `rad`.
        let gizmo_radius = if *style == gpu::SPHERE_GIZMO_STYLE_POLAR_AXIS
          || *style == gpu::SPHERE_GIZMO_STYLE_JET_LINE
        {
          *rad as f64
        } else {
          (*rad as f64) * 2.1
        };

        let au_scale = 1.0 / AU_TO_KM as f64;
        if !sphere_in_frustum(
          cx * au_scale,
          cy * au_scale,
          cz * au_scale,
          gizmo_radius * au_scale,
        ) {
          continue;
        }

        let (obj_near, obj_far) = fit_slab(obj_depth, gizmo_radius);

        let e = per_layer_depth.entry(*layer_idx).or_insert((f64::MAX, f64::NEG_INFINITY));
        e.0 = e.0.min(obj_near);
        e.1 = e.1.max(obj_far);
      }

      // Commit: replace SOI-sphere fallback with per-object tight bounds where
      // visible geometry is present. Invalid entries (obj_far <= obj_near) are skipped.
      for (layer_idx, (obj_near, obj_far)) in per_layer_depth {
        if obj_far > obj_near {
          layer_bounds.insert(layer_idx, (obj_near, obj_far));
        }
      }
    }
    // ------ End depth fitting pass ---------------------------------------------------

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
        let mat = rte.to_transform().to_mat4();
        // Capture the f64 RTE center before to_mat4() truncates it to f32.
        // rte.position = pos_f64 − cam_pos_f64 (computed by compute_rte), so it is
        // already the exact RTE translation in f64, ready for body-fixed sun direction.
        let center_rte_f64 = {
          use aethervk_oshal_rlib::math::vector::Vector3;
          [rte.position.x(), rte.position.y(), rte.position.z()]
        };
        let l = get_or_create_layer!(layer_idx);

        // capture for debug
        // aethervk_oshal_rlib::log!("StaticMesh {} pushed to layer {}", mesh.mesh.id, layer_idx);
        l.draw_calls.push(DrawCall::from_handles_and_matrix(
          res,
          mesh.mesh.indices.len() as u32,
          outline,
          mat,
          center_rte_f64,
          mesh.emissive_color[3],
          [
            mesh.emissive_color[0],
            mesh.emissive_color[1],
            mesh.emissive_color[2],
          ],
          true,
          0,
          layer_idx as u32,
        ));
        if let Some(dc) = l.draw_calls.last_mut() {
          dc.layer_ref_dist = layer_ref_dists.get(&layer_idx).copied().unwrap_or(0.0);
        }
      } else {
        let err = unsafe { gpu_res.unwrap_err_unchecked() };
        aethervk_oshal_rlib::log!("GPU Upload Error: {}", err);
      }
    }

    // 2. Billboards
    let extracted_billboards = extract!(ImageBillboardComponent, |id, i| {
      compute_rte(self, id).map(|(layer_idx, rte)| {
        (
          layer_idx,
          rte.to_transform().to_mat4(),
          i.texture_id,
          i.billboard_type,
        )
      })
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
      compute_rte(self, id)
        .map(|(layer_idx, rte)| (layer_idx, rte.to_transform().to_mat4(), m.clone()))
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
        let mat: Mat4x4f32 = rte.to_transform().to_mat4();
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
          Mat4x4f64::translation(rte.position),
          Mat4x4f32::from_quat_custom_frame(rte.rotation.to_quat()),
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
        for (layer_idx, id, t_mat_f64, r_mat, scale) in extracted_gizmos {
          if let Ok(idx) = device.update_gizmo_instance(id, t_mat_f64, r_mat, pe_handle) {
            get_or_create_layer!(layer_idx)
              .gizmo_calls
              .push(GizmoDrawCall::from_values(pipe, scale, idx));
          }
        }
      } else {
        aethervk_oshal_rlib::log!("GPU Upload Error creating/getting gizmo resources");
      }
    }

    // 6. Sphere Gizmos (Batched - deferred upload)
    let mut sg_batch_buffers = hashbrown::HashMap::<
      u32,
      alloc::vec::Vec<(EntityId, Mat4x4f32, [f64; 3], f32, f32, f32, f32, [f32; 4])>,
    >::with_capacity(16);
    for (layer_idx, id, mat, center, rad, sub, style, dash, color) in extracted_sg {
      get_or_create_layer!(layer_idx);
      sg_batch_buffers
        .entry(layer_idx)
        .or_default()
        .push((id, mat, center, rad, sub, style, dash, color));
    }

    // 7. Trajectories (Batched - deferred upload)
    // per layer: the trajectories and, at the same index, their anchor (composite precision §2.3)
    let mut traj_batch_buffers = hashbrown::HashMap::<
      u32,
      (
        alloc::vec::Vec<(EntityId, TrajectoryComponent, Mat4x4f32)>,
        alloc::vec::Vec<Option<trajectory_rte::AnchorGpu>>,
      ),
    >::with_capacity(16);
    let identity = <Mat4x4f32 as aethervk_oshal_rlib::math::matrix::SquareMatrix>::identity();
    let view_extent_hint = trajectory_rte::ViewScale::from_projection(&cam_comp.projection);
    let extracted_traj = extract!(TrajectoryComponent, |id, traj| {
      // Camera-relative control points computed in f64 and only then cast to f32: the GPU never
      // sees absolute AU (f32 ulp ~1.2e-7 AU = 18 km at 1 AU), so tracks stay exact up close.
      let extra = match self.with_component(
        id,
        |e: &crate::scene::trajectory::EffectiveTrajectoryComponent| e.head_segment(),
      ) {
        Some(Some(seg)) => seg,
        _ => alloc::vec::Vec::new(),
      };
      compute_rte(self, id).map(|(layer_idx, mut rte)| {
        // a comet line under the observer's light time: drawn where the telescope sees the comet
        if layer_idx == 0 {
          if let Some(o) = light_time_overlay.track_offset(id) {
            rte.position = rte.position + o;
          }
        }
        // layer-0 pieces near a micro frame origin are anchored there (empty candidate list
        // while PRECISE_COMPOSITE is off: the output is camera_relative_control_points')
        let (rel, anchor) = if layer_idx == 0 {
          trajectory_rte::control_points_with_anchor(
            traj,
            &extra,
            &rte,
            view_extent_hint,
            &anchor_candidates,
          )
        } else {
          (
            trajectory_rte::camera_relative_control_points(traj, &extra, &rte, view_extent_hint),
            None,
          )
        };
        // dust trajectory lines diagnostics (`DUST_VIEW_TRAJECTORIES`): once per change of
        // the piece count, what reaches the trajectory batch
        if self
          .with_component(id, |_: &crate::scene::trajectory::DustTrajectoryLinesComponent| ())
          .is_some()
        {
          static LAST_PIECES: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(usize::MAX);
          let pieces = rel.len() / 4;
          if LAST_PIECES.swap(pieces, core::sync::atomic::Ordering::Relaxed) != pieces {
            aethervk_oshal_rlib::log!(
              "[Dust trajectories] draw: entity {} layer {layer_idx}: {} segments -> {pieces} pieces, anchored {}, rte |pos| {:.3e}",
              id.as_ffi(),
              traj.control_points_f64.len() / 4,
              anchor.is_some(),
              {
                let p = rte.position;
                (p.x() * p.x() + p.y() * p.y() + p.z() * p.z()).sqrt()
              }
            );
          }
        }
        let mut t = traj.clone();
        t.control_points = rel;
        (layer_idx, id, t, identity, anchor)
      })
    });
    for (layer_idx, id, traj, mat, anchor) in extracted_traj {
      get_or_create_layer!(layer_idx);
      let e = traj_batch_buffers.entry(layer_idx).or_default();
      e.0.push((id, traj, mat));
      e.1.push(anchor);
    }

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

    // 9. Particles (dust v3), see `Scene::dust_draw_calls`: not children of the comet
    let dust_calls: alloc::vec::Vec<(u32, alloc::vec::Vec<DustDrawCall>)> = self
      .dust_draw_calls(camera_entity)
      .into_iter()
      .filter(|(layer_idx, _)| camera_in_frames.contains_key(layer_idx))
      .collect();
    if !dust_calls.is_empty() {
      for (layer_idx, calls) in dust_calls {
        get_or_create_layer!(layer_idx).dust_calls.extend(calls);
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
            rte.to_transform().to_mat4(),
            rte.scale.x(),
            l.near as f32,
            l.far as f32,
            l.frame_scale,
            rel_pos.to_f32().into(),
          ));
        } else {
          aethervk_oshal_rlib::log!("GPU Error Uploading Cursor Resources");
        }
      }
    }

    // Sun
    if let Some((rad_km, id)) = self.query2_first_res_without::<_, _, HiddenComponent, _, _>(
      |id, _t: &TransformComponent, s: &SunComponent| {
        if hidden_set.contains(&id) {
          None
        } else {
          Some(s.radius_km)
        }
      },
    ) {
      if let Some((layer_idx, mut rte)) = compute_rte(self, id) {
        if let Ok(pipe) = device.get_sun_pipeline_key(pe_handle) {
          let l = get_or_create_layer!(layer_idx);

          if layer_idx > 0 {
            use aethervk_oshal_rlib::math::vector::Vector3;
            let (cx, cy, cz) = (
              rte.position.x() as f64,
              rte.position.y() as f64,
              rte.position.z() as f64,
            );
            let obj_dist = (cx * cx + cy * cy + cz * cz).sqrt();
            // rad_km is already in km, matching the micro-layer's local unit.
            // For a macro layer (layer_idx == 0) this branch is skipped.
            let obj_radius = 2.0 * (rad_km as f64);
            let obj_near = (obj_dist - obj_radius * 1.05).max(0.001);
            let obj_far = obj_dist + obj_radius * 1.05;

            l.near = l.near.min(obj_near);
            l.far = l.far.max(obj_far);
          }

          let sun_cam = render_scene.camera_data.rebuild_for_layer(l.near, l.far, l.frame_scale);
          l.sun_call = Some(SunDrawCall::from_model_and_camera(
            rte.to_mat4_f64(),
            &sun_cam,
            pipe,
            id,
            rad_km,
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
        let sky_layer_idx = self.ancestor_depth_layer(id);
        debug_assert_eq!(
          sky_layer_idx, 0,
          "SkyComponent entity must be a direct child of root (depth_layer=0); \
           drawing it in a micro layer would incorrectly render the sky with micro near/far planes."
        );
        let l = get_or_create_layer!(sky_layer_idx);
        let sky_cam = render_scene.camera_data.rebuild_for_layer(l.near, l.far, l.frame_scale);
        // projection matrix inversion can fail.
        l.sky_call = SkyDrawCall::from_camera(&sky_cam, pipe, sky_rotation_offset).ok();
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
        aethervk_oshal_rlib::log!("GPU Error getting grid resources");
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
        aethervk_oshal_rlib::log!("GPU Error getting background resources");
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

    // ------ 8. Indicators -----------------------------------------------------------------
    {
      let w = window_extent[0] as f32;
      let h = window_extent[1] as f32;

      let cam_pos_km = Vec3f64::from_components(
        cam_global_f64.position.x() * AU_TO_KM,
        cam_global_f64.position.y() * AU_TO_KM,
        cam_global_f64.position.z() * AU_TO_KM,
      );

      use aethervk_oshal_rlib::math::{
        matrix::{Matrix, mat4f64::Mat4x4f64},
        vector::vec4f64::Vec4f64,
      };
      let view_proj_f64: Mat4x4f64 = camera_data.proj_f64 * camera_data.view_f64;

      let mut indicator_inputs: alloc::vec::Vec<indicator_layout::IndicatorInput> =
        alloc::vec::Vec::new();

      let mut atlas_candidates: alloc::vec::Vec<(
        alloc::sync::Arc<crate::scene::text::FontAtlas>,
        u64,
      )> = alloc::vec::Vec::new();

      // Track NDC of referential indicators so trajectories can avoid them
      let mut parent_ndc_map: hashbrown::HashMap<EntityId, [f32; 2]> = hashbrown::HashMap::new();

      // 8a. Basic IndicatorComponent
      self.query1_without::<IndicatorComponent, HiddenComponent, _>(
        |id, ind: &IndicatorComponent| {
          if hidden_set.contains(&id) {
            return;
          }

          let rte_km = ind.global_position_km - cam_pos_km;
          let rte_au = aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
            rte_km.x() / AU_TO_KM,
            rte_km.y() / AU_TO_KM,
            rte_km.z() / AU_TO_KM,
          );
          let clip = view_proj_f64.mul_vector(Vec4f64::from_components(
            rte_au.x(),
            rte_au.y(),
            rte_au.z(),
            1.0,
          ));

          if clip.w() <= 0.0 {
            return;
          }
          let ndc_x = clip.x() / clip.w();
          let ndc_y = clip.y() / clip.w();
          if ndc_x < -1.0 || ndc_x > 1.0 || ndc_y < -1.0 || ndc_y > 1.0 {
            return;
          }

          let px = (ndc_x as f32 + 1.0) * 0.5 * w;
          let py = (ndc_y as f32 + 1.0) * 0.5 * h;

          let cam_dist_km =
            (rte_km.x() * rte_km.x() + rte_km.y() * rte_km.y() + rte_km.z() * rte_km.z()).sqrt();

          let desired_px = match camera_data.projection_params {
            CameraProjectionParams::Perspective { .. } => {
              if cam_dist_km > 1e-6 {
                (ind.desired_label_distance_km / cam_dist_km * proj_scale as f64).clamp(20.0, 300.0)
                  as f32
              } else {
                80.0
              }
            }
            CameraProjectionParams::Orthographic { .. } => {
              ((ind.desired_label_distance_km / AU_TO_KM) * proj_scale as f64).clamp(20.0, 300.0)
                as f32
            }
          };

          indicator_inputs.push(indicator_layout::IndicatorInput {
            screen_pos: [px, py],
            cam_dist_km,
            desired_px_dist: desired_px,
            label: ind.label.clone(),
            text_color: ind.text_color,
          });

          atlas_candidates.push((ind.font_atlas.clone(), ind.font_hash));
          let _ = device.allocate_rasterized_font_atlas(
            cmd_buffer,
            ind.font_hash,
            ind.font_atlas.clone(),
          );
        },
      );

      // 8b. ReferentialIndicatorComponent
      self.query1_without::<ReferentialIndicatorComponent, HiddenComponent, _>(|id, ref_ind| {
        if hidden_set.contains(&id) {
          return;
        }

        // Project the target's global (AU) position in the macro layer whatever frame it lives in:
        // a comet body sits in a micro frame (layer 1), and its label is still drawn on the macro
        // overlay. f64 camera-relative, as for every RTE path.
        let Some(mut target_global) = self.global_transform_f64(ref_ind.target_entity) else {
          return;
        };
        let Some(cam_macro) = camera_in_frames.get(&0) else {
          return;
        };
        // the comet's label sits on the nucleus the telescope sees (light-time displaced)
        if let Some(o) = light_time_overlay.label_offset(ref_ind.target_entity) {
          target_global.position = target_global.position + o;
        }
        let rte = crate::scene::HighResTransformComponent {
          position: target_global.position - cam_macro.position,
          rotation: target_global.rotation,
          scale: target_global.scale,
        };

        let clip = view_proj_f64.mul_vector(Vec4f64::from_components(
          rte.position.x(),
          rte.position.y(),
          rte.position.z(),
          1.0,
        ));

        if clip.w() <= 0.0 {
          return;
        }
        let ndc_x = (clip.x() / clip.w()) as f32;
        let ndc_y = (clip.y() / clip.w()) as f32;

        // Track NDC for trajectory avoidance, even if slightly offscreen
        parent_ndc_map.insert(ref_ind.target_entity, [ndc_x, ndc_y]);

        if ndc_x < -1.0 || ndc_x > 1.0 || ndc_y < -1.0 || ndc_y > 1.0 {
          return;
        }

        let px = (ndc_x + 1.0) * 0.5 * w;
        let py = (ndc_y + 1.0) * 0.5 * h;

        let cam_dist_km = rte.position.length() as f64 * AU_TO_KM;
        let desired_px = match camera_data.projection_params {
          CameraProjectionParams::Perspective { .. } => {
            if cam_dist_km > 1e-6 {
              (ref_ind.desired_label_distance_km / cam_dist_km * proj_scale as f64)
                .clamp(20.0, 300.0) as f32
            } else {
              80.0
            }
          }
          CameraProjectionParams::Orthographic { .. } => {
            ((ref_ind.desired_label_distance_km / AU_TO_KM) * proj_scale as f64).clamp(20.0, 300.0)
              as f32
          }
        };

        indicator_inputs.push(indicator_layout::IndicatorInput {
          screen_pos: [px, py],
          cam_dist_km,
          desired_px_dist: desired_px,
          label: ref_ind.label.clone(),
          text_color: ref_ind.text_color,
        });

        atlas_candidates.push((ref_ind.font_atlas.clone(), ref_ind.font_hash));
        let _ = device.allocate_rasterized_font_atlas(
          cmd_buffer,
          ref_ind.font_hash,
          ref_ind.font_atlas.clone(),
        );
      });

      // 8c. TrajectoryIndicatorComponent
      let dt_s = unscaled_time_delta_us as f32 / 1_000_000.0;
      self.query1_without::<TrajectoryIndicatorComponent, HiddenComponent, _>(|id, traj_ind| {
        if hidden_set.contains(&id) {
          return;
        }

        let traj =
          match self.with_component(traj_ind.target_entity, |c: &TrajectoryComponent| c.clone()) {
            Some(t) => t,
            None => return,
          };

        let (layer_idx, rte) = match compute_rte(self, traj_ind.target_entity) {
          Some(v) => v,
          None => return,
        };
        if layer_idx != 0 {
          return; // Macro only
        }

        // Build translation matrix manually using from_cols
        let model_f64 = Mat4x4f64::from_cols(
          Vec4f64::from_components(1.0, 0.0, 0.0, 0.0),
          Vec4f64::from_components(0.0, 1.0, 0.0, 0.0),
          Vec4f64::from_components(0.0, 0.0, 1.0, 0.0),
          Vec4f64::from_components(rte.position.x(), rte.position.y(), rte.position.z(), 1.0),
        );
        let mvp_f64 = view_proj_f64 * model_f64;

        let samples =
          trajectory_indicator::sample_and_project_trajectory(&traj.control_points, &mvp_f64, 16);
        if samples.is_empty() {
          return;
        }

        let parent_ndc = parent_ndc_map.get(&traj_ind.target_entity).copied();
        let best_idx = match trajectory_indicator::find_best_trajectory_anchor(&samples, parent_ndc)
        {
          Some(i) => i,
          None => return,
        };

        let ideal_t = samples[best_idx].global_t;
        let smoothed_t =
          trajectory_indicator::smooth_trajectory_t(traj_ind.get_current_t(), ideal_t, dt_s, 5.0);
        traj_ind.set_current_t(smoothed_t);

        let exact_pos =
          match trajectory_indicator::evaluate_bezier_at(&traj.control_points, smoothed_t) {
            Some(p) => p,
            None => return,
          };

        let clip = mvp_f64.mul_vector(Vec4f64::from_components(
          exact_pos[0],
          exact_pos[1],
          exact_pos[2],
          1.0,
        ));
        if clip.w() <= 0.0 {
          return;
        }
        let ndc_x = (clip.x() / clip.w()) as f32;
        let ndc_y = (clip.y() / clip.w()) as f32;
        if ndc_x < -1.0 || ndc_x > 1.0 || ndc_y < -1.0 || ndc_y > 1.0 {
          return;
        }

        let px = (ndc_x + 1.0) * 0.5 * w;
        let py = (ndc_y + 1.0) * 0.5 * h;
        let cam_dist_km = rte.position.length() as f64 * AU_TO_KM;

        indicator_inputs.push(indicator_layout::IndicatorInput {
          screen_pos: [px, py],
          cam_dist_km,
          desired_px_dist: 80.0,
          label: traj_ind.label.clone(),
          text_color: traj_ind.text_color,
        });

        atlas_candidates.push((traj_ind.font_atlas.clone(), traj_ind.font_hash));
        let _ = device.allocate_rasterized_font_atlas(
          cmd_buffer,
          traj_ind.font_hash,
          traj_ind.font_atlas.clone(),
        );
      });

      if !indicator_inputs.is_empty() {
        let outputs = indicator_layout::layout_indicators(&indicator_inputs, w, h);

        let mut atlas_info: Option<(alloc::sync::Arc<crate::scene::text::FontAtlas>, u32)> = None;
        for (atlas, hash) in atlas_candidates {
          if let Ok(desc_idx) =
            device.allocate_rasterized_font_atlas(cmd_buffer, hash, atlas.clone())
          {
            atlas_info = Some((atlas, desc_idx));
            break;
          }
        }

        if let Some((font_atlas, descriptor_index)) = atlas_info {
          for output in &outputs {
            gpu_ui.push(segment_to_ui_quad(
              output.seg1_start,
              output.seg1_end,
              indicator_layout::LINE_THICKNESS_PX,
              output.text_color,
            ));
            gpu_ui.push(segment_to_ui_quad(
              output.seg2_start,
              output.seg2_end,
              indicator_layout::LINE_THICKNESS_PX,
              output.text_color,
            ));

            let style = text::TextStyle {
              size_pt: output.text_pts,
              color: output.text_color,
              style_flags: 2, // Enable software bold for better readability
            };
            text::push_text_to_batch(
              &output.label,
              output.text_pos,
              &style,
              &font_atlas,
              descriptor_index,
              &mut text_batch,
            );
          }
        }
      }
    }

    // ------ 8c. Screen-space measurements (reference-position error) ----------------------
    // Endpoints are projected from f64 camera-relative positions; line, ticks and label are
    // overlay UI, so they keep a constant pixel size at any zoom and unit.
    {
      use aethervk_oshal_rlib::math::{
        matrix::{Matrix, mat4f64::Mat4x4f64},
        vector::vec4f64::Vec4f64,
      };
      const TICK_PX: f32 = 6.0;
      const LINE_PX: f32 = 1.5;
      const LABEL_PT: f32 = 13.0;
      let (w, h) = (window_extent[0] as f32, window_extent[1] as f32);
      let vp: Mat4x4f64 = camera_data.proj_f64 * camera_data.view_f64;
      let cam = cam_global_f64.position;
      let to_px = |p: aethervk_oshal_rlib::math::vector::vec3f64::DVec3| -> Option<[f32; 2]> {
        let r = p - cam;
        let c = vp.mul_vector(Vec4f64::from_components(r.x(), r.y(), r.z(), 1.0));
        (c.w() > 0.0).then(|| {
          [
            ((c.x() / c.w()) as f32 + 1.0) * 0.5 * w,
            ((c.y() / c.w()) as f32 + 1.0) * 0.5 * h,
          ]
        })
      };
      self
        .query1_without::<crate::scene::trajectory::ScreenMeasurementComponent, HiddenComponent, _>(
          |id, m| {
            if hidden_set.contains(&id) {
              return;
            }
            let (Some(a), Some(b)) = (to_px(m.from_au), to_px(m.to_au)) else {
              return;
            };
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let len = (dx * dx + dy * dy).sqrt().max(1e-3);
            let (nx, ny) = (-dy / len * TICK_PX, dx / len * TICK_PX); // perpendicular, pixels
            gpu_ui.push(segment_to_ui_quad(a, b, LINE_PX, m.color));
            gpu_ui.push(segment_to_ui_quad(
              [a[0] - nx, a[1] - ny],
              [a[0] + nx, a[1] + ny],
              LINE_PX,
              m.color,
            ));
            gpu_ui.push(segment_to_ui_quad(
              [b[0] - nx, b[1] - ny],
              [b[0] + nx, b[1] + ny],
              LINE_PX,
              m.color,
            ));
            if let Ok(desc) =
              device.allocate_rasterized_font_atlas(cmd_buffer, m.font_hash, m.font_atlas.clone())
            {
              let tw = text::text_width_px(&m.label, LABEL_PT, &m.font_atlas);
              let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
              let style = text::TextStyle {
                size_pt: LABEL_PT,
                color: m.color,
                style_flags: 2,
              };
              // centred, just above the line (screen y grows downwards)
              text::push_text_to_batch(
                &m.label,
                [mid[0] - tw * 0.5, mid[1] - TICK_PX - 4.0],
                &style,
                &m.font_atlas,
                desc,
                &mut text_batch,
              );
            }
          },
        );

      // Jet angles (`JetDirectionLineComponent::show_angles`): longitude arc on the equator from
      // the prime meridian, latitude arc on the jet's meridian from the equator, each with its
      // degree label; the font is the body labels' (any `ReferentialIndicatorComponent`).
      {
        const ARC_PX: f32 = 1.5;
        const ARC_LABEL_PT: f32 = 13.0;
        let mut font: Option<(alloc::sync::Arc<text::FontAtlas>, u64)> = None;
        self.query1(|_, r: &crate::scene::ReferentialIndicatorComponent| {
          if font.is_none() {
            font = Some((r.font_atlas.clone(), r.font_hash));
          }
        });
        self.query1_without::<crate::scene::JetDirectionLineComponent, HiddenComponent, _>(
          |id, line| {
            if !line.is_visible || !line.show_angles || hidden_set.contains(&id) {
              return;
            }
            let Some(nucleus) = self.get_parent(id) else {
              return;
            };
            let (Some(g), Some(gizmo_radius), Some(site)) = (
              self.global_transform_f64(nucleus),
              self.with_component(nucleus, |g: &SphereGizmoComponent| g.radius),
              self.with_component(id, |t: &crate::scene::TransformComponent| t.position),
            ) else {
              return;
            };
            let Some(angles) = jet_angle_arcs(vec3_f64(site), gizmo_radius as f64) else {
              return;
            };
            use aethervk_oshal_rlib::math::quaternion::Quaternion;
            let to_global = |p: [f64; 3]| {
              g.position
                + g.rotation.rotate_vector(aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
                  p[0] / AU_TO_KM as f64,
                  p[1] / AU_TO_KM as f64,
                  p[2] / AU_TO_KM as f64,
                ))
            };
            let font_desc = font.as_ref().and_then(|(atlas, hash)| {
              device
                .allocate_rasterized_font_atlas(cmd_buffer, *hash, atlas.clone())
                .ok()
                .map(|d| (atlas, d))
            });
            for arc in &angles.arcs {
              let px: alloc::vec::Vec<Option<[f32; 2]>> =
                arc.points.iter().map(|p| to_px(to_global(*p))).collect();
              for w in px.windows(2) {
                if let [Some(a), Some(b)] = w {
                  gpu_ui.push(segment_to_ui_quad(*a, *b, ARC_PX, JET_ANGLE_COLOR));
                }
              }
              let (Some((atlas, desc)), Some(anchor)) = (font_desc, to_px(to_global(arc.label_at)))
              else {
                continue;
              };
              let style = text::TextStyle {
                size_pt: ARC_LABEL_PT,
                color: JET_ANGLE_COLOR,
                style_flags: 2,
              };
              text::push_text_to_batch(
                &arc.label,
                [anchor[0] + 6.0, anchor[1] - 4.0],
                &style,
                atlas,
                desc,
                &mut text_batch,
              );
            }
          },
        );
      }

      // Pole labels ("N" / "S" at the ends of a dashed polar axis, `earth_visual`).
      const POLE_LABEL_PT: f32 = 16.0;
      const POLE_LABEL_GAP_PX: f32 = 4.0;
      self.query1_without::<crate::simulation_api::earth_visual::PoleLabelsComponent, HiddenComponent, _>(
        |id, labels| {
          if hidden_set.contains(&id) {
            return;
          }
          let Some([centre, north, south]) = crate::simulation_api::earth_visual::pole_points_global(
            self,
            id,
            labels.radius_km,
            labels.length_factor,
          ) else {
            return;
          };
          let Some(c) = to_px(centre) else {
            return;
          };
          let Ok(desc) = device.allocate_rasterized_font_atlas(
            cmd_buffer,
            labels.font_hash,
            labels.font_atlas.clone(),
          ) else {
            return;
          };
          let style = text::TextStyle {
            size_pt: POLE_LABEL_PT,
            color: labels.text_color,
            style_flags: 2,
          };
          for (end, label) in [(north, "N"), (south, "S")] {
            let Some(e) = to_px(end) else {
              continue;
            };
            let tw = text::text_width_px(label, POLE_LABEL_PT, &labels.font_atlas);
            // capitals span the ascent above the baseline, which is where `push_text_to_batch`
            // puts its start position
            let ascent = labels.font_atlas.ascent * POLE_LABEL_PT / labels.font_atlas.scale.y;
            let top_left = crate::simulation_api::earth_visual::label_top_left(
              c,
              e,
              tw,
              ascent,
              POLE_LABEL_GAP_PX,
            );
            text::push_text_to_batch(
              label,
              [top_left[0], top_left[1] + ascent],
              &style,
              &labels.font_atlas,
              desc,
              &mut text_batch,
            );
          }
        },
      );
    }

    if !gpu_ui.is_empty() {
      render_scene.ui_call = device.upload_ui(cmd_buffer, &gpu_ui).ok().flatten();
    }
    if !text_batch.is_empty() {
      render_scene.text2_call = device.upload_text2(cmd_buffer, &text_batch).ok().flatten();
    }

    // ------ 7. Batch Uploads & Finalization -----------------------------------------------
    let mut depth_layers: alloc::vec::Vec<RenderLayer> = layer_map
      .into_values()
      .map(|mut l| {
        if let Some(sg_list) = sg_batch_buffers.remove(&l.layer_index) {
          // same per-layer camera `draw_layer_content` draws with (near/far are final here)
          let layer_camera =
            render_scene.camera_data.rebuild_for_layer(l.near, l.far, l.frame_scale);
          let sg_data: alloc::vec::Vec<_> = sg_list
            .into_iter()
            .filter_map(|(id, m, center, r, sub, style, dash_count, color)| {
              device.allocate_sphere_gizmo_instance(id).ok().map(|idx| {
                (
                  idx,
                  gpu::SphereGizmoDataGpu {
                    model: m.into(),
                    radius: r,
                    subdivisions: sub,
                    style,
                    dash_count,
                    center_clip: crate::gpu::frame::center_clip_f64(&layer_camera, center),
                    color,
                  },
                )
              })
            })
            .collect();
          l.sphere_gizmo_batch_call =
            device.upload_sphere_gizmos_batch(cmd_buffer, &sg_data).ok().flatten();
        }
        if let Some((traj_list, anchors)) = traj_batch_buffers.remove(&l.layer_index) {
          let gpu_anchors: alloc::vec::Vec<gpu::TrajectoryAnchorGpu> =
            if anchors.iter().any(|a| a.is_some()) {
              // the camera draw_layer_content draws this layer's trajectories with
              let layer_camera =
                render_scene.camera_data.rebuild_for_layer(l.near, l.far, l.frame_scale);
              anchors
                .iter()
                .map(|a| match a {
                  Some(a) => gpu::TrajectoryAnchorGpu {
                    clip: crate::gpu::frame::center_clip_f64(&layer_camera, a.origin_rte),
                    dist: a.dist,
                    inv_len: a.inv_len,
                    layer: a.layer,
                  },
                  None => gpu::TrajectoryAnchorGpu::NONE,
                })
                .collect()
            } else {
              alloc::vec::Vec::new()
            };
          l.trajectory_call = device
            .upload_trajectories_with_anchors(cmd_buffer, &traj_list, &gpu_anchors)
            .ok()
            .flatten();
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
          "\x1b[36m[MULTI-SCALE] Frame {} | pos=({:.4},{:.4},{:.4}) yaw={:.1}\u{00b0} pitch={:.1}\u{00b0}\x1b[0m",
          frame,
          camera_data.absolute_pos.x(),
          camera_data.absolute_pos.y(),
          camera_data.absolute_pos.z(),
          camera_yaw_deg(&camera_data.rot),
          camera_pitch_deg(&camera_data.rot),
        );
      }
    }

    Ok(render_scene)
  }
}

/// Yaw = azimuth of the camera's forward direction in the XY plane,
/// measured CCW from +X (degrees). Engine convention: forward = rotate(0, -1, 0).
#[cfg(debug_assertions)]
fn camera_yaw_deg(rot: &aethervk_oshal_rlib::math::vector::vec4::Quat) -> f32 {
  use aethervk_oshal_rlib::math::quaternion::Quaternion as _;
  let fwd = rot.rotate_vector(Vec3f32::from_components(0.0, -1.0, 0.0));
  fwd.y().atan2(fwd.x()).to_degrees()
}

/// Pitch = elevation of the camera's forward direction above the XY plane (degrees).
/// Positive = looking upward (+Z). Engine convention: forward = rotate(0, -1, 0).
#[cfg(debug_assertions)]
fn camera_pitch_deg(rot: &aethervk_oshal_rlib::math::vector::vec4::Quat) -> f32 {
  use aethervk_oshal_rlib::math::quaternion::Quaternion as _;
  let fwd = rot.rotate_vector(Vec3f32::from_components(0.0, -1.0, 0.0));
  // Clamp to [-1, 1] before asin: slerp/retarget cycles can accumulate tiny FP rounding
  // errors that push fwd.z() fractionally outside the unit sphere. On Linux glibc raises
  // SIGFPE for out-of-domain asinf; clamping is the mathematically correct fix since
  // fwd is by definition a unit vector with z ∈ [-1, 1].
  fwd.z().clamp(-1.0, 1.0).asin().to_degrees()
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

/// Convert a 2D line segment into a rotated `UiElementGpu` rectangle.
///
/// The rectangle's centre is at the segment midpoint, its width equals the segment
/// length, and its height equals `thickness_px`. The `rotation` field of
/// `UiElementGpu` is set to the angle of the segment direction (radians, using
/// `atan2(dy, dx)` where +x is right and +y is down in pixel space).
///
/// `bounds` carries `[cx - len/2, cy - thickness/2, len, thickness]` — the UI shader
/// rotates around the rectangle's centre before rasterising.
fn segment_to_ui_quad(
  p0: [f32; 2],
  p1: [f32; 2],
  thickness_px: f32,
  color: [f32; 4],
) -> gpu::UiElementGpu {
  let dx = p1[0] - p0[0];
  let dy = p1[1] - p0[1];
  let length = (dx * dx + dy * dy).sqrt().max(1.0);
  let angle = dy.atan2(dx); // radians in pixel-space (+y down); CCW from +x in math convention
  let cx = (p0[0] + p1[0]) * 0.5;
  let cy = (p0[1] + p1[1]) * 0.5;

  gpu::UiElementGpu {
    bounds: [
      cx - length * 0.5,
      cy - thickness_px * 0.5,
      length,
      thickness_px,
    ],
    clip_rect: [-9999.0, -9999.0, 9999.0, 9999.0],
    color_start: color,
    color_end: color,
    color_border: [0.0; 4],
    color_shadow: [0.0; 4],
    border_radius: [0.0; 4],
    shadow_params: [0.0; 4],
    gradient_dir: [1.0, 0.0],
    border_width: 0.0,
    texture_id: 0xFFFF_FFFF,
    flags: 0,
    opacity: color[3],
    rotation: angle,
    _pad: 0,
  }
}

/// Transform of an object relative to the camera, in the layer units (AU for the macro layer, km
/// for micro layers, matching each layer's viewProj).
///
/// The object scale is already in those units: it is never divided by the camera scale. A camera
/// parented inside a micro frame (CometOrbiting, EarthPosition) carries that frame's 1/AU_TO_KM in
/// its global scale, and dividing by it blew every macro object (trajectories, cursor, ...) up by
/// ~1.5e8 (no_trajectories.rdc: trajectory NDC ~1e15).
impl Scene {
  /// Dust draw calls of the frame, `(depth layer, one call per age tier)` per visible particle
  /// system. Ranges and evaluation parameters come from the host state the logic thread maintains;
  /// positions are evaluated by `frame::prepare_dust` before the pass.
  ///
  /// Particles are NOT children of the comet: the jet entity (child of the comet body) only places
  /// the emission. The draw translation is `anchor − eye` from heliocentric values (the anchor is
  /// a fixed point, `DustDrawState::anchor_m`), evaluated at the time of the committed transforms
  /// the camera comes from ([`Scene::sim_time_s`]). Neither the comet nor the jet transforms enter:
  /// moving them never moves an emitted particle. The entity only picks the depth layer
  /// (occlusion against the nucleus) and the stream color.
  pub fn dust_draw_calls(
    &self,
    camera_entity: EntityId,
  ) -> alloc::vec::Vec<(u32, alloc::vec::Vec<DustDrawCall>)> {
    let Some(cam) = self.global_transform_f64(camera_entity) else {
      return alloc::vec::Vec::new();
    };
    let au = crate::scene::dust::AU_M;
    let eye_m = [
      cam.position.x() * au,
      cam.position.y() * au,
      cam.position.z() * au,
    ];
    // Earth observer light time: the dust is evaluated at the retarded epoch `t − delay` (one
    // epoch for the whole cloud, the nucleus': the spread of Δ across a tail is ≪ 1 s), drawn
    // from the true eye. Clusters emitted after `t − delay` get a negative age and are culled:
    // their light has not arrived yet. The simulation clock is untouched.
    let delay_s = self
      .with_component(
        camera_entity,
        |l: &crate::simulation_api::earth_observer::LightTimeViewComponent| l.delay_s,
      )
      .unwrap_or(0.0);
    let scene_time_s = self.sim_time_s().map(|t| t - delay_s);
    // the telescope's nucleus: displaced by the light-time offset like the comet's micro layer
    let lt_offset_m = self
      .with_component(
        camera_entity,
        |l: &crate::simulation_api::earth_observer::LightTimeViewComponent| l.offset_au,
      )
      .map_or([0.0; 3], |o| [o[0] * au, o[1] * au, o[2] * au]);
    // no nested scene lookups inside the query callback (see the deadlock note above)
    let mut systems = alloc::vec::Vec::new();
    self.query1_without::<ParticleSystemComponent, HiddenComponent, _>(|id, ps| {
      let states = ps.dust.lock().draw_states();
      if !states.is_empty() {
        systems.push((id, states, ps.draw_params.stream_color));
      }
    });
    systems
      .into_iter()
      .map(|(id, states, stream_color)| {
        let layer_idx = self.ancestor_depth_layer(id);
        // diagnostics only (dust trace): the nucleus never enters the draw
        let nucleus_m = self.get_parent(id).and_then(|p| self.global_transform_f64(p)).map(|t| {
          [
            t.position.x() * au,
            t.position.y() * au,
            t.position.z() * au,
          ]
        });
        let units_per_m = if layer_idx == 0 {
          1.0 / au
        } else {
          1.0e-3 // micro layers are in km
        };
        let nucleus_rte = nucleus_m.map(|n| {
          dust_rte_position(
            [n[0] + lt_offset_m[0], n[1] + lt_offset_m[1], n[2] + lt_offset_m[2]],
            eye_m,
            units_per_m,
          )
        });
        let calls = states
          .into_iter()
          .map(|state| {
            let state = match scene_time_s {
              Some(t) => state.at_time(t),
              None => state,
            };
            DustDrawCall {
              entity_id: id,
              rte_position: dust_rte_position(state.anchor_m, eye_m, units_per_m),
              units_per_m,
              stream_color,
              state,
              eye_m,
              nucleus_m,
              nucleus_rte,
              // the render thread stamps the scene's setting (`RenderScene::set_dust_volume_detail`)
              volume_detail: crate::scene::dust::DUST_VOLUME_DETAIL_DEFAULT,
              haze_detail: crate::scene::dust::DUST_HAZE_DETAIL_DEFAULT,
            }
          })
          .collect();
        (layer_idx, calls)
      })
      .collect()
  }
}

/// Dust draw translation in layer units: `(anchor − eye) · units_per_m`, both heliocentric metres
/// (f64). The drawn particle is at `(r − anchor) + (anchor − eye) = r − eye` whatever the anchor.
pub fn dust_rte_position(anchor_m: [f64; 3], eye_m: [f64; 3], units_per_m: f64) -> [f64; 3] {
  [
    (anchor_m[0] - eye_m[0]) * units_per_m,
    (anchor_m[1] - eye_m[1]) * units_per_m,
    (anchor_m[2] - eye_m[2]) * units_per_m,
  ]
}

pub fn rte_relative_to_camera(
  object_position: aethervk_oshal_rlib::math::vector::vec3f64::DVec3,
  object_rotation: aethervk_oshal_rlib::math::vector::vec4f64::Quat64,
  object_scale: Vec3f32,
  camera_in_frame: &crate::scene::HighResTransformComponent,
) -> crate::scene::HighResTransformComponent {
  crate::scene::HighResTransformComponent {
    position: object_position - camera_in_frame.position,
    rotation: object_rotation,
    scale: object_scale,
  }
}

/// Emissive factor of a jet marker whose site faces away from the Sun.
pub(crate) const UNLIT_JET_MARKER_DIM: f32 = 0.25;

/// Whether the jet site `jet` (a child of the nucleus, translated to its surface) faces the Sun,
/// the heliocentric origin: `(site − nucleus) · (−nucleus) > 0`, the dust emission's lit test at
/// the current rotation. `None` without a parent or transforms.
/// Absolute minimum perspective near plane (1 m) to handle camera-inside-mesh.
const DEPTH_NEAR_FLOOR: f64 = 0.001; // km (= 1 m)
/// Perspective near plane when within 2 m of the surface (0.1 mm).
const DEPTH_NEAR_MIN: f64 = 1.0e-7; // km
/// 5% padding on both sides to prevent near/far edge clipping.
const DEPTH_MARGIN: f64 = 1.05;

/// The macro layer's (near, far) from the UI's camera planes. An orthographic camera that shows
/// what lies behind it (no [`crate::scene::CAMERA_HIDE_BEHIND`]) gets the mirrored slab
/// `(−far, far)`: ortho depth is linear, so a negative near is valid and costs only precision.
pub(crate) fn macro_layer_depth(near: f64, far: f64, is_ortho: bool, hide_behind: bool) -> (f64, f64) {
  if is_ortho && !hide_behind {
    (-far.abs(), far)
  } else {
    (near, far)
  }
}

/// The (near, far) slab enclosing a sphere of radius `r` centred at the signed view depth `depth`
/// (along the camera forward axis, view −Y). Orthographic projections have no w=0 singularity:
/// the near plane may be negative, which shows geometry behind the eye plane, unless
/// `hide_behind` clamps it at the eye. Perspective needs a strictly positive near plane.
pub(crate) fn fit_depth_slab(depth: f64, r: f64, is_ortho: bool, hide_behind: bool) -> (f64, f64) {
  if is_ortho {
    let (near, far) = (depth - r * DEPTH_MARGIN, depth + r * DEPTH_MARGIN);
    if hide_behind {
      // wholly behind: an empty slab (far <= near), which the commit step skips
      (near.max(0.0), far.max(0.0))
    } else {
      (near, far)
    }
  } else {
    let d = depth.abs();
    let surface_floor = if d > r {
      ((d - r) * 0.5).clamp(DEPTH_NEAR_MIN, DEPTH_NEAR_FLOOR)
    } else {
      DEPTH_NEAR_FLOOR
    };
    (
      (d - r * DEPTH_MARGIN).max(surface_floor),
      d + r * DEPTH_MARGIN,
    )
  }
}

fn vec3_f64(v: Vec3f32) -> [f64; 3] {
  use aethervk_oshal_rlib::math::vector::Vector3;
  [v.x() as f64, v.y() as f64, v.z() as f64]
}

/// Colour of the jet angle arcs and their labels.
pub(crate) const JET_ANGLE_COLOR: [f32; 4] = [1.0, 0.85, 0.35, 1.0];
/// Segments of a 90° jet angle arc (proportionally fewer for a smaller angle, at least 2).
pub(crate) const JET_ANGLE_ARC_SEGMENTS_PER_RIGHT_ANGLE: usize = 24;
/// Radius of the jet angle arcs over the gizmo radius: inside the jet line (`1.125 ×`).
pub(crate) const JET_ANGLE_ARC_OVER_GIZMO_RADIUS: f64 = 0.8;

/// A jet's site direction (nucleus frame) → the jet line's local frame for the gizmo pipeline:
/// orthonormal, local +Z along the site, X/Y any perpendicular pair. `None` for a zero site.
pub(crate) fn jet_line_frame(site: [f64; 3]) -> Option<Mat4x4f32> {
  use aethervk_oshal_rlib::math::vector::vec4::Vec4f32;
  let n = (site[0] * site[0] + site[1] * site[1] + site[2] * site[2]).sqrt();
  if !(n > 0.0) || !n.is_finite() {
    return None;
  }
  let z = [site[0] / n, site[1] / n, site[2] / n];
  // a helper axis not parallel to z
  let h = if z[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
  let cross = |a: [f64; 3], b: [f64; 3]| {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
  };
  let x = cross(h, z);
  let xn = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
  let x = [x[0] / xn, x[1] / xn, x[2] / xn];
  let y = cross(z, x);
  let col = |v: [f64; 3]| Vec4f32::from_components(v[0] as f32, v[1] as f32, v[2] as f32, 0.0);
  Some(Mat4x4f32 {
    x: col(x),
    y: col(y),
    z: col(z),
    w: Vec4f32::from_components(0.0, 0.0, 0.0, 1.0),
  })
}

/// One jet angle arc: points in the nucleus frame (km) and its label.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct JetAngleArc {
  pub points: alloc::vec::Vec<[f64; 3]>,
  /// Where the label starts (the arc's midpoint).
  pub label_at: [f64; 3],
  pub label: alloc::string::String,
}

/// The two arcs of a jet's angles, nucleus frame (km), at `0.8 ×` the gizmo radius:
/// `arcs[0]` longitude, on the equator from the prime meridian (+X) to the jet's meridian;
/// `arcs[1]` latitude, on that meridian from the equator to the jet's direction.
/// Labels "lon 45.0°" / "lat -30.0°" (east longitude, the C# convention). `None` for a zero site.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct JetAngleArcs {
  pub lat_rad: f64,
  pub lon_rad: f64,
  pub arcs: [JetAngleArc; 2],
}

pub(crate) fn jet_angle_arcs(site: [f64; 3], gizmo_radius_km: f64) -> Option<JetAngleArcs> {
  let n = (site[0] * site[0] + site[1] * site[1] + site[2] * site[2]).sqrt();
  if !(n > 0.0) || !n.is_finite() {
    return None;
  }
  let lat = (site[2] / n).clamp(-1.0, 1.0).asin();
  let lon = site[1].atan2(site[0]);
  let r = gizmo_radius_km * JET_ANGLE_ARC_OVER_GIZMO_RADIUS;
  let segments = |angle: f64| {
    ((angle.abs() / core::f64::consts::FRAC_PI_2 * JET_ANGLE_ARC_SEGMENTS_PER_RIGHT_ANGLE as f64)
      .ceil() as usize)
      .max(2)
  };
  // a point at latitude `phi`, longitude `lam`, radius r
  let at = |phi: f64, lam: f64| [r * phi.cos() * lam.cos(), r * phi.cos() * lam.sin(), r * phi.sin()];
  let sweep = |count: usize, f: &dyn Fn(f64) -> [f64; 3]| -> alloc::vec::Vec<[f64; 3]> {
    (0..=count).map(|i| f(i as f64 / count as f64)).collect()
  };
  let lon_n = segments(lon);
  let lat_n = segments(lat);
  let lon_arc = JetAngleArc {
    points: sweep(lon_n, &|t| at(0.0, lon * t)),
    label_at: at(0.0, lon * 0.5),
    label: alloc::format!("lon {:.1}°", lon.to_degrees()),
  };
  let lat_arc = JetAngleArc {
    points: sweep(lat_n, &|t| at(lat * t, lon)),
    label_at: at(lat * 0.5, lon),
    label: alloc::format!("lat {:.1}°", lat.to_degrees()),
  };
  Some(JetAngleArcs {
    lat_rad: lat,
    lon_rad: lon,
    arcs: [lon_arc, lat_arc],
  })
}

/// The sphere-gizmo style's `color` slot (`sphere_gizmo.vert`): rgb the comet's velocity direction
/// in the axes of the depth-layer frame `frame` (the gizmo's model matrix is relative to it; world
/// axes without one), a the `SPHERE_GIZMO_PART_*` mask. Without a [`SphereGizmoPartsComponent`]:
/// the default parts and no velocity.
pub(crate) fn sphere_gizmo_parts_color(
  scene: &Scene,
  gizmo: EntityId,
  frame: Option<EntityId>,
) -> [f32; 4] {
  use crate::simulation_api::earth_observer::{qinverse, qnormalize, rotate};
  let Some(p) = scene.with_component(gizmo, |p: &crate::scene::SphereGizmoPartsComponent| *p) else {
    return [0.0, 0.0, 0.0, crate::scene::SPHERE_GIZMO_PARTS_DEFAULT as f32];
  };
  let v = p.velocity_dir.map(f64::from);
  let v = match frame.and_then(|f| scene.global_transform_f64(f)) {
    Some(f) => rotate(
      qinverse(qnormalize([f.rotation[0], f.rotation[1], f.rotation[2], f.rotation[3]])),
      v,
    ),
    None => v,
  };
  [v[0] as f32, v[1] as f32, v[2] as f32, p.parts as f32]
}

pub(crate) fn jet_site_lit(scene: &Scene, jet: EntityId) -> Option<bool> {
  use aethervk_oshal_rlib::math::vector::Vector3;
  let nucleus = scene.global_transform_f64(scene.get_parent(jet)?)?.position;
  let site = scene.global_transform_f64(jet)?.position;
  let normal = site - nucleus;
  let to_sun = -nucleus;
  Some(normal.x() * to_sun.x() + normal.y() * to_sun.y() + normal.z() * to_sun.z() > 0.0)
}

/// Where the Earth observer camera draws its comet on a `extent` (physical px) viewport.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CometScreenPosition {
  /// From the left edge, px.
  pub x_px: f64,
  /// From the top edge, px.
  pub y_px: f64,
  /// Inside the viewport and drawable (in front of a perspective eye).
  pub on_screen: bool,
}

/// Projects the comet the Earth observer `camera` tracks (its `LightTimeViewComponent`) onto a
/// viewport of `extent` physical px, light-time displaced as its label and micro layer are drawn:
/// the same f64 camera-relative projection as the comet's indicator. `None` without an observer
/// comet, transforms or camera, or behind a perspective eye.
pub(crate) fn comet_screen_position(
  scene: &Scene,
  camera: EntityId,
  extent: [u32; 2],
) -> Option<CometScreenPosition> {
  use aethervk_oshal_rlib::math::{
    matrix::{Matrix, mat4f64::Mat4x4f64},
    vector::{Vector4, vec3f64::Vec3f64, vec4f64::Vec4f64},
  };
  if extent[0] == 0 || extent[1] == 0 {
    return None;
  }
  let lt = scene
    .with_component(camera, |l: &crate::simulation_api::earth_observer::LightTimeViewComponent| *l)?;
  let cam_global = scene.global_transform_f64(camera)?;
  let cam_comp = scene.with_component(camera, |c: &CameraComponent| *c)?;
  let target = scene.global_transform_f64(lt.comet)?;
  let camera_data = CameraRenderData::new_f64(&cam_global, &cam_comp, 1.0, extent);
  let offset = Vec3f64::from_components(lt.offset_au[0], lt.offset_au[1], lt.offset_au[2]);
  let rte = target.position + offset - cam_global.position;
  let view_proj: Mat4x4f64 = camera_data.proj_f64 * camera_data.view_f64;
  let clip = view_proj.mul_vector(Vec4f64::from_components(
    aethervk_oshal_rlib::math::vector::Vector3::x(&rte),
    aethervk_oshal_rlib::math::vector::Vector3::y(&rte),
    aethervk_oshal_rlib::math::vector::Vector3::z(&rte),
    1.0,
  ));
  if clip.w() <= 0.0 {
    return None;
  }
  let (ndc_x, ndc_y) = (clip.x() / clip.w(), clip.y() / clip.w());
  Some(CometScreenPosition {
    x_px: (ndc_x + 1.0) * 0.5 * extent[0] as f64,
    y_px: (ndc_y + 1.0) * 0.5 * extent[1] as f64,
    on_screen: (-1.0..=1.0).contains(&ndc_x) && (-1.0..=1.0).contains(&ndc_y),
  })
}

/// The Earth observer's light-time displacement (`earth_observer::LightTimeViewComponent` on the
/// camera) for the comet's root-level overlays: the micro layer moves the comet itself, these
/// move its label (which targets the comet body) and its lines (marked with a
/// `LightTimeTrackComponent`) by the same `offset_au`, whatever layer the comet is on.
#[derive(Debug, Default)]
pub(crate) struct LightTimeOverlay {
  comet: Option<EntityId>,
  offset_au: aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64,
  tracks: alloc::vec::Vec<EntityId>,
}

impl LightTimeOverlay {
  pub(crate) fn of(scene: &Scene, camera: EntityId) -> Self {
    use crate::simulation_api::earth_observer::{LightTimeTrackComponent, LightTimeViewComponent};
    let Some(lt) = scene.with_component(camera, |l: &LightTimeViewComponent| *l) else {
      return Self::default();
    };
    let mut tracks = alloc::vec::Vec::new();
    scene.query1(|id, t: &LightTimeTrackComponent| {
      if t.comet == lt.comet {
        tracks.push(id);
      }
    });
    Self {
      comet: Some(lt.comet),
      offset_au: aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
        lt.offset_au[0],
        lt.offset_au[1],
        lt.offset_au[2],
      ),
      tracks,
    }
  }

  /// displacement (AU) of a label pointing at `target`
  pub(crate) fn label_offset(
    &self,
    target: EntityId,
  ) -> Option<aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64> {
    (self.comet == Some(target)).then_some(self.offset_au)
  }

  /// displacement (AU) of the root-level line `entity`
  pub(crate) fn track_offset(
    &self,
    entity: EntityId,
  ) -> Option<aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64> {
    self.tracks.contains(&entity).then_some(self.offset_au)
  }
}

#[cfg(test)]
mod tests;
