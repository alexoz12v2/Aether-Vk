//! scene_conversion module.

pub mod indicator_layout;
pub mod trajectory_indicator;

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
  os::{
    pool::ThreadPool,
    time::{timeus_t, us_to_300ths_rounded},
  },
};
use function_name::named;

const AU_TO_KM: f64 = 149_597_870.700_f64;

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

    let mut layer_bounds: hashbrown::HashMap<u32, (f64, f64)> =
      hashbrown::HashMap::with_capacity(64);
    let mut layer_frame_scales: hashbrown::HashMap<u32, f32> =
      hashbrown::HashMap::with_capacity(64);

    layer_bounds.insert(0, (macro_near as f64, macro_far as f64));
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
    // Phase 2 — `archetypes.read()` from `query1_without` is now released.
    // Safe to call `get_relative_transform_f64` which re-acquires `archetypes.read()`.
    for entry in frame_entries {
      layer_frame_entities.insert(entry.depth_layer, entry.id);
      if let Some(cam_in_frame_f64) = self.get_relative_transform_f64(camera_entity, entry.id) {
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

    let mut layer_map: hashbrown::HashMap<u32, RenderLayer> = hashbrown::HashMap::with_capacity(16);

    // Ensures layers are lazily construted into our final memory footprint format only when
    // actually used
    macro_rules! get_or_create_layer {
      ($layer_idx:expr) => {
        layer_map.entry($layer_idx).or_insert_with(|| {
          let (near, far) = layer_bounds
            .get(&$layer_idx)
            .copied()
            .unwrap_or((macro_near as f64, macro_far as f64));
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

        let diff = pos_f64 - cam_in_frame.position;

        // For macro layer (0): cam_in_frame.scale ≈ 1.0 (camera global AU scale).
        // Dividing obj_scale by ≈1.0 is harmless — result is in AU, matching AU viewProj. ✓
        //
        // For micro layers (>0): cam_in_frame.scale ≈ 1/frame_scale ≈ 1.49e8 (km per world unit).
        // Dividing obj_scale_km by 1.49e8 converts km→AU, but the micro-layer viewProj uses km
        // (tight near/far computed from dist_local in km). Use obj_scale directly so the result
        // is in km, matching the km viewProj. Without this, a 2 km mesh or 50 km sphere would be
        // scaled down to ~0.01 μm — sub-pixel at any viewing distance.
        let scale = if layer_idx == 0 {
          obj_scale / cam_in_frame.scale
        } else {
          obj_scale // micro: km scale, matches km viewProj — no frame-scale division
        };
        Some((
          layer_idx,
          crate::scene::HighResTransformComponent {
            position: diff,
            rotation: rot,
            scale,
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
          sg.radius,
          sg.subdivisions,
        )
      })
    });

    // ------ Depth fitting pass -------------------------------------------------------
    // Refits micro-layer near/far from actual draw-call geometry rather than the
    // conservative SOI sphere. Only meshes whose bounding sphere passes a frustum
    // cull test contribute to the tight bounds — objects outside the frustum (behind
    // the camera, off to the sides, etc.) must not expand the depth range.
    //
    // The SOI-sphere bounds (computed in Phase 2 above) remain as a fallback if no
    // StaticMeshComponent survives the cull test for a given micro layer.
    //
    // Scale is always 1 for comet meshes (radius is baked into vertex positions via
    // update_uv_sphere_radius_in_place), so bounding radius is read from vertex data.
    //
    // Constants:
    //   DEPTH_NEAR_FLOOR — absolute minimum near plane (1 m) to handle camera-inside-mesh
    //   DEPTH_MARGIN     — 5% padding on both sides to prevent near/far edge clipping
    {
      const DEPTH_NEAR_FLOOR: f64 = 0.001; // km (= 1 m)
      const DEPTH_MARGIN: f64 = 1.05;

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

      for (layer_idx, _id, mesh, rte, _outline) in &extracted_meshes {
        if *layer_idx == 0 {
          continue; // macro layer uses camera near/far from UI — do not touch
        }

        // Bounding radius from vertex positions (scale=1, radius baked into positions).
        let obj_radius = mesh
          .mesh
          .vertices
          .iter()
          .map(|v| {
            let [x, y, z] = v.position;
            ((x * x + y * y + z * z) as f64).sqrt()
          })
          .fold(0.0f64, f64::max);

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
        use aethervk_oshal_rlib::math::vector::Vector4;
        use aethervk_oshal_rlib::math::vector::vec4f64::Vec4f64;
        let view_pos = camera_data.view_f64 * Vec4f64::from_components(cx, cy, cz, 1.0);
        let obj_dist_y = view_pos.y().abs();

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

        let obj_near = (obj_dist_y - obj_radius * DEPTH_MARGIN).max(DEPTH_NEAR_FLOOR);
        let obj_far = obj_dist_y + obj_radius * DEPTH_MARGIN;

        let e = per_layer_depth.entry(*layer_idx).or_insert((f64::MAX, f64::NEG_INFINITY));
        e.0 = e.0.min(obj_near);
        e.1 = e.1.max(obj_far);
      }

      // Sphere gizmos: axes extend to `radius * 1.5` km beyond the sphere surface
      // (sphere_gizmo.vert line 121), with arrowheads adding another `radius * 0.2`
      // (line 151). The full gizmo bounding sphere is therefore `radius * 1.7` km.
      // -> scratch that, it seems that the yellow one is still cut. Trying out 2.1 km
      // extracted_sg tuple: (layer_idx, id, mat, rad, sub)
      for (layer_idx, _id, _mat, rad, _sub) in &extracted_sg {
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
        let obj_dist_y = {
          use aethervk_oshal_rlib::math::vector::Vector4;
          view_pos.y().abs()
        };

        // Bounding radius: full axis + arrowhead envelope.
        let gizmo_radius = (*rad as f64) * 2.1;

        let au_scale = 1.0 / AU_TO_KM as f64;
        if !sphere_in_frustum(
          cx * au_scale,
          cy * au_scale,
          cz * au_scale,
          gizmo_radius * au_scale,
        ) {
          continue;
        }

        let obj_near = (obj_dist_y - gizmo_radius * DEPTH_MARGIN).max(DEPTH_NEAR_FLOOR);
        let obj_far = obj_dist_y + gizmo_radius * DEPTH_MARGIN;

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
        aethervk_oshal_rlib::log!("StaticMesh {} pushed to layer {}", mesh.mesh.id, layer_idx);
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
          Mat4x4f32::from_quat_custom_frame(rte.rotation),
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
        aethervk_oshal_rlib::log!("GPU Upload Error creating/getting gizmo resources")
      }
    }

    // 6. Sphere Gizmos (Batched - deferred upload)
    let mut sg_batch_buffers = hashbrown::HashMap::<
      u32,
      alloc::vec::Vec<(EntityId, Mat4x4f32, f32, f32)>,
    >::with_capacity(16);
    for (layer_idx, id, mat, rad, sub) in extracted_sg {
      get_or_create_layer!(layer_idx);
      sg_batch_buffers.entry(layer_idx).or_default().push((id, mat, rad, sub));
    }

    // 7. Trajectories (Batched - deferred upload)
    let mut traj_batch_buffers = hashbrown::HashMap::<
      u32,
      alloc::vec::Vec<(EntityId, TrajectoryComponent, Mat4x4f32)>,
    >::with_capacity(16);
    let extracted_traj = extract!(TrajectoryComponent, |id, traj| {
      // Note: traj.clone() copies the array of control points
      compute_rte(self, id)
        .map(|(layer_idx, rte)| (layer_idx, id, traj.clone(), rte.to_transform().to_mat4()))
    });
    for (layer_idx, id, traj, mat) in extracted_traj {
      get_or_create_layer!(layer_idx);
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
            rte_mat_f64: rte.to_mat4_f64(),
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

      use aethervk_oshal_rlib::math::matrix::{Matrix, mat4f64::Mat4x4f64};
      use aethervk_oshal_rlib::math::vector::vec4f64::Vec4f64;
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

        let (layer_idx, rte) = match compute_rte(self, ref_ind.target_entity) {
          Some(v) => v,
          None => return,
        };

        if layer_idx != 0 {
          return; // Macro only
        }

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
          "\x1b[36m[MULTI-SCALE] Frame {} | pos=({:.4},{:.4},{:.4}) yaw={:.1}\u{00b0} pitch={:.1}\u{00b0}\x1b[0m",
          frame,
          camera_data.absolute_pos.x(),
          camera_data.absolute_pos.y(),
          camera_data.absolute_pos.z(),
          camera_yaw_deg(&camera_data.rot),
          camera_pitch_deg(&camera_data.rot),
        )
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

#[cfg(test)]
mod tests;