//! gpu_backends module.

use crate::{
  gpu::{
    CommandBuffer, DeviceBuffer, DeviceBvh, Kernels, RenderBackendId, RenderFrontend,
    VULKAN_RENDER_BACKEND, WaitHandle,
  },
  physics::physics_scene::PhysicsScene,
  scene::Scene,
  traits::InitWithRuntime,
  types::{EngineError, EngineResult, GpuError, RuntimeParams},
};
use aethervk_oshal_rlib::{
  math::vector::Vector,
  os::time::{timeus_milliseconds, timeus_t},
};
use alloc::vec::Vec;

#[cfg(all(
  not(target_arch = "wasm32"),
  any(
    target_os = "windows",
    target_os = "linux",
    target_os = "macos",
    target_os = "android",
    target_os = "ios"
  )
))]
pub mod vulkan;

// #[cfg(target_os = "macos")]
// pub(super) mod metal;

// #[cfg(target_os = "windows")]
// pub(super) mod d3d12;

/// TODO: Document this item
pub(self) const MAX_DEVICES: usize = 4;

/// TODO: Document this item
pub fn new_render_frontend(
  ty: RenderBackendId,
  params: &'_ RuntimeParams,
) -> EngineResult<RenderFrontend> {
  match ty {
    #[cfg(all(
      not(target_arch = "wasm32"),
      any(
        target_os = "windows",
        target_os = "linux",
        target_os = "macos",
        target_os = "android",
        target_os = "ios"
      )
    ))]
    VULKAN_RENDER_BACKEND => {
      vulkan::VulkanRenderContext::init_with_runtime(params).map(|back| back.into())
    }
    // #[cfg(target_os = "macos")]
    // METAL_RENDER_BACKEND => {
    //   metal::MetalRenderContext::init_with_runtime(params).map(|back| back.into())
    // }
    // #[cfg(target_os = "windows")]
    // D3D12_RENDER_BACKEND => {
    //   d3d12::D3d12RenderContext::init_with_runtime(params).map(|back| back.into())
    // }
    _ => Err(EngineError::Gpu(GpuError::UnsupportedFeature)),
  }
}

/// TODO: Document this item
pub fn get_available_render_backends() -> Vec<&'static str> {
  let mut backends = Vec::new();

  #[cfg(all(
    not(target_arch = "wasm32"),
    any(
      target_os = "windows",
      target_os = "linux",
      target_os = "macos",
      target_os = "android",
      target_os = "ios"
    )
  ))]
  {
    let params = RuntimeParams {
      render_backend_params: heapless::index_map::FnvIndexMap::new(),
      validation_error_callback: None,
    };
    if let Ok(mut context) = vulkan::VulkanRenderContext::init_with_runtime(&params) {
      use crate::gpu::RenderContext;
      if context.init_device(0, &crate::gpu::DeviceAdditionalParams::new()).is_ok() {
        backends.push("Vulkan");
      }
    }
  }

  #[cfg(target_os = "macos")]
  backends.push("Metal");

  #[cfg(target_os = "windows")]
  backends.push("Direct3D12");

  backends
}

/// TODO: Document this item
pub fn get_available_kernels() -> Vec<&'static str> {
  let mut kernels = alloc::vec!["CPU Scalar", "CPU SSE/AVX", "CPU NEON"];

  #[cfg(all(
    not(target_arch = "wasm32"),
    any(
      target_os = "windows",
      target_os = "linux",
      target_os = "macos",
      target_os = "android",
      target_os = "ios"
    )
  ))]
  {
    let params = RuntimeParams {
      render_backend_params: heapless::index_map::FnvIndexMap::new(),
      validation_error_callback: None,
    };
    if let Ok(mut context) = vulkan::VulkanRenderContext::init_with_runtime(&params) {
      use crate::gpu::RenderContext;
      if context.init_device(0, &crate::gpu::DeviceAdditionalParams::new()).is_ok() {
        kernels.push("Vulkan Compute");
      }
    }
  }

  kernels.push("CUDA");
  kernels
}

struct AutoDiscard<T, F: FnMut(T)> {
  value: Option<T>,
  discard_fn: Option<F>,
}
impl<T, F: FnMut(T)> AutoDiscard<T, F> {
  fn new(value: T, discard_fn: F) -> Self {
    Self {
      value: Some(value),
      discard_fn: Some(discard_fn),
    }
  }
}
impl<T, F: FnMut(T)> Drop for AutoDiscard<T, F> {
  fn drop(&mut self) {
    if let (Some(val), Some(mut f)) = (self.value.take(), self.discard_fn.take()) {
      f(val);
    }
  }
}
impl<T, F: FnMut(T)> core::ops::Deref for AutoDiscard<T, F> {
  type Target = T;
  fn deref(&self) -> &Self::Target {
    self.value.as_ref().unwrap()
  }
}
impl<T, F: FnMut(T)> core::ops::DerefMut for AutoDiscard<T, F> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    self.value.as_mut().unwrap()
  }
}

/// Fixed Update Function in interval $t_0, t_1$.
///
/// The total interval `[t0, t1]` may be large at high time-scales (e.g.
/// `OneMonth` produces ~42 000 s per tick).  To keep the IMEX / Velocity-Verlet
/// integrator numerically stable the interval is subdivided into sub-steps
/// whose duration never exceeds `max_sub_dt_us` microseconds.
///
/// **Sub-step loop** (non-collision path):
/// 1. Particle emission happens **once** at the beginning (for the full dt).
/// 2. The IMEX integration is iterated `n_sub_steps` times with `sub_dt_us`.
///
/// This function is asynchronous with respect to the logic_thread – it should
/// be dispatched inside the pool as a singleton tasklet.
///
/// - we expect the physical_scene to contain BVH of entities with state at $t_0$
pub fn simulation_step<K>(
  kernels: &K,
  physical_scene: &mut PhysicsScene,
  scene: &Scene,
  t0: timeus_t,
  t1: timeus_t,
  collisions_enabled: bool,
  max_sub_dt_us: timeus_t,
) -> EngineResult<Option<crate::gpu::CommandBufferSyncInfo>>
where
  K: Kernels + ?Sized,
{
  let total_dt = t1 - t0;
  // Compute sub-step count.  Clamp max_sub_dt_us to at least 1 μs.
  let capped = if max_sub_dt_us > 0 {
    max_sub_dt_us
  } else {
    total_dt.max(1)
  };
  let n_sub_steps = if total_dt <= capped {
    1usize
  } else {
    ((total_dt as f64 / capped as f64).ceil() as usize).max(1)
  };

  aethervk_oshal_rlib::log!(
    "simulation_step running! dt_us: {}, sub_steps: {}, collisions_enabled: {}",
    total_dt,
    n_sub_steps,
    collisions_enabled
  );
  let mut cmd = kernels.create_command_buffer()?;

  // ── shader_debug_sync: isolate each dispatch for GPU-hang debugging ────────
  // When the feature is active, every sync!("name") submits the cmd, waits
  // for GPU completion with a 5 s timeout, then allocates a fresh cmd.
  // The last log line before silence identifies the hanging shader.
  #[cfg(feature = "shader_debug_sync")]
  macro_rules! sync {
    ($name:expr) => {{
      aethervk_oshal_rlib::log!("[SHADER-SYNC] {} — submitting...", $name);
      // Safety: we read cmd out (leaving the slot logically uninit), pass it to
      // debug_sync_barrier which consumes it, then write the fresh cmd back.
      // If the barrier fails we panic — acceptable for a debug-only feature.
      let old_cmd = unsafe { core::ptr::read(&raw const cmd) };
      let new_cmd = match kernels.debug_sync_barrier(old_cmd) {
        Ok(c) => c,
        Err(e) => panic!(
          "[SHADER-SYNC] debug_sync_barrier failed for {}: {:?}",
          $name, e
        ),
      };
      unsafe { core::ptr::write(&raw mut cmd, new_cmd) };
      // Validate VMA debug margins — corruption check pinpoints the guilty shader.
      if let Err(e) = kernels.check_corruption($name) {
        panic!(
          "[SHADER-SYNC] VMA CORRUPTION detected after '{}': {:?}",
          $name, e
        );
      }
      aethervk_oshal_rlib::log!("[SHADER-SYNC] {} — completed ✓", $name);
    }};
  }
  #[cfg(not(feature = "shader_debug_sync"))]
  macro_rules! sync {
    ($name:expr) => {{}};
  }

  let mut current_time = t0;
  let end_time = t1;
  #[cfg(any(test, feature = "collisions"))]
  let mut old_cmds = alloc::vec::Vec::<K::Cmd>::new();
  #[cfg(any(test, feature = "collisions"))]
  let time_collision_delta = timeus_milliseconds(30);
  #[cfg(any(test, feature = "collisions"))]
  let mut collision_iters = 0;
  #[cfg(any(test, feature = "collisions"))]
  const MAX_BOUNCES: usize = 5;

  // ── 0. Build + upload per-tick motion TLAS (CPU-built, GPU-uploaded) ────────
  // The scene-root TLAS is a motion, multi-branch BVH (N = subgroup_size)
  // with leaves: body BLAS, particle BLAS (sentinel-patched for GPU LBVH), or
  // micro-frame sub-TLAS roots.  Its BDA replaces the recycled particle-LBVH
  // address previously used as tlas_bvh_addr in bp_scene / bp_particle_self.
  let dt_s = total_dt as f32 / 1_000_000.0;
  use crate::physics::tlas_builder::build_scene_motion_tlas;
  let (tlas_bytes, tlas_root_idx) = match kernels.subgroup_size().map(|s| s as u32).unwrap_or(32) {
    128 => build_scene_motion_tlas::<128>(physical_scene),
    64 => build_scene_motion_tlas::<64>(physical_scene),
    32 => build_scene_motion_tlas::<32>(physical_scene),
    16 => build_scene_motion_tlas::<16>(physical_scene),
    8 => build_scene_motion_tlas::<8>(physical_scene),
    4 => build_scene_motion_tlas::<4>(physical_scene),
    _ => build_scene_motion_tlas::<32>(physical_scene),
  };
  aethervk_oshal_rlib::log!(
    "tlas_bytes len: {}, tlas_root_idx: {}",
    tlas_bytes.len(),
    tlas_root_idx
  );
  aethervk_oshal_rlib::log!(
    "TLAS DEBUG: tlas_bytes len: {}, tlas_root_idx: {}",
    tlas_bytes.len(),
    tlas_root_idx
  );
  let motion_tlas = AutoDiscard::new(kernels.upload_motion_tlas(&mut cmd, &tlas_bytes)?, |b| {
    kernels.discard_tlas(b)
  });
  let tlas_addr = motion_tlas.address();

  for frame in &mut physical_scene.gpu_frames {
    frame.frame_bda = tlas_addr;
  }

  // ── 1. Build per-frame GPU buffers from ECS ───────────────────────────────
  let kinematics = AutoDiscard::new(
    kernels.build_kinematic_bodies(&mut cmd, physical_scene, scene)?,
    |b| kernels.discard_buffer(b),
  );
  let (rb_buf, wr_buf, n_bodies) = kernels.build_rigid_bodies(&mut cmd, physical_scene, scene)?;
  let mut rigid_bodies = AutoDiscard::new(rb_buf, |b| kernels.discard_buffer(b));
  let mut wrenches = AutoDiscard::new(wr_buf, |b| kernels.discard_buffer(b));
  // build_particles returns one (entity_id, buffer, metadata) per ParticleSystemComponent.
  // Each system gets its own AoSoA GPU buffer so BVH bounds and self-gravity are per-system.
  let mut particle_systems = kernels.build_particles(&mut cmd, scene)?;
  // Flat metadata list (all systems concatenated) used only by write_back_to_scene routing.
  let particle_metadata: alloc::vec::Vec<crate::gpu::ParticleMetadata> =
    particle_systems.iter().flat_map(|(_, _, meta, _)| meta.iter().copied()).collect();
  let (emitters_buf, n_emitters) = kernels.build_emitters(&mut cmd, scene)?;
  let emitters = AutoDiscard::new(emitters_buf, |b| kernels.discard_buffer(b));
  // Reference frames for LCA broad-phase (macro frame is always index 0)
  let frames = AutoDiscard::new(kernels.build_frames(&mut cmd, physical_scene)?, |b| {
    kernels.discard_buffer(b)
  });
  // For collision paths that need a particles BDA (narrow_ccd, restore_dynamics) even in
  // pure rigid-body scenes, we build a 0-element emission-candidates buffer here.
  // It is allocated in the same command batch so the transient pool drains it after cmd.submit().
  #[cfg(any(test, feature = "collisions"))]
  let mut null_particles_buf = AutoDiscard::new(

    kernels.build_emission_candidates(&mut cmd, scene)?,
    |b| kernels.discard_buffer(b),
  );
  aethervk_oshal_rlib::log!("PRINT_ADDR rigid_bodies: 0x{:x}", rigid_bodies.address());
  aethervk_oshal_rlib::log!("PRINT_ADDR frames: 0x{:x}", frames.address());
  sync!("buffer_uploads (kinematics + rigid_bodies + particles + emitters + frames)");


  // ── 2. Sun position (for particle emission) ───────────────────────────────
  let mut sun_pos = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero();
  if let Some((sun_id, _)) =
    scene.query1_first_res::<crate::scene::SunComponent, _, _>(|id, _| Some(id))
  {
    if let Some(pos) = scene.global_transform(sun_id).map(|t| t.position) {
      sun_pos = pos;
    }
  }

  // ── 3. Particle emission ───────────────────────────────────────────────────
  // NOTE: Emission is now handled on the CPU side by `emit_particles_from_circles`
  // in `dispatch_physics_step` (logic_thread.rs).  CPU-emitted particles are
  // already in the ParticleSystemComponent and were picked up by `build_particles`
  // above, so they have proper ParticleMetadata for write_back_to_scene.
  //
  // The GPU `emit_particles` kernel is intentionally SKIPPED here because:
  //  1. It reads from ParticleEmitterCirclesComponent (parent entity) and emits
  //     into the GPU mega-buffer — but those particles have no metadata and would
  //     be lost after write_back_to_scene.
  //  2. Running it would double the particle count for one tick (wasted compute).
  //
  // If GPU-side emission with occlusion testing is needed in the future, the
  // kernel should write metadata alongside particles so write_back can route them.

  // ── 4. IMEX integration — sub-stepped for numerical stability ─────────────
  #[cfg(not(any(test, feature = "collisions")))]
  let collisions_enabled = false;

  if !collisions_enabled {
    // ── Non-collision path: sub-step the IMEX integration ───────────────────
    // TODO: this should be a closure cause collision pipeline should do a substep.
    for sub_step_idx in 0..n_sub_steps {
      // Compute this sub-step's dt.  The last sub-step absorbs rounding remainder.
      let sub_dt = if sub_step_idx == n_sub_steps - 1 {
        end_time - current_time
      } else {
        total_dt / n_sub_steps as timeus_t
      };

      // ── Per-system particle physics ─────────────────────────────────────────
      // VV predictor runs per-system so each buffer is independent.
      for (sys_entity, sys_particles, sys_meta, _skip_self_gravity) in &mut particle_systems {
        if sys_meta.is_empty() { continue; }
        kernels.imex_integrate_particles_p1_p2(&mut cmd, sys_particles, sub_dt)?;
      }
      if !particle_metadata.is_empty() { sync!("imex_integrate_particles_p1_p2"); }

      if n_bodies > 0 {
        // RB: accumulate forces then IMR solve
        kernels.imex_rb_force_assign(&mut cmd, &rigid_bodies, &mut wrenches, n_bodies)?;
        sync!("imex_rb_force_assign");

        kernels.imex_integrate_bodies_p3(
          &mut cmd,
          &mut rigid_bodies,
          &mut wrenches,
          &emitters,
          &frames,
          n_bodies,
          n_emitters,
          sub_dt,
        )?;
        sync!("imex_integrate_bodies_p3");
      }

      // Per-system: BVH → self-gravity → external emitters → VV corrector
      for (sys_entity, sys_particles, sys_meta, skip_self_gravity) in &mut particle_systems {
        if sys_meta.is_empty() { continue; }

        let mut sys_particle_frame_ids = AutoDiscard::new(
          kernels.build_particle_frame_ids(&mut cmd, sys_meta)?,
          |b| kernels.discard_buffer(b),
        );

        if *skip_self_gravity {
          // ── Fast path: test particles (comet dust, tracers) ─────────────────
          // Skip Morton sort, radix sort, LBVH build, and Barnes-Hut self-gravity.
          // Apply external emitter forces directly into AOSOA slots 7-9, then integrate.
          kernels.apply_emitters_direct(
            &mut cmd,
            sys_particles,
            &emitters,
            &frames,
            &*sys_particle_frame_ids,
            n_emitters,
          )?;
          sync!("apply_emitters_direct (test-particle fast path)");
        } else {
          // ── Full path: self-gravitating particle cloud ────────────────────
          let bvh = AutoDiscard::new(
            kernels.build_motion_bvh(
              &mut cmd, &kinematics, &rigid_bodies,
              sys_particles, &mut *sys_particle_frame_ids, sub_dt, *sys_entity,
              None,
            )?,
            |b| kernels.discard_bvh(b),
          );
          sync!("build_motion_bvh");

          kernels.compute_self_gravity(&mut cmd, &bvh, sys_particles)?;
          sync!("compute_self_gravity (barnes_hut)");

          kernels.apply_emitters_to_particles(
            &mut cmd,
            sys_particles,
            &emitters,
            &frames,
            &*sys_particle_frame_ids,
            &*bvh,
            n_emitters,
          )?;
          sync!("apply_emitters_to_particles (Phase A)");

          kernels.accumulate_bvh_forces_to_particles(&mut cmd, sys_particles, &*bvh)?;
          sync!("accumulate_bvh_forces_to_particles (Phase B)");
        }

        kernels.imex_integrate_particles_p4_5(&mut cmd, sys_particles, sub_dt, current_time)?;
        sync!("imex_integrate_particles_p4_5");
      }

      current_time += sub_dt;
    }
  } else {
    #[cfg(any(test, feature = "collisions"))]
    {
      // For collision kernels (snapshot/restore/bp_particle_self/narrow_ccd/
      // apply_collision_responses) that take a single merged &Buffer<f32>, we route
      // through the first particle system's buffer. These kernels deal with RB↔particle
      // interactions; for multi-system scenes, only the first system participates in CCD
      // (a future enhancement can merge all systems into one buffer).
      // All per-system physics (BVH, self-gravity, emitters, VV) loop over all systems.
      while current_time < end_time {
        let dt = end_time - current_time;

        aethervk_oshal_rlib::log!("gpu_backends.rs: entering while loop");

        // Snapshot: pass Some(first_sys_buf) when particles exist, or None for pure RB scenes.
        // The new trait takes Option<&Buffer<f32>> so we don't need to build any placeholder buffer.
        let has_particles = !particle_systems.is_empty();
        let snapshot = AutoDiscard::new(
          {
            let p_opt: Option<&K::Buffer<f32>> = if has_particles {
              Some(&particle_systems[0].1)
            } else {
              None
            };
            kernels.snapshot_dynamics(&mut cmd, &rigid_bodies, p_opt)?
          },
          |s| {
            kernels.discard_buffer(s.0);
            if let Some(p) = s.1 { kernels.discard_buffer(p); }
          },
        );






        // ── VV predictor: per-system ──────────────────────────────────────────
        for (_, sys_buf, sys_meta, _) in &mut particle_systems {
          if sys_meta.is_empty() { continue; }
          aethervk_oshal_rlib::log!("gpu_backends.rs: calling imex_integrate_particles_p1_p2");
          kernels.imex_integrate_particles_p1_p2(&mut cmd, sys_buf, dt)?;
        }
        if !particle_metadata.is_empty() { sync!("imex_integrate_particles_p1_p2 (collisions path)"); }

        if n_bodies > 0 {
          aethervk_oshal_rlib::log!("gpu_backends.rs: calling imex_rb_force_assign");
          kernels.imex_rb_force_assign(&mut cmd, &rigid_bodies, &mut wrenches, n_bodies)?;
          sync!("imex_rb_force_assign (collisions path)");
          aethervk_oshal_rlib::log!("gpu_backends.rs: calling imex_integrate_bodies_p3");
          kernels.imex_integrate_bodies_p3(
            &mut cmd,
            &mut rigid_bodies,
            &mut wrenches,
            &emitters,
            &frames,
            n_bodies,
            n_emitters,
            dt,
          )?;
          sync!("imex_integrate_bodies_p3 (collisions path)");
        }

        // ── Per-system: BVH → self-gravity → emitters → VV corrector ─────────
        for (sys_entity, sys_buf, sys_meta, skip_self_gravity) in &mut particle_systems {
          if sys_meta.is_empty() { continue; }
          let mut sys_fids = AutoDiscard::new(
            kernels.build_particle_frame_ids(&mut cmd, sys_meta)?,
            |b| kernels.discard_buffer(b),
          );
          if *skip_self_gravity {
            kernels.apply_emitters_direct(
              &mut cmd, sys_buf, &emitters, &frames, &*sys_fids, n_emitters,
            )?;
          } else {
            let bvh = AutoDiscard::new(
              kernels.build_motion_bvh(
                &mut cmd, &kinematics, &rigid_bodies,
                sys_buf, &mut *sys_fids, dt, *sys_entity, None,
              )?,
              |b| kernels.discard_bvh(b),
            );
            sync!("build_motion_bvh (collisions path)");
            kernels.compute_self_gravity(&mut cmd, &bvh, sys_buf)?;
            sync!("compute_self_gravity (collisions path)");
            kernels.apply_emitters_to_particles(
              &mut cmd, sys_buf, &emitters, &frames, &*sys_fids, &*bvh, n_emitters,
            )?;
            sync!("apply_emitters_to_particles (collisions Phase A)");
            kernels.accumulate_bvh_forces_to_particles(&mut cmd, sys_buf, &*bvh)?;
            sync!("accumulate_bvh_forces_to_particles (collisions Phase B)");
          }
          kernels.imex_integrate_particles_p4_5(&mut cmd, sys_buf, dt, current_time)?;
          sync!("imex_integrate_particles_p4_5 (collisions path)");
        }

        // ── New broad-phase suite ─────────────────────────────────────────────
        let sg = kernels.subgroup_size().map(|s| s as u32).unwrap_or(32);
        let n_rb_entities = n_bodies as u32;
        let actual_particles = particle_metadata.len() as u32;
        let n_ps_entities = if actual_particles > 0 {
          (actual_particles + sg - 1) / sg
        } else {
          0
        };
        let n_entities = n_rb_entities;
        let _ = n_ps_entities;
        aethervk_oshal_rlib::log!(
          "BP_DIAG: n_entities={}, frames.capacity={}, tlas_root_idx={}",
          n_entities,
          frames.capacity(),
          tlas_root_idx
        );

        let tlas_bvh_addr = tlas_addr;
        let mut raw_pairs = AutoDiscard::new(
          kernels.build_list::<crate::gpu::CollisionPair>(&mut cmd, 10_000)?,
          |b| kernels.discard_list(b),
        );
        aethervk_oshal_rlib::log!("PRINT_ADDR raw_pairs: 0x{:x}", raw_pairs.address());
        let rb_rb_pairs = AutoDiscard::new(
          kernels.build_list::<crate::gpu::CollisionPair>(&mut cmd, 4_000)?,
          |b| kernels.discard_list(b),
        );
        let rb_ps_pairs = AutoDiscard::new(
          kernels.build_list::<crate::gpu::CollisionPair>(&mut cmd, 4_000)?,
          |b| kernels.discard_list(b),
        );
        let rb_lca_pairs = AutoDiscard::new(
          kernels.build_list::<crate::gpu::CollisionPair>(&mut cmd, 2_000)?,
          |b| kernels.discard_list(b),
        );
        let query_leaves =
          AutoDiscard::new(kernels.build_leaves(&mut cmd, n_entities as usize)?, |b| {
            kernels.discard_buffer(b)
          });
        let internal_pairs = AutoDiscard::new(
          kernels.build_list::<crate::gpu::CrossPair>(&mut cmd, 2_000)?,
          |b| kernels.discard_list(b),
        );
        let ps_ps_pairs = AutoDiscard::new(
          kernels.build_list::<crate::gpu::CollisionPair>(&mut cmd, 64)?,
          |b| kernels.discard_list(b),
        );
        let sparse_collisions = AutoDiscard::new(
          kernels.build_list::<crate::gpu::CollisionPair>(&mut cmd, 12000)?,
          |b| kernels.discard_list(b),
        );

        kernels.bp_clear(
          &mut cmd,
          raw_pairs.address(),
          rb_rb_pairs.address(),
          rb_ps_pairs.address(),
          rb_lca_pairs.address(),
          internal_pairs.address(),
          sparse_collisions.address(),
        )?;
        sync!("bp_clear");
        kernels.bp_bounds_gen(
          &mut cmd,
          &rigid_bodies,
          query_leaves.address(),
          frames.address(),
          n_rb_entities,
          dt,
        )?;
        sync!("bp_bounds_gen");
        kernels.bp_scene(
          &mut cmd,
          tlas_bvh_addr,
          query_leaves.address(),
          raw_pairs.address(),
          tlas_root_idx,
          n_entities,
        )?;
        sync!("bp_scene");
        kernels.bp_classify(
          &mut cmd,
          &rigid_bodies,
          raw_pairs.address(),
          rb_rb_pairs.address(),
          rb_ps_pairs.address(),
          ps_ps_pairs.address(),
          rb_lca_pairs.address(),
          0,
          n_entities * n_entities,
        )?;
        sync!("bp_classify");
        if frames.capacity() > 1 {
          kernels.bp_cross_lca(
            &mut cmd,
            tlas_bvh_addr,
            frames.address(),
            query_leaves.address(),
            rigid_bodies.address(),
            rb_lca_pairs.address(),
            rb_rb_pairs.address(),
            rb_ps_pairs.address(),
            ps_ps_pairs.address(),
            internal_pairs.address(),
            rb_lca_pairs.capacity() as u32,
            2_000,
            rigid_bodies.capacity() as u32,
          )?;
          sync!("bp_cross_lca");
        }
        // bp_particle_self uses first system's buffer (CCD only sees one merged set).
        if actual_particles > 0 {
          if let Some((_, sys_buf, _, _)) = particle_systems.first_mut() {
            let p_addr = sys_buf.address();
            kernels.bp_particle_self(
              &mut cmd,
              tlas_bvh_addr,
              sys_buf,
              p_addr,
              actual_particles,
              0,
              0.5_f32,
              1_000.0_f32,
            )?;
            sync!("bp_particle_self");
          }
        }

        // ── Narrow phase ─────────────────────────────────────────────────────
        // Use first system's buffer for particle lookups, or null stub if no systems.
        // null_particles_buf is a 0-element buffer allocated at simulate_step start;
        // in pure RB scenes (no particle pairs) the shader never dereferences this address.
        let narrow_p_buf: &K::Buffer<f32> = if has_particles {
          &particle_systems[0].1
        } else {
          &null_particles_buf
        };
        kernels.narrow_ccd(
          &mut cmd,
          &*rb_rb_pairs,
          &rigid_bodies,
          narrow_p_buf,
          frames.address(),
          0,
          (dt as f32) / 1_000_000.0,

          &sparse_collisions,
        )?;
        sync!("narrow_ccd");

        if frames.capacity() > 1 {
          kernels.narrow_ccd_cross_lca(
            &mut cmd,
            &*internal_pairs,
            &rigid_bodies,
            narrow_p_buf,
            frames.address(),
            1,
            (dt as f32) / 1_000_000.0,
            &sparse_collisions,
          )?;
          sync!("narrow_ccd_cross_lca");
        }



        let compacted = AutoDiscard::new(
          kernels.compact_collisions(&mut cmd, &sparse_collisions, time_collision_delta)?,
          |b| kernels.discard_list(b),
        );
        sync!("compact_collisions");

        let tc_buffer = AutoDiscard::new(
          kernels.find_earliest_collision(
            &mut cmd,
            &compacted,
            time_collision_delta as f32 / 1_000_000.0,
          )?,
          |b| kernels.discard_buffer(b),
        );
        let sync_info = cmd.submit()?;
        if let Some(sync) = sync_info {
          kernels.wait_sync(&sync)?;
        }

        #[cfg(test)]
        {
          if crate::gpu_backends::vulkan::physics::READBACK_DIAGNOSTICS
            .load(core::sync::atomic::Ordering::Relaxed)
          {
            if kernels.is_cpu_device() {
              kernels.wait_idle().unwrap();
            }

            let cmp_host = unsafe { (*compacted).mapped_slice().unwrap_or(&[]) };
            let _rb_host = unsafe { rigid_bodies.mapped_slice().unwrap_or(&[]) };
            let _frames_host = unsafe { frames.mapped_slice().unwrap_or(&[]) };
            let cross_host = unsafe { (*internal_pairs).mapped_slice().unwrap_or(&[]) };
            let raw_host = unsafe { (*raw_pairs).mapped_slice().unwrap_or(&[]) };
            let lca_host = unsafe { (*rb_lca_pairs).mapped_slice().unwrap_or(&[]) };

            let compacted_count = cmp_host.len();
            let cross_count = cross_host.len();
            let raw_count = raw_host.len();
            let lca_count = lca_host.len();
            aethervk_oshal_rlib::log!(
              "READBACK: raw_count={}, lca_count={}, cross_count={}, compacted_count={}",
              raw_count,
              lca_count,
              cross_count,
              compacted_count
            );

            let mut body_entity_map = alloc::vec::Vec::new();
            scene.query2_without::<crate::scene::TransformComponent, crate::scene::ColliderComponent, crate::scene::particles::ParticleSystemComponent, _>(|entity, _, _| {
                   body_entity_map.push(entity);
               });

            if compacted_count > 0 {
              for i in 0..compacted_count {
                let pair = &cmp_host[i];
                let id_a = pair.a.entity_id;
                let prim_a = pair.a.primitive_index;
                let id_b = pair.b.entity_id;
                let prim_b = pair.b.primitive_index;

                aethervk_oshal_rlib::log!(
                  "RAW PAIR: id_a={}, id_b={}, prim_a={}, prim_b={}, toi={}",
                  id_a, id_b, prim_a, prim_b, pair.time_of_impact
                );

                let is_lca = pair.is_lca != 0;
                let lca_id = if is_lca { Some(pair.lca_id) } else { None };

                let mut name_a = alloc::string::String::new();
                if let Some(&entity_a) = body_entity_map.get(id_a as usize) {
                  use slotmap::Key;
                  let ffi_a = entity_a.data().as_ffi() as u64;
                  if let Some(n) = scene.get_name(crate::scene::EntityId::from(
                    slotmap::KeyData::from_ffi(ffi_a),
                  )) {
                    name_a = n;
                  }
                }
                if name_a.is_empty() { name_a = alloc::format!("Entity_{}", id_a); }

                let mut name_b = alloc::string::String::new();
                if let Some(&entity_b) = body_entity_map.get(id_b as usize) {
                  use slotmap::Key;
                  let ffi_b = entity_b.data().as_ffi() as u64;
                  if let Some(n) = scene.get_name(crate::scene::EntityId::from(
                    slotmap::KeyData::from_ffi(ffi_b),
                  )) {
                    name_b = n;
                  }
                }
                if name_b.is_empty() { name_b = alloc::format!("Entity_{}", id_b); }

                let mut particle_path_a = None;
                if let Some(&entity_a) = body_entity_map.get(id_a as usize) {
                  if scene.with_component(entity_a, |_: &crate::scene::particles::ParticleSystemComponent| ()).is_some() {
                    use slotmap::Key;
                    let ffi_a = entity_a.data().as_ffi() as u32;
                    if let Some(p_idx) = physical_scene.particle_entity_map.iter().position(|&ffi| ffi == ffi_a) {
                      if let Some(blas) = &physical_scene.particle_blases[p_idx] {
                        particle_path_a = blas.find_path_to_primitive(prim_a as usize);
                      }
                    }
                  }
                }

                let mut particle_path_b = None;
                if let Some(&entity_b) = body_entity_map.get(id_b as usize) {
                  if scene.with_component(entity_b, |_: &crate::scene::particles::ParticleSystemComponent| ()).is_some() {
                    use slotmap::Key;
                    let ffi_b = entity_b.data().as_ffi() as u32;
                    if let Some(p_idx) = physical_scene.particle_entity_map.iter().position(|&ffi| ffi == ffi_b) {
                      if let Some(blas) = &physical_scene.particle_blases[p_idx] {
                        particle_path_b = blas.find_path_to_primitive(prim_b as usize);
                      }
                    }
                  }
                }

                physical_scene.recent_collisions.push(
                  crate::physics::physics_scene::CollisionEvent {
                    entity_a_id: id_a,
                    entity_a_name: name_a,
                    entity_b_id: id_b,
                    entity_b_name: name_b,
                    contact_point: pair.contact_point,
                    contact_normal: pair.contact_normal,
                    penetration_depth: pair.penetration_depth,
                    frame_id: lca_id.unwrap_or(0),
                    is_lca,
                    particle_path_a,
                    particle_path_b,
                  },
                );
              }
            }
          }
        }

        let t_c_raw: u32 = kernels.read_buffer_u32_first(&*tc_buffer).unwrap_or(0xFFFFFFFF);
        aethervk_oshal_rlib::log!("gpu_backends.rs: t_c_raw is {}", t_c_raw);

        let mut next_cmd = kernels.create_command_buffer()?;
        core::mem::swap(&mut cmd, &mut next_cmd);
        old_cmds.push(next_cmd);

        aethervk_oshal_rlib::log!("gpu_backends.rs: t_c is {}", t_c_raw);

        let t_c = if t_c_raw == 0xFFFFFFFF {
          timeus_t::MAX
        } else {
          let t_c_f32 = f32::from_bits(t_c_raw);
          if t_c_f32 < 0.0 {
            aethervk_oshal_rlib::log!(
              "gpu_backends.rs: warning: negative t_c float {}, assuming 0",
              t_c_f32
            );
            0
          } else {
            (t_c_f32 * 1_000_000.0_f32) as timeus_t
          }
        };

        if t_c < dt {
          collision_iters += 1;
          let inelastic = collision_iters >= MAX_BOUNCES;

          // Restore: always restore rigid bodies; restore particles only if present.
          {
            let p_opt: Option<&mut K::Buffer<f32>> = particle_systems.first_mut()
              .map(|(_, buf, _, _)| buf);
            kernels.restore_dynamics(&mut cmd, &mut rigid_bodies, p_opt, &snapshot)?;
          }

          // Re-integrate to t_c: per-system VV predictor
          for (_, sys_buf, sys_meta, _) in &mut particle_systems {
            if sys_meta.is_empty() { continue; }
            kernels.imex_integrate_particles_p1_p2(&mut cmd, sys_buf, t_c)?;
          }
          kernels.imex_rb_force_assign(&mut cmd, &rigid_bodies, &mut wrenches, n_bodies)?;
          kernels.imex_integrate_bodies_p3(
            &mut cmd, &mut rigid_bodies, &mut wrenches, &emitters, &frames,
            n_bodies, n_emitters, t_c,
          )?;

          // Per-system: BVH/self-gravity/emitters/VV corrector at t_c
          for (sys_entity, sys_buf, sys_meta, skip_self_gravity) in &mut particle_systems {
            if sys_meta.is_empty() { continue; }
            let mut sys_fids = AutoDiscard::new(
              kernels.build_particle_frame_ids(&mut cmd, sys_meta)?,
              |b| kernels.discard_buffer(b),
            );
            if *skip_self_gravity {
              kernels.apply_emitters_direct(
                &mut cmd, sys_buf, &emitters, &frames, &*sys_fids, n_emitters,
              )?;
            } else {
              let rewind_bvh = AutoDiscard::new(
                kernels.build_motion_bvh(
                  &mut cmd, &kinematics, &rigid_bodies,
                  sys_buf, &mut *sys_fids, t_c, *sys_entity, None,
                )?,
                |b| kernels.discard_bvh(b),
              );
              kernels.compute_self_gravity(&mut cmd, &rewind_bvh, sys_buf)?;
              kernels.apply_emitters_to_particles(
                &mut cmd, sys_buf, &emitters, &frames, &*sys_fids, &*rewind_bvh, n_emitters,
              )?;
              kernels.accumulate_bvh_forces_to_particles(&mut cmd, sys_buf, &*rewind_bvh)?;
            }
            kernels.imex_integrate_particles_p4_5(&mut cmd, sys_buf, t_c, current_time)?;
          }

          // Collision response: always call, using first system's buffer or null stub.
          // In pure RB scenes (no particle systems), null_particles_buf is used and the
          // shader only reads from it for particle pairs (which don't exist in this case).
          {
            let p_mut: &mut K::Buffer<f32> = if has_particles {
              &mut particle_systems[0].1
            } else {
              // null_particles_buf is AutoDiscard; we need a &mut to its inner Buffer<f32>.
              // Safe: we only borrow it mutably here, and it's only used as a BDA by the shader.
              &mut *null_particles_buf
            };
            kernels.apply_collision_responses(
              &mut cmd, &kinematics, &mut rigid_bodies, p_mut, &compacted, frames.address(), inelastic,
            )?;
          }

          if inelastic {
            let remaining_dt = dt - t_c;
            if remaining_dt > 0 {
              // Per-system VV predictor for remainder
              for (_, sys_buf, sys_meta, _) in &mut particle_systems {
                if sys_meta.is_empty() { continue; }
                kernels.imex_integrate_particles_p1_p2(&mut cmd, sys_buf, remaining_dt)?;
              }
              kernels.imex_rb_force_assign(&mut cmd, &rigid_bodies, &mut wrenches, n_bodies)?;
              kernels.imex_integrate_bodies_p3(
                &mut cmd, &mut rigid_bodies, &mut wrenches, &emitters, &frames,
                n_bodies, n_emitters, remaining_dt,
              )?;
              // Per-system: BVH/self-gravity/emitters/VV corrector for remainder
              for (sys_entity, sys_buf, sys_meta, skip_self_gravity) in &mut particle_systems {
                if sys_meta.is_empty() { continue; }
                let mut sys_fids = AutoDiscard::new(
                  kernels.build_particle_frame_ids(&mut cmd, sys_meta)?,
                  |b| kernels.discard_buffer(b),
                );
                if *skip_self_gravity {
                  kernels.apply_emitters_direct(
                    &mut cmd, sys_buf, &emitters, &frames, &*sys_fids, n_emitters,
                  )?;
                } else {
                  let final_bvh = AutoDiscard::new(
                    kernels.build_motion_bvh(
                      &mut cmd, &kinematics, &rigid_bodies,
                      sys_buf, &mut *sys_fids, remaining_dt, *sys_entity, None,
                    )?,
                    |b| kernels.discard_bvh(b),
                  );
                  kernels.compute_self_gravity(&mut cmd, &final_bvh, sys_buf)?;
                  kernels.apply_emitters_to_particles(
                    &mut cmd, sys_buf, &emitters, &frames, &*sys_fids, &*final_bvh, n_emitters,
                  )?;
                  kernels.accumulate_bvh_forces_to_particles(&mut cmd, sys_buf, &*final_bvh)?;
                }
                #[cfg(test)] println!("!!! EXEC: compute_self_gravity and imex_integrate_particles_p4_5");
                kernels.imex_integrate_particles_p4_5(
                  &mut cmd, sys_buf, remaining_dt, current_time + t_c,
                )?;
              }
            }
            current_time = end_time;
          } else {
            let advance = if t_c == 0 { 1 } else { t_c };
            current_time += advance;
          }
        } else {
          aethervk_oshal_rlib::log!("gpu_backends.rs: calling kernels.write_back_to_scene!");
          current_time = end_time;
        }
      }
    }
  }

  aethervk_oshal_rlib::log!("gpu_backends.rs: calling cmd.submit() before write_back_to_scene!");
  let sync_info = cmd.submit()?;
  if let Some(sync) = &sync_info {
    let _ = kernels.wait_sync(sync);
  }

  aethervk_oshal_rlib::log!("gpu_backends.rs: calling kernels.write_back_to_scene OUTSIDE LOOP!");
  let _ = kernels.write_back_to_scene(
    &mut cmd,
    &rigid_bodies,
    &particle_systems,
    physical_scene,
    scene,
  )?;

  // Explicitly free the per-system particle GPU buffers.
  // VulkanBuffer::drop only logs a warning; the actual VMA allocation must be
  // freed via kernels.discard_buffer (which queues it for timeline-safe deallocation).
  for (_, sys_buf, _, _) in particle_systems {
    kernels.discard_buffer(sys_buf);
  }

  Ok(sync_info)
}