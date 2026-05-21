//! gpu_backends module.

use crate::{
  gpu::{
    CommandBuffer, DeviceBuffer, DeviceBvh, Kernels,
    RenderBackendId, RenderFrontend, VULKAN_RENDER_BACKEND, WaitHandle,
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

/// Fixed Update Function in interval $t_0, t_1$
/// Note that this function should be asynchronous with respect to the logic_thread,
/// meaning this function should be dispatched inside the pool as a singleton tasklet.
/// - we expect the physical_scene to contain BVH of entities with state at $t_0$
#[allow(deprecated)] // step_ode_* kept as fallback during migration
pub fn simulation_step<K>(
  kernels: &K,
  physical_scene: &mut PhysicsScene,
  scene: &Scene,
  t0: timeus_t,
  t1: timeus_t,
  collisions_enabled: bool,
) -> EngineResult<()>
where
  K: Kernels + ?Sized,
{
  aethervk_oshal_rlib::log!(
    "simulation_step running! dt_us: {}, collisions_enabled: {}",
    t1 - t0,
    collisions_enabled
  );
  let mut cmd = kernels.create_command_buffer()?;

  let mut current_time = t0;
  let end_time = t1;
  let time_collision_delta = timeus_milliseconds(30);
  let mut collision_iters = 0;
  const MAX_BOUNCES: usize = 5;

  // ── 1. Build per-frame GPU buffers from ECS ───────────────────────────────
  let kinematics = kernels.build_kinematic_bodies(&mut cmd, physical_scene, scene)?;
  let (mut rigid_bodies, mut wrenches) =
    kernels.build_rigid_bodies(&mut cmd, physical_scene, scene)?;
  let (mut particles, particle_metadata) = kernels.build_particles(&mut cmd, scene)?;
  let emitters = kernels.build_emitters(&mut cmd, scene)?;
  // Reference frames for LCA broad-phase (macro frame is always index 0)
  let frames = kernels.build_frames(&mut cmd, physical_scene)?;

  // ── 2. Sun position (for particle emission) ───────────────────────────────
  let mut sun_pos = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero();
  if let Some((sun_id, _)) =
    scene.query1_first_res::<crate::scene::SunComponent, _, _>(|id, _| Some(id))
  {
    if let Some(pos) = scene.global_transform(sun_id).map(|t| t.position) {
      sun_pos = pos;
    }
  }

  // ── 3. Particle emission ──────────────────────────────────────────────────
  let full_dt = t1 - t0;
  kernels.emit_particles(&mut cmd, &mut particles, physical_scene, scene, sun_pos, full_dt)?;

  // ── 4. IMEX integration + collision loop ─────────────────────────────────
  if !collisions_enabled {
    let dt = end_time - current_time;

    // VV predictor: half-kick + full drift to x_{n+1}
    kernels.imex_integrate_particles_p1_p2(&mut cmd, &mut particles, dt)?;

    // RB: accumulate forces then IMR solve
    kernels.imex_rb_force_assign(&mut cmd, &rigid_bodies, &mut wrenches)?;
    kernels.imex_integrate_bodies_p3(&mut cmd, &mut rigid_bodies, &mut wrenches, dt)?;

    // Build motion BVH for self-gravity
    let bvh = kernels.build_motion_bvh(&mut cmd, &kinematics, &rigid_bodies, &particles, dt)?;
    kernels.compute_self_gravity(&mut cmd, &bvh, &mut particles)?;
    kernels.discard_bvh(bvh);

    // VV corrector: v_{n+1/2} → v_{n+1} using F(x_{n+1})
    kernels.imex_integrate_particles_p4_5(&mut cmd, &mut particles, dt, current_time)?;

    cmd.submit()?;
  } else {
    while current_time < end_time {
      let dt = end_time - current_time;

      // ── Snapshot state for possible CCD rewind ──
      let snapshot = kernels.snapshot_dynamics(&mut cmd, &rigid_bodies, &particles)?;

      // ── IMEX integration to t_{n+1} ──
      kernels.imex_integrate_particles_p1_p2(&mut cmd, &mut particles, dt)?;
      kernels.imex_rb_force_assign(&mut cmd, &rigid_bodies, &mut wrenches)?;
      kernels.imex_integrate_bodies_p3(&mut cmd, &mut rigid_bodies, &mut wrenches, dt)?;

      let bvh = kernels.build_motion_bvh(&mut cmd, &kinematics, &rigid_bodies, &particles, dt)?;
      kernels.compute_self_gravity(&mut cmd, &bvh, &mut particles)?;
      kernels.imex_integrate_particles_p4_5(&mut cmd, &mut particles, dt, current_time)?;

      // ── New broad-phase suite ─────────────────────────────────────────────
      // Allocate pair-list buffers (is_list=true → 16-byte header: [x,y,z,count])
      let n_entities = (rigid_bodies.capacity() + particles.capacity()) as u32;

      // bp_clear: zero all four pair-list counters via raw BDAs
      // (The caller manages the pair-list buffers; for now we reuse bvh.address
      //  as a placeholder TLAS address until a proper TLAS builder is wired in.)
      let tlas_bvh_addr = bvh.address();
      let raw_pairs = kernels.build_list::<crate::gpu::CollisionPair>(&mut cmd, 10_000)?;
      let rb_rb_pairs = kernels.build_list::<crate::gpu::CollisionPair>(&mut cmd, 4_000)?;
      let rb_ps_pairs = kernels.build_list::<crate::gpu::CollisionPair>(&mut cmd, 4_000)?;
      let rb_lca_pairs = kernels.build_list::<crate::gpu::CollisionPair>(&mut cmd, 2_000)?;

      kernels.bp_clear(
        &mut cmd,
        raw_pairs.address(),
        rb_rb_pairs.address(),
        rb_ps_pairs.address(),
        rb_lca_pairs.address(),
      )?;
      kernels.bp_bounds_gen(&mut cmd, &rigid_bodies, raw_pairs.address(), n_entities, dt)?;
      kernels.bp_scene(
        &mut cmd,
        tlas_bvh_addr,
        raw_pairs.address(),
        raw_pairs.address(),
        0, // TLAS root index
        n_entities,
      )?;
      kernels.bp_classify(
        &mut cmd,
        &rigid_bodies,
        raw_pairs.address(),
        rb_rb_pairs.address(),
        rb_ps_pairs.address(),
        rb_lca_pairs.address(),
        n_entities * n_entities, // conservative upper bound on raw pairs
      )?;
      // Cross-LCA: only if micro-frames exist (frames.capacity() > 1)
      let internal_pairs = kernels.build_list::<crate::gpu::CollisionPair>(&mut cmd, 2_000)?;
      if frames.capacity() > 1 {
        kernels.bp_cross_lca(
          &mut cmd,
          &frames,
          rb_lca_pairs.address(),
          internal_pairs.address(),
          frames.capacity() as u32,
        )?;
      }
      // Particle self-collision via Hookean spring forces
      let p_addr = particles.address();
      let p_cap = particles.capacity() as u32;
      kernels.bp_particle_self(
        &mut cmd,
        tlas_bvh_addr,
        &mut particles,
        p_addr,
        p_cap,
        0, // BVH root index
        0.5_f32, // particle radius (m)
        1_000.0_f32, // spring stiffness k
      )?;

      // ── Merge classified pairs → global collision list ────────────────────
      // For now, merge rb_rb + rb_ps via compact_collisions
      let globals = &rb_rb_pairs; // TODO: merge rb_rb + rb_ps + internal_pairs
      let compacted = kernels.compact_collisions(&mut cmd, globals, time_collision_delta)?;

      let tc_buffer = kernels.find_earliest_collision(&mut cmd, &compacted)?;
      let tc_future = tc_buffer.enqueue_read_to_cpu(&mut cmd)?;

      kernels.discard_bvh(bvh);
      kernels.discard_list(raw_pairs);
      kernels.discard_list(rb_lca_pairs);
      kernels.discard_list(rb_ps_pairs);
      kernels.discard_list(internal_pairs);

      cmd.submit()?;
      let tc_host = tc_future.wait()?;

      let t_c = tc_host.first().copied().unwrap_or(timeus_t::MAX);

      // FIX: Only trigger response mechanisms if a collision happens inside the current timestep
      if t_c < dt {
        collision_iters += 1;
        let inelastic = collision_iters >= MAX_BOUNCES;

        kernels.restore_dynamics(&mut cmd, &mut rigid_bodies, &mut particles, &snapshot)?;

        // Re-integrate to t_c
        kernels.imex_integrate_particles_p1_p2(&mut cmd, &mut particles, t_c)?;
        kernels.imex_rb_force_assign(&mut cmd, &rigid_bodies, &mut wrenches)?;
        kernels.imex_integrate_bodies_p3(&mut cmd, &mut rigid_bodies, &mut wrenches, t_c)?;

        let rewind_bvh =
          kernels.build_motion_bvh(&mut cmd, &kinematics, &rigid_bodies, &particles, t_c)?;
        kernels.compute_self_gravity(&mut cmd, &rewind_bvh, &mut particles)?;
        kernels.imex_integrate_particles_p4_5(&mut cmd, &mut particles, t_c, current_time)?;
        kernels.discard_bvh(rewind_bvh);

        // Apply either an elastic or inelastic response at the proper time t_c
        kernels.apply_collision_responses(
          &mut cmd,
          &kinematics,
          &mut rigid_bodies,
          &mut particles,
          &compacted,
          inelastic,
        )?;

        kernels.discard_list(compacted);
        kernels.discard_list(rb_rb_pairs);

        if inelastic {
          // If we resolved resting contact, integrate the remainder directly.
          let remaining_dt = dt - t_c;
          if remaining_dt > 0 {
            kernels.imex_integrate_particles_p1_p2(&mut cmd, &mut particles, remaining_dt)?;
            kernels.imex_rb_force_assign(&mut cmd, &rigid_bodies, &mut wrenches)?;
            kernels.imex_integrate_bodies_p3(
              &mut cmd,
              &mut rigid_bodies,
              &mut wrenches,
              remaining_dt,
            )?;
            let final_bvh = kernels.build_motion_bvh(
              &mut cmd,
              &kinematics,
              &rigid_bodies,
              &particles,
              remaining_dt,
            )?;
            kernels.compute_self_gravity(&mut cmd, &final_bvh, &mut particles)?;
            kernels.imex_integrate_particles_p4_5(
              &mut cmd,
              &mut particles,
              remaining_dt,
              current_time + t_c,
            )?;
            kernels.discard_bvh(final_bvh);
          }
          current_time = end_time; // CCD complete for this frame
        } else {
          let advance = if t_c == 0 { 1 } else { t_c };
          current_time += advance;
        }
      } else {
        // No collision before dt — accept the step.
        kernels.discard_list(compacted);
        kernels.discard_list(rb_rb_pairs);
        current_time = end_time;
      }
    }
  }

  kernels.write_back_to_scene(&mut cmd, &rigid_bodies, &particles, &particle_metadata, physical_scene, scene)?;

  // Discard per-frame buffers via the timeline-safe discard pool.
  kernels.discard_buffer(kinematics);
  kernels.discard_buffer(rigid_bodies);
  kernels.discard_buffer(wrenches);
  kernels.discard_buffer(particles);
  kernels.discard_buffer(emitters);
  kernels.discard_buffer(frames);

  Ok(())
}