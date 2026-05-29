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
) -> EngineResult<Option<crate::gpu::CommandBufferSyncInfo>>
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
  let dt_s = (t1 - t0) as f32 / 1_000_000.0;
  use crate::physics::tlas_builder::build_scene_motion_tlas;
  let (tlas_bytes, tlas_root_idx) = match kernels.subgroup_size().map(|s| s as u32).unwrap_or(32) {
    64 => build_scene_motion_tlas::<64>(physical_scene),
    16 => build_scene_motion_tlas::<16>(physical_scene),
    _ => build_scene_motion_tlas::<32>(physical_scene),
  };
  aethervk_oshal_rlib::log!(
    "tlas_bytes len: {}, tlas_root_idx: {}",
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
  let (rb_buf, wr_buf) = kernels.build_rigid_bodies(&mut cmd, physical_scene, scene)?;
  let mut rigid_bodies = AutoDiscard::new(rb_buf, |b| kernels.discard_buffer(b));
  let mut wrenches = AutoDiscard::new(wr_buf, |b| kernels.discard_buffer(b));
  let (p_buf, particle_metadata) = kernels.build_particles(&mut cmd, scene)?;
  let mut particles = AutoDiscard::new(p_buf, |b| kernels.discard_buffer(b));
  // Upload per-particle frame indices derived from ParticleMetadata.parent_frame_id.
  // Consumed by apply_emitters_to_particles.comp via BDA.
  let particle_frame_ids = AutoDiscard::new(
    kernels.build_particle_frame_ids(&mut cmd, &particle_metadata)?,
    |b| kernels.discard_buffer(b),
  );
  let emitters = AutoDiscard::new(kernels.build_emitters(&mut cmd, scene)?, |b| {
    kernels.discard_buffer(b)
  });
  // Reference frames for LCA broad-phase (macro frame is always index 0)
  let frames = AutoDiscard::new(kernels.build_frames(&mut cmd, physical_scene)?, |b| {
    kernels.discard_buffer(b)
  });
  aethervk_oshal_rlib::log!("PRINT_ADDR rigid_bodies: 0x{:x}", rigid_bodies.address());
  aethervk_oshal_rlib::log!("PRINT_ADDR frames: 0x{:x}", frames.address());

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
  kernels.emit_particles(
    &mut cmd,
    &mut particles,
    physical_scene,
    scene,
    sun_pos,
    full_dt,
  )?;

  // ── 4. IMEX integration + collision loop ─────────────────────────────────
  #[cfg(not(any(test, feature = "collisions")))]
  let collisions_enabled = false;

  if !collisions_enabled {
    let dt = end_time - current_time;

    // VV predictor: half-kick + full drift to x_{n+1}
    kernels.imex_integrate_particles_p1_p2(&mut cmd, &mut particles, dt)?;

    // RB: accumulate forces then IMR solve
    kernels.imex_rb_force_assign(&mut cmd, &rigid_bodies, &mut wrenches)?;
    kernels.imex_integrate_bodies_p3(
      &mut cmd,
      &mut rigid_bodies,
      &mut wrenches,
      &emitters,
      &frames,
      dt,
    )?;

    // Build motion BVH for self-gravity
    let bvh = AutoDiscard::new(
      kernels.build_motion_bvh(&mut cmd, &kinematics, &rigid_bodies, &particles, dt)?,
      |b| kernels.discard_bvh(b),
    );
    kernels.compute_self_gravity(&mut cmd, &bvh, &mut particles)?;

    // Apply macro-frame gravity emitters to microframe particles (cross-frame transform)
    kernels.apply_emitters_to_particles(
      &mut cmd,
      &mut particles,
      &emitters,
      &frames,
      &particle_frame_ids,
      emitters.capacity() as u32,
    )?;

    // VV corrector: v_{n+1/2} → v_{n+1} using F(x_{n+1})
    kernels.imex_integrate_particles_p4_5(&mut cmd, &mut particles, dt, current_time)?;
  } else {
    #[cfg(any(test, feature = "collisions"))]
    {
      while current_time < end_time {
        let dt = end_time - current_time;

        aethervk_oshal_rlib::log!("gpu_backends.rs: entering while loop");
        let snapshot = AutoDiscard::new(
          kernels.snapshot_dynamics(&mut cmd, &rigid_bodies, &particles)?,
          |s| {
            kernels.discard_buffer(s.0);
            kernels.discard_buffer(s.1);
          },
        );

        aethervk_oshal_rlib::log!("gpu_backends.rs: calling imex_integrate_particles_p1_p2");
        // ── IMEX integration to t_{n+1} ──
        kernels.imex_integrate_particles_p1_p2(&mut cmd, &mut particles, dt)?;
        aethervk_oshal_rlib::log!("gpu_backends.rs: calling imex_rb_force_assign");
        kernels.imex_rb_force_assign(&mut cmd, &rigid_bodies, &mut wrenches)?;
        aethervk_oshal_rlib::log!("gpu_backends.rs: calling imex_integrate_bodies_p3");
        kernels.imex_integrate_bodies_p3(
          &mut cmd,
          &mut rigid_bodies,
          &mut wrenches,
          &emitters,
          &frames,
          dt,
        )?;

        let bvh = AutoDiscard::new(
          kernels.build_motion_bvh(&mut cmd, &kinematics, &rigid_bodies, &particles, dt)?,
          |b| kernels.discard_bvh(b),
        );
        kernels.compute_self_gravity(&mut cmd, &bvh, &mut particles)?;
        // Cross-frame gravity: macro emitters → micro particles
        kernels.apply_emitters_to_particles(
          &mut cmd,
          &mut particles,
          &emitters,
          &frames,
          &particle_frame_ids,
          emitters.capacity() as u32,
        )?;
        kernels.imex_integrate_particles_p4_5(&mut cmd, &mut particles, dt, current_time)?;

        // ── New broad-phase suite ─────────────────────────────────────────────
        // Allocate pair-list buffers (is_list=true → 16-byte header: [x,y,z,count])
        let n_entities = rigid_bodies.capacity() as u32 + (particles.capacity() as u32 / 32);

        // bp_clear: zero all four pair-list counters via raw BDAs
        // (The caller manages the pair-list buffers; for now we reuse bvh.address
        //  as a placeholder TLAS address until a proper TLAS builder is wired in.)
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

        let sparse_collisions = AutoDiscard::new(
          kernels.build_list::<crate::gpu::CollisionPair>(&mut cmd, 10000)?,
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
        kernels.bp_bounds_gen(
          &mut cmd,
          &rigid_bodies,
          query_leaves.address(),
          frames.address(),
          n_entities,
          dt,
        )?;
        kernels.bp_scene(
          &mut cmd,
          tlas_bvh_addr,
          query_leaves.address(),
          raw_pairs.address(),
          tlas_root_idx,
          n_entities,
        )?;
        kernels.bp_classify(
          &mut cmd,
          &rigid_bodies,
          raw_pairs.address(),
          rb_rb_pairs.address(),
          rb_ps_pairs.address(),
          0,                       // out_ps_ps
          rb_lca_pairs.address(),  // out_macro_lca
          0,                       // out_lca_lca
          n_entities * n_entities, // conservative upper bound on raw pairs
        )?;
        // Cross-LCA: only if micro-frames exist (frames.capacity() > 1)
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
            0, // out_ps_ps
            internal_pairs.address(),
            rb_lca_pairs.capacity() as u32,
            2_000,
            rigid_bodies.capacity() as u32,
          )?;
        }
        // Particle self-collision via Hookean spring forces
        let p_copy = ash::vk::BufferCopy::default()
          .size((particles.capacity().max(1) * 10 * core::mem::size_of::<f32>()) as u64);
        let p_addr = particles.address();
        let p_cap = (particles.capacity() as u32 / 320) * 32;
        kernels.bp_particle_self(
          &mut cmd,
          tlas_bvh_addr,
          &mut particles,
          p_addr,
          p_cap,
          0,           // BVH root index
          0.5_f32,     // particle radius (m)
          1_000.0_f32, // spring stiffness k
        )?;

        // ── Merge classified pairs → global collision list ────────────────────
        // 1) Standard CCD for rb_rb_pairs
        kernels.narrow_ccd(
          &mut cmd,
          &*rb_rb_pairs,
          &rigid_bodies,
          &particles,
          frames.address(),
          0,
          (dt as f32) / 1_000_000.0,
          &sparse_collisions,
        )?;

        // 2) Cross-LCA CCD for internal_pairs
        if frames.capacity() > 1 {
          kernels.narrow_ccd_cross_lca(
            &mut cmd,
            &*internal_pairs,
            &rigid_bodies,
            &particles,
            frames.address(),
            1,
            (dt as f32) / 1_000_000.0,
            &sparse_collisions,
          )?;
        }

        let compacted = AutoDiscard::new(
          kernels.compact_collisions(&mut cmd, &sparse_collisions, time_collision_delta)?,
          |b| kernels.discard_list(b),
        );

        let tc_buffer = AutoDiscard::new(
          kernels.find_earliest_collision(
            &mut cmd,
            &compacted,
            time_collision_delta as f32 / 1_000_000.0,
          )?,
          |b| kernels.discard_buffer(b),
        );
        let tc_future = (*tc_buffer).enqueue_read_to_cpu(&mut cmd)?;

        let sync_info = cmd.submit()?;
        if let Some(sync) = sync_info {
          kernels.wait_sync(&sync)?;
        }

        #[cfg(test)]
        {
          let mut read_cmd = kernels.create_command_buffer()?;
          let cmp_fut = (*compacted).enqueue_read_to_cpu(&mut read_cmd)?;
          let rb_fut = rigid_bodies.enqueue_read_to_cpu(&mut read_cmd)?;
          let frames_fut = frames.enqueue_read_to_cpu(&mut read_cmd)?;
          let cross_fut = (*internal_pairs).enqueue_read_to_cpu(&mut read_cmd)?;
          if let Some(read_sync) = read_cmd.submit()? {
            kernels.wait_sync(&read_sync)?;
            if let (Ok(cmp_host), Ok(rb_host), Ok(frames_host), Ok(cross_host)) = (
              cmp_fut.wait(),
              rb_fut.wait(),
              frames_fut.wait(),
              cross_fut.wait(),
            ) {
              let compacted_count = cmp_host.len();
              let cross_count = cross_host.len();
              aethervk_oshal_rlib::log!(
                "READBACK: compacted_count={}, cross_count={}",
                compacted_count,
                cross_count
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
                    id_a,
                    id_b,
                    prim_a,
                    prim_b,
                    pair.time_of_impact
                  );

                  let is_lca = pair.is_lca != 0;
                  let lca_id = if is_lca { Some(pair.lca_id) } else { None };

                  let mut name_a = alloc::string::String::new();
                  // pair.a.entity_id is a dense GPU body index; map to slotmap FFI key
                  if let Some(&entity_a) = body_entity_map.get(id_a as usize) {
                    use slotmap::Key;
                    let ffi_a = entity_a.data().as_ffi() as u64;
                    if let Some(n) = scene.get_name(crate::scene::EntityId::from(
                      slotmap::KeyData::from_ffi(ffi_a),
                    )) {
                      name_a = n;
                    }
                  }
                  if name_a.is_empty() {
                    name_a = alloc::format!("Entity_{}", id_a);
                  }

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
                  if name_b.is_empty() {
                    name_b = alloc::format!("Entity_{}", id_b);
                  }

                  let mut particle_path_a = None;
                  if let Some(&entity_a) = body_entity_map.get(id_a as usize) {
                    if scene
                      .with_component(
                        entity_a,
                        |_: &crate::scene::particles::ParticleSystemComponent| (),
                      )
                      .is_some()
                    {
                      use slotmap::Key;
                      let ffi_a = entity_a.data().as_ffi() as u32;
                      if let Some(p_idx) =
                        physical_scene.particle_entity_map.iter().position(|&ffi| ffi == ffi_a)
                      {
                        if let Some(blas) = &physical_scene.particle_blases[p_idx] {
                          particle_path_a = blas.find_path_to_primitive(prim_a as usize);
                        }
                      }
                    }
                  }

                  let mut particle_path_b = None;
                  if let Some(&entity_b) = body_entity_map.get(id_b as usize) {
                    if scene
                      .with_component(
                        entity_b,
                        |_: &crate::scene::particles::ParticleSystemComponent| (),
                      )
                      .is_some()
                    {
                      use slotmap::Key;
                      let ffi_b = entity_b.data().as_ffi() as u32;
                      if let Some(p_idx) =
                        physical_scene.particle_entity_map.iter().position(|&ffi| ffi == ffi_b)
                      {
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
        }

        let mut next_cmd = kernels.create_command_buffer()?;
        core::mem::swap(&mut cmd, &mut next_cmd);
        old_cmds.push(next_cmd);
        let tc_host = tc_future.wait()?;
        let t_c = tc_host.get(0).copied().unwrap_or(0);
        aethervk_oshal_rlib::log!("gpu_backends.rs: t_c is {}", t_c);

        let t_c = if t_c == 0xFFFFFFFF {
          timeus_t::MAX
        } else {
          let t_c_f32 = f32::from_bits(t_c as u32);
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

          kernels.restore_dynamics(&mut cmd, &mut rigid_bodies, &mut particles, &snapshot)?;

          // Re-integrate to t_c
          kernels.imex_integrate_particles_p1_p2(&mut cmd, &mut particles, t_c)?;
          kernels.imex_rb_force_assign(&mut cmd, &rigid_bodies, &mut wrenches)?;
          kernels.imex_integrate_bodies_p3(
            &mut cmd,
            &mut rigid_bodies,
            &mut wrenches,
            &emitters,
            &frames,
            t_c,
          )?;

          let rewind_bvh = AutoDiscard::new(
            kernels.build_motion_bvh(&mut cmd, &kinematics, &rigid_bodies, &particles, t_c)?,
            |b| kernels.discard_bvh(b),
          );
          kernels.compute_self_gravity(&mut cmd, &rewind_bvh, &mut particles)?;
          kernels.apply_emitters_to_particles(
            &mut cmd,
            &mut particles,
            &emitters,
            &frames,
            &particle_frame_ids,
            emitters.capacity() as u32,
          )?;
          kernels.imex_integrate_particles_p4_5(&mut cmd, &mut particles, t_c, current_time)?;

          // Apply either an elastic or inelastic response at the proper time t_c
          kernels.apply_collision_responses(
            &mut cmd,
            &kinematics,
            &mut rigid_bodies,
            &mut particles,
            &compacted,
            frames.address(),
            inelastic,
          )?;

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
                &emitters,
                &frames,
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
              kernels.apply_emitters_to_particles(
                &mut cmd,
                &mut particles,
                &emitters,
                &frames,
                &particle_frame_ids,
                emitters.capacity() as u32,
              )?;
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
          aethervk_oshal_rlib::log!("gpu_backends.rs: calling kernels.write_back_to_scene!");
          current_time = end_time;
        }
      }
    }
  }

  aethervk_oshal_rlib::log!("gpu_backends.rs: calling kernels.write_back_to_scene OUTSIDE LOOP!");
  let sync_info = kernels.write_back_to_scene(
    &mut cmd,
    &rigid_bodies,
    &particles,
    &particle_metadata,
    physical_scene,
    scene,
  )?;

  Ok(sync_info)
}
