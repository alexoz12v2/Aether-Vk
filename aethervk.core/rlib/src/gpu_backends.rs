//! gpu_backends module.

use crate::{
  gpu::{
    CommandBuffer, D3D12_RENDER_BACKEND, DeviceBuffer, Kernels, METAL_RENDER_BACKEND,
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
  let kinematics = kernels.build_kinematic_bodies(&mut cmd, physical_scene, scene)?;
  let mut rigid_bodies = kernels.build_rigid_bodies(&mut cmd, physical_scene, scene)?;
  let mut particles = kernels.build_particles(&mut cmd, scene)?;
  let emitters = kernels.build_emitters(&mut cmd, scene)?;

  let mut sun_pos = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero();
  if let Some((sun_id, _)) =
    scene.query1_first_res::<crate::scene::SunComponent, _, _>(|id, _| Some(id))
  {
    if let Some(pos) = scene.global_transform(sun_id).map(|t| t.position) {
      sun_pos = pos;
    }
  }

  let full_dt = t1 - t0;
  kernels.emit_particles(
    &mut cmd,
    &mut particles,
    physical_scene,
    scene,
    sun_pos,
    full_dt,
  )?;

  if !collisions_enabled {
    let dt = end_time - current_time;
    let _snapshot = kernels.snapshot_dynamics(&mut cmd, &rigid_bodies, &particles)?;

    kernels.step_ode_p1_p2(&mut cmd, &mut particles, dt)?;
    kernels.step_ode_p3_p4(&mut cmd, &kinematics, &mut rigid_bodies, &emitters, dt)?;

    let bvh = kernels.build_motion_bvh(&mut cmd, &kinematics, &rigid_bodies, &particles, dt)?;
    kernels.compute_self_gravity(&mut cmd, &bvh, &mut particles)?;
    kernels.step_ode_p5(&mut cmd, &kinematics, &mut particles, &emitters, dt)?;

    cmd.submit()?;
  } else {
    while current_time < end_time {
      let dt = end_time - current_time;

      let snapshot = kernels.snapshot_dynamics(&mut cmd, &rigid_bodies, &particles)?;

      kernels.step_ode_p1_p2(&mut cmd, &mut particles, dt)?;
      kernels.step_ode_p3_p4(&mut cmd, &kinematics, &mut rigid_bodies, &emitters, dt)?;
      let bvh = kernels.build_motion_bvh(&mut cmd, &kinematics, &rigid_bodies, &particles, dt)?;
      kernels.compute_self_gravity(&mut cmd, &bvh, &mut particles)?;
      kernels.step_ode_p5(&mut cmd, &kinematics, &mut particles, &emitters, dt)?;

      let potentials = kernels.self_intersect_scene(&mut cmd, &bvh)?;
      let globals =
        kernels.intersect_instances(&mut cmd, &potentials, &rigid_bodies, &particles)?;
      let compacted = kernels.compact_collisions(&mut cmd, &globals, time_collision_delta)?;

      let tc_buffer = kernels.find_earliest_collision(&mut cmd, &compacted)?;
      let tc_future = tc_buffer.enqueue_read_to_cpu(&mut cmd)?;

      cmd.submit()?;
      let tc_host = tc_future.wait()?;

      let t_c = tc_host.first().copied().unwrap_or(timeus_t::MAX);
      let inelastic = collision_iters >= MAX_BOUNCES;

      if t_c < dt && !inelastic {
        collision_iters += 1;

        kernels.restore_dynamics(&mut cmd, &mut rigid_bodies, &mut particles, &snapshot)?;

        kernels.step_ode_p1_p2(&mut cmd, &mut particles, t_c)?;
        kernels.step_ode_p3_p4(&mut cmd, &kinematics, &mut rigid_bodies, &emitters, t_c)?;

        let rewind_bvh =
          kernels.build_motion_bvh(&mut cmd, &kinematics, &rigid_bodies, &particles, t_c)?;
        kernels.compute_self_gravity(&mut cmd, &rewind_bvh, &mut particles)?;
        kernels.step_ode_p5(&mut cmd, &kinematics, &mut particles, &emitters, t_c)?;

        kernels.apply_collision_responses(
          &mut cmd,
          &mut rigid_bodies,
          &mut particles,
          &compacted,
          false,
        )?;

        let advance = if t_c == 0 { 1 } else { t_c };
        current_time += advance;
      } else {
        kernels.apply_collision_responses(
          &mut cmd,
          &mut rigid_bodies,
          &mut particles,
          &compacted,
          inelastic,
        )?;
        current_time = end_time;
      }
    }
  }

  kernels.write_back_to_scene(&mut cmd, &rigid_bodies, &particles, physical_scene, scene)?;
  Ok(())
}
