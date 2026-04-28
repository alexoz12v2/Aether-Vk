use crate::{
  gpu::{
    RenderBackendId, RenderFrontend, VULKAN_RENDER_BACKEND, METAL_RENDER_BACKEND,
    D3D12_RENDER_BACKEND,
  },
  traits::InitWithRuntime,
  types::{EngineError, EngineResult, GpuError, RuntimeParams},
};
use alloc::vec::Vec;
use aethervk_oshal_rlib::os::time::{timeus_milliseconds, timeus_t};
use crate::gpu::{CommandBuffer, DeviceBuffer, Kernels, WaitHandle};
use crate::physics::physics_scene::PhysicsScene;
use crate::scene::Scene;

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
pub(super) mod vulkan;

// #[cfg(target_os = "macos")]
// pub(super) mod metal;

// #[cfg(target_os = "windows")]
// pub(super) mod d3d12;

pub(self) const MAX_DEVICES: usize = 4;

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
      if context
        .init_device(0, &crate::gpu::DeviceAdditionalParams::new())
        .is_ok()
      {
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
      if context
        .init_device(0, &crate::gpu::DeviceAdditionalParams::new())
        .is_ok()
      {
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
) -> EngineResult<()>
where
  K: Kernels + ?Sized,
{
  let mut cmd = kernels.create_command_buffer()?;

  let mut current_time = t0;
  let end_time = t1;
  let time_collision_delta = timeus_milliseconds(30);
  let mut collision_iters = 0;
  const MAX_BOUNCES: usize = 5;

  // 1. Build List of Kinematic Bodies, whose step is simulated exclusively on CPU, by using an almanac,
  // which is a SPICE data kernel which should contain the data for the next position given the current time instance
  let kinematics = kernels.build_kinematic_bodies(&mut cmd, physical_scene, scene)?;

  // 2. Build List of Dynamic Bodies (comets and particle systems)
  let mut dynamics = kernels.build_dynamic_bodies(&mut cmd, physical_scene, scene)?;

  while current_time < end_time {
    let dt = end_time - current_time;

    // Snapshot state for Continuous Collision Detection (CCD) rewinding
    let snapshot = kernels.snapshot_dynamics(&mut cmd, &dynamics)?;

    // 3. Given list of kinematic bodies, dynamic bodies, and their positions in the scene,
    // compute all gravitation force pairs between all bodies. Inside particle system, this translates into applying
    // gravitational/position dependent forces to batches of particles (i.e. to a node of BVH whose number of embedded particles is less than a threshold)
    // 4. For Each Dynamic Body, simulate a step for it by using ODE Solver.
    kernels.compute_forces(&mut cmd, &kinematics, &mut dynamics)?;
    kernels.step_ode(&mut cmd, &mut dynamics, dt)?;

    // 5. Collision detection and response loop
    // - Build backend specific motionPhysicalScene, containing a BVH whose leaf nodes are AABB bounding the motion of
    // all bodies in the scene. Then, for each leaf node, we have its own local representation of BVH bounding each motion. Still to decide whether to
    // compute it on the fly given the linear intra-step motion assumption and object frame BVH stored in the scene component, or if storing is better
    // given that kernels potentially execute on GPU, probably store it or use a caching mechanism
    // - self intersect of leaf nodes of scene level. Build a list of potential collisions
    // - intersect the instance level BVHs from the potential collision list. For each intersection we need to store pair of entities intersecting (what if it is a particle inside a particle system? We need a way to identify that)
    // This should build a global collision list
    // - group the global collision list: stream compaction such that, for each pair of objects, we keep only the earliest time collisions (if some collisions are under a time_collision_delta, then keep both)
    // then, for each an object involved in more than one collision, discard all but the earliest one (if some collisions are under a time_collision_delta, we keep both)
    // - compute and apply collision responses and contact forces to all objects
    // - given the earliest collision time $t_c$, rewind the simulation to that time and simulate again till $t_1$
    // after N collisions, all impacts become inelastic collisions, such that we don't get stuck on a loop
    let bvh = kernels.build_motion_bvh(&mut cmd, &dynamics)?;
    let potentials = kernels.self_intersect_scene(&mut cmd, &bvh)?;
    let globals = kernels.intersect_instances(&mut cmd, &potentials)?;
    let compacted = kernels.compact_collisions(&mut cmd, &globals, time_collision_delta)?;

    // Queue a parallel reduction extracting strictly $t_c$ to avoid
    // downloading the massive collisions array over the PCIe bus
    let tc_buffer = kernels.find_earliest_collision(&mut cmd, &compacted)?;
    let tc_future = tc_buffer.enqueue_read_to_cpu(&mut cmd)?;

    // --- SYNCHRONIZATION POINT ---
    // Submit command graph to hardware and yield thread until transfer finishes!
    cmd.submit()?;
    let tc_host = tc_future.wait()?;

    let t_c = tc_host.first().copied().unwrap_or(timeus_t::MAX);
    let inelastic = collision_iters >= MAX_BOUNCES;

    if t_c < dt {
      // Collision occurred mid-step -> Rewind!
      collision_iters += 1;

      kernels.restore_dynamics(&mut cmd, &mut dynamics, &snapshot)?;

      // Re-simulate precisely up to impact
      kernels.compute_forces(&mut cmd, &kinematics, &mut dynamics)?;
      kernels.step_ode(&mut cmd, &mut dynamics, t_c)?;

      // Apply impacts exactly at t_c
      kernels.apply_collision_responses(&mut cmd, &mut dynamics, &compacted, inelastic)?;

      current_time += t_c;
      continue;
    }

    // Stepped cleanly without mid-step collisions
    kernels.apply_collision_responses(&mut cmd, &mut dynamics, &compacted, inelastic)?;
    current_time = end_time;
  }

  // 6. Update Scene and PhysicsScene to reflect changes in the simulation from the backend specific representation
  kernels.write_back_to_scene(&mut cmd, &dynamics, physical_scene, scene)?;
  Ok(())
}
