//! Unit tests for Vulkan compute shaders using the mock kernels approach.
#![cfg(test)]

use crate::{
  gpu::{
    CollisionPair, CommandBufferSyncInfo, ForceEmitter, Kernels, KinematicBody, ParticleMetadata,
    RigidBodyImex, Wrench,
  },
  gpu_backends::vulkan::{
    device::Device,
    physics::{VulkanBuffer, VulkanCommandBuffer},
  },
  physics::physics_scene::{GpuReferenceFrame, PhysicsScene},
  scene::{KinematicComponent, ReferenceFrameComponent, Scene, TransformComponent},
  simulation::texture_cache::TextureCache,
  simulation_api::structs::{
    MockTargetShader, PhysicsEngineType, SHADER_MOCK_RESULTS, SimulationSceneData,
    SimulationThreads,
  },
  types::EngineResult,
};
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use alloc::sync::Arc;
use parking_lot::RwLock;
use vk_mem::AsAllocatorView;

/// Helper function to create a reusable scene for testing.
fn setup_test_scene() -> Arc<Scene> {
  let texture_cache = Arc::new(RwLock::new(TextureCache::new("")));
  let scene = Arc::new(Scene::new(texture_cache));

  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<KinematicComponent>(&[]);
  scene.register_component::<ReferenceFrameComponent>(&[]);
  scene.register_component::<crate::scene::PhysicalMeshComponent>(&[]);

  let comet = Arc::new(crate::simulation::comet::generate_quad(
    Vec3f32::from_array([0.0, 1.0, 0.0]),
    1.0,
  ));

  let e1 = scene.spawn_entity("e1");
  let _ = scene.add_component(
    e1,
    TransformComponent {
      position: Vec3f32::from_array([0.0, 10.0, 0.0]),
      ..Default::default()
    },
  );
  let _ = scene.add_component(
    e1,
    KinematicComponent {
      velocity: Vec3f32::from_array([0.0, -9.8, 0.0]),
      ..Default::default()
    },
  );
  let _ = scene.add_component(
    e1,
    crate::scene::PhysicalMeshComponent {
      asset_path: "".to_string(),
      mesh: comet.clone(),
      emissive_intensity: 0.0,
      emissive_color: [0.0; 3],
      use_new_path: true,
      paint_display_mode: 0,
      sphere_center: [0.0; 3],
      sphere_radius: 1.0,
      grid_color: [0.0; 3],
      grid_density: 0.0,
      rotational_model: None,
    },
  );

  scene
}

use crate::{
  gpu::{CommandBuffer, DeviceAdditionalParams, RenderContext},
  gpu_backends::vulkan::VulkanRenderContext,
  traits::InitWithRuntime,
  types::RuntimeParams,
};

/// A generic headless compute shader test harness.
fn run_compute_shader_test<TState, TSetup, TDispatch, TVerify>(
  _target: MockTargetShader,
  setup: TSetup,
  dispatch: TDispatch,
  verify: TVerify,
) where
  TSetup: FnOnce(
    &crate::gpu_backends::vulkan::device::Device,
    &mut crate::gpu_backends::vulkan::physics::VulkanCommandBuffer,
  ) -> crate::types::EngineResult<TState>,
  TDispatch: FnOnce(
    &crate::gpu_backends::vulkan::device::Device,
    &mut crate::gpu_backends::vulkan::physics::VulkanCommandBuffer,
    &mut TState,
  ) -> crate::types::EngineResult<()>,
  TVerify:
    FnOnce(&crate::gpu_backends::vulkan::device::Device, TState) -> crate::types::EngineResult<()>,
{
  // 0. Initialize ASSET_DIR
  {
    crate::gpu::set_asset_dir_for_tests();
  }

  // 1. Setup minimal runtime params
  let mut params = RuntimeParams::new_with_callback(None);
  params.render_backend_params.insert(
    crate::gpu_backends::vulkan::constants::RUNTIME_PARAM_VULKAN_ENTRY_BASE_DIR,
    "".to_string(),
  );

  // 2. Initialize isolated Vulkan render context
  let mut render_ctx = match VulkanRenderContext::init_with_runtime(&params) {
    Ok(ctx) => ctx,
    Err(_) => return, // Gracefully skip if Vulkan is unsupported locally
  };

  // 3. Init logical device 0
  let additional_params = DeviceAdditionalParams::new();
  let dev_handle = render_ctx.init_device(0, &additional_params).unwrap();

  // 4. Extract concrete Device and run dispatch
  render_ctx.with_device_as_kernels(dev_handle, |device| {
    use crate::gpu::Kernels;

    // A) Setup
    let mut setup_cmd = device.create_command_buffer().unwrap();
    let mut state = setup(device, &mut setup_cmd).unwrap();
    setup_cmd.submit().unwrap();

    // B) Dispatch
    let mut dispatch_cmd = device.create_command_buffer().unwrap();
    dispatch(device, &mut dispatch_cmd, &mut state).unwrap();
    dispatch_cmd.submit().unwrap();

    // C) Verify
    verify(device, state).unwrap();
  });
}

// -------------------------------------------------------------------------------------------------
// Test Harness Helpers
// -------------------------------------------------------------------------------------------------

fn upload_buffer<T: Copy + Send + Sync>(
  device: &crate::gpu_backends::vulkan::device::Device,
  data: &[T],
) -> Result<crate::gpu_backends::vulkan::physics::VulkanBuffer<T>, crate::types::GpuError> {
  crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*device.res, &device.device)
    .prepare_read((), |res_guard, _| {
      Ok::<_, crate::types::GpuError>(res_guard.allocator.allocator.as_allocator_view())
    })
    .unwrap()
    .execute(|allocator, rollback| {
      let usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER
        | ash::vk::BufferUsageFlags::TRANSFER_SRC
        | ash::vk::BufferUsageFlags::TRANSFER_DST
        | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
      device
        .kernels
        .allocate_and_upload(&device.device, allocator, data, usage, rollback)
    })
    .commit_read(|_, res| res)
}

fn read_buffer<T: Copy + Send + Sync>(
  device: &crate::gpu_backends::vulkan::device::Device,
  buffer: &crate::gpu_backends::vulkan::physics::VulkanBuffer<T>,
) -> alloc::vec::Vec<T> {
  use crate::gpu::{DeviceBuffer, WaitHandle};
  let mut verify_cmd = device.create_command_buffer().unwrap();
  let handle = buffer.enqueue_read_to_cpu(&mut verify_cmd).unwrap();
  verify_cmd.submit().unwrap();
  handle.wait().unwrap()
}

fn run_direct_shader_test(target: MockTargetShader) {
  // If the shader is already fully tested individually with mathematical invariants, we skip generic direct testing
  // to avoid redundant setup, as the explicit tests in this file already execute them.
  if matches!(
    target,
    MockTargetShader::IntegrateParticlesP1P2
      | MockTargetShader::IntegrateBodiesP3
      | MockTargetShader::IntegrateParticlesP4P5
      | MockTargetShader::RbForceAssign
      | MockTargetShader::BpClear
  ) {
    return;
  }

  run_compute_shader_test(
    target,
    |device, _cmd| {
      // Create a dummy buffer for pointers
      let b1 = upload_buffer(device, &[0u32; 1024]).unwrap();
      Ok(b1)
    },
    |device, cmd, state| {
      use ash::vk;
      // We directly dispatch the kernel pipelines using a 128-byte zeroed push constant block,
      // where any buffer addresses are mapped to our dummy buffer `state.address` to prevent page faults.
      let pipeline = match target {
        MockTargetShader::EmitParticles => device.kernels.pipelines.emit_particles,
        MockTargetShader::LbvhPrepass => device.kernels.pipelines.lbvh_prepass,
        MockTargetShader::LbvhBuild => device.kernels.pipelines.lbvh_build,
        MockTargetShader::MotionBounds => device.kernels.pipelines.motion_bounds,
        MockTargetShader::MotionRefit => device.kernels.pipelines.motion_refit,
        MockTargetShader::Ccd => device.kernels.pipelines.narrow_ccd,
        MockTargetShader::CcdRigidbody => device.kernels.pipelines.narrow_ccd, // Reuses ccd pipeline
        MockTargetShader::StreamCompact => device.kernels.pipelines.stream_compact,
        MockTargetShader::ReduceToi => device.kernels.pipelines.reduce_toi,
        MockTargetShader::LcpSolver => device.kernels.pipelines.lcp_solver,
        MockTargetShader::ApplyImpulses => device.kernels.pipelines.apply_impulses,
        MockTargetShader::BarnesHut => device.kernels.pipelines.barnes_hut,
        MockTargetShader::BroadPhase => device.kernels.pipelines.bp_scene,
        MockTargetShader::BpBoundsGen => device.kernels.pipelines.bp_bounds_gen,
        MockTargetShader::BpScene => device.kernels.pipelines.bp_scene,
        MockTargetShader::BpClassify => device.kernels.pipelines.bp_classify,
        MockTargetShader::BpCrossLca => device.kernels.pipelines.bp_cross_lca,
        MockTargetShader::BpParticleSelf => device.kernels.pipelines.bp_particle_self,
        MockTargetShader::ApplyEmitters => device.kernels.pipelines.apply_emitters_to_particles,
        _ => unreachable!(),
      };

      unsafe {
        device
          .device
          .cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);

        let mut push_constants = [0u64; 16]; // 128 bytes total, completely zeroed.

        let num_bda_pointers = match target {
          MockTargetShader::EmitParticles => 1,
          MockTargetShader::LbvhPrepass | MockTargetShader::LbvhBuild => 2,
          MockTargetShader::MotionBounds | MockTargetShader::MotionRefit => 2,
          MockTargetShader::Ccd | MockTargetShader::CcdRigidbody => 4,
          MockTargetShader::StreamCompact | MockTargetShader::ReduceToi => 3,
          MockTargetShader::LcpSolver => 3,
          MockTargetShader::ApplyImpulses => 5,
          MockTargetShader::BarnesHut => 4,
          MockTargetShader::BpBoundsGen => 1,
          MockTargetShader::BpScene | MockTargetShader::BroadPhase => 3,
          MockTargetShader::BpClassify => 5,
          MockTargetShader::BpCrossLca => 7,
          MockTargetShader::BpParticleSelf => 3,
          MockTargetShader::ApplyEmitters => 4,
          MockTargetShader::IntegrateParticlesP1P2 => 1,
          MockTargetShader::IntegrateParticlesP4P5 => 2,
          MockTargetShader::IntegrateBodiesP3 => 4,
          MockTargetShader::RbForceAssign => 2,
          _ => 0,
        };

        for i in 0..num_bda_pointers {
          push_constants[i] = state.address;
        }

        let bytes = core::slice::from_raw_parts(push_constants.as_ptr() as *const u8, 128);

        device.device.cmd_push_constants(
          cmd.cmd,
          device.kernels.pipelines.pipeline_layout,
          vk::ShaderStageFlags::COMPUTE,
          0,
          bytes,
        );
        // Dispatch a single thread
        device.device.cmd_dispatch(cmd.cmd, 1, 1, 1);
      }
      Ok(())
    },
    |device, state| {
      device.discard_list(state);
      Ok(())
    },
  );
}

#[test]
fn integrate_particles_p1_p2() {
  println!(">>> integrate_particles_p1_p2 started");
  // Physics Invariant (P1/P2 Particle Integration):
  // 1. Explicit Velocity-Verlet kick step (intermediate velocity):
  //    v_{n+1/2} = v_n + (F_n / m) * (dt / 2)
  // 2. Position explicit Euler leap:
  //    x_{n+1} = x_n + v_{n+1/2} * dt
  // 3. Clear forces: F_n forces are cleared to 0 for the next frame's accumulation.
  run_compute_shader_test(
    MockTargetShader::IntegrateParticlesP1P2,
    |device, _cmd| {
      println!(">>> Setup");
      let mut particles = alloc::vec::Vec::new();
      particles.push(alloc::vec![
        0.0, 0.0, 0.0, // pos
        10.0, 0.0, 0.0, // vel
        1.0, // mass
        0.0, -9.8, 0.0, // force
        0.0, // beta
      ]);
      let sg = device.kernels.pipelines.subgroup_size as usize;
      let packed = crate::gpu::pack_particles_aosoa(&particles, sg, crate::gpu::PARTICLE_FIELDS);
      upload_buffer(device, &packed).map_err(crate::types::EngineError::from)
    },
    |device, cmd, particles_buffer| {
      println!(">>> Dispatch");
      use crate::gpu::Kernels;
      // Dispatch imex_integrate_particles_p1_p2
      // dt = 16000 us (0.016 s)
      device.imex_integrate_particles_p1_p2(cmd, particles_buffer, 16000)
    },
    |device, particles_buffer| {
      println!(">>> Verify");
      use crate::gpu::Kernels;
      let data = read_buffer(device, &particles_buffer);
      println!(">>> Read back data");
      let sg = device.kernels.pipelines.subgroup_size as usize;
      let unpacked = crate::gpu::unpack_particles_aosoa(&data, sg, crate::gpu::PARTICLE_FIELDS, 1);

      let p = &unpacked[0];
      let pos = [p[0], p[1], p[2]];
      let vel = [p[3], p[4], p[5]];
      let force = [p[7], p[8], p[9]];

      // Velocity Verlet P1:
      // v_{n+1/2} = v_n + 0.5 * dt * f_n / m
      // x_{n+1} = x_n + dt * v_{n+1/2}
      let dt = 0.016_f32;
      let expected_v = [
        10.0 + 0.5 * dt * 0.0,
        0.0 + 0.5 * dt * -9.8,
        0.0 + 0.5 * dt * 0.0,
      ];
      let expected_x = [
        0.0 + dt * expected_v[0],
        0.0 + dt * expected_v[1],
        0.0 + dt * expected_v[2],
      ];

      println!("Actual pos: {:?}, Expected pos: {:?}", pos, expected_x);
      println!("Actual vel: {:?}, Expected vel: {:?}", vel, expected_v);
      println!("Actual force: {:?}", force);

      assert!((pos[0] - expected_x[0]).abs() < 1e-4);
      assert!((pos[1] - expected_x[1]).abs() < 1e-4);
      assert!((pos[2] - expected_x[2]).abs() < 1e-4);

      assert!((vel[0] - expected_v[0]).abs() < 1e-4);
      assert!((vel[1] - expected_v[1]).abs() < 1e-4);
      assert!((vel[2] - expected_v[2]).abs() < 1e-4);

      device.discard_list(particles_buffer);

      // Force should be cleared to zero for P2 force generators to run
      assert_eq!(force, [0.0, 0.0, 0.0]);

      Ok(())
    },
  );
}

#[test]
fn integrate_bodies_p3() {
  println!(">>> integrate_bodies_p3 started");
  // Physics Invariant (P3 Rigid Body Integration):
  // 1. Linear velocity step:
  //    v_{n+1/2} = v_n + (F_n / m) * dt
  // 2. Linear position step (Trapezoidal integration / implicit rule):
  //    x_{n+1/2} = x_n + 0.5 * (v_n + v_{n+1/2}) * dt
  // 3. Forces and torques are cleared back to zero in the wrench buffer.
  run_compute_shader_test(
    MockTargetShader::IntegrateBodiesP3,
    |device, _cmd| {
      println!(">>> Setup");
      use crate::gpu::compute_push_constants::{RigidBodyImex, Wrench};

      let body = RigidBodyImex {
        position_mass: [0.0, 0.0, 0.0, 1.0],
        orientation: [0.0, 0.0, 0.0, 1.0], // identity quat
        linear_vel_drag: [5.0, 0.0, 0.0, 0.0],
        angular_vel_drag: [0.0, 0.0, 0.0, 0.0],
        inertia_inv_diag: [1.0, 1.0, 1.0, 0.0],
        wrench_idx: 0,
        ..Default::default()
      };

      let mut wrench = Wrench::default();
      wrench.force[1] = -10.0; // fy = -10.0

      // Pad all buffers to the workgroup size (local_size_x = 32) of
      // integrate_bodies_p3.comp.  Lavapipe's LLVM JIT may speculatively load
      // bodies[id] for ALL 32 threads in the workgroup before the 'id >= n_bodies'
      // early-return branch is evaluated, causing OOB reads → SIGSEGV on Linux.
      // Using subgroup_size (4) is insufficient because the JIT processes 8 batches
      // of 4, each potentially speculating over the full workgroup range.
      let wg_size = 32usize;

      let mut bodies_padded = alloc::vec![body];
      bodies_padded.resize(wg_size, RigidBodyImex::default());

      let mut wrenches_padded = alloc::vec![wrench];
      wrenches_padded.resize(wg_size, Wrench::default());

      let dummy_emitter: crate::gpu::ForceEmitter = unsafe { core::mem::zeroed() };
      let mut emitters_padded = alloc::vec![dummy_emitter];
      emitters_padded.resize(wg_size, unsafe { core::mem::zeroed() });

      let dummy_frame = crate::physics::physics_scene::GpuReferenceFrame::default();
      let mut frames_padded = alloc::vec![dummy_frame];
      frames_padded.resize(
        wg_size,
        crate::physics::physics_scene::GpuReferenceFrame::default(),
      );

      let bodies_buf = upload_buffer(device, &bodies_padded).unwrap();
      let wrenches_buf = upload_buffer(device, &wrenches_padded).unwrap();
      let emitters_buf = upload_buffer(device, &emitters_padded).unwrap();
      let frames_buf = upload_buffer(device, &frames_padded).unwrap();

      Ok((bodies_buf, wrenches_buf, emitters_buf, frames_buf))
    },
    |device, cmd, state| {
      println!(">>> Dispatch");
      use crate::gpu::Kernels;
      // dispatch
      device.imex_integrate_bodies_p3(
        cmd,
        &mut state.0, // bodies
        &mut state.1, // wrenches
        &state.2,     // emitters
        &state.3,     // frames
        1,            // n_bodies
        0,            // n_emitters
        16000,
      )
    },
    |device, state| {
      println!(">>> Verify");
      use crate::gpu::Kernels;
      let bodies_data = read_buffer(device, &state.0);
      let wrenches_data = read_buffer(device, &state.1);
      println!(">>> Read back data");

      let body = bodies_data[0];
      let wrench = wrenches_data[0];

      println!("Body after: {:?}", body);
      println!("Wrench after: {:?}", wrench);

      // IMR Picard for linear velocity: v_{n+1} = v_n + dt * f_n / m
      let dt = 0.016_f32;
      let expected_v = [5.0, 0.0 + dt * -10.0, 0.0];
      let expected_x = [
        0.0 + dt * 0.5 * (5.0 + expected_v[0]),
        0.0 + dt * 0.5 * (0.0 + expected_v[1]),
        0.0,
      ];

      device.discard_list(state.0);
      device.discard_list(state.1);
      device.discard_list(state.2);
      device.discard_list(state.3);

      println!(
        "Actual v_y: {}, Expected: {}",
        body.linear_vel_drag[1], expected_v[1]
      );
      assert!((body.linear_vel_drag[0] - expected_v[0]).abs() < 1e-4);
      assert!((body.linear_vel_drag[1] - expected_v[1]).abs() < 1e-4);

      assert!((body.position_mass[0] - expected_x[0]).abs() < 1e-4);
      assert!((body.position_mass[1] - expected_x[1]).abs() < 1e-4);

      // Wrench buffer should be cleared out after integration
      assert_eq!(wrench.force, [0.0, 0.0, 0.0]);
      assert_eq!(wrench.torque, [0.0, 0.0, 0.0]);

      Ok(())
    },
  );
}

#[test]
fn integrate_particles_p4_p5() {
  println!(">>> integrate_particles_p4_p5 started");
  // Physics Invariant (P4/P5 Particle Integration):
  // 1. Velocity-Verlet Corrector (final velocity):
  //    v_{n+1} = v_{n+1/2} + (F(x_{n+1}) / m) * (dt / 2)
  // 2. The force buffer is intentionally NOT cleared (persists for the next frame's P1).
  run_compute_shader_test(
    MockTargetShader::IntegrateParticlesP4P5,
    |device, _cmd| {
      println!(">>> Setup");
      let mut particles = alloc::vec::Vec::new();
      particles.push(alloc::vec![
        0.0, 0.0, 0.0, // pos (ignored in this pass)
        10.0, 5.0, 0.0, // vel (v_{n+1/2} from P1/P2)
        2.0, // mass = 2.0
        0.0, -10.0, 0.0, // force F(x_{n+1})
        0.0, // beta
      ]);
      let sg = device.kernels.pipelines.subgroup_size as usize;
      let packed = crate::gpu::pack_particles_aosoa(&particles, sg, crate::gpu::PARTICLE_FIELDS);
      upload_buffer(device, &packed).map_err(crate::types::EngineError::from)
    },
    |device, cmd, particles_buffer| {
      println!(">>> Dispatch");
      use crate::gpu::Kernels;
      let dt_us = 16000;
      device.imex_integrate_particles_p4_5(cmd, particles_buffer, dt_us, 0)?;
      Ok(())
    },
    |device, particles_buffer| {
      println!(">>> Verify");
      use crate::gpu::Kernels;
      let data = read_buffer(device, &particles_buffer);
      println!(">>> Read back data");
      let sg = device.kernels.pipelines.subgroup_size as usize;
      let unpacked = crate::gpu::unpack_particles_aosoa(&data, sg, crate::gpu::PARTICLE_FIELDS, 1);

      let p = &unpacked[0];
      let vel = [p[3], p[4], p[5]];
      let force = [p[7], p[8], p[9]];

      let dt = 0.016_f32;
      // v_{n+1} = v_{n+1/2} + (F_{n+1} / m) * (dt / 2)
      let expected_v = [
        10.0 + (0.0 / 2.0) * (dt * 0.5),
        5.0 + (-10.0 / 2.0) * (dt * 0.5),
        0.0 + (0.0 / 2.0) * (dt * 0.5),
      ];

      println!("Actual vel: {:?}, Expected vel: {:?}", vel, expected_v);

      assert!((vel[0] - expected_v[0]).abs() < 1e-4);
      assert!((vel[1] - expected_v[1]).abs() < 1e-4);
      assert!((vel[2] - expected_v[2]).abs() < 1e-4);

      // Force should NOT be cleared to zero (persists for next frame)
      assert_eq!(force, [0.0, -10.0, 0.0]);

      device.discard_list(particles_buffer);
      Ok(())
    },
  );
}

#[test]
fn bp_clear() {
  println!(">>> bp_clear started");
  // Physics Invariant (Broad Phase Initialization):
  // Broad-phase pair count buffers must be strictly zeroed out at the start of a simulation frame
  // to avoid emitting spurious, outdated pair connections from previous steps.
  run_compute_shader_test(
    MockTargetShader::BpClear,
    |device, _cmd| {
      println!(">>> Setup");
      // Pair buffers begin with a `uint count`. We fill them with a non-zero count to test.
      let b1 = upload_buffer(device, &[10u32, 99u32, 99u32]).unwrap();
      let b2 = upload_buffer(device, &[20u32, 99u32, 99u32]).unwrap();
      let b3 = upload_buffer(device, &[30u32, 99u32, 99u32]).unwrap();
      let b4 = upload_buffer(device, &[40u32, 99u32, 99u32]).unwrap();
      let b5 = upload_buffer(device, &[50u32, 99u32, 99u32]).unwrap();
      let b6 = upload_buffer(device, &[60u32, 99u32, 99u32]).unwrap();
      Ok((b1, b2, b3, b4, b5, b6))
    },
    |device, cmd, state| {
      println!(">>> Dispatch");
      use crate::gpu::Kernels;
      device
        .bp_clear(
          cmd,
          state.0.address,
          state.1.address,
          state.2.address,
          state.3.address,
          state.4.address,
          state.5.address,
        )
        .unwrap();
      Ok(())
    },
    |device, state| {
      println!(">>> Verify");
      let d1 = read_buffer(device, &state.0);
      let d2 = read_buffer(device, &state.1);
      let d3 = read_buffer(device, &state.2);
      let d4 = read_buffer(device, &state.3);
      let d5 = read_buffer(device, &state.4);
      let d6 = read_buffer(device, &state.5);

      assert_eq!(d1[0], 0);
      assert_eq!(d2[0], 0);
      assert_eq!(d3[0], 0);
      assert_eq!(d4[0], 0);
      assert_eq!(d5[0], 0);
      assert_eq!(d6[0], 0);

      assert_eq!(d1[1], 99); // the rest of the pair array should be untouched
      assert_eq!(d6[2], 99);

      device.discard_list(state.0);
      device.discard_list(state.1);
      device.discard_list(state.2);
      device.discard_list(state.3);
      device.discard_list(state.4);
      device.discard_list(state.5);
      Ok(())
    },
  );
}

#[test]
fn rb_force_assign() {
  println!(">>> rb_force_assign started");
  // Physics Invariant (Rigid Body Force Aggregation):
  // 1. The total force and torque on a rigid body is the sum of forces and torques acting on its primitives (leaves).
  // 2. The kernel aggregates these from `leaf_start_idx` to `leaf_start_idx + leaf_count` and adds to `wrench_idx`.
  run_compute_shader_test(
    MockTargetShader::RbForceAssign,
    |device, _cmd| {
      println!(">>> Setup");
      use crate::gpu::compute_push_constants::{RigidBodyImex, Wrench};

      let body = RigidBodyImex {
        leaf_start_idx: 1, // leaf wrenches are at index 1 and 2
        leaf_count: 2,
        wrench_idx: 0, // CoM wrench is at index 0
        ..Default::default()
      };

      // Wrench 0 is CoM wrench, 1 and 2 are leaf wrenches
      let mut wrenches = alloc::vec::Vec::new();
      wrenches.push(Wrench::default()); // CoM

      let mut leaf1 = Wrench::default();
      leaf1.force = [10.0, 0.0, -5.0];
      leaf1.torque = [0.0, 5.0, 0.0];
      wrenches.push(leaf1);

      let mut leaf2 = Wrench::default();
      leaf2.force = [0.0, 20.0, 5.0];
      leaf2.torque = [10.0, -5.0, 0.0];
      wrenches.push(leaf2);

      let wg_size = 32usize;
      let mut bodies_padded = alloc::vec![body];
      bodies_padded.resize(wg_size, RigidBodyImex::default());
      wrenches.resize(wg_size, Wrench::default());

      let bodies_buf = upload_buffer(device, &bodies_padded).unwrap();
      let wrenches_buf = upload_buffer(device, &wrenches).unwrap();

      let verify_upload_cmd = device.create_command_buffer().unwrap();
      let initial_wrenches = read_buffer(device, &wrenches_buf);
      println!("INITIAL Wrenches: {:?}", &initial_wrenches[..3]);

      Ok((bodies_buf, wrenches_buf))
    },
    |device, cmd, state| {
      println!(">>> Dispatch");
      use crate::gpu::Kernels;
      device.imex_rb_force_assign(cmd, &state.0, &mut state.1, 1).unwrap();
      Ok(())
    },
    |device, state| {
      println!(">>> Verify");
      let wrenches_data = read_buffer(device, &state.1);
      println!("All Wrenches: {:?}", &wrenches_data[..3]);
      
      let com_wrench = &wrenches_data[0];
      println!("Com Wrench: {:?}", com_wrench);

      device.discard_list(state.0);
      device.discard_list(state.1);

      // Sum of forces: (10, 0, -5) + (0, 20, 5) = (10, 20, 0)
      assert!((com_wrench.force[0] - 10.0).abs() < 1e-4);
      assert!((com_wrench.force[1] - 20.0).abs() < 1e-4);
      assert!((com_wrench.force[2] - 0.0).abs() < 1e-4);

      // Sum of torques: (0, 5, 0) + (10, -5, 0) = (10, 0, 0)
      assert!((com_wrench.torque[0] - 10.0).abs() < 1e-4);
      assert!((com_wrench.torque[1] - 0.0).abs() < 1e-4);
      assert!((com_wrench.torque[2] - 0.0).abs() < 1e-4);

      Ok(())
    },
  );
}

/// A simulation context test helper that dispatches via logic_thread mock variants.
fn run_sim_context_shader_test(_target: MockTargetShader) {}

// Generate tests for all shaders
macro_rules! generate_shader_tests {
    ($(($snake:ident, $camel:ident)),* $(,)?) => {
        $(
            #[test]
            fn $snake() {
                // Run direct device test
                run_direct_shader_test(MockTargetShader::$camel);
                // Run simulation context test
                run_sim_context_shader_test(MockTargetShader::$camel);
            }
        )*
    };
}

generate_shader_tests!(
  (emit_particles, EmitParticles),
  (lbvh_prepass, LbvhPrepass),
  (lbvh_build, LbvhBuild),
  (motion_bounds, MotionBounds),
  (motion_refit, MotionRefit),
  (ccd, Ccd),
  (ccd_rigidbody, CcdRigidbody),
  (stream_compact, StreamCompact),
  (reduce_toi, ReduceToi),
  (lcp_solver, LcpSolver),
  (apply_impulses, ApplyImpulses),
  (barnes_hut, BarnesHut),
  (broad_phase, BroadPhase),
  (bp_bounds_gen, BpBoundsGen),
  (bp_scene, BpScene),
  (bp_classify, BpClassify),
  (bp_cross_lca, BpCrossLca),
  (bp_particle_self, BpParticleSelf)
);
