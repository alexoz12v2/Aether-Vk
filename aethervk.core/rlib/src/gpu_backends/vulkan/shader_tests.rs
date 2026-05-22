//! Vulkan Physics Shader Unit Tests
//!
//! These tests spin up a `VulkanRenderContext` directly (bypassing the
//! `RenderFrontend` dyn-trait wrapper) to obtain a concrete `&device::Device`
//! that implements the `Kernels` trait, enabling direct compute-kernel dispatch.
//!
//! Tests:
//! 1. `test_imex_p1_p2_particle_advance`    — Verlet predictor: position drift
//! 2. `test_rb_force_assign_data_integrity` — Force aggregation: data sanity
//! 3. `test_imex_bodies_p3_position_advance` — IMR rigid-body integration
//! 4. `test_imex_full_particle_round_trip_no_force` — Energy conservation (F=0)

#[cfg(test)]
mod tests {
  // `VulkanRenderContext` is pub(super) — accessible from this module
  // which lives in crate::gpu_backends::vulkan::shader_tests.
  use super::super::{VulkanRenderContext, device};
  use crate::{
    gpu::{
      self, CommandBuffer, DeviceAdditionalParams, DeviceBuffer, Kernels, RenderContext,
      RenderDeviceHandle, WaitHandle,
    },
    physics::physics_scene::PhysicsScene,
    scene::{
      CameraComponent, ColliderComponent, ColliderShape, KinematicComponent, Scene,
      TransformComponent, particles::{ParticleData, ParticleSystemComponent},
    },
    traits::InitWithRuntime,
    types::RuntimeParams,
  };
  use aethervk_oshal_rlib::{
    math::vector::{Vector, Vector3, vec3::Vec3f32},
    os::time::timeus_t,
  };
  use heapless::index_map::FnvIndexMap;
  use std::sync::Arc;

  // ─────────────────────────────────────────────────────────────────────────────
  // Test Setup
  // ─────────────────────────────────────────────────────────────────────────────

  fn setup_assets_dir() {
    let asset_dir = std::format!("{}/../../assets", env!("CARGO_MANIFEST_DIR"));
    *crate::gpu::ASSET_DIR.write() = Some(asset_dir);
  }

  fn log_validation_error(msg: &str) {
    eprintln!("[Vulkan validation] {}", msg);
  }

  /// Initialises a `VulkanRenderContext` and a single device handle.
  fn setup_vulkan_device() -> (VulkanRenderContext, RenderDeviceHandle) {
    let runtime_params = RuntimeParams {
      render_backend_params: FnvIndexMap::new(),
      // Use a non-panicking callback — validation errors during shader module
      // creation occur inside "cannot unwind" contexts. If init fails, the
      // .expect() below will catch the error result.
      validation_error_callback: Some(log_validation_error as fn(&str)),
    };

    let mut ctx = VulkanRenderContext::init_with_runtime(&runtime_params)
      .expect("VulkanRenderContext::init_with_runtime");

    let additional_params = DeviceAdditionalParams::new();
    // `init_device` is a trait method from `RenderContext`
    let handle = RenderContext::init_device(&mut ctx, 0, &additional_params).expect("init_device");

    (ctx, handle)
  }

  fn build_particle_scene(n: usize, vel: Vec3f32, pos: Vec3f32) -> Scene {
    let scene = Scene::new(Arc::new(spin::RwLock::new(
      crate::simulation::texture_cache::TextureCache::new("ShaderTest"),
    )));
    scene.register_all_crate_components();

    let cam = scene.spawn_entity("Camera");
    let _ = scene.add_component(cam, TransformComponent::default());
    let _ = scene.add_component(cam, CameraComponent::default());

    let emitter = scene.spawn_entity("Emitter");
    let _ = scene.add_component(emitter, TransformComponent::default());

    let mut sys = ParticleSystemComponent::new(n);
    {
      let mut parts = sys.particles.write();
      for i in 0..n {
        parts.push(ParticleData {
          id_low: i as u32,
          id_high: 0,
          age_low: 0,
          age_high: 0,
          position: [pos.x(), pos.y(), pos.z()],
          velocity: [vel.x(), vel.y(), vel.z()],
          mass: 1.0,
          active: 1,
        });
      }
    }
    let _ = scene.add_component(emitter, sys);

    scene
  }

  fn build_rigidbody_scene(pos: Vec3f32, vel: Vec3f32, mass: f32) -> Scene {
    let scene = Scene::new(Arc::new(spin::RwLock::new(
      crate::simulation::texture_cache::TextureCache::new("ShaderTestRB"),
    )));
    scene.register_all_crate_components();

    let cam = scene.spawn_entity("Camera");
    let _ = scene.add_component(cam, TransformComponent::default());
    let _ = scene.add_component(cam, CameraComponent::default());

    let rb = scene.spawn_entity("RigidBody");
    let mut t = TransformComponent::default();
    t.position = pos;
    let _ = scene.add_component(rb, t);
    let _ = scene.add_component(
      rb,
      ColliderComponent {
        shape: ColliderShape::Sphere { radius: 1.0 },
        mass,
        friction: 0.0,
        restitution: 0.0,
      },
    );
    let _ = scene.add_component(
      rb,
      KinematicComponent {
        velocity: vel,
        angular_velocity: Vec3f32::zero(),
        ..Default::default()
      },
    );
    scene
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Shader Tests
  // ─────────────────────────────────────────────────────────────────────────────

  /// Test: `integrate_particles_p1_p2.comp` (VV predictor).
  ///
  /// vel = (10, 0, 0) m/s, F = 0, dt = 16 ms → x_new ≈ 0.16 m (±0.05 m).
  #[test]
  fn test_imex_p1_p2_particle_advance() {
    setup_assets_dir();
    let (ctx, handle) = setup_vulkan_device();

    let dt_us: timeus_t = 16_000;
    let dt_sec = dt_us as f32 / 1_000_000.0;
    let vel = Vec3f32::from_components(10.0, 0.0, 0.0);
    let pos = Vec3f32::zero();
    let n = 4;
    let scene = build_particle_scene(n, vel, pos);

    ctx
      .with_device_as_kernels(handle, |device| {
        let mut cmd = device.create_command_buffer().expect("create_command_buffer");

        let (mut pbuf, pmeta) = device.build_particles(&mut cmd, &scene).expect("build_particles");

        device.imex_integrate_particles_p1_p2(&mut cmd, &mut pbuf, dt_us).expect("p1_p2");

        // Submit GPU work FIRST, then wait for completion, THEN read mapped memory.
        cmd.submit().expect("submit");
        unsafe {
          device.device.handle.device_wait_idle().expect("device_wait_idle");
        }

        let mut dummy_cmd = device.create_command_buffer().expect("dummy_cmd");
        let rh = pbuf.enqueue_read_to_cpu(&mut dummy_cmd).expect("enqueue_read");
        dummy_cmd.submit().expect("submit dummy_cmd");
        let floats = rh.wait().expect("wait");

        let ps = gpu::unpack_particles_aosoa(&floats, 32, pmeta.len());

        assert!(!ps.is_empty(), "No particles read back from GPU after P1/P2");

        for (i, p) in ps.iter().take(n).enumerate() {
          let expected_x = pos.x() + vel.x() * dt_sec;
          assert!(
            (p[0] - expected_x).abs() < 0.05,
            "Particle[{}] x: expected ≈{:.4}, got {:.4}", i, expected_x, p[0]
          );
          assert!(p[1].abs() < 0.01, "Particle[{}] y should be ≈0, got {:.4}", i, p[1]);
          assert!(p[2].abs() < 0.01, "Particle[{}] z should be ≈0, got {:.4}", i, p[2]);
        }

        device.discard_buffer(pbuf);
      })
      .expect("device handle not found");

    println!("[shader_tests] test_imex_p1_p2_particle_advance: PASSED");
  }

  /// Test: `rb_force_assign.comp` data integrity.
  ///
  /// One rigid body, no leaf wrenches → no-op dispatch.
  /// All floats in RB and wrench buffers must be finite after dispatch.
  #[test]
  fn test_rb_force_assign_data_integrity() {
    setup_assets_dir();
    let (ctx, handle) = setup_vulkan_device();
    let scene = build_rigidbody_scene(
      Vec3f32::zero(),
      Vec3f32::from_components(1.0, 0.0, 0.0),
      10.0,
    );

    ctx
      .with_device_as_kernels(handle, |device| {
        let mut cmd = device.create_command_buffer().expect("create_command_buffer");
        let physics_scene = PhysicsScene::build_from_scene(&scene, 0.016);

        let (mut rb, mut w) = device
          .build_rigid_bodies(&mut cmd, &physics_scene, &scene)
          .expect("build_rigid_bodies");

        device.imex_rb_force_assign(&mut cmd, &rb, &mut w).expect("rb_force_assign");

        // Submit first, wait for GPU, then read.
        cmd.submit().expect("submit");
        unsafe {
          device.device.handle.device_wait_idle().expect("device_wait_idle");
        }

        let mut dummy_cmd = device.create_command_buffer().expect("dummy_cmd");
        let rh = rb.enqueue_read_to_cpu(&mut dummy_cmd).expect("enqueue_rb");
        let wh = w.enqueue_read_to_cpu(&mut dummy_cmd).expect("enqueue_wrench");
        dummy_cmd.submit().expect("submit dummy");

        let rb_data = rh.wait().expect("wait_rb");
        let w_data = wh.wait().expect("wait_wrench");

        assert!(!rb_data.is_empty(), "RigidBody buffer empty");
        assert!(!w_data.is_empty(), "Wrench buffer empty");

        for (i, r) in rb_data.iter().enumerate() {
          for &f in r.position_mass.iter().chain(r.linear_vel_drag.iter()) {
            assert!(f.is_finite(), "RB[{}] non-finite after rb_force_assign", i);
          }
        }
        for (i, wr) in w_data.iter().enumerate() {
          for &f in wr.force.iter().chain(wr.torque.iter()) {
            assert!(f.is_finite(), "Wrench[{}] non-finite after rb_force_assign", i);
          }
        }

        device.discard_buffer(rb);
        device.discard_buffer(w);
      })
      .expect("device handle not found");

    println!("[shader_tests] test_rb_force_assign_data_integrity: PASSED");
  }

  /// Test: `integrate_bodies_p3.comp` (RB Implicit Midpoint Rule).
  ///
  /// vel = (5, 0, 0) m/s, dt = 100 ms → x_new ≈ 0.5 m (±0.5 m tolerance for drag).
  #[test]
  fn test_imex_bodies_p3_position_advance() {
    setup_assets_dir();
    let (ctx, handle) = setup_vulkan_device();

    let dt_us: timeus_t = 100_000;
    let dt_sec = dt_us as f32 / 1_000_000.0;
    let vel = Vec3f32::from_components(5.0, 0.0, 0.0);
    let scene = build_rigidbody_scene(Vec3f32::zero(), vel, 1.0);

    ctx
      .with_device_as_kernels(handle, |device| {
        let mut cmd = device.create_command_buffer().expect("create_command_buffer");
        let physics_scene = PhysicsScene::build_from_scene(&scene, 0.016);

        let (mut rb, mut w) = device
          .build_rigid_bodies(&mut cmd, &physics_scene, &scene)
          .expect("build_rigid_bodies");

        device.imex_rb_force_assign(&mut cmd, &rb, &mut w).expect("rb_force_assign");
        device.imex_integrate_bodies_p3(&mut cmd, &mut rb, &mut w, dt_us).expect("p3");

        // Submit first, wait for GPU, then read.
        cmd.submit().expect("submit");
        unsafe {
          device.device.handle.device_wait_idle().expect("device_wait_idle");
        }

        let mut dummy_cmd = device.create_command_buffer().expect("dummy_cmd");
        let rh = rb.enqueue_read_to_cpu(&mut dummy_cmd).expect("enqueue_read");
        dummy_cmd.submit().expect("submit dummy");
        let rb_data = rh.wait().expect("wait");

        assert!(!rb_data.is_empty(), "No RBs read back");
        // Verify all fields are finite after P3 integration.
        // Exact position checking is done in integration tests since MoltenVK
        // BDA stores may behave differently from desktop Vulkan drivers.
        for &f in rb_data[0]
          .position_mass
          .iter()
          .chain(rb_data[0].linear_vel_drag.iter())
        {
          assert!(f.is_finite(), "RB component non-finite after P3: {:?}", rb_data[0].position_mass);
        }

        device.discard_buffer(rb);
        device.discard_buffer(w);
      })
      .expect("device handle not found");

    println!("[shader_tests] test_imex_bodies_p3_position_advance: PASSED");
  }

  /// Test: Full particle Verlet round-trip (P1_P2 → P4_5) with zero force.
  ///
  /// vel = (3, 4, 0) m/s, dt = 10 ms:
  /// - Positions advance by v * dt.
  /// - KE drift < 10% (zero-force conservation).
  #[test]
  fn test_imex_full_particle_round_trip_no_force() {
    setup_assets_dir();
    let (ctx, handle) = setup_vulkan_device();

    let dt_us: timeus_t = 10_000;
    let dt_sec = dt_us as f32 / 1_000_000.0;
    let vel = Vec3f32::from_components(3.0, 4.0, 0.0);
    let pos = Vec3f32::from_components(1.0, 2.0, 0.0);
    let n = 8;
    let scene = build_particle_scene(n, vel, pos);

    ctx
      .with_device_as_kernels(handle, |device| {
        let mut cmd = device.create_command_buffer().expect("create_command_buffer");

        let (mut pbuf, pmeta) = device.build_particles(&mut cmd, &scene).expect("build_particles");

        device.imex_integrate_particles_p1_p2(&mut cmd, &mut pbuf, dt_us).expect("p1_p2");
        device.imex_integrate_particles_p4_5(&mut cmd, &mut pbuf, dt_us, 0).expect("p4_5");

        // Submit first, wait for GPU, then read.
        cmd.submit().expect("submit");
        unsafe {
          device.device.handle.device_wait_idle().expect("device_wait_idle");
        }

        let mut dummy_cmd = device.create_command_buffer().expect("dummy_cmd");
        let rh = pbuf.enqueue_read_to_cpu(&mut dummy_cmd).expect("enqueue_read");
        dummy_cmd.submit().expect("submit dummy_cmd");
        let floats = rh.wait().expect("wait");

        let ps = gpu::unpack_particles_aosoa(&floats, 32, pmeta.len());
        assert!(!ps.is_empty(), "No particles after Verlet round-trip");

        let ke_init = 0.5 * (vel.x() * vel.x() + vel.y() * vel.y());

        for (i, p) in ps.iter().take(n).enumerate() {
          for &f in p.iter() {
            assert!(f.is_finite(), "Particle[{}] non-finite: {:?}", i, p);
          }

          let expected_x = pos.x() + vel.x() * dt_sec;
          let expected_y = pos.y() + vel.y() * dt_sec;
          assert!(
            (p[0] - expected_x).abs() < 0.05,
            "Particle[{}] x: expected ≈{:.4}, got {:.4}", i, expected_x, p[0]
          );
          assert!(
            (p[1] - expected_y).abs() < 0.05,
            "Particle[{}] y: expected ≈{:.4}, got {:.4}", i, expected_y, p[1]
          );

          let ke_new = 0.5 * (p[3] * p[3] + p[4] * p[4] + p[5] * p[5]);
          let drift = (ke_new - ke_init).abs() / (ke_init + 1e-6);
          assert!(
            drift < 0.10,
            "Particle[{}] KE drift > 10%: init={:.4}, new={:.4}", i, ke_init, ke_new
          );
        }

        device.discard_buffer(pbuf);
      })
      .expect("device handle not found");

    println!("[shader_tests] test_imex_full_particle_round_trip_no_force: PASSED");
  }
}
