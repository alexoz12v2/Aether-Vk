use aethervk_core_rlib::{
  gpu::PresentationEngineHandle,
  scene::{
    ui::{ScreenSpaceTextComponent, Transform2DComponent, UiComponent},
    CameraComponent, EntityId, TransformComponent,
  },
  simulation_api::{
    components_api::{CameraParams, PerspectiveCameraParams},
    SimulationContext,
  },
  types::{EngineResult, GpuResult},
};
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, vec4::Quat, Vector, Vector3},
};
use rand::Rng;
use rayon::prelude::*;
use std::sync::Arc;
use test_utils::{
  cycle_get_asset_path_from_exe,
  sim_app::{run_simulation_app, SimulationDelegate},
};
use winit::window::Window;

struct ForceTestDelegate {
  camera_entity: u64,
  quad_entity: Option<EntityId>,
  particle_sys_entity: Option<EntityId>,
  ui_text_entity: Option<EntityId>,
  font_atlas: Option<Arc<aethervk_core_rlib::scene::text::FontAtlas>>,
  font_hash: u64,
  startup_time: std::time::Instant,
}

impl SimulationDelegate for ForceTestDelegate {
  fn create_scene(&mut self, ctx: &mut SimulationContext) -> EngineResult<u64> {
    ctx.create_empty_scene(true)
  }

  fn on_setup(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    pe_handle: PresentationEngineHandle,
    _window: &Window,
  ) -> EngineResult<()> {
    let root_entity = ctx.spawn_entity(scene_id, "root").unwrap();
    ctx
      .add_transform_component(
        scene_id,
        root_entity,
        Vec3f32::zero(),
        Quat::identity(),
        Vec3f32::one(),
      )
      .unwrap();

    let scene_ctx = ctx.get_scene(scene_id).unwrap();
    let scene_ctx_write = scene_ctx.write();
    let scene = &scene_ctx_write.scene;

    // Spawn Quad to initialize hierarchy so get_root works!
    let quad_entity = scene.spawn_entity("quad");
    self.quad_entity = Some(quad_entity);
    let root_entity_id = scene_ctx_write.get_entity(root_entity).unwrap();
    scene.set_parent(quad_entity, Some(root_entity_id));
    scene
      .add_component(
        quad_entity,
        TransformComponent {
          position: Vec3f32::from_components(-8.0, -15.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::one(),
        },
      )
      .unwrap();

    let quad_mesh = aethervk_core_rlib::simulation::comet::generate_quad(
      Vec3f32::from_components(1.0, 0.0, 0.0),
      5.0,
    );
    let quad_mesh_arc = Arc::new(quad_mesh);
    scene
      .add_component(
        quad_entity,
        aethervk_core_rlib::scene::PhysicalMeshComponent {
          asset_path: "procedural_quad".to_string(),
          mesh: quad_mesh_arc.clone(),
          emissive_intensity: -1.0,
          emissive_color: [0.5, 0.5, 0.5],
          use_new_path: true,
          paint_display_mode: 0,
          sphere_center: [0.0, 0.0, 0.0],
          sphere_radius: 1.0,
          grid_color: [0.0, 0.0, 0.0],
          grid_density: 1.0,
        },
      )
      .unwrap();

    scene
      .add_component(
        quad_entity,
        aethervk_core_rlib::scene::KinematicComponent::default(),
      )
      .unwrap();

    drop(scene_ctx_write); // Drop write lock to call add_perspective_camera

    let camera_entity = ctx
      .add_perspective_camera(scene_id, pe_handle, "camera", 45.0, 0.1, 1000.0)
      .unwrap()
      .get();
    ctx.set_parent(scene_id, camera_entity, root_entity).unwrap();
    ctx
      .set_transform_component(
        scene_id,
        camera_entity,
        Vec3f32::from_components(0.0, 0.0, 0.0),
        Quat::identity(),
        Vec3f32::one(),
      )
      .unwrap();
    self.camera_entity = camera_entity;

    // Load font for UI
    let asset_dir = test_utils::cycle_get_asset_path_from_exe(true);
    let font_path = test_utils::get_monospace_font_path_from_asset_path(&asset_dir);
    let font_atlas =
      aethervk_core_rlib::scene::text::FontAtlas::from_path(font_path.to_str().unwrap(), 32.0)
        .unwrap();
    let font_hash = font_atlas.hash_metadata();
    let font_arc = Arc::new(font_atlas);
    self.font_atlas = Some(font_arc.clone());
    self.font_hash = font_hash;

    let scene_ctx_write2 = scene_ctx.write();
    let scene = &scene_ctx_write2.scene;

    // Setup Force Emitter
    let emitter_entity = scene.spawn_entity("force_emitter");
    scene
      .add_component(
        emitter_entity,
        TransformComponent {
          position: Vec3f32::from_components(-8.0, -15.0, 0.0), // Origin
          rotation: Quat::identity(),
          scale: Vec3f32::one(),
        },
      )
      .unwrap();

    scene
      .add_component(
        emitter_entity,
        aethervk_core_rlib::scene::ForceEmitterComponent::Planar {
          normal: Vec3f32::from_components(1.0, 0.0, 0.0),
          base_force: 0.005,
          trunc_distance: 10.0,
        },
      )
      .unwrap();

    // Setup Particles natively on the Quad Entity!
    let config = aethervk_core_rlib::scene::particles::ParticleEmitterComponent {
      uv_distribution: aethervk_core_rlib::math::distribution::Distribution2D::new(
        &[1.0, 1.0, 1.0, 1.0],
        2,
        2,
      ),
      delta: 16000,
      max_particles: 1_000_000,
      velocity_intensity: aethervk_core_rlib::scene::particles::GaussianParams {
        mean: 5.0,
        std_dev: 2.0,
        min: 0.0,
        max: 10.0,
      },
      emission_count: aethervk_core_rlib::scene::particles::GaussianParams {
        mean: 1600.0,
        std_dev: 200.0,
        min: 0.0,
        max: 2000.0,
      },
      particle_radius: 0.05,
      density: 1.0,
      lifetime: 10_000_000, // 10 seconds
      color: [0.2, 0.6, 1.0, 0.8],
      beta: 0.0,
      use_particle2: true, // Requested Particle2 pipeline
    };
    scene.add_component(quad_entity, config).unwrap();

    let particle_sys =
      aethervk_core_rlib::scene::particles::ParticleSystemComponent::new(1_000_000);
    scene.add_component(quad_entity, particle_sys).unwrap();

    // Add UI Text
    let ui_text_entity = scene.spawn_entity("ui_text");
    self.ui_text_entity = Some(ui_text_entity);
    let mut t2d = Transform2DComponent::default();
    t2d.local_position = [20.0, 20.0];
    t2d.global_depth = 2;
    scene.add_component(ui_text_entity, t2d).unwrap();
    scene
      .add_component(
        ui_text_entity,
        ScreenSpaceTextComponent {
          text: "Particles: 0".to_string(),
          font_atlas: font_arc.clone(),
          font_hash,
          color: [1.0, 1.0, 1.0, 1.0],
          points: 24.0,
          use_new_path: true,
        },
      )
      .unwrap();

    // Un-pause simulation out of the gate
    let _ = ctx.threads.logic_thread.tx().try_send(
      aethervk_core_rlib::simulation_api::structs::LogicCommand::SetSceneTimeScale {
        scene_id,
        scale: aethervk_core_rlib::simulation_api::structs::TimeScale::RealTime,
      },
    );
    let _ =
      ctx.threads.logic_thread.tx().try_send(
        aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id },
      );

    #[cfg(feature = "use_vulkan_physics")]
    let engine_type = aethervk_core_rlib::simulation_api::structs::PhysicsEngineType::VulkanCompute;
    #[cfg(not(feature = "use_vulkan_physics"))]
    let engine_type = aethervk_core_rlib::simulation_api::structs::PhysicsEngineType::CpuSimd;

    let _ = ctx.threads.logic_thread.tx().try_send(
      aethervk_core_rlib::simulation_api::structs::LogicCommand::SetPhysicsEngineType {
        scene_id,
        engine_type,
      },
    );

    Ok(())
  }

  fn on_about_to_wait(&mut self, ctx: &mut SimulationContext, scene_id: u64, _delta_time: f32) {
    let scene_ctx = ctx.get_scene(scene_id).unwrap();
    let scene_ctx_read = scene_ctx.read();
    let scene = &scene_ctx_read.scene;

    let mut count = 0;
    if let Some(quad_id) = self.quad_entity {
      let _ = scene
        .with_component::<aethervk_core_rlib::scene::particles::ParticleSystemComponent, _, _>(
          quad_id,
          |ps| {
            count = ps.particles.read().len();
          },
        );
    }

    if count > 0 {
      let elapsed = std::time::Instant::now().duration_since(self.startup_time).as_millis();
      if elapsed % 1000 < 30 {
        println!("Active Particles: {}", count);
      }
    }

    // Update UI text
    if let Some(ui_id) = self.ui_text_entity {
      let _ = scene.with_component_mut::<ScreenSpaceTextComponent, _, _>(ui_id, |text_comp| {
        text_comp.text = format!("Particles: {}", count);
      });
    }
  }
}

fn main() {
  let delegate = ForceTestDelegate {
    camera_entity: 0,
    quad_entity: None,
    particle_sys_entity: None,
    ui_text_entity: None,
    font_atlas: None,
    font_hash: 0,
    startup_time: std::time::Instant::now(),
  };

  run_simulation_app("Particle Force Test", delegate);
}

#[cfg(test)]
mod tests {
  use super::*;
  use aethervk_core_rlib::{
    gpu::{CommandBuffer, DeviceBuffer, Kernels, ParticleGpu, WaitHandle},
    gpu_backends::vulkan::{device::Device, physics::VulkanComputeKernels},
    physics::cpu_kernels::CpuSimdKernels,
  };

  #[test]
  fn test_barnes_hut_parity() {
    let asset_dir = test_utils::cycle_get_asset_path_from_exe(true).to_string_lossy().to_string();
    aethervk_core_rlib::gpu::ASSET_DIR.write().replace(asset_dir);

    let mut ctx =
      SimulationContext::startup(aethervk_core_rlib::gpu::VULKAN_RENDER_BACKEND, None).unwrap();

    let scene_id = ctx.create_empty_scene(true).unwrap();
    let root_entity = ctx.spawn_entity(scene_id, "root").unwrap();
    ctx
      .add_transform_component(
        scene_id,
        root_entity,
        Vec3f32::zero(),
        Quat::identity(),
        Vec3f32::one(),
      )
      .unwrap();

    // Spawn particle system
    let ps_entity = ctx.spawn_entity(scene_id, "ps").unwrap();
    ctx.set_parent(scene_id, ps_entity, root_entity).unwrap();
    ctx
      .add_transform_component(
        scene_id,
        ps_entity,
        Vec3f32::zero(),
        Quat::identity(),
        Vec3f32::one(),
      )
      .unwrap();

    let mut ps_comp = aethervk_core_rlib::scene::particles::ParticleSystemComponent::new(10);
    {
      let mut p_write = ps_comp.particles.write();
      p_write.push(aethervk_core_rlib::scene::particles::Particle {
        position: [0.0, 0.0, 0.0],
        velocity: [0.0, 0.0, 0.0],
        color: [1.0, 1.0, 1.0, 1.0],
        mass: 10.0,
        age_and_active: (0 | 1 << 63) as u64,
        uv_scale: [1.0, 1.0],
        uv_offset: [0.0, 0.0],
      });
      p_write.push(aethervk_core_rlib::scene::particles::Particle {
        position: [2.0, 0.0, 0.0],
        velocity: [0.0, 0.0, 0.0],
        color: [1.0, 1.0, 1.0, 1.0],
        mass: 5.0,
        age_and_active: (0 | 1 << 63) as u64,
        uv_scale: [1.0, 1.0],
        uv_offset: [0.0, 0.0],
      });
    }

    ctx
      .get_scene(scene_id)
      .unwrap()
      .write()
      .scene
      .add_component(ps_entity, ps_comp)
      .unwrap();

    let thread_pool =
      std::sync::Arc::new(aethervk_oshal_rlib::os::pool::ThreadPool::new(4).unwrap());
    let cpu_kernels = CpuSimdKernels {
      thread_pool: thread_pool.clone(),
    };

    let scene_ctx = ctx.get_scene(scene_id).unwrap();
    let scene_read = scene_ctx.read();
    let scene = &scene_read.scene;

    // CPU PASS
    let mut cpu_cmd = cpu_kernels.create_command_buffer().unwrap();
    let kinematics = cpu_kernels
      .build_kinematic_bodies(
        &mut cpu_cmd,
        &aethervk_core_rlib::physics::physics_scene::PhysicsScene::default(),
        scene,
      )
      .unwrap();
    let rigid_bodies = cpu_kernels
      .build_rigid_bodies(
        &mut cpu_cmd,
        &aethervk_core_rlib::physics::physics_scene::PhysicsScene::default(),
        scene,
      )
      .unwrap();
    let mut cpu_particles = cpu_kernels.build_particles(&mut cpu_cmd, scene).unwrap();
    let cpu_bvh = cpu_kernels
      .build_motion_bvh(
        &mut cpu_cmd,
        &kinematics,
        &rigid_bodies,
        &cpu_particles,
        16000,
      )
      .unwrap();

    cpu_kernels
      .compute_self_gravity(&mut cpu_cmd, &cpu_bvh, &mut cpu_particles)
      .unwrap();
    cpu_cmd.submit().unwrap();

    let cpu_result = cpu_particles.data.clone();

    // GPU PASS
    let mut gpu_result = vec![];
    ctx
      .with_device(|dev| {
        let actual_device = dev.as_any().downcast_ref::<Device>().unwrap();
        let gpu_kernels = VulkanComputeKernels {
          device: actual_device.device.clone(),
          pipelines: actual_device.res.read().pipelines.physics.clone(),
          addresses: actual_device.res.read().physics_addresses.clone(),
          thread_pool: thread_pool.clone(),
        };

        let mut gpu_cmd = gpu_kernels.create_command_buffer().unwrap();
        actual_device.begin_command_buffer(gpu_cmd.cmd).unwrap();

        let g_kinematics = gpu_kernels
          .build_kinematic_bodies(
            &mut gpu_cmd,
            &aethervk_core_rlib::physics::physics_scene::PhysicsScene::default(),
            scene,
          )
          .unwrap();
        let g_rigid_bodies = gpu_kernels
          .build_rigid_bodies(
            &mut gpu_cmd,
            &aethervk_core_rlib::physics::physics_scene::PhysicsScene::default(),
            scene,
          )
          .unwrap();
        let mut g_particles = gpu_kernels.build_particles(&mut gpu_cmd, scene).unwrap();
        let g_bvh = gpu_kernels
          .build_motion_bvh(
            &mut gpu_cmd,
            &g_kinematics,
            &g_rigid_bodies,
            &g_particles,
            16000,
          )
          .unwrap();

        gpu_kernels
          .compute_self_gravity(&mut gpu_cmd, &g_bvh, &mut g_particles)
          .unwrap();

        let read_handle = g_particles.enqueue_read_to_cpu(&mut gpu_cmd).unwrap();

        actual_device.end_command_buffer(gpu_cmd.cmd).unwrap();

        let submit_info =
          ash::vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&gpu_cmd.cmd));
        let fence = actual_device.create_fence().unwrap();
        unsafe {
          actual_device
            .device
            .queue_submit(
              actual_device.queues.graphics.queue,
              std::slice::from_ref(&submit_info),
              fence,
            )
            .unwrap();
          actual_device
            .device
            .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
            .unwrap();
        }

        gpu_result = read_handle.wait().unwrap();

        aethervk_core_rlib::types::GpuResult::Ok(())
      })
      .unwrap();

    // COMPARE
    assert_eq!(cpu_result.len(), gpu_result.len());
    for (c, g) in cpu_result.iter().zip(gpu_result.iter()) {
      assert!(
        (c.force[0] - g.force[0]).abs() < 1e-4,
        "CPU force x {} != GPU force x {}",
        c.force[0],
        g.force[0]
      );
      assert!(
        (c.force[1] - g.force[1]).abs() < 1e-4,
        "CPU force y {} != GPU force y {}",
        c.force[1],
        g.force[1]
      );
      assert!(
        (c.force[2] - g.force[2]).abs() < 1e-4,
        "CPU force z {} != GPU force z {}",
        c.force[2],
        g.force[2]
      );
    }
  }
}
