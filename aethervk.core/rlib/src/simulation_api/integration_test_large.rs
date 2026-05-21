#[cfg(test)]
mod tests {
  use crate::simulation_api::SimulationContext;
  use crate::simulation_api::structs::{PhysicsEngineType, TimeScale, LogicCommand};
  use crate::scene::{CameraComponent, TransformComponent, SunComponent, SkyComponent};
  use aethervk_oshal_rlib::math::vector::{vec3::Vec3f32, vec4::Quat, Vector, Vector3};
  use aethervk_oshal_rlib::math::quaternion::Quaternion;
  use alloc::format;
  use crate::scene::{ForceEmitterComponent, particles::ParticleSystemComponent};

  fn save_state(ctx: &SimulationContext, scene_id: u64, comet_id: u64, filename: &str) {
    use std::fs::File;
    use core::fmt::Write;
    let scene_ctx = ctx.scenes.read().scenes.get(&scene_id).unwrap().clone();
    let mut scene_ctx_w = scene_ctx.write();
    let scene = &mut scene_ctx_w.scene;
    
    let mut out = alloc::string::String::new();
    let comet_eid = slotmap::KeyData::from_ffi(comet_id).into();
    scene.with_component(comet_eid, |t: &crate::scene::TransformComponent| {
      let _ = writeln!(out, "Comet Pos: {:?}", t.position);
    });
    
    scene.with_component(comet_eid, |sys: &crate::scene::particles::ParticleSystemComponent| {
      let particles = sys.particles.read();
      let _ = writeln!(out, "Particles: {}", particles.len());
      for (i, p) in particles.iter().take(10).enumerate() {
        let _ = writeln!(out, "  [{}] pos: {:?}", i, p.position);
      }
    });
    
    if let Ok(mut f) = File::create(filename) {
      let _ = std::io::Write::write_all(&mut f, out.as_bytes());
    }
  }
  fn panic_error_callback(msg: &str) {
    panic!("Vulkan Error: {}", msg);
  }

  fn get_test_context() -> Option<*mut SimulationContext> {
    let asset_dir = format!("{}/../../assets", env!("CARGO_MANIFEST_DIR"));
    SimulationContext::set_asset_path(&asset_dir);

    let ctx_ptr = SimulationContext::startup(crate::gpu::VULKAN_RENDER_BACKEND, Some(panic_error_callback));
    if let Ok(boxed) = ctx_ptr {
      return Some(alloc::boxed::Box::into_raw(boxed));
    }
    None
  }

  #[test]
  fn test_large_physics_render_integration() {
    if let Some(ctx_ptr) = get_test_context() {
      unsafe {
        let ctx = &mut *ctx_ptr;
        let scene_id = ctx.create_empty_scene().unwrap();

        // Ensure we use Vulkan physics engine
        let _ = ctx.threads.logic_thread.tx().try_send(LogicCommand::SetPhysicsEngineType { 
          scene_id, 
          engine_type: PhysicsEngineType::VulkanCompute 
        });

        // Create Sun (Macroframe) - 1 AU away
        let sun_name = alloc::ffi::CString::new("Sun").unwrap();
        let sun_id = ctx.spawn_entity(scene_id, sun_name.to_str().unwrap()).unwrap();
        let sun_dist = 1.496e11; // 1 AU
        ctx.add_transform_component(
          scene_id,
          sun_id,
          Vec3f32::from_components(0.0, 0.0, sun_dist as f32),
          Quat::identity(),
          Vec3f32::from_components(1.0, 1.0, 1.0),
        ).unwrap();
        ctx.add_sun_component(scene_id, sun_id, (255, 255, 255), 1.0).unwrap();

        // Create Sky
        let sky_name = alloc::ffi::CString::new("Sky").unwrap();
        let sky_id = ctx.spawn_entity(scene_id, sky_name.to_str().unwrap()).unwrap();
        ctx.add_sky_component(scene_id, sky_id).unwrap();

        // Create Comet (Microframe)
        let comet_name = alloc::ffi::CString::new("Comet").unwrap();
        let comet_id = ctx.spawn_entity(scene_id, comet_name.to_str().unwrap()).unwrap();
        ctx.add_transform_component(
          scene_id,
          comet_id,
          Vec3f32::from_components(0.0, 0.0, 0.0),
          Quat::identity(),
          Vec3f32::from_components(1.0, 1.0, 1.0),
        ).unwrap();
        
        let path = alloc::format!("{}/../../assets/Comet.glb", env!("CARGO_MANIFEST_DIR"));
        let path_buf = aethervk_oshal_rlib::os::fs::PathBuf::from(&path);
        
        // Sphere diameter ~2km -> radius ~1km = 1000m
        ctx.add_physical_mesh_component(scene_id, comet_id, &path_buf, 1000.0, [1.0, 1.0, 1.0]).unwrap();

        let mut particle_sys = ParticleSystemComponent::new(100_000);
        let comet_radius = 1000.0;
        
        // Add beta compensation (force pushing comet outwards from sun)
        // Sun is at Z = 1.496e11. We add a custom force evaluator for beta compensation
        // We will just add a Planar ForceEmitterComponent to simulate constant force
        let comet_eid = slotmap::KeyData::from_ffi(comet_id).into();
        let _ = ctx.scenes.read().scenes.get(&scene_id).unwrap().write().scene.add_component(comet_eid, ForceEmitterComponent::Planar {
          normal: Vec3f32::from_components(0.0, 0.0, -1.0),
          base_force: 9.81 * 1000.0, // Compensate mass * g
          trunc_distance: 2.0e11,
        });

        // Add 10,000 particles at random locations near comet (as a stress test)
        {
          let mut parts = particle_sys.particles.write();
          for i in 0..10_000 {
            parts.push(crate::scene::particles::ParticleData {
              id_low: i as u32,
              id_high: 0,
              age_low: 0,
              age_high: 0,
              position: [0.0, 0.0, comet_radius + (i as f32) * 0.1],
              velocity: [0.0, 0.0, 0.0],
              mass: 1.0,
              active: 1,
            });
          }
        }
        let _ = ctx.scenes.read().scenes.get(&scene_id).unwrap().write().scene.add_component(comet_eid, particle_sys);

        // Torque injection via ForceEvaluatorComponent
        // Let's just add a RigidbodyImex manual update or ForceEvaluator if implemented
        // Or we can add angular velocity to the rigid body if it had one.
        // Wait, to do physics, we need a ColliderComponent for the comet!
        let _ = ctx.scenes.read().scenes.get(&scene_id).unwrap().write().scene.add_component(comet_eid, crate::scene::ColliderComponent {
           shape: crate::scene::ColliderShape::Sphere { radius: comet_radius },
           mass: 1_000_000.0,
           friction: 0.5,
           restitution: 0.5,
        });

        // Save Initial State
        save_state(ctx, scene_id, comet_id, "initial_state.txt");
        
        let _ = ctx.threads.logic_thread.tx().try_send(crate::simulation_api::structs::LogicCommand::PlayScene { scene_id });

        // Simulate for 10 seconds
        aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_secs(10));

        let _ = ctx.threads.logic_thread.tx().try_send(crate::simulation_api::structs::LogicCommand::PauseScene { scene_id });
        aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(100));

        // Save Final State
        save_state(ctx, scene_id, comet_id, "final_state.txt");
        let _ = alloc::boxed::Box::from_raw(ctx_ptr);
      }
    }
  }
}
