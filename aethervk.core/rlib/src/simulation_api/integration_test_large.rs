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
  
  // Storage for our 4 PEs task_ids
  static LAST_PE_TASK_IDS: [core::sync::atomic::AtomicU64; 4] = [
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
  ];

  extern "C" fn visual_render_callback_impl(scene_id: u64, pe_id: u64, render_generation: u64) {
    // Basic modulus mapping assuming PE handles 1, 2, 3, 4
    let idx = (pe_id as usize).saturating_sub(1) % 4;
    LAST_PE_TASK_IDS[idx].store(render_generation, core::sync::atomic::Ordering::Release);
  }

  #[test]
  fn test_visual_physics_render_sync() {
    // Reset static state from any previous test run in this process.
    LAST_PE_TASK_IDS[0].store(0, core::sync::atomic::Ordering::Release);
    LAST_PE_TASK_IDS[1].store(0, core::sync::atomic::Ordering::Release);
    LAST_PE_TASK_IDS[2].store(0, core::sync::atomic::Ordering::Release);
    LAST_PE_TASK_IDS[3].store(0, core::sync::atomic::Ordering::Release);

    if let Some(ctx_ptr) = get_test_context() {
      // RAII guard: ensures ctx_ptr is always freed even if an assertion panics,
      // so background threads are joined and nextest does not time out.
      struct CtxGuard(*mut SimulationContext);
      impl Drop for CtxGuard {
        fn drop(&mut self) {
          unsafe { let _ = alloc::boxed::Box::from_raw(self.0); }
        }
      }
      let _guard = CtxGuard(ctx_ptr);

      unsafe {
        let ctx = &mut *ctx_ptr;
        let scene_id = ctx.create_default_scene().unwrap();

        // 1. Setup Vulkan Compute Physics
        let _ = ctx.threads.logic_thread.tx().try_send(crate::simulation_api::structs::LogicCommand::SetPhysicsEngineType { 
          scene_id, 
          engine_type: PhysicsEngineType::VulkanCompute 
        });

        // 2. Setup 4 Windowless Presentation Engines and Cameras
        let width = 512;
        let height = 512;
        
        let pe_1 = ctx.create_presentation_engine(scene_id, width, height).unwrap();
        let pe_2 = ctx.create_presentation_engine(scene_id, width, height).unwrap();
        let pe_3 = ctx.create_presentation_engine(scene_id, width, height).unwrap();
        let pe_4 = ctx.create_presentation_engine(scene_id, width, height).unwrap();
        
        let cam_1 = ctx.add_perspective_camera(scene_id, pe_1, "cam1", 45.0, 0.1, 1000.0).unwrap();
        let cam_2 = ctx.add_perspective_camera(scene_id, pe_2, "cam2", 45.0, 0.1, 1000.0).unwrap();
        let cam_3 = ctx.add_perspective_camera(scene_id, pe_3, "cam3", 45.0, 0.1, 1000.0).unwrap();
        let cam_4 = ctx.add_perspective_camera(scene_id, pe_4, "cam4", 45.0, 0.1, 1000.0).unwrap();
        
        // Position cameras around the origin (100 units away)
        {
          let scene_ctx = ctx.scenes.read().scenes.get(&scene_id).unwrap().clone();
          let mut scene_write = scene_ctx.write();
          scene_write.scene.with_component_mut(slotmap::KeyData::from_ffi(cam_1.get()).into(), |t: &mut crate::scene::TransformComponent| {
            t.position = Vec3f32::from_components(0.0, 100.0, 100.0);
          });
          scene_write.scene.with_component_mut(slotmap::KeyData::from_ffi(cam_2.get()).into(), |t: &mut crate::scene::TransformComponent| {
            t.position = Vec3f32::from_components(100.0, 0.0, 100.0);
          });
          scene_write.scene.with_component_mut(slotmap::KeyData::from_ffi(cam_3.get()).into(), |t: &mut crate::scene::TransformComponent| {
            t.position = Vec3f32::from_components(-100.0, 0.0, 100.0);
          });
          scene_write.scene.with_component_mut(slotmap::KeyData::from_ffi(cam_4.get()).into(), |t: &mut crate::scene::TransformComponent| {
            t.position = Vec3f32::from_components(0.0, -100.0, 100.0);
          });
        }

        // 3. Setup Scene Entities
        let comet_name = alloc::ffi::CString::new("Comet").unwrap();
        let comet_id = ctx.spawn_entity(scene_id, comet_name.to_str().unwrap()).unwrap();
        ctx.add_transform_component(
          scene_id, comet_id, Vec3f32::from_components(0.0, 0.0, 0.0), Quat::identity(), Vec3f32::from_components(1.0, 1.0, 1.0)
        ).unwrap();

        let path = alloc::format!("{}/../../assets/Comet.glb", env!("CARGO_MANIFEST_DIR"));
        let path_buf = aethervk_oshal_rlib::os::fs::PathBuf::from(&path);
        ctx.add_physical_mesh_component(scene_id, comet_id, &path_buf, 10.0, [1.0, 1.0, 1.0]).unwrap();
        
        // Force and Collider
        let comet_eid = slotmap::KeyData::from_ffi(comet_id).into();
        let _ = ctx.scenes.read().scenes.get(&scene_id).unwrap().write().scene.add_component(comet_eid, ForceEmitterComponent::Planar {
          normal: Vec3f32::from_components(0.0, 1.0, 0.0),
          base_force: 1000.0,
          trunc_distance: 1000.0,
        });
        let _ = ctx.scenes.read().scenes.get(&scene_id).unwrap().write().scene.add_component(comet_eid, crate::scene::ColliderComponent {
           shape: crate::scene::ColliderShape::Sphere { radius: 10.0 },
           mass: 10.0,
           friction: 0.5,
           restitution: 0.5,
        });
        let _ = ctx.scenes.read().scenes.get(&scene_id).unwrap().write().scene.add_component(comet_eid, crate::scene::KinematicComponent::default());

        // 4. Hook Callback
        SimulationContext::set_render_callback(Some(visual_render_callback_impl));
        
        let wait_for_images = |tag: &str| {
            let mut attempts = 0;
            let mut ready = false;
            while attempts < 200 {
                let id1 = LAST_PE_TASK_IDS[0].load(core::sync::atomic::Ordering::Acquire);
                let id2 = LAST_PE_TASK_IDS[1].load(core::sync::atomic::Ordering::Acquire);
                let id3 = LAST_PE_TASK_IDS[2].load(core::sync::atomic::Ordering::Acquire);
                let id4 = LAST_PE_TASK_IDS[3].load(core::sync::atomic::Ordering::Acquire);
                if id1 > 0 && id2 > 0 && id3 > 0 && id4 > 0 {
                    ready = true;
                    break;
                }
                std::thread::sleep(core::time::Duration::from_millis(10));
                attempts += 1;
            }
            if !ready { return; }
            
            // Wait for completion and download
            let tids = [LAST_PE_TASK_IDS[0].load(core::sync::atomic::Ordering::Acquire), LAST_PE_TASK_IDS[1].load(core::sync::atomic::Ordering::Acquire), LAST_PE_TASK_IDS[2].load(core::sync::atomic::Ordering::Acquire), LAST_PE_TASK_IDS[3].load(core::sync::atomic::Ordering::Acquire)];
            for (i, &tid) in tids.iter().enumerate() {
                let mut status = ctx.get_task_status(tid);
                let mut attempt = 0;
                while matches!(status, crate::simulation_api::structs::TaskStatusCode::Pending) && attempt < 100 {
                    std::thread::sleep(core::time::Duration::from_millis(10));
                    status = ctx.get_task_status(tid);
                    attempt += 1;
                }
                
                let mut buffer = vec![0u8; (width * height * 4) as usize];
                if ctx.download_image(tid, buffer.as_mut_ptr(), buffer.len()) {
                    let _ = image::save_buffer(
                        alloc::format!("output_cam_{}_{}.png", i + 1, tag),
                        &buffer,
                        width,
                        height,
                        image::ColorType::Rgba8
                    );
                }
            }
            // Reset for next pass
            LAST_PE_TASK_IDS[0].store(0, core::sync::atomic::Ordering::Release);
            LAST_PE_TASK_IDS[1].store(0, core::sync::atomic::Ordering::Release);
            LAST_PE_TASK_IDS[2].store(0, core::sync::atomic::Ordering::Release);
            LAST_PE_TASK_IDS[3].store(0, core::sync::atomic::Ordering::Release);
        };

        // Output Initial State
        let _ = ctx.threads.logic_thread.tx().try_send(crate::simulation_api::structs::LogicCommand::SetSceneTimeScale { 
          scene_id, 
          scale: crate::simulation_api::structs::TimeScale::RealTime
        });
        let _ = ctx.threads.logic_thread.tx().try_send(crate::simulation_api::structs::LogicCommand::PlayScene { scene_id });
        wait_for_images("initial");
        
        // Wait and Output Final State
        aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_secs(2));
        let _ = ctx.threads.logic_thread.tx().try_send(crate::simulation_api::structs::LogicCommand::PauseScene { scene_id });
        aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(100));
        
        // Final position assertions
        let scene_ctx = ctx.scenes.read().scenes.get(&scene_id).unwrap().clone();
        scene_ctx.write().scene.with_component(comet_eid, |t: &crate::scene::TransformComponent| {
            // Started at 0, Force applied along Y axis (0, 1, 0)
            assert!(t.position.y() > 5.0, "Comet did not move along Y axis! Final pos: {:?}", t.position);
        });

        wait_for_images("final");
        
        // _guard's Drop calls Box::from_raw(ctx_ptr) automatically.
      }
    }
  }
}
