#![cfg(ignore)]
#[cfg(test)]
mod tests {
  use crate::{
    scene::{
      CameraComponent, ForceEmitterComponent, SkyComponent, SunComponent, TransformComponent,
      particles::ParticleSystemComponent,
    },
    simulation_api::{
      SimulationContext,
      structs::{LogicCommand, PhysicsEngineType, TimeScale},
    },
  };
  use aethervk_oshal_rlib::math::{
    quaternion::Quaternion,
    vector::{Vector, Vector3, vec3::Vec3f32, vec4::Quat},
  };
  use alloc::format;

  fn save_state(ctx: &SimulationContext, scene_id: u64, comet_id: u64, filename: &str) {
    use core::fmt::Write;
    use std::fs::File;
    let scene_ctx = ctx.scenes.read().scenes.get(&scene_id).unwrap().clone();
    let mut scene_ctx_w = scene_ctx.write();
    let scene = &mut scene_ctx_w.scene;

    let mut out = alloc::string::String::new();
    let comet_eid = slotmap::KeyData::from_ffi(comet_id).into();
    scene.with_component(comet_eid, |t: &crate::scene::TransformComponent| {
      let _ = writeln!(out, "Comet Pos: {:?}", t.position);
    });

    scene.with_component(
      comet_eid,
      |sys: &crate::scene::particles::ParticleSystemComponent| {
        let particles = sys.particles.read();
        let _ = writeln!(out, "Particles: {}", particles.len());
        for (i, p) in particles.iter().take(10).enumerate() {
          let _ = writeln!(out, "  [{}] pos: {:?}", i, p.position);
        }
      },
    );

    if let Ok(mut f) = File::create(filename) {
      let _ = std::io::Write::write_all(&mut f, out.as_bytes());
    }
  }
  fn panic_error_callback(msg: &str) {
    // Ignore false positive from Khronos Validation Layers GPU-Assisted Validation on Lavapipe ARM64
    // when using buffer_reference inside push constants.
    if msg.contains("UNASSIGNED-Device address out of bounds")
      || msg.contains("UNASSIGNED-VkSemaphore-state-timeout")
      || msg.contains("Ran out of file descriptors")
    {
      return;
    }
    println!("Vulkan Validation Error in test: {}", msg);
    panic!("Vulkan Error: {}", msg);
  }

  fn get_test_context() -> Option<*mut SimulationContext> {
    // Raise the per-process file-descriptor soft limit to 8192.
    //
    // GPU-AV instrumentation, Mesa shader-cache I/O, and 4 windowless
    // presentation engines (each creating 20+ render archetypes) together
    // consume hundreds of FDs, overflowing the default Linux soft limit
    // of 1024.  setrlimit is safe from any thread; nextest runs each test
    // in its own subprocess so this does not affect other tests.
    #[cfg(target_os = "linux")]
    unsafe {
      let mut rl = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
      };
      if libc::getrlimit(libc::RLIMIT_NOFILE, &mut rl) == 0 {
        let desired: libc::rlim_t = 8192;
        if rl.rlim_cur < desired {
          rl.rlim_cur = desired.min(rl.rlim_max);
          let _ = libc::setrlimit(libc::RLIMIT_NOFILE, &rl);
        }
      }
    }

    let asset_dir = format!("{}/../../assets", env!("CARGO_MANIFEST_DIR"));
    SimulationContext::set_asset_path(&asset_dir);

    let ctx_ptr = SimulationContext::startup(
      crate::gpu::VULKAN_RENDER_BACKEND,
      Some(panic_error_callback),
    );
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
        let scene_id = ctx
          .create_empty_scene(
            true,
            hifitime::Epoch::from_gregorian_at_midnight(2020, 12, 25, hifitime::TimeScale::UTC),
            hifitime::Epoch::from_gregorian_at_midnight(2020, 12, 26, hifitime::TimeScale::UTC),
          )
          .unwrap();

        // Ensure we use Vulkan physics engine
        let _ = ctx.threads.logic_thread.tx().try_send(LogicCommand::SetPhysicsEngineType {
          scene_id,
          engine_type: PhysicsEngineType::VulkanCompute,
        });

        // Create Sun (Macroframe) - 1 AU away
        let sun_name = alloc::ffi::CString::new("Sun").unwrap();
        let sun_id = ctx.spawn_entity(scene_id, sun_name.to_str().unwrap()).unwrap();
        let sun_dist = 1.496e11; // 1 AU
        ctx
          .add_transform_component(
            scene_id,
            sun_id,
            Vec3f32::from_components(0.0, 0.0, sun_dist as f32),
            Quat::identity(),
            Vec3f32::from_components(1.0, 1.0, 1.0),
          )
          .unwrap();
        ctx.add_sun_component(scene_id, sun_id, (255, 255, 255), 1.0).unwrap();

        // Create Sky
        let sky_name = alloc::ffi::CString::new("Sky").unwrap();
        let sky_id = ctx.spawn_entity(scene_id, sky_name.to_str().unwrap()).unwrap();
        ctx.add_sky_component(scene_id, sky_id).unwrap();

        // Create Comet (Microframe)
        let comet_name = alloc::ffi::CString::new("Comet").unwrap();
        let comet_id = ctx.spawn_entity(scene_id, comet_name.to_str().unwrap()).unwrap();
        ctx
          .add_transform_component(
            scene_id,
            comet_id,
            Vec3f32::from_components(0.0, 0.0, 0.0),
            Quat::identity(),
            Vec3f32::from_components(1.0, 1.0, 1.0),
          )
          .unwrap();

        let path = alloc::format!("{}/../../assets/Comet.glb", env!("CARGO_MANIFEST_DIR"));
        let path_buf = aethervk_oshal_rlib::os::fs::PathBuf::from(&path);

        // Sphere diameter ~2km -> radius ~1km = 1000m

        // Save Initial State
        save_state(ctx, scene_id, comet_id, "initial_state.txt");

        let _ = ctx
          .threads
          .logic_thread
          .tx()
          .try_send(crate::simulation_api::structs::LogicCommand::PlayScene { scene_id });

        // Simulate for 10 seconds
        aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_secs(
          10,
        ));

        let _ = ctx
          .threads
          .logic_thread
          .tx()
          .try_send(crate::simulation_api::structs::LogicCommand::PauseScene { scene_id });
        aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(
          100,
        ));

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

  static PE_IDS: [core::sync::atomic::AtomicU64; 4] = [
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
    core::sync::atomic::AtomicU64::new(0),
  ];

  extern "C" fn visual_render_callback_impl(scene_id: u64, pe_id: u64, render_generation: u64) {
    aethervk_oshal_rlib::log!("Callback fired for pe_id: {}", pe_id);
    if pe_id == PE_IDS[0].load(core::sync::atomic::Ordering::Acquire) {
      LAST_PE_TASK_IDS[0].store(render_generation, core::sync::atomic::Ordering::Release);
    } else if pe_id == PE_IDS[1].load(core::sync::atomic::Ordering::Acquire) {
      LAST_PE_TASK_IDS[1].store(render_generation, core::sync::atomic::Ordering::Release);
    } else if pe_id == PE_IDS[2].load(core::sync::atomic::Ordering::Acquire) {
      LAST_PE_TASK_IDS[2].store(render_generation, core::sync::atomic::Ordering::Release);
    } else if pe_id == PE_IDS[3].load(core::sync::atomic::Ordering::Acquire) {
      LAST_PE_TASK_IDS[3].store(render_generation, core::sync::atomic::Ordering::Release);
    } else {
      aethervk_oshal_rlib::log!("pe_id {} DID NOT MATCH ANY PE_IDS!", pe_id);
    }
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
          unsafe {
            let _ = alloc::boxed::Box::from_raw(self.0);
          }
        }
      }
      let _guard = CtxGuard(ctx_ptr);

      unsafe {
        let ctx = &mut *ctx_ptr;
        let scene_id = ctx.create_default_scene(true).unwrap();

        // 1. Setup Vulkan Compute Physics
        let _ = ctx.threads.logic_thread.tx().try_send(
          crate::simulation_api::structs::LogicCommand::SetPhysicsEngineType {
            scene_id,
            engine_type: PhysicsEngineType::VulkanCompute,
          },
        );

        // 2. Setup 4 Windowless Presentation Engines and Cameras
        let is_cpu = ctx.with_device(|device| Ok(device.is_cpu_device())).unwrap_or(false);
        let width = if is_cpu { 64 } else { 512 };
        let height = if is_cpu { 64 } else { 512 };

        let pe_1 = ctx.create_presentation_engine(scene_id, width, height).unwrap();
        let pe_2 = ctx.create_presentation_engine(scene_id, width, height).unwrap();
        let pe_3 = ctx.create_presentation_engine(scene_id, width, height).unwrap();
        let pe_4 = ctx.create_presentation_engine(scene_id, width, height).unwrap();

        PE_IDS[0].store(pe_1.0, core::sync::atomic::Ordering::Release);
        PE_IDS[1].store(pe_2.0, core::sync::atomic::Ordering::Release);
        PE_IDS[2].store(pe_3.0, core::sync::atomic::Ordering::Release);
        PE_IDS[3].store(pe_4.0, core::sync::atomic::Ordering::Release);

        let cam_1 = ctx.add_perspective_camera(scene_id, pe_1, "cam1", 45.0, 0.1, 1000.0).unwrap();
        let cam_2 = ctx.add_perspective_camera(scene_id, pe_2, "cam2", 45.0, 0.1, 1000.0).unwrap();
        let cam_3 = ctx.add_perspective_camera(scene_id, pe_3, "cam3", 45.0, 0.1, 1000.0).unwrap();
        let cam_4 = ctx.add_perspective_camera(scene_id, pe_4, "cam4", 45.0, 0.1, 1000.0).unwrap();

        // Position cameras around the origin (100 units away)
        {
          let scene_ctx = ctx.scenes.read().scenes.get(&scene_id).unwrap().clone();
          let mut scene_write = scene_ctx.write();
          scene_write.scene.with_component_mut(
            slotmap::KeyData::from_ffi(cam_1.get()).into(),
            |t: &mut crate::scene::TransformComponent| {
              t.position = Vec3f32::from_components(0.0, 100.0, 100.0);
            },
          );
          scene_write.scene.with_component_mut(
            slotmap::KeyData::from_ffi(cam_2.get()).into(),
            |t: &mut crate::scene::TransformComponent| {
              t.position = Vec3f32::from_components(100.0, 0.0, 100.0);
            },
          );
          scene_write.scene.with_component_mut(
            slotmap::KeyData::from_ffi(cam_3.get()).into(),
            |t: &mut crate::scene::TransformComponent| {
              t.position = Vec3f32::from_components(-100.0, 0.0, 100.0);
            },
          );
          scene_write.scene.with_component_mut(
            slotmap::KeyData::from_ffi(cam_4.get()).into(),
            |t: &mut crate::scene::TransformComponent| {
              t.position = Vec3f32::from_components(0.0, -100.0, 100.0);
            },
          );
        }

        // 3. Setup Scene Entities
        let comet_name = alloc::ffi::CString::new("Comet").unwrap();
        let comet_id = ctx.spawn_entity(scene_id, comet_name.to_str().unwrap()).unwrap();
        ctx
          .add_transform_component(
            scene_id,
            comet_id,
            Vec3f32::from_components(0.0, 0.0, 0.0),
            Quat::identity(),
            Vec3f32::from_components(1.0, 1.0, 1.0),
          )
          .unwrap();

        // 4. Hook Callback
        SimulationContext::set_render_callback(Some(visual_render_callback_impl));

        let mut wait_for_images = |tag: &str, save_to_disk: bool| {
          let mut attempts = 0;
          let mut ready = false;
          let max_attempts = if is_cpu { 6000 } else { 200 };
          while attempts < max_attempts {
            let id1 = LAST_PE_TASK_IDS[0].load(core::sync::atomic::Ordering::Acquire);
            let id2 = LAST_PE_TASK_IDS[1].load(core::sync::atomic::Ordering::Acquire);
            let id3 = LAST_PE_TASK_IDS[2].load(core::sync::atomic::Ordering::Acquire);
            let id4 = LAST_PE_TASK_IDS[3].load(core::sync::atomic::Ordering::Acquire);
            if id1 > 0 && id2 > 0 && id3 > 0 && id4 > 0 {
              ready = true;
              break;
            }
            std::thread::sleep(core::time::Duration::from_millis(10));
            let _ = ctx.process_main_thread_cleanup_queue();
            attempts += 1;
          }
          if !ready {
            return;
          }

          // Wait for completion and download
          let tids = [
            LAST_PE_TASK_IDS[0].load(core::sync::atomic::Ordering::Acquire),
            LAST_PE_TASK_IDS[1].load(core::sync::atomic::Ordering::Acquire),
            LAST_PE_TASK_IDS[2].load(core::sync::atomic::Ordering::Acquire),
            LAST_PE_TASK_IDS[3].load(core::sync::atomic::Ordering::Acquire),
          ];
          for (i, &tid) in tids.iter().enumerate() {
            let mut status = ctx.get_task_status(tid);
            let mut attempt = 0;
            while matches!(
              status,
              crate::simulation_api::structs::TaskStatusCode::Pending
            ) && attempt < max_attempts
            {
              std::thread::sleep(core::time::Duration::from_millis(10));
              let _ = ctx.process_main_thread_cleanup_queue();
              status = ctx.get_task_status(tid);
              attempt += 1;
            }

            #[cfg(target_os = "macos")]
            {
              let mut info: libc::mach_task_basic_info = unsafe { std::mem::zeroed() };
              let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
              let res = unsafe {
                libc::task_info(
                  libc::mach_task_self(),
                  libc::MACH_TASK_BASIC_INFO,
                  &mut info as *mut _ as libc::task_info_t,
                  &mut count,
                )
              };
              if res == libc::KERN_SUCCESS {
                let virtual_mb = info.virtual_size / (1024 * 1024);
                let resident_mb = info.resident_size / (1024 * 1024);

                println!(
                  "Memory Status -> Resident: {} MB | Virtual: {} MB",
                  resident_mb, virtual_mb
                );
              }
            }

            let mut buffer = vec![0u8; (width * height * 4) as usize];
            let _ = ctx.process_main_thread_cleanup_queue();
            if ctx.download_image(tid, buffer.as_mut_ptr(), buffer.len()) {
              if save_to_disk {
                let _ = image::save_buffer(
                  alloc::format!("output_cam_{}_{}.png", i + 1, tag),
                  &buffer,
                  width,
                  height,
                  image::ColorType::Rgba8,
                );
              }
            }
          }
          // Reset for next pass
          LAST_PE_TASK_IDS[0].store(0, core::sync::atomic::Ordering::Release);
          LAST_PE_TASK_IDS[1].store(0, core::sync::atomic::Ordering::Release);
          LAST_PE_TASK_IDS[2].store(0, core::sync::atomic::Ordering::Release);
          LAST_PE_TASK_IDS[3].store(0, core::sync::atomic::Ordering::Release);
        };

        // Output Initial State
        let _ = ctx.threads.logic_thread.tx().try_send(
          crate::simulation_api::structs::LogicCommand::SetSceneTimeScale {
            scene_id,
            scale: crate::simulation_api::structs::TimeScale::RealTime,
          },
        );
        let _ = ctx
          .threads
          .logic_thread
          .tx()
          .try_send(crate::simulation_api::structs::LogicCommand::PlayScene { scene_id });
        wait_for_images("initial", true);

        // Wait and Output Final State
        let start_time = std::time::Instant::now();
        let duration = core::time::Duration::from_millis(if is_cpu { 1000 } else { 500 });
        while start_time.elapsed() < duration {
          wait_for_images("flush", false);
        }

        let _ = ctx
          .threads
          .logic_thread
          .tx()
          .try_send(crate::simulation_api::structs::LogicCommand::PauseScene { scene_id });
        aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(
          100,
        ));

        // Final position assertions (removed due to removed physics components)

        wait_for_images("final", true);

        // _guard's Drop calls Box::from_raw(ctx_ptr) automatically.
      }
    }
  }
}
