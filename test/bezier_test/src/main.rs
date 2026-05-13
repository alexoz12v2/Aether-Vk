#[cfg(all(debug_assertions, feature = "debug_gpu"))]
#[global_allocator]
static ALLOC: aethervk_oshal_rlib::os::memory::tracking::TrackingAllocator<std::alloc::System> = 
  aethervk_oshal_rlib::os::memory::tracking::TrackingAllocator(std::alloc::System);

use aethervk_core_rlib::gpu::{
  new_render_frontend, DeviceAdditionalParams, PresentationEngineParams, VULKAN_RENDER_BACKEND,
};
use aethervk_core_rlib::scene::{
  trajectory::TrajectoryComponent, CameraComponent, EntityId, Scene, TransformComponent,
};
use aethervk_core_rlib::types::RuntimeParams;
use aethervk_oshal_rlib::math::matrix::{mat4::Mat4x4f32, Matrix4};
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::{vec3::Vec3f32, vec4::Quat};
use aethervk_oshal_rlib::os::time::timeus_milliseconds;
use test_utils::{cycle_get_asset_path_from_exe, AppEvent};
use winit::window::Window;

fn panic_on_validation_error(msg: &str) {
  panic!("Vulkan validation error occurred during testing: {}", msg);
}

fn main() {
  *aethervk_core_rlib::gpu::ASSET_DIR.write() =
    Some(cycle_get_asset_path_from_exe(true).to_string_lossy().to_string());

  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(3).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: Default::default(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });

  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();
  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend.write().init_device(0, &additional_params).unwrap();

  render_frontend
    .with_device(render_device_handle, |device| {
      device.wire_callbacks(pool_arc.clone())
    })
    .unwrap();

  let width = 800;
  let height = 600;

  let mut event_loop_opt: Option<winit::event_loop::EventLoop<AppEvent>> = None;
  let mut window_opt: Option<Window> = None;

  #[cfg(feature = "windowed")]
  {
    let mut event_loop_builder =
      winit::event_loop::EventLoopBuilder::<test_utils::AppEvent>::with_user_event();
    #[cfg(target_os = "macos")]
    {
      use winit::platform::macos::EventLoopBuilderExtMacOS;
      event_loop_builder.with_default_menu(false);
    }

    let event_loop = event_loop_builder.build().unwrap();
    let proxy = event_loop.create_proxy();
    let proxy_ptr = unsafe { std::ptr::NonNull::new_unchecked(Box::into_raw(Box::new(proxy))) };

    let window = winit::window::WindowBuilder::new()
      .with_title("Bezier Trajectory Test")
      .with_inner_size(winit::dpi::PhysicalSize::new(width, height))
      .build(&event_loop)
      .unwrap();
    test_utils::setup_resize_hook(&window, proxy_ptr);

    event_loop_opt = Some(event_loop);
    window_opt = Some(window);
  }

  let (presentation_engine, _window_info) = {
    #[cfg(feature = "windowed")]
    {
      let window = window_opt.as_ref().unwrap();
      let (native_handles, window_info) = test_utils::get_handle_and_window_info_create_layer(
        &render_frontend,
        render_device_handle,
        window,
      );

      #[cfg(target_os = "macos")]
      {
        window_info.metal_layer.setDrawableSize(objc2_core_foundation::CGSize {
          width: width as f64,
          height: height as f64,
        });
      }

      let params = PresentationEngineParams {
        ty: aethervk_core_rlib::gpu::PresentationEngineType::Window,
        window_info: native_handles,
        width,
        height,
        vsync: true,
      };
      (
        render_frontend
          .with_device(render_device_handle, |device| {
            let pe = device.create_presentation_engine(&params)?;
            device.init_archetypes(pe)?;
            aethervk_core_rlib::types::GpuResult::Ok(pe)
          })
          .unwrap(),
        window_info,
      )
    }
    #[cfg(not(feature = "windowed"))]
    {
      let params = PresentationEngineParams::windowless(width, height);
      (
        render_frontend
          .with_device(render_device_handle, |device| {
            let pe = device.create_presentation_engine(&params)?;
            device.init_archetypes(pe)?;
            aethervk_core_rlib::types::GpuResult::Ok(pe)
          })
          .unwrap(),
        (),
      )
    }
  };

  let scene = Scene::new(std::sync::Arc::new(spin::RwLock::new(aethervk_core_rlib::simulation::texture_cache::TextureCache::new("AetherVk"))));
  scene.register_all_crate_components();

  let root_e = scene.spawn_entity("root");
  scene
    .add_component(
      root_e,
      TransformComponent {
        position: Vec3f32::from_array([0.0, 0.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();

  let orbit_root_e = scene.spawn_entity("orbit_root");
  scene.set_parent(orbit_root_e, Some(root_e));
  scene
    .add_component(
      orbit_root_e,
      TransformComponent {
        position: Vec3f32::from_array([0.0, 0.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();

  let cam_e = scene.spawn_entity("camera");
  scene.set_parent(cam_e, Some(root_e));
  scene
    .add_component(
      cam_e,
      TransformComponent {
        position: Vec3f32::from_array([0.0, 50.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();
  scene
    .add_component(
      cam_e,
      CameraComponent::new_persp(
          45.0f32.to_radians(), width as f32 / height as f32, 0.1, 100.0,
        ),
    )
    .unwrap();

  // Create a single undulating infinity symbol
  let traj_e = scene.spawn_entity("infinity_orbit");
  scene.set_parent(traj_e, Some(orbit_root_e));

  let mut cps = std::vec::Vec::new();
  // Provide enough points (e.g. 32 segments * 4 points = 128 points) to allow smooth chasing.
  // The actual positions are completely driven by `animate_infinity_chase`, so we just initialize with zeros.
  for _ in 0..128 {
    cps.push([0.0, 0.0, 0.0, 1.0]);
  }

  scene
    .add_component(
      traj_e,
      TransformComponent {
        position: Vec3f32::from_array([0.0, 0.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();
  scene
    .add_component(
      traj_e,
      TrajectoryComponent {
        control_points: cps,
        color: [1.0, 1.0, 0.0, 1.0], // Yellow
        line_width: 5.0,             // Make it thick to see easily
        texture_id: 0,
        subdivisions_per_segment: 32, // Smooth curve
      },
    )
    .unwrap();

  let shared_width = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(width));
  let shared_height = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(height));
  let quit_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

  let r_width = shared_width.clone();
  let r_height = shared_height.clone();
  let r_quit = quit_flag.clone();
  let render_frontend_bg = render_frontend.clone();
  let pool_arc_clone = pool_arc.clone();

  let render_thread = std::thread::spawn(move || {
    let mut frames = 0;
    let mut time_readings = aethervk_oshal_rlib::os::time::TimeInfo::new(
      timeus_milliseconds(16),
      timeus_milliseconds(100),
      1.0,
    );
    let mut total_time_us: aethervk_oshal_rlib::os::time::timeus_t = 0;

    let mut is_windowed = true;
    #[cfg(not(feature = "windowed"))]
    {
      is_windowed = false;
    }

    while !r_quit.load(std::sync::atomic::Ordering::Relaxed) {
      let loop_start = std::time::Instant::now();
      time_readings.ut_update();
      let dt_us = 16_666;

      let current_width = r_width.load(std::sync::atomic::Ordering::Relaxed);
      let current_height = r_height.load(std::sync::atomic::Ordering::Relaxed);

      if current_width > 0 && current_height > 0 {
        scene.with_component_mut(traj_e, |traj: &mut TrajectoryComponent| {
          let scale_x = 10.0;
          let scale_y = 5.0;
          let speed = 2.0;
          traj.animate_infinity_chase(dt_us, &mut total_time_us, scale_x, scale_y, speed);
        });

        let mut should_exit_headless = false;
        #[cfg(target_os = "macos")]
        objc2::rc::autoreleasepool(|_| {
          render_frontend_bg.with_device(render_device_handle, |device| {
            let task_id = device.create_task();
            device.start_frame()?;
            let acquire_result = device.acquire_next_image(presentation_engine)?;
            if acquire_result.status.needs_resize() {
              device.success_task(task_id);
              return Ok(());
            }
            let cmd_buffer_handle = device.get_command_buffer()?;
            let _scoped_cmd =
              aethervk_core_rlib::gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

            use aethervk_core_rlib::gpu::scene_conversion::SceneConversionExt;
            let extracted =
              scene.convert_scene(cam_e, false, Some(&*pool_arc_clone), [current_width, current_height]).unwrap();
            let render_scene = extracted
              .build_render_scene(
                device,
                presentation_engine,
                cmd_buffer_handle,
                time_readings.current(),
                [current_width, current_height],
                "bezier_test",
              )
              .unwrap();

            device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
            let scoped_rp = aethervk_core_rlib::gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

            device.set_viewport(
              cmd_buffer_handle,
              &aethervk_core_rlib::gpu::Viewport::from_extent([current_width, current_height]),
            )?;
            device.set_scissor(
              cmd_buffer_handle,
              &aethervk_core_rlib::gpu::Rect2D::from_extent([current_width, current_height]),
            )?;

            aethervk_core_rlib::gpu::frame::render_frame(
              device,
              cmd_buffer_handle,
              &render_scene,
              presentation_engine,
            )?;
            scoped_rp.end()?;

            if !is_windowed && frames < 3 {
              device.record_windowless_download(cmd_buffer_handle, presentation_engine, task_id)?;
            }

            _scoped_cmd.submit()?;

            let _present_status = device.present(
              presentation_engine,
              acquire_result.image_index as usize,
              acquire_result.frame_index as usize,
            )?;

            if !is_windowed {
              while !device.is_task_completed(task_id)? {
                std::thread::yield_now();
              }

              if frames < 3 {
                let mut buffer = vec![0u8; (current_width * current_height * 4) as usize];
                device.read_windowless_download(task_id, &mut buffer)?;

                let mut export_buffer = buffer.clone();
                for chunk in export_buffer.chunks_exact_mut(4) {
                  chunk.swap(0, 2);
                }

                image::save_buffer(
                  &format!("test_frame_{}.png", frames),
                  &export_buffer,
                  current_width,
                  current_height,
                  image::ColorType::Rgba8,
                )
                .unwrap();
              }
              if frames >= 3 {
                  should_exit_headless = true;
              }
            }

            Ok::<(), aethervk_core_rlib::types::GpuError>(())
          }).unwrap();
        });

        #[cfg(not(target_os = "macos"))]
        render_frontend_bg.with_device(render_device_handle, |device| {
          let task_id = device.create_task();
          device.start_frame()?;
          let acquire_result = device.acquire_next_image(presentation_engine)?;
          if acquire_result.status.needs_resize() {
            device.success_task(task_id);
            return Ok(());
          }
          let cmd_buffer_handle = device.get_command_buffer()?;
          let _scoped_cmd =
            aethervk_core_rlib::gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

          use aethervk_core_rlib::gpu::scene_conversion::SceneConversionExt;
          let extracted =
            scene.convert_scene(cam_e, false, Some(&*pool_arc_clone), [current_width, current_height]).unwrap();
          let render_scene = extracted
            .build_render_scene(
              device,
              presentation_engine,
              cmd_buffer_handle,
              time_readings.current(),
              [current_width, current_height],
              "bezier_test",
            )
            .unwrap();

          device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
          let scoped_rp = aethervk_core_rlib::gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

          device.set_viewport(
            cmd_buffer_handle,
            &aethervk_core_rlib::gpu::Viewport::from_extent([current_width, current_height]),
          )?;
          device.set_scissor(
            cmd_buffer_handle,
            &aethervk_core_rlib::gpu::Rect2D::from_extent([current_width, current_height]),
          )?;

          aethervk_core_rlib::gpu::frame::render_frame(
            device,
            cmd_buffer_handle,
            &render_scene,
            presentation_engine,
          )?;
          scoped_rp.end()?;

          if !is_windowed && frames < 3 {
            device.record_windowless_download(cmd_buffer_handle, presentation_engine, task_id)?;
          }

          _scoped_cmd.submit()?;

          let _present_status = device.present(
            presentation_engine,
            acquire_result.image_index as usize,
            acquire_result.frame_index as usize,
          )?;

          if !is_windowed {
            while !device.is_task_completed(task_id)? {
              std::thread::yield_now();
            }

            if frames < 3 {
              let mut buffer = vec![0u8; (current_width * current_height * 4) as usize];
              device.read_windowless_download(task_id, &mut buffer)?;

              let mut export_buffer = buffer.clone();
              for chunk in export_buffer.chunks_exact_mut(4) {
                chunk.swap(0, 2);
              }

              image::save_buffer(
                &format!("test_frame_{}.png", frames),
                &export_buffer,
                current_width,
                current_height,
                image::ColorType::Rgba8,
              )
              .unwrap();
            }
            if frames >= 3 {
                should_exit_headless = true;
            }
          }

          Ok::<(), aethervk_core_rlib::types::GpuError>(())
        }).unwrap();

        if should_exit_headless {
            r_quit.store(true, std::sync::atomic::Ordering::Relaxed);
        }

        frames += 1;
        if frames % 60 == 0 {
          let mem = aethervk_oshal_rlib::os::memory::query_process_memory();
          println!("Frame {}: Resident memory: {} MB", frames, mem.physical_bytes / 1024 / 1024);
        }
        #[cfg(all(debug_assertions, feature = "debug_gpu"))]
        {
          if frames % 120 == 0 {
            aethervk_oshal_rlib::os::memory::tracking::print_memory_state();
          }
        }
      }

      let elapsed = loop_start.elapsed();
      let target_dt = std::time::Duration::from_millis(16);
      if elapsed < target_dt {
          std::thread::sleep(target_dt - elapsed);
      }
    }
  });

  #[cfg(feature = "windowed")]
  {
    let event_loop = event_loop_opt.unwrap();
    let window = window_opt.unwrap();

    let mut is_resizing = false;
    let mut needs_resize = false;
    let mut resize_width = 0;
    let mut resize_height = 0;

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);

    event_loop
      .run(move |event, elwt| {
        use winit::event::{Event, WindowEvent};
        match event {
          Event::UserEvent(test_utils::AppEvent::ResizeStarted) => {
            is_resizing = true;
          }
          Event::UserEvent(test_utils::AppEvent::ResizeEnded) => {
            is_resizing = false;
            needs_resize = true;
          }
          Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
          } => {
            quit_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            elwt.exit();
          }
          Event::AboutToWait => {
            if needs_resize && !is_resizing {
              if resize_width > 0 && resize_height > 0 {
                render_frontend
                  .with_device(render_device_handle, |device| {
                    device.resize_presentation_engine(
                      presentation_engine,
                      resize_width,
                      resize_height,
                    )
                  })
                  .unwrap();
                shared_width.store(resize_width, std::sync::atomic::Ordering::Relaxed);
                shared_height.store(resize_height, std::sync::atomic::Ordering::Relaxed);
              }
              needs_resize = false;
            }
          }
          Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
          } => {
            if size.width > 0 && size.height > 0 {
              resize_width = size.width;
              resize_height = size.height;
              #[cfg(target_os = "macos")]
              {
                _window_info.metal_layer.setDrawableSize(objc2_core_foundation::CGSize {
                  width: size.width as f64,
                  height: size.height as f64,
                });
              }
              if !is_resizing {
                needs_resize = true;
              }
            }
          }
          _ => (),
        }
      })
      .unwrap();

    render_thread.join().unwrap();
  }

  #[cfg(not(feature = "windowed"))]
  {
    // Wait for the render thread to finish its 3 frames
    render_thread.join().unwrap();
    println!("Rendered 3 frames in headless mode");

    // Verify animation by comparing pixels
    let img0 = image::open("test_frame_0.png").unwrap().into_rgba8();
    let img1 = image::open("test_frame_1.png").unwrap().into_rgba8();

    let mut diff_pixels = 0;
    let mut colored_pixels_0 = 0;
    for (p0, p1) in img0.pixels().zip(img1.pixels()) {
      if p0[0] > 0 || p0[1] > 0 || p0[2] > 0 {
        colored_pixels_0 += 1;
      }
      if p0 != p1 {
        diff_pixels += 1;
      }
    }

    println!("Total colored pixels in frame 0: {}", colored_pixels_0);
    println!(
      "Total different pixels between frame 0 and 1: {}",
      diff_pixels
    );
    assert!(
      diff_pixels > 0,
      "Animation failed! Frame 0 and 1 are identical."
    );

    // Check connected components of the yellow pixels
    let mut yellow_pixels = std::collections::HashSet::new();
    for (y, row) in img0.enumerate_rows() {
      for (x, _, pixel) in row {
        if pixel[0] > 100 && pixel[1] > 100 && pixel[2] < 50 {
          yellow_pixels.insert((x as i32, y as i32));
        }
      }
    }

    let mut visited = std::collections::HashSet::new();
    let mut components = 0;

    for &start_pixel in &yellow_pixels {
      if !visited.contains(&start_pixel) {
        components += 1;
        let mut queue = std::vec::Vec::new();
        queue.push(start_pixel);
        visited.insert(start_pixel);

        while let Some((cx, cy)) = queue.pop() {
          for dx in -1..=1 {
            for dy in -1..=1 {
              if dx == 0 && dy == 0 {
                continue;
              }
              let nx = cx + dx;
              let ny = cy + dy;
              if yellow_pixels.contains(&(nx, ny)) && !visited.contains(&(nx, ny)) {
                visited.insert((nx, ny));
                queue.push((nx, ny));
              }
            }
          }
        }
      }
    }

    println!("Found {} connected component(s).", components);
    assert!(
      components == 1,
      "The curve is broken into {} disconnected pieces!",
      components
    );
  }
}
