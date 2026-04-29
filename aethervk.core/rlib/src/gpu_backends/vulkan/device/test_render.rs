use crate::{
    gpu::{
        self,
        new_render_frontend,
        DeviceAdditionalParams,
        PresentationEngineParams,
        frame::{BillboardDrawCall, CursorDrawCall, RenderScene},
        VULKAN_RENDER_BACKEND
    },
    // TODO render module shouldn't be aware of ECS
    scene::{
        BillboardType, CameraComponent, GridComponent,
        Scene, SkyComponent, SunComponent, TransformComponent,
    },
    types::RuntimeParams,
};
use aethervk_oshal_rlib::math::{
    matrix::{mat4::Mat4x4f32, Matrix4, SquareMatrix},
    quaternion::Quaternion,
    vector::{vec3::Vec3f32, vec4::Quat},
};
use heapless::index_map::FnvIndexMap;

// TODO: test about text rendering in different fonts (system font and packaged font)

fn setup_assets_dir() {
    let mut home_dir = std::env::current_exe().unwrap();
    let mut iter = 0;
    while !home_dir.join("assets").is_dir() && iter < 32 {
      home_dir.pop();
      iter += 1;
    }
    *crate::gpu::ASSET_DIR.write() = Some(home_dir.join("assets").to_str().unwrap().to_string());
}

#[test]
fn test_render_particles_windowless() {
  setup_assets_dir();
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend
    .write()
    .init_device(0, &additional_params)
    .unwrap();

  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  render_frontend
    .with_device(render_device_handle, |device| {
      device.wire_callbacks(pool_arc.clone())
    })
    .unwrap();

  let width = 256;
  let height = 256;

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
        device.init_archetypes(pe)?;
        crate::types::GpuResult::Ok(pe)
      })
      .unwrap()
  };

  let scene = Scene::new();

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;

      let mut render_scene = RenderScene::new((
        TransformComponent {
          position: Vec3f32::from_array([0.0, 10.0, 0.0]),
          rotation: Quat::identity(),
          scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
        },
        CameraComponent {
          projection: Mat4x4f32::perspective_vk(45.0f32.to_radians(), 1.0, 0.1, 100.0),
          near_plane: 0.1,
          far_plane: 100.0,
        },
      ));

      let particle_sys_e = scene.spawn_entity("particles");
      let mut sys = crate::scene::particles::ParticleSystemComponent::new(
        crate::scene::particles::ParticleEmitterConfig {
          uv_center: [0.5, 0.5],
          uv_radius: 0.1,
          delta: 1000,
          max_particles: 10,
          velocity_intensity: crate::scene::particles::GaussianParams { mean: 1.0, std_dev: 0.0, min: 0.0, max: 1.0 },
          emission_count: crate::scene::particles::GaussianParams { mean: 1.0, std_dev: 0.0, min: 0.0, max: 1.0 },
          particle_radius: 2.0,
          density: 1.0,
          lifetime: 1000000,
          color: [1.0, 0.5, 0.25, 1.0],
          beta: 0.0,
        }
      );
      
      let mut p = crate::scene::particles::ParticleData {
        id_low: 0,
        id_high: 0,
        _pad0: [0; 2],
        position: [0.0, 0.0, 0.0],
        _pad1: 0,
        velocity: [0.0, 0.0, 0.0],
        age_low: 0,
        age_high: 0,
        mass: 1.0,
        active: 1,
        _pad2: 0,
      };
      p.set_id(0);
      p.set_age(0);
      sys.particles.push(p);

      render_scene.add_renderable(
        device,
        particle_sys_e,
        Mat4x4f32::identity(),
        crate::scene::RenderableDataRef::ParticleSystem(&sys),
        presentation_engine,
        "particle_sys_test",
        false,
        [0.0, 0.0, 0.0, 0.0],
      )?;

      {
        let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

        let extent = device.get_presentation_engine_extent(presentation_engine)?;
        device.set_viewport(
          cmd_buffer_handle,
          &gpu::Viewport {
            x: 0.0,
            y: extent[1] as f32,
            width: extent[0] as f32,
            height: -(extent[1] as f32),
            min_depth: 0.0,
            max_depth: 1.0,
          },
        )?;

        device.set_scissor(
          cmd_buffer_handle,
          &gpu::Rect2D {
            offset: [0, 0],
            extent,
          },
        )?;

        device.render_frame(cmd_buffer_handle, &render_scene)?;
        scoped_rp.end()?;
        device.record_windowless_download(cmd_buffer_handle, presentation_engine, task_id)?;
      }

      device.present(
        presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      )?;

      while !device.is_task_completed(task_id)? {
        std::thread::sleep(std::time::Duration::from_millis(10));
      }

      let mut buffer = vec![0u8; (width * height * 4) as usize];
      device.read_windowless_download(task_id, &mut buffer)?;

      let mut found_color = false;
      for chunk in buffer.chunks_exact(4) {
        // BGRA format
        let b = chunk[0];
        let g = chunk[1];
        let r = chunk[2];
        if r > 200 && g > 100 && b > 50 {
          found_color = true;
          break;
        }
      }

      let mut export_buffer = buffer.clone();
      for chunk in export_buffer.chunks_exact_mut(4) {
        chunk.swap(0, 2); // BGRA to RGBA
      }
      let row_stride = (width * 4) as usize;
      for y in 0..(height as usize / 2) {
        let top_row_start = y * row_stride;
        let bottom_row_start = ((height as usize) - 1 - y) * row_stride;
        for x in 0..row_stride {
          export_buffer.swap(top_row_start + x, bottom_row_start + x);
        }
      }
      image::save_buffer(
          "rendered_particles.png",
          &export_buffer,
          width,
          height,
          image::ColorType::Rgba8,
      ).expect("Failed to save rendered png");

      assert!(found_color, "Particle color not found in the rendered image!");

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}

#[test]
fn test_render_all_archetypes_windowless() {
  setup_assets_dir();
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend
    .write()
    .init_device(0, &additional_params)
    .unwrap();

  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  render_frontend
    .with_device(render_device_handle, |device| {
      device.wire_callbacks(pool_arc.clone())
    })
    .unwrap();

  let width = 256;
  let height = 256;

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
        device.init_archetypes(pe)?;
        device.generate_sky()?;
        crate::types::GpuResult::Ok(pe)
      })
      .unwrap()
  };

  let scene = Scene::new();
  let sky_e = scene.spawn_entity("sky");
  let sun_e = scene.spawn_entity("sun");
  let grid_e = scene.spawn_entity("grid");

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;

      let cursor_res = device.get_or_create_cursor_resources(presentation_engine)?;
      let billboard_res = device.get_or_create_billboard_resources(presentation_engine)?;

      let mut render_scene = RenderScene::new((
        TransformComponent {
          position: Vec3f32::from_array([0.0, 10.0, 0.0]),
          rotation: Quat::identity(),
          scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
        },
        CameraComponent {
          projection: Mat4x4f32::perspective_vk(45.0f32.to_radians(), 1.0, 0.1, 100.0),
          near_plane: 0.1,
          far_plane: 100.0,
        },
      ));

      let sky_pipeline = device.get_sky_pipeline_key()?;
      render_scene.sky_call = Some(gpu::frame::SkyDrawCall::from_camera(&render_scene.camera_data, sky_pipeline)?);

      let sun_pipeline = device.get_sun_pipeline_key()?;
      render_scene.sun_call = Some(gpu::frame::SunDrawCall::from_model_and_camera(
        Mat4x4f32::identity(),
        &render_scene.camera_data,
        sun_pipeline,
        sun_e,
      )?);

      let grid_pipeline = device.get_grid_pipeline_kay()?;
      render_scene.grid_call = Some(gpu::frame::GridDrawCall::new(grid_pipeline, 0.1, 1.0, [0.5, 0.5, 0.5]));

      let gizmo_resources = device.get_or_create_gizmo_resources(presentation_engine)?;
      let gizmo_idx = device.update_gizmo_instance(sun_e, Mat4x4f32::identity())?;
      render_scene.gizmo_calls.push(gpu::frame::GizmoDrawCall::from_values(
        gizmo_resources.pipeline,
        2.0,
        gizmo_idx,
      ));

      render_scene.cursor_call = Some(CursorDrawCall {
        pipeline: cursor_res.pipeline,
        vertex_count: 36,
        model_matrix: Mat4x4f32::identity(),
        cursor_size: 0.05,
      });

      render_scene.billboard_calls.push(BillboardDrawCall {
        pipeline: billboard_res.pipeline,
        texture_id: 0,
        vertex_count: 4,
        model_matrix: Mat4x4f32::identity(),
        billboard_type: BillboardType::WorldSpace {
          width: 1.0,
          height: 1.0,
        },
      });

      {
        let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

        device.update_sun(cmd_buffer_handle, sun_e, (64, 64, 64))?;

        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

        let extent = device.get_presentation_engine_extent(presentation_engine)?;
        device.set_viewport(
          cmd_buffer_handle,
          &gpu::Viewport {
            x: 0.0,
            y: extent[1] as f32,
            width: extent[0] as f32,
            height: -(extent[1] as f32),
            min_depth: 0.0,
            max_depth: 1.0,
          },
        )?;

        device.set_scissor(
          cmd_buffer_handle,
          &gpu::Rect2D {
            offset: [0, 0],
            extent,
          },
        )?;

        device
          .render_frame(cmd_buffer_handle, &render_scene)
          .map_err(|e| {
            println!("TR: render_frame failed {:?}", e);
            e
          })?;
        scoped_rp.end()?;
        device.record_windowless_download(cmd_buffer_handle, presentation_engine, task_id)?;
      }

      println!("Before present");
      device.present(
        presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      )?;
      println!("After present");

      // Wait for completion
      while !device.is_task_completed(task_id)? {
        std::thread::sleep(std::time::Duration::from_millis(10));
      }
      
      // Download image
      let mut buffer = vec![0u8; (width * height * 4) as usize];
      device.read_windowless_download(task_id, &mut buffer)?;

      let sum: u32 = buffer.iter().map(|&b| b as u32).sum();
      println!("Sum of buffer is {}", sum);

      assert!(sum > 0, "Buffer is completely empty!");

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  // Stop the thread pool before asserting to avoid hanging
  drop(render_frontend);

  // Validate that there were no Vulkan validation errors
  let errors = super::utils::VULKAN_ERROR_MESSAGES.lock().unwrap();
  if !errors.is_empty() {
    println!("Vulkan Validation Errors encountered:");
    for err in errors.iter() {
      println!("{}", err);
    }
    panic!("Vulkan validation errors occurred during testing");
  }
}

#[test]
fn test_render_empty_scene_graceful() {
  setup_assets_dir();
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend
    .write()
    .init_device(0, &additional_params)
    .unwrap();

  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  render_frontend
    .with_device(render_device_handle, |device| {
      device.wire_callbacks(pool_arc.clone())
    })
    .unwrap();

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(16, 16);
    render_frontend
      .with_device(render_device_handle, |device| {
        device.create_presentation_engine(&params)
      })
      .unwrap()
  };

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;

      let render_scene = RenderScene::new((
        TransformComponent {
          position: Vec3f32::from_array([0.0, 0.0, 0.0]),
          rotation: Quat::identity(),
          scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
        },
        CameraComponent {
          projection: Mat4x4f32::perspective_vk(45.0f32.to_radians(), 1.0, 0.1, 100.0),
          near_plane: 0.1,
          far_plane: 100.0,
        },
      ));

      {
        let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;
        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
        device.set_viewport(
          cmd_buffer_handle,
          &gpu::Viewport {
            x: 0.0,
            y: 16.0,
            width: 16.0,
            height: -16.0,
            min_depth: 0.0,
            max_depth: 1.0,
          },
        )?;
        device.set_scissor(
          cmd_buffer_handle,
          &gpu::Rect2D {
            offset: [0, 0],
            extent: [16, 16],
          },
        )?;
        device
          .render_frame(cmd_buffer_handle, &render_scene)
          .map_err(|e| {
            println!("TR: render_frame failed {:?}", e);
            e
          })?;
        scoped_rp.end()?;
      }
      device.present(
        presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      )?;

      std::thread::sleep(std::time::Duration::from_millis(50));
      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}

#[test]
fn test_layout_transition_on_failed_update() {
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();
  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend
    .write()
    .init_device(0, &additional_params)
    .unwrap();

  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  render_frontend
    .with_device(render_device_handle, |device| {
      device.wire_callbacks(pool_arc.clone())
    })
    .unwrap();

  let presentation_engine = render_frontend
    .with_device(render_device_handle, |device| {
      let params = PresentationEngineParams::windowless(16, 16);
      let pe = device.create_presentation_engine(&params)?;
      // WE DELIBERATELY SKIP init_archetypes(pe) TO CAUSE update_sun TO FAIL!
      crate::types::GpuResult::Ok(pe)
    })
    .unwrap();

  let scene = Scene::new();
  let sun_e = scene.spawn_entity("sun");

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;

      let sun_comp = SunComponent {
        resolution: (64, 64, 64),
      };

      {
        let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

        // This will fail because archetypes aren't initialized, but we catch/ignore it
        // just like the real `simulation_api.rs` does now.
        let _ = device.update_sun(cmd_buffer_handle, sun_e, (64, 64, 64));

        // Even if update_sun failed, begin_render_pass MUST succeed to transition the image!
        // But wait! begin_render_pass relies on rendering archetypes? NO, it relies on RenderPasses cache!
        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
        scoped_rp.end()?;
        device.record_windowless_download(cmd_buffer_handle, presentation_engine, task_id)?;
      }

      device.present(
        presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      )?;

      while !device.is_task_completed(task_id)? {
        std::thread::sleep(std::time::Duration::from_millis(10));
      }

      let mut buffer = vec![0u8; (16 * 16 * 4) as usize];
      device.read_windowless_download(task_id, &mut buffer)?;

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}

#[test]
fn test_render_text_system_font_async() {
  setup_assets_dir();
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend
    .write()
    .init_device(0, &additional_params)
    .unwrap();

  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  render_frontend
    .with_device(render_device_handle, |device| {
      device.wire_callbacks(pool_arc.clone())
    })
    .unwrap();

  let width = 512;
  let height = 256;

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
        device.init_archetypes(pe)?;
        crate::types::GpuResult::Ok(pe)
      })
      .unwrap()
  };

  // Run rendering and polling in separate threads to test async capability
  let frontend_clone = render_frontend.clone();
  
  let render_thread = std::thread::spawn(move || {
    frontend_clone
      .with_device(render_device_handle, |device| {
        let task_id = device.create_task();
        device.start_frame()?;
        let acquire_result = device.acquire_next_image(presentation_engine)?;
        let cmd_buffer_handle = device.get_command_buffer()?;

        // System font test
        let atlas = crate::scene::text::FontAtlas::from_path(
            aethervk_oshal_rlib::os::FONT_PATH,
            32.0,
        ).expect("Failed to load system font");
        
        let atlas_width = atlas.width;
        let atlas_height = atlas.height;
        let atlas_image_data = atlas.image_data.clone();
        
        // Export the atlas for verification
        image::save_buffer(
          "system_font_atlas.png",
          &atlas_image_data,
          atlas_width,
          atlas_height,
          image::ColorType::L8,
        ).expect("Failed to save atlas png");

        // Verify glyph presence
        let glyph_info = atlas.glyphs.get(&'A').expect("A not found in font");
        assert!(glyph_info.size[0] > 0.0 && glyph_info.size[1] > 0.0);

        let font_hash = atlas.hash_metadata();
        let font_id = device.allocate_rasterized_font_atlas(font_hash, atlas)?;

        {
          let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;
          device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
          let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

          device.set_viewport(
            cmd_buffer_handle,
            &gpu::Viewport {
              x: 0.0,
              y: height as f32,
              width: width as f32,
              height: -(height as f32),
              min_depth: 0.0,
              max_depth: 1.0,
            },
          )?;

          device.set_scissor(
            cmd_buffer_handle,
            &gpu::Rect2D {
              offset: [0, 0],
              extent: [width, height],
            },
          )?;

          device.prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer_handle)?;
          device.render_text(
            cmd_buffer_handle,
            "AetherVk Async Test",
            [-0.8, -0.8], // NDC space
            [width as f32, height as f32],
            (font_hash, font_id),
            48.0,
            [0.5, 1.0, 0.5, 1.0],
          )?;

          scoped_rp.end()?;
          device.record_windowless_download(cmd_buffer_handle, presentation_engine, task_id)?;
        }

        device.present(
          presentation_engine,
          acquire_result.image_index as usize,
          acquire_result.frame_index as usize,
        )?;

        crate::types::GpuResult::Ok(task_id)
      })
      .unwrap()
  });

  let task_id = render_thread.join().unwrap();

  // Async polling in the main thread
  render_frontend
    .with_device(render_device_handle, |device| {
      while !device.is_task_completed(task_id)? {
        std::thread::yield_now();
      }

      let mut buffer = vec![0u8; (width * height * 4) as usize];
      device.read_windowless_download(task_id, &mut buffer)?;

      // Verify the rendering wasn't completely empty
      let sum: u64 = buffer.iter().map(|&b| b as u64).sum();
      assert!(sum > 0, "Rendered text buffer is completely empty!");
      
      let mut unique_colors = std::collections::HashSet::new();
      for chunk in buffer.chunks_exact(4) {
        unique_colors.insert((chunk[0], chunk[1], chunk[2], chunk[3]));
      }
      assert!(unique_colors.len() > 1, "Image is completely uniform color!");
      
      // Convert BGRA to RGBA
      for chunk in buffer.chunks_exact_mut(4) {
        chunk.swap(0, 2);
      }

      // Flip vertically
      let row_stride = (width * 4) as usize;
      for y in 0..(height as usize / 2) {
        let top_row_start = y * row_stride;
        let bottom_row_start = ((height as usize) - 1 - y) * row_stride;
        for x in 0..row_stride {
          buffer.swap(top_row_start + x, bottom_row_start + x);
        }
      }

      // Export rendered text
      image::save_buffer(
          "rendered_system_text.png",
          &buffer,
          width,
          height,
          image::ColorType::Rgba8,
      ).expect("Failed to save rendered png");

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}

#[test]
fn test_render_text_asset_font_async() {
  setup_assets_dir();
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend
    .write()
    .init_device(0, &additional_params)
    .unwrap();

  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  render_frontend
    .with_device(render_device_handle, |device| {
      device.wire_callbacks(pool_arc.clone())
    })
    .unwrap();

  let width = 512;
  let height = 256;

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
        device.init_archetypes(pe)?;
        crate::types::GpuResult::Ok(pe)
      })
      .unwrap()
  };

  // Run rendering and polling in separate threads to test async capability
  let frontend_clone = render_frontend.clone();
  
  let render_thread = std::thread::spawn(move || {
    frontend_clone
      .with_device(render_device_handle, |device| {
        let task_id = device.create_task();
        device.start_frame()?;
        let acquire_result = device.acquire_next_image(presentation_engine)?;
        let cmd_buffer_handle = device.get_command_buffer()?;

        // Use font under asset folder
        let asset_font_path = format!("{}/fonts/JetBrainsMono-Regular.ttf", crate::gpu::ASSET_DIR.read().as_ref().unwrap());
        let atlas = crate::scene::text::FontAtlas::from_path(
            &asset_font_path,
            32.0,
        ).expect("Failed to load asset font");
        
        let atlas_width = atlas.width;
        let atlas_height = atlas.height;
        let atlas_image_data = atlas.image_data.clone();
        
        // Export the atlas for verification
        image::save_buffer(
          "asset_font_atlas.png",
          &atlas_image_data,
          atlas_width,
          atlas_height,
          image::ColorType::L8,
        ).expect("Failed to save atlas png");

        // Verify glyph presence
        let glyph_info = atlas.glyphs.get(&'W').expect("W not found in font");
        assert!(glyph_info.size[0] > 0.0 && glyph_info.size[1] > 0.0);

        let font_hash = atlas.hash_metadata();
        let font_id = device.allocate_rasterized_font_atlas(font_hash, atlas)?;

        {
          let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;
          device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
          let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

          device.set_viewport(
            cmd_buffer_handle,
            &gpu::Viewport {
              x: 0.0,
              y: height as f32,
              width: width as f32,
              height: -(height as f32),
              min_depth: 0.0,
              max_depth: 1.0,
            },
          )?;

          device.set_scissor(
            cmd_buffer_handle,
            &gpu::Rect2D {
              offset: [0, 0],
              extent: [width, height],
            },
          )?;

          device.prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer_handle)?;
          device.render_text(
            cmd_buffer_handle,
            "AetherVk Async Test",
            [-0.8, -0.8], // NDC space
            [width as f32, height as f32],
            (font_hash, font_id),
            48.0,
            [0.5, 1.0, 0.5, 1.0],
          )?;

          scoped_rp.end()?;
          device.record_windowless_download(cmd_buffer_handle, presentation_engine, task_id)?;
        }

        device.present(
          presentation_engine,
          acquire_result.image_index as usize,
          acquire_result.frame_index as usize,
        )?;

        crate::types::GpuResult::Ok(task_id)
      })
      .unwrap()
  });

  let task_id = render_thread.join().unwrap();

  // Async polling in the main thread
  render_frontend
    .with_device(render_device_handle, |device| {
      while !device.is_task_completed(task_id)? {
        std::thread::yield_now();
      }

      let mut buffer = vec![0u8; (width * height * 4) as usize];
      device.read_windowless_download(task_id, &mut buffer)?;

      // Verify the rendering wasn't completely empty
      let sum: u64 = buffer.iter().map(|&b| b as u64).sum();
      assert!(sum > 0, "Rendered text buffer is completely empty!");
      
      let mut unique_colors = std::collections::HashSet::new();
      for chunk in buffer.chunks_exact(4) {
        unique_colors.insert((chunk[0], chunk[1], chunk[2], chunk[3]));
      }
      assert!(unique_colors.len() > 1, "Image is completely uniform color!");
      
      // Convert BGRA to RGBA
      for chunk in buffer.chunks_exact_mut(4) {
        chunk.swap(0, 2);
      }

      // Flip vertically
      let row_stride = (width * 4) as usize;
      for y in 0..(height as usize / 2) {
        let top_row_start = y * row_stride;
        let bottom_row_start = ((height as usize) - 1 - y) * row_stride;
        for x in 0..row_stride {
          buffer.swap(top_row_start + x, bottom_row_start + x);
        }
      }

      // Export rendered text
      image::save_buffer(
          "rendered_asset_text.png",
          &buffer,
          width,
          height,
          image::ColorType::Rgba8,
      ).expect("Failed to save rendered png");

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}
