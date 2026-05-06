use super::*;
use crate::gpu::{RenderDeviceHandle, RenderFrontend, ScopedCommandBuffer, ScopedRenderPass};
use crate::scene::PhysicalMeshComponent;
use crate::{
  gpu::{
    self, DeviceAdditionalParams, PresentationEngineParams, VULKAN_RENDER_BACKEND,
    frame::{BillboardDrawCall, CursorDrawCall, RenderScene},
    new_render_frontend,
  },
  // TODO render module shouldn't be aware of ECS
  scene::{
    BillboardType, CameraComponent, GridComponent, Scene, SkyComponent, SunComponent,
    TransformComponent,
  },
  types::RuntimeParams,
};
use aethervk_oshal_rlib::math::vector::Vector3;
use aethervk_oshal_rlib::math::{
  matrix::{Matrix4, SquareMatrix, mat4::Mat4x4f32},
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, vec4::Quat},
};
use heapless::index_map::FnvIndexMap;
use std::any::{Any, TypeId};
use std::sync::Arc;

// TODO: test about text rendering in different fonts (system font and packaged font)
// TODO: move into integration tests folder

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
  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
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

      let mut render_scene = RenderScene::new(
        (
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
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
      );

      let particle_sys_e = scene.spawn_entity("particles");
      let mut sys = crate::scene::particles::ParticleSystemComponent::new(
        crate::scene::particles::ParticleEmitterConfig {
          uv_distribution: crate::math::distribution::Distribution2D::new(
            &[1.0, 1.0, 1.0, 1.0],
            2,
            2,
          ),
          delta: 1000,
          max_particles: 10,
          velocity_intensity: crate::scene::particles::GaussianParams {
            mean: 1.0,
            std_dev: 0.0,
            min: 0.0,
            max: 1.0,
          },
          emission_count: crate::scene::particles::GaussianParams {
            mean: 1.0,
            std_dev: 0.0,
            min: 0.0,
            max: 1.0,
          },
          particle_radius: 2.0,
          density: 1.0,
          lifetime: 1000000,
          color: [1.0, 0.5, 0.25, 1.0],
          beta: 0.0,
        },
      );

      let mut p = crate::scene::particles::ParticleData {
        id_low: 0,
        id_high: 0,
        age_low: 0,
        age_high: 0,
        position: [0.0, 0.0, 0.0],
        mass: 1.0,
        velocity: [0.0, 0.0, 0.0],
        active: 1,
      };
      p.set_id(0);
      p.set_age(0);
      sys.particles.write().push(p);

      render_scene.add_renderable(
        cmd_buffer_handle,
        device,
        particle_sys_e,
        Mat4x4f32::identity(),
        crate::scene::RenderableDataRef::ParticleSystem(&sys),
        presentation_engine,
        "particle_sys_test",
        false,
        [0.0, 0.0, 0.0, 0.0],
      )?;

      let scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;
      device.upload_particle_systems(cmd_buffer_handle, &mut render_scene.particle_calls)?;
      device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
      let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
      let extent = device.get_presentation_engine_extent(presentation_engine)?;

      device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent))?;
      device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent))?;

      gpu::frame::render_frame(
        device,
        cmd_buffer_handle,
        &render_scene,
        presentation_engine,
      )?;
      scoped_rp.end()?;
      device.record_windowless_download(cmd_buffer_handle, presentation_engine, task_id)?;

      scoped_cmd.submit().unwrap();

      device.present(
        presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      )?;

      while !device.is_task_completed(task_id)? {
        std::thread::sleep(std::time::Duration::from_millis(10));
      }
      device.success_task(task_id);

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
        "test_rendered_particles.png",
        &export_buffer,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .expect("Failed to save rendered png");

      assert!(
        found_color,
        "Particle color not found in the rendered image!"
      );

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
  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend.write().init_device(0, &additional_params).unwrap();

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

      let cursor_res = device.get_cursor_resources(presentation_engine)?;
      let billboard_res = device.get_billboard_resources(presentation_engine)?;

      let mut render_scene = RenderScene::new(
        (
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
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
      );

      let sky_pipeline = device.get_sky_pipeline_key(presentation_engine)?;
      render_scene.sky_call = Some(gpu::frame::SkyDrawCall::from_camera(
        &render_scene.camera_data,
        sky_pipeline,
      )?);

      let sun_pipeline = device.get_sun_pipeline_key(presentation_engine)?;
      render_scene.sun_call = Some(gpu::frame::SunDrawCall::from_model_and_camera(
        Mat4x4f32::identity(),
        &render_scene.camera_data,
        sun_pipeline,
        sun_e,
        0.6,
      )?);

      let grid_pipeline = device.get_grid_pipeline_kay(presentation_engine)?;
      render_scene.grid_call = Some(gpu::frame::GridDrawCall::new(
        grid_pipeline,
        0.1,
        1.0,
        [0.5, 0.5, 0.5],
      ));

      let gizmo_resources = device.get_gizmo_resources(presentation_engine)?;
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

        device.update_sun(cmd_buffer_handle, sun_e, (64, 64, 64), 0.6)?;

        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

        let extent = device.get_presentation_engine_extent(presentation_engine)?;
        device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent))?;

        device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent))?;

        gpu::frame::render_frame(
          device,
          cmd_buffer_handle,
          &render_scene,
          presentation_engine,
        )
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
      device.success_task(task_id);

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
  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend.write().init_device(0, &additional_params).unwrap();

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

      let render_scene = RenderScene::new(
        (
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
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [16, 16],
      );

      {
        let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;
        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
        device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent([16, 16]))?;
        device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent([16, 16]))?;
        gpu::frame::render_frame(
          device,
          cmd_buffer_handle,
          &render_scene,
          presentation_engine,
        )
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
  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();
  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend.write().init_device(0, &additional_params).unwrap();

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
        radius: 0.6,
      };

      {
        let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

        // This will fail because archetypes aren't initialized, but we catch/ignore it
        // just like the real `simulation_api.rs` does now.
        let _ = device.update_sun(cmd_buffer_handle, sun_e, (64, 64, 64), 0.6);

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
      device.success_task(task_id);

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
  // Must be dropped after render frontend!
  let pool_arc = std::sync::Arc::new(aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap());

  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
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
        let atlas =
          crate::scene::text::FontAtlas::from_path(aethervk_oshal_rlib::os::FONT_PATH, 32.0)
            .expect("Failed to load system font");

        let atlas_width = atlas.width;
        let atlas_height = atlas.height;
        let atlas_image_data = atlas.image_data.clone();

        // Export the atlas for verification
        image::save_buffer(
          "test_system_font_atlas.png",
          &atlas_image_data,
          atlas_width,
          atlas_height,
          image::ColorType::L8,
        )
        .expect("Failed to save atlas png");

        // Verify glyph presence
        let glyph_info = atlas.glyphs.get(&'A').expect("A not found in font");
        assert!(glyph_info.size[0] > 0.0 && glyph_info.size[1] > 0.0);

        let font_hash = atlas.hash_metadata();

        {
          let _scoped_cmd =
            gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

          let font_id =
            device.allocate_rasterized_font_atlas(cmd_buffer_handle, font_hash, atlas)?;

          device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
          let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

          device.set_viewport(
            cmd_buffer_handle,
            &gpu::Viewport::from_extent([width, height]),
          )?;

          device.set_scissor(
            cmd_buffer_handle,
            &gpu::Rect2D::from_extent([width, height]),
          )?;

          device.prepare_text_archetype_for_render_and_bind_pipeline(
            cmd_buffer_handle,
            presentation_engine,
          )?;
          device.render_text(
            cmd_buffer_handle,
            "AetherVk Async Test",
            [-0.8, 0.0], // NDC space
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

      // Export rendered text
      // Convert BGRA to RGBA? TODO: Downloaded image should also return pixel format, we can't just assume BGRA
      image::save_buffer(
        "test_rendered_system_text.png",
        &buffer,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .expect("Failed to save rendered png");

      let mut unique_colors = std::collections::HashSet::new();
      for chunk in buffer.chunks_exact(4) {
        unique_colors.insert((chunk[0], chunk[1], chunk[2], chunk[3]));
        if unique_colors.len() > 1 {
          break;
        }
      }
      assert!(
        unique_colors.len() > 1,
        "Image is completely uniform color!"
      );

      Ok(())
    })
    .unwrap();
}

#[test]
fn test_render_text_asset_font_async() {
  setup_assets_dir();
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let pool_arc = std::sync::Arc::new(aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap());
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
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

  let width = 512;
  let height = 256;

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
        device.init_archetypes(pe)?;
        Ok(pe)
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
        let asset_font_path = format!(
          "{}/fonts/JetBrainsMono-Regular.ttf",
          crate::gpu::ASSET_DIR.read().as_ref().unwrap()
        );
        let atlas = crate::scene::text::FontAtlas::from_path(&asset_font_path, 32.0)
          .expect("Failed to load asset font");

        let atlas_width = atlas.width;
        let atlas_height = atlas.height;
        let atlas_image_data = atlas.image_data.clone();

        // Export the atlas for verification
        image::save_buffer(
          "test_asset_font_atlas.png",
          &atlas_image_data,
          atlas_width,
          atlas_height,
          image::ColorType::L8,
        )
        .expect("Failed to save atlas png");

        // Verify glyph presence
        let glyph_info = atlas.glyphs.get(&'W').expect("W not found in font");
        assert!(glyph_info.size[0] > 0.0 && glyph_info.size[1] > 0.0);

        let font_hash = atlas.hash_metadata();

        {
          let _scoped_cmd =
            gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

          let font_id =
            device.allocate_rasterized_font_atlas(cmd_buffer_handle, font_hash, atlas)?;

          device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
          let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

          device.set_viewport(
            cmd_buffer_handle,
            &gpu::Viewport::from_extent([width, height]),
          )?;

          device.set_scissor(
            cmd_buffer_handle,
            &gpu::Rect2D::from_extent([width, height]),
          )?;

          device.prepare_text_archetype_for_render_and_bind_pipeline(
            cmd_buffer_handle,
            presentation_engine,
          )?;
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
      assert!(
        unique_colors.len() > 1,
        "Image is completely uniform color!"
      );

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
        "test_rendered_asset_text.png",
        &buffer,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .expect("Failed to save rendered png");

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}

#[test]
fn test_render_particles_multithreaded() {
  setup_assets_dir();
  fn multithread_panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(multithread_panic_on_validation_error as fn(&str)),
  });
  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(3).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend.write().init_device(0, &additional_params).unwrap();

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
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<crate::scene::PhysicalMeshComponent>(&[std::any::TypeId::of::<
    TransformComponent,
  >()]);
  scene.register_component::<CameraComponent>(&[std::any::TypeId::of::<TransformComponent>()]);
  scene.register_component::<crate::scene::particles::ParticleSystemComponent>(&[
    std::any::TypeId::of::<TransformComponent>(),
  ]);
  scene.register_component::<SunComponent>(&[std::any::TypeId::of::<TransformComponent>()]);

  let mesh_entity = scene.spawn_entity("mesh");
  let particle_sys_e = scene.spawn_entity("particles");
  let sun_e = scene.spawn_entity("sun");

  let asset_path = format!(
    "{}/Comet.glb",
    crate::gpu::ASSET_DIR.read().as_ref().unwrap()
  );
  let loaded_mesh = crate::simulation::comet::load_comet_from_gltf(&asset_path, false)
    .expect("Failed to load mesh");
  let mesh_arc = std::sync::Arc::from(loaded_mesh);

  scene
    .add_component(
      sun_e,
      TransformComponent {
        position: Vec3f32::from_array([10.0, 10.0, 10.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();

  scene
    .add_component(
      sun_e,
      SunComponent {
        resolution: (64, 64, 64),
        radius: 0.6,
      },
    )
    .unwrap();

  scene
    .add_component(
      mesh_entity,
      TransformComponent {
        position: Vec3f32::from_array([0.0, 0.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();

  scene
    .add_component(
      mesh_entity,
      crate::scene::PhysicalMeshComponent {
        asset_path: asset_path.clone(),
        mesh: mesh_arc.clone(),
        emissive_intensity: 1.0,
        emissive_color: [0.5, 0.5, 0.5], // Emissive gray
      },
    )
    .unwrap();

  let sys = crate::scene::particles::ParticleSystemComponent::new(
    crate::scene::particles::ParticleEmitterConfig {
      uv_distribution: crate::math::distribution::Distribution2D::new(&[1.0, 1.0, 1.0, 1.0], 2, 2),
      delta: 1000,
      max_particles: 100,
      velocity_intensity: crate::scene::particles::GaussianParams {
        mean: 5.0,
        std_dev: 0.1,
        min: 0.0,
        max: 10.0,
      },
      emission_count: crate::scene::particles::GaussianParams {
        mean: 50.0,
        std_dev: 0.0,
        min: 0.0,
        max: 100.0,
      },
      particle_radius: 0.5,
      density: 1.0,
      lifetime: 10000000,
      color: [0.0, 1.0, 0.0, 1.0], // Green
      beta: 0.0,
    },
  );

  scene
    .add_component(
      particle_sys_e,
      TransformComponent {
        position: Vec3f32::from_array([0.0, 0.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();

  scene.add_component(particle_sys_e, sys).unwrap();

  let scene_arc = std::sync::Arc::new(scene);
  let (tx, rx) = std::sync::mpsc::channel();
  let (done_tx, done_rx) = std::sync::mpsc::channel();

  let scene_physics = scene_arc.clone();
  let mesh_arc_physics = mesh_arc.clone();

  let physics_thread = std::thread::spawn(move || {
    let mut time = 0;
    let dt = 0.01; // 10ms
    let uv_grid = crate::simulation::comet::uv_grid::UvGrid::new(
      &mesh_arc_physics.vertices,
      &mesh_arc_physics.indices,
      32,
    );

    while time <= 100 {
      scene_physics.query1_mut::<crate::scene::particles::ParticleSystemComponent, _>(|_, sys| {
        sys.accumulator += (dt * 1_000_000.0) as i64;
        while sys.accumulator >= sys.config.delta {
          sys.accumulator -= sys.config.delta;
          let u_emission = [0.5, 0.5];
          let mut u_particles = Vec::new();
          for _ in 0..100 {
            u_particles.push([0.5, 0.5, 0.5, 0.5]);
          }
          sys.emit_particles(
            &mesh_arc_physics,
            &uv_grid,
            Vec3f32::from_array([0.0, 0.0, 0.0]),
            Quat::identity(),
            Vec3f32::from_components(1.0, 1.0, 1.0),
            &u_emission,
            &u_particles,
          );
        }

        for p in sys.particles.write().iter_mut().filter(|p| p.active != 0) {
          p.position[0] += p.velocity[0] * dt;
          p.position[1] += p.velocity[1] * dt;
          p.position[2] += p.velocity[2] * dt;
        }
      });

      tx.send(time).unwrap();
      std::thread::sleep(std::time::Duration::from_millis(50));
      time += 10;
    }
    tx.send(-1).unwrap();
  });

  let render_frontend_clone = render_frontend.clone();
  let scene_render = scene_arc.clone();
  let done_tx_clone = done_tx.clone();

  let render_thread = std::thread::spawn(move || {
    while let Ok(time) = rx.recv() {
      if time == -1 {
        break;
      }

      let task_id = render_frontend_clone
        .with_device(render_device_handle, |device| {
          let task_id = device.create_task();
          device.start_frame()?;
          let acquire_result = device.acquire_next_image(presentation_engine)?;
          println!("Thread: acquired. getting command buffer...");
          let cmd_buffer_handle = device.get_command_buffer()?;
          let _scoped_cmd =
            gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

          let mut render_scene = RenderScene::new(
            (
              TransformComponent {
                position: Vec3f32::from_array([0.0, -10.0, 0.0]),
                rotation: Quat::from_axis_angle(
                  Vec3f32::from_array([0.0, 0.0, 1.0]),
                  std::f32::consts::PI,
                ),
                scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
              },
              CameraComponent {
                projection: Mat4x4f32::perspective_vk(45.0f32.to_radians(), 1.0, 0.1, 100.0),
                near_plane: 0.1,
                far_plane: 100.0,
              },
            ),
            aethervk_oshal_rlib::os::time::TimeReadings::default(),
            [width, height],
          );
          let sun_pipeline = device.get_sun_pipeline_key(presentation_engine)?;
          render_scene.sun_call = Some(gpu::frame::SunDrawCall::from_model_and_camera(
            Mat4x4f32::identity(),
            &render_scene.camera_data,
            sun_pipeline,
            sun_e,
            0.6,
          )?);

          let res = match device.get_physical_mesh_resources(mesh_entity, presentation_engine) {
            Ok(r) => r,
            Err(_) => device.create_physical_mesh_resources(
              cmd_buffer_handle,
              mesh_entity,
              &crate::scene::PhysicalMeshComponent {
                asset_path: "".to_string(),
                mesh: mesh_arc.clone(),
                emissive_intensity: 0.0,
                emissive_color: [0.0; 3],
              },
              presentation_engine,
              "",
            )?,
          };

          let outline: Option<[f32; 4]> = None;
          render_scene.draw_calls.push(gpu::frame::DrawCall::from_handles_and_matrix(
            res,
            mesh_arc.indices.len() as u32,
            outline,
            Mat4x4f32::identity(),
            1.0,
            [0.5, 0.5, 0.5],
          ));

          scene_render.with_component(
            particle_sys_e,
            |sys: &crate::scene::particles::ParticleSystemComponent| {
              render_scene
                .add_renderable(
                  cmd_buffer_handle,
                  device,
                  particle_sys_e,
                  Mat4x4f32::identity(),
                  crate::scene::RenderableDataRef::ParticleSystem(sys),
                  presentation_engine,
                  "particle_sys_test",
                  false,
                  [0.0, 0.0, 0.0, 0.0],
                )
                .unwrap();
            },
          );

          {
            device.update_sun(cmd_buffer_handle, sun_e, (64, 64, 64), 0.6)?;
            device.upload_particle_systems(cmd_buffer_handle, &mut render_scene.particle_calls)?;
            device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
            let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

            let extent = device.get_presentation_engine_extent(presentation_engine)?;
            device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent))?;
            device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent))?;
            gpu::frame::render_frame(
              device,
              cmd_buffer_handle,
              &render_scene,
              presentation_engine,
            )?;
            scoped_rp.end()?;
            device.record_windowless_download(cmd_buffer_handle, presentation_engine, task_id)?;
          }
          _scoped_cmd.submit()?;
          println!("TR: submitted task {}", task_id);

          device.present(
            presentation_engine,
            acquire_result.image_index as usize,
            acquire_result.frame_index as usize,
          )?;
          println!("TR: presented task {}", task_id);
          crate::types::GpuResult::Ok(task_id)
        })
        .unwrap();

      done_tx_clone.send((time, task_id)).unwrap();
    }
  });

  let render_frontend_save = render_frontend.clone();

  let save_thread = std::thread::spawn(move || {
    while let Ok((time, task_id)) = done_rx.recv() {
      render_frontend_save
        .with_device(render_device_handle, |device| {
          while !device.is_task_completed(task_id)? {
            std::thread::yield_now();
          }
          let mut buffer = vec![0u8; (width * height * 4) as usize];
          device.read_windowless_download(task_id, &mut buffer)?;

          let mut found_gray = false;
          let mut found_green = false;

          let mut max_r = 0;
          let mut max_g = 0;
          let mut max_b = 0;

          for chunk in buffer.chunks_exact(4) {
            let b = chunk[0];
            let g = chunk[1];
            let r = chunk[2];

            if r > max_r {
              max_r = r;
            }
            if g > max_g {
              max_g = g;
            }
            if b > max_b {
              max_b = b;
            }

            // Look for bright green
            if g > 200 && r < 50 && b < 50 {
              found_green = true;
            }

            // Look for gray plane (with sun lighting, typically > 0 and balanced)
            if r > 20 && r < 200 && g > 20 && g < 200 && b > 20 && b < 200 {
              if (r as i32 - g as i32).abs() < 25 && (g as i32 - b as i32).abs() < 25 {
                found_gray = true;
              }
            }
          }

          println!(
            "Time {}ms: Max RGB: ({}, {}, {}). found_gray={}, found_green={}",
            time, max_r, max_g, max_b, found_gray, found_green
          );

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
          assert!(found_gray, "Gray plane missing at {}ms", time);
          assert!(found_green, "Green particles missing at {}ms", time);

          crate::types::GpuResult::Ok(())
        })
        .unwrap();
    }
  });

  physics_thread.join().unwrap();
  render_thread.join().unwrap();
  drop(done_tx); // close channel to terminate save_thread
  save_thread.join().unwrap();

  drop(render_frontend);

  let errors = super::utils::VULKAN_ERROR_MESSAGES.lock().unwrap();
  if !errors.is_empty() {
    panic!("Vulkan validation errors occurred during multithreaded testing");
  }
}

struct DepthTestSetupScene {
  render_frontend: RenderFrontend,
  render_device_handle: RenderDeviceHandle,
  presentation_engine: PresentationEngineHandle,
  transform: TransformComponent,
  entity_id: EntityId,
  mesh: PhysicalMeshComponent,
  scene: Scene,
  thread_pool: Arc<aethervk_oshal_rlib::os::pool::ThreadPool>,
}

fn depth_test_setup_scene() -> DepthTestSetupScene {
  setup_assets_dir();
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend.write().init_device(0, &additional_params).unwrap();

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

  render_frontend
    .with_device(render_device_handle, |device| {
      device.wire_callbacks(pool_arc.clone())
    })
    .unwrap();

  // TODO: remove EntityId dependency on physical mesh resource creation
  let scene = Scene::new();
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<PhysicalMeshComponent>(&[TypeId::of::<TransformComponent>()]);
  let entity_id = scene.spawn_entity("mesh");
  let mesh = Arc::new(crate::simulation::comet::generate_uv_sphere(
    2.0, 10, 10, 1.0,
  ));
  println!(
    "Mesh vertices: {}, indices: {}",
    mesh.vertices.len(),
    mesh.indices.len()
  );
  let mesh_comp = PhysicalMeshComponent {
    mesh,
    emissive_intensity: 1.0,
    emissive_color: [1.0, 1.0, 1.0],
    asset_path: "test".to_string(),
  };
  let transform = TransformComponent {
    position: Vec3f32::from_array([0.0, -5.0, 0.0]), // place it in front of camera (-y is forward)
    rotation: Quat::identity(),
    scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
  };
  scene.add_component(entity_id, transform.clone()).unwrap();
  scene.add_component(entity_id, mesh_comp.clone()).unwrap();

  DepthTestSetupScene {
    render_frontend,
    render_device_handle,
    presentation_engine,
    transform,
    entity_id,
    mesh: mesh_comp,
    scene,
    thread_pool: pool_arc,
  }
}

#[test]
// Note: comment when debugging.
// #[ntest::timeout(20_000)] // 20s timeout in case you don't run cargo nextest run (there's a busy loop here)
fn test_depth_stencil_separation() {
  let test_data = depth_test_setup_scene();

  // 1. Perspective
  test_data
    .render_frontend
    .with_device(test_data.render_device_handle, |device| {
      let [width, height] =
        device.get_presentation_engine_extent(test_data.presentation_engine).unwrap();

      let task_id = device.create_task();
      device.start_frame().unwrap();
      let acquire_result = device.acquire_next_image(test_data.presentation_engine).unwrap();
      let cmd_buffer_handle = device.get_command_buffer().unwrap();

      let mut render_scene = RenderScene::new(
        (
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
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
      );

      let model_matrix = test_data.transform.to_mat4();

      println!("Camera pos: {:?}", render_scene.camera_data.pos);
      println!("Camera view: {:?}", render_scene.camera_data.view);
      println!("Camera proj: {:?}", render_scene.camera_data.proj);
      println!("Model matrix: {:?}", model_matrix);
      println!(
        "MVP matrix: {:?}",
        render_scene.camera_data.view_proj * model_matrix
      );

      let cmd_buf_guard =
        ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id)).unwrap();

      render_scene
        .add_renderable(
          cmd_buffer_handle,
          device,
          test_data.entity_id,
          model_matrix,
          crate::scene::RenderableDataRef::PhysicalMesh(&test_data.mesh),
          test_data.presentation_engine,
          "debug_name",
          false,    // draw_outline
          [1.0; 4], // outline_color
        )
        .unwrap();

      device
        .begin_render_pass(
          cmd_buffer_handle,
          test_data.presentation_engine,
          &acquire_result,
        )
        .unwrap();
      let render_pass_guard = ScopedRenderPass::new(device, cmd_buffer_handle);

      device.set_viewport(
        cmd_buffer_handle,
        &gpu::Viewport::from_extent([width, height]),
      )?;
      device.set_scissor(
        cmd_buffer_handle,
        &gpu::Rect2D::from_extent([width, height]),
      )?;
      crate::gpu::frame::render_frame(
        device,
        cmd_buffer_handle,
        &render_scene,
        test_data.presentation_engine,
      )
      .unwrap();

      render_pass_guard.end().unwrap();

      let actual_device = device.as_any().downcast_ref::<Device>().unwrap();
      actual_device
        .record_test_depth_stencil_download(
          cmd_buffer_handle,
          test_data.presentation_engine,
          task_id,
        )
        .unwrap();

      cmd_buf_guard.submit().unwrap();

      while !device.is_task_completed(task_id).unwrap() {
        core::hint::spin_loop();
      }

      let depth_size = (width * height * 4) as usize;
      let stencil_size = (width * height * 1) as usize;
      let mut buffer = vec![0u8; depth_size + stencil_size];
      println!("Format: {:?}", actual_device.depth_stencil_format);
      actual_device.read_windowless_download(task_id, &mut buffer).unwrap();
      println!("First 32 bytes of buffer: {:?}", &buffer[0..32]);

      let (depth, stencil) = actual_device.separate_depth_stencil(&buffer, width, height);

      let mut min_depth = 1.0f32;
      let mut max_depth = 0.0f32;
      for &d in &depth {
        if d < min_depth {
          min_depth = d;
        }
        if d > max_depth {
          max_depth = d;
        }
      }
      println!("Min depth: {}, Max depth: {}", min_depth, max_depth);

      // save before assertions
      let depth_u8: Vec<u8> = depth.iter().map(|&d| (d * 255.0).clamp(0.0, 255.0) as u8).collect();
      image::save_buffer(
        "test_output_depth_perspective.png",
        &depth_u8,
        width,
        height,
        image::ColorType::L8,
      )
      .unwrap();
      image::save_buffer(
        "test_output_stencil_perspective.png",
        &stencil,
        width,
        height,
        image::ColorType::L8,
      )
      .unwrap();

      let mut found_mesh = false;
      println!(
        "Center depth: {}, Center stencil: {}",
        depth[128 * 256 + 128],
        stencil[128 * 256 + 128]
      );
      println!(
        "Middle 5 depths: {:?}",
        &depth[(128 * 256 + 126)..(128 * 256 + 131)]
      );
      for i in 0..(width * height) as usize {
        if depth[i] < 1.0 {
          found_mesh = true;
          // check stencil
          assert_eq!(
            stencil[i], 255,
            "Stencil should be 255 where mesh is drawn. depth was {}",
            depth[i]
          );
        } else {
          // background
          assert_eq!(stencil[i], 0, "Stencil should be 0 where mesh is NOT drawn");
        }
      }
      assert!(
        found_mesh,
        "Mesh should be visible in perspective projection"
      );

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  // 2. Orthographic
  test_data
    .render_frontend
    .with_device(test_data.render_device_handle, |device| {
      let [width, height] =
        device.get_presentation_engine_extent(test_data.presentation_engine).unwrap();
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(test_data.presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;

      let mut render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([0.0, 0.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent {
            projection: Mat4x4f32::orthographic_vk(-5.0, 5.0, -5.0, 5.0, 0.1, 100.0),
            near_plane: 0.1,
            far_plane: 100.0,
          },
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
      );

      let model_matrix = test_data.transform.to_mat4();

      render_scene.add_renderable(
        cmd_buffer_handle,
        device,
        test_data.entity_id,
        model_matrix,
        crate::scene::RenderableDataRef::PhysicalMesh(&test_data.mesh),
        test_data.presentation_engine,
        "debug_name",
        false,    // draw_outline
        [0.0; 4], // outline_color
      )?;

      let cmd_scope = ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;
      device.begin_render_pass(
        cmd_buffer_handle,
        test_data.presentation_engine,
        &acquire_result,
      )?;
      let render_pass_guard = ScopedRenderPass::new(device, cmd_buffer_handle);

      device.set_viewport(
        cmd_buffer_handle,
        &gpu::Viewport::from_extent([width, height]),
      )?;
      device.set_scissor(
        cmd_buffer_handle,
        &gpu::Rect2D::from_extent([width, height]),
      )?;

      crate::gpu::frame::render_frame(
        device,
        cmd_buffer_handle,
        &render_scene,
        test_data.presentation_engine,
      )?;

      render_pass_guard.end()?;

      let actual_device =
        device.as_any().downcast_ref::<crate::gpu_backends::vulkan::device::Device>().unwrap();
      actual_device.record_test_depth_stencil_download(
        cmd_buffer_handle,
        test_data.presentation_engine,
        task_id,
      )?;

      cmd_scope.submit()?;
      while !device.is_task_completed(task_id)? {
        core::hint::spin_loop();
      }

      let depth_size = (width * height * 4) as usize;
      let stencil_size = (width * height * 1) as usize;
      let mut buffer = vec![0u8; depth_size + stencil_size];
      actual_device.read_windowless_download(task_id, &mut buffer)?;

      let (depth, stencil) = actual_device.separate_depth_stencil(&buffer, width, height);

      let mut found_mesh = false;
      println!(
        "Center depth: {}, Center stencil: {}",
        depth[128 * 256 + 128],
        stencil[128 * 256 + 128]
      );
      println!(
        "Middle 5 depths: {:?}",
        &depth[(128 * 256 + 126)..(128 * 256 + 131)]
      );
      for i in 0..(width * height) as usize {
        if depth[i] < 1.0 {
          found_mesh = true;
          // check stencil
          assert_eq!(stencil[i], 255, "Stencil should be 255 where mesh is drawn");
        }
      }
      assert!(
        found_mesh,
        "Mesh should be visible in orthographic projection"
      );

      let depth_u8: Vec<u8> = depth.iter().map(|&d| (d * 255.0).clamp(0.0, 255.0) as u8).collect();
      image::save_buffer(
        "test_output_depth_orthographic.png",
        &depth_u8,
        width,
        height,
        image::ColorType::L8,
      )
      .unwrap();
      image::save_buffer(
        "test_output_stencil_orthographic.png",
        &stencil,
        width,
        height,
        image::ColorType::L8,
      )
      .unwrap();

      crate::types::GpuResult::Ok(())
    })
    .unwrap();
}

#[test]
// Note: comment when debugging.
// #[ntest::timeout(20_000)] // 20s timeout in case you don't run cargo nextest run (there's a busy loop here)
fn test_depth_color_image() {
  let test_data = depth_test_setup_scene();

  // 1. Perspective
  test_data
    .render_frontend
    .with_device(test_data.render_device_handle, |device| {
      let [width, height] =
        device.get_presentation_engine_extent(test_data.presentation_engine).unwrap();

      let task_id = device.create_task();
      device.start_frame().unwrap();
      let acquire_result = device.acquire_next_image(test_data.presentation_engine).unwrap();
      let cmd_buffer_handle = device.get_command_buffer().unwrap();

      let mut render_scene = RenderScene::new(
        (
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
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
      );

      let model_matrix = test_data.transform.to_mat4();

      println!("Camera pos: {:?}", render_scene.camera_data.pos);
      println!("Camera view: {:?}", render_scene.camera_data.view);
      println!("Camera proj: {:?}", render_scene.camera_data.proj);
      println!("Model matrix: {:?}", model_matrix);
      println!(
        "MVP matrix: {:?}",
        render_scene.camera_data.view_proj * model_matrix
      );

      let cmd_buf_guard =
        ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id)).unwrap();

      render_scene
        .add_renderable(
          cmd_buffer_handle,
          device,
          test_data.entity_id,
          model_matrix,
          crate::scene::RenderableDataRef::PhysicalMesh(&test_data.mesh),
          test_data.presentation_engine,
          "debug_name",
          false,    // draw_outline
          [1.0; 4], // outline_color
        )
        .unwrap();

      device
        .begin_render_pass(
          cmd_buffer_handle,
          test_data.presentation_engine,
          &acquire_result,
        )
        .unwrap();
      let render_pass_guard = ScopedRenderPass::new(device, cmd_buffer_handle);

      device.set_viewport(
        cmd_buffer_handle,
        &gpu::Viewport::from_extent([width, height]),
      )?;
      device.set_scissor(
        cmd_buffer_handle,
        &gpu::Rect2D::from_extent([width, height]),
      )?;
      crate::gpu::frame::render_frame(
        device,
        cmd_buffer_handle,
        &render_scene,
        test_data.presentation_engine,
      )
      .unwrap();

      render_pass_guard.end().unwrap();

      device
        .record_windowless_download(cmd_buffer_handle, test_data.presentation_engine, task_id)
        .unwrap();

      cmd_buf_guard.submit().unwrap();

      while !device.is_task_completed(task_id).unwrap() {
        core::hint::spin_loop();
      }

      let size = (width * height * 4) as usize;
      let mut buffer = vec![0u8; size];
      device.read_windowless_download(task_id, &mut buffer).unwrap();

      // save before assertions
      image::save_buffer(
        "test_output_depth_color_image.png",
        &buffer,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .unwrap();

      let mut found_mesh = false;
      println!(
        "Center color: {:?}",
        [
          buffer[(128 * 256 + 128) * 4],
          buffer[(128 * 256 + 128) * 4 + 1],
          buffer[(128 * 256 + 128) * 4 + 2],
        ]
      );
      let mut max_val = 0;
      for i in 0..(width * height) as usize {
        let color = [
          buffer[i * 4] as f32 / 255.0,
          buffer[i * 4 + 1] as f32 / 255.0,
          buffer[i * 4 + 2] as f32 / 255.0,
        ];
        if buffer[i * 4] > max_val {
          max_val = buffer[i * 4];
        }
        if buffer[i * 4 + 1] > max_val {
          max_val = buffer[i * 4 + 1];
        }
        if buffer[i * 4 + 2] > max_val {
          max_val = buffer[i * 4 + 2];
        }
        if color[0] != 0.0 || color[1] != 0.0 || color[2] != 0.0 {
          found_mesh = true;
        }
      }
      println!("Max pixel value in RGB: {}", max_val);
      assert!(
        found_mesh,
        "Mesh should be visible in perspective projection"
      );

      crate::types::GpuResult::Ok(())
    })
    .unwrap();
}

#[test]
fn test_sun_rendering() {
  setup_assets_dir();
  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend.write().init_device(0, &additional_params).unwrap();

  let width = 256;
  let height = 256;

  render_frontend
    .with_device(render_device_handle, |device| {
      device.wire_callbacks(pool_arc.clone())
    })
    .unwrap();

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
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<PhysicalMeshComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<SunComponent>(&[TypeId::of::<TransformComponent>()]);
  let sun_e = scene.spawn_entity("sun");

  let asset_path = format!(
    "{}/Comet.glb",
    crate::gpu::ASSET_DIR.read().as_ref().unwrap()
  );
  let mesh = Arc::new(crate::simulation::comet::load_comet_from_gltf(&asset_path, false).unwrap());

  // Radius of sun volume is hardcoded to 0.6 in update_sun. So we scale the mesh to 0.5.
  let transform = TransformComponent {
    position: Vec3f32::from_array([0.0, -5.0, 0.0]), // Place in front of camera (-y is forward)
    rotation: Quat::identity(),
    scale: Vec3f32::from_array([0.5, 0.5, 0.5]),
  };

  let mesh_comp = PhysicalMeshComponent {
    mesh: mesh.clone(),
    emissive_intensity: 5.0,
    emissive_color: [1.0, 0.5, 0.1], // Orange-ish emissive core
    asset_path: asset_path.clone(),
  };

  scene.add_component(sun_e, transform.clone()).unwrap();
  scene.add_component(sun_e, mesh_comp.clone()).unwrap();
  scene
    .add_component(
      sun_e,
      SunComponent {
        resolution: (64, 64, 64),
        radius: 0.8,
      },
    )
    .unwrap();

  let mut frame1_sun_buffer = vec![];
  let mut frame2_sun_buffer = vec![];

  // We will run 2 frames to check animation
  for frame in 0..2 {
    render_frontend
      .with_device(render_device_handle, |device| {
        let render_task_id = device.create_task();

        device.start_frame()?;
        let acquire_result = device.acquire_next_image(presentation_engine)?;
        let cmd_buffer_handle = device.get_command_buffer()?;

        let mut render_scene = RenderScene::new(
          (
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
          ),
          aethervk_oshal_rlib::os::time::TimeReadings::default(),
          [width, height],
        );

        let sun_pipeline = device.get_sun_pipeline_key(presentation_engine)?;
        let model_matrix = transform.to_mat4();

        render_scene.sun_call = Some(gpu::frame::SunDrawCall::from_model_and_camera(
          model_matrix,
          &render_scene.camera_data,
          sun_pipeline,
          sun_e,
          0.8,
        )?);

        let cmd_buf_guard =
          gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(render_task_id))?;

        // 1. Update Sun (Generates 3D texture)
        device.update_sun(cmd_buffer_handle, sun_e, (64, 64, 64), 0.6)?;

        // 2. Download the Sun 3D texture
        let actual_device =
          device.as_any().downcast_ref::<crate::gpu_backends::vulkan::device::Device>().unwrap();
        // Use a unique task ID just for the map entry, but wait on the command buffer task ID
        let sun_task_id = device.create_task();
        actual_device.record_test_sun_download(cmd_buffer_handle, sun_e, sun_task_id)?;

        // 3. Render physical mesh
        render_scene.add_renderable(
          cmd_buffer_handle,
          device,
          sun_e,
          model_matrix,
          crate::scene::RenderableDataRef::PhysicalMesh(&mesh_comp),
          presentation_engine,
          "sun_mesh",
          false,
          [0.0; 4],
        )?;

        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

        let extent = device.get_presentation_engine_extent(presentation_engine)?;
        device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent)).unwrap();

        device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent)).unwrap();

        // 4. Render Scene (Draws Physical Mesh, then Sun Volume over it)
        gpu::frame::render_frame(
          device,
          cmd_buffer_handle,
          &render_scene,
          presentation_engine,
        )
        .unwrap();

        scoped_rp.end()?;

        // 5. Download rendered color image
        device.record_windowless_download(
          cmd_buffer_handle,
          presentation_engine,
          render_task_id,
        )?;

        cmd_buf_guard.submit()?;

        while !device.is_task_completed(render_task_id)? {
          core::hint::spin_loop();
        }

        // Mark sun_task_id as success manually since we only submitted render_task_id
        device.success_task(sun_task_id);

        // Verify Sun 3D Texture
        let sun_size = (64 * 64 * 64 * 8) as usize; // R16G16B16A16 = 8 bytes
        let mut sun_buffer = vec![0u8; sun_size];
        actual_device.read_windowless_download(sun_task_id, &mut sun_buffer)?;

        // Verify Rendered Scene Color
        let size = (width * height * 4) as usize;
        let mut buffer = vec![0u8; size];
        device.read_windowless_download(render_task_id, &mut buffer)?;

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
          &format!("test_sun_rendering_frame_{}.png", frame),
          &export_buffer,
          width,
          height,
          image::ColorType::Rgba8,
        )
        .expect("Failed to save rendered png");

        let mut max_val = 0;
        let mut found_scene = false;
        for i in 0..(width * height) as usize {
          if buffer[i * 4] > max_val {
            max_val = buffer[i * 4];
          }
          if buffer[i * 4 + 1] > max_val {
            max_val = buffer[i * 4 + 1];
          }
          if buffer[i * 4 + 2] > max_val {
            max_val = buffer[i * 4 + 2];
          }
          if buffer[i * 4] > 0 || buffer[i * 4 + 1] > 0 || buffer[i * 4 + 2] > 0 {
            found_scene = true;
          }
        }

        assert!(
          found_scene,
          "Sun rendering resulted in an empty image at frame {}!",
          frame
        );

        let mut sun_max = 0;
        for b in &sun_buffer {
          if *b > sun_max {
            sun_max = *b;
          }
        }
        assert!(
          sun_max > 0,
          "Sun 3D volume texture is completely black at frame {}!",
          frame
        );

        if frame == 0 {
          frame1_sun_buffer = sun_buffer;
        } else {
          frame2_sun_buffer = sun_buffer;
        }

        device.present(
          presentation_engine,
          acquire_result.image_index as usize,
          acquire_result.frame_index as usize,
        )?;

        crate::types::GpuResult::Ok(())
      })
      .unwrap();
  }

  assert_ne!(
    frame1_sun_buffer, frame2_sun_buffer,
    "Sun volume did not animate between frames!"
  );

  drop(render_frontend);
}

#[test]
fn test_sun_rendering_volume_only() {
  setup_assets_dir();
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });
  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend.write().init_device(0, &additional_params).unwrap();

  let width = 256;
  let height = 256;

  render_frontend
    .with_device(render_device_handle, |device| {
      device.wire_callbacks(pool_arc.clone())
    })
    .unwrap();

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
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<SunComponent>(&[TypeId::of::<TransformComponent>()]);
  let sun_e = scene.spawn_entity("sun");

  let transform = TransformComponent {
    position: Vec3f32::from_array([0.0, -5.0, 0.0]), // Place in front of camera (-y is forward)
    rotation: Quat::identity(),
    scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
  };

  scene.add_component(sun_e, transform.clone()).unwrap();
  scene
    .add_component(
      sun_e,
      SunComponent {
        resolution: (64, 64, 64),
        radius: 0.8,
      },
    )
    .unwrap();

  render_frontend
    .with_device(render_device_handle, |device| {
      let render_task_id = device.create_task();

      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;

      let mut render_scene = RenderScene::new(
        (
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
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
      );

      let sun_pipeline = device.get_sun_pipeline_key(presentation_engine)?;
      let model_matrix = transform.to_mat4();

      render_scene.sun_call = Some(gpu::frame::SunDrawCall::from_model_and_camera(
        model_matrix,
        &render_scene.camera_data,
        sun_pipeline,
        sun_e,
        0.8,
      )?);

      let cmd_buf_guard =
        gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(render_task_id))?;

      // 1. Update Sun (Generates 3D texture)
      device.update_sun(cmd_buffer_handle, sun_e, (64, 64, 64), 0.8)?;

      // 2. Download the Sun 3D texture
      let actual_device =
        device.as_any().downcast_ref::<crate::gpu_backends::vulkan::device::Device>().unwrap();
      let sun_task_id = device.create_task();
      actual_device.record_test_sun_download(cmd_buffer_handle, sun_e, sun_task_id)?;

      // 3. Render Scene
      device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
      let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

      let extent = device.get_presentation_engine_extent(presentation_engine)?;
      device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent))?;

      device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent))?;

      gpu::frame::render_frame(
        device,
        cmd_buffer_handle,
        &render_scene,
        presentation_engine,
      )?;

      scoped_rp.end()?;

      // 4. Download rendered color image
      device.record_windowless_download(cmd_buffer_handle, presentation_engine, render_task_id)?;

      cmd_buf_guard.submit()?;

      while !device.is_task_completed(render_task_id)? {
        core::hint::spin_loop();
      }

      device.success_task(sun_task_id);

      // Verify Sun 3D Texture
      let sun_size = (64 * 64 * 64 * 8) as usize; // R16G16B16A16 = 8 bytes
      let mut sun_buffer = vec![0u8; sun_size];
      actual_device.read_windowless_download(sun_task_id, &mut sun_buffer)?;

      // Verify Rendered Scene Color
      let size = (width * height * 4) as usize;
      let mut buffer = vec![0u8; size];
      device.read_windowless_download(render_task_id, &mut buffer)?;

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
        "test_sun_rendering_volume_only.png",
        &export_buffer,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .expect("Failed to save rendered png");

      let mut found_volume_pixels = false;
      for i in 0..(width * height) as usize {
        if buffer[i * 4] > 0 || buffer[i * 4 + 1] > 0 || buffer[i * 4 + 2] > 0 {
          found_volume_pixels = true;
          break;
        }
      }
      assert!(
        found_volume_pixels,
        "Sun volume rendering alone resulted in an empty image!"
      );

      let mut found_semitransparent = false;
      // Evaluate 3D texture alpha (16-bit float)
      for i in 0..(64 * 64 * 64) {
        let alpha_bytes = [sun_buffer[i * 8 + 6], sun_buffer[i * 8 + 7]];
        let alpha_u16 = u16::from_le_bytes(alpha_bytes);
        // alpha_u16 == 0x3C00 is 1.0 in f16. alpha_u16 == 0 is 0.0.
        if alpha_u16 > 0 && alpha_u16 < 0x3C00 {
          found_semitransparent = true;
          break;
        }
      }
      assert!(
        found_semitransparent,
        "Sun volume 3D texture does not have semitransparent voxels!"
      );

      device.present(
        presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      )?;

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}

#[test]
fn test_render_particles_stress() {
  setup_assets_dir();
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });
  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend.write().init_device(0, &additional_params).unwrap();

  render_frontend
    .with_device(render_device_handle, |device| {
      device.wire_callbacks(pool_arc.clone())
    })
    .unwrap();

  let width = 512;
  let height = 512;

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

      let mut render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([0.0, 0.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent {
            projection: Mat4x4f32::perspective_vk(
              45.0f32.to_radians(),
              width as f32 / height as f32,
              0.1,
              1000.0,
            ),
            near_plane: 0.1,
            far_plane: 1000.0,
          },
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
      );

      let particle_sys_e = scene.spawn_entity("stress_particles");
      let mut sys = crate::scene::particles::ParticleSystemComponent::new(
        crate::scene::particles::ParticleEmitterConfig {
          uv_distribution: crate::math::distribution::Distribution2D::new(
            &[1.0, 1.0, 1.0, 1.0],
            2,
            2,
          ),
          delta: 1000,
          max_particles: 1_000_000,
          velocity_intensity: crate::scene::particles::GaussianParams {
            mean: 1.0,
            std_dev: 0.0,
            min: 0.0,
            max: 1.0,
          },
          emission_count: crate::scene::particles::GaussianParams {
            mean: 1.0,
            std_dev: 0.0,
            min: 0.0,
            max: 1.0,
          },
          particle_radius: 0.5,
          density: 1.0,
          lifetime: 1000000,
          color: [1.0, 0.8, 0.2, 1.0],
          beta: 0.0,
        },
      );

      let mut seed = 12345u32;
      let mut next_rand = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed as f32) / (u32::MAX as f32)
      };

      let mut particles = sys.particles.write();
      for i in 0..1_000_000 {
        // Box from -200 to +200 on x,z from -200 to 0 in y
        let x = (next_rand() - 0.5) * 400.0;
        let y = (next_rand() - 1.0) * 200.0;
        let z = (next_rand() - 0.5) * 400.0;

        let mut p = crate::scene::particles::ParticleData {
          id_low: 0,
          id_high: 0,
          age_low: 0,
          age_high: 0,
          position: [x, y, z],
          mass: 1.0,
          velocity: [0.0, 0.0, 0.0],
          active: 1,
        };
        p.set_id(i as u64);
        p.set_age(0);
        particles.push(p);
      }
      drop(particles);

      render_scene.add_renderable(
        cmd_buffer_handle,
        device,
        particle_sys_e,
        Mat4x4f32::identity(),
        crate::scene::RenderableDataRef::ParticleSystem(&sys),
        presentation_engine,
        "particle_sys_stress",
        false,
        [0.0, 0.0, 0.0, 0.0],
      )?;

      let scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;
      device.upload_particle_systems(cmd_buffer_handle, &mut render_scene.particle_calls)?;
      device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
      let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
      let extent = device.get_presentation_engine_extent(presentation_engine)?;

      device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent))?;
      device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent))?;

      gpu::frame::render_frame(
        device,
        cmd_buffer_handle,
        &render_scene,
        presentation_engine,
      )?;
      scoped_rp.end()?;
      device.record_windowless_download(cmd_buffer_handle, presentation_engine, task_id)?;

      scoped_cmd.submit().unwrap();

      device.present(
        presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      )?;

      while !device.is_task_completed(task_id)? {
        core::hint::spin_loop();
      }

      let size = (width * height * 4) as usize;
      let mut buffer = vec![0u8; size];
      device.read_windowless_download(task_id, &mut buffer)?;

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
        "test_render_particles_stress.png",
        &export_buffer,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .expect("Failed to save rendered png");

      let mut found_something = false;
      for i in 0..(width * height) as usize {
        if export_buffer[i * 4] != 0
          || export_buffer[i * 4 + 1] != 0
          || export_buffer[i * 4 + 2] != 0
        {
          found_something = true;
        }
      }
      assert!(found_something, "Particles should be visible in image");

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}

#[test]
fn test_outline_rendering_windowless() {
  setup_assets_dir();
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });
  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend.write().init_device(0, &additional_params).unwrap();

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
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<PhysicalMeshComponent>(&[TypeId::of::<TransformComponent>()]);
  let mesh_e = scene.spawn_entity("mesh");

  let transform = TransformComponent {
    position: Vec3f32::from_array([5.0, -5.0, 0.0]), // looking at -Y is forward
    rotation: Quat::identity(),
    scale: Vec3f32::from_array([1.0, 1.2, 1.0]),
  };

  let mesh_comp = PhysicalMeshComponent {
    asset_path: "cube".to_string(), // use the built-in cube
    mesh: Arc::new(crate::simulation::comet::generate_uv_sphere(
      0.5, 16, 16, 1.0,
    )),
    emissive_intensity: 1.0,
    emissive_color: [1.0; 3],
  };

  scene.add_component(mesh_e, transform.clone()).unwrap();
  scene.add_component(mesh_e, mesh_comp.clone()).unwrap();

  render_frontend
    .with_device(render_device_handle, |device| {
      // Must acquire resources to render physical meshes

      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let task_id = device.create_task();
      let cmd_buffer_handle = device.get_command_buffer()?;
      let scoped_cmd =
        gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id)).unwrap();

      let mut render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([5.0, 0.0, 0.0]),
            rotation: Quat::identity(), // identity looks towards -Y (forward)
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent {
            projection: Mat4x4f32::perspective_vk(
              45.0f32.to_radians(),
              width as f32 / height as f32,
              0.1,
              100.0,
            ),
            near_plane: 0.1,
            far_plane: 100.0,
          },
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
      );

      let model_matrix = transform.to_mat4();

      render_scene.add_renderable(
        cmd_buffer_handle,
        device,
        mesh_e,
        model_matrix,
        crate::scene::RenderableDataRef::PhysicalMesh(&mesh_comp),
        presentation_engine,
        "mesh_with_outline",
        true,                 // draw_outline (YES)
        [1.0, 0.0, 0.0, 1.0], // outline_color (RED)
      )?;

      device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
      let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
      let extent = device.get_presentation_engine_extent(presentation_engine)?;

      device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent)).unwrap();
      device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent)).unwrap();

      gpu::frame::render_frame(
        device,
        cmd_buffer_handle,
        &render_scene,
        presentation_engine,
      )
      .unwrap();
      scoped_rp.end().unwrap();
      device.record_windowless_download(cmd_buffer_handle, presentation_engine, task_id).unwrap();

      scoped_cmd.submit().unwrap();

      device.present(
        presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      )?;

      while !device.is_task_completed(task_id)? {
        core::hint::spin_loop();
      }

      let size = (width * height * 4) as usize;
      let mut buffer = vec![0u8; size];
      device.read_windowless_download(task_id, &mut buffer)?;

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
        "test_outline_rendering_windowless.png",
        &export_buffer,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .expect("Failed to save rendered png");

      // Verify that there is at least one red pixel which indicates an outline
      let mut found_red = false;
      for i in 0..(width * height) as usize {
        let r = buffer[i * 4 + 2]; // B G R A
        let g = buffer[i * 4 + 1];
        let b = buffer[i * 4 + 0];
        if r > 200 && g < 50 && b < 50 {
          found_red = true;
          break;
        }
      }

      assert!(
        found_red,
        "No outline found! The image does not contain red pixels."
      );

      Ok(())
    })
    .unwrap();
}

#[test]
fn test_outline_toggled_after_upload() {
  setup_assets_dir();
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });
  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
  let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

  let additional_params = DeviceAdditionalParams::new();
  let render_device_handle = render_frontend.write().init_device(0, &additional_params).unwrap();

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
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<PhysicalMeshComponent>(&[TypeId::of::<TransformComponent>()]);
  let mesh_e = scene.spawn_entity("mesh");

  let transform = TransformComponent {
    position: Vec3f32::from_array([5.0, -5.0, 0.0]), // placed at -Y is forward
    rotation: Quat::identity(),
    scale: Vec3f32::from_array([1.0, 1.2, 1.0]),
  };

  let mesh_comp = PhysicalMeshComponent {
    asset_path: "cube".to_string(),
    mesh: Arc::new(crate::simulation::comet::generate_uv_sphere(
      0.5, 16, 16, 1.0,
    )),
    emissive_intensity: 1.0,
    emissive_color: [1.0; 3],
  };

  scene.add_component(mesh_e, transform.clone()).unwrap();
  scene.add_component(mesh_e, mesh_comp.clone()).unwrap();

  render_frontend
    .with_device(render_device_handle, |device| {
      // FRAME 1: Upload with draw_outline = false
      {
        let task_id = device.create_task();
        device.start_frame()?;
        let acquire_result = device.acquire_next_image(presentation_engine)?;
        let cmd_buffer_handle = device.get_command_buffer()?;
        let scoped_cmd =
          gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id)).unwrap();

        let mut render_scene = RenderScene::new(
          (
            TransformComponent {
              position: Vec3f32::from_array([5.0, 0.0, 0.0]),
              rotation: Quat::identity(),
              scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
            },
            CameraComponent {
              projection: Mat4x4f32::perspective_vk(
                45.0f32.to_radians(),
                width as f32 / height as f32,
                0.1,
                100.0,
              ),
              near_plane: 0.1,
              far_plane: 100.0,
            },
          ),
          aethervk_oshal_rlib::os::time::TimeReadings::default(),
          [width, height],
        );

        let model_matrix = transform.to_mat4();

        render_scene.add_renderable(
          cmd_buffer_handle,
          device,
          mesh_e,
          model_matrix,
          crate::scene::RenderableDataRef::PhysicalMesh(&mesh_comp),
          presentation_engine,
          "mesh_no_outline",
          false,
          [0.0, 0.0, 0.0, 0.0],
        )?;

        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
        let extent = device.get_presentation_engine_extent(presentation_engine)?;

        device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent)).unwrap();
        device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent)).unwrap();

        gpu::frame::render_frame(
          device,
          cmd_buffer_handle,
          &render_scene,
          presentation_engine,
        )
        .unwrap();

        scoped_rp.end().unwrap();
        device.record_windowless_download(cmd_buffer_handle, presentation_engine, task_id).unwrap();

        scoped_cmd.submit().unwrap();

        device.present(
          presentation_engine,
          acquire_result.image_index as usize,
          acquire_result.frame_index as usize,
        )?;

        while !device.is_task_completed(task_id)? {
          core::hint::spin_loop();
        }
      }

      // Simulate resize / pipeline reload
      device.resize_presentation_engine(presentation_engine, width, height)?;

      // FRAME 2: Render with draw_outline = true
      {
        let task_id = device.create_task();
        device.start_frame()?;
        let acquire_result = device.acquire_next_image(presentation_engine)?;
        let cmd_buffer_handle = device.get_command_buffer()?;
        let scoped_cmd =
          gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id)).unwrap();

        let mut render_scene = RenderScene::new(
          (
            TransformComponent {
              position: Vec3f32::from_array([5.0, 0.0, 0.0]),
              rotation: Quat::identity(),
              scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
            },
            CameraComponent {
              projection: Mat4x4f32::perspective_vk(
                45.0f32.to_radians(),
                width as f32 / height as f32,
                0.1,
                100.0,
              ),
              near_plane: 0.1,
              far_plane: 100.0,
            },
          ),
          aethervk_oshal_rlib::os::time::TimeReadings::default(),
          [width, height],
        );

        let model_matrix = transform.to_mat4();

        // This triggers `get_physical_mesh_resources` instead of `create_physical_mesh_resources`
        render_scene.add_renderable(
          cmd_buffer_handle,
          device,
          mesh_e,
          model_matrix,
          crate::scene::RenderableDataRef::PhysicalMesh(&mesh_comp),
          presentation_engine,
          "mesh_with_outline",
          true,
          [1.0, 0.0, 0.0, 1.0], // RED
        )?;

        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
        let extent = device.get_presentation_engine_extent(presentation_engine)?;

        device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent)).unwrap();
        device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent)).unwrap();

        gpu::frame::render_frame(
          device,
          cmd_buffer_handle,
          &render_scene,
          presentation_engine,
        )
        .unwrap();

        scoped_rp.end().unwrap();
        device.record_windowless_download(cmd_buffer_handle, presentation_engine, task_id).unwrap();

        scoped_cmd.submit().unwrap();

        device.present(
          presentation_engine,
          acquire_result.image_index as usize,
          acquire_result.frame_index as usize,
        )?;

        while !device.is_task_completed(task_id)? {
          core::hint::spin_loop();
        }

        let size = (width * height * 4) as usize;
        let mut buffer = vec![0u8; size];
        device.read_windowless_download(task_id, &mut buffer)?;

        let mut found_red = false;
        for i in 0..(width * height) as usize {
          let r = buffer[i * 4 + 2];
          let g = buffer[i * 4 + 1];
          let b = buffer[i * 4 + 0];
          if r > 200 && g < 50 && b < 50 {
            found_red = true;
            break;
          }
        }

        assert!(found_red, "No outline found after toggling!");
      }

      Ok(())
    })
    .unwrap();
}

#[test]
fn test_render_concurrent_resize() {
  setup_assets_dir();
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });
  let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
  let pool_arc = std::sync::Arc::new(pool);
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

  // Create 3 presentation engines: two windowless, one windowed (headless)
  let (pe1, pe2, pe3) = render_frontend
    .with_device(render_device_handle, |device| {
      let params_wl1 = PresentationEngineParams::windowless(width, height);
      let pe1 = device.create_presentation_engine(&params_wl1)?;

      let params_wl2 = PresentationEngineParams::windowless(width, height);
      let pe2 = device.create_presentation_engine(&params_wl2)?;

      let params_win = PresentationEngineParams {
        width,
        height,
        vsync: false,
        window_info: gpu::OpaqueNativeHandleInfo {
          ptr0: core::ptr::null_mut(),
          ptr1: core::ptr::null_mut(),
        },
        ty: gpu::PresentationEngineType::Window,
      };
      let pe3 = device.create_presentation_engine(&params_win)?;

      device.init_archetypes(pe1)?;
      device.init_archetypes(pe2)?;
      device.init_archetypes(pe3)?;
      
      device.generate_sky()?;
      
      crate::types::GpuResult::Ok((pe1, pe2, pe3))
    })
    .unwrap();
    
  let engines = [pe1, pe2, pe3];

  let scene = Scene::new();
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<PhysicalMeshComponent>(&[TypeId::of::<TransformComponent>()]);
  let entity_id = scene.spawn_entity("mesh");
  let mesh = Arc::new(crate::simulation::comet::generate_uv_sphere(
    2.0, 10, 10, 1.0,
  ));
  let mesh_comp = PhysicalMeshComponent {
    mesh,
    emissive_intensity: 1.0,
    emissive_color: [1.0, 1.0, 1.0],
    asset_path: "test".to_string(),
  };
  let transform = TransformComponent {
    position: Vec3f32::from_array([0.0, -5.0, 0.0]),
    rotation: Quat::identity(),
    scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
  };
  scene.add_component(entity_id, transform.clone()).unwrap();
  scene.add_component(entity_id, mesh_comp.clone()).unwrap();

  let stop_signal = Arc::new(core::sync::atomic::AtomicBool::new(false));

  let render_frontend_render = render_frontend.clone();
  let stop_signal_render = stop_signal.clone();
  let render_thread = std::thread::spawn(move || {
    let mut pe_idx = 0;
    while !stop_signal_render.load(core::sync::atomic::Ordering::Relaxed) {
      let pe = engines[pe_idx];
      pe_idx = (pe_idx + 1) % engines.len();
      
      let _ = render_frontend_render
        .with_device(render_device_handle, |device| {
          let task_id = device.create_task();
          device.start_frame()?;
          
          let extent = device.get_presentation_engine_extent(pe)?;
          println!("Thread: acquire_next_image...");
          let acquire_result = match device.acquire_next_image(pe) {
            Ok(res) => res,
            Err(e) => return Err(e),
          };
          println!("Thread: acquired. getting command buffer...");
          let cmd_buffer_handle = device.get_command_buffer()?;

          let mut render_scene = RenderScene::new(
            (
              TransformComponent {
                position: Vec3f32::from_array([0.0, 0.0, 0.0]),
                rotation: Quat::identity(),
                scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
              },
              CameraComponent {
                projection: Mat4x4f32::perspective_vk(45.0f32.to_radians(), extent[0] as f32 / extent[1] as f32, 0.1, 100.0),
                near_plane: 0.1,
                far_plane: 100.0,
              },
            ),
            aethervk_oshal_rlib::os::time::TimeReadings::default(),
            extent,
          );

          let sky_pipeline = device.get_sky_pipeline_key(pe)?;
          render_scene.sky_call = Some(gpu::frame::SkyDrawCall::from_camera(
            &render_scene.camera_data,
            sky_pipeline,
          )?);

          let model_matrix = transform.to_mat4();

          let cmd_scope = ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

          println!("Thread: add_renderable...");
          render_scene.add_renderable(
            cmd_buffer_handle,
            device,
            entity_id,
            model_matrix,
            crate::scene::RenderableDataRef::PhysicalMesh(&mesh_comp),
            pe,
            "emissive_sphere",
            false,
            [0.0; 4],
          )?;

          {
            device.begin_render_pass(cmd_buffer_handle, pe, &acquire_result)?;
            let mut render_pass_guard = ScopedRenderPass::new(device, cmd_buffer_handle);

            device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent))?;
            device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent))?;
            
            crate::gpu::frame::render_frame(
              device,
              cmd_buffer_handle,
              &render_scene,
              pe,
            )?;
            render_pass_guard.end()?;
            if device.is_presentation_engine_windowless(pe)? {
               device.record_windowless_download(cmd_buffer_handle, pe, task_id)?;
            }
            println!("Thread: submitting...");
            cmd_scope.submit()?;
          }
          
          println!("Thread: presenting...");
          device.present(pe, acquire_result.image_index as usize, acquire_result.frame_index as usize)?;
          println!("Thread: reading download...");
          
          if device.is_presentation_engine_windowless(pe)? {
            let actual_device = device.as_any().downcast_ref::<Device>().unwrap();
            let size = (extent[0] * extent[1] * 4) as usize;
            let mut buffer = alloc::vec::Vec::with_capacity(size);
            unsafe { buffer.set_len(size) };
            
            let start_time = std::time::Instant::now();
            while !device.is_task_completed(task_id)? {
              if start_time.elapsed().as_secs() > 2 {
                println!("Task {} is stuck! Timeline cached: {}", task_id, actual_device.res.read().timeline_manager.get_cached_value());
                break;
              }
              core::hint::spin_loop();
            }
            actual_device.read_windowless_download(task_id, &mut buffer)?;
          }
          
          crate::types::GpuResult::Ok(())
        });
    }
  });

  let render_frontend_resize = render_frontend.clone();
  let stop_signal_resize = stop_signal.clone();
  let resize_thread = std::thread::spawn(move || {
    let mut counter = 0;
    let mut dims = [800, 600];
    while !stop_signal_resize.load(core::sync::atomic::Ordering::Relaxed) {
      let pe = engines[counter % engines.len()];
      let width = dims[0];
      let height = dims[1];
      
      dims.swap(0, 1);
      
      let res = render_frontend_resize
        .with_device(render_device_handle, |device| {
          device.resize_presentation_engine(pe, width, height)
        });
      if let Err(e) = res {
        println!("Resize thread error: {:?}", e);
      }
        
      counter += 1;
      std::thread::sleep(std::time::Duration::from_millis(3));
    }
  });

  std::thread::sleep(std::time::Duration::from_secs(5));
  println!("Main thread: setting stop_signal");
  stop_signal.store(true, core::sync::atomic::Ordering::Relaxed);

  println!("Main thread: waiting for render_thread");
  render_thread.join().unwrap();
  println!("Main thread: waiting for resize_thread");
  match resize_thread.join() {
      Ok(_) => println!("Resize thread joined successfully"),
      Err(e) => {
          if let Some(s) = e.downcast_ref::<&str>() {
              println!("Resize thread panicked with: {}", s);
          } else if let Some(s) = e.downcast_ref::<String>() {
              println!("Resize thread panicked with: {}", s);
          } else {
              println!("Resize thread panicked with unknown type");
          }
      }
  }
  println!("Main thread: all joined");

  drop(render_frontend);

  let errors = super::utils::VULKAN_ERROR_MESSAGES.lock().unwrap();
  if !errors.is_empty() {
    panic!("Vulkan validation errors occurred during testing");
  }
}
