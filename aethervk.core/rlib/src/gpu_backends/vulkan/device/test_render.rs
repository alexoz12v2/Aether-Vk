//! test_render module.

use super::*;
use crate::{
  gpu::{
    self, DeviceAdditionalParams, PresentationEngineParams, RenderDeviceHandle, RenderFrontend,
    ScopedCommandBuffer, ScopedRenderPass, VULKAN_RENDER_BACKEND,
    frame::{BillboardDrawCall, CursorDrawCall, RenderScene},
    new_render_frontend,
    scene_conversion::SceneConversionExt,
  },
  math::collision::{
    bounds::AABB,
    linear_bvh::{LinearBVH, LinearBVHHeader, LinearBVHNode, LinearBound},
  },
  scene,
  scene::{
    BillboardType, CameraComponent, PhysicalMeshComponent, Scene, SunComponent, TransformComponent,
  },
  types::RuntimeParams,
};
use aethervk_oshal_rlib::{
  math::{
    matrix::{Matrix4, SquareMatrix, mat4::Mat4x4f32},
    quaternion::Quaternion,
    vector::{Vector, Vector3, Vector4, vec3::Vec3f32, vec4::Quat},
  },
  os::pool::ThreadPool,
};
use heapless::index_map::FnvIndexMap;
use std::sync::Arc;
// TODO: test about text rendering in different fonts (system font and packaged font)
// TODO: move into integration tests folder

fn setup_assets_dir() {
  if let Ok(mut errors) = crate::gpu_backends::vulkan::utils::VULKAN_ERROR_MESSAGES.lock() {
    errors.clear();
  }

  let mut home_dir = std::env::current_exe().unwrap();
  let mut iter = 0;
  while !home_dir.join("assets").is_dir() && iter < 32 {
    home_dir.pop();
    iter += 1;
  }
  *crate::gpu::ASSET_DIR.write() = Some(home_dir.join("assets").to_str().unwrap().to_string());
}

fn setup_render_frontend_for_tests(
  with_windowless: bool,
) -> (
  Arc<ThreadPool>,
  RenderFrontend,
  RenderDeviceHandle,
  Option<PresentationEngineHandle>,
) {
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }

  // Create a channel to safely transfer the constructed data out of the thread
  use std::sync::mpsc;
  let (tx, rx) = mpsc::channel();

  let th =
    aethervk_oshal_rlib::os::thread::Builder::new().stack_size(8 * 1024 * 1024).spawn(move || {
      let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
      let pool_arc = Arc::new(pool);

      let runtime_params = Box::new(RuntimeParams {
        render_backend_params: FnvIndexMap::new(),
        validation_error_callback: Some(panic_on_validation_error as fn(&str)),
      });

      let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();

      let additional_params = DeviceAdditionalParams::new();
      let render_device_handle =
        render_frontend.write().init_device(0, &additional_params).unwrap();

      render_frontend
        .with_device(render_device_handle, |device| {
          device.wire_callbacks(Arc::clone(&pool_arc))
        })
        .unwrap();

      let presentation_engine = if with_windowless {
        let width = 256;
        let height = 256;

        let params = PresentationEngineParams::windowless(width, height);
        Some(
          render_frontend
            .with_device(render_device_handle, |device| {
              let pe = device.create_presentation_engine(&params)?;
              device.generate_sky()?;
              crate::types::GpuResult::Ok(pe)
            })
            .unwrap(),
        )
      } else {
        None
      };

      // Send the fully initialized tuple back to the main thread.
      // Ownership is safely transferred here.
      tx.send((
        pool_arc,
        render_frontend,
        render_device_handle,
        presentation_engine,
      ))
      .expect("Failed to send setup data from thread");
    });

  // Wait for the thread to complete its execution
  // (Depending on the custom thread lib, you might need a second .unwrap() here if join() returns a Result)
  let _ = th.unwrap().join();

  // Receive the data from the channel and return it
  rx.recv().expect("Failed to receive setup data")
}

fn test_render_particles_windowless_impl(use_particle2: bool) {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();
  let [width, height] = render_frontend
    .with_device(render_device_handle, |device| {
      device.get_presentation_engine_extent(presentation_engine)
    })
    .unwrap();

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

      let mut render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([0.0, 10.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 0.1, 100.0),
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
      );

      let particle_sys_e = scene.spawn_entity("particles");
      let config = crate::scene::particles::ParticleEmitterComponent {
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
        use_particle2,
      };
      let mut sys = crate::scene::particles::ParticleSystemComponent::new(config.max_particles);

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
        crate::scene::RenderableDataRef::ParticleSystem(&sys, &config),
        presentation_engine,
        "particle_sys_test",
        false,
        [0.0, 0.0, 0.0, 0.0],
      )?;

      let scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;
      device.upload_particle_systems(cmd_buffer_handle, &mut render_scene.particle_calls)?;
      device.upload_particle2_systems(cmd_buffer_handle, &mut render_scene.particle2_calls)?;
      device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
      let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
      let extent = device.get_presentation_engine_extent(presentation_engine)?;

      device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent))?;
      device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent))?;

      gpu::frame::render_frame(
        device,
        cmd_buffer_handle,
        presentation_engine,
        &render_scene,
      )?;
      scoped_rp.end()?;
      device.record_windowless_download(cmd_buffer_handle, task_id)?;

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
      let mut max_r = 0;
      let mut max_g = 0;
      let mut max_b = 0;
      for chunk in buffer.chunks_exact(4) {
        // BGRA format
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
        if r > 100 && g > 60 && b > 30 {
          found_color = true;
          break;
        }
      }
      println!(
        "Windowless particle test: Max RGB: ({}, {}, {})",
        max_r, max_g, max_b
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
      image::save_buffer(
        if use_particle2 {
          "test_rendered_particles2.png"
        } else {
          "test_rendered_particles.png"
        },
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
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();
  let [width, height] = render_frontend
    .with_device(render_device_handle, |device| {
      device.get_presentation_engine_extent(presentation_engine)
    })
    .unwrap();

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));
  let sky_e = scene.spawn_entity("sky");
  let sun_e = scene.spawn_entity("sun");
  let grid_e = scene.spawn_entity("grid");

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

      let cursor_res = device.get_cursor_resources(presentation_engine)?;
      let billboard_res = device.get_billboard_resources(presentation_engine)?;

      let mut render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([0.0, 10.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 0.1, 100.0),
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
      let gizmo_idx =
        device.update_gizmo_instance(sun_e, Mat4x4f32::identity(), presentation_engine)?;
      render_scene.gizmo_calls.push(gpu::frame::GizmoDrawCall::from_values(
        gizmo_resources.pipeline,
        2.0,
        gizmo_idx,
      ));

      let sphere_gizmo_idx = device.allocate_sphere_gizmo_instance(sun_e)?;
      let sphere_gizmo_pipeline = device.get_sphere_gizmo_pipeline_key(presentation_engine)?;
      let sphere_gizmo_data = vec![(
        sphere_gizmo_idx,
        crate::gpu::SphereGizmoDataGpu {
          model: Mat4x4f32::identity().into(),
          radius: 1.0,
          subdivisions: 12.0,
          _pad: [0.0, 0.0],
        },
      )];

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
        device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;
        device.update_sun(cmd_buffer_handle, sun_e, (64, 64, 64), 0.6)?;

        render_scene.sphere_gizmo_batch_call =
          device.upload_sphere_gizmos_batch(cmd_buffer_handle, &sphere_gizmo_data)?;

        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

        let extent = device.get_presentation_engine_extent(presentation_engine)?;
        device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent))?;

        device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent))?;

        gpu::frame::render_frame(
          device,
          cmd_buffer_handle,
          presentation_engine,
          &render_scene,
        )
        .map_err(|e| {
          println!("TR: render_frame failed {:?}", e);
          e
        })?;
        scoped_rp.end()?;
        device.record_windowless_download(cmd_buffer_handle, task_id)?;
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
fn test_render_sphere_gizmo_windowless() {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();
  let [width, height] = render_frontend
    .with_device(render_device_handle, |device| {
      device.get_presentation_engine_extent(presentation_engine)
    })
    .unwrap();

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));
  let gizmo_e = scene.spawn_entity("gizmo");

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

      let mut render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([0.0, 5.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 0.1, 100.0),
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
      );

      let sphere_gizmo_idx = device.allocate_sphere_gizmo_instance(gizmo_e)?;
      let sphere_gizmo_pipeline = device.get_sphere_gizmo_pipeline_key(presentation_engine)?;
      let mut model = Mat4x4f32::identity();
      model.w =
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, -5.0, 0.0, 1.0);
      let sphere_gizmo_data = vec![(
        sphere_gizmo_idx,
        crate::gpu::SphereGizmoDataGpu {
          model: model.into(),
          radius: 2.0,
          subdivisions: 12.0,
          _pad: [0.0, 0.0],
        },
      )];

      {
        let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;
        device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

        render_scene.sphere_gizmo_batch_call =
          device.upload_sphere_gizmos_batch(cmd_buffer_handle, &sphere_gizmo_data)?;

        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

        let extent = device.get_presentation_engine_extent(presentation_engine)?;
        device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent))?;
        device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent))?;

        gpu::frame::render_frame(
          device,
          cmd_buffer_handle,
          presentation_engine,
          &render_scene,
        )?;
        scoped_rp.end()?;
        device.record_windowless_download(cmd_buffer_handle, task_id)?;
      }

      device.present(
        presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      )?;

      // Wait for completion
      while !device.is_task_completed(task_id)? {
        std::thread::sleep(std::time::Duration::from_millis(10));
      }
      device.success_task(task_id);

      // Download image
      let mut buffer = vec![0u8; (width * height * 4) as usize];
      device.read_windowless_download(task_id, &mut buffer)?;

      let sum: u64 = buffer.iter().map(|&b| b as u64).sum();
      println!("Sum of buffer is {}", sum);

      // Save it to inspect manually before assertions
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
        "test_rendered_sphere_gizmo.png",
        &export_buffer,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .expect("Failed to save rendered png");

      assert!(
        sum > 0,
        "Buffer is completely empty, gizmo failed to render!"
      );

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);

  let errors = super::utils::VULKAN_ERROR_MESSAGES.lock().unwrap();
  if !errors.is_empty() {
    panic!("Vulkan validation errors occurred during testing");
  }
}

#[test]
fn test_sphere_gizmo_persistent_allocator() {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));
  let e1 = scene.spawn_entity("e1");
  let e2 = scene.spawn_entity("e2");
  let e3 = scene.spawn_entity("e3");

  render_frontend
    .with_device(render_device_handle, |device| {
      let idx1 = device.allocate_sphere_gizmo_instance(e1)?;
      let idx2 = device.allocate_sphere_gizmo_instance(e2)?;
      assert_eq!(idx1, 0);
      assert_eq!(idx2, 1);

      // Allocating again should return the same index
      let idx1_again = device.allocate_sphere_gizmo_instance(e1)?;
      assert_eq!(idx1_again, 0);

      // Free e1
      device.free_sphere_gizmo_instance(e1)?;

      // Allocate e3, should reuse e1's index (0)
      let idx3 = device.allocate_sphere_gizmo_instance(e3)?;
      assert_eq!(idx3, 0);

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}

#[test]
fn test_render_empty_scene_graceful() {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();
  let [width, height] = render_frontend
    .with_device(render_device_handle, |device| {
      device.get_presentation_engine_extent(presentation_engine)
    })
    .unwrap();

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

      let render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([0.0, 0.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 0.1, 100.0),
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [16, 16],
      );

      {
        let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;
        device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;
        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
        device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent([16, 16]))?;
        device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent([16, 16]))?;
        gpu::frame::render_frame(
          device,
          cmd_buffer_handle,
          presentation_engine,
          &render_scene,
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
  setup_assets_dir();
  fn panic_on_validation_error(msg: &str) {
    panic!("Vulkan validation error occurred during testing: {}", msg);
  }
  let runtime_params = Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: Some(panic_on_validation_error as fn(&str)),
  });
  let (pool_arc, render_frontend, render_device_handle) = {
    let rp_ptr = Box::into_raw(runtime_params) as usize;
    let result_arc = std::sync::Arc::new(std::sync::Mutex::new(None));
    let result_clone = result_arc.clone();

    let th = aethervk_oshal_rlib::os::thread::Builder::new()
      .stack_size(8 * 1024 * 1024)
      .spawn(move || {
        let rp = unsafe { Box::from_raw(rp_ptr as *mut RuntimeParams) };
        let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
        let pool_arc = std::sync::Arc::new(pool);
        let render_frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &rp).unwrap();
        let additional_params = DeviceAdditionalParams::new();
        let render_device_handle =
          render_frontend.write().init_device(0, &additional_params).unwrap();

        render_frontend
          .with_device(render_device_handle, |device| {
            device.wire_callbacks(pool_arc.clone())
          })
          .unwrap();

        *result_clone.lock().unwrap() = Some((pool_arc, render_frontend, render_device_handle));
      })
      .unwrap();
    th.join();
    result_arc.lock().unwrap().take().unwrap()
  };

  let presentation_engine = render_frontend
    .with_device(render_device_handle, |device| {
      let params = PresentationEngineParams::windowless(16, 16);
      let pe = device.create_presentation_engine(&params)?;
      crate::types::GpuResult::Ok(pe)
    })
    .unwrap();

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));
  let sun_e = scene.spawn_entity("sun");
  let _ = scene.add_component(
    sun_e,
    SunComponent {
      resolution: (64, 64, 64),
      radius: 0.6,
    },
  );

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

      {
        let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;
        device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;
        // This will fail because archetypes aren't initialized, but we catch/ignore it
        // just like the real `simulation_api.rs` does now.
        device.update_sun(cmd_buffer_handle, sun_e, (64, 64, 64), 0.6)?;

        // Even if update_sun failed, begin_render_pass MUST succeed to transition the image!
        // But wait! begin_render_pass relies on rendering archetypes? NO, it relies on RenderPasses cache!
        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
        scoped_rp.end()?;
        device.record_windowless_download(cmd_buffer_handle, task_id)?;
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
  let (pool_arc, render_frontend, render_device_handle, _) = setup_render_frontend_for_tests(false);

  let width = 512;
  let height = 256;

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
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
        device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

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
          device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

          let font_id = device.allocate_rasterized_font_atlas(
            cmd_buffer_handle,
            font_hash,
            alloc::sync::Arc::new(atlas),
          )?;

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

          device.prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer_handle)?;
          let w = width as f32;
          let h = height as f32;
          #[rustfmt::skip]
          let view_proj = [
            2.0 / w, 0.0, 0.0, 0.0,
            0.0, 2.0 / h, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -1.0, -1.0, 0.0, 1.0,
          ];

          device.render_text(
            cmd_buffer_handle,
            "AetherVk Async Test",
            [10.0, 50.0], // Pixel space
            view_proj,
            (font_hash, font_id),
            48.0,
            [0.5, 1.0, 0.5, 1.0],
          )?;

          scoped_rp.end()?;
          device.record_windowless_download(cmd_buffer_handle, task_id)?;
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
  let (pool_arc, render_frontend, render_device_handle, _) = setup_render_frontend_for_tests(false);

  let width = 512;
  let height = 256;

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
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
        device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

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
          device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;
          let font_id = device.allocate_rasterized_font_atlas(
            cmd_buffer_handle,
            font_hash,
            alloc::sync::Arc::new(atlas),
          )?;

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

          device.prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer_handle)?;
          let w = width as f32;
          let h = height as f32;
          #[rustfmt::skip]
          let view_proj = [
            2.0 / w, 0.0, 0.0, 0.0,
            0.0, 2.0 / h, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -1.0, -1.0, 0.0, 1.0,
          ];

          device.render_text(
            cmd_buffer_handle,
            "AetherVk Async Test",
            [10.0, 10.0], // Pixel space
            view_proj,
            (font_hash, font_id),
            48.0,
            [0.5, 1.0, 0.5, 1.0],
          )?;

          scoped_rp.end()?;
          device.record_windowless_download(cmd_buffer_handle, task_id)?;
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
fn test_render_particles_multithreaded_impl(use_particle2: bool) {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();
  let [width, height] = render_frontend
    .with_device(render_device_handle, |device| {
      device.get_presentation_engine_extent(presentation_engine)
    })
    .unwrap();

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));
  scene.register_all_crate_components();

  let mesh_entity = scene.spawn_entity("mesh");
  let particle_sys_e = scene.spawn_entity("particles");
  let sun_e = scene.spawn_entity("sun");

  let asset_path = format!(
    "{}/Comet.glb",
    crate::gpu::ASSET_DIR.read().as_ref().unwrap()
  );
  let loaded_mesh = crate::simulation::comet::load_comet_from_gltf(&asset_path, false, None)
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
        position: Vec3f32::from_array([0.0, 5.0, 0.0]),
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
        emissive_color: [0.5, 0.5, 0.5],
        use_new_path: false,
        paint_display_mode: 0,
        sphere_center: [0.0, 0.0, 0.0],
        sphere_radius: 1.0,
        grid_color: [0.0, 0.0, 0.0],
        grid_density: 1.0, // Emissive gray
      },
    )
    .unwrap();

  let config = crate::scene::particles::ParticleEmitterComponent {
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
    use_particle2,
  };
  let sys = crate::scene::particles::ParticleSystemComponent::new(config.max_particles);

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

  scene
    .add_component(
      particle_sys_e,
      crate::scene::PhysicalMeshComponent {
        asset_path: "".to_string(),
        mesh: mesh_arc.clone(),
        emissive_intensity: 0.0,
        emissive_color: [0.0; 3],
        use_new_path: false,
        paint_display_mode: 0,
        sphere_center: [0.0, 0.0, 0.0],
        sphere_radius: 1.0,
        grid_color: [0.0, 0.0, 0.0],
        grid_density: 1.0,
      },
    )
    .unwrap();

  scene.add_component(particle_sys_e, config.clone()).unwrap();
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
      scene_physics.query2_mut::<crate::scene::particles::ParticleSystemComponent, crate::scene::particles::ParticleEmitterComponent, _>(|_, sys, config| {
        sys.accumulator += (dt * 1_000_000.0) as i64;
        while sys.accumulator >= config.delta {
          sys.accumulator -= config.delta;
          let u_emission = [0.5, 0.5];
          let mut u_particles = Vec::new();
          for i in 0..100 {
            u_particles.push([((i * 13) % 100) as f32 / 100.0, ((i * 27) % 100) as f32 / 100.0, 0.5, 0.5]);
          }
          let mesh_transform = scene_physics.global_transform(mesh_entity).unwrap();
          sys.emit_particles(
            config,
            &mesh_arc_physics,
            &uv_grid,
            mesh_transform.position,
            mesh_transform.rotation,
            mesh_transform.scale,
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
          device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;
          let _scoped_cmd =
            gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;
          device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

          let mut render_scene = RenderScene::new(
            (
              TransformComponent {
                position: Vec3f32::from_array([0.0, 10.0, 0.0]),
                rotation: Quat::identity(),
                scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
              },
              CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 0.1, 100.0),
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

          let asset_hash = mesh_arc.id;
          let res = match device.get_physical_mesh_resources(asset_hash, presentation_engine) {
            Ok(r) => r,
            Err(_) => device.create_physical_mesh_resources(
              cmd_buffer_handle,
              asset_hash,
              &crate::scene::PhysicalMeshComponent {
                asset_path: "".to_string(),
                mesh: mesh_arc.clone(),
                emissive_intensity: 0.0,
                emissive_color: [0.0; 3],
                use_new_path: false,
                paint_display_mode: 0,
                sphere_center: [0.0, 0.0, 0.0],
                sphere_radius: 1.0,
                grid_color: [0.0, 0.0, 0.0],
                grid_density: 1.0,
              },
              presentation_engine,
              "",
            )?,
          };

          let mesh_transform = scene_render.global_transform(mesh_entity).unwrap();
          let mut rel_transform = mesh_transform.clone();
          rel_transform.position = mesh_transform.position - render_scene.camera_data.absolute_pos;
          let outline: Option<[f32; 4]> = None;
          render_scene.draw_calls.push(gpu::frame::DrawCall::from_handles_and_matrix(
            res,
            mesh_arc.indices.len() as u32,
            outline,
            rel_transform.to_mat4(),
            1.0,
            [0.5, 0.5, 0.5],
            false,
            0,
            [0.0, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
            1.0,
          ));

          let config = scene_render
            .with_component(
              particle_sys_e,
              |c: &crate::scene::particles::ParticleEmitterComponent| c.clone(),
            )
            .unwrap();
          scene_render.with_component(
            particle_sys_e,
            |sys: &crate::scene::particles::ParticleSystemComponent| {
              render_scene
                .add_renderable(
                  cmd_buffer_handle,
                  device,
                  particle_sys_e,
                  Mat4x4f32::identity(),
                  crate::scene::RenderableDataRef::ParticleSystem(sys, &config),
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
            device
              .upload_particle2_systems(cmd_buffer_handle, &mut render_scene.particle2_calls)?;
            device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
            let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

            let extent = device.get_presentation_engine_extent(presentation_engine)?;
            device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent))?;
            device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent))?;
            gpu::frame::render_frame(
              device,
              cmd_buffer_handle,
              presentation_engine,
              &render_scene,
            )?;
            scoped_rp.end()?;
            device.record_windowless_download(cmd_buffer_handle, task_id)?;
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
            let r_threshold = if use_particle2 { 150 } else { 50 };
            let b_threshold = if use_particle2 { 150 } else { 50 };
            if g > 200 && r < r_threshold && b < b_threshold {
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
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();

  // TODO: remove EntityId dependency on physical mesh resource creation
  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));
  scene.register_all_crate_components();
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
    use_new_path: false,
    paint_display_mode: 0,
    sphere_center: [0.0, 0.0, 0.0],
    sphere_radius: 1.0,
    grid_color: [0.0, 0.0, 0.0],
    grid_density: 1.0,
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
      device
        .set_command_buffer_presentation_engine(cmd_buffer_handle, test_data.presentation_engine)?;

      let mut render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([0.0, 0.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 0.1, 100.0),
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
      device
        .set_command_buffer_presentation_engine(cmd_buffer_handle, test_data.presentation_engine)?;
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
        test_data.presentation_engine,
        &render_scene,
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
      device
        .set_command_buffer_presentation_engine(cmd_buffer_handle, test_data.presentation_engine)?;

      let mut render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([0.0, 0.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent::new_ortho(-5.0, 5.0, -5.0, 5.0, 0.1, 100.0),
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
        test_data.presentation_engine,
        &render_scene,
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
      device
        .set_command_buffer_presentation_engine(cmd_buffer_handle, test_data.presentation_engine)?;

      let mut render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([0.0, 0.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 0.1, 100.0),
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
      device
        .set_command_buffer_presentation_engine(cmd_buffer_handle, test_data.presentation_engine)?;
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
        test_data.presentation_engine,
        &render_scene,
      )
      .unwrap();

      render_pass_guard.end().unwrap();

      device.record_windowless_download(cmd_buffer_handle, task_id).unwrap();

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
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();
  let [width, height] = render_frontend
    .with_device(render_device_handle, |device| {
      device.get_presentation_engine_extent(presentation_engine)
    })
    .unwrap();

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));
  scene.register_all_crate_components();
  let sun_e = scene.spawn_entity("sun");

  let asset_path = format!(
    "{}/Comet.glb",
    crate::gpu::ASSET_DIR.read().as_ref().unwrap()
  );
  let mesh =
    Arc::new(crate::simulation::comet::load_comet_from_gltf(&asset_path, false, None).unwrap());

  // Radius of sun volume is hardcoded to 0.6 in update_sun. So we scale the mesh to 0.5.
  let transform = TransformComponent {
    position: Vec3f32::from_array([0.0, -5.0, 0.0]), // Place in front of camera (-y is forward)
    rotation: Quat::identity(),
    scale: Vec3f32::from_array([0.5, 0.5, 0.5]),
  };

  let mesh_comp = PhysicalMeshComponent {
    mesh: mesh.clone(),
    emissive_intensity: 5.0,
    emissive_color: [1.0, 0.5, 0.1],
    use_new_path: false,
    paint_display_mode: 0,
    sphere_center: [0.0, 0.0, 0.0],
    sphere_radius: 1.0,
    grid_color: [0.0, 0.0, 0.0],
    grid_density: 1.0, // Orange-ish emissive core
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
        device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

        let mut render_scene = RenderScene::new(
          (
            TransformComponent {
              position: Vec3f32::from_array([0.0, 0.0, 0.0]),
              rotation: Quat::identity(),
              scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
            },
            CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 0.1, 100.0),
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
          presentation_engine,
          &render_scene,
        )
        .unwrap();

        scoped_rp.end()?;

        // 5. Download rendered color image
        device.record_windowless_download(cmd_buffer_handle, render_task_id)?;

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
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();
  let [width, height] = render_frontend
    .with_device(render_device_handle, |device| {
      device.get_presentation_engine_extent(presentation_engine)
    })
    .unwrap();

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));
  scene.register_all_crate_components();
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
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

      let mut render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([0.0, 0.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 0.1, 100.0),
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
        presentation_engine,
        &render_scene,
      )?;

      scoped_rp.end()?;

      // 4. Download rendered color image
      device.record_windowless_download(cmd_buffer_handle, render_task_id)?;

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

fn test_render_particles_stress_impl(use_particle2: bool) {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, _) = setup_render_frontend_for_tests(false);

  let width = 512;
  let height = 512;

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
        crate::types::GpuResult::Ok(pe)
      })
      .unwrap()
  };

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

      let mut render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([0.0, 0.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent::new_persp(
            45.0f32.to_radians(),
            width as f32 / height as f32,
            0.1,
            1000.0,
          ),
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
      );

      let particle_sys_e = scene.spawn_entity("stress_particles");
      let config = crate::scene::particles::ParticleEmitterComponent {
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
        use_particle2,
      };
      let mut sys = crate::scene::particles::ParticleSystemComponent::new(config.max_particles);

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
        crate::scene::RenderableDataRef::ParticleSystem(&sys, &config),
        presentation_engine,
        "particle_sys_stress",
        false,
        [0.0, 0.0, 0.0, 0.0],
      )?;

      let scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;
      device.upload_particle_systems(cmd_buffer_handle, &mut render_scene.particle_calls)?;
      device.upload_particle2_systems(cmd_buffer_handle, &mut render_scene.particle2_calls)?;
      device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
      let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
      let extent = device.get_presentation_engine_extent(presentation_engine)?;

      device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent))?;
      device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent))?;

      gpu::frame::render_frame(
        device,
        cmd_buffer_handle,
        presentation_engine,
        &render_scene,
      )?;
      scoped_rp.end()?;
      device.record_windowless_download(cmd_buffer_handle, task_id)?;

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
        if use_particle2 {
          "test_render_particles2_stress.png"
        } else {
          "test_render_particles_stress.png"
        },
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
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();
  let [width, height] = render_frontend
    .with_device(render_device_handle, |device| {
      device.get_presentation_engine_extent(presentation_engine)
    })
    .unwrap();

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));
  scene.register_all_crate_components();
  for _ in 0..30 {
    scene.spawn_entity("dummy");
  }
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
    use_new_path: false,
    paint_display_mode: 0,
    sphere_center: [0.0, 0.0, 0.0],
    sphere_radius: 1.0,
    grid_color: [0.0, 0.0, 0.0],
    grid_density: 1.0,
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
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;
      let scoped_cmd =
        gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id)).unwrap();

      let mut render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([5.0, 0.0, 0.0]),
            rotation: Quat::identity(), // identity looks towards -Y (forward)
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent::new_persp(
            45.0f32.to_radians(),
            width as f32 / height as f32,
            0.1,
            100.0,
          ),
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
      );

      let mut relative_transform = transform.clone();
      relative_transform.position = transform.position - Vec3f32::from_array([5.0, 0.0, 0.0]);
      let model_matrix = relative_transform.to_mat4();

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

      let draw_call = render_scene.draw_calls.last().unwrap();
      println!(
        "outline_pipeline is some: {}",
        draw_call.outline_pipeline.is_some()
      );
      println!("draw_outline is: {}", draw_call.draw_outline);

      device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
      let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
      let extent = device.get_presentation_engine_extent(presentation_engine)?;

      device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent)).unwrap();
      device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent)).unwrap();

      gpu::frame::render_frame(
        device,
        cmd_buffer_handle,
        presentation_engine,
        &render_scene,
      )
      .unwrap();
      scoped_rp.end().unwrap();
      device.record_windowless_download(cmd_buffer_handle, task_id).unwrap();

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
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();
  let [width, height] = render_frontend
    .with_device(render_device_handle, |device| {
      device.get_presentation_engine_extent(presentation_engine)
    })
    .unwrap();

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));
  scene.register_all_crate_components();
  for _ in 0..30 {
    scene.spawn_entity("dummy");
  }
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
    use_new_path: false,
    paint_display_mode: 0,
    sphere_center: [0.0, 0.0, 0.0],
    sphere_radius: 1.0,
    grid_color: [0.0, 0.0, 0.0],
    grid_density: 1.0,
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
        device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;
        let scoped_cmd =
          gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id)).unwrap();

        let mut render_scene = RenderScene::new(
          (
            TransformComponent {
              position: Vec3f32::from_array([5.0, 0.0, 0.0]),
              rotation: Quat::identity(),
              scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
            },
            CameraComponent::new_persp(
              45.0f32.to_radians(),
              width as f32 / height as f32,
              0.1,
              100.0,
            ),
          ),
          aethervk_oshal_rlib::os::time::TimeReadings::default(),
          [width, height],
        );

        let mut relative_transform = transform.clone();
        relative_transform.position = transform.position - Vec3f32::from_array([5.0, 0.0, 0.0]);
        let model_matrix = relative_transform.to_mat4();

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
          presentation_engine,
          &render_scene,
        )
        .unwrap();

        scoped_rp.end().unwrap();
        device.record_windowless_download(cmd_buffer_handle, task_id).unwrap();

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
        device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;
        let scoped_cmd =
          gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id)).unwrap();

        let mut render_scene = RenderScene::new(
          (
            TransformComponent {
              position: Vec3f32::from_array([5.0, 0.0, 0.0]),
              rotation: Quat::identity(),
              scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
            },
            CameraComponent::new_persp(
              45.0f32.to_radians(),
              width as f32 / height as f32,
              0.1,
              100.0,
            ),
          ),
          aethervk_oshal_rlib::os::time::TimeReadings::default(),
          [width, height],
        );

        let mut relative_transform = transform.clone();
        relative_transform.position = transform.position - Vec3f32::from_array([5.0, 0.0, 0.0]);
        let model_matrix = relative_transform.to_mat4();

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
          presentation_engine,
          &render_scene,
        )
        .unwrap();

        scoped_rp.end().unwrap();
        device.record_windowless_download(cmd_buffer_handle, task_id).unwrap();

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
  let (pool_arc, render_frontend, render_device_handle, _) = setup_render_frontend_for_tests(false);

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
        buffer_count: 3,
      };
      let pe3 = device.create_presentation_engine(&params_win)?;

      device.generate_sky()?;

      crate::types::GpuResult::Ok((pe1, pe2, pe3))
    })
    .unwrap();

  let engines = [pe1, pe2, pe3];

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));
  scene.register_all_crate_components();
  let entity_id = scene.spawn_entity("mesh");
  let mesh = Arc::new(crate::simulation::comet::generate_uv_sphere(
    2.0, 10, 10, 1.0,
  ));
  let mesh_comp = PhysicalMeshComponent {
    mesh,
    emissive_intensity: 1.0,
    emissive_color: [1.0, 1.0, 1.0],
    use_new_path: false,
    paint_display_mode: 0,
    sphere_center: [0.0, 0.0, 0.0],
    sphere_radius: 1.0,
    grid_color: [0.0, 0.0, 0.0],
    grid_density: 1.0,
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

      let _ = render_frontend_render.with_device(render_device_handle, |device| {
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
        device.set_command_buffer_presentation_engine(cmd_buffer_handle, pe)?;

        let mut render_scene = RenderScene::new(
          (
            TransformComponent {
              position: Vec3f32::from_array([0.0, 0.0, 0.0]),
              rotation: Quat::identity(),
              scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
            },
            CameraComponent::new_persp(
              45.0f32.to_radians(),
              extent[0] as f32 / extent[1] as f32,
              0.1,
              100.0,
            ),
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

          crate::gpu::frame::render_frame(device, cmd_buffer_handle, pe, &render_scene)?;
          render_pass_guard.end()?;
          if device.is_presentation_engine_windowless(pe)? {
            device.record_windowless_download(cmd_buffer_handle, task_id)?;
          }
          println!("Thread: submitting...");
          cmd_scope.submit()?;
        }

        println!("Thread: presenting...");
        device.present(
          pe,
          acquire_result.image_index as usize,
          acquire_result.frame_index as usize,
        )?;
        println!("Thread: reading download...");

        if device.is_presentation_engine_windowless(pe)? {
          let actual_device = device.as_any().downcast_ref::<Device>().unwrap();
          let size = (extent[0] * extent[1] * 4) as usize;
          let mut buffer = alloc::vec::Vec::with_capacity(size);
          unsafe { buffer.set_len(size) };

          let start_time = std::time::Instant::now();
          while !device.is_task_completed(task_id)? {
            if start_time.elapsed().as_secs() > 2 {
              println!(
                "Task {} is stuck! Timeline cached: {}",
                task_id,
                DebugTrackedRwLock::read(&actual_device.res).timeline_manager.get_cached_value()
              );
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

      let res = render_frontend_resize.with_device(render_device_handle, |device| {
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

#[test]
fn test_physical_mesh2_variants() {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();
  let [width, height] = render_frontend
    .with_device(render_device_handle, |device| {
      device.get_presentation_engine_extent(presentation_engine)
    })
    .unwrap();

  let width = 512;
  let height = 128; // Wide image for 4 spheres

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
        crate::types::GpuResult::Ok(pe)
      })
      .unwrap()
  };

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));

  // Create 4 entities
  let e_normal = scene.spawn_entity("normal");
  let e_emissive = scene.spawn_entity("emissive");
  let e_painted = scene.spawn_entity("painted");
  let e_outline = scene.spawn_entity("outline");

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

      let mut render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([0.0, 10.0, 0.0]), // Camera looking at origin
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent::new_ortho(-8.0, 8.0, -2.0, 2.0, 0.1, 100.0),
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
      );

      let asset_path = format!(
        "{}/Comet.glb",
        crate::gpu::ASSET_DIR.read().as_ref().unwrap()
      );
      let mesh_arc = std::sync::Arc::new(
        crate::simulation::comet::load_comet_from_gltf(&asset_path, false, None).unwrap(),
      );

      // Common helper to add mesh to render scene
      let mut add_mesh = |entity: crate::scene::EntityId,
                          pos: f32,
                          intensity: f32,
                          color: [f32; 3],
                          outline: Option<[f32; 4]>,
                          _paint_mode: u32|
       -> GpuResult<()> {
        // Wait, physical mesh doesn't expose paint mode yet in PhysicalMeshComponent.
        // We will just test normal, emissive, outline since paint mode is hardcoded in device.rs to 0 right now.
        // Let's modify the component or just test the 3 accessible states + default.

        let mesh_comp = PhysicalMeshComponent {
          asset_path: asset_path.clone(),
          mesh: mesh_arc.clone(),
          emissive_intensity: intensity,
          emissive_color: color,
          use_new_path: false,
          paint_display_mode: 0,
          sphere_center: [0.0, 0.0, 0.0],
          sphere_radius: 1.0,
          grid_color: [0.0, 0.0, 0.0],
          grid_density: 1.0,
        };

        let t = TransformComponent {
          position: Vec3f32::from_array([pos, 0.0, 0.0]),
          rotation: Quat::identity(),
          scale: Vec3f32::from_array([0.5, 0.5, 0.5]),
        };

        let asset_hash = mesh_comp.mesh.id;
        let res = match device.get_physical_mesh2_resources(asset_hash, presentation_engine) {
          Ok(r) => r,
          Err(_) => device.create_physical_mesh2_resources(
            cmd_buffer_handle,
            asset_hash,
            &mesh_comp,
            presentation_engine,
            "test_mesh",
          )?,
        };

        let dc = gpu::frame::DrawCall::from_handles_and_matrix(
          res,
          mesh_arc.indices.len() as u32,
          outline,
          t.to_mat4(),
          intensity,
          color,
          true,
          0,
          [0.0, 0.0, 0.0],
          1.0,
          [0.0, 0.0, 0.0],
          1.0,
        );
        render_scene.draw_calls.push(dc);

        Ok(())
      };

      let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

      add_mesh(e_normal, -6.0, 0.0, [0.0, 0.0, 0.0], None, 0)?;
      add_mesh(e_emissive, -2.0, 5.0, [1.0, 0.0, 0.0], None, 0)?;
      add_mesh(e_painted, 2.0, 0.0, [0.0, 1.0, 0.0], None, 1)?; // Paint mode cannot be set directly from comp right now, but it's a variant.
      add_mesh(
        e_outline,
        6.0,
        0.0,
        [0.0, 0.0, 1.0],
        Some([1.0, 1.0, 0.0, 1.0]),
        0,
      )?;

      {
        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

        let extent = device.get_presentation_engine_extent(presentation_engine)?;
        device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent))?;
        device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent))?;

        gpu::frame::render_frame(
          device,
          cmd_buffer_handle,
          presentation_engine,
          &render_scene,
        )?;
        scoped_rp.end()?;
        device.record_windowless_download(cmd_buffer_handle, task_id)?;
      }

      _scoped_cmd.submit().unwrap();

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

      // Save it
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
        "test_physical_mesh2_variants.png",
        &export_buffer,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .expect("Failed to save rendered png");

      // Verify we drew something in 4 separate quadrants roughly.
      let mut obj_count = 0;
      let quadrant_width = width / 4;
      for q in 0..4 {
        let mut hit = false;
        for x in (q * quadrant_width)..((q + 1) * quadrant_width) {
          for y in 0..height {
            let idx = (y * width + x) as usize * 4;
            let b = export_buffer[idx];
            let g = export_buffer[idx + 1];
            let r = export_buffer[idx + 2];
            if r > 10 || g > 10 || b > 10 {
              hit = true;
              break;
            }
          }
          if hit {
            break;
          }
        }
        if hit {
          obj_count += 1;
        }
      }

      assert_eq!(obj_count, 4, "Expected to see 4 distinct objects rendered");

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}

#[test]
fn test_render_particles_windowless() {
  test_render_particles_windowless_impl(false);
}

#[test]
fn test_render_particles2_windowless() {
  test_render_particles_windowless_impl(true);
}

#[test]
fn test_render_particles_multithreaded() {
  test_render_particles_multithreaded_impl(false);
}

#[test]
fn test_render_particles2_multithreaded() {
  test_render_particles_multithreaded_impl(true);
}

#[test]
fn test_render_particles_stress() {
  test_render_particles_stress_impl(false);
}

#[test]
fn test_render_particles2_stress() {
  test_render_particles_stress_impl(true);
}

#[test]
fn test_painting_mode_write_and_verify() {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, _) = setup_render_frontend_for_tests(false);

  let width = 512;
  let height = 512;

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
        Ok(pe)
      })
      .unwrap()
  };

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));
  let e_paint = scene.spawn_entity("paint_mesh");

  let asset_path = format!(
    "{}/Comet.glb",
    crate::gpu::ASSET_DIR.read().as_ref().unwrap()
  );
  let mesh_arc = std::sync::Arc::new(
    crate::simulation::comet::load_comet_from_gltf(&asset_path, false, None).unwrap(),
  );

  let mut render_scene = RenderScene::new(
    (
      TransformComponent {
        position: Vec3f32::from_array([0.0, 3.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
      CameraComponent::new_ortho(-2.0, 2.0, -2.0, 2.0, 0.1, 100.0),
    ),
    aethervk_oshal_rlib::os::time::TimeReadings::default(),
    [width, height],
  );

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

      let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

      let mesh_comp = PhysicalMeshComponent {
        asset_path: asset_path.clone(),
        mesh: mesh_arc.clone(),
        emissive_intensity: 1.0,
        emissive_color: [1.0, 1.0, 1.0],
        use_new_path: true,
        paint_display_mode: 1,
        sphere_center: [0.0, 0.0, 0.0],
        sphere_radius: 1.0,
        grid_color: [0.0, 0.0, 0.0],
        grid_density: 1.0, // Color mode
      };

      let asset_hash = mesh_comp.mesh.id;
      let res = device.create_physical_mesh2_resources(
        cmd_buffer_handle,
        asset_hash,
        &mesh_comp,
        presentation_engine,
        "paint_mesh",
      )?;

      let dc = gpu::frame::DrawCall::from_handles_and_matrix(
        res,
        mesh_arc.indices.len() as u32,
        None,
        Mat4x4f32::identity(),
        1.0,
        [1.0, 1.0, 1.0],
        true,
        1,
        [0.0, 0.0, 0.0],
        1.0,
        [0.0, 0.0, 0.0],
        1.0,
      );
      render_scene.draw_calls.push(dc);

      // 1. Render BEFORE painting
      {
        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        {
          let mut _scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
          let extent = device.get_presentation_engine_extent(presentation_engine)?;
          device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent))?;
          device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent))?;
          gpu::frame::render_frame(
            device,
            cmd_buffer_handle,
            presentation_engine,
            &render_scene,
          )?;
        }
        device.record_windowless_download(cmd_buffer_handle, task_id)?;
        _scoped_cmd.submit().unwrap();
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

      let mut buffer_before = vec![0u8; (width * height * 4) as usize];
      device.read_windowless_download(task_id, &mut buffer_before)?;
      image::save_buffer(
        "paint_before.png",
        &buffer_before,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .ok();

      // 2. PAINT into the buffer
      // We need to access the mapped memory of the emissive_paint_image.
      // This is inside RenderDevice (Vulkan implementation).
      {
        let vk_device =
          device.as_any().downcast_ref::<crate::gpu_backends::vulkan::device::Device>().unwrap();
        let res_guard = DebugTrackedRwLock::read(&vk_device.res);
        let mesh_id = gpu::RenderableInstanceId::from_physical_mesh(mesh_comp.mesh.id);
        let mesh2_res = DebugTrackedRwLock::read(&res_guard.physical_mesh2_resources);
        let paint_image_resource = mesh2_res.as_ref().unwrap().get(&mesh_id).unwrap();
        let paint_image = paint_image_resource.emissive_paint_image.as_ref().unwrap();

        let alloc_info = DebugTrackedRwLock::read(&vk_device.res)
          .allocator
          .allocator
          .get_allocation_info(&paint_image.allocation);
        let mapped_ptr = alloc_info.mapped_data as *mut u8;
        assert!(!mapped_ptr.is_null(), "Paint image must be mmapped");

        // Write a BIG red square in the middle of 1024x1024 texture
        unsafe {
          for y in 400..600 {
            for x in 400..600 {
              let offset = (y * 1024 + x) * 4;
              *mapped_ptr.add(offset) = 255; // R
              *mapped_ptr.add(offset + 1) = 0; // G
              *mapped_ptr.add(offset + 2) = 0; // B
              *mapped_ptr.add(offset + 3) = 255; // A (Distribution)
            }
          }
        }
      } // <-- locks released

      // 3. Render AFTER painting
      device.start_frame()?;
      let task_id_after = device.create_task();
      let acquire_result_after = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle_after = device.get_command_buffer()?;
      device
        .set_command_buffer_presentation_engine(cmd_buffer_handle_after, presentation_engine)?;

      {
        let _scoped_cmd =
          gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle_after, Some(task_id_after))?;
        device.begin_render_pass(
          cmd_buffer_handle_after,
          presentation_engine,
          &acquire_result_after,
        )?;
        {
          let mut _scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle_after);
          let extent = device.get_presentation_engine_extent(presentation_engine)?;
          device.set_viewport(cmd_buffer_handle_after, &gpu::Viewport::from_extent(extent))?;
          device.set_scissor(cmd_buffer_handle_after, &gpu::Rect2D::from_extent(extent))?;
          gpu::frame::render_frame(
            device,
            cmd_buffer_handle_after,
            presentation_engine,
            &render_scene,
          )?;
        }
        device.record_windowless_download(cmd_buffer_handle_after, task_id_after)?;
        _scoped_cmd.submit().unwrap();
      }

      device.present(
        presentation_engine,
        acquire_result_after.image_index as usize,
        acquire_result_after.frame_index as usize,
      )?;
      while !device.is_task_completed(task_id_after)? {
        std::thread::sleep(std::time::Duration::from_millis(10));
      }
      device.success_task(task_id_after);

      let mut buffer_after = vec![0u8; (width * height * 4) as usize];
      device.read_windowless_download(task_id_after, &mut buffer_after)?;
      image::save_buffer(
        "paint_after.png",
        &buffer_after,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .ok();

      // 4. Assert that painting happened
      // Check if some pixels became more red or different
      let mut diff_count = 0;
      for i in 0..buffer_before.len() {
        if buffer_before[i] != buffer_after[i] {
          diff_count += 1;
        }
      }
      assert!(
        diff_count > 0,
        "Rendered image should have changed after painting"
      );

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}

#[test]
fn test_render_multiple_soi_windowless() {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, _) = setup_render_frontend_for_tests(false);

  let width = 512;
  let height = 512;

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
        crate::types::GpuResult::Ok(pe)
      })
      .unwrap()
  };

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));
  scene.register_all_crate_components();

  // 1. Create Macro Frame
  let macro_frame = scene.spawn_entity("macro_frame");
  scene
    .add_component(
      macro_frame,
      TransformComponent {
        position: Vec3f32::from_array([0.0, 0.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();
  scene
    .add_component(
      macro_frame,
      crate::scene::ReferenceFrameComponent {
        frame_type: crate::scene::ReferenceFrameType::Macro,
        scale: 1.0,
        soi_radius: f32::MAX,
        _padding: 0,
      },
    )
    .unwrap();

  // 2. Create Micro Frame A (SOI 1)
  let micro_frame_a = scene.spawn_entity("micro_frame_a");
  scene.set_parent(micro_frame_a, Some(macro_frame));
  scene
    .add_component(
      micro_frame_a,
      TransformComponent {
        position: Vec3f32::from_array([10.0, 0.0, 0.0]), // offset by 10 on X
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();
  scene
    .add_component(
      micro_frame_a,
      crate::scene::ReferenceFrameComponent {
        frame_type: crate::scene::ReferenceFrameType::Micro,
        scale: 0.1, // 10x smaller scale inside
        soi_radius: 50.0,
        _padding: 0,
      },
    )
    .unwrap();

  // 3. Create Micro Frame B (SOI 2)
  let micro_frame_b = scene.spawn_entity("micro_frame_b");
  scene.set_parent(micro_frame_b, Some(macro_frame));
  scene
    .add_component(
      micro_frame_b,
      TransformComponent {
        position: Vec3f32::from_array([-100.0, 0.0, 0.0]), // offset by -100 on X
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();
  scene
    .add_component(
      micro_frame_b,
      crate::scene::ReferenceFrameComponent {
        frame_type: crate::scene::ReferenceFrameType::Micro,
        scale: 1.0, // 1:1 scale
        soi_radius: 50.0,
        _padding: 0,
      },
    )
    .unwrap();

  // 4. Create Micro Frame C (SOI 3) - Centered in front
  let micro_frame_c = scene.spawn_entity("micro_frame_c");
  scene.set_parent(micro_frame_c, Some(macro_frame));
  scene
    .add_component(
      micro_frame_c,
      TransformComponent {
        position: Vec3f32::from_array([0.0, -5.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();
  scene
    .add_component(
      micro_frame_c,
      crate::scene::ReferenceFrameComponent {
        frame_type: crate::scene::ReferenceFrameType::Micro,
        scale: 0.01,
        soi_radius: 50.0,
        _padding: 0,
      },
    )
    .unwrap();

  let mesh = Arc::new(crate::simulation::comet::generate_uv_sphere(
    2.0, 10, 10, 1.0,
  ));

  // Spawn an object in Micro Frame A
  let obj_a = scene.spawn_entity("obj_a");
  scene.set_parent(obj_a, Some(micro_frame_a));
  scene
    .add_component(
      obj_a,
      TransformComponent {
        position: Vec3f32::from_array([0.0, 0.0, 0.0]), // Local to micro_frame_a
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();
  scene
    .add_component(
      obj_a,
      PhysicalMeshComponent {
        mesh: mesh.clone(),
        emissive_intensity: 1.0,
        emissive_color: [1.0, 0.0, 0.0], // Red
        use_new_path: false,
        paint_display_mode: 0,
        sphere_center: [0.0, 0.0, 0.0],
        sphere_radius: 1.0,
        grid_color: [0.0, 0.0, 0.0],
        grid_density: 1.0,
        asset_path: "test".to_string(),
      },
    )
    .unwrap();

  // Spawn an object in Micro Frame B
  let obj_b = scene.spawn_entity("obj_b");
  scene.set_parent(obj_b, Some(micro_frame_b));
  scene
    .add_component(
      obj_b,
      TransformComponent {
        position: Vec3f32::from_array([0.0, 0.0, 0.0]), // Local to micro_frame_b
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();
  scene
    .add_component(
      obj_b,
      PhysicalMeshComponent {
        mesh: mesh.clone(),
        emissive_intensity: 1.0,
        emissive_color: [0.0, 0.0, 1.0], // Blue
        use_new_path: false,
        paint_display_mode: 0,
        sphere_center: [0.0, 0.0, 0.0],
        sphere_radius: 1.0,
        grid_color: [0.0, 0.0, 0.0],
        grid_density: 1.0,
        asset_path: "test".to_string(),
      },
    )
    .unwrap();

  // Spawn an object in Micro Frame C
  let obj_c = scene.spawn_entity("obj_c");
  scene.set_parent(obj_c, Some(micro_frame_c));
  scene
    .add_component(
      obj_c,
      TransformComponent {
        position: Vec3f32::from_array([0.0, 0.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([3.2, 3.2, 3.2]), // 3.2 * 0.01 = 0.032 radius
      },
    )
    .unwrap();
  scene
    .add_component(
      obj_c,
      PhysicalMeshComponent {
        mesh: mesh.clone(),
        emissive_intensity: 1.0,
        emissive_color: [0.0, 1.0, 0.0], // Green
        use_new_path: false,
        paint_display_mode: 0,
        sphere_center: [0.0, 0.0, 0.0],
        sphere_radius: 1.0,
        grid_color: [0.0, 0.0, 0.0],
        grid_density: 1.0,
        asset_path: "test".to_string(),
      },
    )
    .unwrap();

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

      // We place the camera such that we can see both SOIs.
      // Macro space: obj_a is at (100, 0, 0), obj_b is at (-100, 0, 0).
      // We put camera at (0, -200, 0) looking forward (+Y).
      let mut render_scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([0.0, 0.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 0.1, 1000.0),
        ),
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
      );

      // Now we populate draw calls manually simulating what RenderSceneExtraction does
      // We will resolve their world matrices.

      let scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

      // Object A: parent is micro_frame_a (pos 10, scale 0.1).
      // Global pos: 10 + (0 * 0.1) = 10.
      // Let's compute global transform matrix manually.
      let mat_a = Mat4x4f32::translation(Vec3f32::from_components(1.0, -10.0, 0.0))
        * Mat4x4f32::from_scale(Vec3f32::from_components(1.0, 1.0, 1.0));
      let pm_a = scene.with_component(obj_a, |pm: &PhysicalMeshComponent| pm.clone()).unwrap();

      let asset_hash_a = pm_a.mesh.id;
      let res_a = device.create_physical_mesh2_resources(
        cmd_buffer_handle,
        asset_hash_a,
        &pm_a,
        presentation_engine,
        "obj_a",
      )?;
      render_scene.draw_calls.push(gpu::frame::DrawCall::from_handles_and_matrix(
        res_a,
        mesh.indices.len() as u32,
        None,
        mat_a,
        pm_a.emissive_intensity,
        pm_a.emissive_color,
        true,
        0,
        [0.0, 0.0, 0.0],
        1.0,
        [0.0, 0.0, 0.0],
        1.0,
      ));

      // Object B: parent is micro_frame_b (pos -2, scale 1.0)
      let mat_b = Mat4x4f32::translation(Vec3f32::from_components(-1.0, -10.0, 0.0));
      let pm_b = scene.with_component(obj_b, |pm: &PhysicalMeshComponent| pm.clone()).unwrap();

      let asset_hash_b = pm_b.mesh.id;
      let res_b = device.create_physical_mesh2_resources(
        cmd_buffer_handle,
        asset_hash_b,
        &pm_b,
        presentation_engine,
        "obj_b",
      )?;
      render_scene.draw_calls.push(gpu::frame::DrawCall::from_handles_and_matrix(
        res_b,
        mesh.indices.len() as u32,
        None,
        mat_b,
        pm_b.emissive_intensity,
        pm_b.emissive_color,
        true,
        0,
        [0.0, 0.0, 0.0],
        1.0,
        [0.0, 0.0, 0.0],
        1.0,
      ));

      // Object C: parent is micro_frame_c (pos 0, -5, scale 0.01)
      let mat_c = Mat4x4f32::translation(Vec3f32::from_components(0.0, -5.0, 0.0))
        * Mat4x4f32::from_scale(Vec3f32::from_components(0.032, 0.032, 0.032));
      let pm_c = scene.with_component(obj_c, |pm: &PhysicalMeshComponent| pm.clone()).unwrap();

      let asset_hash_c = pm_c.mesh.id;
      let res_c = device.create_physical_mesh2_resources(
        cmd_buffer_handle,
        asset_hash_c,
        &pm_c,
        presentation_engine,
        "obj_c",
      )?;
      render_scene.draw_calls.push(gpu::frame::DrawCall::from_handles_and_matrix(
        res_c,
        mesh.indices.len() as u32,
        None,
        mat_c,
        pm_c.emissive_intensity,
        pm_c.emissive_color,
        true,
        0,
        [0.0, 0.0, 0.0],
        1.0,
        [0.0, 0.0, 0.0],
        1.0,
      ));

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

      gpu::frame::render_frame(
        device,
        cmd_buffer_handle,
        presentation_engine,
        &render_scene,
      )?;

      scoped_rp.end()?;
      device.record_windowless_download(cmd_buffer_handle, task_id)?;

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

      // Convert BGRA to RGBA and flip vertically
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
        "test_rendered_multiple_soi.png",
        &export_buffer,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .expect("Failed to save rendered png");

      // Verify that red, blue, and green pixels are found
      let mut found_red = false;
      let mut found_blue = false;
      let mut found_green = false;
      let mut max_r = 0;
      let mut max_g = 0;
      let mut max_b = 0;
      for chunk in export_buffer.chunks_exact(4) {
        let r = chunk[0];
        let g = chunk[1];
        let b = chunk[2];
        if r > max_r {
          max_r = r;
        }
        if g > max_g {
          max_g = g;
        }
        if b > max_b {
          max_b = b;
        }
        if r > 100 && g < 100 && b < 100 {
          found_red = true;
        }
        if b > 100 && r < 100 && g < 100 {
          found_blue = true;
        }
        if g > 100 && r < 100 && b < 100 {
          found_green = true;
        }
      }

      println!("Max R: {}, Max G: {}, Max B: {}", max_r, max_g, max_b);

      assert!(found_red, "Red object in Micro Frame A is not visible");
      assert!(found_blue, "Blue object in Micro Frame B is not visible");
      assert!(found_green, "Green object in Micro Frame C is not visible");

      Ok(())
    })
    .unwrap();

  drop(render_frontend);
}

#[test]
fn test_render_weather_ui() {
  setup_assets_dir();
  let (_pool_arc, render_frontend, render_device_handle, _) =
    setup_render_frontend_for_tests(false);
  let width = 800;
  let height = 600;

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
        crate::types::GpuResult::Ok(pe)
      })
      .unwrap()
  };

  render_frontend
    .with_device(render_device_handle, |device| {
      let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
        crate::simulation::texture_cache::TextureCache::new("AetherVk"),
      )));
      scene.register_all_crate_components();

      let root_e = scene.spawn_entity("root");
      scene
        .add_component(
          root_e,
          TransformComponent {
            position: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
        )
        .unwrap();

      let camera_e = scene.spawn_entity("camera");
      scene.set_parent(camera_e, Some(root_e));

      // We need an orthographic projection mapping [0, width] and [0, height] to NDC [-1, 1].
      // Actually, since Y points down in Vulkan NDC, mapping [0, height] to [-1, 1] means Top=0, Bottom=height.
      scene
        .add_component(
          camera_e,
          TransformComponent {
            position: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
        )
        .unwrap();
      scene
        .add_component(
          camera_e,
          CameraComponent::new_ortho(0.0, width as f32, 0.0, height as f32, -1.0, 1.0),
        )
        .unwrap();

      // Background Gradient UI
      let bg_e = scene.spawn_entity("bg_gradient");
      scene
        .add_component(
          bg_e,
          crate::scene::ui::Transform2DComponent {
            size: [width as f32, height as f32],
            local_position: [0.0, 0.0],
            global_depth: 0, // bottom-most
            local_z_index: -100,
            ..Default::default()
          },
        )
        .unwrap();
      scene
        .add_component(
          bg_e,
          crate::scene::ui::UiComponent {
            color_start: [0.5, 0.8, 1.0, 1.0], // light sky blue
            color_end: [0.1, 0.4, 0.8, 1.0],   // darker blue
            gradient_dir: [0.0, 1.0],          // vertical gradient
            opacity: 1.0,
            texture_id: 0xFFFFFFFF,
            ..Default::default()
          },
        )
        .unwrap();

      // UI Widget 1: Main Panel
      let panel_e = scene.spawn_entity("panel");
      scene
        .add_component(
          panel_e,
          crate::scene::ui::Transform2DComponent {
            size: [400.0, 300.0],
            local_position: [100.0, 100.0],
            global_depth: 1,
            ..Default::default()
          },
        )
        .unwrap();
      scene
        .add_component(
          panel_e,
          crate::scene::ui::UiComponent {
            color_start: [0.8, 0.9, 1.0, 0.8], // Light blue transparent
            color_end: [0.5, 0.7, 1.0, 0.8],
            gradient_dir: [0.0, 1.0],
            border_radius: [20.0, 20.0, 20.0, 20.0],
            shadow_params: [0.0, 10.0, 20.0, 5.0],
            color_shadow: [0.0, 0.0, 0.0, 0.3],
            ..Default::default()
          },
        )
        .unwrap();

      // UI Widget 2: Side Panel (Saved Locations)
      let side_panel_e = scene.spawn_entity("side_panel");
      scene
        .add_component(
          side_panel_e,
          crate::scene::ui::Transform2DComponent {
            local_position: [520.0, 100.0],
            size: [200.0, 140.0],
            global_depth: 1,
            ..Default::default()
          },
        )
        .unwrap();
      scene
        .add_component(
          side_panel_e,
          crate::scene::ui::UiComponent {
            color_start: [0.9, 0.95, 1.0, 0.8],
            border_radius: [15.0, 15.0, 15.0, 15.0],
            shadow_params: [0.0, 5.0, 10.0, 0.0],
            color_shadow: [0.0, 0.0, 0.0, 0.15],
            ..Default::default()
          },
        )
        .unwrap();

      // Load icons
      fn load_texture(path: &str, id: u64) -> Option<(u64, Texture)> {
        if let Ok(img) = image::open(path) {
          let img = img.into_rgba8();
          let tex = crate::simulation::comet::Texture {
            width: img.width(),
            height: img.height(),
            data: img.into_raw().into(),
            format: crate::simulation::comet::TexelFormat::R8G8B8A8_UNORM,
            has_mipmaps: false,
          };
          Some((id, tex))
        } else {
          None
        }
      };

      let mut sun_tex = load_texture("../../test_assets/sun.png", 1);
      let mut cloud_tex = load_texture("../../test_assets/cloud.png", 2);

      let sun_e = scene.spawn_entity("sun_icon");
      scene.set_parent(sun_e, Some(root_e));
      // NDC coordinates: Top-left of panel is (100, 100). Center of sun icon around (180, 200)
      // NDC X: (180 / 800) * 2 - 1 = -0.55
      // NDC Y: (200 / 600) * 2 - 1 = -0.33
      // Size pct: 100 / 800 = 0.125
      let mut t = TransformComponent::default();
      t.position = Vec3f32::from_array([-0.55, -0.33, 0.0]);
      scene.add_component(sun_e, t).unwrap();
      scene
        .add_component(
          sun_e,
          crate::scene::ImageBillboardComponent {
            texture_id: 0, // sun_tex, TODO refactor: this should hold file mapped texture data.
            billboard_type: BillboardType::ScreenSpace {
              pct_width: 0.15,
              pct_height: 0.2,
            },
          },
        )
        .unwrap();

      let cloud_e = scene.spawn_entity("cloud_icon");
      scene.set_parent(cloud_e, Some(root_e));
      let mut t = TransformComponent::default();
      t.position = Vec3f32::from_array([-0.45, -0.25, 0.0]); // overlapping sun
      scene.add_component(cloud_e, t).unwrap();
      scene
        .add_component(
          cloud_e,
          crate::scene::ImageBillboardComponent {
            texture_id: 0, // cloud_tex, TODO ECS shouldn't store RenderDevice handles directly!
            billboard_type: BillboardType::ScreenSpace {
              pct_width: 0.15,
              pct_height: 0.2,
            },
          },
        )
        .unwrap();

      // -- Commands --
      let task_id = device.create_task();
      device.start_frame().unwrap();
      let acquire_result = device.acquire_next_image(presentation_engine).unwrap();
      let cmd_buffer_handle = device.get_command_buffer().unwrap();
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

      {
        let _scoped_cmd =
          gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id)).unwrap();

        // Ensure billboard resources are created TODO: check move to first frame in scene_extraction. create should do nothing if already existing
        device.create_billboard_resources(cmd_buffer_handle, presentation_engine).unwrap();

        let sun_id = if let Some((id, tex)) = sun_tex.take() {
          Some(device.add_billboard_texture(cmd_buffer_handle, id, &tex, 0).unwrap())
        } else {
          None
        }
        .unwrap();
        let cloud_id = if let Some((id, tex)) = cloud_tex.take() {
          Some(device.add_billboard_texture(cmd_buffer_handle, id, &tex, 0).unwrap())
        } else {
          None
        }
        .unwrap();
        scene
          .with_component_mut(sun_e, |c: &mut scene::ImageBillboardComponent| {
            c.texture_id = sun_id as _
          })
          .unwrap();
        scene
          .with_component_mut(cloud_e, |c: &mut scene::ImageBillboardComponent| {
            c.texture_id = cloud_id as _
          })
          .unwrap();

        // We need Font
        let atlas =
          crate::scene::text::FontAtlas::from_path(aethervk_oshal_rlib::os::FONT_PATH, 32.0)
            .unwrap();
        let font_hash = atlas.hash_metadata();
        let font_id = device
          .allocate_rasterized_font_atlas(
            cmd_buffer_handle,
            font_hash,
            alloc::sync::Arc::new(atlas),
          )
          .unwrap();

        crate::scene::ui::update_ui_layouts(&scene, [width as f32, height as f32]);

        let render_scene = scene
          .convert_scene(camera_e, false, None, [width, height])
          .unwrap()
          .build_render_scene(
            device,
            presentation_engine,
            cmd_buffer_handle,
            aethervk_oshal_rlib::os::time::TimeReadings::default(),
            [width, height],
            "test_weather",
          )
          .unwrap();

        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result).unwrap();
        let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

        device
          .set_viewport(
            cmd_buffer_handle,
            &gpu::Viewport::from_extent([width, height]),
          )
          .unwrap();
        device
          .set_scissor(
            cmd_buffer_handle,
            &gpu::Rect2D::from_extent([width, height]),
          )
          .unwrap();

        gpu::frame::render_frame(
          device,
          cmd_buffer_handle,
          presentation_engine,
          &render_scene,
        )
        .unwrap();

        // Draw Text Manually for now over the UI
        device.prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer_handle).unwrap();

        let w = width as f32;
        let h = height as f32;
        #[rustfmt::skip]
        let view_proj = [
          2.0 / w, 0.0, 0.0, 0.0,
          0.0, 2.0 / h, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0,
          -1.0, -1.0, 0.0, 1.0,
        ];

        // "San Francisco, CA"
        // Top-left is 120, 130
        device
          .render_text(
            cmd_buffer_handle,
            "San Francisco, CA",
            [120.0, 130.0],
            view_proj,
            (font_hash, font_id),
            24.0,
            [0.0, 0.0, 0.0, 1.0],
          )
          .unwrap();

        // "21 C"
        device
          .render_text(
            cmd_buffer_handle,
            "21 C",
            [360.0, 210.0],
            view_proj,
            (font_hash, font_id),
            48.0,
            [0.0, 0.0, 0.0, 1.0],
          )
          .unwrap();

        // "Partly Cloudy"
        device
          .render_text(
            cmd_buffer_handle,
            "Partly Cloudy",
            [360.0, 255.0],
            view_proj,
            (font_hash, font_id),
            18.0,
            [0.0, 0.0, 0.0, 1.0],
          )
          .unwrap();

        scoped_rp.end().unwrap();
        device.record_windowless_download(cmd_buffer_handle, task_id).unwrap();
      }

      device
        .present(
          presentation_engine,
          acquire_result.image_index as usize,
          acquire_result.frame_index as usize,
        )
        .unwrap();

      while !device.is_task_completed(task_id).unwrap() {
        std::thread::sleep(std::time::Duration::from_millis(10));
      }
      device.success_task(task_id);

      let mut buffer = vec![0u8; (width * height * 4) as usize];
      device.read_windowless_download(task_id, &mut buffer).unwrap();

      let out_path = std::path::Path::new("test_ui_rendering.png");
      if let Some(p) = out_path.parent() {
        std::fs::create_dir_all(p).unwrap();
      }

      // 2. Efficiently swap the Blue and Red channels in-place (BGRA -> RGBA)
      for pixel in buffer.chunks_exact_mut(4) {
        // pixel[0] is B, pixel[1] is G, pixel[2] is R, pixel[3] is A
        pixel.swap(0, 2);
      }
      image::save_buffer(out_path, &buffer, width, height, image::ColorType::Rgba8).unwrap();

      // TODO Determine format mapping (BGRA vs RGBA) based on standard Vulkan swapchain/download behavior
      // In windowless, the format might be BGRA8_UNORM. Let's provide a flexible check.
      let get_pixel = |x: u32, y: u32| -> (u8, u8, u8, u8) {
        let idx = ((y * width + x) * 4) as usize;
        // Return raw layout (could be BGRA)
        (
          buffer[idx],
          buffer[idx + 1],
          buffer[idx + 2],
          buffer[idx + 3],
        )
      };

      // 1. Background Check (Expected Gradient)
      let bg = get_pixel(10, 10);
      assert!(
        bg.2 > 200 && bg.0 < 150,
        "Background should be sky blue gradient, got {:?}",
        bg
      );

      // 2. Main Panel Check (Expected Light Blue blended over Gradient)
      let main_panel = get_pixel(300, 250);
      assert!(
        main_panel.2 > 200 && main_panel.1 > 150 && main_panel.0 > 100,
        "Main panel mismatch: {:?}",
        main_panel
      );

      // 3. Side Panel Check
      let side_panel = get_pixel(620, 170);

      // 4. Sun Check
      let sun_pixel = get_pixel(180, 201);
      assert!(
        sun_pixel.0 > 200 && sun_pixel.1 > 150 && sun_pixel.2 < 180,
        "Sun pixel mismatch: {:?}",
        sun_pixel
      );

      assert!(
        side_panel.0 > 200 && side_panel.1 > 200 && side_panel.2 > 200,
        "Side panel should be light grey/white, got {:?}",
        side_panel
      );

      // Save for visual verification (Note: if format is BGRA, it will look reddish in standard image viewers if saved directly as RGBA)
      // We'll flip BGRA to RGBA if necessary for the saved png.
      // Usually Vulkan on Mac with windowless creates BGRA8Unorm.
      for chunk in buffer.chunks_exact_mut(4) {
        let b = chunk[0];
        let r = chunk[2];
        chunk[0] = r;
        chunk[2] = b;
      }

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}

#[test]
fn test_render_bvhwire2_windowless() {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();
  let [width, height] = render_frontend
    .with_device(render_device_handle, |device| {
      device.get_presentation_engine_extent(presentation_engine)
    })
    .unwrap();

  let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("AetherVk"),
  )));
  scene.register_all_crate_components();
  let mesh_entity = scene.spawn_entity("mesh");

  let asset_path = format!(
    "{}/Comet.glb",
    crate::gpu::ASSET_DIR.read().as_ref().unwrap()
  );
  let mut mesh = crate::simulation::comet::load_comet_from_gltf(&asset_path, false, None).unwrap();

  // Add a dummy BVH to the mesh to ensure something is drawn
  mesh.bvh = Some(LinearBVH::<f32> {
    header: LinearBVHHeader {
      preciseness: 0,
      node_count: 1,
      primitive_count: 0,
    },
    nodes: vec![LinearBVHNode::<f32> {
      mass: 1.0,
      center_of_mass: [0.0, 0.0, 0.0],
      left_child_or_primitive_offset: 0,
      right_child_offset: 0,
      bound: LinearBound::<f32>::AABB(AABB::new(
        Vec3f32::from_array([-1.0, -1.0, -1.0]),
        Vec3f32::from_array([1.0, 1.0, 1.0]),
      )),
      primitive_count: 0,
    }],
    primitives: vec![],
  });

  let mesh_arc = std::sync::Arc::from(mesh);

  scene
    .add_component(
      mesh_entity,
      TransformComponent {
        position: Vec3f32::from_array([0.0, -10.0, 0.0]),
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
        emissive_intensity: 0.0,
        emissive_color: [0.0; 3],
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
      mesh_entity,
      crate::scene::BvhDebugComponent {
        node_render_states: vec![true],
        use_new_path: true,
      },
    )
    .unwrap();

  let camera_entity = scene.spawn_entity("camera");
  scene
    .add_component(
      camera_entity,
      TransformComponent {
        position: Vec3f32::from_array([0.0, 0.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();
  scene
    .add_component(
      camera_entity,
      CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 0.1, 100.0),
    )
    .unwrap();

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let acquire_result = device.acquire_next_image(presentation_engine)?;
      let cmd_buffer_handle = device.get_command_buffer()?;
      device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

      let mut extracted = scene.convert_scene(camera_entity, false, None, [width, height])?;

      let scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

      let render_scene = extracted.build_render_scene(
        device,
        presentation_engine,
        cmd_buffer_handle,
        aethervk_oshal_rlib::os::time::TimeReadings::default(),
        [width, height],
        "test_bvhwire2",
      )?;

      device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
      let scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);

      let extent = device.get_presentation_engine_extent(presentation_engine)?;
      device.set_viewport(cmd_buffer_handle, &gpu::Viewport::from_extent(extent))?;
      device.set_scissor(cmd_buffer_handle, &gpu::Rect2D::from_extent(extent))?;

      gpu::frame::render_frame(
        device,
        cmd_buffer_handle,
        presentation_engine,
        &render_scene,
      )?;
      scoped_rp.end()?;
      device.record_windowless_download(cmd_buffer_handle, task_id)?;

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

      let mut max_r = 0;
      let mut max_g = 0;
      let mut max_b = 0;

      let mut found_green = false;
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
        if g > 200 && r < 50 && b < 50 {
          found_green = true;
        }
      }

      println!("Bvhwire2 Max RGB: ({}, {}, {})", max_r, max_g, max_b);

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
        "test_rendered_bvhwire2.png",
        &export_buffer,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .expect("Failed to save rendered png");

      assert!(
        found_green,
        "BVH wireframe color (green) not found in the rendered image!"
      );

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}

#[test]
fn test_render_uniform_background() {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, presentation_engine) =
    setup_render_frontend_for_tests(true);
  let presentation_engine = presentation_engine.unwrap();
  let [width, height] = render_frontend
    .with_device(render_device_handle, |device| {
      device.get_presentation_engine_extent(presentation_engine)
    })
    .unwrap();

  let frames_to_render = [
    // Frame 0: Solid Red
    crate::gpu::BackgroundPushConstants {
      color_top: [1.0, 0.0, 0.0, 1.0],
      color_bottom: [1.0, 0.0, 0.0, 1.0],
    },
    // Frame 1: Gradient Blue to Green
    crate::gpu::BackgroundPushConstants {
      color_top: [0.0, 0.0, 1.0, 1.0],
      color_bottom: [0.0, 1.0, 0.0, 1.0],
    },
  ];

  for (frame_index, bg_constants) in frames_to_render.iter().enumerate() {
    render_frontend
      .with_device(render_device_handle, |device| {
        let task_id = device.create_task();
        device.start_frame()?;
        let acquire_result = device.acquire_next_image(presentation_engine)?;
        let cmd_buffer_handle = device.get_command_buffer()?;
        device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

        let mut render_scene = RenderScene::new(
          (
            TransformComponent {
              position: Vec3f32::from_array([0.0, 0.0, 0.0]),
              rotation: Quat::identity(),
              scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
            },
            CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 0.1, 100.0),
          ),
          aethervk_oshal_rlib::os::time::TimeReadings::default(),
          [width, height],
        );

        let pipeline = device.get_background_pipeline_key(presentation_engine)?;
        render_scene.background_call = Some(gpu::frame::BackgroundDrawCall {
          pipeline,
          color_top: bg_constants.color_top,
          color_bottom: bg_constants.color_bottom,
        });

        {
          let _scoped_cmd =
            gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;
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

          gpu::frame::render_frame(
            device,
            cmd_buffer_handle,
            presentation_engine,
            &render_scene,
          )?;

          scoped_rp.end()?;
          device.record_windowless_download(cmd_buffer_handle, task_id)?;
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

        let mut buffer = vec![0u8; (width * height * 4) as usize];
        device.read_windowless_download(task_id, &mut buffer)?;

        let mut export_buffer = buffer.clone();
        for chunk in export_buffer.chunks_exact_mut(4) {
          chunk.swap(0, 2); // BGRA to RGBA
        }

        let file_name = format!("test_rendered_background_{}.png", frame_index);
        image::save_buffer(
          &file_name,
          &export_buffer,
          width,
          height,
          image::ColorType::Rgba8,
        )
        .unwrap_or_else(|e| panic!("Failed to save rendered png: {:?}", e));

        // TODO: Vulkan already renders top -> bottom, don't flip. Also, we need support for different pixel formats on download
        let get_pixel = |x: u32, y: u32| -> (u8, u8, u8, u8) {
          let idx = ((y * width + x) * 4) as usize;
          (
            buffer[idx + 2],
            buffer[idx + 1],
            buffer[idx + 0],
            buffer[idx + 3],
          ) // Returns RGBA
        };

        if frame_index == 0 {
          let center = get_pixel(width / 2, height / 2);
          assert!(
            center.0 > 200 && center.1 < 50 && center.2 < 50,
            "Background should be solid red, got {:?}",
            center
          );
        } else if frame_index == 1 {
          let top = get_pixel(width / 2, height - 10);
          let bottom = get_pixel(width / 2, 10);
          assert!(
            top.2 > 200 && top.0 < 50,
            "Top should be blue, got {:?}",
            top
          );
          assert!(
            bottom.1 > 200 && bottom.0 < 50,
            "Bottom should be green, got {:?}",
            bottom
          );
        }

        crate::types::GpuResult::Ok(())
      })
      .unwrap();
  }
  drop(render_frontend);
}

#[test]
fn test_render_text2_basic() {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, _) = setup_render_frontend_for_tests(false);

  let width = 512;
  let height = 256;

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
        Ok(pe)
      })
      .unwrap()
  };

  let frontend_clone = render_frontend.clone();

  let render_thread = std::thread::spawn(move || {
    frontend_clone
      .with_device(render_device_handle, |device| {
        let task_id = device.create_task();
        device.start_frame()?;
        let acquire_result = device.acquire_next_image(presentation_engine)?;
        let cmd_buffer_handle = device.get_command_buffer()?;
        device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

        let atlas = alloc::sync::Arc::new(
          crate::scene::text::FontAtlas::from_path(aethervk_oshal_rlib::os::FONT_PATH, 32.0)
            .expect("Failed to load system font"),
        );
        let font_hash = atlas.hash_metadata();

        {
          let _scoped_cmd =
            gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

          let font_id =
            device.allocate_rasterized_font_atlas(cmd_buffer_handle, font_hash, atlas.clone())?;

          let style = crate::scene::text::TextStyle {
            size_pt: 48.0,
            color: [1.0, 1.0, 1.0, 1.0],
            style_flags: 0,
          };
          let mut text_batch = Vec::new();
          crate::scene::text::push_text_to_batch(
            "Hello Text2!",
            [10.0, 50.0],
            &style,
            &atlas,
            font_id,
            &mut text_batch,
          );

          let batch_call = device.upload_text2(cmd_buffer_handle, &text_batch)?.unwrap();

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

          let camera_data = crate::gpu::frame::CameraRenderData {
            pos: Vec3f32::from_components(0.0, 0.0, 0.0),
            absolute_pos: Vec3f32::from_components(0.0, 0.0, 0.0),
            rot: Quat::identity(),
            view: Mat4x4f32::identity(),
            proj: Mat4x4f32::identity(),
            view_proj: Mat4x4f32::identity(),
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
            near: 0.1,
            far: 100.0,
          };

          crate::gpu::frame::do_draw_text2_batch(
            device,
            &camera_data,
            cmd_buffer_handle,
            presentation_engine,
            &batch_call,
            [width as f32, height as f32],
          )?;

          scoped_rp.end()?;
          device.record_windowless_download(cmd_buffer_handle, task_id)?;
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

  render_frontend
    .with_device(render_device_handle, |device| {
      while !device.is_task_completed(task_id)? {
        std::thread::yield_now();
      }

      let mut buffer = vec![0u8; (width * height * 4) as usize];
      device.read_windowless_download(task_id, &mut buffer)?;

      let sum: u64 = buffer.iter().map(|&b| b as u64).sum();
      assert!(sum > 0, "Rendered text buffer is completely empty!");

      for chunk in buffer.chunks_exact_mut(4) {
        chunk.swap(0, 2);
      }

      image::save_buffer(
        "test_rendered_text2_basic.png",
        &buffer,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .expect("Failed to save rendered png");

      Ok(())
    })
    .unwrap();
}

#[test]
fn test_render_text2_styled() {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, _) = setup_render_frontend_for_tests(false);

  let width = 512;
  let height = 512;

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
        Ok(pe)
      })
      .unwrap()
  };

  let frontend_clone = render_frontend.clone();

  let render_thread = std::thread::spawn(move || {
    frontend_clone
      .with_device(render_device_handle, |device| {
        let task_id = device.create_task();
        device.start_frame()?;
        let acquire_result = device.acquire_next_image(presentation_engine)?;
        let cmd_buffer_handle = device.get_command_buffer()?;
        device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

        let atlas = Arc::new(
          FontAtlas::from_path(aethervk_oshal_rlib::os::FONT_PATH, 32.0)
            .expect("Failed to load system font"),
        );
        let font_hash = atlas.hash_metadata();

        {
          let _scoped_cmd =
            gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

          let font_id = device.allocate_rasterized_font_atlas(
            cmd_buffer_handle,
            font_hash,
            Arc::clone(&atlas),
          )?;

          let style1 = crate::scene::text::TextStyle {
            size_pt: 48.0,
            color: [1.0, 0.0, 0.0, 1.0],
            style_flags: 1,
          }; // Italic
          let style2 = crate::scene::text::TextStyle {
            size_pt: 48.0,
            color: [0.0, 1.0, 0.0, 1.0],
            style_flags: 2,
          }; // Bold
          let style3 = crate::scene::text::TextStyle {
            size_pt: 48.0,
            color: [0.0, 0.0, 1.0, 1.0],
            style_flags: 3,
          }; // Bold Italic

          let mut text_batch = Vec::new();
          crate::scene::text::push_text_to_batch(
            "Italic Text",
            [10.0, 50.0],
            &style1,
            &atlas,
            font_id,
            &mut text_batch,
          );
          crate::scene::text::push_text_to_batch(
            "Bold Text",
            [10.0, 150.0],
            &style2,
            &atlas,
            font_id,
            &mut text_batch,
          );
          crate::scene::text::push_text_to_batch(
            "Bold Italic Text",
            [10.0, 250.0],
            &style3,
            &atlas,
            font_id,
            &mut text_batch,
          );

          let batch_call = device.upload_text2(cmd_buffer_handle, &text_batch)?.unwrap();

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

          let camera_data = crate::gpu::frame::CameraRenderData {
            pos: Vec3f32::from_components(0.0, 0.0, 0.0),
            absolute_pos: Vec3f32::from_components(0.0, 0.0, 0.0),
            rot: Quat::identity(),
            view: Mat4x4f32::identity(),
            proj: Mat4x4f32::identity(),
            view_proj: Mat4x4f32::identity(),
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
            near: 0.1,
            far: 100.0,
          };

          crate::gpu::frame::do_draw_text2_batch(
            device,
            &camera_data,
            cmd_buffer_handle,
            presentation_engine,
            &batch_call,
            [width as f32, height as f32],
          )?;

          scoped_rp.end()?;
          device.record_windowless_download(cmd_buffer_handle, task_id)?;
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

  render_frontend
    .with_device(render_device_handle, |device| {
      while !device.is_task_completed(task_id)? {
        std::thread::yield_now();
      }

      let mut buffer = vec![0u8; (width * height * 4) as usize];
      device.read_windowless_download(task_id, &mut buffer)?;

      for chunk in buffer.chunks_exact_mut(4) {
        chunk.swap(0, 2);
      }

      image::save_buffer(
        "test_rendered_text2_styled.png",
        &buffer,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .expect("Failed to save rendered png");

      Ok(())
    })
    .unwrap();
}

#[test]
fn test_text2_descriptor_allocation() {
  setup_assets_dir();
  let (_pool_arc, render_frontend, render_device_handle, _presentation_engine) =
    setup_render_frontend_for_tests(true);

  render_frontend
    .with_device(render_device_handle, |device| {
      let task_id = device.create_task();
      device.start_frame()?;
      let cmd_buffer_handle = device.get_command_buffer()?;
      device
        .set_command_buffer_presentation_engine(cmd_buffer_handle, _presentation_engine.unwrap())?;

      let atlas = alloc::sync::Arc::new(
        crate::scene::text::FontAtlas::from_path(aethervk_oshal_rlib::os::FONT_PATH, 32.0).unwrap(),
      );

      {
        let _scoped_cmd = gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

        let id1 =
          device.allocate_rasterized_font_atlas(cmd_buffer_handle, 1111, Arc::clone(&atlas))?;
        let id2 =
          device.allocate_rasterized_font_atlas(cmd_buffer_handle, 2222, Arc::clone(&atlas))?;

        assert_ne!(id1, id2);

        device.free_rasterized_font_atlas(1111, id1)?;

        // This should reuse id1's slot if the pipeline advances
        let id3 = device.allocate_rasterized_font_atlas(
          cmd_buffer_handle,
          3333,
          alloc::sync::Arc::clone(&atlas),
        )?;

        // Either it's new or reused, both are valid, but usually free_descriptor_indices acts as LIFO stack
        assert_eq!(id1, id3);
      }
      device.success_task(task_id);

      Ok(())
    })
    .unwrap();
}

#[test]
fn test_render_text2_street_art() {
  setup_assets_dir();
  let (pool_arc, render_frontend, render_device_handle, _) = setup_render_frontend_for_tests(false);

  let width = 800;
  let height = 600;

  let presentation_engine = {
    let params = PresentationEngineParams::windowless(width, height);
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
        Ok(pe)
      })
      .unwrap()
  };

  let frontend_clone = render_frontend.clone();

  let render_thread = std::thread::spawn(move || {
    frontend_clone
      .with_device(render_device_handle, |device| {
        let task_id = device.create_task();
        device.start_frame()?;
        let acquire_result = device.acquire_next_image(presentation_engine)?;
        let cmd_buffer_handle = device.get_command_buffer()?;
        device.set_command_buffer_presentation_engine(cmd_buffer_handle, presentation_engine)?;

        let sys_atlas = Arc::new(
          FontAtlas::from_path(aethervk_oshal_rlib::os::FONT_PATH, 48.0)
            .expect("Failed to load system font"),
        );

        let asset_font_path = format!(
          "{}/fonts/JetBrainsMono-Regular.ttf",
          crate::gpu::ASSET_DIR.read().as_ref().unwrap()
        );
        let asset_atlas = Arc::new(
          FontAtlas::from_path(&asset_font_path, 48.0).expect("Failed to load asset font"),
        );

        let sys_hash = sys_atlas.hash_metadata();
        let asset_hash = asset_atlas.hash_metadata();

        {
          let _scoped_cmd =
            gpu::ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

          let sys_id = device.allocate_rasterized_font_atlas(
            cmd_buffer_handle,
            sys_hash,
            Arc::clone(&sys_atlas),
          )?;

          let asset_id = device.allocate_rasterized_font_atlas(
            cmd_buffer_handle,
            asset_hash,
            Arc::clone(&asset_atlas),
          )?;

          let mut text_batch = Vec::new();

          // Background layer (system font, bold italic, large, gray shadow)
          let style_bg = crate::scene::text::TextStyle {
            size_pt: 120.0,
            color: [0.2, 0.2, 0.2, 1.0],
            style_flags: 3,
          };
          crate::scene::text::push_text_to_batch(
            "VULKAN",
            [50.0, 150.0],
            &style_bg,
            &sys_atlas,
            sys_id,
            &mut text_batch,
          );
          crate::scene::text::push_text_to_batch(
            "AETHER",
            [100.0, 300.0],
            &style_bg,
            &sys_atlas,
            sys_id,
            &mut text_batch,
          );

          // Foreground layer (asset font, italic, colorful)
          let style_fg1 = crate::scene::text::TextStyle {
            size_pt: 100.0,
            color: [1.0, 0.0, 0.5, 1.0],
            style_flags: 1,
          };
          crate::scene::text::push_text_to_batch(
            "VULKAN",
            [40.0, 140.0],
            &style_fg1,
            &asset_atlas,
            asset_id,
            &mut text_batch,
          );

          let style_fg2 = crate::scene::text::TextStyle {
            size_pt: 100.0,
            color: [0.0, 1.0, 0.8, 1.0],
            style_flags: 1,
          };
          crate::scene::text::push_text_to_batch(
            "AETHER",
            [90.0, 290.0],
            &style_fg2,
            &asset_atlas,
            asset_id,
            &mut text_batch,
          );

          // Details/Tags (system font, bold, small)
          let style_tag = crate::scene::text::TextStyle {
            size_pt: 30.0,
            color: [1.0, 1.0, 0.0, 1.0],
            style_flags: 2,
          };
          crate::scene::text::push_text_to_batch(
            "street art edition",
            [300.0, 200.0],
            &style_tag,
            &sys_atlas,
            sys_id,
            &mut text_batch,
          );

          let style_tag2 = crate::scene::text::TextStyle {
            size_pt: 20.0,
            color: [1.0, 1.0, 1.0, 0.8],
            style_flags: 0,
          };
          crate::scene::text::push_text_to_batch(
            "Text2 Archetype",
            [600.0, 550.0],
            &style_tag2,
            &asset_atlas,
            asset_id,
            &mut text_batch,
          );

          // Lots of overlapping "graffiti" tags
          for i in 0..10 {
            let style_random = crate::scene::text::TextStyle {
              size_pt: 40.0 + (i as f32 * 10.0),
              color: [
                0.5 + (i as f32 * 0.05),
                0.1 * i as f32,
                1.0 - (i as f32 * 0.05),
                0.6,
              ],
              style_flags: i % 4,
            };
            let atlas = if i % 2 == 0 { &sys_atlas } else { &asset_atlas };
            let id = if i % 2 == 0 { sys_id } else { asset_id };
            crate::scene::text::push_text_to_batch(
              "TAG",
              [20.0 * i as f32, 400.0 + (i as f32 * 15.0)],
              &style_random,
              atlas,
              id,
              &mut text_batch,
            );
          }

          let batch_call = device.upload_text2(cmd_buffer_handle, &text_batch)?.unwrap();

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

          let camera_data = crate::gpu::frame::CameraRenderData {
            pos: Vec3f32::from_components(0.0, 0.0, 0.0),
            absolute_pos: Vec3f32::from_components(0.0, 0.0, 0.0),
            rot: Quat::identity(),
            view: Mat4x4f32::identity(),
            proj: Mat4x4f32::identity(),
            view_proj: Mat4x4f32::identity(),
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
            near: 0.1,
            far: 100.0,
          };

          crate::gpu::frame::do_draw_text2_batch(
            device,
            &camera_data,
            cmd_buffer_handle,
            presentation_engine,
            &batch_call,
            [width as f32, height as f32],
          )?;

          scoped_rp.end()?;
          device.record_windowless_download(cmd_buffer_handle, task_id)?;
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

  render_frontend
    .with_device(render_device_handle, |device| {
      while !device.is_task_completed(task_id)? {
        std::thread::yield_now();
      }

      let mut buffer = vec![0u8; (width * height * 4) as usize];
      device.read_windowless_download(task_id, &mut buffer)?;

      for chunk in buffer.chunks_exact_mut(4) {
        chunk.swap(0, 2);
      }

      image::save_buffer(
        "test_rendered_text2_street_art.png",
        &buffer,
        width,
        height,
        image::ColorType::Rgba8,
      )
      .expect("Failed to save rendered png");

      Ok(())
    })
    .unwrap();
}
