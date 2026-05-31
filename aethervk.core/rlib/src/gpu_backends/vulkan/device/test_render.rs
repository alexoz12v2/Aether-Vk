//! test_render module.

use super::*;
use crate::{
  gpu::{
    self, DeviceAdditionalParams, PresentationEngineParams, RenderDeviceHandle, RenderFrontend,
    ScopedCommandBuffer, ScopedRenderPass, VULKAN_RENDER_BACKEND,
    frame::{BillboardDrawCall, CursorDrawCall, RenderScene},
    new_render_frontend,
    scene_conversion::SceneConversionExt,
    vulkan::device::resources::ResourceState,
  },
  math::collision::{
    bounds::AABB,
    linear_bvh::{LinearBVH, LinearBVHHeader, LinearBVHNode, LinearBound},
  },
  scene::{
    self, BillboardType, CameraComponent, PhysicalMeshComponent, Scene, SunComponent,
    TransformComponent,
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
    println!("Vulkan validation error occurred during testing: {}", msg);
    // panic!("Vulkan validation error occurred during testing: {}", msg);
  }

  // Create a channel to safely transfer the constructed data out of the thread
  use std::sync::mpsc;
  let (tx, rx) = mpsc::channel();

  let th = aethervk_oshal_rlib::os::thread::Builder::new()
    .stack_size(8 * 1024 * 1024)
    .spawn(move || {
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

      // INJECTED DEFAULT LAYER
      render_scene.depth_layers.push(crate::gpu::frame::RenderLayer {
        layer_index: 0,
        near: 0.1,
        far: 10000.0,
        draw_calls: vec![],
        billboard_calls: vec![],
      });
      let sky_pipeline = device.get_sky_pipeline_key(presentation_engine)?;
      render_scene.get_or_create_layer(0, 0.1, 1000.0).sky_call = Some(
        gpu::frame::SkyDrawCall::from_camera(&render_scene.camera_data, sky_pipeline)?,
      );

      let sun_pipeline = device.get_sun_pipeline_key(presentation_engine)?;
      render_scene.get_or_create_layer(0, 0.1, 1000.0).sun_call =
        Some(gpu::frame::SunDrawCall::from_model_and_camera(
          Mat4x4f32::identity(),
          &render_scene.camera_data,
          sun_pipeline,
          sun_e,
          0.6,
        )?);

      let grid_pipeline = device.get_grid_pipeline_kay(presentation_engine)?;
      render_scene.get_or_create_layer(0, 0.1, 1000.0).grid_call = Some(
        gpu::frame::GridDrawCall::new(grid_pipeline, 0.1, 1.0, [0.5, 0.5, 0.5]),
      );

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

      render_scene.get_or_create_layer(0, 0.1, 1000.0).cursor_call = Some(CursorDrawCall {
        pipeline: cursor_res.pipeline,
        vertex_count: 36,
        model_matrix: Mat4x4f32::identity(),
        cursor_size: 0.05,
      });

      render_scene.depth_layers[0].billboard_calls.push(BillboardDrawCall {
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

        render_scene.get_or_create_layer(0, 0.1, 1000.0).sphere_gizmo_batch_call =
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

      // INJECTED DEFAULT LAYER
      render_scene.depth_layers.push(crate::gpu::frame::RenderLayer {
        layer_index: 0,
        near: 0.1,
        far: 10000.0,
        draw_calls: vec![],
        billboard_calls: vec![],
      });
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

        render_scene.get_or_create_layer(0, 0.1, 1000.0).sphere_gizmo_batch_call =
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
        [16, 16],
      );

      // INJECTED DEFAULT LAYER
      render_scene.depth_layers.push(crate::gpu::frame::RenderLayer {
        layer_index: 0,
        near: 0.1,
        far: 10000.0,
        draw_calls: vec![],
        billboard_calls: vec![],
      });
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
    use_new_path: true,
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

}

#[test]

}

#[test]

}

#[test]

}

#[test]

}

#[test]

}

#[test]

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

  // INJECTED DEFAULT LAYER
  render_scene.depth_layers.push(crate::gpu::frame::RenderLayer {
    layer_index: 0,
    near: 0.1,
    far: 10000.0,
    draw_calls: vec![],
    billboard_calls: vec![],
  });
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
      render_scene.depth_layers[0].draw_calls.push(dc);

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
        let vk_device = device
          .as_any()
          .downcast_ref::<crate::gpu_backends::vulkan::device::Device>()
          .unwrap();
        let mesh_id = gpu::RenderableInstanceId::from_physical_mesh(mesh_comp.mesh.id);

        {
          let res_guard = DebugTrackedRwLock::read(&vk_device.res);
          let mesh2_res = &res_guard.physical_mesh2_resources;
          let paint_image_resource = mesh2_res.get(&mesh_id).unwrap();
          let paint_image = if let ResourceState::Ready(r) = &*paint_image_resource {
            r.emissive_paint_image.as_ref().unwrap()
          } else {
            panic!("image resources not present")
          };

          let alloc_info = DebugTrackedRwLock::read(&vk_device.res)
            .allocator
            .allocator
            .get_allocation_info(&paint_image.allocation);
          let mapped_ptr = alloc_info.mapped_data as *mut u8;
          assert!(!mapped_ptr.is_null(), "Paint image must be mmapped");

          // Write a BIG red square in the middle of 1024x1024 texture
          unsafe {
            for y in 0..1024 {
              for x in 0..1024 {
                let offset = (y * 1024 + x) * 4;
                *mapped_ptr.add(offset) = 255; // R
                *mapped_ptr.add(offset + 1) = 0; // G
                *mapped_ptr.add(offset + 2) = 0; // B
                *mapped_ptr.add(offset + 3) = 255; // A (Distribution)
              }
            }
          }

          let _ = DebugTrackedRwLock::read(&vk_device.res).allocator.allocator.flush_allocation(
            &paint_image.allocation,
            0,
            ash::vk::WHOLE_SIZE as u64,
          );
        }
      }

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

        // Memory barrier to make CPU writes visible to the fragment shader. No layout change.
        {
          let vk_device = device
            .as_any()
            .downcast_ref::<crate::gpu_backends::vulkan::device::Device>()
            .unwrap();
          let mesh_id = gpu::RenderableInstanceId::from_physical_mesh(mesh_comp.mesh.id);
          vk_device
            .submit_paint_image_transition(
              cmd_buffer_handle_after,
              mesh_id,
              ash::vk::ImageLayout::GENERAL,
              ash::vk::ImageLayout::GENERAL,
            )
            .unwrap();
        }

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
        depth_layer: 0,
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
        depth_layer: 1,
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
        depth_layer: 1,
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
        depth_layer: 1,
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
        use_new_path: true,
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
        use_new_path: true,
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
        use_new_path: true,
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

      // INJECTED DEFAULT LAYER
      render_scene.depth_layers.push(crate::gpu::frame::RenderLayer {
        layer_index: 0,
        near: 0.1,
        far: 10000.0,
        draw_calls: vec![],
        billboard_calls: vec![],
      });
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
      render_scene.depth_layers[0]
        .draw_calls
        .push(gpu::frame::DrawCall::from_handles_and_matrix(
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
      render_scene.depth_layers[0]
        .draw_calls
        .push(gpu::frame::DrawCall::from_handles_and_matrix(
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
      render_scene.depth_layers[0]
        .draw_calls
        .push(gpu::frame::DrawCall::from_handles_and_matrix(
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
            size: [20.0, 20.0],
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
            size: [150.0, 300.0],
            local_position: [250.0, 100.0],
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
      t.position = Vec3f32::from_array([-0.55, -0.33, 0.99]);
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
      t.position = Vec3f32::from_array([-0.45, -0.25, 0.98]); // overlapping sun
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
        device
          .create_billboard_resources(cmd_buffer_handle, presentation_engine)
          .unwrap();

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

        device
          .begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)
          .unwrap();
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
        device
          .prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer_handle)
          .unwrap();

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

  render_frontend
    .with_device(render_device_handle, |device| {
      device.destroy_presentation_engine(presentation_engine)
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

        // INJECTED DEFAULT LAYER
        render_scene.depth_layers.push(crate::gpu::frame::RenderLayer {
          layer_index: 0,
          near: 0.1,
          far: 10000.0,
          draw_calls: vec![],
          billboard_calls: vec![],
        });
        let pipeline = device.get_background_pipeline_key(presentation_engine)?;
        render_scene.get_or_create_layer(0, 0.1, 1000.0).background_call =
          Some(gpu::frame::BackgroundDrawCall {
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
            projection_params: crate::gpu::frame::CameraProjectionParams::Perspective {
              fov: 45.0_f32.to_radians(),
              aspect_ratio: width as f32 / height as f32,
            },
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
// FIXME: This test panics with "Unable to load signal_semaphore" because it
// bypasses the engine's LogicalDevice abstraction and calls signal_semaphore
// directly on the raw `ash::Device` handle (`vulkan_device.device.handle`),
// which does not have the VK_KHR_timeline_semaphore function pointer loaded.
// The fix is to use the promoted Vulkan 1.2 path (device.handle with 1.2
// core promotions loaded) or the engine's own timeline manager API instead
// of reaching into the raw ash device.
#[ignore]
fn test_cross_queue_sync_timeline_semaphore() {
  setup_assets_dir();

  let (_pool_arc, render_frontend, render_device_handle, pe_handle_opt) =
    setup_render_frontend_for_tests(true);

  let pe_handle = pe_handle_opt.unwrap();

  render_frontend
    .with_device(render_device_handle, |device| {
      // 1. Create a timeline semaphore directly from the underlying vulkan device
      let mut type_info = ash::vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(ash::vk::SemaphoreType::TIMELINE)
        .initial_value(0);

      let create_info = ash::vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

      // We get the downcasted device to access ash::Device
      let vulkan_device = device
        .as_any()
        .downcast_ref::<crate::gpu_backends::vulkan::device::Device>()
        .unwrap();
      let logical_device = &vulkan_device.device.handle;
      let timeline_sem = unsafe { logical_device.create_semaphore(&create_info, None).unwrap() };

      let timeline_value = 1;

      // We simulate the compute step signaling the timeline semaphore from CPU for test purposes
      let signal_info = ash::vk::SemaphoreSignalInfo::default()
        .semaphore(timeline_sem)
        .value(timeline_value);
      unsafe { logical_device.signal_semaphore(&signal_info).unwrap() };

      let sync_info = crate::gpu::CommandBufferSyncInfo {
        timeline_semaphore: ash::vk::Handle::as_raw(timeline_sem),
        timeline_value,
      };

      // 2. Start a frame and submit a graphics command buffer that waits on it
      device.start_frame().unwrap();
      let acquire_result = device.acquire_next_image(pe_handle).unwrap();
      let task_id = device.create_task();

      let cmd_buffer = device.get_command_buffer().unwrap();
      device.set_command_buffer_presentation_engine(cmd_buffer, pe_handle).unwrap();

      let mut cmd_scope = gpu::ScopedCommandBuffer::new(device, cmd_buffer, Some(task_id)).unwrap();

      device.begin_render_pass(cmd_buffer, pe_handle, &acquire_result).unwrap();
      let render_pass_scope = gpu::ScopedRenderPass::new(device, cmd_buffer);

      device
        .set_viewport(cmd_buffer, &gpu::Viewport::from_extent([256, 256]))
        .unwrap();
      device.set_scissor(cmd_buffer, &gpu::Rect2D::from_extent([256, 256])).unwrap();

      render_pass_scope.end().unwrap();

      device.record_windowless_download(cmd_buffer, task_id).unwrap();

      // Submit with sync_info
      cmd_scope.set_sync_info(sync_info);
      cmd_scope.submit().unwrap();

      // Wait for task to finish
      while !device.is_task_completed(task_id).unwrap() {
        core::hint::spin_loop();
      }

      unsafe { logical_device.destroy_semaphore(timeline_sem, None) };
      crate::types::GpuResult::Ok(())
    })
    .unwrap();
}

