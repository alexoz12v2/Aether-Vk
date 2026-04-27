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

#[test]
fn test_render_all_archetypes_windowless() {
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
          position: Vec3f32::from_array([0.0, 0.0, -10.0]),
          rotation: Quat::identity(),
          scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
        },
        CameraComponent {
          projection: Mat4x4f32::perspective_vk(45.0f32.to_radians(), 1.0, 0.1, 100.0),
          near_plane: 0.1,
          far_plane: 100.0,
        },
      ));

      render_scene.sky = Some((sky_e, SkyComponent {}));

      render_scene.sun = Some((
        sun_e,
        SunComponent {
          resolution: (64, 64, 64),
        },
        TransformComponent {
          position: Vec3f32::from_array([0.0, 0.0, 0.0]),
          rotation: Quat::identity(),
          scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
        },
      ));

      render_scene.grid = Some((grid_e, GridComponent {}));

      render_scene.cursor_calls.push(CursorDrawCall {
        pipeline: cursor_res.pipeline,
        vertex_count: 36,
        model_matrix: Mat4x4f32::identity(),
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

        let sun_comp = SunComponent {
          resolution: (64, 64, 64),
        };
        device.update_sun(cmd_buffer_handle, sun_e, &sun_comp)?;

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
      }

      println!("Before present");
      device.present(
        presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      )?;
      println!("After present");

      // Wait for completion
      std::thread::sleep(std::time::Duration::from_millis(200));
      // Download image
      let mut buffer = vec![0u8; (width * height * 4) as usize];
      device.download_windowless_image(presentation_engine, &mut buffer, task_id)?;

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
        let _ = device.update_sun(cmd_buffer_handle, sun_e, &sun_comp);

        // Even if update_sun failed, begin_render_pass MUST succeed to transition the image!
        // But wait! begin_render_pass relies on rendering archetypes? NO, it relies on RenderPasses cache!
        device.begin_render_pass(cmd_buffer_handle, presentation_engine, &acquire_result)?;
        let mut scoped_rp = gpu::ScopedRenderPass::new(device, cmd_buffer_handle);
        scoped_rp.end()?;
      }

      device.present(
        presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      )?;

      std::thread::sleep(std::time::Duration::from_millis(50));

      let mut buffer = vec![0u8; (16 * 16 * 4) as usize];
      device.download_windowless_image(presentation_engine, &mut buffer, task_id)?;

      crate::types::GpuResult::Ok(())
    })
    .unwrap();

  drop(render_frontend);
}
