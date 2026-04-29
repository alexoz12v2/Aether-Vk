use aethervk_core_rlib::{
  gpu::{self, RenderDevice},
  scene::{
    CameraComponent, EntityId, PhysicalMeshComponent, Scene, SkyComponent, SunComponent,
    TransformComponent,
  },
  types::RuntimeParams,
};
use aethervk_oshal_rlib::math::{
  matrix::{Matrix4, mat4::Mat4x4f32},
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, vec4::Quat, Vector3},
};
use heapless::index_map::FnvIndexMap;
use std::sync::{Arc};
use aethervk_core_rlib::gpu::{FrameCancelGuard, ScopedCommandBuffer, ScopedRenderPass};
use aethervk_core_rlib::types::GpuError;
use test_utils::scene_to_render_scene;

macro_rules! try_task {
  ($task:expr, $expr:expr) => {
    $expr.map_err(|err| {
      $task.error = Some(err.clone());
      err
    })?
  };
}

#[repr(C)]
struct RenderPayloadData<'a> {
  presentation_engine: gpu::PresentationEngineHandle,
  scene: &'a Scene,
  camera_entity: EntityId,
  mesh_entity: EntityId,
  sun_entity: EntityId,
  sky_entity: EntityId,
  width: u32,
  height: u32,
}

fn main() {
  let runtime_params = Box::leak(Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
    validation_error_callback: None,
  }));

  let additional_params = gpu::DeviceAdditionalParams::new();

  // must outlive render frontend
  let thread_pool = Arc::new(aethervk_oshal_rlib::os::pool::ThreadPool::new(4).unwrap());

  {
    let render_frontend =
      gpu::new_render_frontend(gpu::VULKAN_RENDER_BACKEND, runtime_params).unwrap();
    let render_device_handle = render_frontend
      .write()
      .init_device(0, &additional_params)
      .unwrap();
    render_frontend
      .with_device(render_device_handle, |device| {
        device.wire_callbacks(Arc::clone(&thread_pool))
      })
      .unwrap();

    let width = 800;
    let height = 600;

    let asset_path = {
      let mut path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();
      while !path.join("assets").exists() {
        path = path.parent().unwrap().to_owned();
      }
      path.join("assets")
    };

    let mut guard = aethervk_core_rlib::gpu::ASSET_DIR.write();
    *guard = Some(asset_path.to_str().unwrap().to_string());
    drop(guard);

    let presentation_engine = {
      let params = gpu::PresentationEngineParams::windowless(width, height);
      render_frontend
        .with_device(render_device_handle, |device| {
          let res = device.create_presentation_engine(&params);
          println!("create_presentation_engine result: {:?}", res);
          device
            .generate_sky()
            .expect("Failed to generate background sky map!");
          res
        })
        .unwrap()
    };

    let mut scene = Scene::new();
    scene.register_component::<TransformComponent>(&[]);
    scene
      .register_component::<PhysicalMeshComponent>(&[std::any::TypeId::of::<TransformComponent>()]);
    scene.register_component::<CameraComponent>(&[std::any::TypeId::of::<TransformComponent>()]);
    scene.register_component::<SunComponent>(&[std::any::TypeId::of::<TransformComponent>()]);
    scene.register_component::<SkyComponent>(&[]);

    let camera_entity = scene.spawn_entity("entity");
    scene
      .add_component(
        camera_entity,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 5.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene
      .add_component(
        camera_entity,
        CameraComponent {
          projection: Mat4x4f32::perspective_vk(
            std::f32::consts::FRAC_PI_4,
            width as f32 / height as f32,
            0.1,
            100.0,
          ),
          near_plane: 0.1,
          far_plane: 100.0,
        },
      )
      .unwrap();

    let mesh_entity = scene.spawn_entity("entity");
    scene
      .add_component(
        mesh_entity,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();

    let model_path = {
      let mut home_dir = std::env::current_exe().unwrap();
      let mut iter: i32 = 0;
      const MAX_ITER: i32 = 32;
      while {
        let d = home_dir.join("assets/Comet.glb");
        !d.is_file() && iter < MAX_ITER
      } {
        home_dir.pop();
        iter += 1;
      }
      home_dir.join("assets/Comet.glb")
    };
    let comet = Arc::from(
      aethervk_core_rlib::simulation::comet::load_comet_from_gltf(
        model_path.to_str().unwrap(),
        false,
      )
      .expect("Failed to load comet"),
    );
    scene
      .add_component(
        mesh_entity,
        PhysicalMeshComponent {
          asset_path: "".to_string(),
          mesh: comet,
          emissive_intensity: 0.0,
          emissive_color: [0.0, 0.0, 0.0],
        },
      )
      .unwrap();

    let sky_entity = scene.spawn_entity("sky");
    scene.add_component(sky_entity, SkyComponent {}).unwrap();
    let sun_entity = scene.spawn_entity("sun");
    scene
      .add_component(
        sun_entity,
        TransformComponent {
          position: Vec3f32::from_components(10.0, 10.0, 10.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene
      .add_component(
        sun_entity,
        SunComponent {
          resolution: (128, 128, 128),
        },
      )
      .unwrap();

    render_frontend
      .with_device(render_device_handle, |device| {
        device.init_archetypes(presentation_engine).unwrap();
        let render_scene =
          scene_to_render_scene(&scene, device, presentation_engine, camera_entity, false)?;

        device.start_frame().map_err(|e| {
          println!("start_frame failed: {:?}", e);
          e
        })?;
        let mut task = Task::new(device, device.create_task());
        let acquire_result = try_task!(
          task,
          device.acquire_next_image(presentation_engine).map_err(|e| {
            println!("acquire_next_image failed: {:?}", e);
            e
          })
        );
        let present_guard = FrameCancelGuard::new(device, presentation_engine, acquire_result);

        let cmd_buffer = try_task!(task, device.get_command_buffer());
        let cmd_guard = ScopedCommandBuffer::new(device, cmd_buffer, Some(task.task_id))?;
        try_task!(task, device.begin_command_buffer(cmd_buffer));
        if let Some(sun_call) = &render_scene.sun_call {
          // TODO move to kernels
          try_task!(
            task,
            device.update_sun(cmd_buffer, sun_call.entity, (128, 128, 128))
          );
        }
        try_task!(
          task,
          device.begin_render_pass(cmd_buffer, presentation_engine, &acquire_result)
        );
        let render_pass_guard = ScopedRenderPass::new(device, cmd_buffer);

        let extent = try_task!(
          task,
          device.get_presentation_engine_extent(presentation_engine)
        );
        try_task!(
          task,
          device.set_viewport(cmd_buffer, &gpu::Viewport::from_extent(extent))
        );
        try_task!(
          task,
          device.set_scissor(cmd_buffer, &gpu::Rect2D::from_extent(extent))
        );

        try_task!(task, device.render_frame(cmd_buffer, &render_scene));
        try_task!(task, render_pass_guard.end());
        cmd_guard.submit().unwrap();
        present_guard.defuse();

        println!("Waiting for task {} to complete...", task.task_id);
        loop {
          match device.is_task_completed(task.task_id) {
            Ok(true) => {
              println!("Task {} completed successfully.", task.task_id);
              break;
            }
            Ok(false) => {
              std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Err(e) => {
              panic!("Task {} failed with error: {:?}", task.task_id, e);
            }
          }
        }

        let _present_status = device
          .present(
            presentation_engine,
            acquire_result.image_index as usize,
            acquire_result.frame_index as usize,
          )
          .map_err(|e| {
            println!("present failed: {:?}", e);
            e
          })?;

        let mut buffer = vec![0u8; (width * height * 4) as usize];
        device
          .download_windowless_image(presentation_engine, &mut buffer, Some(1))
          .map_err(|e| {
            println!("download_windowless_image failed: {:?}", e);
            e
          })?;

        println!("First few bytes: {:?}", &buffer[0..16]);
        let mut unique_pixels = std::collections::HashSet::new();
        let mut blue_count = 0;
        for chunk in buffer.chunks(4) {
          if chunk.len() == 4 {
            let p = [chunk[0], chunk[1], chunk[2], chunk[3]];
            unique_pixels.insert(p);
            if p == [0, 0, 255, 255] {
              blue_count += 1;
            }
          }
        }
        println!("Unique pixels count: {:?}", unique_pixels.len());
        println!("Blue pixels count: {}", blue_count);

        let out_path = std::env::current_exe()
          .unwrap()
          .parent()
          .unwrap()
          .join("output.png");
        image::save_buffer(
          out_path.clone(),
          &buffer,
          width,
          height,
          image::ColorType::Rgba8,
        )
        .unwrap();
        println!("Image saved to {:?}", out_path);

        Ok(())
      })
      .unwrap();
  }
}

struct Task<'a> {
  device: &'a dyn RenderDevice,
  task_id: u64,
  error: Option<GpuError>,
}

impl<'a> Task<'a> {
  fn new(device: &'a dyn RenderDevice, task_id: u64) -> Task<'a> {
    Task {
      device,
      task_id,
      error: None,
    }
  }
}

impl<'a> Drop for Task<'a> {
  fn drop(&mut self) {
    if let Some(err) = self.error.take() {
      self.device.fail_task(self.task_id, err);
    }
  }
}
