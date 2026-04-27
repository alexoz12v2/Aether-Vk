use aethervk_core_rlib::{
  gpu::{
    self, RenderDevice,
    frame::{RenderScene},
  },
  scene::{
    CameraComponent, EntityId, PhysicalMeshComponent, Scene, SkyComponent, SunComponent,
    TransformComponent,
  },
  types::RuntimeParams,
};
use aethervk_oshal_rlib::math::{
  matrix::{SquareMatrix, Matrix4, mat4::Mat4x4f32},
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, vec4::Quat, Vector3},
};
use heapless::index_map::FnvIndexMap;
use std::sync::{Arc};
use aethervk_core_rlib::gpu::RwLock;

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

    let mut payload = RenderPayloadData {
      presentation_engine,
      scene: &scene,
      camera_entity,
      mesh_entity,
      sun_entity,
      sky_entity,
      width,
      height,
    };

    render_frontend
      .with_device(render_device_handle, |device| {
        let payload = unsafe { &mut *(&mut payload as *mut _ as *mut RenderPayloadData) };
        device.init_archetypes(payload.presentation_engine).unwrap();

        device.start_frame().map_err(|e| {
          println!("start_frame failed: {:?}", e);
          e
        })?;
        let acquire_result = device
          .acquire_next_image(payload.presentation_engine)
          .map_err(|e| {
            println!("acquire_next_image failed: {:?}", e);
            e
          })?;

        let mut camera_transform = TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        };
        let mut camera_component = CameraComponent {
          projection: Mat4x4f32::perspective_vk(
            std::f32::consts::FRAC_PI_4,
            payload.width as f32 / payload.height as f32,
            0.1,
            100.0,
          ),
          near_plane: 0.1,
          far_plane: 100.0,
        };
        payload
          .scene
          .with_component(payload.camera_entity, |c: &TransformComponent| {
            camera_transform = *c
          });
        payload
          .scene
          .with_component(payload.camera_entity, |c: &CameraComponent| {
            camera_component = *c
          });

        let mut render_scene = RenderScene::new((camera_transform, camera_component));

        payload
          .scene
          .with_component(payload.mesh_entity, |mesh: &PhysicalMeshComponent| {
            render_scene
              .add_renderable(
                device,
                payload.mesh_entity,
                Mat4x4f32::identity(),
                aethervk_core_rlib::scene::RenderableDataRef::PhysicalMesh(mesh),
                payload.presentation_engine,
                "mesh",
                false,
                [1.0, 1.0, 1.0, 1.0],
              )
              .unwrap();
          });

        let mut sun_transform = TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        };
        let mut sun_component = SunComponent {
          resolution: (128, 128, 128),
        };
        let mut sky_component = SkyComponent {};
        payload
          .scene
          .with_component(payload.sky_entity, |c: &SkyComponent| sky_component = *c);
        payload
          .scene
          .with_component(payload.sun_entity, |c: &TransformComponent| {
            sun_transform = *c
          });
        payload
          .scene
          .with_component(payload.sun_entity, |c: &SunComponent| sun_component = *c);

        render_scene.sun = Some((payload.sun_entity, sun_component, sun_transform.into()));
        render_scene.sky = Some((payload.sky_entity, sky_component));

        let cmd_buffer = device.get_command_buffer().unwrap();
        device.begin_command_buffer(cmd_buffer).unwrap();
        device
          .update_sun(cmd_buffer, payload.sun_entity, &sun_component)
          .unwrap();
        device
          .begin_render_pass(cmd_buffer, payload.presentation_engine, &acquire_result)
          .unwrap();

        let extent = device
          .get_presentation_engine_extent(payload.presentation_engine)
          .unwrap();
        let root_viewport = gpu::Viewport {
          x: 0.0,
          y: 0.0,
          width: extent[0] as f32,
          height: extent[1] as f32,
          min_depth: 0.0,
          max_depth: 1.0,
        };
        device.set_viewport(cmd_buffer, &root_viewport).unwrap();
        device
          .set_scissor(
            cmd_buffer,
            &gpu::Rect2D {
              offset: [0, 0],
              extent,
            },
          )
          .unwrap();

        device.render_frame(cmd_buffer, &render_scene).unwrap();
        device.end_render_pass(cmd_buffer).unwrap();

        let task_id = device.create_task();
        device
          .submit_command_buffer(cmd_buffer, Some(task_id))
          .unwrap();

        println!("Waiting for task {} to complete...", task_id);
        loop {
          match device.is_task_completed(task_id) {
            Ok(true) => {
              println!("Task {} completed successfully.", task_id);
              break;
            }
            Ok(false) => {
              std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Err(e) => {
              panic!("Task {} failed with error: {:?}", task_id, e);
            }
          }
        }

        let _present_status = device
          .present(
            payload.presentation_engine,
            acquire_result.image_index as usize,
            acquire_result.frame_index as usize,
          )
          .map_err(|e| {
            println!("present failed: {:?}", e);
            e
          })?;

        let mut buffer = vec![0u8; (payload.width * payload.height * 4) as usize];
        device
          .download_windowless_image(payload.presentation_engine, &mut buffer, Some(1))
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
          payload.width,
          payload.height,
          image::ColorType::Rgba8,
        )
        .unwrap();
        println!("Image saved to {:?}", out_path);

        Ok(())
      })
      .unwrap();
  }
}
