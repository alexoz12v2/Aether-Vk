use super::*;
use crate::simulation_api::SimulationContext;
use alloc::{boxed::Box, sync::Arc, collections::BTreeMap};
use core::ffi::{c_char, CStr};
use aethervk_core_rlib::scene::{
  BvhDebugComponent, FollowingComponent, HiddenComponent, MarkersComponent, SelectedComponent,
};

impl SimulationContext {
  pub fn startup(backend: *const c_char, width: u32, height: u32) -> Result<*mut SimulationContext, EngineError> {
    let backend_str = if backend.is_null() {
      ""
    } else {
      unsafe { CStr::from_ptr(backend).to_str().unwrap_or("") }
    };

    if backend_str != "Vulkan" {
      oshal::log!("Unsupported backend: {}", backend_str);
      return Err(EngineError::Gpu(GpuError::InvalidState));
    }

    let thread_pool = Arc::new(aethervk_oshal_rlib::os::pool::ThreadPool::new(4).map_err(|e| {
      oshal::log!("Failed to create thread pool: {:?}", e);
      EngineError::Gpu(GpuError::InvalidState)
    })?);

    let width = if width == 0 { 800 } else { width };
    let height = if height == 0 { 600 } else { height };

    let runtime_params = Box::leak(Box::new(RuntimeParams {
      render_backend_params: FnvIndexMap::new(),
    }));

    let frontend = Arc::new(
      gpu::new_render_frontend(gpu::VULKAN_RENDER_BACKEND, runtime_params).map_err(|e| {
        oshal::log!("Failed to create render frontend: {:?}", e);
        EngineError::from(e)
      })?,
    );
    let additional_params = gpu::DeviceAdditionalParams::new();
    let render_device_handle = frontend
      .take_mut_and(|context| Ok(context.init_device(0, &additional_params)?))
      .ok_or(EngineError::InvalidNullArgument)?
      .map_err(|e| {
        oshal::log!("Failed to init device: {:?}", e);
        EngineError::from(e)
      })?;

    let mut presentation_engine_result: GpuResult<gpu::PresentationEngineHandle> =
      Err(aethervk_core_rlib::types::GpuError::InvalidState);

    let result = frontend
      .take_and(|context| {
        context
          .deref_device_and(
            render_device_handle,
            &mut (
              &gpu::PresentationEngineParams::windowless(width, height),
              &mut presentation_engine_result,
            ) as *mut _ as *mut core::ffi::c_void,
            |device, data| {
              let (params, out) = unsafe {
                &mut *(data
                  as *mut (
                    &gpu::PresentationEngineParams,
                    *mut GpuResult<gpu::PresentationEngineHandle>,
                  ))
              };
              let pe = device.create_presentation_engine(*params)?;
              device.init_archetypes(pe)?;
              unsafe { **out = Ok(pe) };
              Ok(())
            },
          )
          .ok_or(aethervk_core_rlib::types::EngineError::InvalidNullArgument)
          .and_then(|r| r.map_err(aethervk_core_rlib::types::EngineError::from))
      })
      .ok_or(EngineError::InvalidNullArgument)??;

    let presentation_engine = presentation_engine_result.map_err(|e| {
      oshal::log!("Failed to create presentation engine: {:?}", e);
      EngineError::from(e)
    })?;

    let (render_tx, render_rx) = mpsc::channel(128);

    let scene = Scene::new();
    scene.register_component::<TransformComponent>(&[]);
    scene.register_component::<PhysicalMeshComponent>(&[TypeId::of::<TransformComponent>()]);
    scene.register_component::<CameraComponent>(&[TypeId::of::<TransformComponent>()]);
    scene.register_component::<CursorComponent>(&[TypeId::of::<TransformComponent>()]);
    scene.register_component::<SunComponent>(&[TypeId::of::<TransformComponent>()]);
    scene.register_component::<SkyComponent>(&[]);
    scene.register_component::<GridComponent>(&[]);
    scene.register_component::<MarkersComponent>(&[TypeId::of::<TransformComponent>()]);
    scene.register_component::<SelectedComponent>(&[]);
    scene.register_component::<FollowingComponent>(&[]);
    scene.register_component::<HiddenComponent>(&[]);
    scene.register_component::<BvhDebugComponent>(&[]);

    let root_entity = scene.spawn_entity("root");
    let _ = scene.add_component(
      root_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    );

    let camera_entity = scene.spawn_entity("camera");
    let _ = scene.add_component(
      camera_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, -400.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    );
    let _ = scene.add_component(
      camera_entity,
      CameraComponent {
        projection: Mat4x4f32::perspective_vk(
          45.0f32.to_radians(),
          width as f32 / height as f32,
          0.1,
          10000.0,
        ),
        near_plane: 0.1,
        far_plane: 10000.0,
      },
    );
    scene.set_parent(camera_entity, Some(root_entity));

    let sky_entity = scene.spawn_entity("sky");
    let _ = scene.add_component(sky_entity, SkyComponent {});
    scene.set_parent(sky_entity, Some(root_entity));

    let cursor_entity = scene.spawn_entity("cursor");
    let _ = scene.add_component(
      cursor_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    );
    let _ = scene.add_component(cursor_entity, CursorComponent {});
    scene.set_parent(cursor_entity, Some(root_entity));

    let sun_entity = scene.spawn_entity("sun");
    // recursive method here to diagonalize inertia tensor (even though it's a sphere so inertia should be close formula)
    let sun_sphere = {
      let res =
        thread_pool.spawn_tasklet(|| simulation::comet::generate_uv_sphere(0.45 * 0.95, 64, 64));
      match res {
        Ok(handle) => {
          oshal::log!("waiting for sun generate_uv_sphere tasklet");
          handle.wait()
        }
        Err(e) => {
          oshal::log!("Failed to spawn sun sphere tasklet: {:?}", e);
          return Err(EngineError::Gpu(GpuError::InvalidState));
        }
      }
    };

    let _ = scene.add_component(
      sun_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    );
    let _ = scene.add_component(
      sun_entity,
      SunComponent {
        resolution: (128, 128, 128),
      },
    );
    let _ = scene.add_component(
      sun_entity,
      PhysicalMeshComponent {
        asset_path: alloc::string::String::new(),
        mesh: alloc::sync::Arc::from(sun_sphere),
        emissive_intensity: 0.9,
        emissive_color: [1.0, 0.35, 0.02],
      },
    );
    scene.set_parent(sun_entity, Some(root_entity));

    let grid_entity = scene.spawn_entity("grid");
    let _ = scene.add_component(grid_entity, GridComponent {});
    scene.set_parent(grid_entity, Some(root_entity));

    let physics_scene = Arc::new(RwLock::new(
      aethervk_core_rlib::physics::physics_scene::PhysicsScene::build_from_scene(&scene),
    ));

    let time_info = Arc::new(RwLock::new(aethervk_oshal_rlib::os::time::TimeInfo::new(
      16667, 100000, 1.0,
    )));

    let scene_arc = Arc::new(scene);

    let render_thread_handle = start_render_thread(
      render_rx,
      Arc::clone(&scene_arc),
      Arc::clone(&frontend),
      render_device_handle,
      presentation_engine,
      cursor_entity,
      sun_entity,
    );

    // Wire callbacks for completion tracking
    let result = frontend
      .take_and(|context| {
        context
          .deref_device_and(
            render_device_handle,
            &mut Arc::clone(&thread_pool) as *mut _ as *mut core::ffi::c_void,
            |device, data| {
              let pool = unsafe { &*(data as *mut Arc<aethervk_oshal_rlib::os::pool::ThreadPool>) };
              device.wire_callbacks(Arc::clone(pool))
            },
          )
          .ok_or(aethervk_core_rlib::types::EngineError::InvalidNullArgument)
          .and_then(|r| r.map_err(aethervk_core_rlib::types::EngineError::from))
      })
      .ok_or(EngineError::InvalidNullArgument)??;

    let _ = render_tx.try_send(RenderCommand::GenerateSky);

    let (logic_tx, logic_rx) = mpsc::channel(128);
    let logic_thread_handle = start_logic_thread(logic_rx);

    let mut ctx = Box::new(SimulationContext {
      scenes: BTreeMap::new(),
      active_scene_id: 0,
      next_scene_id: 1,
      presentation_engine,
      render_frontend: frontend,
      render_device_handle,
      render_tx,
      render_thread_handle,
      logic_tx,
      logic_thread_handle,
      asset_path: None,
      window_width: width,
      window_height: height,
      logic_state: RwLock::new(Box::new(LogicState::default())),
      model_registry: BTreeMap::new(),
      next_model_id: 1,
      mesh_cache: Arc::new(aethervk_core_rlib::scene::AssetCache::new()),
      thread_pool,
      time_info,
      clear_color: [0.0, 0.0, 0.0, 1.0],
    });

    let scene_ctx = Arc::new(RwLock::new(SceneContext {
      scene: scene_arc,
      physics_scene,
      entity_map: BTreeMap::new(),
      next_entity_id: 1,
      root_entity,
      active_camera_entity: camera_entity,
      cursor_entity,
      sun_entity,
      grid_entity,
      outlines_enabled: Arc::new(AtomicBool::new(false)),
    }));

    ctx.scenes.insert(0, scene_ctx);

    if let Some(mut active) = ctx.active_scene_mut() {
      active.register_entity(root_entity);
      active.register_entity(camera_entity);
      active.register_entity(cursor_entity);
      active.register_entity(sun_entity);
      active.register_entity(grid_entity);
    }

    Ok(Box::into_raw(ctx))
  }

  pub fn shutdown(&mut self) {
    self.stop_threads();
    self.mesh_cache.clear();
  }

  pub fn start_threads(&mut self) {
    // TODO: implement it if needed, or remove
  }

  pub fn stop_threads(&mut self) {
    let _ = self.render_tx.try_send(RenderCommand::Shutdown);
    let _ = self.logic_tx.try_send(LogicCommand::default());
    if let Some(handle) = self.render_thread_handle.take() {
      handle.join();
    }
    if let Some(handle) = self.logic_thread_handle.take() {
      handle.join();
    }
  }

  pub fn simulation_tick(&mut self) {
    // Update time
    {
      let mut time = self.time_info.write();
      time.ut_update();
    }

    // Concurrent physics rebuild
    if let Some(active) = self.active_scene() {
      let workload = Box::new(PhysicsRebuildWorkload {
        scene: Arc::clone(&active.scene),
        physics_scene: Arc::clone(&active.physics_scene),
      });
      let _ = self.thread_pool.scatter(vec![workload]);
      self.thread_pool.gather(); 
    }
  }

  pub fn render_tick(&mut self) -> u64 {
    let mut task_id = 0;
    let _ = self.render_frontend.take_and(|context| {
      context
        .deref_device_and(
          self.render_device_handle,
          &mut task_id as *mut _ as *mut core::ffi::c_void,
          |device, data| {
            let tid = unsafe { &mut *(data as *mut u64) };
            *tid = device.create_task();
            Ok(())
          },
        )
        .ok_or(aethervk_core_rlib::types::EngineError::InvalidNullArgument)
        .and_then(|r| r.map_err(aethervk_core_rlib::types::EngineError::from))
    });

    if let Some(packet) = collect_render_packet(self) {
      let _ = self
        .render_tx
        .try_send(RenderCommand::RenderFrame { packet, task_id });
    }

    task_id
  }

  pub fn set_active_camera(&mut self, camera: u64) -> Result<(), EngineError> {
    let mut active_scene = self.active_scene_mut().ok_or(EngineError::InvalidNullArgument)?;
    if let Some(entity_id) = active_scene.get_entity(camera) {
      active_scene.active_camera_entity = entity_id;
      Ok(())
    } else {
      Err(EngineError::Gpu(GpuError::InvalidState))
    }
  }

  pub fn set_clear_color(&mut self, r: f32, g: f32, b: f32, a: f32) {
    let _ = self
      .render_tx
      .try_send(RenderCommand::SetClearColor([r, g, b, a]));
  }

  pub fn download_image(&mut self, buffer_ptr: *mut u8, buffer_size: usize) -> bool {
    if buffer_ptr.is_null() {
      return false;
    }
    let mut success = false;
    let done_signal = Arc::new(AtomicBool::new(false));

    let _ = self.render_tx.try_send(RenderCommand::DownloadImage {
      buffer: SendPtrMut(buffer_ptr),
      buffer_size,
      success: SendPtrMut(&mut success),
      done_signal: Arc::clone(&done_signal),
    });

    while !done_signal.load(core::sync::atomic::Ordering::Acquire) {
      oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1));
    }

    success
  }
}
