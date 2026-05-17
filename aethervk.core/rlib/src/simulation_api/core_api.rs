//! core_api module.

use crate::{
  expect_scene, expect_scene_and_entity, gpu,
  gpu::{PresentationEngineHandle, WeakRenderFrontendExt},
  scene::{CameraComponent, TransformComponent},
  simulation::texture_cache::TextureCache,
  simulation_api::{
    SimulationContext,
    structs::{
      LogicState, LogicThreadParams, RenderThreadParams, SimulationSceneData,
      SimulationTaskManager, SimulationThreads,
    },
  },
  types::{EngineError, EngineResult, GpuError},
};
use aethervk_oshal_rlib as oshal;
use alloc::{boxed::Box, string::ToString, sync::Arc};
use core::ptr::addr_of_mut;
use oshal::{
  os,
  os::time::{timeus_milliseconds, timeus_t},
};
use spin::RwLock;
// TODO add scene validate

impl SimulationContext {
  // TODO adjust
  const NUM_WORKERS: usize = 8;
  const INITIAL_MAXIMUM_DELTA_TIME: timeus_t = timeus_milliseconds(16);
  const INITIAL_FIXED_DELTA_TIME: timeus_t = timeus_milliseconds(16);
  const INITIAL_TIME_SCALE: f32 = 1.0;

  /// TODO: Document this item
  pub fn startup(
    backend: gpu::RenderBackendId,
    error_debug_callback: Option<fn(&str)>,
  ) -> EngineResult<Box<SimulationContext>> {
    let mut boxed_uninit = Box::<SimulationContext>::new_uninit();
    unsafe {
      let ptr = boxed_uninit.as_mut_ptr();

      // 1. Initialize scene data
      let scenes = Arc::new(RwLock::new(SimulationSceneData::new()));
      addr_of_mut!((*ptr).scenes).write(Arc::clone(&scenes));

      // 1.5 Initialize task manager
      let task_manager = Arc::new(RwLock::new(SimulationTaskManager::new()));
      addr_of_mut!((*ptr).task_manager).write(Arc::clone(&task_manager));

      // 3. initialize threads
      let render_thread_thread_pool =
        Arc::new(os::pool::ThreadPool::new(Self::NUM_WORKERS).map_err(|e| EngineError::from(e))?);
      let logic_thread_thread_pool = Arc::clone(&render_thread_thread_pool);

      let logic_state = Arc::new(RwLock::new(LogicState::default()));
      addr_of_mut!((*ptr).logic_state).write(Arc::clone(&logic_state));

      let texture_cache = Arc::new(RwLock::new(TextureCache::new("AetherVk")));
      addr_of_mut!((*ptr).texture_cache).write(Arc::clone(&texture_cache));

      // TODO test: if this fails, render frontend should drop.
      let render_thread_params =
        RenderThreadParams::new(backend, error_debug_callback, render_thread_thread_pool)?;
      let logic_thread_params = LogicThreadParams::new(
        logic_thread_thread_pool,
        Arc::clone(&task_manager),
        logic_state,
        Arc::clone(&scenes),
        crate::simulation_api::structs::SendPtrMut(ptr as *mut core::ffi::c_void),
      );
      let render_proxy = (
        render_thread_params.render_frontend.weak_self(),
        render_thread_params.render_device_handle,
      );
      addr_of_mut!((*ptr).threads).write(SimulationThreads::new_running(
        render_thread_params,
        logic_thread_params,
      )?);

      // 4. Render Proxy
      addr_of_mut!((*ptr).render_proxy).write(render_proxy);

      Ok(boxed_uninit.assume_init())
    }
  }

  /// TODO: Document this item
  pub fn render_frontend(&self) -> Option<gpu::RenderFrontend> {
    self.render_proxy.0.as_frontend()
  }

  /// TODO: Document this item
  pub fn render_device_handle(&self) -> gpu::RenderDeviceHandle {
    self.render_proxy.1
  }

  /// Returns the mapped memory pointer of the emissive paint image for a given physical mesh instance
  pub fn get_emissive_paint_image_mapped_ptr(
    &self,
    mesh_id: crate::gpu::RenderableInstanceId,
  ) -> Option<*mut u8> {
    let mut mapped_ptr = None;
    let _ = self.with_device(|device| {
      mapped_ptr = device.get_emissive_paint_image_mapped_ptr(mesh_id);
      Ok(())
    });
    mapped_ptr
  }

  /// TODO: Document this item
  pub fn create_presentation_engine_windowed(
    &self,
    scene_id: u64,
    width: u32,
    height: u32,
    window_info: crate::gpu::OpaqueNativeHandleInfo,
  ) -> EngineResult<PresentationEngineHandle> {
    let scene_ctx = expect_scene!(
      self.get_scene(scene_id),
      "scene_api:create_presentation_engine"
    );
    self.with_device(|render_device| {
      let params = gpu::PresentationEngineParams {
        width,
        height,
        vsync: true,
        ty: gpu::PresentationEngineType::Window,
        window_info,
        buffer_count: 3,
      };
      let h = render_device.create_presentation_engine(&params)?;
      render_device.init_archetypes(h)?;
      let read_scene = scene_ctx.read();
      let mut presentation_engines = read_scene.presentation_engines.write();
      if presentation_engines.insert(h, crate::simulation_api::structs::PresentationEngineData { is_windowless: false, camera_entity: None, }, ).is_none() {
        Ok(h)
      } else {
        if let Err(e) = render_device.destroy_presentation_engine(h) {
          oshal::log!(
            "core_api:create_presentation_engine_windowed | failed to destroy presentation engine: {:?}",
            e
          );
        }
        Err(GpuError::InvalidState("core_api:create_presentation_engine_windowed | couldn't insert presentation engine inside map".to_string(),
        ))
      }
    })
  }

  pub fn add_perspective_camera(
    &self,
    scene_id: u64,
    presentation_engine: PresentationEngineHandle,
    name: &str,
    fov: f32,
    near: f32,
    far: f32,
  ) -> EngineResult<core::num::NonZeroU64> {
    let scene_ctx = expect_scene!(
      self.get_scene(scene_id),
      "scene_api:create_presentation_engine"
    );
    let mut scene_write = scene_ctx.write();
    let root_entity =
      scene_write.scene.get_root().ok_or(EngineError::InvalidOperation("empty scene"))?;
    let camera_entity = {
      let mut presentation_engines = scene_write.presentation_engines.write();
      let presentation_engine_data = presentation_engines.get_mut(&presentation_engine).ok_or(EngineError::InvalidOperation("[SimulationContext] core_api:add_perspective_camera: inexistent presentation engine for given scene"))?;
      if presentation_engine_data.camera_entity.is_some() {
        return Err(EngineError::InvalidOperation(
          "[SimulationContext] core_api:add_perspective_camera: presentation engine for given scene already has a camera",
        ));
      }
      let [width, height] =
        self.with_device(|device| device.get_presentation_engine_extent(presentation_engine))?;
      let camera_entity = scene_write.scene.spawn_camera(
        name,
        Some(root_entity),
        TransformComponent::default(),
        CameraComponent::new_persp(fov, width as f32 / height as f32, near, far),
      );
      presentation_engine_data.camera_entity = Some(camera_entity);
      camera_entity
    };

    let camera_id = scene_write.register_entity(camera_entity);
    Ok(core::num::NonZeroU64::new(camera_id).unwrap())
  }

  pub fn add_orthographic_camera(
    &self,
    scene_id: u64,
    presentation_engine: PresentationEngineHandle,
    name: &str,
    left: f32,
    bottom: f32,
    near: f32,
    far: f32,
  ) -> EngineResult<core::num::NonZeroU64> {
    let scene_ctx = expect_scene!(
      self.get_scene(scene_id),
      "scene_api:create_presentation_engine"
    );
    let mut scene_write = scene_ctx.write();
    let root_entity =
      scene_write.scene.get_root().ok_or(EngineError::InvalidOperation("empty scene"))?;
    let camera_entity = {
      let mut presentation_engines = scene_write.presentation_engines.write();
      let presentation_engine_data = presentation_engines.get_mut(&presentation_engine).ok_or(EngineError::InvalidOperation("[SimulationContext] core_api:add_perspective_camera: inexistent presentation engine for given scene"))?;
      if presentation_engine_data.camera_entity.is_some() {
        return Err(EngineError::InvalidOperation(
          "[SimulationContext] core_api:add_orthographic_camera: presentation engine for given scene already has a camera",
        ));
      }
      let [width, height] =
        self.with_device(|device| device.get_presentation_engine_extent(presentation_engine))?;
      let camera_entity = scene_write.scene.spawn_camera(
        name,
        Some(root_entity),
        TransformComponent::default(),
        CameraComponent::new_ortho(
          left,
          left + width as f32,
          bottom,
          bottom + height as f32,
          near,
          far,
        ),
      );
      presentation_engine_data.camera_entity = Some(camera_entity);
      camera_entity
    };

    let camera_id = scene_write.register_entity(camera_entity);
    Ok(core::num::NonZeroU64::new(camera_id).unwrap())
  }

  /// TODO: Document this item
  pub fn create_presentation_engine(
    &self,
    scene_id: u64,
    width: u32,
    height: u32,
  ) -> EngineResult<PresentationEngineHandle> {
    let scene_ctx = expect_scene!(
      self.get_scene(scene_id),
      "scene_api:create_presentation_engine"
    );
    self.with_device(|render_device| {
      let params = gpu::PresentationEngineParams::windowless(width, height);
      let h = render_device.create_presentation_engine(&params)?;
      render_device.init_archetypes(h)?;
      let read_scene = scene_ctx.read();
      let mut presentation_engines = read_scene.presentation_engines.write();
      if presentation_engines
        .insert(
          h,
          crate::simulation_api::structs::PresentationEngineData {
            is_windowless: true,
            camera_entity: None,
          },
        )
        .is_none()
      {
        Ok(h)
      } else {
        if let Err(e) = render_device.destroy_presentation_engine(h) {
          oshal::log!(
            "core_api:create_presentation_engine | failed to destroy presentation engine: {:?}",
            e
          );
        }
        Err(GpuError::InvalidState(
          "core_api:create_presentation_engine | couldn't insert presentation engine inside map"
            .to_string(),
        ))
      }
    })
  }

  /// TODO: Document this item
  pub fn destroy_presentation_engine(
    &self,
    scene_id: u64,
    handle: PresentationEngineHandle,
  ) -> EngineResult<()> {
    let scene_ctx = expect_scene!(
      self.get_scene(scene_id),
      "scene_api:create_presentation_engine"
    );

    let camera_to_remove = {
      let read_scene = scene_ctx.read();
      let mut presentation_engines = read_scene.presentation_engines.write();
      if let Some(pe_data) = presentation_engines.remove(&handle) {
        pe_data.camera_entity
      } else {
        return Err(
          GpuError::InvalidState(
            "core_api:destroy_presentation_engine | presentation engine not found".to_string(),
          )
          .into(),
        );
      }
    };

    if let Some(camera_internal_id) = camera_to_remove {
      let mut write_scene = scene_ctx.write();
      write_scene.scene.remove_entity(camera_internal_id);
      write_scene.entity_map.retain(|_, v| *v != camera_internal_id);
    }

    self.with_device(|render_device| {
      render_device.destroy_presentation_engine(handle)?;
      Ok(())
    })
  }

  /// TODO: Document this item
  pub fn dispatch_logic_command_custom(
    &self,
    custom_fn: fn(
      &crate::simulation_api::structs::LogicThreadContext,
      *mut core::ffi::c_void,
    ) -> EngineResult<crate::simulation_api::structs::SimulationTaskResult>,
    user_data: Option<crate::simulation_api::structs::SendPtrMut<core::ffi::c_void>>,
  ) -> EngineResult<core::num::NonZero<u64>> {
    let mut task_manager = self.task_manager.write();
    let task_id = task_manager.create_task();
    self
      .threads
      .logic_thread
      .tx()
      .try_send(crate::simulation_api::structs::LogicCommand::Custom {
        task_id: task_id.get(),
        custom_fn,
        user_data,
      })
      .map_err(|_| EngineError::InvalidOperation("logic thread closed"))?;
    Ok(task_id)
  }
}
