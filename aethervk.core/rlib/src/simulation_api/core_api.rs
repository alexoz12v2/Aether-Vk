use crate::{
  expect_scene, expect_scene_and_entity, gpu,
  gpu::PresentationEngineHandle,
  gpu::WeakRenderFrontendExt,
  simulation_api::SimulationContext,
  simulation_api::structs::{
    LogicState, LogicThreadParams, RenderThreadParams, SimulationSceneData, SimulationTaskManager,
    SimulationThreads,
  },
  types::GpuError,
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib as oshal;
use alloc::{boxed::Box, sync::Arc};
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

      // TODO test: if this fails, render frontend should drop.
      let render_thread_params = RenderThreadParams::new(backend, None, render_thread_thread_pool)?;
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

  pub fn render_frontend(&self) -> Option<gpu::RenderFrontend> {
    self.render_proxy.0.as_frontend()
  }

  pub fn render_device_handle(&self) -> gpu::RenderDeviceHandle {
    self.render_proxy.1
  }

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
      };
      let h = render_device.create_presentation_engine(&params)?;
      render_device.init_archetypes(h)?;
      let read_scene = scene_ctx.read();
      let mut presentation_engines = read_scene.presentation_engines.write();
      if presentation_engines.insert(h, false).is_none() {
        Ok(h)
      } else {
        if let Err(e) = render_device.destroy_presentation_engine(h) {
          oshal::log!(
            "core_api:create_presentation_engine_windowed | failed to destroy presentation engine: {:?}",
            e
          );
        }
        Err(GpuError::InvalidState(
          "core_api:create_presentation_engine_windowed | couldn't insert presentation engine inside map",
        ))
      }
    })
  }

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
      if presentation_engines.insert(h, true).is_none() {
        Ok(h)
      } else {
        if let Err(e) = render_device.destroy_presentation_engine(h) {
          oshal::log!(
            "core_api:create_presentation_engine | failed to destroy presentation engine: {:?}",
            e
          );
        }
        Err(GpuError::InvalidState(
          "core_api:create_presentation_engine | couldn't insert presentation engine inside map",
        ))
      }
    })
  }

  pub fn destroy_presentation_engine(
    &self,
    scene_id: u64,
    handle: PresentationEngineHandle,
  ) -> EngineResult<()> {
    let scene_ctx = expect_scene!(
      self.get_scene(scene_id),
      "scene_api:create_presentation_engine"
    );
    self.with_device(|render_device| {
      let read_scene = scene_ctx.read();
      let mut presentation_engines = read_scene.presentation_engines.write();
      if let Some(_) = presentation_engines.remove(&handle) {
        render_device.destroy_presentation_engine(handle)?;
        Ok(())
      } else {
        Err(GpuError::InvalidState(
          "core_api:destroy_presentation_engine | presentation engine not found",
        ))
      }
    })
  }

  pub fn set_active_camera(&self, scene_id: u64, camera: u64) -> EngineResult<()> {
    let (scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      camera,
      "core_api:set_active_camera"
    );
    scene.write().active_camera_entity = Some(entity_id);
    Ok(())
  }

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
