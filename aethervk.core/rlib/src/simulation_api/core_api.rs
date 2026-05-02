use crate::{
  gpu::PresentationEngineHandle,
  expect_scene, expect_scene_and_entity,
  simulation_api::structs::{
    LogicState, LogicThreadParams, RenderCommand, RenderThreadParams, SimulationSceneData,
    SimulationTaskManager, SimulationThreads,
  },
  simulation_api::SimulationContext,
  gpu::WeakRenderFrontendExt,
  types::GpuError,
  gpu,
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib as oshal;
use oshal::{
  os::time::{timeus_milliseconds, timeus_t},
  os,
};
use alloc::{boxed::Box, sync::Arc, collections::BTreeSet};
use core::{num::NonZero, ptr::addr_of_mut};
use spin::RwLock;
use crate::simulation_api::structs::CustomRenderCallback;
// TODO add scene validate

impl SimulationContext {
  // TODO adjust
  const NUM_WORKERS: usize = 8;
  const INITIAL_MAXIMUM_DELTA_TIME: timeus_t = timeus_milliseconds(16);
  const INITIAL_FIXED_DELTA_TIME: timeus_t = timeus_milliseconds(16);
  const INITIAL_TIME_SCALE: f32 = 1.0;

  pub fn startup(backend: gpu::RenderBackendId, error_debug_callback: Option<fn(&str)>) -> EngineResult<Box<SimulationContext>> {
    let mut boxed_uninit = Box::<SimulationContext>::new_uninit();
    unsafe {
      let ptr = boxed_uninit.as_mut_ptr();

      // 1. Initialize scene data
      let scenes = Arc::new(RwLock::new(SimulationSceneData::new()));
      addr_of_mut!((*ptr).scenes).write(Arc::clone(&scenes));

      // 1.5 Initialize task manager
      let task_manager = Arc::new(RwLock::new(SimulationTaskManager::new()));
      addr_of_mut!((*ptr).task_manager).write(Arc::clone(&task_manager));

      // 2. initialize presentation engine container
      addr_of_mut!((*ptr).presentation_engines).write(RwLock::new(BTreeSet::new()));

      // 3. initialize threads
      let render_thread_thread_pool =
        Arc::new(os::pool::ThreadPool::new(Self::NUM_WORKERS).map_err(|e| EngineError::from(e))?);
      let logic_thread_thread_pool = Arc::clone(&render_thread_thread_pool);
      let render_thread_time_info = Arc::new(RwLock::new(os::time::TimeInfo::new(
        Self::INITIAL_FIXED_DELTA_TIME,
        Self::INITIAL_MAXIMUM_DELTA_TIME,
        Self::INITIAL_TIME_SCALE,
      )));
      let logic_thread_time_info = Arc::clone(&render_thread_time_info);

      let logic_state = Arc::new(RwLock::new(LogicState::default()));
      addr_of_mut!((*ptr).logic_state).write(Arc::clone(&logic_state));

      // TODO test: if this fails, render frontend should drop.
      let render_thread_params = RenderThreadParams::new(
        backend,
        None,
        render_thread_thread_pool,
        render_thread_time_info,
      )?;
      let logic_thread_params = LogicThreadParams::new(
        logic_thread_thread_pool,
        logic_thread_time_info,
        Arc::clone(&task_manager),
        logic_state,
        Arc::clone(&scenes),
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
    width: u32,
    height: u32,
    window_info: crate::gpu::OpaqueNativeHandleInfo,
  ) -> EngineResult<PresentationEngineHandle> {
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
      let mut presentation_engines = self.presentation_engines.write();
      if presentation_engines.insert(h) {
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
    width: u32,
    height: u32,
  ) -> EngineResult<PresentationEngineHandle> {
    self.with_device(|render_device| {
      let params = gpu::PresentationEngineParams::windowless(width, height);
      let h = render_device.create_presentation_engine(&params)?;
      render_device.init_archetypes(h)?;
      let mut presentation_engines = self.presentation_engines.write();
      if presentation_engines.insert(h) {
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

  // TODO: async, return a task_id
  pub fn simulation_tick(
    &self,
    scene_id: u64,
    delta_time: f64,
  ) -> EngineResult<core::num::NonZero<u64>> {
    let mut task_manager = self.task_manager.write();
    let task_id = task_manager.create_task();
    self
      .threads
      .logic_thread
      .tx()
      .try_send(
        crate::simulation_api::structs::LogicCommand::SimulationTick {
          task_id: task_id.get(),
          scene_id,
          delta_time,
        },
      )
      .map_err(|_| EngineError::InvalidOperation("logic thread closed"))?;
    Ok(task_id)
  }

  // TODO all task methods throughout the whole crate return EngineResult<NonZero<u64>>
  // TODO if task insertion failed, FFI wrapper returns 0
  pub fn render_tick(
    &self,
    presentation_engine_handle: PresentationEngineHandle,
    scene_id: u64,
    window_extent: [u32; 2],
  ) -> EngineResult<core::num::NonZero<u64>> {
    self.render_tick_custom(presentation_engine_handle, scene_id, window_extent, None)
  }

  pub fn render_tick_custom(
    &self,
    presentation_engine_handle: PresentationEngineHandle,
    scene_id: u64,
    window_extent: [u32; 2],
    custom: Option<CustomRenderCallback>,
  ) -> EngineResult<core::num::NonZero<u64>> {
    let task_id = self
      .render_proxy
      .0
      .as_frontend()
      .ok_or(EngineError::InvalidOperation("render_frontend"))
      .and_then(|context| {
        context
          .with_device(self.render_proxy.1, |device| Ok(device.create_task()))
          .map_err(|e| EngineError::from(e))
      })?;

    let scenes = self.scenes.read();
    let active = scenes
      .get(&scene_id)
      .ok_or(EngineError::InvalidOperation("scene not found"))?;
    let scene = Arc::clone(&active);
    let _ = self
      .threads
      .render_thread
      .tx()
      .try_send(RenderCommand::RenderFrame(
        crate::simulation_api::structs::RenderFrame {
          presentation_engine_handle,
          task_id,
          scene,
          render_physical_meshes_outline: active
            .read()
            .outlines_enabled
            .load(core::sync::atomic::Ordering::Acquire),
          camera_entity: active.read().active_camera_entity.unwrap_or_default(),
          window_width: window_extent[0],
          window_height: window_extent[1],
          clear_color: [0.0, 0.0, 0.0, 1.0],
          sun_entity: active.read().sun_entity,
          sky_entity: active.read().sky_entity,
          cursor_entity: active.read().cursor_entity,
          custom_render_callback: custom,
        },
      ));

    Ok(unsafe { core::num::NonZero::new_unchecked(task_id) })
  }

  pub fn set_active_camera(&self, scene_id: u64, camera: u64) -> EngineResult<()> {
    let (mut scene, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      camera,
      "core_api:set_active_camera"
    );
    scene.write().active_camera_entity = Some(entity_id);
    Ok(())
  }

  pub fn dispatch_logic_command_custom(
    &self,
    custom_fn: fn(&crate::simulation_api::structs::LogicThreadContext, *mut core::ffi::c_void),
    user_data: Option<crate::simulation_api::structs::SendPtrMut<core::ffi::c_void>>,
  ) -> EngineResult<()> {
    self
      .threads
      .logic_thread
      .tx()
      .try_send(crate::simulation_api::structs::LogicCommand::Custom {
        custom_fn,
        user_data,
      })
      .map_err(|_| EngineError::InvalidOperation("logic thread closed"))
  }
}
