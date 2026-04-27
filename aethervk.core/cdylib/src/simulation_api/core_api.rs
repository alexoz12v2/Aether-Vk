use aethervk_oshal_rlib as oshal;
use aethervk_core_rlib as rlib;
use crate::simulation_api::SimulationContext;
use crate::structs::{LogicThreadParams, RenderThreadParams, SimulationSceneData, SimulationThreads};
use crate::{expect_scene, expect_scene_and_entity};
use rlib::gpu::PresentationEngineHandle;
use aethervk_oshal_rlib::os::time::{timeus_milliseconds, timeus_t};
use alloc::{boxed::Box, sync::Arc};
use alloc::collections::BTreeSet;
use core::ptr::addr_of_mut;
use spin::RwLock;
use aethervk_core_rlib::types::GpuError;
use rlib::gpu;
use rlib::types::{EngineError, EngineResult};
use oshal::os;

// TODO add scene validate

impl SimulationContext {
  // TODO adjust
  const NUM_WORKERS: usize = 8;
  const INITIAL_MAXIMUM_DELTA_TIME: timeus_t = timeus_milliseconds(16);
  const INITIAL_FIXED_DELTA_TIME: timeus_t = timeus_milliseconds(16);
  const INITIAL_TIME_SCALE: f32 = 1.0;

  pub fn startup(backend: gpu::RenderBackendId) -> EngineResult<Box<SimulationContext>> {
    let mut boxed_uninit = Box::<SimulationContext>::new_uninit();
    unsafe {
      let ptr = boxed_uninit.as_mut_ptr();

      // 1. Initialize scene data
      SimulationSceneData::new_inplace(addr_of_mut!((*ptr).scenes));

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
      // TODO test: if this fails, render frontend should drop.
      let render_thread_params = RenderThreadParams::new(
        backend,
        None,
        render_thread_thread_pool,
        render_thread_time_info,
      )?;
      let logic_thread_params =
        LogicThreadParams::new(logic_thread_thread_pool, logic_thread_time_info);
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

  pub fn create_presentation_engine(
    &mut self,
    width: u32,
    height: u32,
  ) -> EngineResult<PresentationEngineHandle> {
    self.with_device(|render_device| {
      let params = gpu::PresentationEngineParams::windowless(width, height);
      let h = render_device.create_presentation_engine(&params)?;
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
  pub fn simulation_tick(&mut self, scene_id: u64) -> EngineResult<()> {
    // TODO: everything on logic thread, not here
    // // Update time
    // let current_time = {
    //   let mut time = self.time_info.write();
    //   time.ut_update();
    //   time.current()
    // };
    // // TODO implement here fixed update (on thread?)
    // // Concurrent physics rebuild
    // let scene = expect_scene!(self.get_scene(scene_id), "core_api:simulation_tick");
    // let ps = scene
    //   .physics_scene
    //   .as_ref()
    //   .ok_or(EngineError::InvalidOperation(
    //     "core_api:simulation_tick scene doesn't have a physics scene associated to it",
    //   ))?;
    // let workload = Box::new(PhysicsRebuildWorkload {
    //   scene: Arc::clone(&scene.scene),
    //   physics_scene: Arc::clone(ps),
    // });
    // let _ = self.thread_pool.scatter(vec![workload]);
    // self.thread_pool.gather();
    Ok(())
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
      if let Some(active) = self.active_scene() {
        let scene = Arc::clone(&active.scene);
        let _ = self.render_tx.try_send(RenderCommand::RenderFrame {
          packet,
          scene,
          task_id,
        });
      }
    }

    task_id
  }

  pub fn set_active_camera(&mut self, scene_id: u64, camera: u64) -> EngineResult<()> {
    let (mut scene, entity_id) = expect_scene_and_entity!(self.get_scene_mut(scene_id), camera, "core_api:set_active_camera");
    scene.active_camera_entity = Some(entity_id);
    Ok(())
  }

  // TODO return a result of task, this will be async
  pub fn download_image(&mut self, buffer_ptr: *mut u8, buffer_size: usize) -> bool {
    if buffer_ptr.is_null() {
      return false;
    }
    let mut success = false;
    let done_signal = Arc::new(AtomicBool::new(false));

    if self
      .render_tx
      .try_send(RenderCommand::DownloadImage {
        buffer: SendPtrMut(buffer_ptr),
        buffer_size,
        success: SendPtrMut(&mut success),
        done_signal: Arc::clone(&done_signal),
      })
      .is_err()
    {
      return false;
    }

    while !done_signal.load(core::sync::atomic::Ordering::Acquire) {
      oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1));
    }

    success
  }
}
