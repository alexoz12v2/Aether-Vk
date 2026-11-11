//! core_api module.

use crate::{
  expect_scene,
  gpu::{self, ASSET_DIR, PresentationEngineHandle, WeakRenderFrontendExt},
  scene::{CameraComponent, EntityId, TransformComponent},
  simulation::texture_cache::TextureCache,
  simulation_api::{
    SimulationContext,
    structs::{
      LogicState, LogicThreadParams, RenderThreadParams, SimulationSceneData, SimulationThreads,
    },
  },
  types::{EngineError, EngineResult, GpuError},
};
use aethervk_oshal_rlib::{self as oshal, os, os::files::MappedFile};
use alloc::{boxed::Box, string::ToString, sync::Arc};
use core::ptr::addr_of_mut;
use parking_lot::RwLock;

impl SimulationContext {
  const NUM_WORKERS: usize = 8;

  pub fn startup(error_debug_callback: Option<fn(&str)>) -> EngineResult<Box<SimulationContext>> {
    #[cfg(test)]
    crate::gpu::set_asset_dir_for_tests();
    let backend: gpu::RenderBackendId = gpu::VULKAN_RENDER_BACKEND;
    let mut boxed_uninit = Box::<SimulationContext>::new_uninit();
    unsafe {
      let ptr = boxed_uninit.as_mut_ptr();

      // 1. Initialize scene data
      let scenes = Arc::new(RwLock::new(SimulationSceneData::new()));
      addr_of_mut!((*ptr).scenes).write(Arc::clone(&scenes));

      // 3. initialize threads
      let render_thread_thread_pool =
        Arc::new(os::pool::ThreadPool::new(Self::NUM_WORKERS).map_err(EngineError::from)?);
      let logic_thread_thread_pool = Arc::clone(&render_thread_thread_pool);

      let logic_state = Arc::new(RwLock::new(LogicState::default()));
      addr_of_mut!((*ptr).logic_state).write(Arc::clone(&logic_state));

      let texture_cache = Arc::new(RwLock::new(TextureCache::new("AetherVk")));
      addr_of_mut!((*ptr).texture_cache).write(Arc::clone(&texture_cache));

      let mut mixer = crate::audio::AudioMixer::new(44100);

      // Pre-load UI sound assets into the lock-free mixer buffers
      let asset_dir: alloc::string::String = ASSET_DIR.read().clone().unwrap();
      let click = crate::audio::SoundBuffer::from_wav_mapped(
        &MappedFile::new(asset_dir.clone() + "/sounds/ui_click.wav")
          .map_err(|e| EngineError::Io(crate::types::IoError::FileIO(e)))?,
      );
      mixer.load_buffer(click); // AvkSoundEvent::Click == 0

      let grab = crate::audio::SoundBuffer::from_wav_mapped(
        &MappedFile::new(asset_dir.clone() + "/sounds/ui_grab.wav")
          .map_err(|e| EngineError::Io(crate::types::IoError::FileIO(e)))?,
      );
      mixer.load_buffer(grab); // AvkSoundEvent::UiGrab == 1

      let drop = crate::audio::SoundBuffer::from_wav_mapped(
        &MappedFile::new(asset_dir + "/sounds/ui_drop.wav")
          .map_err(|e| EngineError::Io(crate::types::IoError::FileIO(e)))?,
      );
      mixer.load_buffer(drop); // AvkSoundEvent::UiDrop == 2

      let audio_mixer = Arc::new(RwLock::new(mixer));
      addr_of_mut!((*ptr).audio_mixer).write(Arc::clone(&audio_mixer));

      // TODO test: if this fails, render frontend should drop.
      let render_thread_params = RenderThreadParams::new(
        backend,
        error_debug_callback,
        render_thread_thread_pool,
        Arc::new(core::sync::atomic::AtomicBool::new(false)),
      )?;
      let render_proxy = (
        render_thread_params.render_frontend.weak_self(),
        render_thread_params.render_device_handle,
      );
      let logic_thread_params = LogicThreadParams::new(
        logic_thread_thread_pool,
        logic_state,
        Arc::clone(&scenes),
        crate::simulation_api::structs::SendPtrMut(ptr as *mut core::ffi::c_void),
        (render_thread_params.render_frontend.clone(), render_proxy.1),
      );
      addr_of_mut!((*ptr).threads).write(SimulationThreads::new_running(
        render_thread_params,
        logic_thread_params,
      )?);

      // 4. Render Proxy
      addr_of_mut!((*ptr).render_proxy).write(render_proxy);

      // 5. Stall-watcher (Linux and Windows debug only)
      #[cfg(all(any(target_os = "linux", target_os = "windows"), debug_assertions))]
      {
        let sim_active = Arc::new(core::sync::atomic::AtomicBool::new(false));
        let shutdown = Arc::new(core::sync::atomic::AtomicBool::new(false));
        addr_of_mut!((*ptr).stall_watcher_simulation_active).write(Arc::clone(&sim_active));
        addr_of_mut!((*ptr).stall_watcher_shutdown).write(Arc::clone(&shutdown));
        crate::simulation_api::render_stall_watcher::spawn_render_stall_watcher(
          sim_active, shutdown,
        );
      }

      // 6. CPU particle simulation mode (activated by AETHERVK_PARTICLES_CPU=1)
      if aethervk_oshal_rlib::os::env::var("AETHERVK_PARTICLES_CPU").as_deref() == Some("1") {
        crate::gpu_backends::vulkan::physics::enable_cpu_particles();
      }

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
    self.create_presentation_engine_windowed_with_flags(scene_id, width, height, window_info, 0)
  }

  /// [`Self::create_presentation_engine_windowed`] with creation `flags`
  /// (`gpu::PRESENTATION_ENGINE_FLAG_*`; unknown bits are ignored).
  pub fn create_presentation_engine_windowed_with_flags(
    &self,
    scene_id: u64,
    width: u32,
    height: u32,
    window_info: crate::gpu::OpaqueNativeHandleInfo,
    flags: u32,
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
        low_latency: flags & gpu::PRESENTATION_ENGINE_FLAG_LOW_LATENCY != 0,
      };
      let h = render_device.create_presentation_engine(&params)?;
      render_device.init_archetypes(h)?;
      let read_scene = scene_ctx.read();
      let mut presentation_engines = read_scene.presentation_engines.write();
      let pe_data = crate::simulation_api::structs::PresentationEngineData::new(false);
      if flags & gpu::PRESENTATION_ENGINE_FLAG_START_INACTIVE != 0 {
        // paused before the frame loop can see it: inserted under the same write lock
        pe_data.active.store(false, core::sync::atomic::Ordering::Release);
      }
      if presentation_engines.insert(h, pe_data).is_none() {
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
    let root_entity = scene_write
      .scene
      .get_root()
      .ok_or(EngineError::InvalidOperation("empty scene"))?;
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

    let camera_id = EntityId::as_ffi(&camera_entity);
    Ok(core::num::NonZeroU64::new(camera_id).unwrap())
  }

  pub fn add_orthographic_camera(
    &self,
    scene_id: u64,
    presentation_engine: PresentationEngineHandle,
    name: &str,
    scale_factor: f32,
    near: f32,
    far: f32,
  ) -> EngineResult<core::num::NonZeroU64> {
    let scene_ctx = expect_scene!(
      self.get_scene(scene_id),
      "scene_api:create_presentation_engine"
    );
    let mut scene_write = scene_ctx.write();
    let root_entity = scene_write
      .scene
      .get_root()
      .ok_or(EngineError::InvalidOperation("empty scene"))?;
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
          0.0,
          width as f32 * scale_factor,
          0.0,
          height as f32 * scale_factor,
          near,
          far,
        ),
      );
      presentation_engine_data.camera_entity = Some(camera_entity);
      camera_entity
    };

    let camera_id = EntityId::as_ffi(&camera_entity);
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
          crate::simulation_api::structs::PresentationEngineData::new(true),
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

  /// Assigns an existing scene entity as the camera for the given presentation engine.
  /// This overwrites any previous binding. Use this to wire the default scene's `camera`
  /// entity to the PE instead of spawning a second camera at the origin.
  pub fn set_camera_for_presentation_engine(
    &self,
    scene_id: u64,
    presentation_engine: PresentationEngineHandle,
    camera_entity_external_id: u64,
  ) -> EngineResult<()> {
    let scene_ctx = expect_scene!(
      self.get_scene(scene_id),
      "core_api:set_camera_for_presentation_engine"
    );
    let scene_read = scene_ctx.read();
    let internal_id = scene_read.get_entity(camera_entity_external_id).ok_or_else(|| {
      EngineError::InvalidOperation(
        "[SimulationContext] core_api:set_camera_for_presentation_engine: entity not found",
      )
    })?;
    let mut presentation_engines = scene_read.presentation_engines.write();
    let pe_data = presentation_engines
      .get_mut(&presentation_engine)
      .ok_or_else(|| EngineError::InvalidOperation(
        "[SimulationContext] core_api:set_camera_for_presentation_engine: presentation engine not found",
      ))?;
    pe_data.camera_entity = Some(internal_id);
    Ok(())
  }

  /// Removes presentation engine `handle` (and its camera) from `scene_id` and destroys it before
  /// returning.
  ///
  /// The destruction runs on the render thread between two frame batches
  /// (`structs::RetireRequests`), the one point where none of the engine's images is acquired,
  /// being recorded or presented; this thread waits for it, then drains the main-thread cleanup
  /// queue, which destroys a windowed engine's swapchain and surface. A caller on the UI thread
  /// may therefore destroy the native window and close its display connection as soon as this
  /// returns. (Deferred to the discard pool, the swapchain outlived its X11 window and display
  /// connection: the render thread's next acquire crashed in the driver's surface query, and the
  /// leftover destroyed at process exit crashed a worker.) Without a running render thread the
  /// same happens inline.
  ///
  /// The camera goes last: the batches already queued for the engine are either rendered with it
  /// (ahead of the retirement) or skipped (behind it). Removed first, every queued batch failed
  /// in `build_render_scene`, and the render loop's failure backoff stretched the drain to 1 s
  /// per batch.
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

    // no scene lock is held from here: the render thread reads the scene for each frame
    let retired = if self.retire_on_render_thread(handle) {
      Ok(())
    } else {
      self.with_device(|render_device| render_device.retire_presentation_engine(handle))
    };
    // a windowed engine's swapchain and surface wait on the main-thread cleanup queue
    self.with_device(|render_device| render_device.process_main_thread_cleanup_queue())?;

    if let Some(camera_internal_id) = camera_to_remove {
      let mut write_scene = scene_ctx.write();
      // an Earth observer that posed this camera goes with it
      write_scene.earth_observers.remove(camera_internal_id);
      write_scene.scene.remove_entity(camera_internal_id);
    }
    retired
  }

  /// Pauses (`active = false`) or resumes presentation engine `handle` of `scene_id`. Paused, the
  /// logic thread builds no frame for it and the render thread neither acquires nor presents it
  /// (a frame already queued is dropped; one already being recorded completes). The engine, its
  /// swapchain and its camera stay. On resume the next frame acquires again: a swapchain that
  /// went stale meanwhile (window resized or remapped) answers OUT_OF_DATE / SUBOPTIMAL and is
  /// recreated by that acquire, as for any resize. False when the scene or the engine is missing.
  pub fn set_viewport_active(
    &self,
    scene_id: u64,
    handle: PresentationEngineHandle,
    active: bool,
  ) -> bool {
    let Some(scene_ctx) = self.get_scene(scene_id) else {
      return false;
    };
    let read_scene = scene_ctx.read();
    let presentation_engines = read_scene.presentation_engines.read();
    match presentation_engines.get(&handle) {
      Some(pe_data) => {
        pe_data.active.store(active, core::sync::atomic::Ordering::Release);
        true
      }
      None => false,
    }
  }

  /// Caps the frame rate of presentation engine `handle` of `scene_id`: the logic thread builds a
  /// frame for it at most once every `interval_us` µs of wall time (0 = every display slot, the
  /// default). The other views are unaffected. False when the scene or the engine is missing.
  pub fn set_viewport_min_frame_interval(
    &self,
    scene_id: u64,
    handle: PresentationEngineHandle,
    interval_us: u64,
  ) -> bool {
    let Some(scene_ctx) = self.get_scene(scene_id) else {
      return false;
    };
    let read_scene = scene_ctx.read();
    let presentation_engines = read_scene.presentation_engines.read();
    match presentation_engines.get(&handle) {
      Some(pe_data) => {
        pe_data
          .min_frame_interval_us
          .store(interval_us, core::sync::atomic::Ordering::Release);
        true
      }
      None => false,
    }
  }

  /// Whether presentation engine `handle` of `scene_id` is active ([`Self::set_viewport_active`]);
  /// `None` when the scene or the engine is missing.
  pub fn viewport_active(&self, scene_id: u64, handle: PresentationEngineHandle) -> Option<bool> {
    let scene_ctx = self.get_scene(scene_id)?;
    let read_scene = scene_ctx.read();
    let presentation_engines = read_scene.presentation_engines.read();
    presentation_engines
      .get(&handle)
      .map(|d| d.active.load(core::sync::atomic::Ordering::Acquire))
  }

  /// Frames of presentation engine `handle` (of any scene) that the render thread recorded,
  /// submitted and passed to `present` since the engine was created (windowless engines: frames
  /// submitted). 0 for an unknown engine.
  pub fn viewport_frame_count(&self, handle: PresentationEngineHandle) -> u64 {
    let scenes = self.scenes.read();
    let ids: alloc::vec::Vec<u64> = scenes.keys().copied().collect();
    for id in ids {
      if let Some(scene_ctx) = scenes.get_scene(id) {
        let read_scene = scene_ctx.read();
        if let Some(d) = read_scene.presentation_engines.read().get(&handle) {
          return d.frames_presented.load(core::sync::atomic::Ordering::Acquire);
        }
      }
    }
    0
  }

  /// Hands presentation engine `handle` to the render thread for destruction and waits until it
  /// is done. `false` when no render thread runs (never spawned, or exited): the caller destroys
  /// inline.
  fn retire_on_render_thread(&self, handle: PresentationEngineHandle) -> bool {
    use core::sync::atomic::{AtomicBool, Ordering};
    if !self.threads.render_thread_running() {
      return false;
    }
    let done = Arc::new(AtomicBool::new(false));
    self.threads.retire_requests.lock().push((handle, Arc::clone(&done)));
    // at most the batch in progress (an acquire is capped at 5 s); a stuck render thread is
    // reported, as Device::drop does for its polling task
    let started_us = os::time::get_monotonic_time();
    let mut warned_s = 0;
    while !done.load(Ordering::Acquire) {
      os::native::this_thread::yield_now();
      let waited_s = (os::time::get_monotonic_time() - started_us) / 1_000_000;
      if waited_s >= warned_s + 5 {
        warned_s = waited_s;
        oshal::log!(
          "destroy_presentation_engine: still waiting for the render thread to retire PE {:?} \
           after {} s",
          handle,
          waited_s
        );
      }
    }
    true
  }

  /// Drains main-thread cleanup queue. Call from UI thread tick.
  /// For windowless-only usage this is a no-op.
  pub fn process_main_thread_cleanup_queue(&self) -> crate::types::EngineResult<()> {
    self.with_device(|device| device.process_main_thread_cleanup_queue())
  }

  /// Flushes all windowed PE resources. Call from UI thread before shutdown.
  /// For windowless-only usage this is a no-op.
  pub fn flush_main_thread_cleanup_queue(&self) -> crate::types::EngineResult<()> {
    self.with_device(|device| device.flush_main_thread_cleanup_queue())
  }
}
