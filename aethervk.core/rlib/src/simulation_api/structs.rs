//! structs module.

use crate::{
  gpu::{self, DeviceAdditionalParams, PresentationEngineHandle},
  gpu_backends::new_render_frontend,
  scene::{
    AlmanacPlanet, BodyRotationalModel, EntityId, Scene, StaticMeshComponent, TransformComponent,
  },
  simulation::{self, almanac::AlmanacPackedData},
  simulation_api::{logic_thread::start_logic_thread, render_thread::start_render_thread},
  types::{EngineError, EngineResult, GpuError, GpuResult, RuntimeParams},
};
use aethervk_oshal_rlib::{
  self as oshal,
  math::{
    quaternion::Quaternion,
    vector::{Vector3, vec3::Vec3f32, vec4::Quat},
  },
  os::{
    self,
    pool::tasklet::TaskletHandle,
    thread::Thread,
    time::{timeus_milliseconds, timeus_t},
  },
};
use alloc::{
  collections::{BTreeMap, BTreeSet},
  string::{String, ToString},
  sync::Arc,
};
use core::{
  cell::RefCell,
  sync::atomic::{AtomicBool, AtomicU64},
};
use parking_lot::RwLock;
use thingbuf::mpsc;

/// target rate for physical simulation, in unscaled(read) monotonic time
pub const UNSCALED_FIXED_DELTA_US: timeus_t = timeus_milliseconds(16);

/// Osculating Keplerian orbital elements from the JPL Small-Body Database (SBDB).
///
/// Used by [`LogicCommand::TryInitComet`] and [`LogicCommand::BuildCometTrajectory`] to
/// generate an analytical orbit track (full ellipse or large hyperbola arc) independently
/// of the SPK file time coverage. The elements are in the **Heliocentric Ecliptic** frame
/// (J2000 ecliptic plane), which must be rotated by Earth's obliquity (~23.44°) before
/// storing as Heliocentric Equatorial (ICRF) control points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeplerianElements {
  /// Orbital eccentricity (e ≥ 0; <1 = ellipse, ≥1 = hyperbola).
  pub eccentricity: f64,
  /// Perihelion distance (AU).
  pub perihelion_distance_au: f64,
  /// Inclination to the J2000 ecliptic plane (degrees).
  pub inclination_deg: f64,
  /// Longitude of the ascending node Ω (degrees).
  pub longitude_of_ascending_node_deg: f64,
  /// Argument of perihelion ω (degrees).
  pub argument_of_perihelion_deg: f64,
}

pub mod particle_constants {
  pub const MEAN_INTRA_GRAINS_DISTANCE_MM: f32 = 1_f32;
  pub const MIN_CUMULATED_MASS_G: f32 = 0.001_f32; // from bvh_utils.glsl
}

/// Drop wrapper for a thread whose function uses a receiver, and the struct wraps
/// its transmitter. Invariant: Once this class is constructed through new, and its
/// safety conditions are satisfied, we can unwrap_unchecked both members
pub struct ThreadTxContainer<T> {
  tx: Option<mpsc::Sender<T>>,
  handle: Option<Thread>,
}

impl<T> ThreadTxContainer<T> {
  fn empty() -> Self {
    Self {
      tx: None,
      handle: None,
    }
  }

  pub fn tx(&self) -> &mpsc::Sender<T> {
    self.tx.as_ref().unwrap()
  }

  pub fn tx_opt(&self) -> Option<&mpsc::Sender<T>> {
    self.tx.as_ref()
  }
}

impl<T> Drop for ThreadTxContainer<T> {
  fn drop(&mut self) {
    oshal::log!("ThreadTxContainer drop started. Dropping tx...");
    // 1. Drop the Sender first.
    // This closes the channel. The Receiver in the background thread
    // will now yield `None` or an error, signaling the thread to exit.
    drop(self.tx.take());

    // 2. Take the Thread handle out of the Option and join it.
    // Because the channel is closed, the thread will finish its work
    // and this join will safely unblock.
    if let Some(thread) = self.handle.take() {
      oshal::log!("ThreadTxContainer joining thread...");
      thread.join();
      oshal::log!("ThreadTxContainer thread joined.");
    }
  }
}

// --------------------- Members of SimulationContext ---------------------------
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod, Hash)]
pub struct SceneEntityId {
  pub scene_id: u64,
  pub entity_id: u64,
}

impl SceneEntityId {
  pub fn new(scene_id: u64, entity_id: EntityId) -> Self {
    Self {
      scene_id,
      entity_id: entity_id.as_ffi(),
    }
  }
}

/// The three entity IDs that make up a planet or comet subtree.
/// Stored in [`SceneContext`] so the logic thread can find them without name-based queries.
#[derive(Debug, Clone, Copy)]
pub struct SubtreeEntities {
  /// The Micro frame entity (AU scale, parent = root_entity). Repositioned on frame shifts.
  pub subtree: EntityId,
  /// The body entity (km scale, parent = subtree). Holds AlmanacPlanet when driven.
  pub body: EntityId,
  /// The orbit trajectory entity (AU scale, parent = root_entity). Holds TrajectoryComponent.
  /// NOTE: orbit must be a child of root (depth_layer=0) so its AU control points are rendered
  /// in the heliocentric frame. Placing it in the Micro frame (depth_layer=1) would cause the
  /// renderer's RTE path to scale AU control points by AU_TO_KM, breaking the trajectory.
  pub orbit: EntityId,
}

#[derive(Debug, Clone)]
pub struct CartesianStateComet {
  pub transform: TransformComponent,
  pub almanac_planet: AlmanacPlanet,
  pub body_rotational_model: Option<BodyRotationalModel>,
}

#[derive(Debug, Clone)]
pub struct CartesianState {
  pub comet_state: Option<CartesianStateComet>,

  pub parent_frame: EntityId,
  pub parent_frame_transform: TransformComponent,
}

impl CartesianState {
  pub fn new_comet(
    transform: TransformComponent,
    almanac_planet: AlmanacPlanet,
    body_rotational_model: Option<BodyRotationalModel>,
    parent_frame: EntityId,
    parent_frame_transform: TransformComponent,
  ) -> Self {
    Self {
      comet_state: Some(CartesianStateComet {
        transform,
        almanac_planet,
        body_rotational_model,
      }),
      parent_frame,
      parent_frame_transform,
    }
  }

  pub fn new_frame(parent_frame: EntityId, parent_frame_transform: TransformComponent) -> Self {
    Self {
      comet_state: None,
      parent_frame,
      parent_frame_transform,
    }
  }

  pub fn frame_data(
    cartesian_state_cache: &dashmap::DashMap<SceneEntityId, CartesianState>,
    scene_id: u64,
    entity_id: EntityId,
  ) -> Option<TransformComponent> {
    cartesian_state_cache
      .get(&SceneEntityId {
        scene_id,
        entity_id: entity_id.as_ffi(),
      })
      .map(|state| state.parent_frame_transform)
  }
}

/// owned by logic thread context
pub struct SimulationSceneData {
  /// Scene state: Scene map
  pub scenes: BTreeMap<u64, Arc<RwLock<SceneContext>>>,
  /// Centralized time system. One for each scene which needs it (so inserted if absent at
  /// first simulation play, and removed when simulation stops)
  pub time_managers: dashmap::DashMap<u64, oshal::os::time::v2::TimeManager>,
  /// Scene state: next available id. Steadily incremented
  next_scene_id: u64,
  /// mesh cache shared among all scenes
  pub(crate) mesh_cache: Arc<crate::scene::AssetCache<simulation::comet::Comet>>,
  /// Loaded GLTF Models. Necessary with the asset cache because when a model is evicted,
  /// the string used as key in the cache is eliminated
  pub model_registry: BTreeMap<u64, String>,
  /// model_registry next available id. Steadily incremented
  next_model_id: u64,
  /// Cache used by the simulation to store computed next positions in `fixed_update` phase
  /// before the next cross sync window
  pub cartesian_state_cache: dashmap::DashMap<SceneEntityId, CartesianState>,
}

impl Default for SimulationSceneData {
  fn default() -> Self {
    Self::new()
  }
}

impl SimulationSceneData {
  /// # Safety
  /// - `ptr` should be a pointer to a piece of memory with size and alignment compatible for [`SimulationSceneData`]
  pub unsafe fn new_inplace(ptr: *mut Self) {
    unsafe { ptr.write(Self::new()) }
  }

  pub fn next_scene_id(&self) -> u64 {
    self.next_scene_id
  }

  pub fn new() -> Self {
    Self {
      time_managers: dashmap::DashMap::with_capacity(16),
      scenes: BTreeMap::new(),
      next_scene_id: 1,
      mesh_cache: Arc::new(crate::scene::AssetCache::new()),
      model_registry: Default::default(),
      next_model_id: 1,
      cartesian_state_cache: dashmap::DashMap::with_capacity(16),
    }
  }

  pub fn get_scene(&self, scene_id: u64) -> Option<Arc<RwLock<SceneContext>>> {
    self.scenes.get(&scene_id).cloned()
  }

  /// Insert a new scene in `scenes` and properly increment next free id counter.
  /// return current id
  pub fn insert_scene(&mut self, scene_ctx: Arc<RwLock<SceneContext>>) -> u64 {
    debug_assert!(Arc::strong_count(&scene_ctx) == 1 && Arc::weak_count(&scene_ctx) == 0);
    let new_id = self.next_scene_id;
    debug_assert!(new_id != 0);
    let _ = self.scenes.insert(new_id, scene_ctx);
    self.next_scene_id += 1;
    new_id
  }
}

impl core::ops::Deref for SimulationSceneData {
  type Target = BTreeMap<u64, Arc<RwLock<SceneContext>>>;

  fn deref(&self) -> &Self::Target {
    &self.scenes
  }
}

impl core::ops::DerefMut for SimulationSceneData {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.scenes
  }
}

/// container for threads in a simulation context. It's not necessary to develop a stop threads
/// function, cause the drop function for this struct should take care of it. We don't care
/// about the drop order of the two. The only thing we care about is to send the Shutdown command
pub struct SimulationThreads {
  /// Render thread handle (has task thread pool arc handle)
  pub render_thread: ThreadTxContainer<RenderCommand>,
  /// Render thread feedback receiver (note, follows `render_thread`, therefore dropped after it)
  pub render_feedback_rx: Option<mpsc::Receiver<RenderFeedback>>,
  /// Logic thread handle (has task thread pool arc handle)
  pub logic_thread: ThreadTxContainer<LogicCommand>,
  /// Logic thread feedback receiver (note, follows `logic_thread`, therefore dropped after it)
  pub logic_feedback_rx: Option<mpsc::Receiver<LogicFeedback>>,
  /// Task pool handle (duplicated here so that it outlives threads). Should not be accessed
  /// by FFI caller threads
  pool: Arc<os::pool::ThreadPool>,
  /// Audio device thread handle
  pub audio_device: Option<alloc::boxed::Box<dyn oshal::os::audio::AudioDevice + Send + Sync>>,
  /// Shared flag that is set to `true` early in `drop` to signal the render thread that it
  /// must not call `vkQueuePresentKHR` for any new frames.  On Linux/X11 with NVIDIA, the
  /// present call can block indefinitely waiting for a DRI3 completion event that nobody will
  /// deliver once Avalonia's event loop has stopped — which is exactly the state we are in
  /// when `ShutdownSync` is invoked from the UI thread.
  ///
  /// The Arc is cloned into [`RenderThreadContext`] so the render thread can observe it
  /// without any lifetime coupling to `SimulationThreads`.
  pub skip_present: Arc<core::sync::atomic::AtomicBool>,
}

impl Drop for SimulationThreads {
  fn drop(&mut self) {
    oshal::log!("SimulationThreads drop started");

    // ── Shutdown watchdog ──────────────────────────────────────────────────────
    // Spawn a thread that forcibly terminates the process after 5 seconds.
    // This guards against `pthread_join` / `WaitForSingleObject` hanging forever
    // (e.g. due to a deadlock in the render or logic thread during shutdown).
    // If the normal shutdown sequence completes first, the process exits cleanly
    // and this thread is killed with it. The handle is intentionally not joined.
    //
    // We use `_exit` (Unix) / `TerminateProcess` (Windows) rather than `exit`
    // to bypass atexit handlers and C++ destructors, which could themselves block
    // if the process is already deadlocked.
    const SHUTDOWN_WATCHDOG_S: u64 = 8;
    let _watchdog = oshal::os::thread::Builder::new()
      .name(alloc::format!("shutdown_watchdog"))
      .spawn(move || {
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_secs(
          SHUTDOWN_WATCHDOG_S,
        ));
        oshal::log!(
          "[shutdown] watchdog fired after {} s — forcing process exit.",
          SHUTDOWN_WATCHDOG_S
        );

        #[cfg(debug_assertions)]
        {
          #[cfg(any(unix, target_os = "macos"))]
          {
            oshal::log!("Attempting to print backtraces for all threads...");
            let pid = unsafe { libc::getpid() };
            let debugger = if cfg!(target_os = "macos") {
              "lldb"
            } else {
              "gdb"
            };
            let cmd = if cfg!(target_os = "macos") {
              alloc::format!(
                "{} -p {} -o 'thread backtrace all' -o 'quit'",
                debugger,
                pid
              )
            } else {
              // full is too much output
              alloc::format!(
                "{} -p {} -ex 'thread apply all bt' -ex 'quit' --batch",
                debugger,
                pid
              )
            };

            if let Ok(c_cmd) = alloc::ffi::CString::new(cmd) {
              unsafe {
                libc::system(c_cmd.as_ptr());
              }
            }
          }
          #[cfg(windows)]
          {
            oshal::log!("Watchdog backtrace (Windows does not support native all-thread trace):");
            oshal::os::debug::print_stacktrace();
          }
        }

        // SAFETY: called only on a terminal shutdown path after all normal
        // shutdown attempts have timed out.
        #[cfg(any(unix, target_os = "macos"))]
        unsafe {
          libc::_exit(1);
        }
        #[cfg(windows)]
        unsafe {
          use windows::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
          let _ = TerminateProcess(GetCurrentProcess(), 1);
        }
      });
    // Intentionally drop without joining — the thread runs detached.
    // Either the process exits cleanly (watchdog killed with it) or the watchdog
    // fires and terminates us after SHUTDOWN_WATCHDOG_S seconds.
    drop(_watchdog);

    // ── Phase 1: Signal "no more presents" ────────────────────────────────────
    // Set the flag BEFORE closing channels so the render thread observes it
    // between the end of its current command and the channel-closed check.
    //
    // This is a best-effort guard for the common case: if the render thread is
    // *between frames* when shutdown starts, it will see the flag and skip the
    // next `vkQueuePresentKHR` call entirely.  If `vkQueuePresentKHR` is already
    // in-flight, the timed join below (Phase 4) is the safety net.
    //
    // Background: on Linux/X11 + NVIDIA, `vkQueuePresentKHR` blocks waiting for a
    // DRI3 present-completion event from the X server.  That event is delivered
    // through the Xlib connection that Avalonia's event loop reads.  But by the
    // time `ShutdownSync` is called from the UI thread, Avalonia's event loop has
    // already stopped — so nobody reads the X socket, the call never returns, and
    // a plain `pthread_join` on the main thread deadlocks.
    use core::sync::atomic::Ordering;
    self.skip_present.store(true, Ordering::Release);
    oshal::log!("SimulationThreads: skip_present flag set");

    // ── Phase 2: Close logic channel ──────────────────────────────────────────
    // Drop the sender — this closes the channel from the sender side.
    // The logic thread's try_recv loop returns on TryRecvError::Closed (line ~326
    // of logic_thread.rs), so closing the channel is the reliable shutdown signal.
    // Using try_send(Shutdown) was unreliable: if the 128-slot channel is full
    // (e.g. many commands queued during heavy use), try_send silently fails and
    // handle.join() hangs forever.
    self.logic_thread.tx.take();
    self.logic_feedback_rx = None;

    // ── Phase 3: Join logic thread (safe — no X11 involvement) ────────────────
    // The logic thread does not call Vulkan WSI / X11 present, so joining it
    // directly from the main thread is always safe.
    if let Some(handle) = self.logic_thread.handle.take() {
      oshal::log!("SimulationThreads: joining logic thread...");
      handle.join();
      oshal::log!("SimulationThreads: logic thread joined.");
    }

    // ── Phase 4: Close render channel ─────────────────────────────────────────
    // MUST drop render thread BEFORE gathering the pool!
    // Dropping the render thread drops the `Device`, which sets `callback_stop_signal`
    // to true. This stops `TimelinePollingWorkload` which is running on the pool.
    self.render_thread.tx.take();
    self.render_feedback_rx = None;

    // ── Phase 5: Timed join of render thread ──────────────────────────────────
    // Do NOT join the render thread directly from the main thread.  If
    // `vkQueuePresentKHR` is still in-flight (i.e. the skip_present flag arrived
    // too late), that call can block indefinitely on X11/NVIDIA, deadlocking the
    // main thread in `pthread_join`.
    //
    // Instead: spawn a tiny helper thread that calls `thread.join()`, and wait for
    // it via a `parking_lot::Condvar` with a 2-second timeout.  The join itself
    // still happens, preserving RAII "we know exactly when threads exit" semantics;
    // it just happens off the main thread.  If the 2-second timeout fires, the
    // watchdog at SHUTDOWN_WATCHDOG_S will terminate the process regardless.
    if let Some(render_handle) = self.render_thread.handle.take() {
      oshal::log!("SimulationThreads: timed-join of render thread (2 s timeout)...");

      let pair = Arc::new((
        parking_lot::Mutex::<bool>::new(false), // joined?
        parking_lot::Condvar::new(),
      ));
      let pair_for_helper = Arc::clone(&pair);

      let join_thread = oshal::os::thread::Builder::new()
        .name(alloc::format!("render_join"))
        .spawn(move || {
          render_handle.join();
          let (lock, cvar) = &*pair_for_helper;
          *lock.lock() = true;
          cvar.notify_one();
          oshal::log!("SimulationThreads: render thread joined (from helper thread).");
        });
      // Detach the join thread — we synchronise via condvar, not by joining it.
      drop(join_thread);

      const RENDER_JOIN_TIMEOUT_MS: u64 = 6_000;
      let (lock, cvar) = &*pair;
      let mut done = lock.lock();
      if !*done {
        let timed_out = cvar
          .wait_for(&mut done, core::time::Duration::from_millis(RENDER_JOIN_TIMEOUT_MS))
          .timed_out();
        if timed_out {
          oshal::log!(
            "SimulationThreads: render thread join timed out after {} ms — \
             vkQueuePresentKHR may be deadlocked on X11/NVIDIA. \
             Watchdog will terminate the process in {} s.",
            RENDER_JOIN_TIMEOUT_MS,
            SHUTDOWN_WATCHDOG_S,
          );
        }
      }
    }

    // ── Phase 6: Pool gather + audio ──────────────────────────────────────────
    // Ensure all logic-launched tasklets are finished before shutting down the renderer.
    oshal::log!("SimulationThreads waiting for thread pool tasks to complete...");
    self.pool.gather();

    // Stop the audio device thread
    if let Some(mut audio) = self.audio_device.take() {
      audio.stop();
    }

    oshal::log!("SimulationThreads drop finished");
  }
}

pub struct LogicWorkload {
  pub cmd: LogicCommand,
  pub ctx: alloc::sync::Arc<LogicThreadContext>,
}

impl LogicThreadContext {
  /// Loads an almanac file into the shared logic state.
  ///
  /// Returns the discovered primary NAIF body ID (0 if unknown, multi-body, or already loaded).
  /// Emitting the `ExternalState::AlmanacImported` callback is the caller's responsibility.
  pub fn load_almanac_file_internal(&self, path: &str) -> EngineResult<i32> {
    let mut logic = self.logic_state.write();
    if logic.almanac_data.file_names.iter().any(|f| f == path) {
      return Ok(0); // already loaded; NAIF ID not re-discovered
    }

    let path_buf = oshal::os::fs::PathBuf::from(path);
    logic.almanac_data.load_almanac(&path_buf)?;

    // Discover the primary NAIF body ID from the freshly-loaded SPK data.
    // Exclude SSB (0) and well-known solar-system bodies (1–999) to isolate
    // comet/asteroid IDs, which JPL Horizons places at 1 000 000+ or negative.
    let naif_id = logic
      .almanac_data
      .almanac
      .spk_domains()
      .ok()
      .and_then(|domains| {
        let targets: alloc::vec::Vec<i32> = domains
          .into_iter()
          .map(|(id, _)| id)
          .filter(|&id| id != 0 && !(1..=999).contains(&id))
          .collect();
        if targets.len() == 1 {
          Some(targets[0])
        } else {
          None
        }
      })
      .unwrap_or(0);

    Ok(naif_id)
  }

  pub fn unload_almanac_file_internal(&self, path: &str) -> EngineResult<()> {
    let mut logic = self.logic_state.write();
    logic.almanac_data.unload_almanac_spk(path)
  }
}

impl SimulationSceneData {
  pub fn import_model_from_mesh(
    &mut self,
    path: &str,
    mesh: crate::simulation::comet::Comet,
  ) -> u64 {
    let model_id = self.next_model_id;
    self.next_model_id += 1;
    self.mesh_cache.insert(path.to_string(), mesh);
    self.model_registry.insert(model_id, path.to_string());
    model_id
  }
}

impl SimulationThreads {
  /// whether the render thread is currently spawned and its channel is active
  pub fn render_thread_running(&self) -> bool {
    let result = self.render_thread.handle.is_some();
    debug_assert!(
      !result || (self.render_thread.tx.is_some() && self.render_feedback_rx.is_some())
    );
    result
  }

  /// TODO: Document this item
  pub fn logic_thread_running(&self) -> bool {
    let result = self.logic_thread.handle.is_some();
    debug_assert!(!result || (self.logic_thread.tx.is_some() && self.logic_feedback_rx.is_some()));
    result
  }

  /// TODO: Document this item
  pub fn new_running(
    render_thread_params: RenderThreadParams,
    logic_thread_params: LogicThreadParams,
  ) -> EngineResult<Self> {
    oshal::os::debug::fpe::setup_fpu_panic();
    // Extract skip_present from the render params before moving them into
    // start_render_thread, so we can store our own Arc clone in `self`.
    let skip_present = Arc::clone(&render_thread_params.skip_present);
    let mut this = Self::new_idle_with_skip_present(skip_present)?;
    this.start_render_thread(render_thread_params)?;
    this.start_logic_thread(logic_thread_params)?;
    Ok(this)
  }

  /// Creates the thread pool only, with a fresh `skip_present` flag.
  pub fn new_idle() -> EngineResult<Self> {
    Self::new_idle_with_skip_present(Arc::new(core::sync::atomic::AtomicBool::new(false)))
  }

  /// Internal helper: creates the thread pool and wires up a caller-supplied `skip_present` Arc.
  fn new_idle_with_skip_present(
    skip_present: Arc<core::sync::atomic::AtomicBool>,
  ) -> EngineResult<Self> {
    let thread_pool = Arc::new(os::pool::ThreadPool::new(4).map_err(|e| {
      oshal::log!("Failed to create thread pool: {:?}", e);
      EngineError::InvalidOperation("core_api:startup | failed to create thread pool")
    })?);

    Ok(Self {
      pool: thread_pool,
      render_thread: ThreadTxContainer::empty(),
      render_feedback_rx: None,
      logic_thread: ThreadTxContainer::empty(),
      logic_feedback_rx: None,
      audio_device: None,
      skip_present,
    })
  }

  /// TODO: Document this item
  pub fn start_render_thread(&mut self, params: RenderThreadParams) -> EngineResult<()> {
    if self.render_thread_running() {
      return Err(EngineError::InvalidOperation(
        "SimulationThreads::start_render_thread | render thread already running",
      ));
    }
    let (render_tx, render_rx) = mpsc::channel(params.channel_capacity);
    let (render_feedback_tx, render_feedback_rx) = mpsc::channel(params.channel_capacity);
    let render_thread_handle =
      start_render_thread(render_rx, params.to_context(render_feedback_tx))?;
    self.render_thread.tx = Some(render_tx);
    self.render_thread.handle = Some(render_thread_handle);
    self.render_feedback_rx = Some(render_feedback_rx);
    Ok(())
  }

  /// TODO: Document this item
  pub fn start_logic_thread(&mut self, params: LogicThreadParams) -> EngineResult<()> {
    if self.logic_thread_running() || !self.render_thread_running() {
      return Err(EngineError::InvalidOperation(
        "SimulationThreads::start_logic_thread | logic thread already running | should start after render thread",
      ));
    }

    let (logic_tx, logic_rx) = mpsc::channel(params.channel_capacity);
    let (logic_feedback_tx, logic_feedback_rx) = mpsc::channel(params.channel_capacity);
    let logic_thread_handle = start_logic_thread(
      logic_rx,
      alloc::sync::Arc::new(params.to_context(
        logic_feedback_tx,
        self.render_thread.tx.as_ref().unwrap().clone(),
      )),
    )?;
    self.logic_thread.tx = Some(logic_tx);
    self.logic_thread.handle = Some(logic_thread_handle);
    self.logic_feedback_rx = Some(logic_feedback_rx);
    Ok(())
  }

  /// TODO: Document this item
  pub fn stop_render_thread(&mut self) -> EngineResult<()> {
    if !self.render_thread_running() {
      return Err(EngineError::InvalidOperation(
        "SimulationThreads::stop_render_thread | render thread not running",
      ));
    }
    let render_tx = self.render_thread.tx();
    render_tx.try_send(RenderCommand::default()).map_err(|e| {
      oshal::log!("Failed to send stop command to render thread: {:?}", e);
      EngineError::InvalidOperation(
        "SimulationThreads::stop_render_thread failed to send shutdown message",
      )
    })?;
    // drop function will join
    self.render_thread = ThreadTxContainer::empty();
    Ok(())
  }

  /// TODO: Document this item
  pub fn stop_logic_thread(&mut self) -> EngineResult<()> {
    if !self.logic_thread_running() {
      return Err(EngineError::InvalidOperation(
        "SimulationThreads::stop_logic_thread | logic thread not running",
      ));
    }

    let logic_tx = self.logic_thread.tx();
    logic_tx.try_send(LogicCommand::default()).map_err(|e| {
      oshal::log!("Failed to send stop command to logic thread: {:?}", e);
      EngineError::InvalidOperation(
        "SimulationThreads::stop_logic_thread failed to send shutdown message",
      )
    })?;
    // drop function will join
    self.logic_thread = ThreadTxContainer::empty();
    self.logic_feedback_rx = None;
    Ok(())
  }
}

// --------------------- Internal Types: Logic Thread ---------------------------

// TODO remove if not useful after time rework
#[derive(Clone, Default)]
pub enum LogicFeedback {
  #[default]
  Empty,
}

#[derive(Default, Clone, Debug)]
pub enum LogicCommand {
  #[default]
  Shutdown,

  RotateCamera {
    camera_entity: EntityId,
    scene: Arc<RwLock<SceneContext>>,
    delta_x: f32,
    delta_y: f32,
  },
  ZoomCamera {
    camera_entity: EntityId,
    scene: Arc<RwLock<SceneContext>>,
    amount: f32,
  },
  ResetCamera {
    camera_entity: EntityId,
    scene: Arc<RwLock<SceneContext>>,
  },
  PanCamera {
    camera_entity: EntityId,
    scene: Arc<RwLock<SceneContext>>,
    delta_x: f32,
    delta_y: f32,
  },

  MoveCursor {
    scene: Arc<RwLock<SceneContext>>,
    delta_x: f32,
    delta_y: f32,
    delta_z: f32,
  },

  SnapToEntity {
    snap_entity: EntityId,
    target_entity: EntityId,
    scene: Arc<RwLock<SceneContext>>,
  },
  StartSimlulation {
    scene_id: u64,
    speed: oshal::os::time::v2::SimSpeed,
  },
  PlaySceneToEnd {
    scene_id: u64,
    speed: oshal::os::time::v2::SimSpeed,
  },
  PauseScene {
    scene_id: u64,
  },
  PlayScene {
    scene_id: u64,
    speed: oshal::os::time::v2::SimSpeed,
  },
  SnapshotScene {
    scene_id: u64,
  },

  ImportModel {
    task_id: u64,
    path: String,
  },
  LoadAlmanac {
    task_id: u64,
    path: String,
  },
  UnloadAlmanac {
    task_id: u64,
    path: String,
  },
  UpdateTrajectoryForSpk {
    task_id: u64,
    scene_id: u64,
    entity_id: u64,
    start_epoch_tai_sec: f64,
    end_epoch_tai_sec: f64,
    sample_step_days: f64,
    spk_id: i32,
  },
  RestoreSnapshot {
    scene_id: u64,
  },
  /// Set the visibility (hidden/visible) of an entity and all its descendants.
  /// Dispatched asynchronously to the logic thread to avoid spin-wait deadlocks.
  SetEntityVisibility {
    scene_id: u64,
    entity: u64,
    visible: bool,
  },

  SetEpochRange {
    scene_id: u64,
    start: hifitime::Epoch,
    end: hifitime::Epoch,
  },
  /// Phase 1 of Two-Phase Commit: Validates proposed timeline without mutating ECS.
  TryInitComet {
    scene_id: u64,
    spk_id: i32,
    proposed_start: hifitime::Epoch,
    proposed_end: hifitime::Epoch,
    /// Osculating Keplerian elements from SBDB, used to generate the analytical orbit track.
    keplerian_elements: KeplerianElements,
  },
  /// Phase 1 Async Math: Background thread generates trajectory points.
  BuildCometTrajectory {
    scene_id: u64,
    spk_id: i32,
    start_epoch_tai_sec: f64,
    end_epoch_tai_sec: f64,
    sample_step_days: f64,
    /// Osculating Keplerian elements from SBDB, forwarded from TryInitComet.
    keplerian_elements: KeplerianElements,
  },
  /// Internal command dispatched by the UnloadAlmanac handler for comet SPK cleanup.
  /// Removes AlmanacPlanet and TrajectoryComponent, resets comet to 1 AU +X default.
  CleanupComet {
    scene_id: u64,
  },

  /// Animate the camera's `HighResTransformComponent` to a target position/rotation.
  /// If a `TransformAnimationComponent` is already active on the camera entity, it is
  /// **retargeted in-flight** via `retarget()` — preserving speed, avoiding any snap.
  ///
  /// Submitted by the FFI caller thread from `avkSimulationContext_addCameraAnimation`.
  AnimateCameraTo {
    scene_id: u64,
    camera_id: u64, // external (FFI) entity id
    target_pos: aethervk_oshal_rlib::math::vector::vec3f64::DVec3,
    target_rot: aethervk_oshal_rlib::math::vector::vec4::Quat,
    duration_s: f32,
  },

  /// Apply an immediate (non-animated) camera transform and/or projection update.
  ///
  /// Submitted by `avkSimulationContext_transformStaticCamera`. Writes
  /// [`HighResTransformComponent`](crate::scene::HighResTransformComponent) directly via
  /// `set_global_transform_f64` (no `TransformAnimationComponent` is created or retargeted).
  SetCameraTransform {
    scene_id: u64,
    camera_id: u64, // external (FFI) entity id
    /// `Some((pos, rot))` → write world-space position + rotation via `set_global_transform_f64`.
    transform: Option<(
      aethervk_oshal_rlib::math::vector::vec3f64::DVec3,
      aethervk_oshal_rlib::math::vector::vec4::Quat,
    )>,
    /// `Some(proj)` → overwrite `CameraComponent::projection`.
    projection: Option<crate::scene::CameraProjection>,
  },
}

impl LogicCommand {
  const PARSING_ERROR: &str = "LogicCommand::new | parsing error";
}

pub struct LogicState {
  pub almanac_data: AlmanacPackedData,
  /// Optional callback to request SPK data download from the host application.
  /// Called when epoch range validation finds that almanac coverage is insufficient.
  /// Parameters: (spk_id, start_epoch_str, end_epoch_str) → returns file path of downloaded SPK, or null.
  pub almanac_invalidation_callback: Option<
    extern "C" fn(
      i32,
      *const core::ffi::c_char,
      *const core::ffi::c_char,
    ) -> *const core::ffi::c_char,
  >,
}

impl Default for LogicState {
  fn default() -> Self {
    Self {
      almanac_data: AlmanacPackedData::default(),
      almanac_invalidation_callback: None,
    }
  }
}

// --------------------- Internal Types: Render Thread ---------------------------

#[derive(Clone, Default)]
pub enum RenderTaskStatus {
  #[default]
  Completed,
  Pending,
  Error(GpuError),
}

impl AsRef<RenderTaskStatus> for RenderTaskStatus {
  fn as_ref(&self) -> &RenderTaskStatus {
    self
  }
}

#[derive(Clone, Default)]
pub enum RenderFeedback {
  #[default]
  Empty,

  TaskCreated(Option<u64>),
  TaskQueryStatus(RenderTaskStatus),
}

#[derive(Clone, Copy, Debug)]
pub struct CustomRenderCallback {
  pub after_render_frame_fn: fn(
    &dyn crate::gpu::RenderDevice,
    crate::gpu::CommandBufferHandle,
    gpu::PresentationEngineHandle,
    &crate::gpu::RenderScene,
    *mut core::ffi::c_void,
  ) -> GpuResult<()>,
  pub on_first_render_fn: fn(
    &dyn crate::gpu::RenderDevice,
    crate::gpu::CommandBufferHandle,
    gpu::PresentationEngineHandle,
    &crate::gpu::RenderScene,
    *mut core::ffi::c_void,
  ) -> GpuResult<()>,
  pub user_data: SendPtrMut<core::ffi::c_void>,
}

/// Invariant: presentation engine is not null and exists inside simulation context and render device
#[derive(Clone, Debug)]
pub struct RenderFrame {
  pub presentation_engine_handle: PresentationEngineHandle,
  pub task_id: Arc<AtomicU64>,
  pub scene: Arc<RwLock<SceneContext>>,
  pub render_physical_meshes_outline: bool,
  pub camera_entity: EntityId,
  pub clear_color: [f32; 4],
  pub sun_entity: Option<EntityId>,
  pub sky_entity: Option<EntityId>,
  pub cursor_entity: Option<EntityId>,
  pub custom_render_callback: Option<CustomRenderCallback>,

  /// Step 4 of the cross sync procedure: if the compute workload which is invoking this command
  /// generated a cross sync procedure, we need to wait on a particular compute timeline value to
  /// finish, and then insert graphics queue acquire commands. Whether to insert commands or not is
  /// signaled by cross queue AND presence of this sync value
  /// This is the compute timeline value to wait for
  ///
  /// Note: We are packing only the release value and not the `vk::Semaphore` itself cause we know
  /// we are implicitly referring to the compute queue global timeline semaphore
  /// `vulkan_device.kernels.timeline`. It can implicitly be taken by
  /// `vulkan_device.kernels.next_submit_value`, but that would create a race condition between the
  /// render thread and the physics tasklet threads
  pub particle_acquire_sync: Option<u64>,

  /// particles constants to reproduce cluster params coherent with what the logic thread computes
  pub mean_intra_grains_distance_mm: f32,
  /// particles constants to reproduce cluster params coherent with what the logic thread computes
  pub min_cumulated_mass_g: f32,
}

/// Invariant: width and height are valid, presentation engine is inside simulation context and render device
#[derive(Clone, Debug)]
pub struct Resize {
  pub presentation_engine_handle: PresentationEngineHandle,
  pub width: u32,
  pub height: u32,
}

/// Struct pointed to in the [`RenderCommand::SyncParticleRelease`], whose writes are protected by
/// a memory barrier issued through an atomic load/store on `feedback`
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncParticleReleaseFeedback {
  pub timeline_semaphore: ash::vk::Semaphore,
  pub timeline_release_value: u64,
}

unsafe impl bytemuck::Zeroable for SyncParticleReleaseFeedback {}
unsafe impl bytemuck::Pod for SyncParticleReleaseFeedback {}

#[derive(Clone, Default)]
pub enum RenderCommand {
  #[default]
  Shutdown,
  RenderFrames(alloc::vec::Vec<RenderFrame>),
  Resize(Resize),
  GenerateSky,
  /// Step 1 of Cross Sync 4 steps procedure to hand over compute owned updates to the render
  /// thread. The render thread will write its generated task_id into `feedback`
  SyncParticleRelease {
    // Note: it could also be a raw pointer instead of a shared one, cause we know that the caller
    // will outlive this command as it will poll the content of this atomic
    feedback: alloc::sync::Arc<core::sync::atomic::AtomicU64>,
    feedback_ptr: SendPtrMut<SyncParticleReleaseFeedback>,
  },
  /// Instructs the render thread to bracket the next frame for this PE with
  /// RenderDoc `StartFrameCapture` / `EndFrameCapture`, capturing only that
  /// windowed swapchain.  Emitted only in debug builds.
  #[cfg(debug_assertions)]
  CaptureNextFrame {
    pe_handle: crate::gpu::PresentationEngineHandle,
  },
}

unsafe impl Send for RenderCommand {}

// --------------------- Internal Types ---------------------------

#[derive(Debug)]
/// TODO: Document this item
pub struct SendPtr<T: ?Sized>(pub *const T);
unsafe impl<T: ?Sized> Send for SendPtr<T> {}
unsafe impl<T: ?Sized> Sync for SendPtr<T> {}

impl<T: ?Sized> Clone for SendPtr<T> {
  fn clone(&self) -> Self {
    *self // This works because the underlying *const T is Copy
  }
}

impl<T: ?Sized> Copy for SendPtr<T> {}

/// After rust edition 2021, the compiler captures in closures only used fields, not the entire
/// struct, therefore, you cannot access with .0 the inner pointer otherwise you bypass Sync + Send
impl<T: ?Sized> SendPtr<T> {
  #[inline(always)]
  /// TODO: Document this item
  pub fn get(self) -> *const T {
    self.0
  }
}

#[derive(Debug)]
/// TODO: Document this item
pub struct SendPtrMut<T: ?Sized>(pub *mut T);
unsafe impl<T: ?Sized> Send for SendPtrMut<T> {}
unsafe impl<T: ?Sized> Sync for SendPtrMut<T> {}

impl<T: ?Sized> Clone for SendPtrMut<T> {
  fn clone(&self) -> Self {
    *self // This works because the underlying *mut T is Copy
  }
}

impl<T: ?Sized> Copy for SendPtrMut<T> {}

/// After rust edition 2021, the compiler captures in closures only used fields, not the entire
/// struct, therefore, you cannot access with .0 the inner pointer otherwise you bypass Sync + Send
impl<T: ?Sized> SendPtrMut<T> {
  #[inline(always)]
  /// TODO: Document this item
  pub fn get(self) -> *mut T {
    self.0
  }
}

#[cfg(test)]
lazy_static::lazy_static! {
  pub static ref SHADER_MOCK_RESULTS: std::sync::Mutex<std::collections::HashMap<u64, alloc::vec::Vec<u8>>> = std::sync::Mutex::new(std::collections::HashMap::new());
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MockTargetShader {
  EmitParticles,
  LbvhPrepass,
  LbvhBuild,
  MotionBounds,
  MotionRefit,
  Ccd,
  CcdRigidbody,
  StreamCompact,
  ReduceToi,
  LcpSolver,
  ApplyImpulses,
  BarnesHut,
  BroadPhase,
  IntegrateParticlesP1P2,
  IntegrateBodiesP3,
  IntegrateParticlesP4P5,
  RbForceAssign,
  BpClear,
  BpBoundsGen,
  BpScene,
  BpClassify,
  BpCrossLca,
  BpParticleSelf,
  ApplyEmitters,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhysicsEngineType {
  #[default]
  VulkanCompute,
  #[cfg(test)]
  Mock(MockTargetShader),
}

#[derive(Clone, Debug)]
pub struct PresentationEngineData {
  pub is_windowless: bool,
  pub camera_entity: Option<EntityId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicsDeviceSelfSync {
  /// compute timeline semaphore
  pub timeline_handle: ash::vk::Semaphore,
  /// target compute timeline value
  pub timeline_value: u64,
  /// latest query time of the timeline semaphore (zero init)
  pub latest_query_us: oshal::os::time::timeus_t,
  /// recomputed backoff interval for each query (doubled each failed query)
  pub query_exp_backoff_us: oshal::os::time::timeus_t,
}

impl PhysicsDeviceSelfSync {
  pub fn new(timeline_handle: ash::vk::Semaphore, timeline_value: u64) -> Self {
    use oshal::os::time::timeus_t;
    const SIMULATION_DISPATCH_CHECK_INTERVAL_US: timeus_t = 8;
    Self {
      timeline_handle,
      timeline_value,
      latest_query_us: 0,
      query_exp_backoff_us: SIMULATION_DISPATCH_CHECK_INTERVAL_US,
    }
  }

  // return `true` if GPU side completed, `false` if time not elapsed or GPU side not completed
  pub fn try_wait(
    &mut self,
    vulkan_device: &crate::gpu_backends::vulkan::device::LogicalDevice,
    now_unscaled_us: oshal::os::time::timeus_t,
    delta_unscaled_us: oshal::os::time::timeus_t,
  ) -> bool {
    if delta_unscaled_us < self.query_exp_backoff_us {
      return false;
    }

    // Count every poll in debug builds so debug_perf can report rate + backtraces.
    #[cfg(all(debug_assertions, target_os = "linux"))]
    crate::simulation_api::debug_perf::traced_semaphore_poll();

    if let Ok(value) = unsafe {
      vulkan_device
        .timeline_semaphore
        .get_semaphore_counter_value(self.timeline_handle)
    } {
      self.latest_query_us = now_unscaled_us;
      debug_assert!(self.timeline_value <= value);
      if self.timeline_value == value {
        true
      } else {
        // Cap at 8ms to prevent exponential freeze on slow GPU dispatches.
        // Without this cap, after ~20 missed polls the backoff exceeds 8 seconds.
        const MAX_BACKOFF_US: oshal::os::time::timeus_t = 8_000;
        self.query_exp_backoff_us = (self.query_exp_backoff_us << 1).min(MAX_BACKOFF_US);
        false
      }
    } else {
      false
    }
  }

  /// Blocking wait: parks the thread until the GPU signals the timeline value.
  ///
  /// Uses `vkWaitSemaphoresKHR` — the correct API for waiting on GPU completion.
  /// The calling thread consumes **zero CPU** while parked (kernel-level wait,
  /// signalled via GPU interrupt).
  ///
  /// `timeout_ns`: hard deadline in nanoseconds. Returns `true` if the GPU
  /// signalled within the deadline, `false` if the deadline expired.
  pub fn blocking_wait(
    &self,
    vulkan_device: &crate::gpu_backends::vulkan::device::LogicalDevice,
    timeout_ns: u64,
  ) -> bool {
    vulkan_device
      .wait_for_semaphore_value(self.timeline_handle, self.timeline_value, timeout_ns)
      .is_ok()
  }
}

#[derive(Debug)]
pub struct SceneContext {
  pub scene: Arc<Scene>,

  /// Contains the last render task id for the given scene
  pub last_render_task: core::sync::atomic::AtomicU64,

  pub pending_cross_sync: bool,

  pub root_entity: EntityId,
  pub cursor_entity: Option<EntityId>,
  pub sun_entity: Option<EntityId>,
  pub grid_entity: Option<EntityId>,
  pub sky_entity: Option<EntityId>,
  pub outlines_enabled: Arc<AtomicBool>,

  /// atomic boolean signaling whether we are or not executing a simulation step *CPU Side*. This
  /// means that when this is `false`, it means that either we compute queue is idle or is still in
  /// flight on previous dispatch, therefore `latest_physics_sync` should also be checked
  pub active_physics_task: core::sync::atomic::AtomicBool,
  /// necessary synchronization primitives for "Self Synchronization" (compute N -> compute N + 1).
  /// This means that it packs timeline handle and value (Note: we are assuming vulkan only for now)
  /// We are also packing last semaphore query time and accumulated backoff time.
  /// No need for synchronization cause it is used by
  /// - (read) logic thread if `active_physics_task` is `false`
  /// - (write) physics tasklet when `active_physics_task` is `true`
  pub latest_physics_sync: Option<PhysicsDeviceSelfSync>,

  pub physics_engine_type: Arc<RwLock<PhysicsEngineType>>,

  /// Time state tracking the simulation unscaled and scaled time. Tracks start_epoch but not its
  /// end, therefore stored separately
  pub time_state: alloc::sync::Arc<spin::RwLock<oshal::os::time::v2::TimeState>>,

  pub presentation_engines: Arc<RwLock<BTreeMap<PresentationEngineHandle, PresentationEngineData>>>,

  /// Necessary for the C# side bulk update (TODO check correctness)
  pub changed_entities: Arc<RwLock<BTreeMap<u64, BTreeSet<u64>>>>,

  /// Tasklet synchronization handle to ensure that we don't have multiple C# bulk update tasklets
  /// running at once
  pub entities_update_tasklet: Option<TaskletHandle<()>>,

  pub custom_render_callback: Option<CustomRenderCallback>,
  pub debug_name: alloc::string::String,

  pub scene_snapshot: Option<alloc::boxed::Box<crate::scene::Scene>>,
  pub particle_snapshot: Option<ParticleSystemSnapshot>,

  /// Earth entity hierarchy (subtree, body, orbit). Populated in create_empty_scene2.
  /// None until the scene is created.
  pub earth: Option<SubtreeEntities>,
  /// Comet entity hierarchy (subtree, body, orbit). Populated in create_empty_scene2.
  pub comet: Option<SubtreeEntities>,
  /// Calendar year for which the Earth orbit TrajectoryComponent is currently built.
  /// None = no trajectory yet. Used to avoid redundant UpdateTrajectoryForSpk dispatches.
  pub earth_orbit_year: Option<i32>,
  /// Calendar year for which the Comet orbit TrajectoryComponent is currently built.
  pub comet_orbit_year: Option<i32>,
}

impl Drop for SceneContext {
  fn drop(&mut self) {
    let pes = self.presentation_engines.read();
    if !pes.is_empty() {
      oshal::log!(
        "WARNING: SceneContext dropped but {} presentation engines are still attached!",
        pes.len()
      );
    }
  }
}

impl SceneContext {
  pub fn register_custom_render_callback(&mut self, callback: Option<CustomRenderCallback>) {
    self.custom_render_callback = callback;
  }

  pub fn with_sun_entity(mut self, sun_entity: EntityId) -> EngineResult<Self> {
    if self.sun_entity.is_some() {
      return Err(EngineError::InvalidOperation(
        "simulation_api:with_sun_entity | sun_entity already present in scene",
      ));
    }
    self.sun_entity = Some(sun_entity);
    Ok(self)
  }

  pub fn with_grid_entity(mut self, grid_entity: EntityId) -> EngineResult<Self> {
    if self.grid_entity.is_some() {
      return Err(EngineError::InvalidOperation(
        "simulation_api:with_grid_entity | grid_entity already present in scene",
      ));
    }
    self.grid_entity = Some(grid_entity);
    Ok(self)
  }

  pub fn with_sky_entity(mut self, sky_entity: EntityId) -> EngineResult<Self> {
    if self.sky_entity.is_some() {
      return Err(EngineError::InvalidOperation(
        "simulation_api:with_sky_entity | sky_entity already present in scene",
      ));
    }
    self.sky_entity = Some(sky_entity);
    Ok(self)
  }

  pub fn new_empty(
    scene: Arc<Scene>,
    root_entity: EntityId,
    time_state: Arc<spin::RwLock<oshal::os::time::v2::TimeState>>,
  ) -> Self {
    Self {
      scene,
      root_entity,
      cursor_entity: None,
      sun_entity: None,
      grid_entity: None,
      sky_entity: None,
      outlines_enabled: Arc::new(AtomicBool::new(false)),
      active_physics_task: core::sync::atomic::AtomicBool::new(false),
      latest_physics_sync: None,
      physics_engine_type: Arc::new(RwLock::new(PhysicsEngineType::VulkanCompute)),
      time_state,
      presentation_engines: Arc::new(RwLock::new(BTreeMap::new())),
      scene_snapshot: None,
      changed_entities: Arc::new(RwLock::new(BTreeMap::new())),
      custom_render_callback: None,
      debug_name: alloc::string::String::new(),
      last_render_task: core::sync::atomic::AtomicU64::new(0),
      pending_cross_sync: false,
      entities_update_tasklet: None,
      particle_snapshot: None,
      earth: None,
      comet: None,
      earth_orbit_year: None,
      comet_orbit_year: None,
    }
  }

  pub fn get_entity(&self, external_id: u64) -> Option<EntityId> {
    Some(EntityId::from_ffi(external_id))
  }

  // TODO: change this to be generic on a ForeignSerializable.
  pub fn mark_component_changed(&self, entity_id: u64, component_id: u64) {
    let mut changed = self.changed_entities.write();
    changed.entry(entity_id).or_insert_with(BTreeSet::new).insert(component_id);
  }
}

// -------------------- Render And Logic Threads Main Data ----------------------------

/// TODO: Document this item
pub struct RenderThreadParams {
  channel_capacity: usize,
  pub render_device_handle: gpu::RenderDeviceHandle,
  pub render_frontend: gpu::RenderFrontend,
  thread_pool: Arc<os::pool::ThreadPool>,
  /// See [`SimulationThreads::skip_present`].  Created by the caller and Arc-shared with
  /// `SimulationThreads` so both sides observe the same flag.
  pub skip_present: Arc<core::sync::atomic::AtomicBool>,
}

impl RenderThreadParams {
  const DEFAULT_CHANNEL_CAPACITY: usize = 128;

  /// TODO: Document this item
  pub(crate) fn new(
    backend: gpu::RenderBackendId,
    error_debug_callback: Option<fn(&str)>,
    thread_pool: Arc<os::pool::ThreadPool>,
    skip_present: Arc<core::sync::atomic::AtomicBool>,
  ) -> EngineResult<Self> {
    let render_frontend = {
      let params = RuntimeParams::new_with_callback(error_debug_callback);
      new_render_frontend(backend, &params)?
    };
    let render_device_handle = {
      let params = DeviceAdditionalParams::new();
      render_frontend
        .write()
        .init_device(0, &params)
        .map_err(|e| EngineError::from(e))?
    };

    render_frontend
      .with_device(render_device_handle, |device| {
        device.wire_callbacks(Arc::clone(&thread_pool))
      })
      .map_err(|e| EngineError::from(e))?;

    Ok(RenderThreadParams {
      channel_capacity: Self::DEFAULT_CHANNEL_CAPACITY,
      render_frontend,
      render_device_handle,
      thread_pool,
      skip_present,
    })
  }

  /// TODO: Document this item
  pub fn to_context(self, render_feedback_tx: mpsc::Sender<RenderFeedback>) -> RenderThreadContext {
    RenderThreadContext {
      render_feedback_tx,
      render_frontend: RefCell::new(Some(self.render_frontend)),
      render_device_handle: self.render_device_handle,
      thread_pool: self.thread_pool,
      skip_present: self.skip_present,
    }
  }
}

/// Struct whose lifetime is equal to the render thread's lifetime
pub struct RenderThreadContext {
  pub render_feedback_tx: mpsc::Sender<RenderFeedback>,
  pub render_frontend: RefCell<Option<gpu::RenderFrontend>>,
  pub render_device_handle: gpu::RenderDeviceHandle,
  /// Thread pool for task submission shared between render thread and logic thread
  pub thread_pool: Arc<os::pool::ThreadPool>,
  /// Set to `true` by [`SimulationThreads::drop`] early in the shutdown sequence
  /// to prevent the render thread from calling `vkQueuePresentKHR` on any new
  /// frame.  See [`SimulationThreads::skip_present`] for the full rationale.
  pub skip_present: Arc<core::sync::atomic::AtomicBool>,
}

impl RenderThreadContext {
  pub(crate) fn is_render_single_ownership(&self) -> bool {
    let render_frontend = self.render_frontend.borrow();
    if render_frontend.is_none() {
      return false;
    }
    let render_frontend = unsafe { render_frontend.as_ref().unwrap_unchecked() };
    let strong = Arc::strong_count(&render_frontend);
    let weak = Arc::weak_count(&render_frontend);
    oshal::log!(
      "is_render_single_ownership | strong: {}, weak: {}",
      strong,
      weak
    );
    strong == 1 && weak == 1
  }
}

pub struct LogicThreadParams {
  channel_capacity: usize,
  /// Thread pool for task submission shared between render thread and logic thread
  pub thread_pool: Arc<os::pool::ThreadPool>,
  pub logic_state: Arc<RwLock<LogicState>>,
  pub scenes: Arc<RwLock<SimulationSceneData>>,
  pub ctx_ptr: SendPtrMut<core::ffi::c_void>,
  pub kernels: (gpu::RenderFrontend, gpu::RenderDeviceHandle),
}

impl LogicThreadParams {
  const DEFAULT_CHANNEL_CAPACITY: usize = 128;

  pub fn new(
    thread_pool: Arc<os::pool::ThreadPool>,
    logic_state: Arc<RwLock<LogicState>>,
    scenes: Arc<RwLock<SimulationSceneData>>,
    ctx_ptr: SendPtrMut<core::ffi::c_void>,
    kernels: (gpu::RenderFrontend, gpu::RenderDeviceHandle),
  ) -> Self {
    Self {
      channel_capacity: Self::DEFAULT_CHANNEL_CAPACITY,
      thread_pool,
      logic_state,
      scenes,
      ctx_ptr,
      kernels,
    }
  }

  pub fn to_context(
    self,
    logic_feedback_tx: mpsc::Sender<LogicFeedback>,
    render_tx: mpsc::Sender<RenderCommand>,
  ) -> LogicThreadContext {
    LogicThreadContext {
      logic_state: self.logic_state,
      thread_pool: self.thread_pool,
      logic_feedback_tx,
      scenes: self.scenes,
      ctx_ptr: self.ctx_ptr,
      render_tx,
      kernels: self.kernels,
    }
  }
}

/// Struct whose lifetime is equal to the logic thread's lifetime
#[derive(Clone)]
pub struct LogicThreadContext {
  pub logic_state: Arc<RwLock<LogicState>>,
  /// Thread pool for task submission shared between render thread and logic thread
  pub thread_pool: Arc<os::pool::ThreadPool>,
  /// Transmission channel to send back data to FFI caller threads
  pub logic_feedback_tx: mpsc::Sender<LogicFeedback>,
  pub scenes: Arc<RwLock<SimulationSceneData>>,
  pub ctx_ptr: SendPtrMut<core::ffi::c_void>,
  pub render_tx: mpsc::Sender<RenderCommand>,
  pub kernels: (gpu::RenderFrontend, gpu::RenderDeviceHandle),
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum TaskStatusCode {
  #[default]
  Pending = 0,
  Completed = 1,
  Error = 2,
  Invalid = -1,
}

impl TaskStatusCode {
  pub fn from_render(value: &RenderTaskStatus) -> Self {
    match value.as_ref() {
      RenderTaskStatus::Completed => TaskStatusCode::Completed,
      RenderTaskStatus::Pending => TaskStatusCode::Pending,
      RenderTaskStatus::Error(_) => TaskStatusCode::Error,
    }
  }
}

/// Struct to hold the binary dump of the GPU buffers for particle system. Can't differentiate
/// between scenes
#[derive(Clone, Default)]
pub struct ParticleSystemSnapshot {
  pub global_buffer: alloc::vec::Vec<u8>,
  pub free_list: alloc::vec::Vec<u8>,
  pub page_tables: alloc::collections::BTreeMap<u64, alloc::vec::Vec<u8>>,
}

impl core::fmt::Debug for ParticleSystemSnapshot {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    core::fmt::write(f, format_args!("ParticleSystemSnapshot {{ ... }}"))
  }
}