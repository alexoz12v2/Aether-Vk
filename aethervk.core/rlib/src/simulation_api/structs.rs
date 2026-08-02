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
}

impl Drop for SimulationThreads {
  fn drop(&mut self) {
    oshal::log!("SimulationThreads drop started");

    // Drop the sender — this closes the channel from the sender side.
    // The logic thread's try_recv loop returns on TryRecvError::Closed (line ~326
    // of logic_thread.rs), so closing the channel is the reliable shutdown signal.
    // Using try_send(Shutdown) was unreliable: if the 128-slot channel is full
    // (e.g. many commands queued during heavy use), try_send silently fails and
    // handle.join() hangs forever.
    self.logic_thread.tx.take();
    self.logic_feedback_rx = None;
    if let Some(handle) = self.logic_thread.handle.take() {
      handle.join();
    }

    // Same pattern for render thread.
    // MUST drop render thread BEFORE gathering the pool!
    // Dropping the render thread drops the `Device`, which sets `callback_stop_signal`
    // to true. This stops `TimelinePollingWorkload` which is running on the pool.
    self.render_thread.tx.take();
    self.render_feedback_rx = None;
    if let Some(handle) = self.render_thread.handle.take() {
      handle.join();
    }

    // Ensure all logic-launched tasklets are finished before shutting down the renderer
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
  pub fn load_almanac_file_internal(&self, path: &str) -> EngineResult<()> {
    let mut logic = self.logic_state.write();
    if logic.almanac_data.file_names.iter().any(|f| f == path) {
      return Ok(());
    }

    let path_buf = oshal::os::fs::PathBuf::from(path);
    logic.almanac_data.load_almanac(&path_buf)
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
    let mut this = Self::new_idle()?;
    this.start_render_thread(render_thread_params)?;
    this.start_logic_thread(logic_thread_params)?;
    Ok(this)
  }

  /// Creates the thread pool only.
  pub fn new_idle() -> EngineResult<Self> {
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
  // TODO for now commented out. If needed, implement the Scene Observer concept
  // FollowEntity {
  //   snap_entity: EntityId,
  //   entity_id: EntityId,
  //   scene: Arc<RwLock<SceneContext>>,
  //   unfollow_other: bool,
  // },
  // TODO for now commented out. If needed, implement the Scene Observer concept
  // UnfollowEntity {
  //   entity_id: EntityId,
  //   scene: Arc<RwLock<SceneContext>>,
  // },
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
        self.query_exp_backoff_us <<= 1;
        false
      }
    } else {
      false
    }
  }
}

#[derive(Debug)]
pub struct SceneContext {
  pub scene: Arc<Scene>,
  pub entity_map: BTreeMap<u64, EntityId>,
  next_entity_id: u64,

  /// Contains the last render task id for the given scene
  pub last_render_task: core::sync::atomic::AtomicU64,

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
  /// TODO: Document this item
  pub(crate) fn register_present_entities(&mut self) {
    self.register_entity(self.root_entity);
    if self.cursor_entity.is_some() {
      let cursor_entity = unsafe { self.cursor_entity.unwrap_unchecked() };
      self.register_entity(cursor_entity);
    }
    if self.sun_entity.is_some() {
      let sun_entity = unsafe { self.sun_entity.unwrap_unchecked() };
      self.register_entity(sun_entity);
    }
    if self.grid_entity.is_some() {
      let grid_entity = unsafe { self.grid_entity.unwrap_unchecked() };
      self.register_entity(grid_entity);
    }
    if self.sky_entity.is_some() {
      let sky_entity = unsafe { self.sky_entity.unwrap_unchecked() };
      self.register_entity(sky_entity);
    }
  }

  /// TODO: Document this item
  pub fn register_custom_render_callback(&mut self, callback: Option<CustomRenderCallback>) {
    self.custom_render_callback = callback;
  }

  fn with_new_entity_inserted(mut self, entity_id: EntityId) -> EngineResult<Self> {
    if self.entity_map.insert(self.next_entity_id, entity_id).is_some() {
      return Err(EngineError::InvalidOperation(
        "simulation_api:with_new_entity_inserted | Failed to insert entity into entity_map",
      ));
    }
    self.next_entity_id += 1;
    Ok(self)
  }

  pub fn with_cursor_entity(mut self, cursor_entity: EntityId) -> EngineResult<Self> {
    if self.cursor_entity.is_some() {
      return Err(EngineError::InvalidOperation(
        "simulation_api:with_cursor_entity | cursor_entity already present in scene",
      ));
    }
    self.cursor_entity = Some(cursor_entity);
    self.with_new_entity_inserted(cursor_entity)
  }

  pub fn with_sun_entity(mut self, sun_entity: EntityId) -> EngineResult<Self> {
    if self.sun_entity.is_some() {
      return Err(EngineError::InvalidOperation(
        "simulation_api:with_sun_entity | sun_entity already present in scene",
      ));
    }
    self.sun_entity = Some(sun_entity);
    self.with_new_entity_inserted(sun_entity)
  }

  pub fn with_grid_entity(mut self, grid_entity: EntityId) -> EngineResult<Self> {
    if self.grid_entity.is_some() {
      return Err(EngineError::InvalidOperation(
        "simulation_api:with_grid_entity | grid_entity already present in scene",
      ));
    }
    self.grid_entity = Some(grid_entity);
    self.with_new_entity_inserted(grid_entity)
  }

  pub fn with_sky_entity(mut self, sky_entity: EntityId) -> EngineResult<Self> {
    if self.sky_entity.is_some() {
      return Err(EngineError::InvalidOperation(
        "simulation_api:with_sky_entity | sky_entity already present in scene",
      ));
    }
    self.sky_entity = Some(sky_entity);
    self.with_new_entity_inserted(sky_entity)
  }

  pub fn new_empty(
    scene: Arc<Scene>,
    root_entity: EntityId,
    time_state: Arc<spin::RwLock<oshal::os::time::v2::TimeState>>,
  ) -> Self {
    let mut entity_map = BTreeMap::new();
    entity_map.insert(1, root_entity);
    Self {
      scene,
      entity_map,
      next_entity_id: 2,
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
      entities_update_tasklet: None,
      particle_snapshot: None,
    }
  }

  pub fn register_entity(&mut self, id: EntityId) -> u64 {
    let external_id = self.next_entity_id;
    self.next_entity_id += 1;
    let _ = self.entity_map.insert(external_id, id);
    external_id
  }

  pub fn get_entity(&self, external_id: u64) -> Option<EntityId> {
    self.entity_map.get(&external_id).copied()
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
}

impl RenderThreadParams {
  const DEFAULT_CHANNEL_CAPACITY: usize = 128;

  /// TODO: Document this item
  pub(crate) fn new(
    backend: gpu::RenderBackendId,
    error_debug_callback: Option<fn(&str)>,
    thread_pool: Arc<os::pool::ThreadPool>,
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
    })
  }

  /// TODO: Document this item
  pub fn to_context(self, render_feedback_tx: mpsc::Sender<RenderFeedback>) -> RenderThreadContext {
    RenderThreadContext {
      render_feedback_tx,
      render_frontend: RefCell::new(Some(self.render_frontend)),
      render_device_handle: self.render_device_handle,
      thread_pool: self.thread_pool,
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