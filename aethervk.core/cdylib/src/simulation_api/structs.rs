use aethervk_core_rlib as rlib;
use aethervk_oshal_rlib as oshal;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::sync::atomic::AtomicBool;
use spin::rwlock::RwLock;
use thingbuf::mpsc;
use aethervk_core_rlib::gpu::DeviceAdditionalParams;
use aethervk_core_rlib::types::{GpuError, RuntimeParams};
use rlib::{gpu, physics, simulation};
use rlib::gpu::PresentationEngineHandle;
use rlib::scene::{CameraComponent, EntityId, Scene, TransformComponent};
use rlib::simulation::almanac::AlmanacPackedData;
use rlib::types::{EngineError, EngineResult};
use oshal::math::matrix::mat4::Mat4x4f32;
use oshal::os;
use oshal::os::thread::Thread;
use crate::logic_thread::start_logic_thread;
use crate::render_thread::start_render_thread;
use crate::{expect_entity, SimulationContext};

// --------------------- FFI Types ---------------------------

#[repr(u32)]
pub enum FfiRenderTaskStatus {
  Completed = 0,
  Pending = 1,
  Error = 2,
}

// TODO sync with C#
#[repr(u32)]
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum FfiLogicCommandType {
  #[default]
  Shutdown = 0,

  RotateCamera = 1,
  ZoomCamera = 2,
  ResetCamera = 3,
  PanCamera = 4,

  PanCursor = 5,
  MoveCursor = 6,

  SnapToEntity = 7,
  FollowEntity = 8,
  UnfollowEntity = 9,

  FeedbackGetTimeScale = 10,
  FeedbackGetDateTimeUTC = 11,
  FeedbackGetDateTimeLimitsUTC = 12,
  // TODO: probably we'll need Ephemeris duration
}

// TODO modify in C#
#[repr(C, align(4))]
#[derive(Default, Clone, Copy, Debug)]
pub struct FfiLogicCommand {
  pub cmd_type: FfiLogicCommandType,
  pub payload: [u8; 28],
}

impl FfiLogicCommand {
  pub fn get_u32_u64x3_at_start(&self) -> Option<(u32, u64, u64, u64)> {
    let first = self.get_u32_at_offset(0)?;
    let second = self.get_u64_at_offset(4)?;
    let third = self.get_u64_at_offset(12)?;
    let fourth = self.get_u64_at_offset(20)?;
    Some((first, second, third, fourth))
  }

  pub fn get_u64_f32x3_at_start(&self) -> Option<(u64, f32, f32, f32)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_f32_at_offset(12)?;
    let third = self.get_f32_at_offset(16)?;
    let fourth = self.get_f32_at_offset(20)?;
    Some((first, second, third, fourth))
  }

  pub fn get_u64_f32x2_at_start(&self) -> Option<(u64, f32, f32)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_f32_at_offset(12)?;
    let third = self.get_f32_at_offset(16)?;
    Some((first, second, third))
  }

  pub fn get_u64x2_f32_at_start(&self) -> Option<(u64, u64, f32)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_u64_at_offset(12)?;
    let third = self.get_f32_at_offset(20)?;
    Some((first, second, third))
  }

  pub fn get_u64x2_f32x2_at_start(&self) -> Option<(u64, u64, f32, f32)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_u64_at_offset(12)?;
    let third = self.get_f32_at_offset(20)?;
    let fourth = self.get_f32_at_offset(24)?;
    Some((first, second, third, fourth))
  }

  /// Utility for commands which take 2 u64 from start, aligned to 8 (eg 2 entities and a scene id)
  pub fn get_u64x2_at_start(&self) -> Option<(u64, u64)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_u64_at_offset(12)?;
    Some((first, second))
  }

  /// Utility for commands which take 3 u64 from start, aligned to 8 (eg 2 entities and a scene id)
  pub fn get_u64x3_at_start(&self) -> Option<(u64, u64, u64)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_u64_at_offset(12)?;
    let third = self.get_u64_at_offset(20)?;
    Some((first, second, third))
  }

  /// Utility for commands which takes 3 floats from start (eg delta_x and delta_y)
  pub fn get_f32x3_at_start(&self) -> Option<(f32, f32, f32)> {
    let first = self.get_f32_at_offset(0)?;
    let second = self.get_f32_at_offset(4)?;
    let third = self.get_f32_at_offset(8)?;
    Some((first, second, third))
  }

  /// Utility for commands which takes 2 floats from start (eg delta_x and delta_y)
  pub fn get_f32x2_at_start(&self) -> Option<(f32, f32)> {
    let first = self.get_f32_at_offset(0)?;
    let second = self.get_f32_at_offset(4)?;
    Some((first, second))
  }

  /// Safely reads a u32 from the payload.
  /// Payload starts at struct offset 4, so payload offset must be a multiple of 4.
  pub fn get_u32_at_offset(&self, offset: usize) -> Option<u32> {
    let size = core::mem::size_of::<u32>();

    // (offset + 4) % 4 == 0 simplifies down to offset % 4 == 0
    if offset % 4 != 0 || offset + size > self.payload.len() {
      return None;
    }

    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&self.payload[offset..offset + size]);
    Some(u32::from_ne_bytes(bytes))
  }

  /// Safely reads an f32 from the payload.
  /// Payload starts at struct offset 4, so payload offset must be a multiple of 4.
  pub fn get_f32_at_offset(&self, offset: usize) -> Option<f32> {
    let size = core::mem::size_of::<f32>();

    if offset % 4 != 0 || offset + size > self.payload.len() {
      return None;
    }

    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&self.payload[offset..offset + size]);
    Some(f32::from_ne_bytes(bytes))
  }

  /// Safely reads a u64 from the payload.
  /// Because the payload itself starts at struct offset 4,
  /// the absolute memory offset is (offset + 4).
  /// This absolute offset must be a multiple of 8.
  pub fn get_u64_at_offset(&self, offset: usize) -> Option<u64> {
    let size = core::mem::size_of::<u64>();

    // Ensure the *absolute* struct offset is 8-byte aligned.
    // Valid payload offsets for u64: 4, 12, 20.
    if (offset + 4) % 8 != 0 || offset + size > self.payload.len() {
      return None;
    }

    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&self.payload[offset..offset + size]);
    Some(u64::from_ne_bytes(bytes))
  }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)] // Optional: Keeps memory layout predictable if interfacing with C
pub struct FfiMarker {
  pub position: [f32; 3], // [x, y, z]
  pub color: [f32; 3],    // [r, g, b]
  pub size: f32,
}

impl From<FfiMarker> for rlib::scene::Marker {
  fn from(value: FfiMarker) -> Self {
    Self {
      local_pos: value.position,
      color: value.color,
      size: value.size,
    }
  }
}

#[repr(u32)]
pub enum FfiNodeType {
  AABB = 0,
  OBB = 1,
}

#[repr(C)]
pub struct FfiBvhNode {
  pub node_type: FfiNodeType,
  pub min_x: f32,
  pub min_y: f32,
  pub min_z: f32,
  pub max_x: f32,
  pub max_y: f32,
  pub max_z: f32,
  pub center_x: f32,
  pub center_y: f32,
  pub center_z: f32,
  pub extents_x: f32,
  pub extents_y: f32,
  pub extents_z: f32,
  pub left_child: u32,
  pub right_child: u32,
  pub primitive_count: u32,
}

impl FfiBvhNode {
  pub fn from_offsets(left_child: u32, right_child: u32, primitive_count: u32) -> Self {
    Self {
      node_type: FfiNodeType::AABB,
      min_x: 0.0,
      min_y: 0.0,
      min_z: 0.0,
      max_x: 0.0,
      max_y: 0.0,
      max_z: 0.0,
      center_x: 0.0,
      center_y: 0.0,
      center_z: 0.0,
      extents_x: 0.0,
      extents_y: 0.0,
      extents_z: 0.0,
      left_child,
      right_child,
      primitive_count,
    }
  }
}

// --------------------- Drop Wrapper Types ---------------------------

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

  /// given thread should own the receiver.
  unsafe fn new(tx: mpsc::Sender<T>, thread: Thread) -> Self {
    Self {
      tx: Some(tx),
      handle: Some(thread),
    }
  }

  pub fn tx(&self) -> &mpsc::Sender<T> {
    self.tx.as_ref().unwrap()
  }
}

impl<T> Drop for ThreadTxContainer<T> {
  fn drop(&mut self) {
    // 1. Drop the Sender first.
    // This closes the channel. The Receiver in the background thread
    // will now yield `None` or an error, signaling the thread to exit.
    drop(self.tx.take());

    // 2. Take the Thread handle out of the Option and join it.
    // Because the channel is closed, the thread will finish its work
    // and this join will safely unblock.
    if let Some(thread) = self.handle.take() {
      thread.join();
    }
  }
}

// --------------------- Members of SimulationContext ---------------------------

pub struct SimulationSceneData {
  /// Scene state: Scene map
  pub scenes: BTreeMap<u64, Arc<RwLock<SceneContext>>>,
  /// Scene state: next available id. Steadily incremented
  next_scene_id: u64,
  /// mesh cache shared among all scenes
  mesh_cache: Arc<rlib::scene::AssetCache<simulation::comet::Comet>>,
  /// Loaded GLTF Models. Necessary with the asset cache because when a model is evicted,
  /// the string used as key in the cache is eliminated
  pub model_registry: BTreeMap<u64, String>,
  /// model_registry next available id. Steadily incremented
  next_model_id: u64,
}

impl SimulationSceneData {
  pub fn new_inplace(ptr: *mut Self) {
    unsafe { ptr.write(Self::new()) }
  }

  pub fn new() -> Self {
    Self {
      scenes: BTreeMap::new(),
      next_scene_id: 1,
      mesh_cache: Arc::new(rlib::scene::AssetCache::new()),
      model_registry: Default::default(),
      next_model_id: 1,
    }
  }

  /// Insert a new scene in `scenes` and properly increment next free id counter.
  /// return current id
  /// TODO: like Arc, do &mut Self if there will be a conflict, since we implement Deref
  pub fn insert_scene(&mut self, scene_ctx: Arc<RwLock<SceneContext>>) -> u64 {
    debug_assert!(Arc::strong_count(&scene_ctx) == 1 && Arc::weak_count(&scene_ctx) == 0);
    let new_id = self.next_scene_id;
    debug_assert!(new_id != 0);
    // Why unwrap: If you insert a scene only through this method, we will always use
    // fresh new values starting from 1
    self.scenes.insert(new_id, scene_ctx).unwrap();
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
}

impl SimulationThreads {
  pub fn render_thread_running(&self) -> bool {
    let result = self.render_thread.handle.is_some();
    debug_assert!(self.render_thread.tx.is_some() && self.render_feedback_rx.is_some());
    result
  }

  pub fn logic_thread_running(&self) -> bool {
    let result = self.logic_thread.handle.is_some();
    debug_assert!(self.logic_thread.tx.is_some() && self.logic_feedback_rx.is_some());
    result
  }

  pub fn new_running(
    render_thread_params: RenderThreadParams,
    logic_thread_params: LogicThreadParams,
  ) -> EngineResult<Self> {
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
    })
  }

  pub fn start_render_thread(&mut self, params: RenderThreadParams) -> EngineResult<()> {
    if self.render_thread_running() {
      return Err(EngineError::InvalidOperation(
        "SimulationThreads::start_render_thread | render thread already running",
      ));
    }
    let (render_tx, render_rx) = mpsc::channel(params.channel_capacity);
    let (render_feedback_tx, render_feedback_rx) = mpsc::channel(params.channel_capacity);
    let render_thread_handle = start_render_thread(render_rx, params.to_context(render_feedback_tx))?;
    self.render_thread.tx = Some(render_tx);
    self.render_thread.handle = Some(render_thread_handle);
    self.render_feedback_rx = Some(render_feedback_rx);
    Ok(())
  }

  pub fn start_logic_thread(&mut self, params: LogicThreadParams) -> EngineResult<()> {
    if self.logic_thread_running() {
      return Err(EngineError::InvalidOperation(
        "SimulationThreads::start_logic_thread | logic thread already running",
      ));
    }

    let (logic_tx, logic_rx) = mpsc::channel(params.channel_capacity);
    let (logic_feedback_tx, logic_feedback_rx) = mpsc::channel(params.channel_capacity);
    let logic_thread_handle = start_logic_thread(logic_rx, params.to_context(logic_feedback_tx))?;
    self.logic_thread.tx = Some(logic_tx);
    self.logic_thread.handle = Some(logic_thread_handle);
    self.logic_feedback_rx = Some(logic_feedback_rx);
    Ok(())
  }

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
#[derive(Clone, Default)]
pub enum LogicFeedback {
  #[default]
  Empty,
  TimeScale(TimeScale),
  TimeReadings(os::time::TimeReadings),
}

/// Assumes there's one cursor in scene
#[derive(Clone, Debug)]
pub struct RotateCamera {
  pub camera_entity: EntityId,
  pub scene: Arc<RwLock<SceneContext>>,
  pub delta_x: f32,
  pub delta_y: f32,
}
/// Assumes there's one cursor in scene
#[derive(Clone, Debug)]
pub struct ZoomCamera {
  pub camera_entity: EntityId,
  pub scene: Arc<RwLock<SceneContext>>,
  pub amount: f32,
}
/// Assumes there's one cursor in scene
#[derive(Clone, Debug)]
pub struct ResetCamera {
  pub camera_entity: EntityId,
  pub scene: Arc<RwLock<SceneContext>>,
}
/// Assumes there's one cursor in scene
#[derive(Clone, Debug)]
pub struct PanCamera {
  pub camera_entity: EntityId,
  pub scene: Arc<RwLock<SceneContext>>,
  pub delta_x: f32,
  pub delta_y: f32,
}

/// Assumes there's one cursor in scene
#[derive(Clone, Debug)]
pub struct PanCursor {
  pub scene: Arc<RwLock<SceneContext>>,
  pub delta_x: f32,
  pub delta_y: f32,
}
/// Assumes there's one cursor in scene
#[derive(Clone, Debug)]
pub struct MoveCursor {
  pub scene: Arc<RwLock<SceneContext>>,
  pub delta_x: f32,
  pub delta_y: f32,
  pub delta_z: f32,
}

/// Invariant: the entity does exist in the scene.
#[derive(Clone, Debug)]
pub struct SnapToEntity {
  pub snap_entity: EntityId,
  pub target_entity: EntityId,
  pub scene: Arc<RwLock<SceneContext>>,
}
/// Invariant: the entity does exist in the scene. Includes a snap
#[derive(Clone, Debug)]
pub struct FollowEntity {
  pub snap_entity: EntityId,
  pub entity_id: EntityId,
  pub scene: Arc<RwLock<SceneContext>>,
  pub unfollow_other: bool,
}
/// Invariant: the entity does exist in the scene.
#[derive(Clone, Debug)]
pub struct UnfollowEntity {
  pub entity_id: EntityId,
  pub scene: Arc<RwLock<SceneContext>>,
}

#[derive(Clone, Default, Debug)]
pub enum LogicCommand {
  #[default]
  Shutdown,

  RotateCamera(RotateCamera),
  ZoomCamera(ZoomCamera),
  ResetCamera(ResetCamera),
  PanCamera(PanCamera),

  PanCursor(PanCursor),
  MoveCursor(MoveCursor),

  SnapToEntity(SnapToEntity),
  FollowEntity(FollowEntity),
  UnfollowEntity(UnfollowEntity),

  FeedbackGetTimeScale,
  FeedbackGetDateTimeUTC,
  FeedbackGetDateTimeLimitsUTC,
}

impl LogicCommand {
  const PARSING_ERROR: &str = "LogicCommand::new | parsing error";

  // TODO add check for floats that they are valid, normalized, finite floats
  pub fn new(ctx: &SimulationContext, ffi_logic_command: FfiLogicCommand) -> EngineResult<Self> {
    match ffi_logic_command.cmd_type {
      FfiLogicCommandType::Shutdown => Ok(LogicCommand::Shutdown),
      FfiLogicCommandType::RotateCamera => {
        let (camera_entity_id, scene_id, delta_x, delta_y) = ffi_logic_command
          .get_u64x2_f32x2_at_start()
          .ok_or(EngineError::InvalidOperation(Self::PARSING_ERROR))?;
        let scene = ctx
          .scenes
          .get(&scene_id)
          .ok_or_else(|| EngineError::InvalidOperation(Self::PARSING_ERROR))
          .cloned()?;
        let scene_read = scene.read();
        let camera_entity =
          expect_entity!(scene_read, camera_entity_id, "LogicCommand::RotateCamera");
        drop(scene_read);
        Ok(LogicCommand::RotateCamera(RotateCamera {
          camera_entity,
          scene,
          delta_x,
          delta_y,
        }))
      }
      FfiLogicCommandType::ZoomCamera => {
        let (camera_entity_id, scene_id, amount) = ffi_logic_command
          .get_u64x2_f32_at_start()
          .ok_or(EngineError::InvalidOperation(Self::PARSING_ERROR))?;
        let scene = ctx
          .scenes
          .get(&scene_id)
          .ok_or(EngineError::InvalidOperation(
            "LogicCommand::ZoomCamera scene not present",
          ))
          .cloned()?;
        let scene_read = scene.read();
        let camera_entity =
          expect_entity!(scene_read, camera_entity_id, "LogicCommand::ZoomCamera");
        drop(scene_read);
        Ok(LogicCommand::ZoomCamera(ZoomCamera {
          camera_entity,
          scene,
          amount,
        }))
      }
      FfiLogicCommandType::ResetCamera => {
        let (camera_entity_id, scene_id) = ffi_logic_command
          .get_u64x2_at_start()
          .ok_or(EngineError::InvalidOperation(Self::PARSING_ERROR))?;
        let scene = ctx
          .scenes
          .get(&scene_id)
          .ok_or(EngineError::InvalidOperation(
            "LogicCommand::ResetCamera scene not present",
          ))
          .cloned()?;
        let scene_read = scene.read();
        let camera_entity =
          expect_entity!(scene_read, camera_entity_id, "LogicCommand::ResetCamera");
        drop(scene_read);
        Ok(LogicCommand::ResetCamera(ResetCamera {
          camera_entity,
          scene,
        }))
      }
      FfiLogicCommandType::PanCamera => {
        let (camera_entity_id, scene_id, delta_x, delta_y) = ffi_logic_command
          .get_u64x2_f32x2_at_start()
          .ok_or(EngineError::InvalidOperation(Self::PARSING_ERROR))?;
        let scene = ctx
          .scenes
          .get(&scene_id)
          .ok_or(EngineError::InvalidOperation(Self::PARSING_ERROR))
          .cloned()?;
        let scene_read = scene.read();
        let camera_entity =
          expect_entity!(scene_read, camera_entity_id, "LogicCommand::ResetCamera");
        drop(scene_read);
        Ok(LogicCommand::PanCamera(PanCamera {
          camera_entity,
          scene,
          delta_x,
          delta_y,
        }))
      }
      FfiLogicCommandType::PanCursor => {
        let (scene_id, delta_x, delta_y) = ffi_logic_command
          .get_u64_f32x2_at_start()
          .ok_or(EngineError::InvalidOperation(Self::PARSING_ERROR))?;
        let scene = ctx
          .scenes
          .get(&scene_id)
          .ok_or(EngineError::InvalidOperation(Self::PARSING_ERROR))
          .cloned()?;
        Ok(LogicCommand::PanCursor(PanCursor {
          scene,
          delta_x,
          delta_y,
        }))
      }
      FfiLogicCommandType::MoveCursor => {
        let (scene_id, delta_x, delta_y, delta_z) = ffi_logic_command
          .get_u64_f32x3_at_start()
          .ok_or(EngineError::InvalidOperation(Self::PARSING_ERROR))?;
        let scene = ctx
          .scenes
          .get(&scene_id)
          .ok_or(EngineError::InvalidOperation(Self::PARSING_ERROR))
          .cloned()?;
        Ok(LogicCommand::MoveCursor(MoveCursor {
          scene,
          delta_x,
          delta_y,
          delta_z,
        }))
      }
      FfiLogicCommandType::SnapToEntity => {
        let (snap_entity_id, target_entity_id, scene_id) =
          ffi_logic_command
            .get_u64x3_at_start()
            .ok_or(EngineError::InvalidOperation(Self::PARSING_ERROR))?;
        let scene = ctx
          .scenes
          .get(&scene_id)
          .ok_or(EngineError::InvalidOperation(
            "LogicCommand::new | SnapToEntity | scene not found",
          ))?
          .clone();
        let scene_read = scene.read();
        let snap_entity = expect_entity!(
          scene_read,
          snap_entity_id,
          "LogicCommand::new | SnapToEntity"
        );
        let target_entity = expect_entity!(
          scene_read,
          target_entity_id,
          "LogicCommand::new | SnapToEntity"
        );
        drop(scene_read);
        Ok(LogicCommand::SnapToEntity(SnapToEntity {
          snap_entity,
          target_entity,
          scene,
        }))
      }
      FfiLogicCommandType::FollowEntity => {
        let (unfollow_other_u32, snap_entity_ext_id, entity_ext_id, scene_id) = ffi_logic_command
          .get_u32_u64x3_at_start()
          .ok_or(EngineError::InvalidOperation(Self::PARSING_ERROR))?;
        let scene = ctx
          .scenes
          .get(&scene_id)
          .ok_or(EngineError::InvalidOperation(
            "LogicCommand::new | SnapToEntity | scene not found",
          ))?
          .clone();
        let scene_read = scene.read();
        let snap_entity = expect_entity!(
          scene_read,
          snap_entity_ext_id,
          "LogicCommand::new | FollowEntity"
        );
        let entity_id = expect_entity!(
          scene_read,
          entity_ext_id,
          "LogicCommand::new | SnapToEntity"
        );
        let unfollow_other = unfollow_other_u32 != 0;
        drop(scene_read);
        Ok(LogicCommand::FollowEntity(FollowEntity {
          snap_entity,
          entity_id,
          scene,
          unfollow_other,
        }))
      }
      FfiLogicCommandType::UnfollowEntity => {
        let (entity_ext_id, scene_id) = ffi_logic_command
          .get_u64x2_at_start()
          .ok_or(EngineError::InvalidOperation(Self::PARSING_ERROR))?;
        let scene = ctx
          .scenes
          .get(&scene_id)
          .ok_or(EngineError::InvalidOperation(
            "LogicCommand::new | UnfollowEntity | scene not found",
          ))?
          .clone();
        let scene_read = scene.read();
        let entity_id = expect_entity!(
          scene_read,
          entity_ext_id,
          "LogicCommand::new | UnfollowEntity"
        );
        drop(scene_read);
        Ok(LogicCommand::UnfollowEntity(UnfollowEntity {
          entity_id,
          scene,
        }))
      }
      FfiLogicCommandType::FeedbackGetTimeScale => Ok(LogicCommand::FeedbackGetTimeScale),
      FfiLogicCommandType::FeedbackGetDateTimeUTC => Ok(LogicCommand::FeedbackGetDateTimeUTC),
      FfiLogicCommandType::FeedbackGetDateTimeLimitsUTC => {
        Ok(LogicCommand::FeedbackGetDateTimeLimitsUTC)
      }
    }
  }
}

#[derive(Clone, Copy, PartialEq, Default)]
pub enum TimeScale {
  Stopped,
  #[default]
  OneDay,
  OneWeek,
  OneMonth,
}

impl TimeScale {
  fn to_days_per_st_second(self) -> f64 {
    match self {
      TimeScale::Stopped => 0.0,
      TimeScale::OneDay => 1.0,
      TimeScale::OneWeek => 7.0,
      TimeScale::OneMonth => 30.436875,
    }
  }
}

pub struct LogicState {
  pub almanac_data: AlmanacPackedData,
  pub current_scale: TimeScale,
  pub current_epoch: anise::time::Epoch,
  pub epoch_start: anise::time::Epoch,
  pub epoch_end: anise::time::Epoch,
  pub st_seconds_elapsed: f64,
}

impl Default for LogicState {
  fn default() -> Self {
    Self {
      almanac_data: AlmanacPackedData::default(),
      current_scale: TimeScale::Stopped,
      current_epoch: anise::time::Epoch::from_gregorian_utc_at_midnight(2000, 1, 1),
      epoch_start: anise::time::Epoch::from_gregorian_utc_at_midnight(2000, 1, 1),
      epoch_end: anise::time::Epoch::from_gregorian_utc_at_midnight(2100, 1, 1),
      st_seconds_elapsed: 0.0,
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

impl From<RenderTaskStatus> for FfiRenderTaskStatus {
  fn from(value: RenderTaskStatus) -> Self {
    match value {
      RenderTaskStatus::Completed => FfiRenderTaskStatus::Completed,
      RenderTaskStatus::Pending => FfiRenderTaskStatus::Pending,
      RenderTaskStatus::Error(_) => FfiRenderTaskStatus::Error,
    }
  }
}

#[derive(Clone, Default)]
pub enum RenderFeedback {
  #[default]
  Empty,

  TaskCreated(Option<u64>),
  TaskQueryStatus(RenderTaskStatus),
}

/// Invariant: presentation engine is not null and exists inside simulation context and render device
#[derive(Clone, Debug)]
pub struct RenderFrame {
  pub presentation_engine_handle: PresentationEngineHandle,
  pub scene: Arc<RwLock<SceneContext>>,
  pub render_physical_meshes_outline: bool,
  pub camera_entity: EntityId,
  pub window_width: u32,
  pub window_height: u32,
  pub clear_color: [f32; 4],
  pub sun_entity: Option<EntityId>,
  pub sky_entity: Option<EntityId>,
  pub cursor_entity: Option<EntityId>,
}

#[derive(Clone, Debug)]
pub struct DownloadImage {
  pub task_id: u64,
  pub buffer: SendPtrMut<u8>,
  pub buffer_size: usize,
}

/// Invariant: width and height are valid, presentation engine is inside simulation context and render device
#[derive(Clone, Debug)]
pub struct Resize {
  pub presentation_engine_handle: PresentationEngineHandle,
  pub width: u32,
  pub height: u32,
}

#[derive(Clone, Default)]
pub enum RenderCommand {
  #[default]
  Shutdown,

  RenderFrame(RenderFrame),
  DownloadImage(DownloadImage),

  Resize(Resize),

  /// TODO move to compute
  GenerateSky,
}

unsafe impl Send for RenderCommand {}

// --------------------- Internal Types ---------------------------

#[derive(Clone, Copy, Debug)]
struct SendPtr<T>(pub *const T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

#[derive(Clone, Copy, Debug)]
pub struct SendPtrMut<T>(pub *mut T);
unsafe impl<T> Send for SendPtrMut<T> {}
unsafe impl<T> Sync for SendPtrMut<T> {}

#[derive(Clone, Debug)]
pub struct SceneContext {
  pub scene: Arc<Scene>,
  pub entity_map: BTreeMap<u64, EntityId>,
  next_entity_id: u64,
  pub root_entity: EntityId,
  pub active_camera_entity: Option<EntityId>,
  pub cursor_entity: Option<EntityId>,
  pub sun_entity: Option<EntityId>,
  pub grid_entity: Option<EntityId>,
  pub sky_entity: Option<EntityId>,
  pub outlines_enabled: Arc<AtomicBool>,
  pub physics_scene: Option<Arc<RwLock<physics::physics_scene::PhysicsScene>>>,
}

impl SceneContext {
  pub(crate) fn register_present_entities(&mut self) {
    self.register_entity(self.root_entity);
    if self.active_camera_entity.is_some() {
      let active_camera_entity = unsafe { self.active_camera_entity.unwrap_unchecked() };
      self.register_entity(active_camera_entity);
    }
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

  fn with_new_entity_inserted(mut self, entity_id: EntityId) -> EngineResult<Self> {
    self
      .entity_map
      .insert(self.next_entity_id, entity_id)
      .ok_or(EngineError::InvalidOperation(
        "simulation_api:with_new_entity_inserted | Failed to insert entity into entity_map",
      ))?;
    self.next_entity_id = self.next_entity_id + 1;
    Ok(self)
  }

  pub fn with_active_camera_entity(mut self, active_camera_entity: EntityId) -> EngineResult<Self> {
    if self.active_camera_entity.is_some() {
      return Err(EngineError::InvalidOperation(
        "simulation_api:with_active_camera_entity | active_camera_entity already present in scene context",
      ));
    }
    self.active_camera_entity = Some(active_camera_entity);
    self.with_new_entity_inserted(active_camera_entity)
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

  pub fn with_physics_scene(self) -> Self {
    Self {
      physics_scene: Some(Arc::new(RwLock::new(
        physics::physics_scene::PhysicsScene::build_from_scene(self.scene.as_ref()),
      ))),
      ..self
    }
  }

  pub fn new_empty(scene: Arc<Scene>, root_entity: EntityId) -> Self {
    Self {
      scene,
      entity_map: BTreeMap::new(),
      next_entity_id: 1,
      root_entity,
      active_camera_entity: None,
      cursor_entity: None,
      sun_entity: None,
      grid_entity: None,
      sky_entity: None,
      outlines_enabled: Arc::new(AtomicBool::new(false)),
      physics_scene: None,
    }
  }

  pub fn register_entity(&mut self, id: EntityId) -> u64 {
    let external_id = self.next_entity_id;
    self.next_entity_id += 1;
    // unwrap because, being even increasing, it should never fail.
    self.entity_map.insert(external_id, id).unwrap();
    external_id
  }

  pub fn get_entity(&self, external_id: u64) -> Option<EntityId> {
    self.entity_map.get(&external_id).copied()
  }
}

// -------------------- Render And Logic Threads Main Data ----------------------------

pub struct RenderThreadParams {
  channel_capacity: usize,
  pub render_device_handle: gpu::RenderDeviceHandle,
  pub render_frontend: gpu::RenderFrontend,
  thread_pool: Arc<os::pool::ThreadPool>,
  time_info: Arc<RwLock<os::time::TimeInfo>>,
}

impl RenderThreadParams {
  const DEFAULT_CHANNEL_CAPACITY: usize = 128;

  pub(crate) fn new(
    backend: gpu::RenderBackendId,
    error_debug_callback: Option<fn(&str)>,
    thread_pool: Arc<os::pool::ThreadPool>,
    time_info: Arc<RwLock<os::time::TimeInfo>>,
  ) -> EngineResult<Self> {
    let render_frontend = {
      let params = RuntimeParams::new_with_callback(error_debug_callback);
      gpu::new_render_frontend(backend, &params)?
    };
    let render_device_handle = {
      let params = DeviceAdditionalParams::new();
      render_frontend
        .write()
        .init_device(0, &params)
        .map_err(|e| EngineError::from(e))?
    };

    Ok(RenderThreadParams {
      channel_capacity: Self::DEFAULT_CHANNEL_CAPACITY,
      render_frontend,
      render_device_handle,
      thread_pool,
      time_info,
    })
  }

  pub fn to_context(self, render_feedback_tx: mpsc::Sender<RenderFeedback>) -> RenderThreadContext {
    RenderThreadContext {
      render_feedback_tx,
      render_frontend: RefCell::new(Some(self.render_frontend)),
      render_device_handle: self.render_device_handle,
      thread_pool: self.thread_pool,
      time_info: self.time_info,
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
  /// Timing state shared between render thread, simulation thread, and all caller threads
  /// from FFI
  /// TODO: Alternative: Render and Logic Commands take TimeReadings as param. is it better?
  pub time_info: Arc<RwLock<os::time::TimeInfo>>,
}

impl RenderThreadContext {
  pub(crate) fn is_render_single_ownership(&self) -> bool {
    let render_frontend = self.render_frontend.borrow();
    if render_frontend.is_none() {
      return false;
    }
    let render_frontend = unsafe { render_frontend.as_ref().unwrap_unchecked() };
    // Weak count comes from the Simulation Context
    Arc::strong_count(&render_frontend) == 1 && Arc::weak_count(&render_frontend) == 1
  }
}

pub struct LogicThreadParams {
  channel_capacity: usize,
  /// Thread pool for task submission shared between render thread and logic thread
  pub thread_pool: Arc<os::pool::ThreadPool>,
  /// Timing state shared between render thread, simulation thread, and all caller threads
  /// from FFI
  /// TODO: Alternative: Render and Logic Commands take TimeReadings as param. is it better?
  pub time_info: Arc<RwLock<os::time::TimeInfo>>,
}

impl LogicThreadParams {
  const DEFAULT_CHANNEL_CAPACITY: usize = 128;

  pub fn new(
    thread_pool: Arc<os::pool::ThreadPool>,
    time_info: Arc<RwLock<os::time::TimeInfo>>,
  ) -> Self {
    Self {
      channel_capacity: Self::DEFAULT_CHANNEL_CAPACITY,
      thread_pool,
      time_info,
    }
  }

  pub fn to_context(self, logic_feedback_tx: mpsc::Sender<LogicFeedback>) -> LogicThreadContext {
    LogicThreadContext {
      // TODO investigate
      logic_state: Default::default(),
      thread_pool: self.thread_pool,
      time_info: self.time_info,
      logic_feedback_tx,
    }
  }
}

/// Struct whose lifetime is equal to the logic thread's lifetime
pub struct LogicThreadContext {
  pub logic_state: RwLock<Box<LogicState>>,
  /// Thread pool for task submission shared between render thread and logic thread
  pub thread_pool: Arc<os::pool::ThreadPool>,
  /// Timing state shared between render thread, simulation thread, and all caller threads
  /// from FFI
  /// TODO: Alternative: Render and Logic Commands take TimeReadings as param. is it better?
  pub time_info: Arc<RwLock<os::time::TimeInfo>>,
  /// Transmission channel to send back data to FFI caller threads
  pub logic_feedback_tx: mpsc::Sender<LogicFeedback>,
}
