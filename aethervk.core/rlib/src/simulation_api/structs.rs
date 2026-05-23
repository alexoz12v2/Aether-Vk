//! structs module.

use crate::{
  gpu,
  gpu::{DeviceAdditionalParams, PresentationEngineHandle},
  physics,
  physics::physics_scene::math::PhysicsSceneMathExt,
  scene::{CameraComponent, EntityId, Scene, TransformComponent},
  simulation,
  simulation::almanac::{AlmanacPackedData, KinematicState},
  simulation_api::{logic_thread::start_logic_thread, render_thread::start_render_thread},
  types::{EngineError, EngineResult, GpuError, GpuResult, RuntimeParams},
};
use aethervk_oshal_rlib as oshal;
use alloc::{
  collections::{BTreeMap, BTreeSet},
  string::String,
  sync::Arc,
  vec::Vec,
};
use core::{
  cell::{RefCell, UnsafeCell},
  mem::MaybeUninit,
  sync::atomic::{AtomicBool, AtomicU64},
};
use oshal::{
  math::{
    matrix::{Matrix4, MatrixVectorMul, SquareMatrix, mat4::Mat4x4f32},
    quaternion::Quaternion,
    vector::{
      Vector, Vector3, Vector4,
      vec3::Vec3f32,
      vec4::{Quat, Vec4f32},
    },
  },
  os,
  os::thread::Thread,
};
use spin::rwlock::RwLock;
use thingbuf::mpsc;
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

  /// TODO: Document this item
  pub fn tx(&self) -> &mpsc::Sender<T> {
    self.tx.as_ref().unwrap()
  }

  /// TODO: Document this item
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

/// TODO: Document this item
pub struct SimulationSceneData {
  /// Scene state: Scene map
  pub scenes: BTreeMap<u64, Arc<RwLock<SceneContext>>>,
  /// Scene state: next available id. Steadily incremented
  next_scene_id: u64,
  /// mesh cache shared among all scenes
  pub(crate) mesh_cache: Arc<crate::scene::AssetCache<simulation::comet::Comet>>,
  /// Loaded GLTF Models. Necessary with the asset cache because when a model is evicted,
  /// the string used as key in the cache is eliminated
  pub model_registry: BTreeMap<u64, String>,
  /// model_registry next available id. Steadily incremented
  next_model_id: u64,
}

impl SimulationSceneData {
  /// TODO: Document this item
  pub fn new_inplace(ptr: *mut Self) {
    unsafe { ptr.write(Self::new()) }
  }

  /// TODO: Document this item
  pub fn new() -> Self {
    Self {
      scenes: BTreeMap::new(),
      next_scene_id: 1,
      mesh_cache: Arc::new(crate::scene::AssetCache::new()),
      model_registry: Default::default(),
      next_model_id: 1,
    }
  }

  /// TODO: Document this item
  pub fn get_scene(&self, scene_id: u64) -> Option<Arc<RwLock<SceneContext>>> {
    self.scenes.get(&scene_id).cloned()
  }

  /// Insert a new scene in `scenes` and properly increment next free id counter.
  /// return current id
  /// TODO: like Arc, do &mut Self if there will be a conflict, since we implement Deref
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
}

impl Drop for SimulationThreads {
  fn drop(&mut self) {
    oshal::log!("SimulationThreads drop started");
    if let Some(tx) = self.logic_thread.tx.take() {
      let _ = tx.try_send(LogicCommand::Shutdown);
    }
    self.logic_feedback_rx = None;
    if let Some(handle) = self.logic_thread.handle.take() {
      let _ = handle.join();
    }

    // Ensure all logic-launched tasklets are finished before shutting down the renderer
    oshal::log!("SimulationThreads waiting for thread pool tasks to complete...");
    self.pool.gather();

    if let Some(tx) = self.render_thread.tx.take() {
      let _ = tx.try_send(RenderCommand::Shutdown);
    }
    self.render_feedback_rx = None;
    if let Some(handle) = self.render_thread.handle.take() {
      let _ = handle.join();
    }
    oshal::log!("SimulationThreads drop finished");
  }
}

/// TODO: Document this item
pub struct LogicWorkload {
  pub cmd: LogicCommand,
  pub ctx: alloc::sync::Arc<LogicThreadContext>,
}

impl LogicThreadContext {
  /// TODO: Document this item
  pub fn load_almanac_file_internal(&self, path: &str) -> EngineResult<()> {
    let mut logic = self.logic_state.write();
    if logic.almanac_data.file_names.iter().any(|f| f == path) {
      return Ok(());
    }

    let path_buf = oshal::os::fs::PathBuf::from(path);
    logic.almanac_data.load_almanac(&path_buf)
  }

  /// TODO: Document this item
  pub fn unload_almanac_file_internal(&self, path: &str) -> EngineResult<()> {
    let mut logic = self.logic_state.write();
    logic.almanac_data.unload_almanac_spk(path)
  }

  /// TODO: Document this item
  pub fn raycast_ndc_internal(
    &self,
    scene_id: u64,
    camera_id: u64,
    ndc_x: f32,
    ndc_y: f32,
  ) -> EngineResult<RaycastResult> {
    let (ro, rd) = {
      let scenes = self.scenes.read();
      let active =
        scenes.get(&scene_id).ok_or(EngineError::InvalidOperation("scene not found"))?.read();
      let active_camera_entity =
        active.get_entity(camera_id).ok_or(EngineError::InvalidOperation("no camera found"))?;

      let mut view = Mat4x4f32::identity();
      active
        .scene
        .with_component(active_camera_entity, |c: &TransformComponent| {
          let right = c.rotation.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
          let up = c.rotation.rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));
          let forward = c.rotation.rotate_vector(Vec3f32::from_components(0.0, -1.0, 0.0));
          view = Mat4x4f32::look_at_axes(right, forward, up, c.position);
        })
        .ok_or(EngineError::InvalidOperation("camera transform missing"))?;

      let mut view_proj_inv = Mat4x4f32::identity();
      active
        .scene
        .with_component(active_camera_entity, |cam: &CameraComponent| {
          let proj = cam.get_projection_matrix();
          let view_proj = proj * view;
          view_proj_inv = view_proj.inverse().unwrap_or(Mat4x4f32::identity());
        })
        .ok_or(EngineError::InvalidOperation("camera component missing"))?;

      let ndc_near = Vec4f32::from_components(ndc_x, ndc_y, 0.0, 1.0);
      let ndc_far = Vec4f32::from_components(ndc_x, ndc_y, 1.0, 1.0);
      let mut world_near = view_proj_inv.mul_vector(ndc_near);
      let mut world_far = view_proj_inv.mul_vector(ndc_far);
      if world_near.w() != 0.0 {
        world_near = world_near / world_near.w();
      }
      if world_far.w() != 0.0 {
        world_far = world_far / world_far.w();
      }
      let ro = Vec3f32::from_components(world_near.x(), world_near.y(), world_near.z());
      let target = Vec3f32::from_components(world_far.x(), world_far.y(), world_far.z());
      let delta = target - ro;
      (ro, delta.normalize())
    };

    aethervk_oshal_rlib::log!(
      "DEBUG: raycast_ndc_internal ro=[{},{},{}] rd=[{},{},{}]",
      ro.x(),
      ro.y(),
      ro.z(),
      rd.x(),
      rd.y(),
      rd.z()
    );

    self.raycast_internal(scene_id, ro, rd)
  }

  /// TODO: Document this item
  pub fn raycast_internal(
    &self,
    scene_id: u64,
    ro: Vec3f32,
    rd: Vec3f32,
  ) -> EngineResult<RaycastResult> {
    use crate::physics::physics_scene::math::closest_intersection;
    let scenes = self.scenes.read();
    let scene_ctx =
      scenes.get(&scene_id).ok_or(EngineError::InvalidOperation("scene not found"))?.read();
    let ps_lock = scene_ctx
      .physics_scene
      .as_ref()
      .ok_or(EngineError::InvalidOperation("physics scene missing"))?;
    let ps = ps_lock.read();

    let st_lock = scene_ctx
      .selection_tlas
      .as_ref()
      .ok_or(EngineError::InvalidOperation("selection tlas missing"))?;
    let st = st_lock.read();

    let ray = crate::math::collision::intersection::Ray {
      origin: ro,
      direction: rd,
      length: f32::MAX,
    };

    let mut hit_instances = alloc::vec::Vec::new();
    if !st.is_empty() {
      use aethervk_oshal_rlib::math::vector::{Vector, Vector3, vec3::Vec3f32};
      use slotmap::Key;
      let mut stack = alloc::vec![0];
      while let Some(node_idx) = stack.pop() {
        if node_idx as usize >= st.len() { continue; }
        let node = &st[node_idx as usize];

        for i in 0..32 {
          let meta = node.metadata[i];
          if meta == 0 { continue; }

          let bmin = Vec3f32::from_components(node.min_x[i], node.min_y[i], node.min_z[i]);
          let bmax = Vec3f32::from_components(node.max_x[i], node.max_y[i], node.max_z[i]);
          let aabb = crate::math::collision::bounds::AABB::new(bmin, bmax);

          if crate::math::collision::intersection::intersect_ray_aabb(&ray, &aabb) {
            if (meta & 0x8000_0000) != 0 {
              let entity_ffi = (((meta & 0x7FFF_FFFF) as u64) << 32) | (node.child_indices[i] as u64);
              let entity = crate::scene::EntityId::from(slotmap::KeyData::from_ffi(entity_ffi));
              hit_instances.push(entity);
            } else {
              stack.push(node.child_indices[i]);
            }
          }
        }
      }
    }
    let intersections: Vec<((f32, Vec3f32, [f32; 2]), EntityId)> = scene_ctx
      .scene
      .query2_res::<crate::scene::PhysicalMeshComponent, TransformComponent, _, (f32, Vec3f32, [f32; 2])>(
      |entity, mesh, transform| {
        if !hit_instances.contains(&entity) || mesh.mesh.bvh.is_none() {
          return None;
        }
        let global_transform = scene_ctx.scene.global_transform(entity).unwrap_or(*transform);
        let model_matrix = Mat4x4f32::translation(global_transform.position)
          * <Mat4x4f32 as oshal::math::matrix::Matrix4>::from_quat_custom_frame(global_transform.rotation)
          * Mat4x4f32::from_scale(global_transform.scale);
        ps.intersect_mesh_bvh_math(ro, rd, model_matrix, mesh, ray.length)
      },
    );

    if let Some((_, hit_point, hit_uv, hit_entity)) = closest_intersection(&intersections) {
      let external_id = scene_ctx
        .entity_map
        .iter()
        .find(|&(_, v)| *v == hit_entity)
        .map(|(ext, _)| *ext)
        .unwrap_or(0);
      return Ok(Some(RayCastHit {
        entity_ext_id: external_id,
        p: hit_point,
        uv: hit_uv,
      }));
    }

    Ok(None)
  }
}

impl SimulationSceneData {
  /// TODO: Document this item
  pub fn import_model_from_mesh(
    &mut self,
    path: String,
    mesh: crate::simulation::comet::Comet,
  ) -> u64 {
    let model_id = self.next_model_id;
    self.next_model_id += 1;
    self.mesh_cache.insert(path.clone(), mesh);
    self.model_registry.insert(model_id, path);
    model_id
  }

  /// TODO: Document this item
  pub fn spawn_model_instance_internal(
    &mut self,
    scene_id: u64,
    model_id: u64,
    name: &str,
  ) -> EngineResult<u64> {
    let path_str = self
      .model_registry
      .get(&model_id)
      .ok_or(EngineError::InvalidOperation("model not found"))?
      .clone();
    let mesh_arc = if let Some(cached) = self.mesh_cache.get(&path_str) {
      cached
    } else {
      return Err(EngineError::InvalidOperation("mesh not found in cache"));
    };

    let scene_ctx_lock =
      self.scenes.get(&scene_id).ok_or(EngineError::InvalidOperation("no scene"))?;
    let mut scene_ctx = scene_ctx_lock.write();
    let entity_id = scene_ctx.scene.spawn_entity(name);
    scene_ctx.scene.add_component(
      entity_id,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )?;
    scene_ctx.scene.add_component(
      entity_id,
      crate::scene::PhysicalMeshComponent {
        asset_path: path_str,
        mesh: mesh_arc,
        emissive_intensity: 0.0,
        emissive_color: [0.0, 0.0, 0.0],
        use_new_path: false,
        paint_display_mode: 0,
        sphere_center: [0.0, 0.0, 0.0],
        sphere_radius: 1.0,
        grid_color: [0.0, 0.0, 0.0],
        grid_density: 1.0,
      },
    )?;
    let root = scene_ctx.root_entity;
    scene_ctx.scene.set_parent(entity_id, Some(root));
    Ok(scene_ctx.register_entity(entity_id))
  }
}

impl SimulationThreads {
  /// TODO: Document this item
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
#[derive(Clone, Default)]
/// TODO: Document this item
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

#[derive(Clone, Debug)]
/// TODO: Document this item
pub struct TogglePaintMode {
  pub scene_id: u64,
  pub entity_id: EntityId,
}

#[derive(Clone, Debug)]
/// TODO: Document this item
pub enum LogicCommand {
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

  TogglePaintMode(TogglePaintMode),

  FeedbackGetSceneTimeScale {
    scene_id: u64,
  },
  FeedbackGetSceneDateTimeUTC {
    scene_id: u64,
  },
  FeedbackGetSceneDateTimeLimitsUTC {
    scene_id: u64,
  },

  SetSceneTimeScale {
    scene_id: u64,
    scale: TimeScale,
  },
  SetSceneEpoch {
    scene_id: u64,
    epoch_tai_seconds: f64,
  },
  SetPhysicsEngineType {
    scene_id: u64,
    engine_type: PhysicsEngineType,
  },
  PauseScene {
    scene_id: u64,
  },
  StepScene {
    scene_id: u64,
    step_days: f64,
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
  LoadCometSpk {
    task_id: u64,
    spk_id: i32,
    frame: anise::frames::Frame,
    epoch: anise::time::Epoch,
  },
  SpawnModelInstance {
    task_id: u64,
    scene_id: u64,
    model_id: u64,
    name: String,
  },
  RaycastNdc {
    task_id: u64,
    scene_id: u64,
    camera_id: u64,
    ndc_x: f32,
    ndc_y: f32,
  },
  Raycast {
    task_id: u64,
    scene_id: u64,
    ro: Vec3f32,
    rd: Vec3f32,
  },
  UpdateTrajectoryForSpk {
    task_id: u64,
    scene_id: u64,
    entity_id: u64,
    spk_id: i32,
    start_epoch_tai_sec: f64,
    end_epoch_tai_sec: f64,
    sample_step_days: f64,
  },
  Custom {
    task_id: u64,
    custom_fn:
      fn(&LogicThreadContext, *mut core::ffi::c_void) -> EngineResult<SimulationTaskResult>,
    user_data: Option<SendPtrMut<core::ffi::c_void>>,
  },
  PlayScene {
    scene_id: u64,
  },
  SnapshotScene {
    scene_id: u64,
  },
  RestoreSnapshot {
    scene_id: u64,
  },
}

impl Default for LogicCommand {
  fn default() -> Self {
    Self::Shutdown
  }
}

impl LogicCommand {
  const PARSING_ERROR: &str = "LogicCommand::new | parsing error";
}

#[derive(Clone, Copy, PartialEq, Default, Debug)]
/// TODO: Document this item
pub enum TimeScale {
  Stopped,
  RealTime,
  #[default]
  OneDay,
  OneWeek,
  OneMonth,
}

impl TimeScale {
  /// TODO: Document this item
  pub fn to_days_per_st_second(self) -> f64 {
    match self {
      TimeScale::Stopped => 0.0,
      TimeScale::RealTime => 1.0 / 86400.0,
      TimeScale::OneDay => 1.0,
      TimeScale::OneWeek => 7.0,
      TimeScale::OneMonth => 30.436875,
    }
  }
}

/// TODO: Document this item
pub struct LogicState {
  pub almanac_data: AlmanacPackedData,
}

impl Default for LogicState {
  fn default() -> Self {
    Self {
      almanac_data: AlmanacPackedData::default(),
    }
  }
}

// --------------------- Internal Types: Render Thread ---------------------------

#[derive(Clone, Default)]
/// TODO: Document this item
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
/// TODO: Document this item
pub enum RenderFeedback {
  #[default]
  Empty,

  TaskCreated(Option<u64>),
  TaskQueryStatus(RenderTaskStatus),
}

pub enum KernelsEnum {
  VulkanCompute(crate::gpu::WeakRenderFrontend, crate::gpu::RenderDeviceHandle),
  CpuSingleThreaded(crate::physics::cpu_kernels::CpuScalarKernels),
  CpuMultiThreaded(crate::physics::cpu_kernels::CpuSimdKernels),
}

#[derive(Clone, Copy, Debug)]
/// TODO: Document this item
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
  pub active_physics_task: alloc::sync::Arc<
    spin::Mutex<
      Option<aethervk_oshal_rlib::os::pool::tasklet::TaskletHandle<crate::types::EngineResult<Option<crate::gpu::CommandBufferSyncInfo>>>>,
    >,
  >,
}

/// Invariant: width and height are valid, presentation engine is inside simulation context and render device
#[derive(Clone, Debug)]
pub struct Resize {
  pub presentation_engine_handle: PresentationEngineHandle,
  pub width: u32,
  pub height: u32,
}

#[derive(Clone, Default)]
/// TODO: Document this item
pub enum RenderCommand {
  #[default]
  Shutdown,

  RenderFrames(alloc::vec::Vec<RenderFrame>),

  Resize(Resize),

  /// TODO move to compute
  GenerateSky,
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

#[derive(Clone, Debug)]
/// TODO: Document this item
pub struct SceneTimeState {
  pub time_info: alloc::sync::Arc<spin::RwLock<aethervk_oshal_rlib::os::time::TimeInfo>>,
  pub current_scale: TimeScale,
  pub current_epoch: anise::time::Epoch,
  pub epoch_start: anise::time::Epoch,
  pub epoch_end: anise::time::Epoch,
  pub st_seconds_elapsed: f64,
  pub is_playing: bool,
  pub manual_step_requests: f64,
  pub is_ticking: alloc::sync::Arc<core::sync::atomic::AtomicBool>,
}

impl Default for SceneTimeState {
  fn default() -> Self {
    Self {
      time_info: alloc::sync::Arc::new(spin::RwLock::new(
        aethervk_oshal_rlib::os::time::TimeInfo::new(
          aethervk_oshal_rlib::os::time::timeus_milliseconds(16),
          aethervk_oshal_rlib::os::time::timeus_milliseconds(100),
          1.0,
        ),
      )),
      current_scale: TimeScale::Stopped,
      current_epoch: anise::time::Epoch::from_gregorian_utc_at_midnight(2000, 1, 1),
      epoch_start: anise::time::Epoch::from_gregorian_utc_at_midnight(2000, 1, 1),
      epoch_end: anise::time::Epoch::from_gregorian_utc_at_midnight(2100, 1, 1),
      st_seconds_elapsed: 0.0,
      is_playing: false,
      manual_step_requests: 0.0,
      is_ticking: alloc::sync::Arc::new(core::sync::atomic::AtomicBool::new(false)),
    }
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
  P1_2Imex,
  P3_4Imex,
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
  P5Imex,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhysicsEngineType {
  CpuScalar,
  CpuSimd,
  VulkanCompute,
  #[cfg(test)]
  Mock(MockTargetShader),
}

impl Default for PhysicsEngineType {
  fn default() -> Self {
    Self::CpuSimd
  }
}

#[derive(Clone, Debug)]
pub struct PresentationEngineData {
  pub is_windowless: bool,
  pub camera_entity: Option<EntityId>,
}

#[derive(Debug)]
pub struct SceneContext {
  pub scene: Arc<Scene>,
  pub entity_map: BTreeMap<u64, EntityId>,
  next_entity_id: u64,
  pub root_entity: EntityId,
  pub cursor_entity: Option<EntityId>,
  pub sun_entity: Option<EntityId>,
  pub grid_entity: Option<EntityId>,
  pub sky_entity: Option<EntityId>,
  pub outlines_enabled: Arc<AtomicBool>,
  pub collisions_enabled: Arc<AtomicBool>,
  pub physics_scene: Option<Arc<RwLock<physics::physics_scene::PhysicsScene>>>,
  pub selection_tlas: Option<Arc<RwLock<alloc::vec::Vec<crate::math::collision::multi_bvh::TlasMultiNode<32>>>>>,
  pub active_physics_task: alloc::sync::Arc<
    spin::Mutex<
      Option<aethervk_oshal_rlib::os::pool::tasklet::TaskletHandle<crate::types::EngineResult<Option<crate::gpu::CommandBufferSyncInfo>>>>,
    >,
  >,
  pub physics_engine_type: Arc<RwLock<PhysicsEngineType>>,
  pub time_state: Arc<RwLock<SceneTimeState>>,
  pub presentation_engines: Arc<RwLock<BTreeMap<PresentationEngineHandle, PresentationEngineData>>>,
  pub changed_entities: Arc<RwLock<BTreeMap<u64, BTreeSet<u64>>>>,
  pub delta_buffer: Arc<RwLock<alloc::boxed::Box<[u64]>>>,
  pub custom_render_callback: Option<CustomRenderCallback>,
  pub debug_name: alloc::string::String,
  pub scene_snapshot: Option<alloc::boxed::Box<crate::scene::Scene>>,
  pub static_tlas: Arc<RwLock<alloc::vec::Vec<crate::math::collision::multi_bvh::TlasMultiNode<32>>>>,
  pub is_static_tlas_dirty: Arc<AtomicBool>,
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

  /// TODO: Document this item
  pub fn with_cursor_entity(mut self, cursor_entity: EntityId) -> EngineResult<Self> {
    if self.cursor_entity.is_some() {
      return Err(EngineError::InvalidOperation(
        "simulation_api:with_cursor_entity | cursor_entity already present in scene",
      ));
    }
    self.cursor_entity = Some(cursor_entity);
    self.with_new_entity_inserted(cursor_entity)
  }

  /// TODO: Document this item
  pub fn with_sun_entity(mut self, sun_entity: EntityId) -> EngineResult<Self> {
    if self.sun_entity.is_some() {
      return Err(EngineError::InvalidOperation(
        "simulation_api:with_sun_entity | sun_entity already present in scene",
      ));
    }
    self.sun_entity = Some(sun_entity);
    self.with_new_entity_inserted(sun_entity)
  }

  /// TODO: Document this item
  pub fn with_grid_entity(mut self, grid_entity: EntityId) -> EngineResult<Self> {
    if self.grid_entity.is_some() {
      return Err(EngineError::InvalidOperation(
        "simulation_api:with_grid_entity | grid_entity already present in scene",
      ));
    }
    self.grid_entity = Some(grid_entity);
    self.with_new_entity_inserted(grid_entity)
  }

  /// TODO: Document this item
  pub fn with_sky_entity(mut self, sky_entity: EntityId) -> EngineResult<Self> {
    if self.sky_entity.is_some() {
      return Err(EngineError::InvalidOperation(
        "simulation_api:with_sky_entity | sky_entity already present in scene",
      ));
    }
    self.sky_entity = Some(sky_entity);
    self.with_new_entity_inserted(sky_entity)
  }

  /// TODO: Document this item
  pub fn with_physics_scene(mut self) -> Self {
    self.physics_scene = Some(Arc::new(RwLock::new(
      physics::physics_scene::PhysicsScene::build_from_scene(self.scene.as_ref(), 0.016),
    )));
    self.selection_tlas = Some(Arc::new(RwLock::new(alloc::vec::Vec::new())));
    self
  }

  /// TODO: Document this item
  pub fn new_empty(scene: Arc<Scene>, root_entity: EntityId) -> Self {
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
      collisions_enabled: Arc::new(AtomicBool::new(false)),
      physics_scene: None,
      selection_tlas: None,
      active_physics_task: alloc::sync::Arc::new(spin::Mutex::new(None)),
      physics_engine_type: Arc::new(RwLock::new(PhysicsEngineType::VulkanCompute)),
      time_state: Arc::new(RwLock::new(SceneTimeState::default())),
      presentation_engines: Arc::new(RwLock::new(BTreeMap::new())),
      scene_snapshot: None,
      static_tlas: Arc::new(RwLock::new(alloc::vec::Vec::new())),
      is_static_tlas_dirty: Arc::new(AtomicBool::new(true)),
      changed_entities: Arc::new(RwLock::new(BTreeMap::new())),
      custom_render_callback: None,
      debug_name: alloc::string::String::new(),
      delta_buffer: Arc::new(RwLock::new(
        alloc::vec![0u64; 131072 /* 1 MiB */].into_boxed_slice(),
      )),
    }
  }

  /// TODO: Document this item
  pub fn register_entity(&mut self, id: EntityId) -> u64 {
    let external_id = self.next_entity_id;
    self.next_entity_id += 1;
    let _ = self.entity_map.insert(external_id, id);
    external_id
  }

  /// TODO: Document this item
  pub fn get_entity(&self, external_id: u64) -> Option<EntityId> {
    self.entity_map.get(&external_id).copied()
  }

  /// TODO: Document this item
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
      gpu::new_render_frontend(backend, &params)?
    };
    let render_device_handle = {
      let params = DeviceAdditionalParams::new();
      render_frontend.write().init_device(0, &params).map_err(|e| EngineError::from(e))?
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
  /// TODO: Document this item
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

/// TODO: Document this item
pub struct LogicThreadParams {
  channel_capacity: usize,
  /// Thread pool for task submission shared between render thread and logic thread
  pub thread_pool: Arc<os::pool::ThreadPool>,
  pub task_manager: Arc<RwLock<SimulationTaskManager>>,
  pub logic_state: Arc<RwLock<LogicState>>,
  pub scenes: Arc<RwLock<SimulationSceneData>>,
  pub ctx_ptr: SendPtrMut<core::ffi::c_void>,
  pub kernels: Arc<RwLock<KernelsEnum>>,
}

impl LogicThreadParams {
  const DEFAULT_CHANNEL_CAPACITY: usize = 128;

  /// TODO: Document this item
  pub fn new(
    thread_pool: Arc<os::pool::ThreadPool>,
    task_manager: Arc<RwLock<SimulationTaskManager>>,
    logic_state: Arc<RwLock<LogicState>>,
    scenes: Arc<RwLock<SimulationSceneData>>,
    ctx_ptr: SendPtrMut<core::ffi::c_void>,
    kernels: Arc<RwLock<KernelsEnum>>,
  ) -> Self {
    Self {
      channel_capacity: Self::DEFAULT_CHANNEL_CAPACITY,
      thread_pool,
      task_manager,
      logic_state,
      scenes,
      ctx_ptr,
      kernels,
    }
  }

  /// TODO: Document this item
  pub fn to_context(
    self,
    logic_feedback_tx: mpsc::Sender<LogicFeedback>,
    render_tx: mpsc::Sender<RenderCommand>,
  ) -> LogicThreadContext {
    LogicThreadContext {
      logic_state: self.logic_state,
      thread_pool: self.thread_pool,
      logic_feedback_tx,
      task_manager: self.task_manager,
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
  pub task_manager: Arc<RwLock<SimulationTaskManager>>,
  pub scenes: Arc<RwLock<SimulationSceneData>>,
  pub ctx_ptr: SendPtrMut<core::ffi::c_void>,
  pub render_tx: mpsc::Sender<RenderCommand>,
  pub kernels: Arc<RwLock<KernelsEnum>>,
}

#[derive(Clone, Copy, Debug)]
/// TODO: Document this item
pub struct RayCastHit {
  pub entity_ext_id: u64,
  pub p: Vec3f32,
  pub uv: [f32; 2],
}

/// TODO: Document this item
pub type RaycastResult = Option<RayCastHit>;

/// TODO: Document this item
pub enum SimulationTaskResult {
  None,
  U64(u64),
  Bool(bool),
  Raycast(RaycastResult),
  Vec3(Vec3f32), // TODO more vector types in oshal (design first by me)
  KinematicState(KinematicState),
  String(String),
}

/// TODO: Document this item
pub enum SimulationTaskStatus {
  Pending,
  Completed(SimulationTaskResult),
  Error(String),
}

impl AsRef<SimulationTaskStatus> for SimulationTaskStatus {
  fn as_ref(&self) -> &SimulationTaskStatus {
    self
  }
}

/// TODO: Document this item
pub struct SimulationTaskManager {
  next_task_id: u64,
  tasks: BTreeMap<u64, SimulationTaskStatus>,
}

impl Drop for SimulationTaskManager {
  fn drop(&mut self) {
    oshal::log!("SimulationTaskManager drop started");
  }
}

impl SimulationTaskManager {
  /// TODO: Document this item
  pub fn new() -> Self {
    Self {
      next_task_id: 1,
      tasks: BTreeMap::new(),
    }
  }

  /// TODO: Document this item
  pub fn create_task(&mut self) -> core::num::NonZero<u64> {
    let id = self.next_task_id | (1u64 << 63);
    self.next_task_id += 1;
    self.tasks.insert(id, SimulationTaskStatus::Pending);
    unsafe { core::num::NonZero::new_unchecked(id) }
  }

  /// TODO: Document this item
  pub fn success_task(&mut self, id: u64, result: SimulationTaskResult) {
    self.tasks.insert(id, SimulationTaskStatus::Completed(result));
  }

  /// TODO: Document this item
  pub fn fail_task(&mut self, id: u64, error: String) {
    self.tasks.insert(id, SimulationTaskStatus::Error(error));
  }

  /// TODO: Document this item
  pub fn get_status(&self, id: u64) -> TaskStatusCode {
    self.tasks.get(&id).map(|t| TaskStatusCode::from_sim(t)).unwrap_or(TaskStatusCode::Invalid)
  }

  /// TODO: Document this item
  pub fn take_result(&mut self, id: u64) -> Option<SimulationTaskResult> {
    if let Some(SimulationTaskStatus::Completed(res)) = self.tasks.remove(&id) {
      Some(res)
    } else {
      None
    }
  }
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
/// TODO: Document this item
pub enum TaskStatusCode {
  #[default]
  Pending = 0,
  Completed = 1,
  Error = 2,
  Invalid = -1,
}

impl TaskStatusCode {
  /// TODO: Document this item
  pub fn from_sim(value: &SimulationTaskStatus) -> Self {
    match value.as_ref() {
      SimulationTaskStatus::Pending => TaskStatusCode::Pending,
      SimulationTaskStatus::Completed(_) => TaskStatusCode::Completed,
      SimulationTaskStatus::Error(_) => TaskStatusCode::Error,
    }
  }
  /// TODO: Document this item
  pub fn from_render(value: &RenderTaskStatus) -> Self {
    match value.as_ref() {
      RenderTaskStatus::Completed => TaskStatusCode::Completed,
      RenderTaskStatus::Pending => TaskStatusCode::Pending,
      RenderTaskStatus::Error(_) => TaskStatusCode::Error,
    }
  }
}

// 1. Group the shared data and the atomic signal into a single struct
/// TODO: Document this item
pub struct SharedState<T> {
  pub done_signal: AtomicBool,
  pub data: UnsafeCell<MaybeUninit<T>>,
}

// We promise the compiler we are handling thread safety manually via the AtomicBool
unsafe impl<T: Send> Sync for SharedState<T> {}
unsafe impl<T: Send> Send for SharedState<T> {}

// 2. The wrapper is now just a single Arc
/// TODO: Document this item
pub struct SharedDataWrapper<T> {
  inner: Arc<SharedState<T>>,
}

// 3. Implement Default
impl<T> Default for SharedDataWrapper<T> {
  fn default() -> Self {
    Self::new()
  }
}

// 4. Implement Clone (This now safely shares BOTH the data and the signal)
impl<T> Clone for SharedDataWrapper<T> {
  fn clone(&self) -> Self {
    Self {
      inner: Arc::clone(&self.inner),
    }
  }
}

impl<T> SharedDataWrapper<T> {
  /// TODO: Document this item
  pub fn new() -> Self {
    Self {
      inner: Arc::new(SharedState {
        done_signal: AtomicBool::new(false),
        data: UnsafeCell::new(MaybeUninit::uninit()),
      }),
    }
  }

  /// TODO: Document this item
  pub unsafe fn write_value(&self, v: T) {
    debug_assert_eq!(
      self.inner.done_signal.load(core::sync::atomic::Ordering::Relaxed),
      false
    );

    // Write the data first
    unsafe { (*self.inner.data.get()).write(v) };

    // Release ordering ensures the memory write is visible to other threads
    // before the flag is set to true.
    self.inner.done_signal.store(true, core::sync::atomic::Ordering::Release);
  }

  // TODO error timeout after 10 ms
  /// TODO: Document this item
  pub unsafe fn read_value(self) -> T {
    loop {
      // Acquire ordering pairs with Release to ensure cache coherency.
      // We can just load with Acquire directly in the loop.
      if self.inner.done_signal.load(core::sync::atomic::Ordering::Acquire) {
        break;
      }
      core::hint::spin_loop();
    }

    // Safely extract the value without waiting for the writer thread to drop its Arc.
    // Since MaybeUninit does not drop its contents, ptr::read safely transfers ownership
    // to us. The Arc will eventually deallocate the wrapper memory later.
    unsafe { core::ptr::read(self.inner.data.get()).assume_init() }
  }
}
