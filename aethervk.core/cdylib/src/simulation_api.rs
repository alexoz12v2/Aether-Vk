use crate::structs::{
  SceneContext, SimulationSceneData, SimulationTaskManager, SimulationTaskResult, FfiRaycastResult,
};
use aethervk_core_rlib as rlib;
use aethervk_core_rlib::gpu::{RenderDeviceHandle, WeakRenderFrontend, WeakRenderFrontendExt};
use aethervk_core_rlib::types::GpuResult;
use aethervk_oshal_rlib as oshal;
use aethervk_oshal_rlib::os;
use aethervk_oshal_rlib::os::pool::WorkloadStatus;
use alloc::collections::BTreeSet;
use alloc::string::ToString;
use alloc::{boxed::Box, string::String, sync::Arc, vec, vec::Vec};
use core::ffi::{c_char, CStr};
use oshal::math::matrix::{Matrix4};
use oshal::math::vector::Vector3;
use oshal::math::{
  matrix::{mat4::Mat4x4f32},
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, vec4::Quat},
};
use rlib::types::EngineResult;
use rlib::{
  gpu::{self, RenderDevice},
  physics,
  scene::{
    CameraComponent, CursorComponent, GridComponent, PhysicalMeshComponent, Scene, SkyComponent,
    SunComponent, TransformComponent,
  },
  simulation,
  types::EngineError,
};
use spin::rwlock::RwLock;
use spin::{RwLockReadGuard, RwLockWriteGuard};

pub mod components_api;
pub mod core_api;
pub mod logic_thread;
pub mod misc_api;
pub mod models_api;
pub mod render_thread;
pub mod scene_api;
pub mod structs;
pub mod time_api;

/// Holds State for the whole simulation. For now, default drop order (from first to last)
/// is fine. Threads are first shut down, then data is deallocated.
/// if this grows more complicated, implement `Drop`
/// creation in [`core_api::SimulationContext::startup`] is in reverse
pub struct SimulationContext {
  /// Necessary evil to let the render thread own a strong reference to render frontend, but
  /// still having the possibility for FFI caller threads to create presentation engines
  pub(crate) render_proxy: (WeakRenderFrontend, RenderDeviceHandle),
  pub(crate) threads: structs::SimulationThreads,
  pub(crate) presentation_engines: RwLock<BTreeSet<gpu::PresentationEngineHandle>>,
  pub(crate) scenes: Arc<RwLock<SimulationSceneData>>,
  pub(crate) task_manager: Arc<RwLock<SimulationTaskManager>>,
  pub(crate) logic_state: Arc<RwLock<structs::LogicState>>,
}

impl Drop for SimulationContext {
  fn drop(&mut self) {
    oshal::log!("SimulationContext drop started");
  }
}

unsafe impl Sync for SimulationContext {}

impl SimulationContext {
  /// Necessary evil function so that FFI caller threads can create a presentation engine on demand
  /// even though render device is owned by render thread
  pub fn with_device<F, R>(&self, f: F) -> EngineResult<R>
  where
    F: FnOnce(&dyn RenderDevice) -> GpuResult<R>,
  {
    let render_device_handle = self.render_proxy.1;
    let render_frontend =
      self
        .render_proxy
        .0
        .as_frontend()
        .ok_or(EngineError::InvalidOperation(
          "SimulationContext::with_device | couldn't upgrade weak pointer to render context",
        ))?;
    render_frontend
      .with_device(render_device_handle, f)
      .map_err(|e| EngineError::from(e))
  }

  pub fn get_scene(&self, scene_id: u64) -> Option<Arc<RwLock<SceneContext>>> {
    self.scenes.read().get(&scene_id).cloned()
  }

}

/// Macro to quickly extract scene, convert option to result and produce a `&'static str` error message
#[macro_export]
macro_rules! expect_scene {
  ($expr:expr, $context:expr) => {
    $expr.ok_or(EngineError::InvalidOperation(concat!(
      $context,
      " | scene not found"
    )))?
  };
}
/// Macro to quickly extract an entity from a scene, convert option to result and produce a `&'static str` error message
#[macro_export]
macro_rules! expect_entity {
  ($scene:expr, $entity:expr, $context:expr) => {
    $scene
      .get_entity($entity)
      .ok_or(EngineError::InvalidOperation(concat!(
        $context,
        " | child entity not found"
      )))?
  };
}
/// Macro to quickly extract both a scene and an entity, returning a tuple (scene, entity_id),
/// while converting options to results with `&'static str` error messages.
#[macro_export]
macro_rules! expect_scene_and_entity {
  ($scene_expr:expr, $entity_expr:expr, $context:expr) => {{
    // Re-use your existing scene macro
    let scene = expect_scene!($scene_expr, $context);

    // Extract the entity
    let entity = scene.read()
      .get_entity($entity_expr)
      .ok_or(EngineError::InvalidOperation(concat!(
        $context,
        " | child entity not found"
      )))?;

    // Return both so they can be destructured
    (scene, entity)
  }};
}

struct PhysicsRebuildWorkload {
  scene: Arc<Scene>,
  physics_scene: Arc<RwLock<physics::physics_scene::PhysicsScene>>,
}

impl os::pool::Workload for PhysicsRebuildWorkload {
  fn execute(&mut self) -> WorkloadStatus {
    let new_physics = physics::physics_scene::PhysicsScene::build_from_scene(&self.scene);
    let mut guard = self.physics_scene.write();
    *guard = new_physics;
    WorkloadStatus::Complete
  }
}

pub(crate) static BREADCRUMB_CALLBACK: core::sync::atomic::AtomicPtr<()> =
  core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

pub(crate) fn emit_breadcrumb(status: u32, msg: &str) {
  let fptr = BREADCRUMB_CALLBACK.load(core::sync::atomic::Ordering::Relaxed);
  if !fptr.is_null() {
    if let Ok(c_msg) = alloc::ffi::CString::new(msg) {
      let cb: extern "C" fn(u32, *const c_char) = unsafe { core::mem::transmute(fptr) };
      cb(status, c_msg.as_ptr());
    }
  }
}

// -------------------- C Exposed API (Async & Stateless) ----------------------------

fn backend_id_from_str(backend_std: &str) -> Option<gpu::RenderBackendId> {
  const VULKAN_BACKEND_STR: &str = "Vulkan";
  match backend_std {
    VULKAN_BACKEND_STR => Some(gpu::VULKAN_RENDER_BACKEND),
    _ => None,
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_startup(
  backend: *const c_char,
) -> *mut SimulationContext {
  let backend_str = if backend.is_null() {
    ""
  } else {
    unsafe { CStr::from_ptr(backend).to_str().unwrap_or("") }
  };
  if let Some(backend) = backend_id_from_str(backend_str) {
    SimulationContext::startup(backend)
      .map(|boxed| Box::into_raw(boxed))
      .unwrap_or_else(|e| {
        oshal::log!("avkSimulationContext_startup failed: {}", e.to_string());
        emit_breadcrumb(1, &alloc::format!("Startup failed: {}", e.to_string()));
        core::ptr::null_mut()
      })
  } else {
    oshal::log!("Unsupported backend: {}", backend_str);
    core::ptr::null_mut()
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_shutdown(ctx: *mut SimulationContext) {
  if !ctx.is_null() {
    let _ = unsafe { Box::from_raw(ctx) };
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTaskStatus(
  ctx: *mut SimulationContext,
  task_id: u64,
) -> i32 {
  if ctx.is_null() {
    return -1;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_task_status(task_id)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTaskResultU64(
  ctx: *mut SimulationContext,
  task_id: u64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_task_result_u64(task_id)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTaskResultBool(
  ctx: *mut SimulationContext,
  task_id: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_task_result_bool(task_id)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTaskResultRaycast(
  ctx: *mut SimulationContext,
  task_id: u64,
  out_hit: *mut structs::FfiRaycastResult,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_task_result_raycast(task_id, out_hit)
}

// --- Scene Management ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_createEmptyScene(ctx: *mut SimulationContext) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.create_empty_scene().unwrap_or_else(|e| {
    oshal::log!("create_empty_scene failed: {}", e);
    0
  })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_createPresentationEngine(
  ctx: *mut SimulationContext,
  width: u32,
  height: u32,
) -> u64 {
  if ctx.is_null() { return 0; }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.create_presentation_engine(width, height).map(|h| h.0).unwrap_or(0)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_createDefaultScene(ctx: *mut SimulationContext) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  match ctx_ref.create_default_scene() {
    Ok(id) => {
      oshal::log!("create_default_scene SUCCESS: {}", id);
      id
    }
    Err(e) => {
      oshal::log!("create_default_scene failed: {}", e);
      0
    }
  }
}

// --- Entity Management (Async) ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnEntity(
  ctx: *mut SimulationContext,
  scene_id: u64,
  name: *const c_char,
) -> u64 {
  if ctx.is_null() || name.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let name_str = unsafe { CStr::from_ptr(name).to_str().unwrap_or("Entity") };
  ctx_ref.spawn_entity(scene_id, name_str).unwrap_or_else(|e| {
    oshal::log!("spawn_entity failed: {}", e);
    0
  })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_removeEntity(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.remove_entity(scene_id, entity).is_ok()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setParent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  parent: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.set_parent(scene_id, entity, parent).is_ok()
}

// --- Components ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addTransformComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  pos_x: f32,
  pos_y: f32,
  pos_z: f32,
  rot_w: f32,
  rot_x: f32,
  rot_y: f32,
  rot_z: f32,
  scale_x: f32,
  scale_y: f32,
  scale_z: f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .add_transform_component(
      scene_id,
      entity,
      Vec3f32::from_components(pos_x, pos_y, pos_z),
      Quat::from_components(rot_w, rot_x, rot_y, rot_z),
      Vec3f32::from_components(scale_x, scale_y, scale_z),
    )
    .is_ok()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setTransformComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  pos_x: f32,
  pos_y: f32,
  pos_z: f32,
  rot_w: f32,
  rot_x: f32,
  rot_y: f32,
  rot_z: f32,
  scale_x: f32,
  scale_y: f32,
  scale_z: f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  if !pos_x.is_finite()
    || !pos_y.is_finite()
    || !pos_z.is_finite()
    || !rot_w.is_finite()
    || !rot_x.is_finite()
    || !rot_y.is_finite()
    || !rot_z.is_finite()
    || !scale_x.is_finite()
    || !scale_y.is_finite()
    || !scale_z.is_finite()
  {
    return false;
  }
  ctx_ref
    .set_transform_component(
      scene_id,
      entity,
      Vec3f32::from_components(pos_x, pos_y, pos_z),
      Quat::from_components(rot_w, rot_x, rot_y, rot_z),
      Vec3f32::from_components(scale_x, scale_y, scale_z),
    )
    .is_ok()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTransformComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  pos_x: *mut f32,
  pos_y: *mut f32,
  pos_z: *mut f32,
  rot_w: *mut f32,
  rot_x: *mut f32,
  rot_y: *mut f32,
  rot_z: *mut f32,
  scale_x: *mut f32,
  scale_y: *mut f32,
  scale_z: *mut f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .get_transform_component(
      scene_id, entity, pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z, scale_x, scale_y, scale_z,
    )
    .is_ok()
}

// --- Queries ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityCount(
  ctx: *mut SimulationContext,
  scene_id: u64,
) -> u32 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_entity_count(scene_id).map(|c| {
    oshal::log!("getEntityCount for scene_id {} returned {}", scene_id, c);
    c
  }).unwrap_or_else(|e| {
    oshal::log!("getEntityCount failed for scene_id {}: {}", scene_id, e);
    0
  })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityIds(
  ctx: *mut SimulationContext,
  scene_id: u64,
  out_ids: *mut u64,
  max_count: u32,
) -> u32 {
  if ctx.is_null() || out_ids.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let ids = unsafe { core::slice::from_raw_parts_mut(out_ids, max_count as usize) };
  ctx_ref.get_entity_ids(scene_id, ids).map(|(n, _)| n).unwrap_or(0)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityName(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  out_name: *mut c_char,
  max_len: u32,
) -> bool {
  if ctx.is_null() || out_name.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let name = unsafe { core::slice::from_raw_parts_mut(out_name, max_len as usize) };
  ctx_ref.get_entity_name(scene_id, entity, name).is_ok()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityParent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_entity_parent(scene_id, entity).unwrap_or(0)
}

// --- Async Heavy Operations ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_importModel(
  ctx: *mut SimulationContext,
  path: *const c_char,
) -> u64 {
  if ctx.is_null() || path.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("").to_string() };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::ImportModel {
      task_id,
      path: path_str,
    });
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_loadAlmanacFile(
  ctx: *mut SimulationContext,
  path: *const c_char,
) -> u64 {
  if ctx.is_null() || path.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("").to_string() };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::LoadAlmanac {
      task_id,
      path: path_str,
    });
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_loadCometSpk(
  ctx: *mut SimulationContext,
  path: *const c_char,
  spkid: u32,
) -> u64 {
  if ctx.is_null() || path.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("").to_string() };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::LoadCometSpk {
      task_id,
      path: path_str,
      spkid,
    });
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnModelInstance(
  ctx: *mut SimulationContext,
  model_id: u64,
  name: *const c_char,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let name_str = if name.is_null() {
    "ModelInstance".to_string()
  } else {
    unsafe {
      CStr::from_ptr(name)
        .to_str()
        .unwrap_or("ModelInstance")
        .to_string()
    }
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::SpawnModelInstance {
      task_id,
      model_id,
      name: name_str,
    });
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_raycastNdc(
  ctx: *mut SimulationContext,
  scene_id: u64,
  ndc_x: f32,
  ndc_y: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .raycast_ndc(scene_id, ndc_x, ndc_y)
    .map(|id| id.get())
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnProceduralSphere(
  ctx: *mut SimulationContext,
  scene_id: u64,
  name: *const c_char,
  radius: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .spawn_procedural_sphere(scene_id, name, radius)
    .unwrap_or_else(|e| {
      oshal::log!("spawn_procedural_sphere failed: {}", e);
      0
    })
}

// --- Camera & Cursor Commands ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_rotateCamera(
  ctx: *mut SimulationContext,
  scene_id: u64,
  camera_entity: u64,
  delta_x: f32,
  delta_y: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let internal_entity = match scene.read().get_entity(camera_entity) {
    Some(e) => e,
    None => return 0,
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::RotateCamera(structs::RotateCamera {
    camera_entity: internal_entity,
    scene,
    delta_x,
    delta_y,
  }));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_zoomCamera(
  ctx: *mut SimulationContext,
  scene_id: u64,
  camera_entity: u64,
  amount: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let internal_entity = match scene.read().get_entity(camera_entity) {
    Some(e) => e,
    None => return 0,
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::ZoomCamera(structs::ZoomCamera {
    camera_entity: internal_entity,
    scene,
    amount,
  }));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_resetCamera(
  ctx: *mut SimulationContext,
  scene_id: u64,
  camera_entity: u64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let internal_entity = match scene.read().get_entity(camera_entity) {
    Some(e) => e,
    None => return 0,
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::ResetCamera(structs::ResetCamera {
    camera_entity: internal_entity,
    scene,
  }));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_panCamera(
  ctx: *mut SimulationContext,
  scene_id: u64,
  camera_entity: u64,
  delta_x: f32,
  delta_y: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let internal_entity = match scene.read().get_entity(camera_entity) {
    Some(e) => e,
    None => return 0,
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::PanCamera(structs::PanCamera {
    camera_entity: internal_entity,
    scene,
    delta_x,
    delta_y,
  }));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_panCursor(
  ctx: *mut SimulationContext,
  scene_id: u64,
  delta_x: f32,
  delta_y: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::PanCursor(structs::PanCursor {
    scene,
    delta_x,
    delta_y,
  }));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_moveCursor(
  ctx: *mut SimulationContext,
  scene_id: u64,
  delta_x: f32,
  delta_y: f32,
  delta_z: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::MoveCursor(structs::MoveCursor {
    scene,
    delta_x,
    delta_y,
    delta_z,
  }));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_snapToEntity(
  ctx: *mut SimulationContext,
  scene_id: u64,
  snap_entity: u64,
  target_entity: u64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let internal_snap = match scene.read().get_entity(snap_entity) {
    Some(e) => e,
    None => return 0,
  };
  let internal_target = match scene.read().get_entity(target_entity) {
    Some(e) => e,
    None => return 0,
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::SnapToEntity(structs::SnapToEntity {
    snap_entity: internal_snap,
    target_entity: internal_target,
    scene,
  }));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_followEntity(
  ctx: *mut SimulationContext,
  scene_id: u64,
  snap_entity: u64,
  target_entity: u64,
  unfollow_other: bool,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let internal_snap = match scene.read().get_entity(snap_entity) {
    Some(e) => e,
    None => return 0,
  };
  let internal_target = match scene.read().get_entity(target_entity) {
    Some(e) => e,
    None => return 0,
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::FollowEntity(structs::FollowEntity {
    snap_entity: internal_snap,
    entity_id: internal_target,
    scene,
    unfollow_other,
  }));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_unfollowEntity(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let internal_entity = match scene.read().get_entity(entity_id) {
    Some(e) => e,
    None => return 0,
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::UnfollowEntity(structs::UnfollowEntity {
    entity_id: internal_entity,
    scene,
  }));
  task_id
}

// --- Callbacks & Asset Path ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setAssetPath(path: *const c_char) {
  SimulationContext::set_asset_path(path)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setLoggerCallback(
  cb: Option<extern "C" fn(*const c_char)>,
) {
  SimulationContext::set_logger_callback(cb)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setBreadcrumbCallback(
  cb: Option<extern "C" fn(u32, *const c_char)>,
) {
  SimulationContext::set_breadcrumb_callback(cb)
}

// --- Tick ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_simulationTick(
  ctx: *mut SimulationContext,
  scene_id: u64,
  delta_time: f64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .simulation_tick(scene_id, delta_time)
    .map(|id| id.get())
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_renderTick(
  ctx: *mut SimulationContext,
  presentation_engine_handle: u64,
  scene_id: u64,
  width: u32,
  height: u32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .render_tick(
      gpu::PresentationEngineHandle(presentation_engine_handle),
      scene_id,
      [width, height],
    )
    .map(|id| id.get())
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_downloadImage(
  ctx: *mut SimulationContext,
  task_id: u64,
  buffer_ptr: *mut u8,
  buffer_size: usize,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.download_image(task_id, buffer_ptr, buffer_size)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addCameraComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  fov: f32,
  aspect: f32,
  near: f32,
  far: f32,
) {
  if ctx.is_null() { return; }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.add_camera_component(scene_id, entity, crate::simulation_api::components_api::CameraParams::Perspective(crate::simulation_api::components_api::PerspectiveCameraParams {
    fov: fov.to_radians(),
    aspect_ratio: aspect,
    near_plane: near,
    far_plane: far,
  }));
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setCameraComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  is_orthographic: bool,
  fov: f32,
  aspect: f32,
  near: f32,
  far: f32,
) {
  if ctx.is_null() { return; }
  let ctx_ref = unsafe { &*ctx };
  let params = if is_orthographic {
    crate::simulation_api::components_api::CameraParams::Orthographic(crate::simulation_api::components_api::OrthographicCameraParams {
      left: -aspect, right: aspect, bottom: -1.0, top: 1.0, near, far
    })
  } else {
    crate::simulation_api::components_api::CameraParams::Perspective(crate::simulation_api::components_api::PerspectiveCameraParams {
      fov: fov.to_radians(), aspect_ratio: aspect, near_plane: near, far_plane: far
    })
  };
  let _ = ctx_ref.set_camera_component(scene_id, entity, params);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getCameraComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  proj_out: *mut f32,
) -> bool {
  if ctx.is_null() || proj_out.is_null() { return false; }
  let ctx_ref = unsafe { &*ctx };
  let mut arr = [0.0; 16];
  if ctx_ref.get_camera_component(scene_id, entity, &mut arr).is_ok() {
    unsafe { core::ptr::copy_nonoverlapping(arr.as_ptr(), proj_out, 16) };
    true
  } else {
    false
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addMeasurementComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  p1x: f32, p1y: f32, p1z: f32,
  p2x: f32, p2y: f32, p2z: f32,
) {
  if ctx.is_null() { return; }
  let ctx_ref = unsafe { &*ctx };
  use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
  let _ = ctx_ref.add_measurement_component(scene_id, entity, Vec3f32::from_components(p1x, p1y, p1z), Vec3f32::from_components(p2x, p2y, p2z));
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addImageBillboardComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  is_screen_space: bool,
  width: f32,
  height: f32,
) {
  if ctx.is_null() { return; }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.add_image_billboard_component(scene_id, entity, is_screen_space, width, height);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setActiveCamera(
  ctx: *mut SimulationContext,
  scene_id: u64,
  camera_entity: u64,
) {
  if ctx.is_null() { return; }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.set_active_camera(scene_id, camera_entity);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setMarkers(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  count: u32,
  px: *const f32,
  py: *const f32,
  pz: *const f32,
  cr: *const f32,
  cg: *const f32,
  cb: *const f32,
  sizes: *const f32,
) {
  if ctx.is_null() || px.is_null() || py.is_null() || pz.is_null() || cr.is_null() || cg.is_null() || cb.is_null() || sizes.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  use crate::structs::FfiMarker;
  use alloc::vec::Vec;

  let mut markers = Vec::with_capacity(count as usize);
  for i in 0..(count as usize) {
    markers.push(FfiMarker {
      position: [unsafe { *px.add(i) }, unsafe { *py.add(i) }, unsafe { *pz.add(i) }],
      color: [unsafe { *cr.add(i) }, unsafe { *cg.add(i) }, unsafe { *cb.add(i) }],
      size: unsafe { *sizes.add(i) },
    });
  }

  let _ = ctx_ref.set_markers(scene_id, entity, &markers);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getBvhNodes(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  count: *mut u32,
) -> *mut crate::structs::FfiBvhNode {
  if ctx.is_null() || count.is_null() { return core::ptr::null_mut(); }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_bvh_nodes(scene_id, entity, count).unwrap_or(core::ptr::null_mut())
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_freeBvhNodes(
  ptr: *mut crate::structs::FfiBvhNode,
  count: u32,
) {
  if !ptr.is_null() && count > 0 {
    let _ = unsafe { alloc::vec::Vec::from_raw_parts(ptr, count as usize, count as usize) };
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setBvhNodeVisibility(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  node_index: u32,
  visible: bool,
) {
  if ctx.is_null() { return; }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.set_bvh_node_visibility(scene_id, entity, node_index, visible);
}

#[cfg(test)]
mod tests;
