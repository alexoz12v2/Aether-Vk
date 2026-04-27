use crate::structs::{SceneContext, SimulationSceneData};
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
  render_proxy: (WeakRenderFrontend, RenderDeviceHandle),
  threads: structs::SimulationThreads,
  presentation_engines: RwLock<BTreeSet<gpu::PresentationEngineHandle>>,
  scenes: SimulationSceneData,
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

  pub fn get_scene(&self, scene_id: u64) -> Option<RwLockReadGuard<'_, SceneContext>> {
    self.scenes.get(&scene_id).map(|l| l.read())
  }

  pub fn get_scene_mut(&mut self, scene_id: u64) -> Option<RwLockWriteGuard<'_, SceneContext>> {
    self.scenes.get_mut(&scene_id).map(|l| l.write())
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
    let entity = scene
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

static BREADCRUMB_CALLBACK: core::sync::atomic::AtomicPtr<()> =
  core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

fn emit_breadcrumb(status: u32, msg: &str) {
  let fptr = BREADCRUMB_CALLBACK.load(core::sync::atomic::Ordering::Relaxed);
  if !fptr.is_null() {
    if let Ok(c_msg) = alloc::ffi::CString::new(msg) {
      let cb: extern "C" fn(u32, *const c_char) = unsafe { core::mem::transmute(fptr) };
      cb(status, c_msg.as_ptr());
    }
  }
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

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_loadAlmanacFile(
  ctx: *mut SimulationContext,
  path: *const c_char,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.load_almanac_file(path) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_loadAlmanacFile failed: {}",
        e.to_string()
      );
      emit_breadcrumb(1, &alloc::format!("Almanac load failed: {}", e.to_string()));
      false
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_importModel(
  ctx: *mut SimulationContext,
  path: *const c_char,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.import_model(path) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!("avkSimulationContext_importModel failed: {}", e.to_string());
      emit_breadcrumb(1, &alloc::format!("Model import failed: {}", e.to_string()));
      0
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_unloadModel(
  ctx: *mut SimulationContext,
  model_id: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.unload_model(model_id)
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
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.spawn_model_instance(model_id, name) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_spawnModelInstance failed: {}",
        e.to_string()
      );
      emit_breadcrumb(1, &alloc::format!("Model spawn failed: {}", e.to_string()));
      0
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getAlmanacLoadedFiles(
  ctx: *mut SimulationContext,
  count: *mut u32,
) -> *mut *mut c_char {
  if ctx.is_null() {
    return core::ptr::null_mut();
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_almanac_loaded_files(count)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setTimeScale(
  ctx: *mut SimulationContext,
  scale: u32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.set_time_scale(scale)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getSimulationTime(
  ctx: *mut SimulationContext,
) -> f64 {
  if ctx.is_null() {
    return 0.0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_simulation_time()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getSimulationTimeUtc(
  ctx: *mut SimulationContext,
  buffer: *mut c_char,
  buffer_len: u32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_simulation_time_utc(buffer, buffer_len)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setSimulationTime(
  ctx: *mut SimulationContext,
  time_tai: f64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.set_simulation_time(time_tai)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_raycastNdc(
  ctx: *mut SimulationContext,
  ndc_x: f32,
  ndc_y: f32,
  out_hit_entity: *mut u64,
  out_px: *mut f32,
  out_py: *mut f32,
  out_pz: *mut f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.raycast_ndc(ndc_x, ndc_y, out_hit_entity, out_px, out_py, out_pz) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!("avkSimulationContext_raycastNdc failed: {}", e.to_string());
      false
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_raycast(
  ctx: *mut SimulationContext,
  ro_x: f32,
  ro_y: f32,
  ro_z: f32,
  rd_x: f32,
  rd_y: f32,
  rd_z: f32,
  out_hit_entity: *mut u64,
  out_px: *mut f32,
  out_py: *mut f32,
  out_pz: *mut f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.raycast(
    ro_x,
    ro_y,
    ro_z,
    rd_x,
    rd_y,
    rd_z,
    out_hit_entity,
    out_px,
    out_py,
    out_pz,
  ) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!("avkSimulationContext_raycast failed: {}", e.to_string());
      false
    }
  }
}

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
  width: u32,
  height: u32,
) -> *mut SimulationContext {
  let backend_str = if backend.is_null() {
    ""
  } else {
    unsafe { CStr::from_ptr(backend).to_str().unwrap_or("") }
  };
  if let Some(backend) = backend_id_from_str(backend_str) {
    SimulationContext::startup(backend, width, height).unwrap_or_else(|e| {
      oshal::log!("avkSimulationContext_startup failed: {}", e.to_string());
      emit_breadcrumb(1, &alloc::format!("Startup failed: {}", e.to_string()));
      core::ptr::null_mut()
    })
  } else {
    oshal::log!("Unsupported backend: {}", backend_str);
    core::ptr::null_mut()
  }
}

/// Expose creation of windowless presentation engine with the given simulation context.
/// Simulation context should have been initialized
/// Returns `0` in case of error. Otherwise, presentation engine id
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_createPresentationEngine(
  ctx: *mut SimulationContext,
  width: u32,
  height: u32,
) -> u64 {
  if ctx.is_null() || width == 0 || height == 0 {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref
    .create_presentation_engine(width, height)
    .map(|h| h.0)
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_shutdown(ctx: *mut SimulationContext) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.shutdown();
  let _ = unsafe { Box::from_raw(ctx) };
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnProceduralSphere(
  ctx: *mut SimulationContext,
  name: *const c_char,
  radius: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.spawn_procedural_sphere(name, radius) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_spawnProceduralSphere failed: {}",
        e.to_string()
      );
      0
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnEntity(
  ctx: *mut SimulationContext,
  name: *const c_char,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  let name_res = if name.is_null() {
    Err(EngineError::InvalidNullArgument)
  } else {
    unsafe {
      CStr::from_ptr(name).to_str().map_err(|_| EngineError::InvalidOperation("avkSimulationContext_spawnEntity | couldn't convert UTF-8 Null terminated entity name (is there a nul byte in the middle?)"))
    }
  };
  match name_res {
    Ok(name) => ctx_ref.spawn_entity(name).unwrap_or_else(|e| {
      oshal::log!("avkSimulationContext_spawnEntity failed: {}", e.to_string());
      0
    }),
    Err(e) => {
      oshal::log!("avkSimulationContext_spawnEntity failed: {}", e.to_string());
      0
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_removeEntity(
  ctx: *mut SimulationContext,
  entity: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.remove_entity(entity) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_removeEntity failed: {}",
        e.to_string()
      );
      false
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setParent(
  ctx: *mut SimulationContext,
  entity: u64,
  parent: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_parent(entity, parent) {
    oshal::log!("avkSimulationContext_setParent failed: {}", e.to_string());
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addTransformComponent(
  ctx: *mut SimulationContext,
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
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_transform_component(
    entity, pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z, scale_x, scale_y, scale_z,
  ) {
    oshal::log!(
      "avkSimulationContext_addTransformComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setTransformComponent(
  ctx: *mut SimulationContext,
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
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_transform_component(
    entity, pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z, scale_x, scale_y, scale_z,
  ) {
    oshal::log!(
      "avkSimulationContext_setTransformComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTransformComponent(
  ctx: *mut SimulationContext,
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
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.get_transform_component(
    entity, pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z, scale_x, scale_y, scale_z,
  ) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_getTransformComponent failed: {}",
        e.to_string()
      );
      false
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getBvhNodes(
  ctx: *mut SimulationContext,
  entity: u64,
  count: *mut u32,
) -> *mut FfiBvhNode {
  if ctx.is_null() {
    return core::ptr::null_mut();
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.get_bvh_nodes(entity, count) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!("avkSimulationContext_getBvhNodes failed: {}", e.to_string());
      core::ptr::null_mut()
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_freeBvhNodes(ptr: *mut FfiBvhNode, count: u32) {
  SimulationContext::free_bvh_nodes(ptr, count)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setBvhNodeVisibility(
  ctx: *mut SimulationContext,
  entity: u64,
  node_index: u32,
  is_visible: bool,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_bvh_node_visibility(entity, node_index, is_visible) {
    oshal::log!(
      "avkSimulationContext_setBvhNodeVisibility failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_simulationTick(ctx: *mut SimulationContext) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.simulation_tick()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_renderTick(ctx: *mut SimulationContext) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.render_tick()
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
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_task_status(task_id)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_resize(
  ctx: *mut SimulationContext,
  width: u32,
  height: u32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.resize(width, height) {
    oshal::log!("avkSimulationContext_resize failed: {}", e.to_string());
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setActiveCamera(
  ctx: *mut SimulationContext,
  camera: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_active_camera(camera) {
    oshal::log!(
      "avkSimulationContext_setActiveCamera failed: {}",
      e.to_string()
    );
  }
}

// TODO error handling
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_processCommand(
  ctx: *mut SimulationContext,
  command: FfiLogicCommand,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  let active_scene = ctx_ref.active_scene_clone().unwrap();
  let command = LogicCommand {
    ffi_logic_command: command,
    active_scene: Some(active_scene),
  };
  ctx_ref.logic_tx.try_send(command).unwrap();
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setClearColor(
  ctx: *mut SimulationContext,
  r: f32,
  g: f32,
  b: f32,
  a: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.set_clear_color(r, g, b, a)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_downloadImage(
  ctx: *mut SimulationContext,
  buffer_ptr: *mut u8,
  buffer_size: usize,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.download_image(buffer_ptr, buffer_size)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setAssetPath(path: *const c_char) {
  SimulationContext::set_asset_path(path)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn aethervk_core_cdylib_log(msg: *const c_char) {
  SimulationContext::log(msg)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityCount(ctx: *mut SimulationContext) -> u32 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_entity_count()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityIds(
  ctx: *mut SimulationContext,
  out_ids: *mut u64,
  max_count: u32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_entity_ids(out_ids, max_count)
}

/// Returns u32::MAX on error, 0 if successful, otherwise the number of missing characters
/// because of insufficient `out_name` buffer length
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityName(
  ctx: *mut SimulationContext,
  entity: u64,
  out_name: *mut c_char,
  max_len: u32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if out_name.is_null() || max_len == 0 {
    oshal::log!("avkSimulationContext_getEntityName | out_name null or max_len = 0");
    return false;
  }
  let name = unsafe { core::slice::from_raw_parts_mut(out_name, max_len as usize) };
  ctx_ref
    .get_entity_name(entity, name)
    .map(|_| true)
    .unwrap_or_else(|err| {
      oshal::log!(
        "avkSimulationContext_getEntityName failed: {}",
        err.to_string()
      );
      false
    })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityParent(
  ctx: *mut SimulationContext,
  entity: u64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_entity_parent(entity)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_createDefaultScene(
  ctx: *mut SimulationContext,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.create_default_scene().unwrap_or_else(|e| {
    oshal::log!(
      "avkSimulationContext_createDefaultScene failed: {}",
      e.to_string()
    );
    0
  })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_createEmptyScene(ctx: *mut SimulationContext) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.create_empty_scene().unwrap_or_else(|e| {
    oshal::log!(
      "avkSimulationContext_createEmptyScene failed: {}",
      e.to_string()
    );
    0
  })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setEntityVisibility(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  visible: bool,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_entity_visibility(scene_id, entity, visible) {
    oshal::log!(
      "avkSimulationContext_setEntityVisibility failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setEntitySelected(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  selected: bool,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_entity_selected(entity, scene_id, selected) {
    oshal::log!(
      "avkSimulationContext_setEntitySelected failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setEntityFollowing(
  ctx: *mut SimulationContext,
  entity: u64,
  following: bool,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_entity_following(entity, following) {
    oshal::log!(
      "avkSimulationContext_setEntityFollowing failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addCameraComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  fov: f32,
  aspect_ratio: f32,
  near_plane: f32,
  far_plane: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_camera_component(entity, fov, aspect_ratio, near_plane, far_plane) {
    oshal::log!(
      "avkSimulationContext_addCameraComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setCameraComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  is_orthographic: bool,
  fov: f32,
  aspect_ratio: f32,
  near_plane: f32,
  far_plane: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_camera_component(
    entity,
    is_orthographic,
    fov,
    aspect_ratio,
    near_plane,
    far_plane,
  ) {
    oshal::log!(
      "avkSimulationContext_setCameraComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getCameraComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  proj_out: *mut f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.get_camera_component(entity, proj_out) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_getCameraComponent failed: {}",
        e.to_string()
      );
      false
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addPhysicalMeshComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  gltf_path: *const c_char,
  emissive_intensity: f32,
  emissive_color_r: f32,
  emissive_color_g: f32,
  emissive_color_b: f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.add_physical_mesh_component(
    entity,
    gltf_path,
    emissive_intensity,
    emissive_color_r,
    emissive_color_g,
    emissive_color_b,
  ) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_addPhysicalMeshComponent failed: {}",
        e.to_string()
      );
      false
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addSkyComponent(
  ctx: *mut SimulationContext,
  entity: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_sky_component(entity) {
    oshal::log!(
      "avkSimulationContext_addSkyComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addCursorComponent(
  ctx: *mut SimulationContext,
  entity: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_cursor_component(entity) {
    oshal::log!(
      "avkSimulationContext_addCursorComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addSunComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  resolution_x: u32,
  resolution_y: u32,
  resolution_z: u32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_sun_component(entity, resolution_x, resolution_y, resolution_z) {
    oshal::log!(
      "avkSimulationContext_addSunComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addGridComponent(
  ctx: *mut SimulationContext,
  entity: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_grid_component(entity) {
    oshal::log!(
      "avkSimulationContext_addGridComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addMeasurementComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  p1_x: f32,
  p1_y: f32,
  p1_z: f32,
  p2_x: f32,
  p2_y: f32,
  p2_z: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_measurement_component(entity, p1_x, p1_y, p1_z, p2_x, p2_y, p2_z) {
    oshal::log!(
      "avkSimulationContext_addMeasurementComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addImageBillboardComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  is_screen_space: bool,
  width: f32,
  height: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_image_billboard_component(entity, is_screen_space, width, height) {
    oshal::log!(
      "avkSimulationContext_addImageBillboardComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setMarkers(
  ctx: *mut SimulationContext,
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
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_markers(entity, count, px, py, pz, cr, cg, cb, sizes) {
    oshal::log!("avkSimulationContext_setMarkers failed: {}", e.to_string());
  }
}

#[cfg(test)]
mod tests;
