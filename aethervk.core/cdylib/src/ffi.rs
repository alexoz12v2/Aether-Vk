use aethervk_core_rlib::simulation_api::*;
use aethervk_core_rlib::simulation_api::structs::*;
use aethervk_oshal_rlib as oshal;
use core::ffi::{c_char, CStr};
use alloc::{boxed::Box, string::ToString};
use core::str::FromStr;
use aethervk_core_rlib::gpu;
use aethervk_core_rlib::math::collision::linear_bvh;
use aethervk_core_rlib::math::collision::linear_bvh::LinearBVHNode;
use aethervk_core_rlib::scene::Marker;
use aethervk_core_rlib::simulation::almanac::SUN_ECLIPJ200;
use aethervk_core_rlib::simulation_api::components_api::{
  CameraParams, OrthographicCameraParams, PerspectiveCameraParams,
};
use aethervk_core_rlib::types::EngineError;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::vec4::Quat;
use aethervk_oshal_rlib::math::vector::{Vector3, Vector4};
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
    SimulationContext::startup(backend, None)
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
  out_hit: *mut FfiRaycastResult,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_task_result_raycast(task_id, out_hit)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTaskResultKinematicState(
  ctx: *mut SimulationContext,
  task_id: u64,
  out_state: *mut FfiKinematicState,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_task_result_kinematic_state(task_id, out_state)
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
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .create_presentation_engine(width, height)
    .map(|h| h.0)
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_createDefaultScene(
  ctx: *mut SimulationContext,
) -> u64 {
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
  ctx_ref
    .spawn_entity(scene_id, name_str)
    .unwrap_or_else(|e| {
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
  ctx_ref
    .get_entity_count(scene_id)
    .map(|c| {
      oshal::log!("getEntityCount for scene_id {} returned {}", scene_id, c);
      c
    })
    .unwrap_or_else(|e| {
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
  ctx_ref
    .get_entity_ids(scene_id, ids)
    .map(|(n, _)| n)
    .unwrap_or(0)
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
pub unsafe extern "C" fn avkSimulationContext_unloadModel(
  ctx: *mut SimulationContext,
  model_id: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.unload_model(model_id);
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

// TODO C# side: builder/helper functions for strings suppoerted by `anise::time::Epoch::from_str("2024-03-24 12:00:00 TDB")`
// alternative (nah): Epoch::from_gregorian_utc_at_midnight(2000, 1, 1). But string is more precise
// TODO C# side: update
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_loadCometSpk(
  ctx: *mut SimulationContext,
  spk_id: i32,
  epoch_raw: *const c_char,
) -> u64 {
  if ctx.is_null() || epoch_raw.is_null() {
    return 0;
  }
  let epoch_opt = unsafe { CStr::from_ptr(epoch_raw) }
    .to_str()
    .ok()
    .and_then(|epoch_str| anise::time::Epoch::from_str(epoch_str).ok());
  if epoch_opt.is_none() {
    // TODO log/breadcrumb?
    return 0;
  }
  let epoch = unsafe { epoch_opt.unwrap_unchecked() };
  let ctx_ref = unsafe { &*ctx };

  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::LoadCometSpk {
      task_id,
      spk_id,
      frame: SUN_ECLIPJ200,
      epoch,
    });
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnModelInstance(
  ctx: *mut SimulationContext,
  scene_id: u64,
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
      scene_id,
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
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::RotateCamera(structs::RotateCamera {
      camera_entity: internal_entity,
      scene,
      delta_x,
      delta_y,
    }));
  task_id
}

// TODO pass the task_id to the command so that logic thread can give feedback. apply this to all functions
// TODO: since this try_send functions are used in test/simulation_test and tests of simulation_api, refactor these commands into methods
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
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::ZoomCamera(structs::ZoomCamera {
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
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::ResetCamera(structs::ResetCamera {
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
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::PanCamera(structs::PanCamera {
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
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::PanCursor(structs::PanCursor {
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
  // TODO pass the task_id to the command so that logic thread can give feedback. apply this to all functions
  // TODO: since this try_send functions are used in test/simulation_test and tests of simulation_api, refactor these commands into methods
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::MoveCursor(structs::MoveCursor {
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
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::SnapToEntity(structs::SnapToEntity {
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
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::FollowEntity(structs::FollowEntity {
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
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::UnfollowEntity(
      structs::UnfollowEntity {
        entity_id: internal_entity,
        scene,
      },
    ));
  task_id
}

// --- Callbacks & Asset Path ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setAssetPath(path: *const c_char) {
  if path.is_null() {
    return;
  }

  if let Ok(path_str) = unsafe { core::ffi::CStr::from_ptr(path) }.to_str() {
    SimulationContext::set_asset_path(path_str)
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getBodyRadius(body_id: i32) -> f32 {
  if let Some(asset_dir) = aethervk_core_rlib::gpu::ASSET_DIR.read().as_ref() {
    let pck_path = alloc::format!("{}/planets/pck00011.tpc", asset_dir);
    if let Some(radii) = aethervk_core_rlib::simulation::pck::read_body_radii(&pck_path, body_id) {
      return (radii[0] / aethervk_core_rlib::simulation::almanac::DISTANCE_SCALE_FACTOR) as f32;
    }
  }
  0.0
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

/// the C# caller should be responsible for polling getTaskStatus and only calling download_image once the status is completed.
/// Given the architecture of the C# async API and the fact that Vulkan windowless downloads are placed in persistently mapped memory buffers, bouncing the download request through the RenderCommand channel was actually an anti-pattern. Here is why:
/// 1. Thread Safety: is_task_completed and read_windowless_download only acquire read/write locks (self.res.read() and pending_downloads.write()). They don't actually need to execute on the Render Thread.
/// 2. No Render Thread Blocking: By having C# call download_image synchronously after polling confirms it's ready, the memory copy happens on the caller thread (e.g., C#'s worker pool), freeing the Render Thread to continue pushing Vulkan commands without stalling
///   during the memory copy.
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

/// fov is in radians
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addCameraComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  is_orthographic: bool,
  fov: f32,
  aspect: f32,
  near: f32,
  far: f32,
  ortho_left: f32,
  ortho_right: f32,
  ortho_bottom: f32,
  ortho_top: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let params = if is_orthographic {
    CameraParams::Orthographic(OrthographicCameraParams {
      left: ortho_left,
      right: ortho_right,
      bottom: ortho_bottom,
      top: ortho_top,
      near,
      far,
    })
  } else {
    CameraParams::Perspective(PerspectiveCameraParams {
      fov: fov,
      aspect_ratio: aspect,
      near_plane: near,
      far_plane: far,
    })
  };
  let _ = ctx_ref.add_camera_component(scene_id, entity, params);
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
  ortho_left: f32,
  ortho_right: f32,
  ortho_bottom: f32,
  ortho_top: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let params = if is_orthographic {
    CameraParams::Orthographic(OrthographicCameraParams {
      left: ortho_left,
      right: ortho_right,
      bottom: ortho_bottom,
      top: ortho_top,
      near,
      far,
    })
  } else {
    CameraParams::Perspective(PerspectiveCameraParams {
      fov: fov.to_radians(),
      aspect_ratio: aspect,
      near_plane: near,
      far_plane: far,
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
  if ctx.is_null() || proj_out.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let mut arr = [0.0; 16];
  if ctx_ref
    .get_camera_component(scene_id, entity, &mut arr)
    .is_ok()
  {
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
  p1x: f32,
  p1y: f32,
  p1z: f32,
  p2x: f32,
  p2y: f32,
  p2z: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
  let _ = ctx_ref.add_measurement_component(
    scene_id,
    entity,
    Vec3f32::from_components(p1x, p1y, p1z),
    Vec3f32::from_components(p2x, p2y, p2z),
  );
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
  if ctx.is_null() {
    return;
  }
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
  if ctx.is_null() {
    return;
  }
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
  if ctx.is_null()
    || px.is_null()
    || py.is_null()
    || pz.is_null()
    || cr.is_null()
    || cg.is_null()
    || cb.is_null()
    || sizes.is_null()
  {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  use alloc::vec::Vec;

  let mut markers = Vec::with_capacity(count as usize);
  for i in 0..(count as usize) {
    markers.push(FfiMarker {
      position: [unsafe { *px.add(i) }, unsafe { *py.add(i) }, unsafe {
        *pz.add(i)
      }],
      color: [unsafe { *cr.add(i) }, unsafe { *cg.add(i) }, unsafe {
        *cb.add(i)
      }],
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
) -> *mut FfiBvhNode {
  if ctx.is_null() || count.is_null() {
    return core::ptr::null_mut();
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .get_bvh_nodes(scene_id, entity, count)
    .unwrap_or(core::ptr::null_mut())
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_freeBvhNodes(ptr: *mut FfiBvhNode, count: u32) {
  if !ptr.is_null() && count > 0 {
    let _ = unsafe { alloc::vec::Vec::from_raw_parts(ptr, count as usize, count as usize) };
  }
}

impl From<LinearBVHNode<f32>> for FfiBvhNode {
  fn from(node: LinearBVHNode<f32>) -> Self {
    let mut ffi_node = FfiBvhNode::from_offsets(
      node.left_child_or_primitive_offset,
      node.right_child_offset,
      node.primitive_count,
    );

    match &node.bound {
      linear_bvh::LinearBound::AABB(aabb) => {
        ffi_node.node_type = FfiNodeType::AABB;
        ffi_node.min_x = aabb.min::<Vec3f32>().x();
        ffi_node.min_y = aabb.min::<Vec3f32>().y();
        ffi_node.min_z = aabb.min::<Vec3f32>().z();
        ffi_node.max_x = aabb.max::<Vec3f32>().x();
        ffi_node.max_y = aabb.max::<Vec3f32>().y();
        ffi_node.max_z = aabb.max::<Vec3f32>().z();
      }
      linear_bvh::LinearBound::OBB(obb) => {
        ffi_node.node_type = FfiNodeType::OBB;
        let t: Vec3f32 = obb.translation();
        let ext: Vec3f32 = obb.half_extent();
        ffi_node.center_x = t.x();
        ffi_node.center_y = t.y();
        ffi_node.center_z = t.z();
        ffi_node.extents_x = ext.x();
        ffi_node.extents_y = ext.y();
        ffi_node.extents_z = ext.z();
      }
    }
    ffi_node
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
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.set_bvh_node_visibility(scene_id, entity, node_index, visible);
}

// --------------------- FFI Types ---------------------------

#[repr(u32)]
pub enum FfiRenderTaskStatus {
  Completed = 0,
  Pending = 1,
  Error = 2,
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

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfiRaycastResult {
  pub hit: bool,
  pub entity: u64,
  pub px: f32,
  pub py: f32,
  pub pz: f32,
}

impl From<RaycastResult> for FfiRaycastResult {
  fn from(value: RaycastResult) -> Self {
    let mut res = Self {
      hit: false,
      entity: 0,
      px: 0.0,
      py: 0.0,
      pz: 0.0,
    };

    if let Some(hit) = value {
      res.hit = true;
      res.entity = hit.entity_ext_id;
      res.px = hit.p.x();
      res.py = hit.p.y();
      res.pz = hit.p.z();
    }

    res
  }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfiKinematicState {
  pub pos_x: f32,
  pub pos_y: f32,
  pub pos_z: f32,
  pub vel_x: f32,
  pub vel_y: f32,
  pub vel_z: f32,
  pub has_rotation: bool,
  pub rot_w: f32,
  pub rot_x: f32,
  pub rot_y: f32,
  pub rot_z: f32,
  pub has_angular_velocity: bool,
  pub ang_vel_x: f32,
  pub ang_vel_y: f32,
  pub ang_vel_z: f32,
}

impl From<aethervk_core_rlib::simulation::almanac::KinematicState> for FfiKinematicState {
  fn from(value: aethervk_core_rlib::simulation::almanac::KinematicState) -> Self {
    let mut res = Self {
      pos_x: value.position.x(),
      pos_y: value.position.y(),
      pos_z: value.position.z(),
      vel_x: value.velocity.x(),
      vel_y: value.velocity.y(),
      vel_z: value.velocity.z(),
      has_rotation: false,
      rot_w: 1.0,
      rot_x: 0.0,
      rot_y: 0.0,
      rot_z: 0.0,
      has_angular_velocity: false,
      ang_vel_x: 0.0,
      ang_vel_y: 0.0,
      ang_vel_z: 0.0,
    };

    if let Some(rot) = value.rotation {
      res.has_rotation = true;
      res.rot_w = rot.0.w();
      res.rot_x = rot.0.x();
      res.rot_y = rot.0.y();
      res.rot_z = rot.0.z();
    }

    if let Some(ang_vel) = value.angular_velocity {
      res.has_angular_velocity = true;
      res.ang_vel_x = ang_vel.x();
      res.ang_vel_y = ang_vel.y();
      res.ang_vel_z = ang_vel.z();
    }

    res
  }
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

impl From<FfiMarker> for Marker {
  fn from(value: FfiMarker) -> Self {
    Self {
      local_pos: value.position,
      color: value.color,
      size: value.size,
    }
  }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FfiNodeType {
  AABB = 0,
  OBB = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
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
