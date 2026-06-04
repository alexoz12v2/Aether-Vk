//! ffi module.

use aethervk_core_rlib::{
  gpu,
  math::collision::{linear_bvh, linear_bvh::LinearBVHNode},
  scene::{ForeignSerializable, Marker},
  simulation_api::{
    comet_api::CometApi,
    components_api::{CameraParams, OrthographicCameraParams, PerspectiveCameraParams},
    structs::*,
    *,
  },
  types::EngineError,
};
use aethervk_oshal_rlib as oshal;
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{Vector3, Vector4, vec3::Vec3f32, vec4::Quat},
};
use alloc::{boxed::Box, string::ToString};
use core::{
  ffi::{CStr, c_char},
  str::FromStr,
};

pub type PanicCallback = extern "C" fn(*const u8, usize);
pub static mut PANIC_CALLBACK: Option<PanicCallback> = None;

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_registerPanicCallback(cb: PanicCallback) {
  unsafe {
    PANIC_CALLBACK = Some(cb);
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
  oshal::log!(">>> avkSimulationContext_shutdown called!");
  if !ctx.is_null() {
    oshal::log!(">>> ctx is not null, from_raw...");
    let ctx_box = unsafe { Box::from_raw(ctx) };
    oshal::log!(">>> sending shutdown...");
    let _ = ctx_box
      .threads
      .logic_thread
      .tx()
      .try_send(aethervk_core_rlib::simulation_api::structs::LogicCommand::Shutdown);
    oshal::log!(">>> shutdown sent, dropping box...");
  }
  oshal::log!(">>> avkSimulationContext_shutdown returning.");
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
  ctx_ref.get_task_status(task_id) as i32
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
pub unsafe extern "C" fn avkSimulationContext_destroyScene(
  ctx: *mut SimulationContext,
  scene_id: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.destroy_scene(scene_id);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_createEmptyScene(ctx: *mut SimulationContext) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.create_empty_scene(false).unwrap_or_else(|e| {
    oshal::log!("create_empty_scene failed: {}", e);
    0
  })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addPerspectiveCamera(
  ctx: *mut SimulationContext,
  scene_id: u64,
  presentation_engine: u64,
  name: *const c_char,
  fov: f32,
  near: f32,
  far: f32,
) -> u64 {
  if ctx.is_null() || name.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  unsafe { core::ffi::CStr::from_ptr(name) }
    .to_str()
    .map_err(|_| EngineError::InvalidNullArgument)
    .and_then(|name_str| {
      ctx_ref
        .add_perspective_camera(
          scene_id,
          gpu::PresentationEngineHandle(presentation_engine),
          name_str,
          fov.to_radians(),
          near,
          far,
        )
        .map(|id| id.get())
    })
    .unwrap_or_else(|e| {
      oshal::log!("add_perspective_camera failed: {}", e);
      0
    })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setCameraForPresentationEngine(
  ctx: *mut SimulationContext,
  scene_id: u64,
  presentation_engine: u64,
  camera_entity_id: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .set_camera_for_presentation_engine(
      scene_id,
      gpu::PresentationEngineHandle(presentation_engine),
      camera_entity_id,
    )
    .map(|_| true)
    .unwrap_or_else(|e| {
      oshal::log!("set_camera_for_presentation_engine failed: {}", e);
      false
    })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addOrthographicCamera(
  ctx: *mut SimulationContext,
  scene_id: u64,
  presentation_engine: u64,
  name: *const c_char,
  scale_factor: f32,
  near: f32,
  far: f32,
) -> u64 {
  if ctx.is_null() || name.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  unsafe { core::ffi::CStr::from_ptr(name) }
    .to_str()
    .map_err(|_| EngineError::InvalidNullArgument)
    .and_then(|name_str| {
      ctx_ref
        .add_orthographic_camera(
          scene_id,
          gpu::PresentationEngineHandle(presentation_engine),
          name_str,
          scale_factor,
          near,
          far,
        )
        .map(|id| id.get())
    })
    .unwrap_or_else(|e| {
      oshal::log!("add_orthographic_camera failed: {}", e);
      0
    })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_createPresentationEngine(
  ctx: *mut SimulationContext,
  width: u32,
  height: u32,
  scene_id: u64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .create_presentation_engine(scene_id, width, height)
    .map(|h| h.0)
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_destroyPresentationEngine(
  ctx: *mut SimulationContext,
  scene_id: u64,
  handle: u64,
) {
  if ctx.is_null() || handle == 0 {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.destroy_presentation_engine(scene_id, gpu::PresentationEngineHandle(handle));
}

/// Process pending main-thread cleanup tasks.
/// Avalonia MUST call this from its UI thread (e.g., in a DispatcherTimer tick).
/// Currently a no-op when only windowless PEs are used.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_processMainThreadCleanupQueue(
  ctx: *mut SimulationContext,
) -> i32 {
  let ctx_ref = unsafe { &*ctx };
  match ctx_ref.process_main_thread_cleanup_queue() {
    Ok(()) => 0,
    Err(_) => -1,
  }
}

/// Flush all window-tied resources before shutdown.
/// Avalonia MUST call this from its UI thread before disposing the SimulationContext.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_flushMainThreadCleanupQueue(
  ctx: *mut SimulationContext,
) -> i32 {
  let ctx_ref = unsafe { &*ctx };
  match ctx_ref.flush_main_thread_cleanup_queue() {
    Ok(()) => 0,
    Err(_) => -1,
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_resize(
  ctx: *mut SimulationContext,
  scene_id: u64,
  handle: u64,
  width: u32,
  height: u32,
) {
  if ctx.is_null() || handle == 0 {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  let _ = ctx_ref.resize(
    scene_id,
    gpu::PresentationEngineHandle(handle),
    width,
    height,
  );
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
  match ctx_ref.create_default_scene(false) {
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

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setSceneDebugName(
  ctx: *mut SimulationContext,
  scene_id: u64,
  name: *const c_char,
) {
  if ctx.is_null() || name.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let name_str = unsafe { core::ffi::CStr::from_ptr(name).to_str().unwrap_or("") };
  if let Some(scene) = ctx_ref.get_scene(scene_id) {
    scene.write().debug_name = alloc::string::String::from(name_str);
  }
}

// --- Entity Management (Async) ---

/// Prints the full scene hierarchy to the log in a tree format (debug builds only).
#[cfg(debug_assertions)]
fn debug_print_scene_hierarchy(
  ctx_ref: &aethervk_core_rlib::simulation_api::SimulationContext,
  scene_id: u64,
) {
  let scene_arc = match ctx_ref.scenes.read().get_scene(scene_id) {
    Some(s) => s,
    None => {
      oshal::log!("[Scene Hierarchy] scene {} not found", scene_id);
      return;
    }
  };
  let scene_ctx = scene_arc.read();

  // Find root entities (those with no parent).
  let all_ext_ids: alloc::vec::Vec<u64> = scene_ctx.entity_map.keys().copied().collect();
  let mut roots: alloc::vec::Vec<u64> = alloc::vec::Vec::new();
  for &ext_id in &all_ext_ids {
    if let Some(int_id) = scene_ctx.get_entity(ext_id) {
      let parent = scene_ctx.scene.get_parent(int_id);
      if parent.is_none() {
        roots.push(ext_id);
      }
    }
  }

  oshal::log!(
    "[Scene Hierarchy] scene={} total_entities={}",
    scene_id,
    all_ext_ids.len()
  );

  // DFS print helper using an explicit stack to avoid recursion in no_std.
  // Stack entries: (ext_id, depth)
  let mut stack: alloc::vec::Vec<(u64, usize)> = roots.iter().map(|&id| (id, 0)).collect();
  // Reverse so first root prints first.
  stack.reverse();

  while let Some((ext_id, depth)) = stack.pop() {
    let int_id = match scene_ctx.get_entity(ext_id) {
      Some(id) => id,
      None => continue,
    };
    let name = scene_ctx
      .scene
      .get_name(int_id)
      .unwrap_or_else(|| alloc::string::String::from("<unnamed>"));
    let components = scene_ctx.scene.get_entity_component_names(int_id);
    let indent: alloc::string::String = (0..depth).map(|_| "  ").collect();
    let comp_str = if components.is_empty() {
      alloc::string::String::from("(no components)")
    } else {
      components.join(", ")
    };
    oshal::log!(
      "{}├─ \"{}\" [ext={}] {{{}}}",
      indent,
      name,
      ext_id,
      comp_str
    );

    // Push children in reverse order so first child prints first.
    if let Some(children) = scene_ctx.scene.get_children(int_id) {
      // Map internal ids back to external ids.
      let mut child_ext: alloc::vec::Vec<u64> = alloc::vec::Vec::new();
      for child_int in &children {
        if let Some((&child_ext_id, _)) =
          scene_ctx.entity_map.iter().find(|&(_, &v)| v == *child_int)
        {
          child_ext.push(child_ext_id);
        }
      }
      child_ext.reverse();
      for cext in child_ext {
        stack.push((cext, depth + 1));
      }
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addAlmanacPlanet(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  naif_id: i32,
  offset_x: f32,
  offset_y: f32,
  offset_z: f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx) = scenes.get(&scene_id) {
    let mut scene_write = scene_ctx.write();
    let int_id = aethervk_core_rlib::scene::EntityId::from_ffi(entity_id);
    let mut planet = aethervk_core_rlib::scene::AlmanacPlanet::new(naif_id, 0.0, 0.0);
    planet.surface_offset_bf = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
      offset_x, offset_y, offset_z,
    );
    return scene_write.scene.add_component(int_id, planet).is_ok();
  }
  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setAlmanacPlanetOffset(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  offset_x: f32,
  offset_y: f32,
  offset_z: f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx) = scenes.get(&scene_id) {
    let mut scene_write = scene_ctx.write();
    let int_id = aethervk_core_rlib::scene::EntityId::from_ffi(entity_id);
    return scene_write
      .scene
      .with_component_mut(
        int_id,
        |comp: &mut aethervk_core_rlib::scene::AlmanacPlanet| {
          comp.surface_offset_bf =
            aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
              offset_x, offset_y, offset_z,
            );
        },
      )
      .is_some();
  }
  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_removeAlmanacPlanet(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx) = scenes.get(&scene_id) {
    let mut scene_write = scene_ctx.write();
    let int_id = aethervk_core_rlib::scene::EntityId::from_ffi(entity_id);
    return scene_write
      .scene
      .remove_component::<aethervk_core_rlib::scene::AlmanacPlanet>(int_id)
      .is_ok();
  }
  false
}

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
  let result = ctx_ref.spawn_entity(scene_id, name_str).unwrap_or_else(|e| {
    oshal::log!("spawn_entity failed: {}", e);
    0
  });

  #[cfg(debug_assertions)]
  debug_print_scene_hierarchy(ctx_ref, scene_id);

  result
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SceneHierarchyNodeDTO {
  pub entity_id: u64,
  pub parent_id: u64, // 0 denotes a root entity with no parent
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getSceneHierarchy(
  ctx: *mut SimulationContext,
  scene_id: u64,
  out_buffer: *mut SceneHierarchyNodeDTO,
  capacity: u32,
  out_count: *mut u32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  if let Some(scene_ctx_rw) = ctx_ref.get_scene(scene_id) {
    let scene_ctx = scene_ctx_rw.read();
    let count = scene_ctx.entity_map.len() as u32;
    if !out_count.is_null() {
      unsafe {
        *out_count = count;
      }
    }
    if capacity < count || out_buffer.is_null() {
      return false;
    }

    let mut reverse_map = alloc::collections::BTreeMap::new();
    for (&ext_id, &int_id) in scene_ctx.entity_map.iter() {
      reverse_map.insert(int_id, ext_id);
    }

    let mut idx = 0;
    for (&ext_id, &int_id) in scene_ctx.entity_map.iter() {
      // Scene has a method to get parent directly, or we can add it
      let parent_internal = scene_ctx.scene.get_parent(int_id);
      let parent_id = parent_internal.and_then(|pid| reverse_map.get(&pid).copied()).unwrap_or(0);
      let dto = SceneHierarchyNodeDTO {
        entity_id: ext_id,
        parent_id,
      };
      unsafe {
        out_buffer.add(idx).write(dto);
      }
      idx += 1;
    }
    return true;
  }
  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  component_id: u64,
  out_buffer: *mut core::ffi::c_void,
) -> bool {
  if ctx.is_null() || out_buffer.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx_rw) = scenes.get(&scene_id) {
    let scene_read = scene_ctx_rw.read();
    if let Some(int_id) = scene_read.get_entity(entity_id) {
      match component_id {
        aethervk_core_rlib::scene::PhysicalMeshComponent::COMPONENT_ID => {
          let mut found = false;
          let _ = scene_read.scene.with_component(int_id, |comp: &aethervk_core_rlib::scene::PhysicalMeshComponent| {
            let foreign = <aethervk_core_rlib::scene::PhysicalMeshComponent as aethervk_core_rlib::scene::ForeignSerializable>::to_foreign(comp);
            unsafe {
              core::ptr::copy_nonoverlapping(&foreign as *const _ as *const u8, out_buffer as *mut u8, core::mem::size_of_val(&foreign));
            }
            found = true;
          });
          if found {
            return true;
          }
        }
        aethervk_core_rlib::scene::TransformComponent::COMPONENT_ID => {
          let mut found = false;
          let _ = scene_read.scene.with_component(int_id, |comp: &aethervk_core_rlib::scene::TransformComponent| {
            let foreign = <aethervk_core_rlib::scene::TransformComponent as aethervk_core_rlib::scene::ForeignSerializable>::to_foreign(comp);
            unsafe {
              core::ptr::copy_nonoverlapping(&foreign as *const _ as *const u8, out_buffer as *mut u8, core::mem::size_of_val(&foreign));
            }
            found = true;
          });
          if found {
            return true;
          }
        }
        aethervk_core_rlib::scene::HighResTransformComponent::COMPONENT_ID => {
          let mut found = false;
          let _ = scene_read.scene.with_component(int_id, |comp: &aethervk_core_rlib::scene::HighResTransformComponent| {
            let foreign = <aethervk_core_rlib::scene::HighResTransformComponent as aethervk_core_rlib::scene::ForeignSerializable>::to_foreign(comp);
            unsafe {
              core::ptr::copy_nonoverlapping(&foreign as *const _ as *const u8, out_buffer as *mut u8, core::mem::size_of_val(&foreign));
            }
            found = true;
          });
          if found {
            return true;
          }
        }
        aethervk_core_rlib::scene::CameraComponent::COMPONENT_ID => {
          let mut found = false;
          let _ = scene_read.scene.with_component(int_id, |comp: &aethervk_core_rlib::scene::CameraComponent| {
            let foreign = <aethervk_core_rlib::scene::CameraComponent as aethervk_core_rlib::scene::ForeignSerializable>::to_foreign(comp);
            unsafe {
              core::ptr::copy_nonoverlapping(&foreign as *const _ as *const u8, out_buffer as *mut u8, core::mem::size_of_val(&foreign));
            }
            found = true;
          });
          if found {
            return true;
          }
        }
        aethervk_core_rlib::scene::SphereGizmoComponent::COMPONENT_ID => {
          let mut found = false;
          let _ = scene_read.scene.with_component(int_id, |comp: &aethervk_core_rlib::scene::SphereGizmoComponent| {
            let foreign = <aethervk_core_rlib::scene::SphereGizmoComponent as aethervk_core_rlib::scene::ForeignSerializable>::to_foreign(comp);
            unsafe {
              core::ptr::copy_nonoverlapping(&foreign as *const _ as *const u8, out_buffer as *mut u8, core::mem::size_of_val(&foreign));
            }
            found = true;
          });
          if found {
            return true;
          }
        }
        aethervk_core_rlib::scene::ScreenSpaceBillboardComponent::COMPONENT_ID => {
          let mut found = false;
          let _ = scene_read.scene.with_component(int_id, |comp: &aethervk_core_rlib::scene::ScreenSpaceBillboardComponent| {
            let foreign = <aethervk_core_rlib::scene::ScreenSpaceBillboardComponent as aethervk_core_rlib::scene::ForeignSerializable>::to_foreign(comp);
            unsafe {
              core::ptr::copy_nonoverlapping(&foreign as *const _ as *const u8, out_buffer as *mut u8, core::mem::size_of_val(&foreign));
            }
            found = true;
          });
          if found {
            return true;
          }
        }
        _ => {}
      }
    }
  }
  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  component_id: u64,
  in_buffer: *const core::ffi::c_void,
) -> bool {
  if ctx.is_null() || in_buffer.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx_rw) = scenes.get(&scene_id) {
    let mut scene_write = scene_ctx_rw.write();
    if let Some(int_id) = scene_write.get_entity(entity_id) {
      match component_id {
        aethervk_core_rlib::scene::PhysicalMeshComponent::COMPONENT_ID => {
          let res = scene_write.scene.with_component_mut(int_id, |comp: &mut aethervk_core_rlib::scene::PhysicalMeshComponent| {
            unsafe { <aethervk_core_rlib::scene::PhysicalMeshComponent as aethervk_core_rlib::scene::ForeignSerializable>::apply_foreign_ptr(comp, in_buffer); }
          });
          if res.is_some() {
            return true;
          }
        }
        aethervk_core_rlib::scene::TransformComponent::COMPONENT_ID => {
          let res = scene_write.scene.with_component_mut(int_id, |comp: &mut aethervk_core_rlib::scene::TransformComponent| {
            unsafe { <aethervk_core_rlib::scene::TransformComponent as aethervk_core_rlib::scene::ForeignSerializable>::apply_foreign_ptr(comp, in_buffer); }
          });
          if res.is_some() {
            return true;
          }
        }
        aethervk_core_rlib::scene::HighResTransformComponent::COMPONENT_ID => {
          let res = scene_write.scene.with_component_mut(int_id, |comp: &mut aethervk_core_rlib::scene::HighResTransformComponent| {
            unsafe { <aethervk_core_rlib::scene::HighResTransformComponent as aethervk_core_rlib::scene::ForeignSerializable>::apply_foreign_ptr(comp, in_buffer); }
          });
          if res.is_some() {
            return true;
          }
        }
        aethervk_core_rlib::scene::CameraComponent::COMPONENT_ID => {
          let res = scene_write.scene.with_component_mut(int_id, |comp: &mut aethervk_core_rlib::scene::CameraComponent| {
            unsafe { <aethervk_core_rlib::scene::CameraComponent as aethervk_core_rlib::scene::ForeignSerializable>::apply_foreign_ptr(comp, in_buffer); }
          });
          if res.is_some() {
            return true;
          }
        }
        aethervk_core_rlib::scene::SphereGizmoComponent::COMPONENT_ID => {
          let res = scene_write.scene.with_component_mut(int_id, |comp: &mut aethervk_core_rlib::scene::SphereGizmoComponent| {
            unsafe { <aethervk_core_rlib::scene::SphereGizmoComponent as aethervk_core_rlib::scene::ForeignSerializable>::apply_foreign_ptr(comp, in_buffer); }
          });
          if res.is_some() {
            return true;
          }
        }
        aethervk_core_rlib::scene::ScreenSpaceBillboardComponent::COMPONENT_ID => {
          let res = scene_write.scene.with_component_mut(int_id, |comp: &mut aethervk_core_rlib::scene::ScreenSpaceBillboardComponent| {
            unsafe { <aethervk_core_rlib::scene::ScreenSpaceBillboardComponent as aethervk_core_rlib::scene::ForeignSerializable>::apply_foreign_ptr(comp, in_buffer); }
          });
          if res.is_some() {
            return true;
          }
        }
        _ => {}
      }
    }
  }
  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setPhysicalMeshEmissiveColor(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  r: f32,
  g: f32,
  b: f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx_rw) = scenes.get(&scene_id) {
    let mut scene_write = scene_ctx_rw.write();
    if let Some(int_id) = scene_write.get_entity(entity_id) {
      let mut found = false;
      let _ = scene_write.scene.with_component_mut(
        int_id,
        |comp: &mut aethervk_core_rlib::scene::PhysicalMeshComponent| {
          comp.emissive_color = [r, g, b];
          comp.emissive_intensity = 1.0;
          found = true;
        },
      );
      if !found {
        let _ = scene_write.scene.with_component_mut(
          int_id,
          |comp: &mut aethervk_core_rlib::scene::StaticMeshComponent| {
            comp.emissive_color = [r, g, b, 1.0];
            found = true;
          },
        );
      }
      return found;
    }
  }
  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnBillboard(
  ctx: *mut SimulationContext,
  scene_id: u64,
  image_path: *const c_char,
  out_entity_id: *mut u64,
) -> bool {
  if ctx.is_null() || image_path.is_null() || out_entity_id.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let _image_path_str = unsafe { CStr::from_ptr(image_path).to_str().unwrap_or("") };

  // Screen-space billboards are rendered by the C# Avalonia overlay, not the Vulkan
  // pipeline. The entity only needs a ScreenSpaceBillboardComponent (added by the
  // C# caller after this returns) — no TransformComponent or ImageBillboardComponent.
  if let Ok(entity_id) = ctx_ref.spawn_entity(scene_id, "Billboard") {
    // Parent billboard entity to the scene root for proper hierarchy.
    // The billboard ignores transforms (no TransformComponent) but should still
    // appear as a child of root in the scene outline.
    let scenes = ctx_ref.scenes.read();
    if let Some(scene_ctx) = scenes.get(&scene_id) {
      let scene_read = scene_ctx.read();
      if let Some(root_id) = scene_read.scene.get_root() {
        let root_ext_id =
          scene_read.entity_map.iter().find(|&(_, &v)| v == root_id).map(|(&k, _)| k);
        if let Some(parent_ext) = root_ext_id {
          drop(scene_read);
          drop(scenes);
          let _ = ctx_ref.set_parent(scene_id, entity_id, parent_ext);
        }
      }
    }
    unsafe {
      *out_entity_id = entity_id;
    }
    return true;
  }
  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnComet(
  ctx: *mut SimulationContext,
  scene_id: u64,
  model_id: u64,
  name: *const c_char,
  pos_x: f32,
  pos_y: f32,
  pos_z: f32,
  rot_w: f32,
  rot_x: f32,
  rot_y: f32,
  rot_z: f32,
  radius_km: f32,
  mass_kg: f32,
  physics_type: u32,
  naif_id: i32,
  // IAU rotational model parameters
  pole_ra_deg: f64,
  pole_dec_deg: f64,
  prime_meridian_deg: f64,
  pole_ra_rate_deg: f64,
  pole_dec_rate_deg: f64,
  rotation_rate_deg: f64,
  // Angular velocity (rad/s)
  angular_vel_x: f32,
  angular_vel_y: f32,
  angular_vel_z: f32,
  out_result: *mut FfiSpawnCometResult,
) -> bool {
  if ctx.is_null() || name.is_null() || out_result.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let name_str = unsafe { CStr::from_ptr(name).to_str().unwrap_or("Comet") };

  // Build rotational model if any non-default values were provided
  let has_rotational_data = pole_ra_deg.abs() > 1e-12
    || (pole_dec_deg - 90.0).abs() > 1e-12
    || prime_meridian_deg.abs() > 1e-12
    || rotation_rate_deg.abs() > 1e-12
    || pole_ra_rate_deg.abs() > 1e-12
    || pole_dec_rate_deg.abs() > 1e-12;

  let rotational_model = if has_rotational_data {
    Some(aethervk_core_rlib::scene::BodyRotationalModel {
      pole_ra: pole_ra_deg,
      pole_dec: pole_dec_deg,
      prime_meridian: prime_meridian_deg,
      pole_ra_rate: pole_ra_rate_deg,
      pole_dec_rate: pole_dec_rate_deg,
      rotation_rate: rotation_rate_deg,
      reference_epoch_jd: 2451545.0, // J2000.0
    })
  } else {
    None
  };

  match ctx_ref.spawn_comet_internal(
    scene_id,
    model_id,
    name_str,
    Vec3f32::from_components(pos_x, pos_y, pos_z),
    Quat::from_components(rot_x, rot_y, rot_z, rot_w),
    radius_km,
    mass_kg,
    physics_type,
    naif_id,
    rotational_model,
    Vec3f32::from_components(angular_vel_x, angular_vel_y, angular_vel_z),
  ) {
    Ok((micro_id, comet_id)) => {
      unsafe {
        core::ptr::write_unaligned(
          out_result,
          FfiSpawnCometResult {
            lca_frame_id: micro_id,
            comet_entity_id: comet_id,
          },
        );
      }
      true
    }
    Err(e) => {
      oshal::log!("avkSimulationContext_spawnComet failed: {:?}", e);
      false
    }
  }
}

/// Updates the IAU rotational model on a comet's `PhysicalMeshComponent` and recomputes
/// the `TransformComponent` rotation from the new model parameters.
///
/// Called from C# when the user edits the RotationalModelEditor in the properties panel.
/// After updating the model, this function:
/// 1. Stores the new `BodyRotationalModel` on the `PhysicalMeshComponent`.
/// 2. Evaluates the orientation at the given Julian Date.
/// 3. Applies the `bf_to_pa` correction and writes the quaternion to `TransformComponent`.
/// 4. Triggers `recalculateJetPoints` to update child emitter positions on the rotated surface.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setRotationalModel(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  pole_ra_deg: f64,
  pole_dec_deg: f64,
  prime_meridian_deg: f64,
  pole_ra_rate_deg: f64,
  pole_dec_rate_deg: f64,
  rotation_rate_deg: f64,
  reference_epoch_jd: f64,
  current_jd: f64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  let scene_ctx_lock = match scenes.get_scene(scene_id) {
    Some(s) => s,
    None => return false,
  };
  let mut scene_ctx = scene_ctx_lock.write();
  let internal_id = match scene_ctx.entity_map.get(&entity_id).copied() {
    Some(id) => id,
    None => return false,
  };

  let model = aethervk_core_rlib::scene::BodyRotationalModel {
    pole_ra: pole_ra_deg,
    pole_dec: pole_dec_deg,
    prime_meridian: prime_meridian_deg,
    pole_ra_rate: pole_ra_rate_deg,
    pole_dec_rate: pole_dec_rate_deg,
    rotation_rate: rotation_rate_deg,
    reference_epoch_jd,
  };

  // 1. Update the rotational model on PhysicalMeshComponent
  let mut bf_to_pa = aethervk_oshal_rlib::math::vector::vec4::Quat::identity();
  let _ = scene_ctx.scene.with_component_mut(
    internal_id,
    |c: &mut aethervk_core_rlib::scene::PhysicalMeshComponent| {
      if let Some(q) = c.mesh.bf_to_pa {
        bf_to_pa = q;
      }
      c.rotational_model = Some(model);
    },
  );

  // 2. Compute orientation from IAU model at the current simulation time
  let iau_quat = model.orientation_at(current_jd);
  // Convert from IAU body frame to object/PA frame
  let sim_rotation = (iau_quat * bf_to_pa.inverse()).normalize();

  // 3. Update TransformComponent rotation (preserving position and scale)
  let _ = scene_ctx.scene.with_component_mut(
    internal_id,
    |tc: &mut aethervk_core_rlib::scene::TransformComponent| {
      tc.rotation = sim_rotation;
    },
  );

  // Note: recalculateJetPoints is called separately from C# after this returns
  true
}

/// Synchronizes `ColliderComponent.shape.radius` and `SphereGizmoComponent.radius`
/// with the given `new_radius_km`. Called from C# when `PhysicalMeshComponent.RadiusKm`
/// changes, so the physics collider and visual gizmo stay in sync.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_syncColliderRadius(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  new_radius_km: f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  let scene_ctx_lock = match scenes.get_scene(scene_id) {
    Some(s) => s,
    None => return false,
  };
  let scene_ctx = scene_ctx_lock.read();
  let internal_id = match scene_ctx.entity_map.get(&entity_id).copied() {
    Some(id) => id,
    None => return false,
  };

  // 1. Update ColliderComponent sphere radius (preserving mass, restitution, friction)
  let _ = scene_ctx.scene.with_component_mut(
    internal_id,
    |collider: &mut aethervk_core_rlib::scene::ColliderComponent| {
      collider.shape = aethervk_core_rlib::scene::ColliderShape::Sphere { radius: new_radius_km };
    },
  );

  // 2. Update SphereGizmoComponent radius
  let _ = scene_ctx.scene.with_component_mut(
    internal_id,
    |gizmo: &mut aethervk_core_rlib::scene::SphereGizmoComponent| {
      gizmo.radius = new_radius_km;
    },
  );

  // Mark the static TLAS as dirty so selection raycasts pick up the new bounds
  scene_ctx.is_static_tlas_dirty.store(true, core::sync::atomic::Ordering::Relaxed);

  true
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnTrajectory(
  ctx: *mut SimulationContext,
  scene_id: u64,
  parent_entity: u64,
  name: *const c_char,
  trajectory: *mut aethervk_core_rlib::gpu::TrajectoryGpu,
  segments: *const aethervk_core_rlib::gpu::RationalBezierGpu,
  segment_count: u32,
) -> u64 {
  if ctx.is_null()
    || name.is_null()
    || trajectory.is_null()
    || segments.is_null()
    || segment_count == 0
  {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let name_str = unsafe { CStr::from_ptr(name).to_str().unwrap_or("Trajectory") };
  let traj = unsafe { *trajectory };
  let segments_slice = unsafe { core::slice::from_raw_parts(segments, segment_count as usize) };

  oshal::log!(
    "avkSimulationContext_spawnTrajectory called! Name: {}",
    name_str
  );
  oshal::log!(
    "  -> Color: {:?}, LineWidth: {}",
    traj.color,
    traj.line_width
  );
  if segment_count > 0 {
    oshal::log!("  -> First Segment CP0: {:?}", segments_slice[0].cp0);
  }

  match ctx_ref.spawn_trajectory_internal(scene_id, parent_entity, name_str, traj, segments_slice) {
    Ok(entity_id) => {
      oshal::log!("  -> Successfully spawned Trajectory Entity: {}", entity_id);
      entity_id
    }
    Err(e) => {
      oshal::log!("avkSimulationContext_spawnTrajectory failed: {:?}", e);
      0
    }
  }
}

/// Spawns a Keplerian ellipse trajectory entity from raw osculating orbital elements,
/// building the 4-segment rational cubic Bézier approximation internally.
///
/// Parameters:
///   a_au        – semi-major axis in AU
///   e           – eccentricity (must be < 1 for an ellipse)
///   i_deg       – inclination in degrees
///   omega_deg   – longitude of ascending node Ω in degrees
///   argperi_deg – argument of periapsis ω in degrees
///   r, g, b, a  – trajectory line colour (0–1)
///   line_width  – line width in pixels
///
/// Returns the external entity id, or 0 on failure.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnTrajectoryFromElements(
  ctx: *mut SimulationContext,
  scene_id: u64,
  parent_entity: u64,
  name: *const c_char,
  a_au: f64,
  e: f64,
  i_deg: f64,
  omega_deg: f64,
  argperi_deg: f64,
  col_r: f32,
  col_g: f32,
  col_b: f32,
  col_a: f32,
  line_width: f32,
) -> u64 {
  if ctx.is_null() || name.is_null() || e >= 1.0 || a_au <= 0.0 {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let name_str = unsafe { CStr::from_ptr(name).to_str().unwrap_or("orbit") };

  oshal::log!(
    "avkSimulationContext_spawnTrajectoryFromElements called! Name: {}",
    name_str
  );
  oshal::log!(
    "  -> Elements: a={}, e={}, i={}, omega={}, w={}",
    a_au,
    e,
    i_deg,
    omega_deg,
    argperi_deg
  );
  oshal::log!(
    "  -> Color: [{}, {}, {}, {}], LineWidth: {}",
    col_r,
    col_g,
    col_b,
    col_a,
    line_width
  );

  // ── Keplerian ellipse → 4-segment rational cubic Bézier ──────────────────
  // Standard NURBS-circle approximation adapted for an ellipse.
  const PI: f64 = core::f64::consts::PI;

  let a = a_au;
  let b = a * (1.0_f64 - e * e).max(0.0).sqrt();
  let c = a * e; // focus offset (shift ellipse so focus is at origin)

  // Bézier weight for the middle control points of a 90° conic arc
  let w_inner: f64 = (1.0 + 2.0_f64.sqrt()) / 3.0;
  let k: f64 = 2.0 - 2.0_f64.sqrt();

  // 4 quadrant arcs (perifocal frame, focus at origin via -c shift)
  //   Each arc: [CP0, CP1, CP2, CP3] where CP = [x, y, weight]
  let quads: [[[f64; 3]; 4]; 4] = [
    [
      [a, 0.0, 1.0],
      [a, b * k, w_inner],
      [a * k, b, w_inner],
      [0.0, b, 1.0],
    ],
    [
      [0.0, b, 1.0],
      [-a * k, b, w_inner],
      [-a, b * k, w_inner],
      [-a, 0.0, 1.0],
    ],
    [
      [-a, 0.0, 1.0],
      [-a, -b * k, w_inner],
      [-a * k, -b, w_inner],
      [0.0, -b, 1.0],
    ],
    [
      [0.0, -b, 1.0],
      [a * k, -b, w_inner],
      [a, -b * k, w_inner],
      [a, 0.0, 1.0],
    ],
  ];

  let i = i_deg * PI / 180.0;
  let Om = omega_deg * PI / 180.0;
  let w = argperi_deg * PI / 180.0;

  let cos_Om = Om.cos();
  let sin_Om = Om.sin();
  let cos_i = i.cos();
  let sin_i = i.sin();
  let cos_w = w.cos();
  let sin_w = w.sin();

  // Rotation matrix rows (perifocal → ecliptic J2000)
  let rxx = cos_Om * cos_w - sin_Om * sin_w * cos_i;
  let rxy = -cos_Om * sin_w - sin_Om * cos_w * cos_i;
  let ryx = sin_Om * cos_w + cos_Om * sin_w * cos_i;
  let ryy = -sin_Om * sin_w + cos_Om * cos_w * cos_i;
  let rzx = sin_w * sin_i;
  let rzy = cos_w * sin_i;

  let mut segs: [aethervk_core_rlib::gpu::RationalBezierGpu; 4] =
    [aethervk_core_rlib::gpu::RationalBezierGpu {
      cp0: [0.0; 4],
      cp1: [0.0; 4],
      cp2: [0.0; 4],
      cp3: [0.0; 4],
    }; 4];

  for (qi, quad) in quads.iter().enumerate() {
    let cps = [
      &mut segs[qi].cp0,
      &mut segs[qi].cp1,
      &mut segs[qi].cp2,
      &mut segs[qi].cp3,
    ];
    for (pi, cp) in cps.into_iter().enumerate() {
      let xp = quad[pi][0] - c; // shift: focus at origin
      let yp = quad[pi][1];
      let wt = quad[pi][2] as f32;

      // Rotate to ecliptic, then pre-multiply by weight (homogeneous form)
      let xe = (rxx * xp + rxy * yp) as f32 * wt;
      let ye = (ryx * xp + ryy * yp) as f32 * wt;
      let ze = (rzx * xp + rzy * yp) as f32 * wt;

      *cp = [xe, ye, ze, wt];
    }
  }

  let traj = aethervk_core_rlib::gpu::TrajectoryGpu {
    segments_ptr: 0,
    _pad0: 0,
    color: [col_r, col_g, col_b, col_a],
    line_width,
    texture_id: 0xFFFF_FFFF,
    _pad1: 0,
  };

  match ctx_ref.spawn_trajectory_internal(scene_id, parent_entity, name_str, traj, &segs) {
    Ok(entity_id) => {
      oshal::log!(
        "  -> Successfully spawned TrajectoryFromElements Entity: {}",
        entity_id
      );
      entity_id
    }
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_spawnTrajectoryFromElements failed: {:?}",
        e
      );
      0
    }
  }
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
      Quat::from_components(rot_x, rot_y, rot_z, rot_w),
      Vec3f32::from_components(scale_x, scale_y, scale_z),
    )
    .is_ok()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addHighResTransformComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  pos_x: f64,
  pos_y: f64,
  pos_z: f64,
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
    .add_highres_transform_component(
      scene_id, entity, pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z, scale_x, scale_y, scale_z,
    )
    .is_ok()
}

#[repr(C)]
pub struct FfiEmissionCircle {
  pub latitude_rad: f32,
  pub longitude_rad: f32,
  pub circle_radius_km: f32,
  pub mass: f32,
  pub color_r: f32,
  pub color_g: f32,
  pub color_b: f32,
  pub color_a: f32,
  pub particles_per_tick: u32,
  pub ttl: u64,
  pub mean_velocity: f32,
  pub velocity_std_dev: f32,
  pub child_entity: u64,
  pub beta: f32,
  pub max_particles: u32,
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setParticleEmitterCirclesComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  circles: *const FfiEmissionCircle,
  count: u32,
) -> bool {
  if ctx.is_null() || (count > 0 && circles.is_null()) {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };

  let slice = if count > 0 {
    unsafe { core::slice::from_raw_parts(circles, count as usize) }
  } else {
    &[]
  };

  let mut rust_circles = alloc::vec::Vec::with_capacity(slice.len());
  for c in slice {
    rust_circles.push(aethervk_core_rlib::scene::EmissionCircle {
      latitude_rad: c.latitude_rad,
      longitude_rad: c.longitude_rad,
      circle_radius_km: c.circle_radius_km,
      mass: c.mass,
      color: [c.color_r, c.color_g, c.color_b, c.color_a],

      cached_point: None,
      cached_normal: None,
      particles_per_tick: c.particles_per_tick,
      ttl: c.ttl,
      mean_velocity: c.mean_velocity,
      velocity_std_dev: c.velocity_std_dev,
      child_entity: if c.child_entity == 0 || c.child_entity == u64::MAX {
        None
      } else {
        Some(aethervk_core_rlib::scene::EntityId::from_ffi(
          c.child_entity,
        ))
      },
      beta: c.beta,
      max_particles: c.max_particles.max(64),
    });
  }

  ctx_ref
    .set_particle_emitter_circles_component(scene_id, entity, rust_circles)
    .is_ok()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_recalculateJetPoints(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
) -> bool {
  use aethervk_oshal_rlib::math::vector::{Vector, Vector3, vec3::Vec3f32};

  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  let scene_ctx_lock = match scenes.get_scene(scene_id) {
    Some(s) => s,
    None => return false,
  };
  let mut scene_ctx = scene_ctx_lock.write();
  let internal_id = match scene_ctx.entity_map.get(&entity_id).copied() {
    Some(id) => id,
    None => return false,
  };

  // We need to fetch both components, but Rust's borrow checker prevents mutably borrowing one while immutably borrowing another from the same struct directly if not split.
  // Actually, we can get them sequentially or use a BVH/Mesh clone if needed, but `mesh` is Arc<Comet> so we can just clone the Arc.
  let mut mesh_arc_opt = None;
  let _ = scene_ctx.scene.with_component(
    internal_id,
    |c: &aethervk_core_rlib::scene::PhysicalMeshComponent| {
      mesh_arc_opt = Some(c.mesh.clone());
    },
  );
  let mesh_arc = match mesh_arc_opt {
    Some(m) => m,
    None => return false,
  };

  let mut comet_scale = 1.0;
  let mut comet_rotation = aethervk_oshal_rlib::math::vector::vec4::Quat::identity();
  let _ = scene_ctx.scene.with_component(
    internal_id,
    |c: &aethervk_core_rlib::scene::TransformComponent| {
      comet_scale = c.scale.x();
      comet_rotation = c.rotation;
    },
  );

  // Compute bounding sphere radius from mesh vertices (max distance from origin)
  let r = mesh_arc
    .vertices
    .iter()
    .map(|v| {
      let p = v.position;
      (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
    })
    .fold(0.0f32, f32::max)
    .max(0.01);

  let mut to_spawn = alloc::vec::Vec::new();
  let mut updates = alloc::vec::Vec::new();

  let _ = scene_ctx.scene.with_component_mut(
    internal_id,
    |circles_comp: &mut aethervk_core_rlib::scene::ParticleEmitterCirclesComponent| {
      for (i, circle) in circles_comp.circles.iter_mut().enumerate() {
        let lat = circle.latitude_rad;
        let lon = circle.longitude_rad;

        // Direction vector from center (spherical coordinates relative to +Z)
        // In our convention: Z is up. Latitude is from XY plane to Z. Longitude is around Z from X.
        let dir_z = lat.sin();
        let dir_x = lat.cos() * lon.cos();
        let dir_y = lat.cos() * lon.sin();
        let dir_user = Vec3f32::from_components(dir_x, dir_y, dir_z);
        // Transform direction from user/IAU frame to object space for raycasting against mesh vertices
        let dir = comet_rotation.inverse().rotate_vector(dir_user);

        // Cast a ray from far away towards the origin.
        let start_dist = r * 2.0;
        let origin = dir * start_dist;
        let ray_dir = dir * -1.0;


        let (pt, norm) = if let Some(ref bvh) = mesh_arc.bvh {
          match bvh.raycast(origin, ray_dir, &mesh_arc.vertices, &mesh_arc.indices) {
            Some((_t, hit_pt, hit_normal)) => {
              ([hit_pt.x(), hit_pt.y(), hit_pt.z()],
               [hit_normal.x(), hit_normal.y(), hit_normal.z()])
            }
            None => {
              // Fallback to bounding sphere
              let hit_pt = dir * r;
              ([hit_pt.x(), hit_pt.y(), hit_pt.z()],
               [dir.x(), dir.y(), dir.z()])
            }
          }
        } else {
          // No BVH available — fallback to bounding sphere
          let hit_pt = dir * r;
          ([hit_pt.x(), hit_pt.y(), hit_pt.z()],
           [dir.x(), dir.y(), dir.z()])
        };
        circle.cached_point = Some(pt);
        circle.cached_normal = Some(norm);

        let pt_vec = Vec3f32::from_components(pt[0], pt[1], pt[2]);
        let scale = (circle.circle_radius_km / comet_scale).max(1e-4);
        let scale_vec = Vec3f32::from_components(scale, scale, scale);
        let t = aethervk_core_rlib::scene::TransformComponent {
          position: pt_vec,
          rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
          scale: scale_vec,
        };

        if circle.child_entity.is_none() {
          to_spawn.push((i, t, circle.color, scale, circle.max_particles));
        } else {
          updates.push((circle.child_entity.unwrap(), t, circle.color));
        }
      }
    },
  );

  for (idx, (i, t, color, gizmo_radius, max_p)) in to_spawn.into_iter().enumerate() {
    let new_id = scene_ctx.scene.spawn_entity("EmissionSphere");
    scene_ctx.scene.set_parent(new_id, Some(internal_id));

    let static_mesh = aethervk_core_rlib::scene::StaticMeshComponent {
      asset_path: "primitives/sphere.obj".into(),
      mesh: alloc::sync::Arc::from(aethervk_core_rlib::simulation::comet::generate_uv_sphere(
        1.0, 6, 6, 0.0,
      )),
      emissive_color: [color[0], color[1], color[2], color[3]],
      is_visible: true,
    };

    let gizmo = aethervk_core_rlib::scene::SphereGizmoComponent {
      radius: gizmo_radius,
      subdivisions: 3.0,
      local_frame: {
        use aethervk_oshal_rlib::math::matrix::SquareMatrix;
        aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::identity()
      },
      is_visible: true,
    };

    let _ = scene_ctx.scene.add_component(new_id, static_mesh);
    let _ = scene_ctx.scene.add_component(new_id, t);
    let _ = scene_ctx.scene.add_component(new_id, gizmo);
    let _ = scene_ctx.scene.add_component(
      new_id,
      aethervk_core_rlib::scene::particles::ParticleSystemComponent::new(
        max_p as usize
      ),
    );

    // Register the entity so it gets an external ID for C# to reference
    let _ext_id = scene_ctx.register_entity(new_id);

    let _ = scene_ctx.scene.with_component_mut(
      internal_id,
      |circles_comp: &mut aethervk_core_rlib::scene::ParticleEmitterCirclesComponent| {
        circles_comp.circles[i].child_entity = Some(new_id);
      },
    );
  }

  for (child_id, t, color) in updates {
    let _ = scene_ctx.scene.with_component_mut(
      child_id,
      |tc: &mut aethervk_core_rlib::scene::TransformComponent| {
        *tc = t;
      },
    );
    let _ = scene_ctx.scene.with_component_mut(
      child_id,
      |sm: &mut aethervk_core_rlib::scene::StaticMeshComponent| {
        sm.emissive_color = [color[0], color[1], color[2], color[3]];
      },
    );
  }

  true
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getParticleEmitterCirclesComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  out_circles: *mut FfiEmissionCircle,
  max_count: u32,
  out_actual_count: *mut u32,
) -> bool {
  if ctx.is_null() || out_actual_count.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };

  // Single lock scope to avoid re-entrant spin::RwLock deadlock
  let scenes = ctx_ref.scenes.read();
  let scene_ctx_lock = match scenes.get_scene(scene_id) {
    Some(s) => s,
    None => return false,
  };
  let scene_ctx = scene_ctx_lock.read();
  let internal_id = match scene_ctx.entity_map.get(&entity).copied() {
    Some(id) => id,
    None => return false,
  };

  let mut circles_opt = None;
  let _ = scene_ctx.scene.with_component(
    internal_id,
    |c: &aethervk_core_rlib::scene::ParticleEmitterCirclesComponent| {
      circles_opt = Some(c.circles.clone());
    },
  );
  let circles = match circles_opt {
    Some(c) => c,
    None => return false,
  };

  let actual_count = circles.len() as u32;
  unsafe {
    *out_actual_count = actual_count;
  }

  if !out_circles.is_null() && max_count > 0 {
    let copy_count = core::cmp::min(max_count, actual_count) as usize;
    let out_slice = unsafe { core::slice::from_raw_parts_mut(out_circles, copy_count) };
    for i in 0..copy_count {
      let c = &circles[i];
      // Resolve internal EntityId to registered external ID for C#
      let ext_child = c.child_entity.map_or(u64::MAX, |internal_id| {
        scene_ctx.entity_map
          .iter()
          .find(|(_, v)| **v == internal_id)
          .map_or(u64::MAX, |(&k, _)| k)
      });
      out_slice[i] = FfiEmissionCircle {
        latitude_rad: c.latitude_rad,
        longitude_rad: c.longitude_rad,
        circle_radius_km: c.circle_radius_km,
        mass: c.mass,
        color_r: c.color[0],
        color_g: c.color[1],
        color_b: c.color[2],
        color_a: c.color[3],
        particles_per_tick: c.particles_per_tick,
        ttl: c.ttl,
        mean_velocity: c.mean_velocity,
        velocity_std_dev: c.velocity_std_dev,
        child_entity: ext_child,
        beta: c.beta,
        max_particles: c.max_particles,
      };
    }
  }
  true
}

/// Patches the `naif_id` field of an `AlmanacPlanet` component on a Kinematic comet entity
/// after spawn. At spawn time `naif_id` is set to 0 (placeholder); this function injects
/// the real SPK/NAIF id so the logic thread can query the almanac correctly each tick.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setAlmanacPlanetNaifId(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  naif_id: i32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  if let Some(scenes) = ctx_ref.scenes.try_write() {
    if let Some(scene_ctx_lock) = scenes.scenes.get(&scene_id) {
      let mut scene_ctx = scene_ctx_lock.write();
      if let Some(internal_id) = scene_ctx.entity_map.get(&entity_id).copied() {
        let mut found = false;
        let _ = scene_ctx.scene.with_component_mut(
          internal_id,
          |planet: &mut aethervk_core_rlib::scene::AlmanacPlanet| {
            planet.naif_id = naif_id;
            found = true;
          },
        );
        return found;
      }
    }
  }
  false
}

/// Sets the initial velocity (km/s in ecliptic J2000) on a Dynamic comet entity's
/// `KinematicComponent`. Must be called after `avkSimulationContext_spawnComet` when
/// `physics_type = 2` (Dynamic) to inject the vis-viva derived velocity.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setKinematicVelocity(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  vx_km_s: f32,
  vy_km_s: f32,
  vz_km_s: f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  // Convert km/s to simulation units (AU/s): 1 km/s = 1 / 149597870.7 AU/s
  const KM_PER_AU: f32 = 149_597_870.7_f32;
  let vx = vx_km_s / KM_PER_AU;
  let vy = vy_km_s / KM_PER_AU;
  let vz = vz_km_s / KM_PER_AU;
  if let Some(scenes) = ctx_ref.scenes.try_write() {
    if let Some(scene_ctx_lock) = scenes.scenes.get(&scene_id) {
      let mut scene_ctx = scene_ctx_lock.write();
      if let Some(internal_id) = scene_ctx.entity_map.get(&entity_id).copied() {
        let mut found = false;
        let _ = scene_ctx.scene.with_component_mut(
          internal_id,
          |kin: &mut aethervk_core_rlib::scene::KinematicComponent| {
            kin.velocity =
              aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(vx, vy, vz);
            found = true;
          },
        );
        return found;
      }
    }
  }
  false
}

// --- Queries ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getMeshBoundingSphereRadius(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
) -> f32 {
  if ctx.is_null() {
    return 1.0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx_rw) = scenes.get(&scene_id) {
    let scene_read = scene_ctx_rw.read();
    if let Some(int_id) = scene_read.get_entity(entity_id) {
      let mut radius = 1.0;
      let _ = scene_read.scene.with_component(
        int_id,
        |comp: &aethervk_core_rlib::scene::PhysicalMeshComponent| {
          radius = comp.sphere_radius;
        },
      );
      return radius;
    }
  }
  1.0
}

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

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityReferenceFrameType(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
) -> u32 {
  if ctx.is_null() {
    return 0; // 0 = Macro by default
  }
  let ctx_ref = unsafe { &*ctx };

  let scenes = ctx_ref.scenes.read();
  let scene_ctx = match scenes.get(&scene_id) {
    Some(s) => s.read(),
    None => return 0,
  };

  let internal_entity = match scene_ctx.get_entity(entity) {
    Some(e) => e,
    None => return 0,
  };

  let scene = &scene_ctx.scene;
  if let Some((_, Some(frame_id))) = scene.frame_relative_transform(internal_entity) {
    if let Some(frame_comp) = scene.with_component(
      frame_id,
      |c: &aethervk_core_rlib::scene::ReferenceFrameComponent| c.clone(),
    ) {
      match frame_comp.frame_type {
        aethervk_core_rlib::scene::ReferenceFrameType::Macro => return 0,
        aethervk_core_rlib::scene::ReferenceFrameType::Micro => return 1,
      }
    }
  }
  0
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
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.set_entity_visibility(scene_id, entity, visible);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setEntityFollowing(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  following: bool,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.set_entity_following(scene_id, entity, following);
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
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.set_entity_selected(scene_id, entity, selected);
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
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::ImportModel {
    task_id,
    path: path_str,
  });
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getImportedModelsCount(
  ctx: *mut SimulationContext,
) -> u32 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  scenes.model_registry.len() as u32
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
pub struct FfiMat3 {
  pub m00: f32,
  pub m10: f32,
  pub m20: f32,
  pub m01: f32,
  pub m11: f32,
  pub m21: f32,
  pub m02: f32,
  pub m12: f32,
  pub m22: f32,
}

impl Default for FfiMat3 {
  fn default() -> Self {
    Self {
      m00: 1.0,
      m10: 0.0,
      m20: 0.0,
      m01: 0.0,
      m11: 1.0,
      m21: 0.0,
      m02: 0.0,
      m12: 0.0,
      m22: 1.0,
    }
  }
}

impl FfiMat3 {
  pub fn from_mat3(mat: oshal::math::matrix::mat3::Mat3f32) -> Self {
    Self {
      m00: mat.x.x(),
      m10: mat.x.y(),
      m20: mat.x.z(),
      m01: mat.y.x(),
      m11: mat.y.y(),
      m21: mat.y.z(),
      m02: mat.z.x(),
      m12: mat.z.y(),
      m22: mat.z.z(),
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getModelLocalFrames(
  ctx: *mut SimulationContext,
  model_id: u64,
  out_user_frame: *mut FfiMat3,
  out_sim_frame: *mut FfiMat3,
) -> bool {
  if ctx.is_null() || out_user_frame.is_null() || out_sim_frame.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };

  if let Some((user_frame, sim_frame)) = ctx_ref.get_model_local_frames(model_id) {
    unsafe {
      *out_user_frame = FfiMat3::from_mat3(user_frame);
      *out_sim_frame = FfiMat3::from_mat3(sim_frame);
    }
    return true;
  }
  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_overrideModelSpherical(
  ctx: *mut SimulationContext,
  model_id: u64,
  radius_km: f32,
  mass_kg: f32,
  user_frame: *const FfiMat3,
) -> bool {
  if ctx.is_null() || user_frame.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };

  // For a spherical or externally defined object, the principal axes are isotropic.
  // Therefore, the simulation frame aligns perfectly with the user's defined local frame.
  let uf = unsafe { &*user_frame };
  let mat = oshal::math::matrix::mat3::Mat3f32 {
    x: oshal::math::vector::vec3::Vec3f32::from_components(uf.m00, uf.m10, uf.m20),
    y: oshal::math::vector::vec3::Vec3f32::from_components(uf.m01, uf.m11, uf.m21),
    z: oshal::math::vector::vec3::Vec3f32::from_components(uf.m02, uf.m12, uf.m22),
  };

  ctx_ref.override_model_spherical(model_id, radius_km, mass_kg, mat)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getImportedModels(
  ctx: *mut SimulationContext,
  out_ids: *mut u64,
  out_paths: *mut *const c_char,
  capacity: u32,
) -> u32 {
  if ctx.is_null() || out_ids.is_null() || out_paths.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();

  let mut count = 0;
  for (id, path) in &scenes.model_registry {
    if count >= capacity {
      break;
    }
    unsafe {
      *out_ids.add(count as usize) = *id;
      let c_str = alloc::ffi::CString::new(path.clone()).unwrap();
      *out_paths.add(count as usize) = c_str.into_raw();
    }
    count += 1;
  }
  count
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
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::LoadAlmanac {
    task_id,
    path: path_str,
  });
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_unloadAlmanacFile(
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
    .try_send(structs::LogicCommand::UnloadAlmanac {
      task_id,
      path: path_str,
    });
  task_id
}
// TODO C# side: builder/helper functions for strings suppoerted by `anise::time::Epoch::from_str("2024-03-24 12:00:00 TDB")`
// alternative (nah): Epoch::from_gregorian_utc_at_midnight(2000, 1, 1). But string is more precise
// TODO C# side: use crate::types::*;

#[repr(u32)]
pub enum AvkSoundEvent {
    UiClick = 0,
    UiGrab = 1,
    UiDrop = 2,
    PhysicsCollisionSoft = 3,
    PhysicsCollisionHard = 4,
}

#[repr(u32)]
pub enum AvkAudioPlaybackMode {
    MonoSpatial = 0,
    StereoDirect = 1,
}

#[repr(C)]
pub struct AvkAudioParams {
    pub volume: f32,
    pub pitch: f32,
    pub pan: f32,
    pub mode: AvkAudioPlaybackMode,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn avkSimulationContext_playSoundEvent(
    ctx: *mut SimulationContext,
    sound_event: AvkSoundEvent,
    params: AvkAudioParams,
) {
    if let Some(ctx_ref) = ctx.as_ref() {
        let mut mixer = ctx_ref.audio_mixer.write();
        
        // For now, map the AvkSoundEvent integer to a generic buffer_id.
        // Once the WAV files are embedded in the core, we will load them into the 
        // mixer upon startup and map them accurately here.
        let buffer_id = sound_event as usize;
        
        mixer.play(buffer_id, aethervk_core_rlib::audio::AvkAudioParams {
            volume: params.volume,
            pitch: params.pitch,
            pan: params.pan,
            mode: match params.mode {
                AvkAudioPlaybackMode::MonoSpatial => aethervk_core_rlib::audio::AvkAudioPlaybackMode::MonoSpatial,
                AvkAudioPlaybackMode::StereoDirect => aethervk_core_rlib::audio::AvkAudioPlaybackMode::StereoDirect,
            },
        });
    }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_parseEpochToTaiSec(
  epoch_raw: *const core::ffi::c_char,
  out_tai_sec: *mut f64,
) -> bool {
  if epoch_raw.is_null() || out_tai_sec.is_null() {
    return false;
  }
  let epoch_opt = unsafe { core::ffi::CStr::from_ptr(epoch_raw) }
    .to_str()
    .ok()
    .and_then(|epoch_str| anise::time::Epoch::from_str(epoch_str).ok());

  if let Some(epoch) = epoch_opt {
    unsafe { *out_tai_sec = epoch.to_unix_seconds() };
    true
  } else {
    false
  }
}

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
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::LoadCometSpk {
    task_id,
    spk_id,
    frame: aethervk_core_rlib::simulation::almanac::SUN_ECLIPJ2000,
    epoch,
  });
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEphemerisPosition(
  ctx: *mut SimulationContext,
  spk_id: i32,
  epoch_tai_sec: f64,
  out_pos: *mut FfiKinematicState,
) -> bool {
  if ctx.is_null() || out_pos.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let frame = aethervk_core_rlib::simulation::almanac::SUN_ECLIPJ2000;
  let epoch = anise::time::Epoch::from_unix_seconds(epoch_tai_sec);

  if let Ok(state) = ctx_ref
    .logic_state
    .read()
    .almanac_data
    .get_ephem_full(spk_id, frame, epoch, true, false)
  {
    unsafe {
      *out_pos = FfiKinematicState::from(state);
    }
    true
  } else {
    false
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_updateTrajectoryForSpk(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  spk_id: i32,
  start_epoch_tai_sec: f64,
  end_epoch_tai_sec: f64,
  sample_step_days: f64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };

  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ =
    ctx_ref
      .threads
      .logic_thread
      .tx()
      .try_send(structs::LogicCommand::UpdateTrajectoryForSpk {
        task_id,
        scene_id,
        entity_id,
        spk_id,
        start_epoch_tai_sec,
        end_epoch_tai_sec,
        sample_step_days,
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
    unsafe { CStr::from_ptr(name).to_str().unwrap_or("ModelInstance").to_string() }
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

/// Result type written to the `out_result` pointer by `avkSimulationContext_spawnComet`.
#[repr(C)]
pub struct FfiSpawnCometResult {
  /// External entity id of the LCA micro-frame parent entity.
  pub lca_frame_id: u64,
  /// External entity id of the comet mesh child entity.
  pub comet_entity_id: u64,
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
/// Synchronously spawns a static mesh entity hierarchy (LCA micro-frame + static mesh).
///
/// Returns `true` on success; `false` on any error. On success `*out_result` is
/// populated with both entity ids.
pub unsafe extern "C" fn avkSimulationContext_spawnStaticMesh(
  ctx: *mut SimulationContext,
  scene_id: u64,
  model_id: u64,
  entity_name: *const c_char,
  pos_x: f32,
  pos_y: f32,
  pos_z: f32,
  rot_w: f32,
  rot_x: f32,
  rot_y: f32,
  rot_z: f32,
  radius_km: f32,
  out_result: *mut FfiSpawnCometResult,
) -> bool {
  use aethervk_oshal_rlib::math::{
    quaternion::Quaternion,
    vector::{Vector3, vec3::Vec3f32, vec4::Quat},
  };

  if ctx.is_null() || entity_name.is_null() || out_result.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let name = unsafe { CStr::from_ptr(entity_name).to_str().unwrap_or("StaticMesh") };
  let pos = Vec3f32::from_components(pos_x, pos_y, pos_z);
  let rot = Quat::from_components(rot_x, rot_y, rot_z, rot_w);
  let scale = Vec3f32::from_components(radius_km, radius_km, radius_km);

  match ctx_ref.spawn_static_mesh_internal(scene_id, model_id, name, pos, rot, scale) {
    Ok((lca_id, mesh_id)) => {
      unsafe {
        core::ptr::write_unaligned(
          out_result,
          FfiSpawnCometResult {
            lca_frame_id: lca_id,
            comet_entity_id: mesh_id,
          },
        );
      }
      true
    }
    Err(e) => {
      oshal::log!("avkSimulationContext_spawnStaticMesh failed: {:?}", e);
      false
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_raycastNdc(
  ctx: *mut SimulationContext,
  scene_id: u64,
  camera_id: u64,
  ndc_x: f32,
  ndc_y: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .raycast_ndc(scene_id, camera_id, ndc_x, ndc_y)
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
  mass: f32,
) -> u64 {
  if ctx.is_null() || mass <= 0.0 || name.is_null() || radius <= 0.0 {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .spawn_procedural_sphere(scene_id, name, radius, mass)
    .unwrap_or_else(|e| {
      oshal::log!("spawn_procedural_sphere failed: {}", e);
      0
    })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnStaticSphere(
  ctx: *mut SimulationContext,
  scene_id: u64,
  name: *const c_char,
  radius: f32,
  mass: f32,
) -> u64 {
  if ctx.is_null() || mass <= 0.0 || name.is_null() || radius <= 0.0 {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.spawn_static_sphere(scene_id, name, radius, mass).unwrap_or_else(|e| {
    oshal::log!("spawn_static_sphere failed: {}", e);
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
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::RotateCamera(
    structs::RotateCamera {
      camera_entity: internal_entity,
      scene,
      delta_x,
      delta_y,
    },
  ));
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
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::ZoomCamera(
    structs::ZoomCamera {
      camera_entity: internal_entity,
      scene,
      amount,
    },
  ));
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
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::ResetCamera(
    structs::ResetCamera {
      camera_entity: internal_entity,
      scene,
    },
  ));
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
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::PanCamera(
    structs::PanCamera {
      camera_entity: internal_entity,
      scene,
      delta_x,
      delta_y,
    },
  ));
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
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::PanCursor(
    structs::PanCursor {
      scene,
      delta_x,
      delta_y,
    },
  ));
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
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::MoveCursor(
    structs::MoveCursor {
      scene,
      delta_x,
      delta_y,
      delta_z,
    },
  ));
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
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::SnapToEntity(
    structs::SnapToEntity {
      snap_entity: internal_snap,
      target_entity: internal_target,
      scene,
    },
  ));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setCameraTransform(
  ctx: *mut SimulationContext,
  scene_id: u64,
  camera_entity: u64,
  px: f64,
  py: f64,
  pz: f64,
  rx: f32,
  ry: f32,
  rz: f32,
  rw: f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene_ctx = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return false,
  };
  let scene_read = scene_ctx.read();
  let internal_camera = match scene_read.get_entity(camera_entity) {
    Some(e) => e,
    None => return false,
  };

  let q = aethervk_oshal_rlib::math::vector::vec4::Quat::from_components(rx, ry, rz, rw);
  let res = scene_read.scene.with_component_mut(
    internal_camera,
    |t: &mut aethervk_core_rlib::scene::HighResTransformComponent| {
      t.position = aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(px, py, pz);
      t.rotation = q;
    },
  );

  res.is_some()
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
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::FollowEntity(
    structs::FollowEntity {
      snap_entity: internal_snap,
      entity_id: internal_target,
      scene,
      unfollow_other,
    },
  ));
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

// --- Change Tracking ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getChangedEntityCount(
  ctx: *mut SimulationContext,
  scene_id: u64,
) -> u32 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx) = scenes.get(&scene_id) {
    let len = scene_ctx.read().changed_entities.read().len();
    return len as u32;
  }
  0
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getChangedEntityIds(
  ctx: *mut SimulationContext,
  scene_id: u64,
  out_ids: *mut u64,
  max_count: u32,
) -> u32 {
  if ctx.is_null() || out_ids.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx) = scenes.get(&scene_id) {
    let read_ctx = scene_ctx.read();
    let changed = read_ctx.changed_entities.read();
    let mut count = 0;
    for (id, _) in changed.iter() {
      if count >= max_count {
        break;
      }
      unsafe { *out_ids.add(count as usize) = *id };
      count += 1;
    }
    return count;
  }
  0
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getChangedComponentCount(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
) -> u32 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx) = scenes.get(&scene_id) {
    let read_ctx = scene_ctx.read();
    let changed = read_ctx.changed_entities.read();
    if let Some(components) = changed.get(&entity_id) {
      return components.len() as u32;
    }
  }
  0
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getChangedComponentNames(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  out_names: *mut *const c_char,
  max_count: u32,
) -> u32 {
  if ctx.is_null() || out_names.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx) = scenes.get(&scene_id) {
    let read_ctx = scene_ctx.read();
    let changed = read_ctx.changed_entities.read();
    if let Some(components) = changed.get(&entity_id) {
      let mut count = 0;
      for name in components.iter() {
        if count >= max_count {
          break;
        }
        // Safely pass strings to C#, assuming components names are static or leaked
        // For now, we return the internal pointer. C# must not free it.
        // Actually, name.as_ptr() is not guaranteed to be null-terminated if it's just String.
        // But since these are strings we insert, let's create a CString and leak it, or just copy it.
        // Better: use a pre-allocated buffer per FFI call or leak CString for now.
        // To be safe, we will assume C# copies the string immediately.
        if let Ok(c_str) = alloc::ffi::CString::new(alloc::format!("{}", name)) {
          unsafe { *out_names.add(count as usize) = c_str.into_raw() };
        }
        count += 1;
      }
      return count;
    }
  }
  0
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_freeComponentNames(
  names: *mut *const c_char,
  count: u32,
) {
  if names.is_null() {
    return;
  }
  for i in 0..count {
    let ptr = unsafe { *names.add(i as usize) };
    if !ptr.is_null() {
      let _ = unsafe { alloc::ffi::CString::from_raw(ptr as *mut c_char) };
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityComponentNames(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_ext_id: u64,
  out_names: *mut *const c_char,
  max_count: u32,
) -> u32 {
  if ctx.is_null() || out_names.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };

  if let Some(scene_ctx) = ctx_ref.get_scene(scene_id) {
    let s = scene_ctx.read();
    if let Some(internal_entity) = s.get_entity(entity_ext_id) {
      // Get the archetype signature for the entity
      let names = s.scene.get_entity_component_names(internal_entity);

      let mut count = 0;
      for name in names {
        if count >= max_count {
          break;
        }
        // CString allocates, C# must call your existing avkSimulationContext_freeComponentNames
        if let Ok(c_str) = alloc::ffi::CString::new(name) {
          unsafe { *out_names.add(count as usize) = c_str.into_raw() };
        }
        count += 1;
      }
      return count;
    }
  }
  0
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

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setSimulationCallback(
  cb: Option<extern "C" fn(u64, u64, u64, *const core::ffi::c_void)>,
) {
  SimulationContext::set_simulation_callback(cb)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setRenderCallback(
  cb: Option<extern "C" fn(u64, u64, u64)>,
) {
  SimulationContext::set_render_callback(cb)
}

// --- Tick & Time ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_playScene(
  ctx: *mut SimulationContext,
  scene_id: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::PlayScene { scene_id });
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_snapshotScene(
  ctx: *mut SimulationContext,
  scene_id: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::SnapshotScene { scene_id });
  true
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_restoreSnapshot(
  ctx: *mut SimulationContext,
  scene_id: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::RestoreSnapshot { scene_id });
  true
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_pauseScene(
  ctx: *mut SimulationContext,
  scene_id: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::PauseScene { scene_id });
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setTimeScale(
  ctx: *mut SimulationContext,
  scene_id: u64,
  scale: u32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let time_scale = match scale {
    1 => structs::TimeScale::OneDay,
    2 => structs::TimeScale::OneWeek,
    3 => structs::TimeScale::OneMonth,
    _ => structs::TimeScale::Stopped,
  };
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::SetSceneTimeScale {
      scene_id,
      scale: time_scale,
    });
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

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addSunComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  res_x: u32,
  res_y: u32,
  res_z: u32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.add_sun_component(scene_id, entity, (res_x, res_y, res_z), 0.6);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addCursorComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.add_cursor_component(scene_id, entity);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addSkyComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.add_sky_component(scene_id, entity);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addGridComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.add_grid_component(scene_id, entity);
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
  width: f32,
  height: f32,
  near: f32,
  far: f32,
  ortho_scale_factor: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let aspect = if height > 0.0 { width / height } else { 1.0 };
  let params = if is_orthographic {
    let w = width * ortho_scale_factor;
    let h = height * ortho_scale_factor;
    CameraParams::Orthographic(OrthographicCameraParams {
      left: -w / 2.0,
      right: w / 2.0,
      bottom: -h / 2.0,
      top: h / 2.0,
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
  let _ = ctx_ref.add_camera_component(scene_id, entity, params);
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiScreenSpaceBillboardDTO {
  pub ndc_x: f32,
  pub ndc_y: f32,
  pub scale: f32,
  pub rotation_deg: f32,
  pub opacity: f32,
  pub z_index: i32,
  pub viewport_id: u64,
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
  ctx_ref.get_bvh_nodes(scene_id, entity, count).unwrap_or(core::ptr::null_mut())
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

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfiTransform {
  pub px: f32,
  pub py: f32,
  pub pz: f32,
  pub rw: f32,
  pub rx: f32,
  pub ry: f32,
  pub rz: f32,
  pub sx: f32,
  pub sy: f32,
  pub sz: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfiHighResTransform {
  pub px: f64,
  pub py: f64,
  pub pz: f64,
  pub rw: f32,
  pub rx: f32,
  pub ry: f32,
  pub rz: f32,
  pub sx: f32,
  pub sy: f32,
  pub sz: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfiCamera {
  pub is_orthographic: bool,
  pub fov: f32,
  pub aspect: f32,
  pub near: f32,
  pub far: f32,
  pub ortho_scale_factor: f32,
  pub focus_distance: f32,
  pub proj: [f32; 16],
}

#[repr(u32)]
/// TODO: Document this item
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
/// TODO: Document this item
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
/// TODO: Document this item
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
/// TODO: Document this item
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
/// TODO: Document this item
pub struct FfiLogicCommand {
  pub cmd_type: FfiLogicCommandType,
  pub payload: [u8; 28],
}

impl FfiLogicCommand {
  /// TODO: Document this item
  pub fn get_u32_u64x3_at_start(&self) -> Option<(u32, u64, u64, u64)> {
    let first = self.get_u32_at_offset(0)?;
    let second = self.get_u64_at_offset(4)?;
    let third = self.get_u64_at_offset(12)?;
    let fourth = self.get_u64_at_offset(20)?;
    Some((first, second, third, fourth))
  }

  /// TODO: Document this item
  pub fn get_u64_f32x3_at_start(&self) -> Option<(u64, f32, f32, f32)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_f32_at_offset(12)?;
    let third = self.get_f32_at_offset(16)?;
    let fourth = self.get_f32_at_offset(20)?;
    Some((first, second, third, fourth))
  }

  /// TODO: Document this item
  pub fn get_u64_f32x2_at_start(&self) -> Option<(u64, f32, f32)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_f32_at_offset(12)?;
    let third = self.get_f32_at_offset(16)?;
    Some((first, second, third))
  }

  /// TODO: Document this item
  pub fn get_u64x2_f32_at_start(&self) -> Option<(u64, u64, f32)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_u64_at_offset(12)?;
    let third = self.get_f32_at_offset(20)?;
    Some((first, second, third))
  }

  /// TODO: Document this item
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
/// TODO: Document this item
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
/// TODO: Document this item
pub enum FfiNodeType {
  AABB = 0,
  OBB = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
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
  /// TODO: Document this item
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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn avkSimulationContext_getSimulationTime(
  ctx: *mut SimulationContext,
  scene_id: u64,
) -> f64 {
  let ctx = unsafe { &*ctx };
  ctx.get_simulation_time(scene_id)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn avkSimulationContext_getSimulationTimeUtc(
  ctx: *mut SimulationContext,
  scene_id: u64,
  buffer: *mut core::ffi::c_char,
  buffer_len: u32,
) -> bool {
  let ctx = unsafe { &*ctx };
  ctx.get_simulation_time_utc(scene_id, buffer, buffer_len)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn avkSimulationContext_setSimulationTime(
  ctx: *mut SimulationContext,
  scene_id: u64,
  time_tai: f64,
) {
  let ctx = unsafe { &*ctx };
  ctx.set_simulation_time(scene_id, time_tai);
}

/// Seek the simulation to a specific epoch, recomputing all ephemeris body positions and
/// rebuilding the TLAS. Dispatches a `SeekEpoch` command to the logic thread.
///
/// P/Invoke: `avkSimulationContext_seekEpoch(IntPtr ctx, ulong sceneId, double epochTai)`
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_seekEpoch(
  ctx: *mut SimulationContext,
  scene_id: u64,
  epoch_tai: f64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::SeekEpoch {
    scene_id,
    epoch_tai_seconds: epoch_tai,
  });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn avkSimulationContext_getEpochLimits(
  ctx: *mut SimulationContext,
  scene_id: u64,
  start_tai: *mut f64,
  end_tai: *mut f64,
) -> bool {
  let ctx = unsafe { &*ctx };
  ctx.get_epoch_limits(scene_id, start_tai, end_tai)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setEpochRange(
  ctx: *mut SimulationContext,
  scene_id: u64,
  start_tai: f64,
  end_tai: f64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.set_epoch_range(scene_id, start_tai, end_tai)
}

/// Check whether the loaded almanac SPK data covers the given epoch interval
/// for Earth (399) and a specified comet NAIF ID.
///
/// P/Invoke: `avkSimulationContext_checkAlmanacCoverage(IntPtr ctx, int cometSpkId, double startTai, double endTai)`
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_checkAlmanacCoverage(
  ctx: *mut SimulationContext,
  comet_spk_id: i32,
  start_tai: f64,
  end_tai: f64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let logic_state = ctx_ref.logic_state.read();
  let almanac = &logic_state.almanac_data;

  let start = anise::time::Epoch::from_unix_seconds(start_tai);
  let end = anise::time::Epoch::from_unix_seconds(end_tai);

  // Check Earth coverage (required for orbit reference frame)
  let earth_ok = almanac.covers_interval(399, start, end);
  // Check comet coverage
  let comet_ok = almanac.covers_interval(comet_spk_id, start, end);

  earth_ok && comet_ok
}

/// Loads an SPK file into a temporary almanac and probes whether ephemeris data
/// can be queried at the given start and end epochs for the specified NAIF ID.
/// Returns true if the SPK covers the requested epoch range.
/// On success or partial success, writes the actual SPK domain and discovered
/// NAIF ID (which may differ from spk_id) to the out pointers.
///
/// P/Invoke: `avkSimulationContext_probeSpkFile(string path, int spkId, double startTai, double endTai, out double domainStart, out double domainEnd, out int discoveredNaifId)`
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_probeSpkFile(
  path: *const c_char,
  spk_id: i32,
  start_tai_sec: f64,
  end_tai_sec: f64,
  out_domain_start_tai_sec: *mut f64,
  out_domain_end_tai_sec: *mut f64,
  out_discovered_naif_id: *mut i32,
) -> bool {
  // Zero out all outputs upfront
  if !out_domain_start_tai_sec.is_null() {
    unsafe { *out_domain_start_tai_sec = 0.0; }
  }
  if !out_domain_end_tai_sec.is_null() {
    unsafe { *out_domain_end_tai_sec = 0.0; }
  }
  if !out_discovered_naif_id.is_null() {
    unsafe { *out_discovered_naif_id = 0; }
  }

  if path.is_null() {
    return false;
  }
  let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };
  let start_epoch = anise::time::Epoch::from_unix_seconds(start_tai_sec);
  let end_epoch = anise::time::Epoch::from_unix_seconds(end_tai_sec);
  let (covers, domain, discovered_id) =
    aethervk_core_rlib::simulation::almanac::AlmanacPackedData::probe_spk_file_with_domain(
      path_str, spk_id, start_epoch, end_epoch,
    );

  if let Some((ds, de)) = domain {
    if !out_domain_start_tai_sec.is_null() {
      unsafe { *out_domain_start_tai_sec = ds.to_unix_seconds(); }
    }
    if !out_domain_end_tai_sec.is_null() {
      unsafe { *out_domain_end_tai_sec = de.to_unix_seconds(); }
    }
  }
  if !out_discovered_naif_id.is_null() {
    unsafe { *out_discovered_naif_id = discovered_id; }
  }

  covers
}

/// Sets a callback that the engine will invoke when it detects missing almanac SPK coverage.
/// The callback signature is: `fn(spk_id: i32, start_epoch_str: *const c_char, end_epoch_str: *const c_char) -> *const c_char`
/// The returned pointer is the file path of the downloaded SPK file (null on failure).
/// P/Invoke: `avkSimulationContext_setAlmanacInvalidationCallback(IntPtr ctx, IntPtr callbackFnPtr)`
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setAlmanacInvalidationCallback(
  ctx: *mut SimulationContext,
  callback: Option<
    extern "C" fn(
      i32,
      *const core::ffi::c_char,
      *const core::ffi::c_char,
    ) -> *const core::ffi::c_char,
  >,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let mut logic_state = ctx_ref.logic_state.write();
  logic_state.almanac_invalidation_callback = callback;
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_ephemeris_position_ffi_signature() {
    // We just test if the symbol exists and compiles.
    let f: unsafe extern "C" fn(*mut SimulationContext, i32, f64, *mut FfiKinematicState) -> bool =
      avkSimulationContext_getEphemerisPosition;
    assert!(f as usize > 0);
  }

  #[test]
  fn test_update_trajectory_ffi_signature() {
    let f: unsafe extern "C" fn(*mut SimulationContext, u64, u64, i32, f64, f64, f64) -> u64 =
      avkSimulationContext_updateTrajectoryForSpk;
    assert!(f as usize > 0);
  }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn avkSimulationContext_addJet(
  ctx: *mut SimulationContext,
  scene_id: u64,
  comet_id: u64,
  radius_km: f32,
  lat: f32,
  lon: f32,
  color_r: f32,
  color_g: f32,
  color_b: f32,
  mass: f32,
  particles_per_tick: u32,
  ttl: f32,
  mean_velocity: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  match ctx_ref.add_jet(
    scene_id,
    comet_id,
    radius_km,
    lat,
    lon,
    color_r,
    color_g,
    color_b,
    mass,
    particles_per_tick,
    ttl,
    mean_velocity,
  ) {
    Ok(id) => id,
    Err(e) => {
      aethervk_oshal_rlib::log!("Error adding jet: {:?}", e);
      0
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addCameraAnimation(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  target_x: f64,
  target_y: f64,
  target_z: f64,
  duration: f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx_guard) = scenes.get(&scene_id) {
    let scene_ctx = scene_ctx_guard.write();
    let internal_id = match scene_ctx.get_entity(entity_id) {
      Some(id) => id,
      None => return false,
    };

    let (start_pos, start_rot) = if let Some(t) = scene_ctx.scene.with_component(
      internal_id,
      |t: &aethervk_core_rlib::scene::HighResTransformComponent| (t.position, t.rotation),
    ) {
      t
    } else {
      return false;
    };

    let anim = aethervk_core_rlib::scene::animation::TransformAnimationComponent {
      start_pos,
      start_rot,
      target_pos: aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
        target_x, target_y, target_z,
      ),
      target_rot: start_rot,
      duration,
      elapsed: 0.0,
      is_finished: false,
    };

    let _ = scene_ctx
      .scene
      .remove_component::<aethervk_core_rlib::scene::animation::TransformAnimationComponent>(
        internal_id,
      );
    let _ = scene_ctx.scene.add_component(internal_id, anim);
    return true;
  }
  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_checkCameraAnimationFinished(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
) -> bool {
  if ctx.is_null() {
    return true;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx_guard) = scenes.get(&scene_id) {
    let scene_ctx = scene_ctx_guard.read();
    let internal_id = match scene_ctx.get_entity(entity_id) {
      Some(id) => id,
      None => return true,
    };

    if let Some(finished) = scene_ctx.scene.with_component(
      internal_id,
      |a: &aethervk_core_rlib::scene::animation::TransformAnimationComponent| a.is_finished,
    ) {
      return finished;
    }
  }
  true
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_removeCameraAnimation(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx_guard) = scenes.get(&scene_id) {
    let scene_ctx = scene_ctx_guard.write();
    let internal_id = match scene_ctx.get_entity(entity_id) {
      Some(id) => id,
      None => return false,
    };

    return scene_ctx
      .scene
      .remove_component::<aethervk_core_rlib::scene::animation::TransformAnimationComponent>(
        internal_id,
      )
      .is_ok();
  }
  false
}
