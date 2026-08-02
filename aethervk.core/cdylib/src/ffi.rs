//! ffi module.

use aethervk_core_rlib::{
  gpu,
  gpu_backends::vulkan,
  scene::ForeignSerializable,
  simulation::almanac::AlmanacPackedData,
  simulation_api::{external_state::CTimeRange, structs::*, *},
};
use aethervk_oshal_rlib as oshal;
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{Vector3, Vector4, vec3::Vec3f32, vec4::Quat},
};
use alloc::{boxed::Box, string::ToString};
use bytemuck::Zeroable;
use core::{
  ffi::{CStr, c_char},
  str::FromStr,
};

pub type PanicCallback = extern "C" fn(*const u8, usize);
pub static mut PANIC_CALLBACK: Option<PanicCallback> = None;

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkRegisterPanicCallback(cb: PanicCallback) {
  unsafe {
    PANIC_CALLBACK = Some(cb);
  }
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_startup() -> *mut SimulationContext {
  SimulationContext::startup(gpu::VULKAN_RENDER_BACKEND, None)
    .map(Box::into_raw)
    .unwrap_or_else(|e| {
      oshal::log!("avkSimulationContext_startup failed: {}", e.to_string());
      emit_breadcrumb(1, &alloc::format!("Startup failed: {}", e.to_string()));
      core::ptr::null_mut()
    })
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_shutdownSync(ctx: *mut SimulationContext) {
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

// TODO remove when rendering switches to Avalonia 11 Composition API
/// # Safety
/// FFI Contract
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

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addViewport(
  ctx: *mut SimulationContext,
  scene_id: u64,
  width: u32,
  height: u32,
  name: *const c_char,
  out_presentation_engine: *mut u64,
  out_camera_entity: *mut u64,
) -> bool {
  // Null check
  if ctx.is_null() {
    return false;
  }

  // Zero init results
  if !out_presentation_engine.is_null() {
    unsafe { *out_presentation_engine = 0 };
  }
  if !out_camera_entity.is_null() {
    unsafe { *out_camera_entity = 0 };
  }

  let ctx_ref = unsafe { ctx.as_ref_unchecked() };
  let name_str = if name.is_null() {
    "DefaultViewport"
  } else {
    unsafe { core::ffi::CStr::from_ptr(name) }.to_str().unwrap_or("DefaultViewport")
  };

  const DEFAULT_FOV: f32 = core::f32::consts::FRAC_PI_4;
  const DEFAULT_NEAR: f32 = 0.001;
  const DEFAULT_FAR: f32 = 10000.0;

  // TODO windowed
  match ctx_ref.create_presentation_engine(scene_id, width, height).and_then(|pe_id| {
    ctx_ref
      .add_perspective_camera(
        scene_id,
        pe_id,
        name_str,
        DEFAULT_FOV,
        DEFAULT_NEAR,
        DEFAULT_FAR,
      )
      .map(|id| (id.get(), pe_id.0))
  }) {
    Ok((cam_id, pe_id)) => {
      if !out_presentation_engine.is_null() {
        unsafe { *out_presentation_engine = pe_id };
      }
      if !out_camera_entity.is_null() {
        unsafe { *out_camera_entity = cam_id };
      }
      true
    }
    Err(_) => {
      // Logic layer is responsible for logging and breadcrumb emission
      false
    }
  }
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_removeViewport(
  ctx: *mut SimulationContext,
  scene_id: u64,
  handle: u64,
) {
  if ctx.is_null() || handle == 0 {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  // Handles destruction of associated camera too
  let _ = ctx_ref.destroy_presentation_engine(scene_id, gpu::PresentationEngineHandle(handle));
}

// TODO: remove/rework when swapping for the Avalonia 11 Composition API
/// # Safety
/// FFI Contract
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

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_importModel(
  ctx: *mut SimulationContext,
  path: *const c_char,
) -> bool {
  if ctx.is_null() || path.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("").to_string() };
  // TODO remove task_id from all logic commands
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::ImportModel {
    task_id: 1,
    path: path_str,
  });
  // Model ID will be communicated via external state SimulationCallback
  true
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_unloadModel(
  ctx: *mut SimulationContext,
  model_id: u64,
) {
  if ctx.is_null() {
    return;
  }
  // TODO: all static mesh components using that model will be swapped for a procedurally generated
  // sphere
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.unload_model(model_id);
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_loadAlmanacFile(
  ctx: *mut SimulationContext,
  path: *const c_char,
) -> bool {
  if ctx.is_null() || path.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("").to_string() };
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::LoadAlmanac {
    task_id: 1,
    path: path_str,
  });
  // cannot safely say that there's no error. only from breadcrumb we'll know. TODO: callback on
  // almanac error so that application can crash or rollback
  true
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_unloadAlmanacFile(
  ctx: *mut SimulationContext,
  path: *const c_char,
) -> bool {
  if ctx.is_null() || path.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("").to_string() };
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::UnloadAlmanac {
      task_id: 1,
      path: path_str,
    });
  // cannot safely say that there's no error. only from breadcrumb we'll know. TODO: callback on
  // almanac error so that application can crash or rollback
  true
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSetAssetPath(path: *const c_char) {
  if path.is_null() {
    return;
  }

  if let Ok(path_str) = unsafe { core::ffi::CStr::from_ptr(path) }.to_str() {
    SimulationContext::set_asset_path(path_str)
  }
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSetLoggerCallback(cb: Option<extern "C" fn(*const c_char)>) {
  SimulationContext::set_logger_callback(cb)
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSetBreadcrumbCallback(
  cb: Option<unsafe extern "C" fn(u32, *const c_char)>,
) {
  SimulationContext::set_breadcrumb_callback(cb)
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSetSimulationCallback(
  cb: Option<unsafe extern "C" fn(u64, u64, u64, *const core::ffi::c_void)>,
) {
  SimulationContext::set_simulation_callback(cb)
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSetExternalStateSimulationCallback(
  cb: Option<ExternalStateSimulationCallback>,
) {
  set_external_state_simulation_callback(cb);
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSetRenderCallback(cb: Option<unsafe extern "C" fn(u64, u64, u64)>) {
  SimulationContext::set_render_callback(cb)
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSetMainThreadDispatchCallback(cb: Option<MainThreadDispatchCallback>) {
  register_main_thread_dispatcher(cb);
}

/// the C# caller should be responsible for polling getTaskStatus and only calling download_image once the status is completed.
/// Given the architecture of the C# async API and the fact that Vulkan windowless downloads are placed in persistently mapped memory buffers, bouncing the download request through the RenderCommand channel was actually an anti-pattern. Here is why:
/// 1. Thread Safety: is_task_completed and read_windowless_download only acquire read/write locks (self.res.read() and pending_downloads.write()). They don't actually need to execute on the Render Thread.
/// 2. No Render Thread Blocking: By having C# call download_image synchronously after polling confirms it's ready, the memory copy happens on the caller thread (e.g., C#'s worker pool), freeing the Render Thread to continue pushing Vulkan commands without stalling
///   during the memory copy.
/// # Safety
/// FFI Contract
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
  unsafe { ctx_ref.download_image(task_id, buffer_ptr, buffer_size) }
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

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addScreenSpaceBillboard(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  image_path: *const c_char,
  ndc_x: f32,
  ndc_y: f32,
  scale: f32,
  rotation_deg: f32,
  opacity: f32,
  z_index: i32,
  viewport_id: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let image_path_str = if image_path.is_null() {
    ""
  } else {
    unsafe { CStr::from_ptr(image_path).to_str().unwrap_or("") }
  };
  ctx_ref
    .add_screen_space_billboard_component(
      scene_id,
      entity,
      image_path_str,
      ndc_x,
      ndc_y,
      scale,
      rotation_deg,
      opacity,
      z_index,
      viewport_id,
    )
    .is_ok()
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setScreenSpaceBillboard(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  ndc_x: f32,
  ndc_y: f32,
  scale: f32,
  rotation_deg: f32,
  opacity: f32,
  z_index: i32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .set_screen_space_billboard(
      scene_id,
      entity,
      ndc_x,
      ndc_y,
      scale,
      rotation_deg,
      opacity,
      z_index,
    )
    .is_ok()
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getScreenSpaceBillboard(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  out_data: *mut FfiScreenSpaceBillboardDTO,
) -> bool {
  if ctx.is_null() || out_data.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  if let Ok(dto) = ctx_ref.get_screen_space_billboard(scene_id, entity) {
    unsafe {
      *out_data = FfiScreenSpaceBillboardDTO {
        ndc_x: dto.ndc_x,
        ndc_y: dto.ndc_y,
        scale: dto.scale,
        rotation_deg: dto.rotation_deg,
        opacity: dto.opacity,
        z_index: dto.z_index,
        viewport_id: dto.viewport_id,
      };
    }
    true
  } else {
    false
  }
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
// TODO update C# side
pub unsafe extern "C" fn avkSimulationContext_setEpochRange(
  ctx: *mut SimulationContext,
  scene_id: u64,
  tai: *const CTimeRange,
) -> bool {
  // null check
  if ctx.is_null() || tai.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let tai_ref = unsafe { tai.as_ref_unchecked() };
  let start = anise::time::Epoch::from_tai_parts(tai_ref.centuries[0], tai_ref.nanoseconds[0]);
  let end = anise::time::Epoch::from_tai_parts(tai_ref.centuries[1], tai_ref.nanoseconds[1]);
  // the only kind of error cdylib logs is the one from sending
  if let Err(e) = ctx_ref.threads.logic_thread.tx().try_send(LogicCommand::SetEpochRange {
    scene_id,
    start,
    end,
  }) {
    emit_breadcrumb(3, "Error while sending setEpochRange command");
    oshal::log!("{}", e);
    false
  } else {
    // set epoch range will yield an external state callback
    true
  }
}

/// Check whether the loaded almanac SPK data covers the given epoch interval
/// for Earth (399) and a specified comet NAIF ID.
///
/// P/Invoke: `avkSimulationContext_checkAlmanacCoverage(IntPtr ctx, int cometSpkId, double startTai, double endTai)`
/// # Safety
/// FFI Contract
// TODO update C# side
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_checkAlmanacCoverage(
  ctx: *mut SimulationContext,
  comet_spk_id: i32,
  tai: *const CTimeRange,
) -> bool {
  // null check
  if ctx.is_null() || tai.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let logic_state = ctx_ref.logic_state.read();
  let almanac = &logic_state.almanac_data;

  let tai_cov = unsafe { tai.as_ref_unchecked() };
  let start = anise::time::Epoch::from_tai_parts(tai_cov.centuries[0], tai_cov.nanoseconds[0]);
  let end = anise::time::Epoch::from_tai_parts(tai_cov.centuries[1], tai_cov.nanoseconds[1]);

  // Check Earth coverage (required for orbit reference frame)
  let earth_ok = almanac.covers_interval(anise::constants::celestial_objects::EARTH, start, end);
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
/// # Safety
/// FFI Contract
// TODO update C# side
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkProbeSpkFile(
  path: *const c_char,
  spk_id: i32,
  tai_parts: *const CTimeRange,
  out_domain_tai_parts: *mut CTimeRange,
  out_discovered_naif_id: *mut i32,
) -> bool {
  // Zero out all outputs upfront
  if !out_domain_tai_parts.is_null() {
    unsafe { *out_domain_tai_parts = CTimeRange::zeroed() };
  }
  if !out_discovered_naif_id.is_null() {
    unsafe { *out_discovered_naif_id = 0 };
  }

  // null check input
  if path.is_null() || tai_parts.is_null() {
    return false;
  }

  let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };
  let tai = unsafe { tai_parts.as_ref_unchecked() };
  let start_epoch = anise::time::Epoch::from_tai_parts(tai.centuries[0], tai.nanoseconds[0]);
  let end_epoch = anise::time::Epoch::from_tai_parts(tai.centuries[1], tai.nanoseconds[1]);
  let (covers, domain, discovered_id) =
    AlmanacPackedData::probe_spk_file_with_domain(path_str, spk_id, start_epoch, end_epoch);

  if let Some((ds, de)) = domain {
    if !out_domain_tai_parts.is_null() {
      unsafe { *out_domain_tai_parts = CTimeRange::new(ds, de) };
    }
  }
  if !out_discovered_naif_id.is_null() {
    unsafe { *out_discovered_naif_id = discovered_id };
  }

  covers
}

/// # Safety
/// FFI Contract
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

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn registerMainThreadDispatcher(cb: Option<MainThreadDispatchCallback>) {
  register_main_thread_dispatcher(cb);
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn executeMainThreadCleanup(
  device_ptr: *const vulkan::device::Device,
  command: u32,
  signal_done: *const core::sync::atomic::AtomicBool,
) {
  unsafe { execute_main_thread_cleanup(device_ptr, command, signal_done) };
}

// ----------------------- Debugging Functionality -------------------------------------------

#[cfg(debug_assertions)]
pub use debug::*;

#[cfg(debug_assertions)]
pub mod debug {
  use super::*;

  #[repr(C)]
  #[derive(Debug, Clone, Copy)]
  pub struct SceneHierarchyNodeDTO {
    pub entity_id: u64,
    pub parent_id: u64, // 0 denotes a root entity with no parent
  }

  /// # Safety
  /// FFI Contract
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

  /// # Safety
  /// FFI Contract
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

  /// # Safety
  /// FFI Contract
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

  /// # Safety
  /// FFI Contract
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

  /// # Safety
  /// FFI Contract
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

  /// # Safety
  /// FFI Contract
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

  /// # Safety
  /// FFI Contract
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
}