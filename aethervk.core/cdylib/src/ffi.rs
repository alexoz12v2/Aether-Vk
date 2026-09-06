//! ffi module.

use aethervk_core_rlib::{
  gpu,
  gpu_backends::vulkan,
  scene::{ForeignSerializable, ParticleSystemComputedDTO, ParticleSystemDTO},
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

/// Startup Parameters
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod)]
pub struct CStartupParameters {
  pub start_range: CTimeRange,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod)]
pub struct CStartupReturn {
  pub earth_planet_entity: u64,
  pub comet_planet_entity: u64,
  pub scene_id: u64,
  pub ctx: isize,
  // compiles only on 64 bit platforms. 32 bit require padding here
}

/// IAU rotational model parameters for a small body (comet/asteroid).
/// All angles in degrees; rates per century or per day as noted.
/// Passed to `avkSimulationContext_setBodyRotationalModel`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
pub struct CBodyRotationalModelDTO {
  /// Right ascension of rotation pole at J2000 (degrees).
  pub pole_ra_deg: f64,
  /// Declination of rotation pole at J2000 (degrees).
  pub pole_dec_deg: f64,
  /// Prime meridian angle at J2000 (degrees).
  pub prime_meridian_deg: f64,
  /// Rate of change of pole RA (degrees/century).
  pub pole_ra_rate_deg_cen: f64,
  /// Rate of change of pole Dec (degrees/century).
  pub pole_dec_rate_deg_cen: f64,
  /// Sidereal rotation rate (degrees/day).
  pub rot_rate_deg_day: f64,
}

/// Osculating Keplerian elements from the JPL Small-Body Database.
/// All angles in degrees; perihelion distance in AU.
/// Passed to `avkSimulationContext_tryInitComet` so the logic thread can draw
/// the analytical orbit track without depending on the SPK time coverage.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
pub struct CKeplerianElementsDTO {
  /// Orbital eccentricity (e ≥ 0; <1 = ellipse, ≥1 = hyperbola/parabola).
  pub eccentricity: f64,
  /// Perihelion distance (AU).
  pub perihelion_distance_au: f64,
  /// Inclination to the ecliptic (degrees).
  pub inclination_deg: f64,
  /// Longitude of the ascending node Ω (degrees).
  pub longitude_of_ascending_node_deg: f64,
  /// Argument of perihelion ω (degrees).
  pub argument_of_perihelion_deg: f64,
}

/// - Should create initial scene
/// - should set initial epoch to 2020 and end epoch 1 month later (TDB)
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_startup(
  params: *const CStartupParameters,
  out: *mut CStartupReturn,
) -> bool {
  if params.is_null() || out.is_null() {
    return false;
  }
  let range = unsafe {
    let crange = params.as_ref().unwrap_unchecked().start_range;
    let min_index = if crange.centuries[0] < crange.centuries[1]
      || (crange.centuries[0] == crange.centuries[1]
        && crange.nanoseconds[0] < crange.nanoseconds[1])
    {
      0
    } else {
      1
    };
    let max_index = 1 - min_index;
    let start = anise::time::Epoch::from_tdb_duration(anise::time::Duration::from_parts(
      crange.centuries[min_index],
      crange.nanoseconds[min_index],
    ));
    let end = anise::time::Epoch::from_tdb_duration(anise::time::Duration::from_parts(
      crange.centuries[max_index],
      crange.nanoseconds[max_index],
    ));
    if end - start < anise::time::Duration::from_days(30.0) {
      oshal::log!(
        "avkSimulationContext_startup failed: start and end epoch must be at least 1 month apart"
      );
      emit_breadcrumb(
        1,
        "Startup failed: start and end epoch must be at least 1 month apart",
      );
      return false;
    }
    [start, end]
  };
  match SimulationContext::startup(None) {
    Ok(ctx_box) => {
      // ── Auto-load Earth almanac files from ASSET_DIR ─────────────────────────────────────
      // avkSetAssetPath must be called before avkSimulationContext_startup for this to work.
      // Loading is done synchronously here so that create_empty_scene2 sees the data
      // immediately and can perform Earth initialization (attach AlmanacPlanet, force_reposition,
      // dispatch trajectory) in the same startup call.
      if let Some(asset_dir) = aethervk_core_rlib::gpu::ASSET_DIR.read().clone() {
        let de442_path = alloc::format!("{}/planets/de442.bsp", asset_dir);
        let bpc_path = alloc::format!("{}/earth_latest_high_prec.bpc", asset_dir);
        let pca_path = alloc::format!("{}/planets/pck00011.pca", asset_dir);
        let gm_path = alloc::format!("{}/planets/gm_de431.pca", asset_dir);
        let mut logic = ctx_box.logic_state.write();
        if let Err(e) = logic.almanac_data.load_almanac(oshal::os::fs::PathBuf::from(&de442_path)) {
          oshal::log!("[startup] de442.bsp load failed: {}", e);
          emit_breadcrumb(
            2,
            "de442.bsp not available at startup — Earth will be at origin",
          );
          panic!("de442.bsp");
        }
        if let Err(e) = logic.almanac_data.load_almanac(oshal::os::fs::PathBuf::from(&bpc_path)) {
          oshal::log!("[startup] earth_latest_high_prec.bpc load failed: {}", e);
          emit_breadcrumb(
            2,
            "earth_latest_high_prec.bpc not available at startup — Earth rotation unavailable",
          );
        }
        if let Err(e) = logic.almanac_data.load_almanac(oshal::os::fs::PathBuf::from(&pca_path)) {
          oshal::log!("[startup] pck00011.pca load failed: {}", e);
          emit_breadcrumb(
            2,
            "pck00011.pca not available at startup — Earth constants unavailable",
          );
          panic!("pck00011.pca");
        }
        if let Err(e) = logic.almanac_data.load_almanac(oshal::os::fs::PathBuf::from(&gm_path)) {
          oshal::log!("[startup] gm_de431.pca load failed: {}", e);
          emit_breadcrumb(
            2,
            "gm_de431.pca not available at startup — Gravity mass unavailable",
          );
          panic!("gm_de431.pca");
        }
        drop(logic);
      }

      // purposefully unwrap to crash
      let scene_return = ctx_box.create_empty_scene2(false, range[0], range[1]).unwrap();
      let out_mut = unsafe { out.as_mut().unwrap_unchecked() };

      out_mut.earth_planet_entity = scene_return.earth_body;
      out_mut.comet_planet_entity = scene_return.comet_body;
      out_mut.scene_id = scene_return.scene_id;
      out_mut.ctx = alloc::boxed::Box::into_raw(ctx_box) as _;

      true
    }
    Err(e) => {
      oshal::log!("avkSimulationContext_startup failed: {}", e.to_string());
      oshal::os::debug::print_aethervk_stacktrace(0, 10);
      emit_breadcrumb(1, &alloc::format!("Startup failed: {}", e));
      false
    }
  }
}

/// Earth initialization is now performed automatically inside `avkSimulationContext_startup` if
/// `avkSetAssetPath` has been called beforehand. This function is no longer needed.
/// Kept as a no-op for ABI compatibility — will be removed in a future version.
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_initEarth() -> u64 {
  0
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
/// - `handle_type == 0` → windowless path (backward compat, no callback fired).
/// - `handle_type > 0` → windowed path; `GET_NATIVE_WINDOW_HANDLE_CALLBACK` must have been
///   registered via `avkSetGetNativeWindowHandleCallback`. Rust fires the callback and
///   spin-waits for the C# UI thread to fill the native handle.
///
/// `handle_type` values match `gpu::NativeHandleType`:
/// - `0` = Unknown / windowless
/// - `1` = Win32   (ptr0 = HINSTANCE, ptr1 = HWND)
/// - `3` = Xlib    (ptr0 = Display*, ptr1 = Window / XID)
/// - `4` = Xcb     (ptr0 = xcb_connection_t*, ptr1 = xcb_window_t)
/// - `5` = Metal   (ptr0 = CAMetalLayer*, ptr1 = 0)
///
/// Note: Wayland (2) is intentionally omitted — Avalonia 11 does not support native Wayland.
///
/// **Must NOT be called from the UI thread in windowed mode** — the callback dispatches work
/// there; calling it from the UI thread would deadlock.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addViewport(
  ctx: *mut SimulationContext,
  scene_id: u64,
  width: u32,
  height: u32,
  name: *const c_char,
  handle_type: u32,
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

  let ctx_ref = unsafe { ctx.as_ref().unwrap() };
  let name_str = if name.is_null() {
    "DefaultViewport"
  } else {
    unsafe { core::ffi::CStr::from_ptr(name) }.to_str().unwrap_or("DefaultViewport")
  };

  const DEFAULT_FOV: f32 = core::f32::consts::FRAC_PI_4;
  const DEFAULT_NEAR: f32 = 0.001;
  const DEFAULT_FAR: f32 = 10000.0;

  // Resolve the NativeHandleType discriminator. `None` means windowless.
  let native_handle_type: Option<gpu::NativeHandleType> = match handle_type {
    0 => None,
    1 => Some(gpu::NativeHandleType::Win32),
    3 => Some(gpu::NativeHandleType::Xlib),
    4 => Some(gpu::NativeHandleType::Xcb),
    5 => Some(gpu::NativeHandleType::Metal),
    _ => {
      emit_breadcrumb(1, "avkSimulationContext_addViewport: unknown handle_type");
      return false;
    }
  };

  let result = if let Some(ht) = native_handle_type {
    // ── Windowed path ─────────────────────────────────────────────────────
    // Request the OS window handle from the C# UI thread. Rust spin-waits until
    // C# fills the handle and signals the AtomicBool (see get_native_window_handle_sync).
    match unsafe { get_native_window_handle_sync() } {
      Some(raw) => {
        let window_info = gpu::OpaqueNativeHandleInfo {
          ptr0: raw.field0 as *mut _,
          ptr1: raw.field1 as *mut _,
          handle_type: ht,
        };
        ctx_ref
          .create_presentation_engine_windowed(scene_id, width, height, window_info)
          .and_then(|pe_id| {
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
          })
      }
      None => {
        emit_breadcrumb(
          1,
          "avkSimulationContext_addViewport: GET_NATIVE_WINDOW_HANDLE_CALLBACK not registered \
           or returned a null handle",
        );
        return false;
      }
    }
  } else {
    // ── Windowless path (backward compat) ─────────────────────────────────
    ctx_ref.create_presentation_engine(scene_id, width, height).and_then(|pe_id| {
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
    })
  };

  match result {
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
// TODO handle windowed, main thread callback to delete swapchain and surface. Shuld rely on either
// timelnie semaphore or present fence. (Actually queue wait idle is affordable here cause you don't
// do this often)
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

/// Enum to simplify management of [`avkSimulationContext_modifyComponent`] command
#[repr(u32)]
enum ModifyComponentCommand {
  Add = 1,
  Edit = 2,
  Remove = 3,
}
impl ModifyComponentCommand {
  fn from_u32(value: u32) -> Option<Self> {
    match value {
      1 => Some(Self::Add),
      2 => Some(Self::Edit),
      3 => Some(Self::Remove),
      _ => None,
    }
  }
}

/// - `in_dto` should be processed as a [`aethervk_core_rlib::scene::ErasedForeignSerializable`]
/// - Buffers are expected to be 8 byte aligned
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_modifyComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  command: u32,
  in_dto: *const core::ffi::c_void,
  out_computed_dto: *mut core::ffi::c_void,
) -> bool {
  // null check
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { ctx.as_ref().unwrap() };
  // command validity check
  match ModifyComponentCommand::from_u32(command) {
    Some(ModifyComponentCommand::Remove) => {}
    Some(ModifyComponentCommand::Add) | Some(ModifyComponentCommand::Edit) => {
      if in_dto.is_null() {
        return false;
      }
    }
    None => return false,
  }
  // execution
  todo!()
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_debugECSPrint(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_count: u32,
  entity_ids: *const u64,
  comp_count: u32,
  comps: *const u64,
) {
  #[cfg(debug_assertions)]
  {
    if ctx.is_null()
      || entity_count == 0
      || comp_count == 0
      || entity_ids.is_null()
      || comps.is_null()
    {
      return;
    }
    let ctx_ref = unsafe { &*ctx };
    let e_ids = unsafe { core::slice::from_raw_parts(entity_ids, entity_count as usize) };
    let c_ids = unsafe { core::slice::from_raw_parts(comps, comp_count as usize) };

    if let Some(scene_arc) = ctx_ref.scenes.read().get_scene(scene_id) {
      let scene_guard = scene_arc.read();
      for &e_id in e_ids {
        let entity = aethervk_core_rlib::scene::EntityId::from_ffi(e_id);
        oshal::log!("--- Entity {} ---", e_id);
        for &c_id in c_ids {
          if c_id == 1 {
            // HighResTransform (ComponentForeignId = 1) -> Print global f64 transform in km
            if let Some(global_t) = scene_guard.scene.global_transform_f64(entity) {
              const AU_TO_KM: f64 = 149_597_870.7_f64;
              let pos_km = global_t.position * AU_TO_KM;
              oshal::log!(
                "Global Transform (km): pos=({:.2}, {:.2}, {:.2}), scale=({:.2e}, {:.2e}, {:.2e})",
                pos_km.x(),
                pos_km.y(),
                pos_km.z(),
                global_t.scale.x(),
                global_t.scale.y(),
                global_t.scale.z()
              );
            } else {
              oshal::log!("Global Transform not available");
            }
            continue;
          }
          if c_id == u64::MAX {
            fn print_entity_tree(
                scene: &aethervk_core_rlib::scene::Scene,
                entity: aethervk_core_rlib::scene::EntityId,
                depth: usize,
            ) {
                let name = scene.get_name(entity).unwrap_or_else(|| "Unknown".into());
                let mut comp_names = scene.get_entity_component_names(entity);
                comp_names.sort();
                let indent = "  ".repeat(depth);
                oshal::log!("{}- Entity {} '{}' [{}]", indent, entity.as_ffi(), name, comp_names.join(", "));
                if let Some(children) = scene.get_children(entity) {
                    for child in children {
                        print_entity_tree(scene, child, depth + 1);
                    }
                }
            }
            let root_to_print = scene_guard.scene.get_parent(entity).unwrap_or(entity);
            oshal::log!("--- Subtree for {} ---", e_id);
            print_entity_tree(&scene_guard.scene, root_to_print, 0);
            
            let name = scene_guard.scene.get_name(root_to_print).unwrap_or_default();
            if let Some(prefix) = name.strip_suffix("_subtree") {
                let orbit_name = alloc::format!("{}_orbit", prefix);
                if let Some(orbit_entity) = scene_guard.scene.get_entity_by_name(&orbit_name) {
                    print_entity_tree(&scene_guard.scene, orbit_entity, 0);
                }
            }
            continue;
          }
          if let Some(_) = scene_guard.scene.with_component_mut_by_id(entity, c_id, |erased| {
            erased.debug_print();
          }) {
            // Successfully printed
          } else {
            oshal::log!("Component {} not found", c_id);
          }
        }
      }
    }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
pub struct AnimationTargetDTO {
  pub pos_x: f32,
  pub pos_y: f32,
  pub pos_z: f32,
  pub rot_x: f32,
  pub rot_y: f32,
  pub rot_z: f32,
  pub rot_w: f32,
  pub duration_s: f32,
}

/// Why is this separate from modifyComponent? because this explicitly calls
/// [`aethervk_core_rlib::scene::animation::TransformAnimationComponent`] `retarget` method in case
/// the previous animation didn't fully play out yet
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addCameraAnimation(
  ctx: *mut SimulationContext,
  scene_id: u64,
  camera_id: u64,
  animation: *const AnimationTargetDTO,
) -> bool {
  // null check
  if ctx.is_null() || animation.is_null() {
    return false;
  }
  let ctx_ref = unsafe { ctx.as_ref().unwrap_unchecked() };
  let anim = unsafe { animation.as_ref().unwrap_unchecked() };

  use aethervk_core_rlib::simulation_api::structs::LogicCommand;
  use aethervk_oshal_rlib::math::vector::{Vector3, vec3f64::DVec3, vec4::Quat};

  let target_pos = DVec3::from_components(anim.pos_x as f64, anim.pos_y as f64, anim.pos_z as f64);
  let target_rot = Quat::from_components(anim.rot_x, anim.rot_y, anim.rot_z, anim.rot_w);

  ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(LogicCommand::AnimateCameraTo {
      scene_id,
      camera_id,
      target_pos,
      target_rot,
      duration_s: anim.duration_s,
    })
    .is_ok()
}

#[derive(Debug)]
pub enum TransformStaticCamera<'a> {
  /// left | right | bottom | top | near | far
  ProjectionOrtho(&'a [f32; 6]),
  /// fov | aspect ratio | near | far
  ProjectionPersp(&'a [f32; 4]),
  /// disp_x | disp_y | disp_z | quat_x | quat_y | quat_z | quat_w
  RotoTranslate(&'a [f32; 7]),
}
impl<'a> TransformStaticCamera<'a> {
  pub fn from_mode_and_buffer(mode: i32, buffer: *const core::ffi::c_void) -> Option<Self> {
    match mode {
      0 => Some(Self::ProjectionOrtho(unsafe {
        &*buffer.cast::<[f32; 6]>()
      })),
      1 => Some(Self::ProjectionPersp(unsafe {
        &*buffer.cast::<[f32; 4]>()
      })),
      2 => Some(Self::RotoTranslate(unsafe { &*buffer.cast::<[f32; 7]>() })),
      _ => None,
    }
  }

  pub fn as_camera_projection(&self) -> Option<aethervk_core_rlib::scene::CameraProjection> {
    use aethervk_core_rlib::scene::CameraProjection;
    match self {
      Self::ProjectionOrtho(arr) => Some(CameraProjection::Orthographic {
        left: arr[0],
        right: arr[1],
        bottom: arr[2],
        top: arr[3],
        near: arr[4],
        far: arr[5],
      }),
      Self::ProjectionPersp(arr) => Some(CameraProjection::Perspective {
        fov: arr[0],
        aspect_ratio: arr[1],
        near: arr[2],
        far: arr[3],
      }),
      Self::RotoTranslate(_) => None,
    }
  }

  /// casting to double precision cause camera always uses it
  pub fn as_srt_transform(&self) -> Option<aethervk_core_rlib::scene::HighResTransformComponent> {
    use aethervk_core_rlib::scene::HighResTransformComponent;
    use aethervk_oshal_rlib::math::vector::{
      Vector3, Vector4, vec3::Vec3f32, vec3f64::DVec3, vec4::Vec4f32,
    };
    match self {
      Self::ProjectionPersp(_) | Self::ProjectionOrtho(_) => None,
      Self::RotoTranslate(arr) => Some(HighResTransformComponent {
        position: DVec3::from_components(arr[0] as f64, arr[1] as f64, arr[2] as f64),
        rotation: Quat(Vec4f32::from_components(arr[3], arr[4], arr[5], arr[6])),
        scale: Vec3f32::one(),
      }),
    }
  }
}

/// modify either transform or projection of a "camera entity"
///
/// # Safety
/// FFI Contract:
/// - `mode = 0` then `buffer` is 4 byte aligned and points to [f32; 6]
/// - `mode = 1` then `buffer` is 4 byte aligned and points to [f32; 4]
/// - `mode = 2` then `buffer` is 4 byte aligned and points to [f32; 7]
///
/// Constraints for buffer
/// - Alignment: buffer must be properly aligned for an f32 (4-byte alignment).
/// - Size/Initialization: The memory buffer points to must contain at least 24 bytes (6 × 4 bytes) of initialized, valid f32 data.
/// - Lifetimes: The data must remain valid and unmodified for the entire lifetime of the resulting &[f32; 6] reference.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_transformStaticCamera(
  ctx: *mut SimulationContext,
  scene_id: u64,
  camera_id: u64,
  mode: i32,
  buffer: *const core::ffi::c_void,
) -> bool {
  // null check
  if ctx.is_null() || buffer.is_null() || mode > 2 {
    return false;
  }
  // cast and return if error
  let transform_request = match TransformStaticCamera::from_mode_and_buffer(mode, buffer) {
    Some(value) => value,
    None => return false,
  };
  // execution
  let ctx_ref = unsafe { ctx.as_ref().unwrap_unchecked() };

  use aethervk_core_rlib::simulation_api::structs::LogicCommand;

  let transform = transform_request.as_srt_transform().map(|srt| (srt.position, srt.rotation));
  let projection = transform_request.as_camera_projection();

  ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(LogicCommand::SetCameraTransform {
      scene_id,
      camera_id,
      transform,
      projection,
    })
    .is_ok()
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_resetSimulationSync(
  ctx: *mut SimulationContext,
  scene_id: u64,
) -> bool {
  // null check
  if ctx.is_null() {
    return false;
  }
  // execution
  //   Notes for rlib implementation: (first 3 steps are in common with pauseSimulationSync)
  //   - check if scene exists and that SimSpeed for its time manager is not Paused
  //   - sets the simspeed to paused
  //   - explicitly waits for the next self sync and cross sync
  //   - reset current epoch to start epoch, discard accumulator in time state
  //   - restore snapshot command and wait for its conclusion (spin wait atomic flag)
  todo!()
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_pauseSimulationSync(
  ctx: *mut SimulationContext,
  scene_id: u64,
) -> bool {
  // null check
  if ctx.is_null() {
    return false;
  }
  // execution
  todo!()
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_startSimulation(
  ctx: *mut SimulationContext,
  scene_id: u64,
  speed: i32,
) -> bool {
  use aethervk_oshal_rlib::os::time::v2::SimSpeed;
  // null check
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { ctx.as_ref().unwrap_unchecked() };
  // sim speed conversion check
  let sim_speed = SimSpeed::from(speed);
  match sim_speed {
    SimSpeed::Realtime | SimSpeed::Paused | SimSpeed::Custom(_) => {
      oshal::log!("Invalid simulation speed value {}", speed);
      return false;
    }
    _ => {}
  };
  // execution
  //   Notes for rlib implementation:
  //   - check that scene exists and that is paused: invariant for paused scenes: no active ptask
  //     debug assert that
  //   - check scene integrity for simulation: comet and earth subtrees exist, and comet subtree has
  //     at least 1 particle system fully configured
  //   - write lock on time manager and time state and change sim speed.
  todo!()
}

/// Propagates the "Jet Common Parameters" from `ps_dto` to all sibling jet entities
/// (other children of `parent_entity` that have a `ParticleSystemComponent`), excluding
/// `skip_entity` (the jet just created or currently being modified).
///
/// Common params: mass_variability_perc, diametre_um, density_gcm3, scattering_efficiency,
/// afrho_0_cm, afrho_power, afrho_cutoff_au, afrho_max_value_cm.
fn propagate_common_params(
  scene: &aethervk_core_rlib::scene::Scene,
  parent_entity: aethervk_core_rlib::scene::EntityId,
  ps_dto: &ParticleSystemDTO,
  skip_entity: aethervk_core_rlib::scene::EntityId,
) {
  let children = match scene.get_children(parent_entity) {
    Some(c) => c,
    None => return,
  };
  for child in children {
    if child == skip_entity {
      continue;
    }
    scene.with_component_mut(
      child,
      |ps: &mut aethervk_core_rlib::scene::particles::ParticleSystemComponent| {
        let ep = &mut ps.emission_params;
        ep.mass_variability_perc = ps_dto.mass_variability_perc;
        ep.diametre_um = ps_dto.diametre_um;
        ep.density_gcm3 = ps_dto.density_gcm3;
        ep.scattering_efficiency = ps_dto.scattering_efficiency;
        ep.afrho_0_cm = ps_dto.afrho_0_cm;
        ep.afrho_power = ps_dto.afrho_power;
        ep.afrho_cutoff_au = ps_dto.afrho_cutoff_au;
        ep.afrho_max_value_cm = ps_dto.afrho_max_value_cm;
      },
    );
  }
}

/// Why isn't this included in modifyComponent? Because this adds a new child entity. Furthermore it
/// picks up "Jet Common Parameters" to sibling particle systems if present.
///
/// Returns computed properties only if you ask for them, by giving a non-null out pointer to a
/// computed DTO
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addParticleSystem(
  ctx: *mut SimulationContext,
  scene_id: u64,
  particle_system: *const ParticleSystemDTO,
  out_ps_id: *mut u64,
  out_ps_computed_props: *mut ParticleSystemComputedDTO,
) -> bool {
  if ctx.is_null() || particle_system.is_null() {
    return false;
  }
  let ctx_ref = unsafe { ctx.as_ref().unwrap_unchecked() };
  let ps_dto = unsafe { particle_system.as_ref().unwrap_unchecked() };

  if !out_ps_id.is_null() {
    unsafe { *out_ps_id = 0 };
  }

  let scenes = ctx_ref.scenes.read();
  let scene_arc = match scenes.get_scene(scene_id) {
    Some(s) => s,
    None => return false,
  };
  let scene_guard = scene_arc.read();
  let comet = match scene_guard.comet {
    Some(c) => c,
    None => {
      emit_breadcrumb(1, "avkSimulationContext_addParticleSystem: comet not found");
      return false;
    }
  };

  let lat = ps_dto.latitude_rad;
  let lon = ps_dto.longitude_rad;
  let r = ps_dto.nucleus_radius_km;
  let pos = Vec3f32::from_components(
    r * lat.cos() * lon.cos(),
    r * lat.cos() * lon.sin(),
    r * lat.sin(),
  );

  let jet_entity = scene_guard.scene.spawn_entity("jet");
  scene_guard.scene.set_parent(jet_entity, Some(comet.body));

  let _ = scene_guard.scene.add_component(
    jet_entity,
    aethervk_core_rlib::scene::TransformComponent {
      position: pos,
      rotation: Quat::identity(),
      scale: Vec3f32::from_components(0.05, 0.05, 0.05),
    },
  );

  let sphere_mesh = alloc::sync::Arc::new(
    aethervk_core_rlib::simulation::comet::generate_uv_sphere(1.0, 8, 8, 1.0, false),
  );
  let _ = scene_guard.scene.add_component(
    jet_entity,
    aethervk_core_rlib::scene::StaticMeshComponent {
      asset_path: alloc::string::String::from("__jet_marker__"),
      mesh: sphere_mesh,
      emissive_color: ps_dto.stream_color,
      is_visible: true,
    },
  );
  debug_assert!(
    scene_guard
      .scene
      .has_component::<aethervk_core_rlib::scene::StaticMeshComponent>(jet_entity)
      == aethervk_core_rlib::scene::HasComponentResultEnum::EntityHasComponent,
    "jet entity must have StaticMeshComponent before ParticleSystemComponent is created"
  );

  let emit_params = aethervk_core_rlib::scene::particles::ParticleSystemEmitParams {
    mass_variability_perc: ps_dto.mass_variability_perc,
    diametre_um: ps_dto.diametre_um,
    density_gcm3: ps_dto.density_gcm3,
    scattering_efficiency: ps_dto.scattering_efficiency,
    afrho_0_cm: ps_dto.afrho_0_cm,
    afrho_power: ps_dto.afrho_power,
    afrho_cutoff_au: ps_dto.afrho_cutoff_au,
    afrho_max_value_cm: ps_dto.afrho_max_value_cm,
    latitude_rad: ps_dto.latitude_rad,
    longitude_rad: ps_dto.longitude_rad,
    aperture_rad: ps_dto.aperture_rad,
    start_velocity_mean: ps_dto.start_velocity_mean,
    start_velocity_std: ps_dto.start_velocity_std,
    seed: ps_dto.seed,
  };

  let draw_params = aethervk_core_rlib::scene::particles::ParticleSystemDrawParams {
    stream_color: ps_dto.stream_color,
  };

  // TODO Add this as a function parameter in the jet common area
  const JET_TTL_US: aethervk_oshal_rlib::os::time::timeus_t = 60_000_000_000i64;

  let render_frontend = match ctx_ref.render_frontend() {
    Some(rf) => rf,
    None => {
      emit_breadcrumb(
        1,
        "avkSimulationContext_addParticleSystem: render frontend not found",
      );
      scene_guard.scene.remove_entity(jet_entity);
      return false;
    }
  };
  let render_device_handle = ctx_ref.render_device_handle();

  let ps_component = match aethervk_core_rlib::scene::particles::ParticleSystemComponent::new(
    render_frontend,
    render_device_handle,
    jet_entity,
    emit_params,
    draw_params,
    JET_TTL_US,
  ) {
    Ok(c) => c,
    Err(e) => {
      emit_breadcrumb(
        1,
        &alloc::format!(
          "avkSimulationContext_addParticleSystem: failed to create particle system component: {}",
          e
        ),
      );
      scene_guard.scene.remove_entity(jet_entity);
      return false;
    }
  };
  let _ = scene_guard.scene.add_component(jet_entity, ps_component);

  if !out_ps_id.is_null() {
    unsafe { *out_ps_id = jet_entity.as_ffi() };
  }
  if !out_ps_computed_props.is_null() {
    unsafe {
      (*out_ps_computed_props).beta = emit_params.beta();
      (*out_ps_computed_props).dust_production_rate_at_1au_kgs =
        emit_params.dust_production_rate_kgs(1.0);
    }
  }

  propagate_common_params(&scene_guard.scene, comet.body, ps_dto, jet_entity);
  true
}

/// Why isn't thid included in modifyComponent? Because this also propagates changes in
/// "Jet Common Parameters" to sibling particle systems if present. Doesn't return new value of
/// common parameters cause if they come from Avalonia, then UI update is caller responsiblity
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_modifyParticleSystem(
  ctx: *mut SimulationContext,
  scene_id: u64,
  ps_id: u64,
  particle_system: *const ParticleSystemDTO,
  out_ps_computed_props: *mut ParticleSystemComputedDTO,
) -> bool {
  if ctx.is_null() || particle_system.is_null() {
    return false;
  }
  let ctx_ref = unsafe { ctx.as_ref().unwrap_unchecked() };
  let ps_dto = unsafe { particle_system.as_ref().unwrap_unchecked() };

  let scenes = ctx_ref.scenes.read();
  let scene_arc = match scenes.get_scene(scene_id) {
    Some(s) => s,
    None => return false,
  };
  let scene_guard = scene_arc.read();

  let jet_eid = aethervk_core_rlib::scene::EntityId::from_ffi(ps_id);

  let lat = ps_dto.latitude_rad;
  let lon = ps_dto.longitude_rad;
  let r = ps_dto.nucleus_radius_km;
  let pos = Vec3f32::from_components(
    r * lat.cos() * lon.cos(),
    r * lat.cos() * lon.sin(),
    r * lat.sin(),
  );

  scene_guard.scene.with_component_mut(
    jet_eid,
    |t: &mut aethervk_core_rlib::scene::TransformComponent| {
      t.position = pos;
    },
  );

  scene_guard.scene.with_component_mut(
    jet_eid,
    |s: &mut aethervk_core_rlib::scene::StaticMeshComponent| {
      s.emissive_color = ps_dto.stream_color;
    },
  );

  let emit_params = aethervk_core_rlib::scene::particles::ParticleSystemEmitParams {
    mass_variability_perc: ps_dto.mass_variability_perc,
    diametre_um: ps_dto.diametre_um,
    density_gcm3: ps_dto.density_gcm3,
    scattering_efficiency: ps_dto.scattering_efficiency,
    afrho_0_cm: ps_dto.afrho_0_cm,
    afrho_power: ps_dto.afrho_power,
    afrho_cutoff_au: ps_dto.afrho_cutoff_au,
    afrho_max_value_cm: ps_dto.afrho_max_value_cm,
    latitude_rad: ps_dto.latitude_rad,
    longitude_rad: ps_dto.longitude_rad,
    aperture_rad: ps_dto.aperture_rad,
    start_velocity_mean: ps_dto.start_velocity_mean,
    start_velocity_std: ps_dto.start_velocity_std,
    seed: ps_dto.seed,
  };

  scene_guard.scene.with_component_mut(
    jet_eid,
    |p: &mut aethervk_core_rlib::scene::ParticleSystemComponent| {
      p.emission_params = emit_params;
      p.draw_params.stream_color = ps_dto.stream_color;
    },
  );

  if !out_ps_computed_props.is_null() {
    unsafe {
      (*out_ps_computed_props).beta = emit_params.beta();
      (*out_ps_computed_props).dust_production_rate_at_1au_kgs =
        emit_params.dust_production_rate_kgs(1.0);
    }
  }

  if let Some(comet) = scene_guard.comet {
    propagate_common_params(&scene_guard.scene, comet.body, ps_dto, jet_eid);
  }
  true
}

/// Removes a jet (particle system) entity from the scene.
/// `ParticleSystemComponent::drop` handles deferred GPU resource deallocation via timeline semaphores.
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_removeParticleSystem(
  ctx: *mut SimulationContext,
  scene_id: u64,
  ps_id: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { ctx.as_ref().unwrap_unchecked() };
  let scenes = ctx_ref.scenes.read();
  let scene_arc = match scenes.get_scene(scene_id) {
    Some(s) => s,
    None => return false,
  };
  let scene_guard = scene_arc.read();
  let jet_eid = aethervk_core_rlib::scene::EntityId::from_ffi(ps_id);
  scene_guard.scene.remove_entity(jet_eid);
  true
}

/// Reconfigures the comet entity in the simulation scene.
///
/// `command_flags` is a bitmask:
/// - `0` — query only: write comet body entity id to `out_comet_id` and return.
/// - `0x2` (DETACH) — remove `AlmanacPlanet` and `TrajectoryComponent`, reset comet to
///   1 AU +X default placement.
///
/// Returns `false` if `ctx` is null or the command could not be enqueued.
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_reconfigureComet(
  ctx: *mut SimulationContext,
  scene_id: u64,
  command_flags: i32,
  spk_id: i32,
  out_comet_id: *mut u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };

  // Always populate out_comet_id from the stored SubtreeEntities if available.
  if !out_comet_id.is_null() {
    let scenes = ctx_ref.scenes.read();
    if let Some(scene_arc) = scenes.get_scene(scene_id)
      && let Some(comet) = scene_arc.read().comet
    {
      unsafe { *out_comet_id = comet.body.as_ffi() };
    }
  }

  if command_flags == 0 {
    return true; // query only
  }

  if command_flags & 0x2 != 0 {
    // DETACH — remove AlmanacPlanet + TrajectoryComponent, reset comet subtree.
    let _ = ctx_ref
      .threads
      .logic_thread
      .tx()
      .try_send(structs::LogicCommand::CleanupComet { scene_id });
    return true;
  }

  true
}

/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_tryInitComet(
  ctx: *mut SimulationContext,
  scene_id: u64,
  spk_id: i32,
  proposed_range: *const external_state::CTimeRange,
  keplerian_elements: *const CKeplerianElementsDTO,
  out_comet_id: *mut u64,
) -> bool {
  if ctx.is_null() || proposed_range.is_null() || keplerian_elements.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };

  if !out_comet_id.is_null() {
    let scenes = ctx_ref.scenes.read();
    if let Some(scene_arc) = scenes.get_scene(scene_id)
      && let Some(comet) = scene_arc.read().comet
    {
      unsafe { *out_comet_id = comet.body.as_ffi() };
    }
  }

  let range = unsafe { *proposed_range };
  let dto = unsafe { *keplerian_elements };
  let elements = KeplerianElements {
    eccentricity: dto.eccentricity,
    perihelion_distance_au: dto.perihelion_distance_au,
    inclination_deg: dto.inclination_deg,
    longitude_of_ascending_node_deg: dto.longitude_of_ascending_node_deg,
    argument_of_perihelion_deg: dto.argument_of_perihelion_deg,
  };
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::TryInitComet {
    scene_id,
    spk_id,
    proposed_start: anise::time::Epoch::from_tdb_duration(anise::time::Duration::from_parts(range.centuries[0], range.nanoseconds[0])),
    proposed_end: anise::time::Epoch::from_tdb_duration(anise::time::Duration::from_parts(range.centuries[1], range.nanoseconds[1])),
    keplerian_elements: elements,
  });

  true
}

/// Synchronously updates the `BodyRotationalModel` ECS component on the comet body entity.
///
/// Safe to call from any thread — the write is protected by the scene's internal lock.
/// The logic thread reads `BodyRotationalModel` on each step tick so changes take effect
/// within one logic frame (~16 ms at 60 Hz).
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setBodyRotationalModel(
  ctx: *mut SimulationContext,
  scene_id: u64,
  comet_body_entity: u64,
  dto: *const CBodyRotationalModelDTO,
) -> bool {
  if ctx.is_null() || dto.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let dto_ref = unsafe { &*dto };

  let model = aethervk_core_rlib::scene::BodyRotationalModel {
    pole_ra: dto_ref.pole_ra_deg,
    pole_dec: dto_ref.pole_dec_deg,
    prime_meridian: dto_ref.prime_meridian_deg,
    pole_ra_rate: dto_ref.pole_ra_rate_deg_cen,
    pole_dec_rate: dto_ref.pole_dec_rate_deg_cen,
    rotation_rate: dto_ref.rot_rate_deg_day,
  };

  let scenes = ctx_ref.scenes.read();
  if let Some(scene_arc) = scenes.get_scene(scene_id) {
    let scene_guard = scene_arc.read();
    // Try to update an existing BodyRotationalModel component.
    // If none exists yet, add it.
    let entity = scene_guard.get_entity(comet_body_entity);
    if let Some(entity_id) = entity {
      let updated = scene_guard.scene.with_component_mut(
        entity_id,
        |m: &mut aethervk_core_rlib::scene::BodyRotationalModel| {
          *m = model;
        },
      );
      if updated.is_none() {
        // Component absent — add it
        let _ = scene_guard.scene.add_component(entity_id, model);
      }
      return true;
    }
  }
  false
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
  // TODO: this should be sync
  let _ = ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::UnloadAlmanac {
      task_id: 1,
      path: path_str,
    });
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

/// Registers the C# callback that Rust calls (synchronously, from a non-UI thread) to obtain
/// the OS native window handle when creating a windowed presentation engine.
///
/// Must be installed before the first call to `avkSimulationContext_addViewport` with
/// `handle_type > 0`. The callback marshals to the C# UI thread, fills the handle, and
/// signals an `AtomicBool` so Rust can stop spin-waiting.
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSetGetNativeWindowHandleCallback(
  cb: Option<GetNativeWindowHandleCallback>,
) {
  register_get_native_window_handle_callback(cb);
}

/// the C# caller should be responsible for polling getTaskStatus and only calling download_image once the status is completed.
/// Given the architecture of the C# async API and the fact that Vulkan windowless downloads are placed in persistently mapped memory buffers, bouncing the download request through the RenderCommand channel was actually an anti-pattern. Here is why:
/// 1. Thread Safety: is_task_completed and read_windowless_download only acquire read/write locks (self.res.read() and pending_downloads.write()). They don't actually need to execute on the Render Thread.
/// 2. No Render Thread Blocking: By having C# call download_image synchronously after polling confirms it's ready, the memory copy happens on the caller thread (e.g., C#'s worker pool), freeing the Render Thread to continue pushing Vulkan commands without stalling
///    during the memory copy.
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
pub unsafe extern "C" fn avkSimulationContext_removeScreenSpaceBillboard(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
) -> bool {
  // null check
  // see if returned thing has a billboard, if so remove it, otherwise breadcrumb and false
  todo!()
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
  let tai_ref = unsafe { tai.as_ref().unwrap_unchecked() };
  let start = anise::time::Epoch::from_tdb_duration(anise::time::Duration::from_parts(tai_ref.centuries[0], tai_ref.nanoseconds[0]));
  let end = anise::time::Epoch::from_tdb_duration(anise::time::Duration::from_parts(tai_ref.centuries[1], tai_ref.nanoseconds[1]));
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

  let tai_cov = unsafe { tai.as_ref().unwrap_unchecked() };
  let start = anise::time::Epoch::from_tdb_duration(anise::time::Duration::from_parts(tai_cov.centuries[0], tai_cov.nanoseconds[0]));
  let end = anise::time::Epoch::from_tdb_duration(anise::time::Duration::from_parts(tai_cov.centuries[1], tai_cov.nanoseconds[1]));

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
  let tai = unsafe { tai_parts.as_ref().unwrap_unchecked() };
  let start_epoch = anise::time::Epoch::from_tdb_duration(anise::time::Duration::from_parts(tai.centuries[0], tai.nanoseconds[0]));
  let end_epoch = anise::time::Epoch::from_tdb_duration(anise::time::Duration::from_parts(tai.centuries[1], tai.nanoseconds[1]));
  let (covers, domain, discovered_id) =
    AlmanacPackedData::probe_spk_file_with_domain(path_str, spk_id, start_epoch, end_epoch);

  if let Some((ds, de)) = domain
    && !out_domain_tai_parts.is_null()
  {
    unsafe { *out_domain_tai_parts = CTimeRange::new(ds, de) };
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
  use aethervk_core_rlib::scene::EntityId;

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
      let count = scene_ctx.scene.entity_count() as u32;
      if !out_count.is_null() {
        unsafe {
          *out_count = count;
        }
      }
      if capacity < count || out_buffer.is_null() {
        return false;
      }

      let mut idx = 0_usize;
      scene_ctx.scene.for_each_entity(|id| {
        let parent_id =
          scene_ctx.scene.get_parent(id).map(|id| EntityId::as_ffi(&id)).unwrap_or(0_u64);
        let dto = SceneHierarchyNodeDTO {
          entity_id: EntityId::as_ffi(&id),
          parent_id,
        };
        unsafe {
          out_buffer.add(idx).write(dto);
          idx += 1;
        }
      });

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
  /// FFI Contract: Must be allocated with [`avkSimulationContext_getEntityComponentNames`]
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
  /// FFI Contract: Must be freed with [`avkSimulationContext_freeComponentNames`]
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

// ── RenderDoc in-application API (debug builds only) ─────────────────────────

/// Returns `1` if the process was launched under RenderDoc and the in-app
/// capture API was successfully acquired, `0` otherwise.
///
/// Call once at startup (or lazily) to determine whether to show the capture
/// button in the UI.  The first call performs the one-time library probe;
/// subsequent calls return the cached result instantly.
///
/// # Safety
/// FFI Contract — no preconditions beyond a valid calling environment.
#[cfg(debug_assertions)]
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkDebug_isRenderDocAvailable() -> u8 {
  if aethervk_core_rlib::gpu_backends::vulkan::renderdoc::is_available() {
    1
  } else {
    0
  }
}

/// Requests RenderDoc to save the next presented frame to a `.rdc` capture file.
///
/// Equivalent to pressing F12 in the RenderDoc UI.  No-op (and safe to call)
/// if RenderDoc is not loaded into this process.
///
/// # Safety
/// FFI Contract — no preconditions beyond a valid calling environment.
#[cfg(debug_assertions)]
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkDebug_triggerCapture() {
  aethervk_core_rlib::gpu_backends::vulkan::renderdoc::trigger_capture();
}

/// Queues a scoped RenderDoc capture for the **next rendered frame** of the
/// given windowed presentation engine.  Only that swapchain is captured —
/// Avalonia's own rendering queues are unaffected.
///
/// Returns 1 on success, 0 if `ctx` is null, RenderDoc is unavailable, or
/// the render thread channel is full / not yet started.
///
/// # Safety
/// `ctx` must be a valid non-null pointer to a live `SimulationContext`.
#[cfg(debug_assertions)]
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkDebug_startScopedCapture(
  ctx: *mut SimulationContext,
  pe_id: u64,
) -> u8 {
  use aethervk_core_rlib::gpu_backends::vulkan::renderdoc;
  use aethervk_core_rlib::simulation_api::render_thread::channel_utils;

  if ctx.is_null() || !renderdoc::is_available() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let pe_handle = gpu::PresentationEngineHandle(pe_id);
  let cmd = RenderCommand::CaptureNextFrame { pe_handle };
  let Some(tx) = ctx_ref.threads.render_thread.tx_opt() else {
    return 0;
  };
  if channel_utils::retry_with_limit(tx, cmd, 10, core::time::Duration::from_millis(1)) {
    1
  } else {
    0
  }
}
/// Debug-only telemetry DTO. All fields are fixed-size for stable C ABI.
/// In release builds this type and its associated FFI function are not compiled.
#[cfg(debug_assertions)]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CDebugTelemetryStatsDTO {
  /// RSS physical RAM from /proc/self/statm (bytes)
  pub os_physical_ram_bytes: u64,
  /// Virtual address space from /proc/self/statm (bytes)
  pub os_virtual_ram_bytes: u64,
  /// CPU heap allocated via TrackingAllocator (bytes)
  pub cpu_allocated_bytes: u64,
  /// GPU VRAM allocated via VMA tracking (bytes)
  pub gpu_allocated_bytes: u64,
  /// Last logic-thread loop wall time (ms)
  pub logic_thread_cpu_time_ms: f64,
  /// Last render-thread CPU submission time (ms, NOT GPU execution time)
  pub render_thread_cpu_time_ms: f64,
  /// Reserved for future VK_QUERY_TYPE_TIMESTAMP GPU timing
  pub reserved_gpu_execution_ms: f64,
}

/// Query engine-side debug telemetry statistics.
/// Only available in debug builds (`cfg(debug_assertions)`).
/// In release builds this symbol is not exported.
///
/// # Safety
/// `out` must be a valid non-null pointer to a `CDebugTelemetryStatsDTO`.
/// The caller is responsible for ensuring `out` is properly aligned and sized.
#[cfg(debug_assertions)]
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getDebugTelemetryStats(
  _ctx: isize,
  out: *mut CDebugTelemetryStatsDTO,
) -> bool {
  if out.is_null() {
    return false;
  }

  use aethervk_oshal_rlib::os::memory;
  use core::sync::atomic::Ordering;

  let proc_mem = memory::query_process_memory();
  let cpu_bytes = memory::tracking::CPU_ALLOCATED.load(Ordering::Relaxed) as u64;
  let gpu_bytes = memory::tracking::GPU_ALLOCATED.load(Ordering::Relaxed) as u64;

  let logic_ms = f64::from_bits(
    aethervk_core_rlib::simulation_api::logic_thread::DEBUG_LOGIC_THREAD_TIME_MS
      .load(Ordering::Relaxed),
  );
  let render_ms = f64::from_bits(
    aethervk_core_rlib::gpu_backends::vulkan::DEBUG_RENDER_THREAD_CPU_TIME_MS
      .load(Ordering::Relaxed),
  );

  let gpu_exec_ms = f64::from_bits(
    aethervk_core_rlib::gpu_backends::vulkan::DEBUG_RENDER_THREAD_GPU_TIME_MS
      .load(Ordering::Relaxed),
  );

  unsafe {
    *out = CDebugTelemetryStatsDTO {
      os_physical_ram_bytes: proc_mem.physical_bytes,
      os_virtual_ram_bytes: proc_mem.virtual_bytes,
      cpu_allocated_bytes: cpu_bytes,
      gpu_allocated_bytes: gpu_bytes,
      logic_thread_cpu_time_ms: logic_ms,
      render_thread_cpu_time_ms: render_ms,
      reserved_gpu_execution_ms: gpu_exec_ms,
    };
  }
  true
}

#[cfg(test)]
mod tests {
  use super::*;
  use anise::time::{Duration, Epoch};

  #[test]
  fn test_epoch_from_tdb_duration_is_j2000() {
    // 0 centuries, 0 nanoseconds should be EXACTLY J2000 TDB.
    let dur_zero = Duration::from_parts(0, 0);
    let epoch_zero = Epoch::from_tdb_duration(dur_zero);
    
    // JDE for J2000 is 2451545.0
    approx::assert_relative_eq!(epoch_zero.to_jde_tdb_days(), 2451545.0, epsilon = 1e-9);

    // Now test the user's specific duration: 812548800000000000 ns (~25.75 years)
    let dur = Duration::from_parts(0, 812548800000000000);
    let epoch = Epoch::from_tdb_duration(dur);
    
    // 812548800 seconds = 812548800 / 86400 = 9404.5 days
    // 2451545.0 + 9404.5 = 2460949.5
    approx::assert_relative_eq!(epoch.to_jde_tdb_days(), 2460949.5, epsilon = 1e-9);
  }
}
