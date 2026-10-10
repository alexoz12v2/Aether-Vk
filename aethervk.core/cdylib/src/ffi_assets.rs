//! FFI for the Imports tab (asset library) and the comet appearance (Settings tab).
//!
//! - `importAsset` is asynchronous (thread pool); completion arrives as the external state
//!   `AssetImported` (id 10) carrying the caller's `request_id`.
//! - The asset list is pulled with `getAssetCount` / `getAssetInfo` / `copyAssetThumbnail`.
//! - `setCometAppearance` validates synchronously (returns a [`AppearanceStatus`] code) and then
//!   queues `LogicCommand::SetCometAppearance`.

use aethervk_core_rlib::{
  simulation::asset_library::{AssetInfo, AssetKind, TextureChannel},
  simulation_api::{
    SimulationContext,
    comet_appearance::{
      CometAppearanceComponent, CometAppearanceWiring, CometDisplayMode, CometVisualOffset,
    },
    structs,
  },
};
use alloc::string::ToString;
use core::ffi::{CStr, c_char};

/// Flat description of an imported asset.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct CAssetInfoDTO {
  pub id: u64,
  /// Texture: width. Mesh: vertex count.
  pub a: u64,
  /// Texture: height. Mesh: index count.
  pub b: u64,
  /// Mesh only: textures bundled with the source file, per channel (albedo, normal, roughness,
  /// ao); `0` = none.
  pub bundled_textures: [u64; 4],
  /// `1` = mesh, `2` = texture.
  pub kind: u32,
  /// Texture only: channel suggested by the source (`0..=3`), `-1` = none.
  pub channel_hint: i32,
  /// Mesh only: `OrientationFix` applied at import (`0` none, `1` winding flipped,
  /// `2` inside-out fixed).
  pub orientation_fix: u32,
  pub _pad: u32,
}

/// Comet appearance, both directions.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct CCometAppearanceDTO {
  /// Mesh asset id, `0` = none.
  pub mesh: u64,
  /// Texture asset ids per channel (albedo, normal, roughness, ao), `0` = none.
  pub textures: [u64; 4],
  /// Intrinsic Z-Y-X rotation, degrees.
  pub yaw_pitch_roll_deg: [f32; 3],
  /// Translation in nucleus radii.
  pub translation_radii: [f32; 3],
  /// `0` = default procedural sphere, `1` = custom mesh.
  pub mode: u32,
  pub _pad: u32,
}

/// Memory accounting of the asset library.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct CAssetLibraryStatsDTO {
  /// File-backed (reclaimable) bytes of mapped cache files.
  pub mapped_bytes: u64,
  /// Anonymous heap bytes held by the library (thumbnails + strings).
  pub heap_bytes: u64,
  pub mesh_count: u32,
  pub texture_count: u32,
}

/// Result of [`avkSimulationContext_setCometAppearance`].
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppearanceStatus {
  /// Validated and queued.
  Queued = 0,
  /// Mode / mesh / texture wiring cannot change while the simulation plays.
  LockedWhileRunning = 1,
  /// A referenced asset does not exist (or has the wrong kind).
  UnknownAsset = 2,
  /// Bad arguments (null pointers, unknown mode, no comet in the scene).
  InvalidArgument = 3,
  /// The logic thread queue rejected the command.
  QueueFull = 4,
  /// An import is in progress; retry when `AssetImported` arrives.
  Busy = 5,
}

fn from_id(id: u64) -> Option<u64> {
  (id != 0).then_some(id)
}

fn to_id(id: Option<u64>) -> u64 {
  id.unwrap_or(0)
}

fn dto_to_appearance(
  dto: &CCometAppearanceDTO,
) -> Option<(CometAppearanceWiring, CometVisualOffset)> {
  let mode = CometDisplayMode::from_u32(dto.mode)?;
  let wiring = CometAppearanceWiring {
    mode,
    mesh: from_id(dto.mesh),
    textures: dto.textures.map(from_id),
  };
  let finite = dto
    .yaw_pitch_roll_deg
    .iter()
    .chain(dto.translation_radii.iter())
    .all(|v| v.is_finite());
  if !finite {
    return None;
  }
  let offset = CometVisualOffset {
    yaw_pitch_roll_deg: dto.yaw_pitch_roll_deg,
    translation_radii: dto.translation_radii,
  };
  Some((wiring, offset))
}

fn appearance_to_dto(a: &CometAppearanceComponent) -> CCometAppearanceDTO {
  CCometAppearanceDTO {
    mesh: to_id(a.wiring.mesh),
    textures: a.wiring.textures.map(to_id),
    yaw_pitch_roll_deg: a.offset.yaw_pitch_roll_deg,
    translation_radii: a.offset.translation_radii,
    mode: a.wiring.mode as u32,
    _pad: 0,
  }
}

fn info_to_dto(info: &AssetInfo) -> CAssetInfoDTO {
  CAssetInfoDTO {
    id: info.id,
    a: info.a,
    b: info.b,
    bundled_textures: info.bundled_textures.map(to_id),
    kind: info.kind as u32,
    channel_hint: info.channel_hint.map_or(-1, |c: TextureChannel| c as i32),
    orientation_fix: info.orientation_fix as u32,
    _pad: 0,
  }
}

/// Copies `s` into `out[..len]` NUL-terminated (truncating on a char boundary). Returns the
/// number of bytes `s` needs, excluding the terminator.
unsafe fn write_c_string(s: &str, out: *mut c_char, len: u32) -> u32 {
  if !out.is_null() && len > 0 {
    let mut n = s.len().min(len as usize - 1);
    while !s.is_char_boundary(n) {
      n -= 1;
    }
    unsafe {
      core::ptr::copy_nonoverlapping(s.as_ptr(), out.cast::<u8>(), n);
      *out.add(n) = 0;
    }
  }
  s.len() as u32
}

/// Starts an asynchronous import of a mesh (obj, ply, gltf, glb) or texture (png, jpg, jpeg,
/// ktx2). `cache_dir` receives the decoded, memory-mapped cache files (use the session folder).
///
/// # Safety
/// FFI Contract: `path` and `cache_dir` are NUL-terminated UTF-8 strings.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_importAsset(
  ctx: *mut SimulationContext,
  request_id: u64,
  path: *const c_char,
  cache_dir: *const c_char,
) -> bool {
  if ctx.is_null() || path.is_null() || cache_dir.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let (Ok(path), Ok(cache_dir)) = (
    unsafe { CStr::from_ptr(path) }.to_str(),
    unsafe { CStr::from_ptr(cache_dir) }.to_str(),
  ) else {
    return false;
  };
  if path.is_empty() || cache_dir.is_empty() {
    return false;
  }
  ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::ImportAsset {
      request_id,
      path: path.to_string(),
      cache_dir: cache_dir.to_string(),
    })
    .is_ok()
}

/// Number of imported assets (meshes first, then textures). Returns `u32::MAX` while an import
/// holds the library (retry after `AssetImported`).
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getAssetCount(ctx: *mut SimulationContext) -> u32 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let library = alloc::sync::Arc::clone(&ctx_ref.scenes.read().asset_library);
  let Some(library) = library.try_read() else {
    return u32::MAX;
  };
  let stats = library.stats();
  stats.mesh_count + stats.texture_count
}

/// Describes asset number `index` (`0..getAssetCount()`). `label` / `key` are optional output
/// buffers (NUL-terminated, truncated); `out_label_len` / `out_key_len` receive the full byte
/// lengths so the caller can retry with larger buffers.
///
/// # Safety
/// FFI Contract: buffers are valid for their stated lengths.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getAssetInfo(
  ctx: *mut SimulationContext,
  index: u32,
  out: *mut CAssetInfoDTO,
  label: *mut c_char,
  label_len: u32,
  out_label_len: *mut u32,
  key: *mut c_char,
  key_len: u32,
  out_key_len: *mut u32,
) -> bool {
  if ctx.is_null() || out.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let library = alloc::sync::Arc::clone(&ctx_ref.scenes.read().asset_library);
  let Some(library) = library.try_read() else {
    return false;
  };
  let assets = library.assets();
  let Some(info) = assets.get(index as usize) else {
    return false;
  };
  unsafe {
    *out = info_to_dto(info);
    let l = write_c_string(&info.label, label, label_len);
    if !out_label_len.is_null() {
      *out_label_len = l;
    }
    let k = write_c_string(&info.key, key, key_len);
    if !out_key_len.is_null() {
      *out_key_len = k;
    }
  }
  true
}

/// Copies the RGBA8 thumbnail of `asset_id`. Call with `out_rgba = null` to only query the
/// size (`out_width * out_height * 4` bytes).
///
/// # Safety
/// FFI Contract: `out_rgba` is valid for `out_len` bytes when non-null.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_copyAssetThumbnail(
  ctx: *mut SimulationContext,
  asset_id: u64,
  out_rgba: *mut u8,
  out_len: u32,
  out_width: *mut u32,
  out_height: *mut u32,
) -> bool {
  if ctx.is_null() || out_width.is_null() || out_height.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let library = alloc::sync::Arc::clone(&ctx_ref.scenes.read().asset_library);
  let Some(library) = library.try_read() else {
    return false;
  };
  let Some(thumb) = library.thumbnail(asset_id) else {
    return false;
  };
  unsafe {
    *out_width = thumb.width;
    *out_height = thumb.height;
  }
  if out_rgba.is_null() {
    return true;
  }
  if (out_len as usize) < thumb.rgba.len() {
    return false;
  }
  unsafe { core::ptr::copy_nonoverlapping(thumb.rgba.as_ptr(), out_rgba, thumb.rgba.len()) };
  true
}

/// Asset library memory accounting (mapped vs heap bytes).
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getAssetLibraryStats(
  ctx: *mut SimulationContext,
  out: *mut CAssetLibraryStatsDTO,
) -> bool {
  if ctx.is_null() || out.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let library = alloc::sync::Arc::clone(&ctx_ref.scenes.read().asset_library);
  let Some(library) = library.try_read() else {
    return false;
  };
  let s = library.stats();
  unsafe {
    *out = CAssetLibraryStatsDTO {
      mapped_bytes: s.mapped_bytes,
      heap_bytes: s.heap_bytes,
      mesh_count: s.mesh_count,
      texture_count: s.texture_count,
    };
  }
  true
}

/// Unloads an imported asset (only while every scene is paused). A comet displaying it is
/// ejected to the procedural sphere (mesh) or has the channel cleared (texture). Completion is
/// reported through the external state `AssetRemoved` (id 11) carrying `request_id`.
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_removeAsset(
  ctx: *mut SimulationContext,
  request_id: u64,
  asset_id: u64,
) -> bool {
  if ctx.is_null() || asset_id == 0 {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::RemoveAsset {
      request_id,
      asset_id,
    })
    .is_ok()
}

/// Requests a comet appearance change. Validation happens here, synchronously, so the UI gets
/// an immediate answer; the change itself is applied by the logic thread before the next frame.
///
/// # Safety
/// FFI Contract: `appearance` points to a valid `CCometAppearanceDTO`.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setCometAppearance(
  ctx: *mut SimulationContext,
  scene_id: u64,
  appearance: *const CCometAppearanceDTO,
) -> i32 {
  if ctx.is_null() || appearance.is_null() {
    return AppearanceStatus::InvalidArgument as i32;
  }
  let ctx_ref = unsafe { &*ctx };
  let Some((wiring, offset)) = dto_to_appearance(unsafe { &*appearance }) else {
    return AppearanceStatus::InvalidArgument as i32;
  };

  let (scene_arc, library) = {
    let scenes = ctx_ref.scenes.read();
    (
      scenes.get_scene(scene_id),
      alloc::sync::Arc::clone(&scenes.asset_library),
    )
  };
  let Some(scene_arc) = scene_arc else {
    return AppearanceStatus::InvalidArgument as i32;
  };
  {
    let scene_guard = scene_arc.read();
    let Some(visual) = scene_guard.comet.and_then(|c| c.visual) else {
      return AppearanceStatus::InvalidArgument as i32;
    };
    let current = scene_guard
      .scene
      .with_component(visual, |a: &CometAppearanceComponent| a.wiring)
      .unwrap_or_default();
    let running =
      scene_guard.time_state.read().speed != aethervk_oshal_rlib::os::time::v2::SimSpeed::Paused;
    if running && current != wiring {
      return AppearanceStatus::LockedWhileRunning as i32;
    }
  }
  if wiring.mode == CometDisplayMode::Custom {
    let Some(library) = library.try_read() else {
      return AppearanceStatus::Busy as i32;
    };
    let mesh_ok = wiring.mesh.is_none_or(|id| library.kind_of(id) == Some(AssetKind::Mesh));
    let textures_ok = wiring
      .textures
      .iter()
      .flatten()
      .all(|id| library.kind_of(*id) == Some(AssetKind::Texture));
    if !mesh_ok || !textures_ok {
      return AppearanceStatus::UnknownAsset as i32;
    }
  }

  match ctx_ref
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::SetCometAppearance {
      scene_id,
      wiring,
      offset,
    }) {
    Ok(()) => AppearanceStatus::Queued as i32,
    Err(_) => AppearanceStatus::QueueFull as i32,
  }
}

/// Reads the current comet appearance (as last applied by the logic thread).
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getCometAppearance(
  ctx: *mut SimulationContext,
  scene_id: u64,
  out: *mut CCometAppearanceDTO,
) -> bool {
  if ctx.is_null() || out.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let Some(scene_arc) = ctx_ref.scenes.read().get_scene(scene_id) else {
    return false;
  };
  let scene_guard = scene_arc.read();
  let Some(visual) = scene_guard.comet.and_then(|c| c.visual) else {
    return false;
  };
  let Some(a) = scene_guard.scene.with_component(visual, |a: &CometAppearanceComponent| *a) else {
    return false;
  };
  unsafe { *out = appearance_to_dto(&a) };
  true
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn dto_layouts_are_stable() {
    // mirrored by `[StructLayout(Sequential)]` structs in INativeRuntimeService.cs
    assert_eq!(core::mem::size_of::<CAssetInfoDTO>(), 72);
    assert_eq!(
      core::mem::size_of::<aethervk_core_rlib::simulation_api::external_state::CAssetRemoved>(),
      24
    );
    assert_eq!(core::mem::size_of::<CCometAppearanceDTO>(), 72);
    assert_eq!(core::mem::size_of::<CAssetLibraryStatsDTO>(), 24);
    assert_eq!(
      core::mem::size_of::<aethervk_core_rlib::simulation_api::external_state::CAssetImported>(),
      24
    );
  }

  #[test]
  fn appearance_dto_roundtrip_and_validation() {
    let dto = CCometAppearanceDTO {
      mesh: 7,
      textures: [3, 0, 0, 9],
      yaw_pitch_roll_deg: [10.0, 20.0, 30.0],
      translation_radii: [0.1, 0.0, -0.2],
      mode: 1,
      _pad: 0,
    };
    let (wiring, offset) = dto_to_appearance(&dto).unwrap();
    assert_eq!(wiring.mode, CometDisplayMode::Custom);
    assert_eq!(wiring.mesh, Some(7));
    assert_eq!(wiring.textures, [Some(3), None, None, Some(9)]);
    let back = appearance_to_dto(&CometAppearanceComponent {
      wiring,
      offset,
      ..Default::default()
    });
    assert_eq!(bytemuck::bytes_of(&back), bytemuck::bytes_of(&dto));

    assert!(dto_to_appearance(&CCometAppearanceDTO { mode: 7, ..dto }).is_none());
    let mut nan = dto;
    nan.yaw_pitch_roll_deg[1] = f32::NAN;
    assert!(dto_to_appearance(&nan).is_none());
  }

  #[test]
  fn c_string_truncation_respects_char_boundaries() {
    let mut buf = [0 as c_char; 5];
    let needed = unsafe { write_c_string("aé漢", buf.as_mut_ptr(), buf.len() as u32) };
    assert_eq!(needed, "aé漢".len() as u32);
    // 4 usable bytes: "a" (1) + "é" (2) fits, "漢" (3) does not
    let got: alloc::vec::Vec<u8> = buf.iter().map(|c| *c as u8).collect();
    assert_eq!(&got[..4], &[b'a', 0xC3, 0xA9, 0]);
  }
}
