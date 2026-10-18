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
  /// Bit 0 ([`ASSET_FLAG_DISPLAYABLE`]): texture with uncompressed texels
  /// ([`avkSimulationContext_copyAssetTexels`] works).
  pub flags: u32,
}

/// [`CAssetInfoDTO::flags`]: the texture's texels can be copied as RGBA8.
pub const ASSET_FLAG_DISPLAYABLE: u32 = 1 << 0;

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
    flags: if info.displayable { ASSET_FLAG_DISPLAYABLE } else { 0 },
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

/// Result of [`avkSimulationContext_copyAssetTexels`].
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexelsStatus {
  /// Copied (or, with a null buffer, sized).
  Ok = 0,
  /// Null context or size pointers.
  InvalidArgument = 1,
  /// No texture with this id.
  UnknownAsset = 2,
  /// Block-compressed texture: no CPU decoder.
  Compressed = 3,
  /// `out_len` < `width · height · 4`; the sizes are written.
  BufferTooSmall = 4,
  /// An import or unload holds the library; retry.
  Busy = 5,
}

/// Copies texture `asset_id` as tightly packed RGBA8 (desktop billboards draw it): grey → grey
/// opaque, grey + alpha, RGB → opaque, RGBA as is. With `out_rgba = null` only the size is written
/// (`out_width · out_height · 4` bytes). Returns a [`TexelsStatus`].
///
/// # Safety
/// FFI Contract: `out_rgba` is valid for `out_len` bytes when non-null.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_copyAssetTexels(
  ctx: *mut SimulationContext,
  asset_id: u64,
  out_rgba: *mut u8,
  out_len: u64,
  out_width: *mut u32,
  out_height: *mut u32,
) -> i32 {
  if ctx.is_null() || out_width.is_null() || out_height.is_null() {
    return TexelsStatus::InvalidArgument as i32;
  }
  let ctx_ref = unsafe { &*ctx };
  let library = alloc::sync::Arc::clone(&ctx_ref.scenes.read().asset_library);
  let Some(library) = library.try_read() else {
    return TexelsStatus::Busy as i32;
  };
  let sized = match library.copy_texture_rgba8(asset_id, None) {
    Ok(wh) => wh,
    Err(e) => return texels_status(e) as i32,
  };
  unsafe {
    *out_width = sized.0;
    *out_height = sized.1;
  }
  if out_rgba.is_null() {
    return TexelsStatus::Ok as i32;
  }
  let out = unsafe { core::slice::from_raw_parts_mut(out_rgba, out_len as usize) };
  match library.copy_texture_rgba8(asset_id, Some(out)) {
    Ok(_) => TexelsStatus::Ok as i32,
    Err(e) => texels_status(e) as i32,
  }
}

fn texels_status(e: aethervk_core_rlib::simulation::asset_library::TexelsError) -> TexelsStatus {
  use aethervk_core_rlib::simulation::asset_library::TexelsError;
  match e {
    TexelsError::UnknownAsset => TexelsStatus::UnknownAsset,
    TexelsError::Compressed => TexelsStatus::Compressed,
    TexelsError::BufferTooSmall => TexelsStatus::BufferTooSmall,
  }
}

/// Writes the source of asset `asset_id` into `dest_dir` (UTF-8, NUL-terminated) as
/// `<content_hash>_<label>.<ext>`: the original file copied verbatim, or the encoded bytes of an
/// image embedded in a glTF/GLB (savefile `assets/` folders). `out_name` receives the file name
/// (NUL-terminated, truncated to `out_name_len`; `out_name_needed` = its full byte length) and
/// `out_hash` the content hash. Returns `0` written, `1` reused (already there), negative on
/// failure: `-1` bad arguments, `-2` unknown asset, `-3` source unreadable, `-4` write failed,
/// `-5` library busy.
///
/// # Safety
/// FFI Contract: `dest_dir` is a valid C string; `out_name` is valid for `out_name_len` bytes
/// when non-null; `out_name_needed` / `out_hash` are null or writable.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_exportAsset(
  ctx: *mut SimulationContext,
  asset_id: u64,
  dest_dir: *const c_char,
  out_name: *mut c_char,
  out_name_len: u32,
  out_name_needed: *mut u32,
  out_hash: *mut u64,
) -> i32 {
  use aethervk_core_rlib::simulation::asset_library::AssetError;
  if ctx.is_null() || dest_dir.is_null() {
    return -1;
  }
  let Ok(dir) = (unsafe { CStr::from_ptr(dest_dir) }).to_str() else {
    return -1;
  };
  let ctx_ref = unsafe { &*ctx };
  let library = alloc::sync::Arc::clone(&ctx_ref.scenes.read().asset_library);
  let Some(library) = library.try_read() else {
    return -5;
  };
  match library.export_asset(asset_id, dir) {
    Ok(e) => {
      unsafe {
        let n = write_c_string(&e.file_name, out_name, out_name_len);
        if !out_name_needed.is_null() {
          *out_name_needed = n;
        }
        if !out_hash.is_null() {
          *out_hash = e.content_hash;
        }
      }
      if e.reused { 1 } else { 0 }
    }
    Err(AssetError::UnknownAsset) => -2,
    Err(AssetError::Load(_)) => -3,
    Err(_) => -4,
  }
}

/// The content hash of asset `asset_id` (the library's dedup hash of the decoded data), written to
/// `out_hash`. `false` for an unknown asset or while an import holds the library. A savefile maps
/// its saved assets to the ids they get when re-imported by this hash.
///
/// # Safety
/// FFI Contract: `out_hash` is writable.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getAssetContentHash(
  ctx: *mut SimulationContext,
  asset_id: u64,
  out_hash: *mut u64,
) -> bool {
  if ctx.is_null() || out_hash.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let library = alloc::sync::Arc::clone(&ctx_ref.scenes.read().asset_library);
  let Some(library) = library.try_read() else {
    return false;
  };
  match library.content_hash_of(asset_id) {
    Some(h) => {
      unsafe { *out_hash = h };
      true
    }
    None => false,
  }
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

// ── Sky image overlay (HUD, Earth-observer panel) ────────────────────────────

/// Facts of a loaded sky image (`avkSimulationContext_loadSkyImage`). `flags` holds a bit per
/// header fact (`aethervk_core_rlib::scene::sky_image::SKY_FLAG_*`): a field whose bit is clear is
/// zero and must not be offered (DATE-OBS bit 0, EXPTIME 1, SITELAT 2, SITELONG 3, SITEELEV 4,
/// SIP ignored 5, PNG / JPEG source 6, WCS from a sidecar 7, frame assumed ICRS 8, TIMESYS assumed
/// UTC 9).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct CSkyImageInfo {
  /// `1` = a TAN WCS was read: the image is registered and drawn
  pub has_wcs: u32,
  pub nx: u32,
  pub ny: u32,
  pub _pad: u32,
  /// start of the exposure: TDB microseconds since J2000 TDB (the scale `seekEpochSync` takes)
  pub date_obs_us: i64,
  pub exptime_s: f64,
  pub site_lat_deg: f64,
  /// east positive
  pub site_lon_deg: f64,
  pub site_elev_m: f64,
  /// `CRVAL1/2` (ICRS / FK5 J2000), the "lock to image" aim
  pub crval_ra_deg: f64,
  pub crval_dec_deg: f64,
  pub flags: u32,
  pub _pad2: u32,
}
const _: () = assert!(core::mem::size_of::<CSkyImageInfo>() == 80);

fn sky_image_info(img: &aethervk_core_rlib::scene::sky_image::SkyImage) -> CSkyImageInfo {
  let m = &img.meta;
  CSkyImageInfo {
    has_wcs: img.wcs.is_some() as u32,
    nx: img.nx,
    ny: img.ny,
    _pad: 0,
    date_obs_us: m.date_obs_tdb_us.unwrap_or(0),
    exptime_s: m.exptime_s.unwrap_or(0.0),
    site_lat_deg: m.site_lat_deg.unwrap_or(0.0),
    site_lon_deg: m.site_lon_deg.unwrap_or(0.0),
    site_elev_m: m.site_elev_m.unwrap_or(0.0),
    crval_ra_deg: img.wcs.map(|w| w.crval_deg[0]).unwrap_or(0.0),
    crval_dec_deg: img.wcs.map(|w| w.crval_deg[1]).unwrap_or(0.0),
    flags: img.flags,
    _pad2: 0,
  }
}

/// Store key of scene `scene_id` (`None`: no such scene).
fn sky_image_key(ctx: *mut SimulationContext, scene_id: u64) -> Option<usize> {
  if ctx.is_null() {
    return None;
  }
  let scene = unsafe { &*ctx }.get_scene(scene_id)?;
  Some(aethervk_core_rlib::scene::sky_image::scene_key(&scene))
}

/// Loads a telescope image (FITS, or PNG / JPEG with a WCS sidecar; `sidecar` may be null: a
/// PNG / JPEG then looks for `<stem>.wcs`) as scene `scene_id`'s sky overlay, replacing any.
/// Synchronous (decode + percentiles: call it off the UI thread). Returns `false` when the file
/// cannot be read or decoded (the reason is logged). An image without a usable TAN WCS returns
/// `true` with `has_wcs = 0` and removes the previous overlay (nothing to register).
///
/// # Safety
/// FFI Contract: `path` (and `sidecar` when not null) are NUL-terminated UTF-8; `out` is valid.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_loadSkyImage(
  ctx: *mut SimulationContext,
  scene_id: u64,
  path: *const c_char,
  sidecar: *const c_char,
  out: *mut CSkyImageInfo,
) -> bool {
  use aethervk_core_rlib::scene::sky_image;
  if path.is_null() || out.is_null() {
    return false;
  }
  let Some(key) = sky_image_key(ctx, scene_id) else {
    return false;
  };
  let Ok(path) = unsafe { CStr::from_ptr(path) }.to_str() else {
    return false;
  };
  let sidecar = if sidecar.is_null() {
    None
  } else {
    match unsafe { CStr::from_ptr(sidecar) }.to_str() {
      Ok(s) => Some(s),
      Err(_) => return false,
    }
  };
  let img = match sky_image::SkyImage::load(path, sidecar) {
    Ok(img) => img,
    Err(e) => {
      aethervk_oshal_rlib::log!("[SkyImage] {path}: {e}");
      return false;
    }
  };
  if let Some(e) = &img.wcs_error {
    aethervk_oshal_rlib::log!("[SkyImage] {path}: WCS refused: {e}");
  }
  if !sky_image::store_insert(key, &img) {
    sky_image::store_remove(key);
  }
  unsafe { out.write(sky_image_info(&img)) };
  true
}

/// Display parameters of scene `scene_id`'s sky overlay: `opacity` in `[0, 1]`, `stretch` the
/// asinh softening (`0` = linear, larger = more of the faint end), `visible` `0` / `1`. `false`
/// without an overlay.
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setSkyImageParams(
  ctx: *mut SimulationContext,
  scene_id: u64,
  opacity: f32,
  stretch: f32,
  visible: u32,
) -> bool {
  use aethervk_core_rlib::scene::sky_image;
  let Some(key) = sky_image_key(ctx, scene_id) else {
    return false;
  };
  sky_image::store_set_params(
    key,
    sky_image::SkyImageParams {
      opacity,
      stretch,
      visible: visible != 0,
    },
  )
}

/// Removes scene `scene_id`'s sky overlay (its GPU resources are retired by the next frame).
/// `false` when there was none.
///
/// # Safety
/// FFI Contract
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_removeSkyImage(
  ctx: *mut SimulationContext,
  scene_id: u64,
) -> bool {
  let Some(key) = sky_image_key(ctx, scene_id) else {
    return false;
  };
  aethervk_core_rlib::scene::sky_image::store_remove(key)
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
    assert_eq!(core::mem::size_of::<CSkyImageInfo>(), 80);
    assert_eq!(core::mem::offset_of!(CSkyImageInfo, date_obs_us), 16);
    assert_eq!(core::mem::offset_of!(CSkyImageInfo, flags), 72);
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
  fn displayable_flag_rides_in_the_former_padding_word() {
    // C# reads it at the same offset the padding had: the DTO size is unchanged (72)
    assert_eq!(core::mem::offset_of!(CAssetInfoDTO, flags), 68);
    let mut info = AssetInfo {
      id: 3,
      kind: AssetKind::Texture,
      key: "k".into(),
      label: "l".into(),
      source_path: alloc::string::String::new(),
      a: 4,
      b: 2,
      channel_hint: None,
      bundled_textures: [None; 4],
      orientation_fix: Default::default(),
      displayable: true,
    };
    assert_eq!(info_to_dto(&info).flags, ASSET_FLAG_DISPLAYABLE);
    info.displayable = false;
    assert_eq!(info_to_dto(&info).flags, 0);
  }

  #[test]
  fn texels_status_codes() {
    use aethervk_core_rlib::simulation::asset_library::TexelsError;
    assert_eq!(texels_status(TexelsError::UnknownAsset) as i32, 2);
    assert_eq!(texels_status(TexelsError::Compressed) as i32, 3);
    assert_eq!(texels_status(TexelsError::BufferTooSmall) as i32, 4);
    let (mut w, mut h) = (0u32, 0u32);
    let st = unsafe {
      avkSimulationContext_copyAssetTexels(core::ptr::null_mut(), 1, core::ptr::null_mut(), 0, &mut w, &mut h)
    };
    assert_eq!(st, TexelsStatus::InvalidArgument as i32);
  }

  /// End to end through a real context (skipped without a Vulkan device): an imported 2×1 RGB PNG
  /// comes back as RGBA8 through the FFI; the size query, a short buffer and an unknown id.
  #[test]
  fn copy_asset_texels_through_a_context() {
    extern crate std;
    const PNG_2X1_RGB: &[u8] = &[
      137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 1, 8, 2, 0,
      0, 0, 123, 64, 232, 221, 0, 0, 0, 15, 73, 68, 65, 84, 120, 156, 99, 224, 18, 145, 59, 49, 45,
      5, 0, 5, 7, 1, 255, 6, 124, 182, 253, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];
    let asset_dir = alloc::format!("{}/../../assets", env!("CARGO_MANIFEST_DIR"));
    SimulationContext::set_asset_path(&asset_dir);
    let Ok(ctx) = SimulationContext::startup(None) else {
      std::println!("Skipping: no Vulkan device");
      return;
    };
    let dir = std::env::temp_dir().join(alloc::format!("avk_ffi_texels_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let png = dir.join("two.png");
    std::fs::write(&png, PNG_2X1_RGB).unwrap();
    let id = {
      let lib = alloc::sync::Arc::clone(&ctx.scenes.read().asset_library);
      let mut lib = lib.write();
      lib.set_cache_dir(dir.join("cache").to_str().unwrap()).unwrap();
      lib.import_file(png.to_str().unwrap()).unwrap().textures[0]
    };
    let ctx_ptr = alloc::boxed::Box::into_raw(ctx);
    let (mut w, mut h) = (0u32, 0u32);
    unsafe {
      let st = avkSimulationContext_copyAssetTexels(ctx_ptr, id, core::ptr::null_mut(), 0, &mut w, &mut h);
      assert_eq!((st, w, h), (TexelsStatus::Ok as i32, 2, 1));
      let mut out = [0u8; 8];
      let st = avkSimulationContext_copyAssetTexels(ctx_ptr, id, out.as_mut_ptr(), 8, &mut w, &mut h);
      assert_eq!(st, TexelsStatus::Ok as i32);
      assert_eq!(out, [10, 20, 30, 255, 200, 150, 100, 255]);
      let mut short = [0u8; 7];
      let st = avkSimulationContext_copyAssetTexels(ctx_ptr, id, short.as_mut_ptr(), 7, &mut w, &mut h);
      assert_eq!(st, TexelsStatus::BufferTooSmall as i32);
      let st = avkSimulationContext_copyAssetTexels(ctx_ptr, id + 1000, core::ptr::null_mut(), 0, &mut w, &mut h);
      assert_eq!(st, TexelsStatus::UnknownAsset as i32);
      // the listing flags it displayable
      let mut dto = CAssetInfoDTO::default();
      let ok = avkSimulationContext_getAssetInfo(
        ctx_ptr, 0, &mut dto, core::ptr::null_mut(), 0, core::ptr::null_mut(), core::ptr::null_mut(), 0, core::ptr::null_mut(),
      );
      assert!(ok && dto.id == id && dto.flags & ASSET_FLAG_DISPLAYABLE != 0);
      // the content hash a savefile maps re-imported assets by: the library's, unknown ids refused
      let mut hash = 0u64;
      assert!(avkSimulationContext_getAssetContentHash(ctx_ptr, id, &mut hash));
      let want = alloc::sync::Arc::clone(&(*ctx_ptr).scenes.read().asset_library).read().content_hash_of(id);
      assert_eq!(Some(hash), want);
      assert_ne!(hash, 0);
      assert!(!avkSimulationContext_getAssetContentHash(ctx_ptr, id + 1000, &mut hash));
      assert!(!avkSimulationContext_getAssetContentHash(ctx_ptr, id, core::ptr::null_mut()));
      {
        let lib = alloc::sync::Arc::clone(&(*ctx_ptr).scenes.read().asset_library);
        lib.write().clear();
      }
      drop(alloc::boxed::Box::from_raw(ctx_ptr));
    }
    let _ = std::fs::remove_dir_all(&dir);
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
