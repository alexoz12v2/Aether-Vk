//! FFI for FITS texture assets (asset library): header metadata as JSON and the display stretch.
//! Contract: `docs/fits_ffi_contract.md`.

use aethervk_core_rlib::{
  scene::sky_image::fits_metadata::{json_f64, json_str},
  simulation::asset_library::AssetLibrary,
  simulation_api::SimulationContext,
};
use alloc::{format, string::String};
use core::ffi::c_char;

/// The JSON document of a FITS asset: header cards, facts and the current stretch.
pub fn fits_metadata_json(library: &AssetLibrary, asset_id: u64) -> Option<String> {
  let f = library.fits(asset_id)?;
  let (w, h) = library
    .texture(asset_id)
    .map(|t| (t.width(), t.height()))
    .unwrap_or((f.width(), f.height()));
  let (mode, transfer) = f.stretch.codes();
  let meta = f.metadata.to_json();
  // splice the asset / texture / stretch fields in front of the metadata object's fields
  let mut j = String::with_capacity(meta.len() + 256);
  j.push_str(&format!(
    "{{\"assetId\":{asset_id},\"textureWidth\":{w},\"textureHeight\":{h},\"rowOrder\":"
  ));
  json_str(&mut j, "topDown");
  j.push_str(if f.flipped { ",\"flipped\":true" } else { ",\"flipped\":false" });
  j.push_str(&format!(
    ",\"stretch\":{{\"mode\":{mode},\"transfer\":{transfer},\"black\":{},\"white\":{}}},",
    json_f64(Some(f.black)),
    json_f64(Some(f.white))
  ));
  j.push_str(&meta[1..]);
  Some(j)
}

/// Header metadata of FITS texture `asset_id` as UTF-8 JSON (schema in
/// `docs/fits_ffi_contract.md`), written NUL-terminated into `out` (truncated to `out_len`;
/// `out = null` only sizes); `out_needed` receives the full byte length (excluding the NUL).
/// `false` when the asset is unknown / not a FITS texture, or while an import holds the library.
///
/// # Safety
/// FFI contract: `ctx` valid; `out` valid for `out_len` bytes when non-null; `out_needed` null or
/// writable.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getFitsMetadataJson(
  ctx: *mut SimulationContext,
  asset_id: u64,
  out: *mut c_char,
  out_len: u32,
  out_needed: *mut u32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let library = alloc::sync::Arc::clone(&ctx_ref.scenes.read().asset_library);
  let Some(library) = library.try_read() else {
    return false;
  };
  let Some(json) = fits_metadata_json(&library, asset_id) else {
    return false;
  };
  let bytes = json.as_bytes();
  if !out.is_null() && out_len > 0 {
    // JSON is ASCII-escaped except for string contents; cut on a char boundary
    let mut n = bytes.len().min(out_len as usize - 1);
    while !json.is_char_boundary(n) {
      n -= 1;
    }
    unsafe {
      core::ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast::<u8>(), n);
      *out.add(n) = 0;
    }
  }
  if !out_needed.is_null() {
    unsafe { *out_needed = bytes.len() as u32 };
  }
  true
}

/// Re-renders the RGBA8 texels of FITS texture `asset_id` from its cached raster (no re-read):
/// `mode` 0 zscale, 1 percentile (`p0` low %, `p1` high %, `0..=100`), 2 manual (`p0` black,
/// `p1` white, data units); `transfer` 0 linear, 1 log, 2 asinh. A later
/// `avkSimulationContext_copyAssetTexels` / `copyAssetThumbnail` returns the new image.
/// `out_black` / `out_white` (nullable) receive the points used. `false` for an unknown or
/// non-FITS asset, invalid codes / parameters or a cache write failure. Waits for the library's
/// write lock (an import, a texel copy): call it off the UI thread.
///
/// # Safety
/// FFI contract: `ctx` valid; `out_black` / `out_white` null or writable.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setFitsStretch(
  ctx: *mut SimulationContext,
  asset_id: u64,
  mode: u32,
  transfer: u32,
  p0: f64,
  p1: f64,
  out_black: *mut f64,
  out_white: *mut f64,
) -> bool {
  use aethervk_core_rlib::scene::sky_image::fits_stretch::FitsStretch;
  if ctx.is_null() {
    return false;
  }
  let Some(stretch) = FitsStretch::from_codes(mode, transfer, p0, p1) else {
    return false;
  };
  let ctx_ref = unsafe { &*ctx };
  let library = alloc::sync::Arc::clone(&ctx_ref.scenes.read().asset_library);
  let mut library = library.write();
  match library.set_fits_stretch(asset_id, stretch) {
    Ok((b, w)) => {
      unsafe {
        if !out_black.is_null() {
          *out_black = b;
        }
        if !out_white.is_null() {
          *out_white = w;
        }
      }
      true
    }
    Err(_) => false,
  }
}

/// The physical f32 raster of FITS texture `asset_id` (`BZERO + BSCALE · raw`, blank = NaN), in
/// texture row order (row 0 = the top row of the displayed image, as
/// `avkSimulationContext_copyAssetTexels` returns it), for measurements on the data rather than on
/// the stretched texels (nucleus detection). With `out = null` only the size is written
/// (`out_width · out_height` floats); `out_len` counts floats. Returns a
/// [`crate::ffi_assets::TexelsStatus`]: `UnknownAsset` also for a texture that is not FITS.
///
/// # Safety
/// FFI contract: `ctx` valid; `out` valid for `out_len` floats when non-null; `out_width` /
/// `out_height` writable.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_copyFitsRaster(
  ctx: *mut SimulationContext,
  asset_id: u64,
  out: *mut f32,
  out_len: u64,
  out_width: *mut u32,
  out_height: *mut u32,
) -> i32 {
  use crate::ffi_assets::TexelsStatus;
  if ctx.is_null() || out_width.is_null() || out_height.is_null() {
    return TexelsStatus::InvalidArgument as i32;
  }
  let ctx_ref = unsafe { &*ctx };
  let library = alloc::sync::Arc::clone(&ctx_ref.scenes.read().asset_library);
  let Some(library) = library.try_read() else {
    return TexelsStatus::Busy as i32;
  };
  copy_fits_raster(&library, asset_id, out, out_len, out_width, out_height) as i32
}

/// [`avkSimulationContext_copyFitsRaster`] on a library (testable without a context).
fn copy_fits_raster(
  library: &AssetLibrary,
  asset_id: u64,
  out: *mut f32,
  out_len: u64,
  out_width: *mut u32,
  out_height: *mut u32,
) -> crate::ffi_assets::TexelsStatus {
  use crate::ffi_assets::TexelsStatus;
  let Some(f) = library.fits(asset_id) else {
    return TexelsStatus::UnknownAsset;
  };
  let (w, h) = (f.width(), f.height());
  unsafe {
    *out_width = w;
    *out_height = h;
  }
  if out.is_null() {
    return TexelsStatus::Ok;
  }
  let raster = f.raster();
  let n = w as usize * h as usize;
  if (out_len as usize) < n || raster.len() < n {
    return TexelsStatus::BufferTooSmall;
  }
  unsafe { core::ptr::copy_nonoverlapping(raster.as_ptr(), out, n) };
  TexelsStatus::Ok
}

#[cfg(test)]
mod tests {
  use super::*;

  /// 4 × 3 BITPIX 16 FITS (ramp) with an OBJECT card.
  fn tiny_fits() -> alloc::vec::Vec<u8> {
    let mut h = alloc::vec::Vec::new();
    for c in [
      "SIMPLE  =                    T",
      "BITPIX  =                   16",
      "NAXIS   =                    2",
      "NAXIS1  =                    4",
      "NAXIS2  =                    3",
      "OBJECT  = 'Test \"q\"'          / a comment",
      "END",
    ] {
      let mut card = [b' '; 80];
      card[..c.len()].copy_from_slice(c.as_bytes());
      h.extend_from_slice(&card);
    }
    h.resize(2880, b' ');
    let mut d = alloc::vec::Vec::new();
    for i in 0..12i16 {
      d.extend_from_slice(&(i * 100).to_be_bytes());
    }
    d.resize(2880, 0);
    h.extend_from_slice(&d);
    h
  }

  #[test]
  fn fits_ffi_json_and_stretch_without_device() {
    extern crate std;
    let dir = std::env::temp_dir().join(format!("avk_ffi_fits_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("t.fits");
    std::fs::write(&p, tiny_fits()).unwrap();
    let mut lib = AssetLibrary::new();
    lib.set_cache_dir(dir.join("cache").to_str().unwrap()).unwrap();
    let id = lib.import_file(p.to_str().unwrap()).unwrap().textures[0];
    let j = fits_metadata_json(&lib, id).unwrap();
    assert!(j.starts_with(&format!("{{\"assetId\":{id},\"textureWidth\":4,\"textureHeight\":3")));
    assert!(j.contains("\"naxis1\":4,\"naxis2\":3,\"bitpix\":16"));
    assert!(j.contains("{\"key\":\"OBJECT\",\"value\":\"Test \\\"q\\\"\",\"comment\":\"a comment\"}"));
    assert!(j.contains("\"object\":\"Test \\\"q\\\"\""));
    assert!(j.contains("\"stretch\":{\"mode\":0,\"transfer\":0"));
    assert!(j.ends_with("}}"));
    assert!(fits_metadata_json(&lib, id + 99).is_none());

    // the physical raster, texture row order: FITS row 0 (0, 100, 200, 300) is the bottom row
    use crate::ffi_assets::TexelsStatus;
    let (mut w, mut h) = (0u32, 0u32);
    assert_eq!(copy_fits_raster(&lib, id, core::ptr::null_mut(), 0, &mut w, &mut h), TexelsStatus::Ok);
    assert_eq!((w, h), (4, 3));
    let mut short = [0f32; 11];
    assert_eq!(
      copy_fits_raster(&lib, id, short.as_mut_ptr(), 11, &mut w, &mut h),
      TexelsStatus::BufferTooSmall
    );
    let mut px = [0f32; 12];
    assert_eq!(copy_fits_raster(&lib, id, px.as_mut_ptr(), 12, &mut w, &mut h), TexelsStatus::Ok);
    assert_eq!(&px[..4], &[800.0, 900.0, 1000.0, 1100.0]);
    assert_eq!(&px[8..], &[0.0, 100.0, 200.0, 300.0]);
    assert_eq!(
      copy_fits_raster(&lib, id + 99, core::ptr::null_mut(), 0, &mut w, &mut h),
      TexelsStatus::UnknownAsset
    );
    lib.clear();
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[test]
  fn fits_ffi_null_context() {
    let (mut b, mut w) = (0.0, 0.0);
    unsafe {
      assert!(!avkSimulationContext_getFitsMetadataJson(
        core::ptr::null_mut(),
        1,
        core::ptr::null_mut(),
        0,
        core::ptr::null_mut()
      ));
      assert!(!avkSimulationContext_setFitsStretch(core::ptr::null_mut(), 1, 0, 0, 0.0, 0.0, &mut b, &mut w));
      let (mut rw, mut rh) = (0u32, 0u32);
      assert_eq!(
        avkSimulationContext_copyFitsRaster(core::ptr::null_mut(), 1, core::ptr::null_mut(), 0, &mut rw, &mut rh),
        crate::ffi_assets::TexelsStatus::InvalidArgument as i32
      );
    }
  }
}
