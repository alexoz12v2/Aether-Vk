//! FITS files imported as texture assets.

use super::*;
use crate::scene::sky_image::{
  fits_stretch::{StretchMode, Transfer},
  fits_tests::fits_bytes,
};
use std::path::PathBuf;

struct Scratch(PathBuf);

impl Scratch {
  fn new(tag: &str) -> Self {
    static N: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("avk_fits_lib_{tag}_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    Self(dir)
  }

  fn write(&self, name: &str, bytes: &[u8]) -> String {
    let p = self.0.join(name);
    std::fs::write(&p, bytes).unwrap();
    p.to_str().unwrap().to_string()
  }

  fn library(&self) -> AssetLibrary {
    let mut lib = AssetLibrary::new();
    lib.set_cache_dir(self.0.join("asset_cache").to_str().unwrap()).unwrap();
    lib
  }
}

impl Drop for Scratch {
  fn drop(&mut self) {
    let _ = std::fs::remove_dir_all(&self.0);
  }
}

/// 4 × 3 BITPIX 16 ramp: FITS row `j` (bottom first) holds `j·4 + i`, `*100`; one BLANK pixel.
fn ramp_fits() -> Vec<u8> {
  let mut v: Vec<f64> = (0..12).map(|i| (i * 100) as f64).collect();
  v[1] = -1.0; // BLANK
  fits_bytes(16, 4, 3, &["BLANK   =                   -1", "OBJECT  = 'ramp    '"], &v)
}

fn texels(lib: &AssetLibrary, id: AssetId) -> Vec<u8> {
  let (w, h) = lib.copy_texture_rgba8(id, None).unwrap();
  let mut out = alloc::vec![0u8; (w * h * 4) as usize];
  lib.copy_texture_rgba8(id, Some(&mut out)).unwrap();
  out
}

#[test]
fn fits_import_yields_a_flipped_rgba8_texture() {
  let s = Scratch::new("import");
  let path = s.write("ramp.FITS", &ramp_fits()); // extension case-insensitive
  let mut lib = s.library();
  let out = lib.import_file(&path).unwrap();
  assert_eq!(out.textures.len(), 1);
  assert_eq!(out.added, out.textures);
  let id = out.textures[0];
  let t = lib.texture(id).unwrap();
  assert_eq!((t.width(), t.height(), t.format()), (4, 3, TexelFormat::R8G8B8A8_UNORM));
  assert_eq!(lib.copy_texture_rgba8(id, None), Ok((4, 3)));
  let px = texels(&lib, id);
  assert_eq!(px.len(), 4 * 3 * 4);
  // grey, opaque
  for p in px.chunks_exact(4) {
    assert!(p[0] == p[1] && p[1] == p[2] && p[3] == 255);
  }
  // texture row 0 = last FITS row (the brightest: 800..1100), last texture row = FITS row 0
  let row = |r: usize| -> Vec<u8> { (0..4).map(|i| px[(r * 4 + i) * 4]).collect() };
  assert_eq!(row(0)[3], 255, "FITS max (1100) at texture (3, 0)");
  assert!(row(0)[0] > row(1)[3] && row(1)[0] > row(2)[3]);
  // the BLANK pixel (FITS (1, 0)) is black, the minimum (FITS (0, 0), 0) too
  assert_eq!(row(2)[1], 0);
  assert_eq!(row(2)[0], 0);
  // FITS side data
  let f = lib.fits(id).unwrap();
  assert_eq!((f.width(), f.height(), f.metadata.bitpix), (4, 3, 16));
  assert_eq!(f.metadata.facts.object.as_deref(), Some("ramp"));
  assert_eq!(f.stretch, FitsStretch::default());
  assert_eq!((f.black, f.white), (0.0, 1100.0));
  assert_eq!(f.raster()[0], 800.0);
  assert!(f.raster()[9].is_nan());
  // listed as a displayable texture; a PNG has no FITS data
  assert!(lib.assets().iter().any(|a| a.id == id && a.displayable));
  // .fit / .fts too, the same raster under another path is the same asset
  let p2 = s.write("copy.fts", &ramp_fits());
  let out2 = lib.import_file(&p2).unwrap();
  assert_eq!(out2.textures, [id]);
  assert!(out2.added.is_empty());
}

#[test]
fn fits_restretch_changes_texels_without_reimport() {
  let s = Scratch::new("restretch");
  let path = s.write("ramp.fit", &ramp_fits());
  let mut lib = s.library();
  let id = lib.import_file(&path).unwrap().textures[0];
  let before = texels(&lib, id);
  let hash = lib.content_hash_of(id);
  let thumb_before = lib.thumbnail(id).unwrap().rgba.clone();
  // the source is gone: the cached raster is used
  std::fs::remove_file(&path).unwrap();
  let (b, w) = lib
    .set_fits_stretch(id, FitsStretch { mode: StretchMode::Manual { black: 0.0, white: 2000.0 }, transfer: Transfer::Linear })
    .unwrap();
  assert_eq!((b, w), (0.0, 2000.0));
  let linear = texels(&lib, id);
  assert_ne!(before, linear);
  assert_eq!(linear[(3) * 4], 140); // FITS (3, 2) = 1100 → 1100 / 2000 · 255
  let (_, _) = lib
    .set_fits_stretch(id, FitsStretch { mode: StretchMode::Manual { black: 0.0, white: 2000.0 }, transfer: Transfer::Asinh })
    .unwrap();
  let asinh = texels(&lib, id);
  assert!(asinh[3 * 4] > linear[3 * 4]);
  assert_ne!(lib.thumbnail(id).unwrap().rgba, thumb_before);
  // identity unchanged
  assert_eq!(lib.content_hash_of(id), hash);
  let f = lib.fits(id).unwrap();
  assert_eq!(f.stretch.transfer, Transfer::Asinh);
  // percentile, back to zscale = the import texels
  lib.set_fits_stretch(id, FitsStretch { mode: StretchMode::Percentile { low: 0.0, high: 100.0 }, transfer: Transfer::Log }).unwrap();
  let (b, w) = lib.set_fits_stretch(id, FitsStretch::default()).unwrap();
  assert_eq!((b, w), (0.0, 1100.0));
  assert_eq!(texels(&lib, id), before);
  // one texel cache file per texture (old ones deleted), plus the raster
  let cache: Vec<_> = std::fs::read_dir(s.0.join("asset_cache")).unwrap().flatten().map(|e| e.file_name()).collect();
  assert_eq!(cache.iter().filter(|n| n.to_string_lossy().ends_with(".avkt")).count(), 1, "{cache:?}");
  assert_eq!(cache.iter().filter(|n| n.to_string_lossy().ends_with(".avkf")).count(), 1, "{cache:?}");
  // not a FITS / unknown
  assert!(matches!(lib.set_fits_stretch(id + 100, FitsStretch::default()), Err(AssetError::NotFits)));
  // remove drops the raster
  lib.remove(id).unwrap();
  assert!(lib.fits(id).is_none());
  let left = std::fs::read_dir(s.0.join("asset_cache")).unwrap().count();
  assert_eq!(left, 0);
}

#[test]
fn fits_bad_file_is_a_fits_error() {
  let s = Scratch::new("bad");
  let path = s.write("bad.fits", b"not a fits file at all");
  let mut lib = s.library();
  assert!(matches!(lib.import_file(&path), Err(AssetError::Fits(_))));
}

#[test]
fn fits_fixture_imports_with_its_header() {
  let src = aethervk_test_assets::test_asset!("fits_lt_67p_header_4x3.fits");
  let s = Scratch::new("fixture");
  let mut lib = s.library();
  let id = lib.import_file(src.to_str().unwrap()).unwrap().textures[0];
  assert_eq!(lib.copy_texture_rgba8(id, None), Ok((4, 3)));
  let f = lib.fits(id).unwrap();
  assert_eq!(f.metadata.facts.telescope.as_deref(), Some("Liverpool Telescope"));
  assert!(f.metadata.cards.len() > 250);
}

/// The user's real file end to end (not in the repo): `--run-ignored all`.
#[test]
#[ignore]
fn fits_real_file_imports() {
  let p = "Y:/Downloads/67p 20211225+26+27+28 Liverpool2m0 g+r+i+z 1x-02.fit";
  if !std::path::Path::new(p).exists() {
    std::println!("skipping: {p} not found");
    return;
  }
  let s = Scratch::new("real");
  let mut lib = s.library();
  let id = lib.import_file(p).unwrap().textures[0];
  let px = texels(&lib, id);
  assert_eq!(lib.copy_texture_rgba8(id, None), Ok((1200, 1200)));
  let f = lib.fits(id).unwrap();
  let mean = px.chunks_exact(4).map(|p| p[0] as f64).sum::<f64>() / (1200.0 * 1200.0);
  std::println!("zscale black {} white {} mean grey {mean:.1}", f.black, f.white);
  std::println!("{:#?}", f.metadata.facts);
  let t0 = std::time::Instant::now();
  let (b, w) = lib.set_fits_stretch(id, FitsStretch { mode: StretchMode::Percentile { low: 0.5, high: 99.5 }, transfer: Transfer::Asinh }).unwrap();
  std::println!("percentile/asinh {b} .. {w} in {:?}", t0.elapsed());
  assert!(w > b);
}

/// The committed image-extension and tile-compressed fixtures import like a primary-HDU file
/// (the ramp, flipped into texture row order); `.fz` is an accepted extension. Each in its own
/// library: the two hold the same raster, so one library would hand back the first asset.
#[cfg(feature = "cfitsio")]
#[test]
fn fits_extension_and_compressed_fixtures_import() {
  // the 4 x 3 ramp 100..1200 with FITS row 2 first
  let flipped: Vec<f32> = (0..3).rev().flat_map(|r| (0..4).map(move |c| ((r * 4 + c + 1) * 100) as f32)).collect();
  for (name, compressed, hdu) in [("fits_image_extension_4x3.fits", false, 2), ("fits_tile_compressed_4x3.fz", true, 2)] {
    let path = aethervk_test_assets::test_asset!(name);
    let s = Scratch::new("ext_fz");
    let mut lib = s.library();
    let out = lib.import_file(path.to_str().unwrap()).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    let id = out.textures[0];
    assert_eq!(out.added, [id], "{name}");
    let f = lib.fits(id).unwrap();
    assert_eq!((f.width(), f.height()), (4, 3), "{name}");
    assert_eq!(&*f.raster(), &flipped[..], "{name}: the ramp in texture row order");
    assert_eq!((f.metadata.hdu, f.metadata.compressed, f.metadata.bitpix), (hdu, compressed, 16), "{name}");
    assert!(f.metadata.to_json().contains(&format!("\"hdu\":{hdu},\"compressed\":{compressed},")), "{name}");
    let t = lib.texture(id).unwrap();
    assert_eq!((t.width(), t.height()), (4, 3), "{name}");
  }
}

/// Without CFITSIO the same two fixtures are refused at import (the pure-Rust parser reads the
/// primary HDU only), with the decode message.
#[cfg(not(feature = "cfitsio"))]
#[test]
fn fits_extension_and_compressed_fixtures_are_refused_without_cfitsio() {
  let s = Scratch::new("ext_fz_legacy");
  let mut lib = s.library();
  for name in ["fits_image_extension_4x3.fits", "fits_tile_compressed_4x3.fz"] {
    let path = aethervk_test_assets::test_asset!(name);
    match lib.import_file(path.to_str().unwrap()) {
      Err(AssetError::Fits(m)) => assert!(m.contains("extensions are not read"), "{name}: {m}"),
      other => panic!("{name}: {other:?}"),
    }
  }
}

/// The committed 4 × 3 × 3 cube imports as its first plane; `set_fits_plane` re-reads the source
/// for another plane (plane p = the ramp + 1000 p), replaces raster and texels, keeps the stretch
/// (its points recomputed on the plane) and reports the plane in the metadata; an out-of-range
/// plane or a non-FITS asset is an error that leaves the asset as it was.
#[test]
fn fits_cube_plane_selection() {
  let src = aethervk_test_assets::test_asset!("fits_cube_4x3x3.fits");
  let s = Scratch::new("cube");
  let mut lib = s.library();
  let id = lib.import_file(src.to_str().unwrap()).unwrap().textures[0];
  let plane = |p: f32| -> Vec<f32> {
    // the ramp 100..1200 + 1000 p, FITS row 2 first
    (0..3).rev().flat_map(|r| (0..4).map(move |c| ((r * 4 + c + 1) * 100) as f32 + 1000.0 * p)).collect()
  };
  let f = lib.fits(id).unwrap();
  assert_eq!((f.metadata.planes, f.metadata.plane), (3, 0));
  assert_eq!(&*f.raster(), &plane(0.0)[..]);
  let texels0 = texels(&lib, id);
  // min / max stretch: the points follow the plane exactly
  lib.set_fits_stretch(id, FitsStretch { mode: StretchMode::Percentile { low: 0.0, high: 100.0 }, transfer: Transfer::Log }).unwrap();
  assert_eq!((lib.fits(id).unwrap().black, lib.fits(id).unwrap().white), (100.0, 1200.0));
  let texels0_log = texels(&lib, id);
  lib.set_fits_plane(id, 2).unwrap();
  let f = lib.fits(id).unwrap();
  assert_eq!((f.metadata.planes, f.metadata.plane), (3, 2));
  assert_eq!(&*f.raster(), &plane(2.0)[..]);
  assert_eq!(f.stretch, FitsStretch { mode: StretchMode::Percentile { low: 0.0, high: 100.0 }, transfer: Transfer::Log });
  assert_eq!((f.black, f.white), (2100.0, 3200.0));
  assert!(f.metadata.to_json().contains("\"planes\":3,\"plane\":2,"));
  assert_eq!(f.metadata.facts.object.as_deref(), Some("cube"));
  // same relative stretch of the same ramp: the same texels as plane 0 under the same stretch
  assert_eq!(texels(&lib, id), texels0_log);
  assert_ne!(texels(&lib, id), texels0);
  assert_eq!(lib.copy_texture_rgba8(id, None), Ok((4, 3)));
  // a manual stretch stays manual across planes
  lib.set_fits_stretch(id, FitsStretch { mode: StretchMode::Manual { black: 0.0, white: 4000.0 }, transfer: Transfer::Linear }).unwrap();
  lib.set_fits_plane(id, 1).unwrap();
  let f = lib.fits(id).unwrap();
  assert_eq!((f.black, f.white, f.metadata.plane), (0.0, 4000.0, 1));
  assert_eq!(&*f.raster(), &plane(1.0)[..]);
  let t1 = texels(&lib, id);
  assert!(t1.chunks_exact(4).all(|p| p[3] == 255));
  // errors leave the asset as it was
  match lib.set_fits_plane(id, 3) {
    Err(AssetError::Fits(m)) => assert!(m.contains("plane 3 out of range"), "{m}"),
    other => panic!("{other:?}"),
  }
  assert_eq!(lib.fits(id).unwrap().metadata.plane, 1);
  assert_eq!(texels(&lib, id), t1);
  assert!(matches!(lib.set_fits_plane(id + 100, 0), Err(AssetError::NotFits)));
  // the same plane again is a no-op
  lib.set_fits_plane(id, 1).unwrap();
  assert_eq!(texels(&lib, id), t1);
  // a plain image has plane 0 only
  let ramp = s.write("ramp.fits", &ramp_fits());
  let id2 = lib.import_file(&ramp).unwrap().textures[0];
  assert!(lib.set_fits_plane(id2, 1).is_err());
  lib.set_fits_plane(id2, 0).unwrap();
  // the cache holds one raster and one texel file per asset: the replaced ones are gone
  let files: Vec<String> = std::fs::read_dir(s.0.join("asset_cache")).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
  assert_eq!(files.iter().filter(|f| f.ends_with(".avkf")).count(), 2, "{files:?}");
}

/// The tile-compressed cube twin behaves the same (CFITSIO reads `.fz`).
#[cfg(feature = "cfitsio")]
#[test]
fn fits_compressed_cube_plane_selection() {
  let src = aethervk_test_assets::test_asset!("fits_cube_4x3x3.fz");
  let s = Scratch::new("cube_fz");
  let mut lib = s.library();
  let id = lib.import_file(src.to_str().unwrap()).unwrap().textures[0];
  let f = lib.fits(id).unwrap();
  assert_eq!((f.metadata.planes, f.metadata.plane, f.metadata.compressed, f.metadata.hdu), (3, 0, true, 2));
  lib.set_fits_plane(id, 2).unwrap();
  let f = lib.fits(id).unwrap();
  assert_eq!(f.metadata.plane, 2);
  assert_eq!(f.raster()[8..12], [2100.0, 2200.0, 2300.0, 2400.0]);
  assert!(lib.set_fits_plane(id, 3).is_err());
}

/// A FITS tile in a HiPS directory layout keeps its HiPS facts (recognised from the import
/// path); a plain image and an image with a WCS carry none.
#[test]
fn fits_hips_tile_import_keeps_its_facts() {
  let s = Scratch::new("hips");
  std::fs::create_dir_all(s.0.join("Norder3/Dir0")).unwrap();
  let ramp: Vec<f64> = (0..64).map(|i| i as f64).collect();
  let rev: Vec<f64> = (0..64).map(|i| (63 - i) as f64).collect();
  let tile = s.write("Norder3/Dir0/Npix5.fits", &fits_bytes(-32, 8, 8, &[], &ramp));
  let plain = s.write("plain.fits", &fits_bytes(-32, 8, 8, &[], &rev));
  let mut lib = s.library();
  let id = lib.import_file(&tile).unwrap().textures[0];
  let h = lib.fits(id).unwrap().hips.as_ref().expect("HiPS facts");
  assert_eq!((h.order, h.npix, h.allsky), (3, Some(5), false));
  let id = lib.import_file(&plain).unwrap().textures[0];
  assert!(lib.fits(id).unwrap().hips.is_none());
  lib.clear();
}
