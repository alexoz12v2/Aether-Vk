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
  let s = Scratch::new("fixture");
  let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_assets/fits_lt_67p_header_4x3.fits");
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
