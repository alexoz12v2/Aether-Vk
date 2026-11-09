//! The FITS test corpus and the WCS / HiPS reference tables (`test_assets/fits_corpus/`, see its
//! README.md).
//!
//! - [`fits_corpus_manifest_is_consistent`]: the committed manifest parses (always runs).
//! - [`fits_corpus_opens_every_image`]: every manifest file of a downloaded corpus
//!   (`AETHERVK_FITS_CORPUS=<dir>`, fetched by `scripts/fetch_fits_corpus.sh`) meets its `expected`
//!   value through [`read_fits`], and the WCS of each opened image is built through [`wcs_check`].
//!   Without the variable it prints `skipped: set AETHERVK_FITS_CORPUS` and passes.
//! - [`fits_wcs_reference_tables_match`]: the engine's WCS against astropy / wcslib (wcstools for
//!   TNX / ZPX, AST for DSS, sunpy for helioprojective) on every committed reference header
//!   (always runs).
//! - [`fits_hips_reference_tables_consistent`]: the HiPS tile pixel convention of the committed
//!   HiPS tables (always runs).
//!
//! The engine's WCS is reached through one adapter, [`engine_wcs`]: it is the only place that names
//! the WCS API, so a new WCS implementation is wired by changing that function (and shrinking
//! [`WCS_KNOWN_GAPS`]).

extern crate std;

use alloc::{
  format,
  string::{String, ToString},
  vec::Vec,
};
use std::{collections::BTreeMap, path::PathBuf, println};

use super::fits_metadata::read_fits;
use super::{FitsHeader, Wcs};


// ─── manifest ────────────────────────────────────────────────────────────────

/// The sources of the corpus (`category` column).
const CATEGORIES: &[&str] = &["astropy-data", "calabretta", "cds", "heasarc", "hips", "nasa-samples", "wcs_extra"];

#[derive(Debug, Clone, PartialEq)]
enum Expected {
  /// a 2-D image (or a cube) the app must open: `Ok`, `nx, ny > 0`, a finite pixel (unless the
  /// note says the file's first plane is all NaN)
  Image,
  /// `Err` whose message contains the text (case-insensitive)
  Refuse(String),
  /// tables only: `Err`
  TableOnly,
  /// a deliberately malformed file: `Ok` or `Err`, never a panic
  Either,
  /// an archive the fetch script extracts; not opened
  Archive,
}

#[derive(Debug, Clone)]
struct Row {
  path: String,
  sha256: String,
  size: u64,
  source: String,
  category: String,
  expected: Expected,
  notes: String,
}

fn parse_expected(s: &str) -> Result<Expected, String> {
  Ok(match s {
    "image" => Expected::Image,
    "table-only" => Expected::TableOnly,
    "either" => Expected::Either,
    "archive" => Expected::Archive,
    _ => match s.strip_prefix("refuse:") {
      Some(r) if !r.trim().is_empty() => Expected::Refuse(r.trim().to_string()),
      _ => return Err(format!("unknown expected value {s:?}")),
    },
  })
}

fn parse_manifest(text: &str) -> Result<Vec<Row>, Vec<String>> {
  let mut rows = Vec::new();
  let mut errors = Vec::new();
  for (i, line) in text.lines().enumerate() {
    if line.starts_with('#') || line.trim().is_empty() {
      continue;
    }
    let f: Vec<&str> = line.split('\t').collect();
    let at = |what: &str| format!("manifest line {}: {what}", i + 1);
    if f.len() != 7 {
      errors.push(at(&format!("{} columns, 7 expected", f.len())));
      continue;
    }
    let sha_ok = f[1].len() == 64 && f[1].bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase());
    if !sha_ok {
      errors.push(at(&format!("bad sha256 {:?}", f[1])));
    }
    let size = match f[2].parse::<u64>() {
      Ok(s) if s > 0 => s,
      _ => {
        errors.push(at(&format!("bad size {:?}", f[2])));
        0
      }
    };
    if !CATEGORIES.contains(&f[4]) {
      errors.push(at(&format!("unknown category {:?}", f[4])));
    }
    if !f[0].starts_with(&format!("{}/", f[4])) {
      errors.push(at(&format!("path {:?} is not under its category {:?}", f[0], f[4])));
    }
    if f[0].contains("..") || f[0].starts_with('/') {
      errors.push(at(&format!("path {:?} leaves the corpus", f[0])));
    }
    if !(f[3].starts_with("https://") || f[3].starts_with("http://") || f[3].starts_with("tar:")) {
      errors.push(at(&format!("source {:?}: http(s):// or tar:<archive> expected", f[3])));
    }
    let expected = match parse_expected(f[5]) {
      Ok(e) => e,
      Err(e) => {
        errors.push(at(&e));
        continue;
      }
    };
    rows.push(Row {
      path: f[0].to_string(),
      sha256: f[1].to_string(),
      size,
      source: f[3].to_string(),
      category: f[4].to_string(),
      expected,
      notes: f[6].to_string(),
    });
  }
  // unique paths and hashes; every tar: source names an archive row
  let mut paths = BTreeMap::new();
  let mut hashes = BTreeMap::new();
  for r in &rows {
    if paths.insert(r.path.clone(), ()).is_some() {
      errors.push(format!("duplicate path {}", r.path));
    }
    if let Some(other) = hashes.insert(r.sha256.clone(), r.path.clone()) {
      errors.push(format!("{} and {other} have the same sha256 (keep one row)", r.path));
    }
  }
  for r in &rows {
    if let Some(arch) = r.source.strip_prefix("tar:")
      && !rows.iter().any(|a| a.path == arch && a.expected == Expected::Archive)
    {
      errors.push(format!("{}: archive {arch} has no `archive` row", r.path));
    }
  }
  if errors.is_empty() { Ok(rows) } else { Err(errors) }
}

fn load_manifest() -> Vec<Row> {
  // committed, never downloaded
  let path = aethervk_test_assets::path("fits_corpus/manifest.tsv").unwrap_or_else(|e| panic!("fits_corpus/manifest.tsv: {e}"));
  let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
  parse_manifest(&text).unwrap_or_else(|errs| panic!("manifest errors:\n  {}", errs.join("\n  ")))
}

#[test]
fn fits_corpus_manifest_is_consistent() {
  let rows = load_manifest();
  assert!(rows.len() > 100, "{} rows", rows.len());
  let mut per: BTreeMap<String, usize> = BTreeMap::new();
  for r in &rows {
    *per.entry(format!("{} {:?}", r.category, r.expected)).or_default() += 1;
  }
  for (k, n) in &per {
    println!("{n:5} {k}");
  }
  // the WCS reference set must be in the manifest: every projection of the Calabretta set
  for p in ["AZP", "TAN", "ZPN", "CEA", "MOL", "COD", "BON", "TSC", "QSC", "HPX"] {
    let name = format!("calabretta/fitswcs_maps/1904-66_{p}.fits");
    assert!(rows.iter().any(|r| r.path == name && r.expected == Expected::Image), "{name}");
  }
}

// ─── filters ─────────────────────────────────────────────────────────────────

/// `*` / `?` glob over the whole string.
fn glob_match(pat: &[u8], s: &[u8]) -> bool {
  match (pat.first(), s.first()) {
    (None, None) => true,
    (Some(b'*'), _) => glob_match(&pat[1..], s) || (!s.is_empty() && glob_match(pat, &s[1..])),
    (Some(b'?'), Some(_)) => glob_match(&pat[1..], &s[1..]),
    (Some(a), Some(b)) if a == b => glob_match(&pat[1..], &s[1..]),
    _ => false,
  }
}

/// `AETHERVK_FITS_CORPUS_FILTER`: a glob when it holds `*` / `?`, else a substring.
fn path_filter_matches(filter: &str, path: &str) -> bool {
  if filter.contains(['*', '?']) {
    glob_match(filter.as_bytes(), path.as_bytes())
  } else {
    path.contains(filter)
  }
}

#[test]
fn fits_corpus_filter_matching() {
  assert!(path_filter_matches("TAN", "calabretta/fitswcs_maps/1904-66_TAN.fits"));
  assert!(path_filter_matches("calabretta/*_T?N.fits", "calabretta/fitswcs_maps/1904-66_TAN.fits"));
  assert!(!path_filter_matches("calabretta/*.gz", "calabretta/fitswcs_maps/1904-66_TAN.fits"));
  assert!(path_filter_matches("*", "x"));
}

// ─── the engine's WCS (the one adapter) ──────────────────────────────────────

/// The engine's sky mapping of a header, as the tests use it.
trait EngineWcs {
  /// `(lon, lat)` degrees in the header's own frame (`RADESYS` / `GLON` / `ELON` …), `None`
  /// outside the projection's domain.
  fn pix_to_native(&self, x: f64, y: f64) -> Option<(f64, f64)>;
  /// FITS 1-based pixel of a native `(lon, lat)`, `None` when it has no image.
  fn native_to_pix(&self, lon: f64, lat: f64) -> Option<(f64, f64)>;
  /// ICRS `(ra, dec)` degrees of a pixel when the implementation converts frames; `None` = not
  /// provided (the ICRS columns of the tables are then not compared).
  fn pix_to_icrs(&self, _x: f64, _y: f64) -> Option<(f64, f64)> {
    None
  }
}

/// `sky_image::wcs::Wcs`: every Paper II projection, SIP / TPV / TNX / ZPX / DSS, all frames.
impl EngineWcs for Wcs {
  fn pix_to_native(&self, x: f64, y: f64) -> Option<(f64, f64)> {
    self.pixel_to_world(x, y)
  }
  fn native_to_pix(&self, lon: f64, lat: f64) -> Option<(f64, f64)> {
    self.world_to_pixel(lon, lat)
  }
  fn pix_to_icrs(&self, x: f64, y: f64) -> Option<(f64, f64)> {
    let (ra, dec) = self.pixel_to_radec_deg(x, y);
    (ra.is_finite() && dec.is_finite()).then_some((ra, dec))
  }
}

/// Builds the engine's WCS of a header: `Ok(None)` = no celestial WCS, `Err` = refused
/// (unsupported). The only function naming the WCS API.
fn engine_wcs(h: &FitsHeader) -> Result<Option<alloc::boxed::Box<dyn EngineWcs>>, String> {
  match Wcs::from_header(h)? {
    None => Ok(None),
    Some(w) => Ok(Some(alloc::boxed::Box::new(w))),
  }
}

/// What [`wcs_check`] found for one corpus image.
#[derive(Debug, Clone, PartialEq)]
enum WcsOutcome {
  /// no celestial `CTYPE`s
  None,
  /// mapping built; centre and corners map, the centre round-trips
  Ok,
  /// the engine refuses the WCS (reason)
  Unsupported(String),
  /// built but wrong (reason): non-finite centre, a round trip off by more than 1e-6 px, a panic
  Broken(String),
}

/// The WCS of an opened corpus image through [`engine_wcs`]: built, the centre and the four
/// corners mapped, the centre round-tripped. Panics are caught and reported as `Broken`.
fn wcs_check(h: &FitsHeader, nx: u32, ny: u32) -> WcsOutcome {
  let run = || -> WcsOutcome {
    let w = match engine_wcs(h) {
      Ok(None) => return WcsOutcome::None,
      Err(e) => return WcsOutcome::Unsupported(e),
      Ok(Some(w)) => w,
    };
    let (cx, cy) = ((nx as f64 + 1.0) / 2.0, (ny as f64 + 1.0) / 2.0);
    let Some((lon, lat)) = w.pix_to_native(cx, cy) else {
      return WcsOutcome::Broken(format!("centre ({cx}, {cy}) has no sky position"));
    };
    if !(lon.is_finite() && lat.is_finite() && lat.abs() <= 90.0 + 1e-9) {
      return WcsOutcome::Broken(format!("centre maps to ({lon}, {lat})"));
    }
    for (x, y) in [(1.0, 1.0), (nx as f64, 1.0), (1.0, ny as f64), (nx as f64, ny as f64)] {
      if let Some((a, b)) = w.pix_to_native(x, y)
        && !(a.is_finite() && b.is_finite())
      {
        return WcsOutcome::Broken(format!("corner ({x}, {y}) maps to ({a}, {b})"));
      }
    }
    // CSC: the COBE forward and inverse polynomials (Paper II section 5.6.2) only approximate each
    // other, so wcslib / astropy themselves round-trip to about 0.2 px; the reference tables
    // compare against astropy's own round trip instead
    let rt_tol = if h.get("CTYPE1").is_some_and(|c| c.contains("-CSC")) { 0.5 } else { 1e-6 };
    match w.native_to_pix(lon, lat) {
      Some((x, y)) if (x - cx).hypot(y - cy) <= rt_tol => WcsOutcome::Ok,
      r => WcsOutcome::Broken(format!("centre round trip ({cx}, {cy}) -> ({lon}, {lat}) -> {r:?}")),
    }
  };
  match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
    Ok(o) => o,
    Err(p) => WcsOutcome::Broken(format!("panic: {}", panic_text(&p))),
  }
}

fn panic_text(p: &alloc::boxed::Box<dyn core::any::Any + Send>) -> String {
  if let Some(s) = p.downcast_ref::<&str>() {
    s.to_string()
  } else if let Some(s) = p.downcast_ref::<String>() {
    s.clone()
  } else {
    "non-string panic".to_string()
  }
}

// ─── the corpus ──────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct Tally {
  opened: usize,
  refused: usize,
  failed: usize,
  panicked: usize,
  wcs_ok: usize,
  wcs_none: usize,
  wcs_unsupported: usize,
  wcs_broken: usize,
}

#[test]
fn fits_corpus_opens_every_image() {
  let Ok(root) = std::env::var("AETHERVK_FITS_CORPUS") else {
    println!("skipped: set AETHERVK_FITS_CORPUS=<dir> (scripts/fetch_fits_corpus.sh <dir>)");
    return;
  };
  let root = PathBuf::from(root);
  let filter = std::env::var("AETHERVK_FITS_CORPUS_FILTER").ok().filter(|f| !f.is_empty());
  let cats: Option<Vec<String>> = std::env::var("AETHERVK_FITS_CORPUS_CATEGORY")
    .ok()
    .filter(|c| !c.is_empty())
    .map(|c| c.split(',').map(|s| s.trim().to_string()).collect());
  let wcs_strict = std::env::var("AETHERVK_FITS_CORPUS_WCS_STRICT").is_ok_and(|v| v == "1");
  let verbose = std::env::var("AETHERVK_FITS_CORPUS_VERBOSE").is_ok_and(|v| v == "1");
  println!(
    "FITS corpus {} (read path: {})",
    root.display(),
    if cfg!(feature = "cfitsio") { "CFITSIO" } else { "pure-Rust primary HDU (no `cfitsio` feature)" }
  );

  let mut tally: BTreeMap<String, Tally> = BTreeMap::new();
  let mut unexpected: Vec<String> = Vec::new();
  let mut wcs_notes: BTreeMap<String, usize> = BTreeMap::new();
  let mut checked = 0usize;
  for row in load_manifest() {
    if row.expected == Expected::Archive {
      continue;
    }
    if let Some(f) = &filter
      && !path_filter_matches(f, &row.path)
    {
      continue;
    }
    if let Some(c) = &cats
      && !c.iter().any(|c| c == &row.category)
    {
      continue;
    }
    let path = root.join(&row.path);
    let bytes = match std::fs::read(&path) {
      Ok(b) => b,
      Err(e) => {
        unexpected.push(format!("{}: missing ({e}); run scripts/fetch_fits_corpus.sh", row.path));
        continue;
      }
    };
    if bytes.len() as u64 != row.size {
      unexpected.push(format!("{}: {} bytes, manifest says {} (fetch again)", row.path, bytes.len(), row.size));
      continue;
    }
    checked += 1;
    let t = tally.entry(row.category.clone()).or_default();
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| read_fits(&bytes)));
    let outcome = match res {
      Err(p) => {
        t.panicked += 1;
        unexpected.push(format!("{}: PANIC {} [{}]", row.path, panic_text(&p), row.notes));
        continue;
      }
      Ok(r) => r,
    };
    match (&row.expected, outcome) {
      (Expected::Image | Expected::Either, Ok((img, _meta))) => {
        // a cube whose first plane is blank in the file itself (the manifest note says so)
        let finite = img.data.iter().any(|v| v.is_finite()) || row.notes.contains("first plane all NaN");
        let good = img.nx > 0 && img.ny > 0 && finite && img.data.len() == img.nx as usize * img.ny as usize;
        if good {
          t.opened += 1;
        } else {
          t.failed += 1;
        }
        if !good && row.expected == Expected::Image {
          unexpected.push(format!(
            "{}: opened {}x{} (hdu {}) with {} pixels, finite: {finite} [{}]",
            row.path,
            img.nx,
            img.ny,
            img.hdu,
            img.data.len(),
            row.notes
          ));
        }
        let w = wcs_check(&img.header, img.nx, img.ny);
        if verbose {
          println!("  {:60} {}x{} hdu {} wcs {w:?}", row.path, img.nx, img.ny, img.hdu);
        }
        match w {
          WcsOutcome::Ok => t.wcs_ok += 1,
          WcsOutcome::None => t.wcs_none += 1,
          WcsOutcome::Unsupported(e) => {
            t.wcs_unsupported += 1;
            *wcs_notes.entry(e.clone()).or_default() += 1;
            if wcs_strict {
              unexpected.push(format!("{}: WCS unsupported: {e}", row.path));
            }
          }
          WcsOutcome::Broken(e) => {
            t.wcs_broken += 1;
            unexpected.push(format!("{}: WCS broken: {e}", row.path));
          }
        }
      }
      // the pure-Rust fallback reads only an uncompressed primary-HDU image: there, a refusal of
      // anything else is the documented behaviour; the corpus still checks it never panics and that
      // what it opens has the right pixels
      (Expected::Image, Err(e)) if !cfg!(feature = "cfitsio") => {
        t.refused += 1;
        if verbose {
          println!("  {:60} refused by the fallback: {e}", row.path);
        }
      }
      (Expected::Image, Err(e)) => {
        t.failed += 1;
        unexpected.push(format!("{}: refused ({e}) but must open [{}]", row.path, row.notes));
      }
      (Expected::Refuse(_) | Expected::TableOnly, Ok((img, _))) => {
        t.failed += 1;
        unexpected.push(format!(
          "{}: opened {}x{} (hdu {}) but expected {:?} [{}]",
          row.path, img.nx, img.ny, img.hdu, row.expected, row.notes
        ));
      }
      (Expected::Refuse(want), Err(e)) => {
        if !cfg!(feature = "cfitsio") || e.to_ascii_lowercase().contains(&want.to_ascii_lowercase()) {
          t.refused += 1;
        } else {
          t.failed += 1;
          unexpected.push(format!("{}: refused with {e:?}, expected a message containing {want:?}", row.path));
        }
      }
      (Expected::TableOnly | Expected::Either, Err(e)) => {
        t.refused += 1;
        if verbose {
          println!("  {:60} refused: {e}", row.path);
        }
      }
      (Expected::Archive, _) => unreachable!(),
    }
  }

  println!(
    "\n{:14} {:>7} {:>7} {:>7} {:>8} | {:>7} {:>7} {:>11} {:>7}",
    "category", "opened", "refused", "failed", "panicked", "wcs ok", "no wcs", "unsupported", "broken"
  );
  let mut total = Tally::default();
  for (c, t) in &tally {
    println!(
      "{c:14} {:>7} {:>7} {:>7} {:>8} | {:>7} {:>7} {:>11} {:>7}",
      t.opened, t.refused, t.failed, t.panicked, t.wcs_ok, t.wcs_none, t.wcs_unsupported, t.wcs_broken
    );
    total.opened += t.opened;
    total.refused += t.refused;
    total.failed += t.failed;
    total.panicked += t.panicked;
    total.wcs_ok += t.wcs_ok;
    total.wcs_none += t.wcs_none;
    total.wcs_unsupported += t.wcs_unsupported;
    total.wcs_broken += t.wcs_broken;
  }
  println!(
    "{:14} {:>7} {:>7} {:>7} {:>8} | {:>7} {:>7} {:>11} {:>7}   ({checked} files)",
    "total",
    total.opened,
    total.refused,
    total.failed,
    total.panicked,
    total.wcs_ok,
    total.wcs_none,
    total.wcs_unsupported,
    total.wcs_broken
  );
  if !wcs_notes.is_empty() {
    println!("\nWCS refusals (not failures unless AETHERVK_FITS_CORPUS_WCS_STRICT=1):");
    for (e, n) in &wcs_notes {
      println!("  {n:4}  {e}");
    }
  }
  assert!(checked > 0 || filter.is_some() || cats.is_some(), "no corpus file checked");
  assert!(unexpected.is_empty(), "{} unexpected results:\n  {}", unexpected.len(), unexpected.join("\n  "));
}

// ─── WCS reference tables ────────────────────────────────────────────────────

/// Reference cases the engine's WCS does not reproduce yet, with the reason. A case listed here
/// must fail (else the entry is stale and the test says so); any case not listed must match.
/// Empty: every table matches (CSC within the single-precision wcslib arithmetic it emulates).
const WCS_KNOWN_GAPS: &[(&str, &str)] = &[];

/// pix → world agreement (degrees of arc) and world → pix agreement (pixels). 1e-7°: every
/// double-precision projection agrees to ~1e-12°; CSC is computed in single precision by wcslib
/// (its polynomial constants are `float`), which the engine emulates to ~1e-8°.
const TOL_ARC_DEG: f64 = 1e-7;
const TOL_PIX: f64 = 1e-6;

struct RefRow {
  x: f64,
  y: f64,
  lon: f64,
  lat: f64,
  ra_icrs: f64,
  dec_icrs: f64,
  /// astropy's own world → pixel of `(lon, lat)`: the pixel a forward transformation must give
  /// (wcslib's CSC inverse polynomial is only an approximation of its forward one, 0.2 px)
  x_rt: f64,
  y_rt: f64,
}

fn parse_f(s: &str) -> f64 {
  if s == "nan" { f64::NAN } else { s.parse().unwrap_or_else(|_| panic!("bad number {s:?}")) }
}

fn arc_deg(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
  let d = core::f64::consts::PI / 180.0;
  let (l1, b1, l2, b2) = (lon1 * d, lat1 * d, lon2 * d, lat2 * d);
  // haversine: accurate at the tiny separations compared here
  let h = ((b2 - b1) / 2.0).sin().powi(2) + b1.cos() * b2.cos() * ((l2 - l1) / 2.0).sin().powi(2);
  2.0 * h.sqrt().min(1.0).asin() / d
}

/// Compares the engine's WCS of one reference case; `Ok(None)` = match, `Ok(Some(reason))` =
/// refused, `Err(reason)` = wrong. A `# icrs_tol_deg: <value>` comment of the table loosens the
/// ICRS check (helioprojective: the engine's truncated Earth ephemeris, 3e-5° = 0.1″).
fn compare_case(hdr: &[u8], tsv: &str) -> Result<Option<String>, String> {
  let h = FitsHeader::parse_text(hdr);
  let icrs_tol = tsv
    .lines()
    .find_map(|l| l.strip_prefix("# icrs_tol_deg:"))
    .map(|v| v.trim().parse::<f64>().expect("icrs_tol_deg"))
    .unwrap_or(1e-6);
  let rows: Vec<RefRow> = tsv
    .lines()
    .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    .map(|l| {
      let f: Vec<f64> = l.split('\t').map(parse_f).collect();
      assert!(f.len() >= 6, "short row {l:?}");
      let (x_rt, y_rt) = if f.len() >= 8 && f[6].is_finite() { (f[6], f[7]) } else { (f[0], f[1]) };
      RefRow { x: f[0], y: f[1], lon: f[2], lat: f[3], ra_icrs: f[4], dec_icrs: f[5], x_rt, y_rt }
    })
    .collect();
  let w = match engine_wcs(&h) {
    Err(e) => return Ok(Some(e)),
    Ok(None) => return Ok(Some("no celestial WCS recognised".to_string())),
    Ok(Some(w)) => w,
  };
  let mut worst = (0.0f64, 0.0f64);
  for r in &rows {
    let got = w.pix_to_native(r.x, r.y);
    if !(r.lon.is_finite() && r.lat.is_finite()) {
      if let Some((a, b)) = got
        && a.is_finite()
        && b.is_finite()
      {
        return Err(format!("({}, {}) is outside the projection (wcslib) but maps to ({a}, {b})", r.x, r.y));
      }
      continue;
    }
    let Some((lon, lat)) = got else {
      return Err(format!("({}, {}) has no sky position, wcslib: ({}, {})", r.x, r.y, r.lon, r.lat));
    };
    let e = arc_deg(lon, lat, r.lon, r.lat);
    worst.0 = worst.0.max(e);
    if !(e <= TOL_ARC_DEG) {
      return Err(format!("pix ({}, {}) -> ({lon}, {lat}), wcslib ({}, {}): {e:.3e} deg", r.x, r.y, r.lon, r.lat));
    }
    match w.native_to_pix(r.lon, r.lat) {
      Some((x, y)) if (x - r.x_rt).hypot(y - r.y_rt) <= TOL_PIX => worst.1 = worst.1.max((x - r.x_rt).hypot(y - r.y_rt)),
      p => return Err(format!("world ({}, {}) -> {p:?}, expected ({}, {})", r.lon, r.lat, r.x_rt, r.y_rt)),
    }
    if r.ra_icrs.is_finite()
      && let Some((ra, dec)) = w.pix_to_icrs(r.x, r.y)
    {
      let e = arc_deg(ra, dec, r.ra_icrs, r.dec_icrs);
      if !(e <= icrs_tol) {
        return Err(format!("pix ({}, {}) ICRS ({ra}, {dec}), reference ({}, {}): {e:.3e} deg", r.x, r.y, r.ra_icrs, r.dec_icrs));
      }
    }
  }
  Ok(None)
}

#[test]
fn fits_wcs_reference_tables_match() {
  let dir = aethervk_test_assets::test_asset_dir!("fits_corpus/wcs_reference");
  let mut names: Vec<String> = std::fs::read_dir(&dir)
    .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
    .filter_map(|e| e.ok()?.file_name().into_string().ok()?.strip_suffix(".tsv").map(str::to_string))
    .collect();
  names.sort();
  assert!(names.len() >= 60, "{} reference cases", names.len());
  let gaps: BTreeMap<&str, &str> = WCS_KNOWN_GAPS.iter().copied().collect();
  let (mut matched, mut refused, mut wrong) = (Vec::new(), Vec::new(), Vec::new());
  let mut problems = Vec::new();
  for name in &names {
    let hdr = std::fs::read(dir.join(format!("{name}.hdr"))).unwrap_or_else(|e| panic!("{name}.hdr: {e}"));
    let tsv = std::fs::read_to_string(dir.join(format!("{name}.tsv"))).unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compare_case(&hdr, &tsv)))
      .unwrap_or_else(|p| Err(format!("panic: {}", panic_text(&p))));
    let failed = match &r {
      Ok(None) => {
        matched.push(name.clone());
        false
      }
      Ok(Some(why)) => {
        refused.push(format!("{name}: {why}"));
        true
      }
      Err(why) => {
        wrong.push(format!("{name}: {why}"));
        true
      }
    };
    match (failed, gaps.get(name.as_str())) {
      (true, None) => problems.push(format!(
        "{name}: {}",
        match &r {
          Ok(Some(w)) => format!("refused: {w}"),
          Err(w) => format!("WRONG: {w}"),
          Ok(None) => unreachable!(),
        }
      )),
      (false, Some(_)) => problems.push(format!("{name}: now matches wcslib, remove it from WCS_KNOWN_GAPS")),
      _ => {}
    }
  }
  for g in gaps.keys() {
    if !names.iter().any(|n| n == g) {
      problems.push(format!("WCS_KNOWN_GAPS names {g}, which has no reference table"));
    }
  }
  println!("{} reference cases: {} match, {} refused, {} wrong", names.len(), matched.len(), refused.len(), wrong.len());
  println!("match: {}", matched.join(" "));
  for r in refused.iter().chain(&wrong) {
    println!("  {r}");
  }
  assert!(problems.is_empty(), "{} WCS reference problems:\n  {}", problems.len(), problems.join("\n  "));
}

// ─── HiPS reference tables ───────────────────────────────────────────────────

/// HiPS tile pixel `(x, y)` (0-based column, row; row 0 = first stored row) of a tile `npix`
/// of width `2^s` → its nested HEALPix index at order + s (README.md: `ix = W-1-y` on the even
/// bits, `iy = x` on the odd bits).
fn hips_pixel_nested(npix: u64, s: u32, x: u64, y: u64) -> u64 {
  let w = 1u64 << s;
  let (ix, iy) = (w - 1 - y, x);
  let mut sub = 0u64;
  for i in 0..s {
    sub |= ((ix >> i) & 1) << (2 * i);
    sub |= ((iy >> i) & 1) << (2 * i + 1);
  }
  (npix << (2 * s)) | sub
}

#[test]
fn fits_hips_reference_tables_consistent() {
  let dir = aethervk_test_assets::test_asset_dir!("fits_corpus/hips_reference");
  for width in [512u64, 64] {
    let s = width.trailing_zeros();
    let text = std::fs::read_to_string(dir.join(format!("tile_pixels_w{width}.tsv"))).unwrap();
    let mut n = 0;
    for l in text.lines().filter(|l| !l.starts_with('#')) {
      let f: Vec<&str> = l.split('\t').collect();
      assert_eq!(f.len(), 16, "{l}");
      let p = |i: usize| f[i].parse::<u64>().unwrap();
      let (order, npix, w, x, y, nested) = (p(0), p(1), p(2), p(3), p(4), p(5));
      assert_eq!(w, width);
      assert!(npix < 12 << (2 * order), "{l}");
      assert_eq!(hips_pixel_nested(npix, s, x, y), nested, "{l}");
      for i in 6..16 {
        let v: f64 = f[i].parse().unwrap();
        assert!(v.is_finite() && v.abs() <= 360.0, "{l}");
      }
      n += 1;
    }
    assert!(n > 100, "{n} rows in tile_pixels_w{width}.tsv");
  }
  let tiles = std::fs::read_to_string(dir.join("tiles.tsv")).unwrap();
  let rows = tiles.lines().filter(|l| !l.starts_with('#')).count();
  assert_eq!(rows, 12 + 48 + 4 + 5, "tiles.tsv rows");
}
