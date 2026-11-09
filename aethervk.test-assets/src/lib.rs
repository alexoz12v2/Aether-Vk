//! Test-only access to the files under `test_assets/`.
//!
//! Most test assets are not committed: `test_assets/ASSETS.tsv` lists groups of them, each packed
//! into a deterministic `.tar.gz` that the tests download on demand (see `test_assets/README.md`).
//! [`path`] / [`dir`] resolve a `test_assets`-relative path:
//!
//! 1. `<repo>/test_assets/<rel>` when it exists (a committed file or a developer's local copy);
//! 2. else the manifest group whose `paths` entry equals `rel` or is a directory prefix of it;
//! 3. that group's cache dir `<cache>/<group>-<sha12>/` once `.complete` is in it, filled under an
//!    exclusive lock on `<cache>/<group>.lock` from `$AETHERVK_TEST_ASSETS_MIRROR/<file>` or a
//!    download of `<base><file>`, checked against the manifest size and sha256.
//!
//! Any failure is an `Err` with the reason; the [`test_asset!`] macro prints
//! `skipped: <rel>: <reason>` and returns from the calling test. With
//! `AETHERVK_TEST_ASSETS_REQUIRED=1` (CI) a failure panics instead, so assets cannot skip silently.
//!
//! Environment: `AETHERVK_TEST_ASSETS_CACHE` (default `<repo>/target/test-assets`),
//! `AETHERVK_TEST_ASSETS_MIRROR` (a directory holding the tarballs), `AETHERVK_TEST_ASSETS_BASE`
//! (overrides the manifest's `# base` URL), `AETHERVK_TEST_ASSETS_REQUIRED`.
//!
//! This crate is a dev-dependency only; nothing shipped links it.

use std::{
  fs::{self, File},
  path::PathBuf,
  process::Command,
  time::Duration,
};

/// The manifest, relative to the repository root.
const MANIFEST: &str = "test_assets/ASSETS.tsv";

/// Resolves a `test_assets`-relative file or directory, downloading its group when needed.
/// Panics instead of returning `Err` when `AETHERVK_TEST_ASSETS_REQUIRED=1`.
pub fn path(rel: &str) -> Result<PathBuf, String> {
  finish(required_from_env(), rel, Config::from_env().and_then(|c| resolve(&c, rel, Kind::Any)))
}

/// [`path`] for a directory: the result must be a directory.
pub fn dir(rel: &str) -> Result<PathBuf, String> {
  finish(required_from_env(), rel, Config::from_env().and_then(|c| resolve(&c, rel, Kind::Dir)))
}

/// `<repo>/test_assets/<rel>` when it exists locally, without any download and never panicking:
/// for optional files that are not in the manifest (a developer's untracked extras).
pub fn local(rel: &str) -> Option<PathBuf> {
  let p = repo_root().ok()?.join("test_assets").join(rel);
  p.exists().then_some(p)
}

/// `<repo>/test_assets`, the source tree the asset tarballs are packed from. Only for helpers that
/// write fixtures there for repacking (a developer action); tests read through [`path`].
pub fn source_root() -> Result<PathBuf, String> {
  Ok(repo_root()?.join("test_assets"))
}

/// The repository root: the first ancestor of `CARGO_MANIFEST_DIR` (the test's crate at run time,
/// else this crate) that holds `test_assets/ASSETS.tsv`.
pub fn repo_root() -> Result<PathBuf, String> {
  let starts = std::env::var_os("CARGO_MANIFEST_DIR")
    .map(PathBuf::from)
    .into_iter()
    .chain([PathBuf::from(env!("CARGO_MANIFEST_DIR"))]);
  for start in starts {
    if let Some(root) = start.ancestors().find(|a| a.join(MANIFEST).is_file()) {
      return Ok(root.to_path_buf());
    }
  }
  Err(format!("no ancestor of CARGO_MANIFEST_DIR holds {MANIFEST}"))
}

/// Resolves `rel` with [`path`], or prints `skipped: <rel>: <reason>` and returns from the
/// calling function (a test returning `()`).
#[macro_export]
macro_rules! test_asset {
  ($rel:expr) => {
    match $crate::path($rel) {
      ::core::result::Result::Ok(p) => p,
      ::core::result::Result::Err(e) => {
        $crate::__skip($rel, &e);
        return;
      }
    }
  };
}

/// [`test_asset!`] for a directory ([`dir`]).
#[macro_export]
macro_rules! test_asset_dir {
  ($rel:expr) => {
    match $crate::dir($rel) {
      ::core::result::Result::Ok(p) => p,
      ::core::result::Result::Err(e) => {
        $crate::__skip($rel, &e);
        return;
      }
    }
  };
}

/// Unwraps a `Result<T, String>` built from [`path`] / [`dir`] calls (a helper gathering several
/// assets), or prints `skipped: <reason>` and returns from the calling test.
#[macro_export]
macro_rules! skip_on_err {
  ($result:expr) => {
    match $result {
      ::core::result::Result::Ok(v) => v,
      ::core::result::Result::Err(e) => {
        $crate::__skip("", &e);
        return;
      }
    }
  };
}

#[doc(hidden)]
pub fn __skip(rel: &str, reason: &str) {
  if rel.is_empty() {
    println!("skipped: {reason}");
  } else {
    println!("skipped: {rel}: {reason}");
  }
}

// ─── resolution ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Kind {
  Any,
  Dir,
}

#[derive(Clone, Debug)]
struct Config {
  repo: PathBuf,
  cache: PathBuf,
  mirror: Option<PathBuf>,
  base: Option<String>,
}

impl Config {
  fn from_env() -> Result<Config, String> {
    Ok(Config::from_env_at(repo_root()?))
  }

  fn from_env_at(repo: PathBuf) -> Config {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty());
    Config {
      cache: var("AETHERVK_TEST_ASSETS_CACHE").map(PathBuf::from).unwrap_or_else(|| repo.join("target/test-assets")),
      mirror: var("AETHERVK_TEST_ASSETS_MIRROR").map(PathBuf::from),
      base: var("AETHERVK_TEST_ASSETS_BASE").map(|v| v.to_string_lossy().into_owned()),
      repo,
    }
  }
}

fn required_from_env() -> bool {
  std::env::var("AETHERVK_TEST_ASSETS_REQUIRED").is_ok_and(|v| v.trim() == "1")
}

fn finish(required: bool, rel: &str, r: Result<PathBuf, String>) -> Result<PathBuf, String> {
  match r {
    Err(e) if required => panic!("test asset {rel}: {e} (AETHERVK_TEST_ASSETS_REQUIRED=1)"),
    r => r,
  }
}

#[derive(Clone, Debug, PartialEq)]
struct Group {
  name: String,
  paths: Vec<String>,
  file: String,
  sha256: String,
  size: u64,
}

#[derive(Debug)]
struct Manifest {
  base: Option<String>,
  groups: Vec<Group>,
}

fn parse_manifest(text: &str) -> Result<Manifest, String> {
  let mut m = Manifest { base: None, groups: Vec::new() };
  for (i, line) in text.lines().enumerate() {
    let line = line.trim_end_matches('\r');
    if line.trim().is_empty() {
      continue;
    }
    if let Some(comment) = line.strip_prefix('#') {
      if let Some(base) = comment.trim().strip_prefix("base ") {
        m.base = Some(base.trim().to_string());
      }
      continue;
    }
    let cols: Vec<&str> = line.split('\t').collect();
    let [name, paths, file, sha256, size] = cols[..] else {
      return Err(format!("{MANIFEST} line {}: {} columns, expected 5", i + 1, cols.len()));
    };
    m.groups.push(Group {
      name: name.to_string(),
      paths: paths.split(',').map(|p| p.trim().trim_end_matches('/').to_string()).filter(|p| !p.is_empty()).collect(),
      file: file.to_string(),
      sha256: sha256.to_ascii_lowercase(),
      size: size.trim().parse().map_err(|e| format!("{MANIFEST} line {}: size {size:?}: {e}", i + 1))?,
    });
  }
  Ok(m)
}

impl Manifest {
  /// The group with a `paths` entry equal to `rel` or a directory prefix of it.
  fn group_of(&self, rel: &str) -> Option<&Group> {
    self.groups.iter().find(|g| {
      g.paths.iter().any(|e| rel == e || rel.strip_prefix(e.as_str()).is_some_and(|rest| rest.starts_with('/')))
    })
  }
}

fn normalise(rel: &str) -> String {
  let rel = rel.replace('\\', "/");
  let rel = rel.trim_start_matches("./").trim_end_matches('/');
  rel.to_string()
}

fn check_kind(p: PathBuf, kind: Kind) -> Result<PathBuf, String> {
  match kind {
    Kind::Dir if !p.is_dir() => Err(format!("{} is not a directory", p.display())),
    _ => Ok(p),
  }
}

fn resolve(cfg: &Config, rel: &str, kind: Kind) -> Result<PathBuf, String> {
  let rel = normalise(rel);
  let local = cfg.repo.join("test_assets").join(&rel);
  if local.exists() {
    return check_kind(local, kind);
  }
  let manifest_path = cfg.repo.join(MANIFEST);
  let text = fs::read_to_string(&manifest_path).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
  let manifest = parse_manifest(&text)?;
  let group = manifest.group_of(&rel).ok_or_else(|| "not in ASSETS.tsv".to_string())?;
  let group_dir = group_dir(cfg, group);
  if !group_dir.join(".complete").is_file() {
    fill_once(cfg, manifest.base.as_deref(), group)?;
  }
  let p = group_dir.join(&rel);
  if !p.exists() {
    return Err(format!("not in the {} tarball ({})", group.name, group.file));
  }
  check_kind(p, kind)
}

fn group_dir(cfg: &Config, g: &Group) -> PathBuf {
  cfg.cache.join(format!("{}-{}", g.name, &g.sha256[..g.sha256.len().min(12)]))
}

/// Groups that failed to fill in this process, with the reason: an offline run tries each group
/// once instead of waiting for a timeout in every test.
static FAILED: std::sync::Mutex<Vec<(String, String)>> = std::sync::Mutex::new(Vec::new());

fn fill_once(cfg: &Config, manifest_base: Option<&str>, g: &Group) -> Result<(), String> {
  let key = format!("{cfg:?} {g:?}");
  let failed = |k: &str| FAILED.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(f, _)| f == k).map(|(_, e)| e.clone());
  if let Some(e) = failed(&key) {
    return Err(e);
  }
  fill(cfg, manifest_base, g).inspect_err(|e| FAILED.lock().unwrap_or_else(|p| p.into_inner()).push((key, e.clone())))
}

#[cfg(test)]
static EXTRACTIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Fills `<cache>/<group>-<sha12>/` under the group's file lock (threads and processes).
fn fill(cfg: &Config, manifest_base: Option<&str>, g: &Group) -> Result<(), String> {
  let cache = &cfg.cache;
  fs::create_dir_all(cache).map_err(|e| format!("create {}: {e}", cache.display()))?;
  let lock_path = cache.join(format!("{}.lock", g.name));
  let lock = File::options()
    .create(true)
    .truncate(false)
    .write(true)
    .open(&lock_path)
    .map_err(|e| format!("open {}: {e}", lock_path.display()))?;
  lock.lock().map_err(|e| format!("lock {}: {e}", lock_path.display()))?;
  let dest = group_dir(cfg, g);
  if dest.join(".complete").is_file() {
    return Ok(());
  }
  let bytes = fetch(cfg, manifest_base, g)?;
  if bytes.len() as u64 != g.size {
    return Err(format!("{}: {} bytes, ASSETS.tsv says {}", g.file, bytes.len(), g.size));
  }
  let got = hex(&sha256(&bytes));
  if got != g.sha256 {
    return Err(format!("{}: sha256 {got}, ASSETS.tsv says {}", g.file, g.sha256));
  }
  let pid = std::process::id();
  let stem = dest.file_name().unwrap().to_string_lossy().into_owned();
  let tarball = cache.join(format!("{stem}.tar.gz.tmp-{pid}"));
  let tmp = cache.join(format!("{stem}.tmp-{pid}"));
  let _ = fs::remove_dir_all(&tmp);
  let result = (|| {
    fs::write(&tarball, &bytes).map_err(|e| format!("write {}: {e}", tarball.display()))?;
    fs::create_dir_all(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
    let out = Command::new("tar")
      .arg("-xzf")
      .arg(&tarball)
      .arg("-C")
      .arg(&tmp)
      .output()
      .map_err(|e| format!("run tar: {e}"))?;
    if !out.status.success() {
      return Err(format!("tar -xzf {}: {} {}", g.file, out.status, String::from_utf8_lossy(&out.stderr).trim()));
    }
    #[cfg(test)]
    EXTRACTIONS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    if dest.exists() {
      // a run that died between the rename and `.complete`
      fs::remove_dir_all(&dest).map_err(|e| format!("remove stale {}: {e}", dest.display()))?;
    }
    fs::rename(&tmp, &dest).map_err(|e| format!("rename to {}: {e}", dest.display()))?;
    fs::write(dest.join(".complete"), format!("{}\n", g.sha256)).map_err(|e| format!("write .complete: {e}"))
  })();
  let _ = fs::remove_file(&tarball);
  let _ = fs::remove_dir_all(&tmp);
  result
}

fn fetch(cfg: &Config, manifest_base: Option<&str>, g: &Group) -> Result<Vec<u8>, String> {
  if let Some(mirror) = &cfg.mirror {
    let p = mirror.join(&g.file);
    return fs::read(&p).map_err(|e| format!("mirror {}: {e}", p.display()));
  }
  let mut base = cfg
    .base
    .clone()
    .or_else(|| manifest_base.map(str::to_string))
    .ok_or_else(|| format!("no `# base` line in {MANIFEST} and AETHERVK_TEST_ASSETS_BASE is unset"))?;
  if !base.ends_with('/') {
    base.push('/');
  }
  let url = format!("{base}{}", g.file);
  let client = reqwest::blocking::Client::builder()
    .connect_timeout(Duration::from_secs(15))
    .timeout(Duration::from_secs(300))
    .build()
    .map_err(|e| format!("http client: {e}"))?;
  let resp = client.get(&url).send().map_err(|e| format!("download {url}: {e}"))?;
  let resp = resp.error_for_status().map_err(|e| format!("download {url}: {e}"))?;
  let bytes = resp.bytes().map_err(|e| format!("download {url}: {e}"))?;
  Ok(bytes.to_vec())
}

fn hex(bytes: &[u8]) -> String {
  bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// FIPS 180-4 SHA-256, std-only.
fn sha256(data: &[u8]) -> [u8; 32] {
  const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ];
  let mut h: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ];
  let mut msg = data.to_vec();
  let bit_len = (data.len() as u64).wrapping_mul(8);
  msg.push(0x80);
  while msg.len() % 64 != 56 {
    msg.push(0);
  }
  msg.extend_from_slice(&bit_len.to_be_bytes());
  let mut w = [0u32; 64];
  for chunk in msg.as_chunks::<64>().0 {
    for (i, word) in chunk.as_chunks::<4>().0.iter().enumerate() {
      w[i] = u32::from_be_bytes(*word);
    }
    for i in 16..64 {
      let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
      let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
    for i in 0..64 {
      let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
      let ch = (e & f) ^ (!e & g);
      let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
      let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
      let maj = (a & b) ^ (a & c) ^ (b & c);
      let t2 = s0.wrapping_add(maj);
      hh = g;
      g = f;
      f = e;
      e = d.wrapping_add(t1);
      d = c;
      c = b;
      b = a;
      a = t1.wrapping_add(t2);
    }
    for (slot, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
      *slot = slot.wrapping_add(v);
    }
  }
  let mut out = [0u8; 32];
  for (i, v) in h.iter().enumerate() {
    out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
  }
  out
}

#[cfg(test)]
mod tests;
