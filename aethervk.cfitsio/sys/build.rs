//! Build script of the CFITSIO sys crate: makes a static `cfitsio` (+ `zlib`) available and emits
//! the link lines. Three sources, first match wins:
//!
//! 1. `AETHERVK_CFITSIO_DIR=<prefix>`: a prebuilt tree with `<prefix>/lib/{libcfitsio.a,libz.a}`
//!    (`cfitsio.lib`, `zlib.lib` on Windows) and `<prefix>/include/fitsio.h` (offline / emergency).
//! 2. `AETHERVK_CFITSIO_SYSTEM=1`: the distribution's shared `cfitsio` through `pkg-config`
//!    (Linux developer convenience; the ABI then comes from whatever version is installed).
//! 3. vcpkg (default): the pinned vcpkg release tarball is downloaded with `curl`, checked against
//!    a pinned SHA-256, unpacked with `tar` into `<repo>/.vcpkg/`, bootstrapped, and asked to
//!    install `cfitsio[bzip2]` (classic mode: zlib + bzip2, non-reentrant) into
//!    `<OUT_DIR>/vcpkg_installed`. The binary cache lives in `<repo>/.vcpkg-bincache` (or
//!    `VCPKG_DEFAULT_BINARY_CACHE`), so a second install is a few-second copy.
//!    `AETHERVK_VCPKG_ROOT` / `VCPKG_ROOT` are honoured when they carry the cfitsio 4.7.0 port.
//!
//! No version-control tool is ever invoked by this script. Everything is std-only.

use std::{
  env, fs,
  path::{Path, PathBuf},
  process::{Command, Output},
  time::{Duration, SystemTime},
};

const VCPKG_TAG: &str = "2026.10.16";
const VCPKG_TARBALL_URL: &str =
  "https://github.com/microsoft/vcpkg/archive/refs/tags/2026.10.16.tar.gz";
const VCPKG_TARBALL_SHA256: &str =
  "c905d9c28dc6792c446647081bda071780aefce91cbdebf8eccd619f7fde8541";
const CFITSIO_PORT_VERSION: &str = "4.7.0";
const BOOTSTRAP_MARKER: &str = ".aethervk-bootstrap-ok";
const LOCK_STALE_AFTER: Duration = Duration::from_secs(2 * 60 * 60);

const HINT: &str = "hint: set AETHERVK_CFITSIO_DIR=<prefix> to a prebuilt cfitsio (lib/ + include/), \
AETHERVK_CFITSIO_SYSTEM=1 to use the distribution's cfitsio through pkg-config, or build the core \
without the `cfitsio` feature (cargo build -p aethervk-core-cdylib --no-default-features)";

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=wrapper.h");
  println!("cargo:rerun-if-changed=src/bindings.rs");
  for v in [
    "AETHERVK_CFITSIO_DIR",
    "AETHERVK_CFITSIO_SYSTEM",
    "AETHERVK_VCPKG_ROOT",
    "VCPKG_ROOT",
    "VCPKG_DEFAULT_BINARY_CACHE",
  ] {
    println!("cargo:rerun-if-env-changed={v}");
  }

  let target = env::var("TARGET").expect("TARGET");
  let host = env::var("HOST").expect("HOST");
  let windows = target.contains("windows");

  if let Some(prefix) = env::var_os("AETHERVK_CFITSIO_DIR") {
    link_prefix(Path::new(&prefix), windows, "AETHERVK_CFITSIO_DIR");
    return;
  }
  if env::var("AETHERVK_CFITSIO_SYSTEM").map(|v| v == "1").unwrap_or(false) {
    link_system();
    return;
  }

  let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
  let repo_root = manifest_dir
    .parent()
    .and_then(Path::parent)
    .expect("crate lives two levels below the repo root")
    .to_path_buf();
  let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));

  let triplet = triplet_of(&target).unwrap_or_else(|| {
    panic!("aethervk-cfitsio-sys: no vcpkg triplet for target `{target}`; {HINT}");
  });
  let host_triplet = triplet_of(&host).unwrap_or(triplet);

  let vcpkg_dir = repo_root.join(".vcpkg");
  fs::create_dir_all(&vcpkg_dir).unwrap_or_else(|e| panic!("create {}: {e}", vcpkg_dir.display()));
  let _lock = DirLock::acquire(&vcpkg_dir.join(".lock"));

  let root = env_vcpkg_root().unwrap_or_else(|| ensure_vcpkg_tree(&vcpkg_dir));
  let exe = ensure_bootstrapped(&root, windows_host(&host));

  let install_root = out_dir.join("vcpkg_installed");
  let bincache = env::var_os("VCPKG_DEFAULT_BINARY_CACHE")
    .map(PathBuf::from)
    .unwrap_or_else(|| repo_root.join(".vcpkg-bincache"));
  fs::create_dir_all(&bincache).unwrap_or_else(|e| panic!("create {}: {e}", bincache.display()));
  fs::create_dir_all(&install_root)
    .unwrap_or_else(|e| panic!("create {}: {e}", install_root.display()));

  let mut cmd = Command::new(&exe);
  cmd
    .arg("install")
    .arg(format!("cfitsio[bzip2]:{triplet}"))
    .arg(format!("--host-triplet={host_triplet}"))
    .arg(format!("--vcpkg-root={}", root.display()))
    .arg(format!("--x-install-root={}", install_root.display()))
    .arg("--no-print-usage")
    .arg("--clean-after-build")
    // an install root left by an earlier feature set is rebuilt rather than refused
    .arg("--recurse")
    .current_dir(&out_dir)
    .env("VCPKG_DISABLE_METRICS", "1")
    .env("VCPKG_ROOT", &root)
    .env("VCPKG_DEFAULT_BINARY_CACHE", &bincache);
  // a manifest-mode variable would make classic mode fail
  cmd.env_remove("VCPKG_MANIFEST_DIR").env_remove("VCPKG_MANIFEST_MODE");
  let out = run(&mut cmd, "vcpkg install cfitsio");
  for line in String::from_utf8_lossy(&out.stdout).lines().filter(|l| {
    l.starts_with("Elapsed time")
      || l.starts_with("Restored")
      || l.starts_with("Total install time")
  }) {
    println!("cargo:warning=vcpkg: {line}");
  }

  let prefix = install_root.join(triplet);
  link_prefix(&prefix, windows, "the vcpkg install");
}

// ───────────────────────────── link emission ─────────────────────────────

fn link_prefix(prefix: &Path, windows: bool, origin: &str) {
  let lib = prefix.join("lib");
  let include = prefix.join("include");
  let (cfitsio, zlib, bzip2) = if windows {
    ("cfitsio.lib", "zlib.lib", "bz2.lib")
  } else {
    ("libcfitsio.a", "libz.a", "libbz2.a")
  };
  for f in [cfitsio, zlib, bzip2] {
    let p = lib.join(f);
    if !p.is_file() {
      panic!(
        "aethervk-cfitsio-sys: {} from {origin} does not exist; {HINT}",
        p.display()
      );
    }
  }
  if !include.join("fitsio.h").is_file() {
    panic!(
      "aethervk-cfitsio-sys: {} from {origin} does not exist; {HINT}",
      include.join("fitsio.h").display()
    );
  }
  println!("cargo:rustc-link-search=native={}", lib.display());
  println!("cargo:rustc-link-lib=static=cfitsio");
  println!(
    "cargo:rustc-link-lib=static={}",
    if windows { "zlib" } else { "z" }
  );
  // bzip2 (the cfitsio `bzip2` feature, and the wrapper's own `BZ2_bzBuffToBuffDecompress`)
  println!("cargo:rustc-link-lib=static=bz2");
  if !windows {
    println!("cargo:rustc-link-lib=m");
  }
  println!("cargo:include={}", include.display());
  println!("cargo:root={}", prefix.display());
}

fn link_system() {
  let version = run(
    Command::new("pkg-config").args(["--modversion", "cfitsio"]),
    "pkg-config --modversion cfitsio",
  );
  let version = String::from_utf8_lossy(&version.stdout).trim().to_string();
  let major: u32 = version.split('.').next().and_then(|s| s.parse().ok()).unwrap_or(0);
  if major < 4 {
    panic!("aethervk-cfitsio-sys: system cfitsio {version} is older than 4.0; {HINT}");
  }
  println!(
    "cargo:warning=aethervk-cfitsio-sys: linking the system cfitsio {version} (shared); the bindings were generated from 4.7.0"
  );
  let libs = run(
    Command::new("pkg-config").args(["--libs", "cfitsio"]),
    "pkg-config --libs cfitsio",
  );
  for flag in String::from_utf8_lossy(&libs.stdout).split_whitespace() {
    if let Some(dir) = flag.strip_prefix("-L") {
      println!("cargo:rustc-link-search=native={dir}");
    } else if let Some(name) = flag.strip_prefix("-l") {
      println!("cargo:rustc-link-lib={name}");
    }
  }
  // the wrapper's `.fits.bz2` path calls libbz2 directly
  println!("cargo:rustc-link-lib=bz2");
  let inc = run(
    Command::new("pkg-config").args(["--variable=includedir", "cfitsio"]),
    "pkg-config --variable=includedir cfitsio",
  );
  println!(
    "cargo:include={}",
    String::from_utf8_lossy(&inc.stdout).trim()
  );
}

fn triplet_of(rust_target: &str) -> Option<&'static str> {
  Some(match rust_target {
    "x86_64-pc-windows-msvc" => "x64-windows-static-md",
    "aarch64-pc-windows-msvc" => "arm64-windows-static-md",
    "x86_64-unknown-linux-gnu" => "x64-linux",
    "aarch64-unknown-linux-gnu" => "arm64-linux",
    "x86_64-apple-darwin" => "x64-osx",
    "aarch64-apple-darwin" => "arm64-osx",
    _ => return None,
  })
}

fn windows_host(host: &str) -> bool {
  host.contains("windows")
}

// ───────────────────────────── vcpkg tree ─────────────────────────────

/// `AETHERVK_VCPKG_ROOT`, else `VCPKG_ROOT`, accepted only when its cfitsio port is the pinned version.
fn env_vcpkg_root() -> Option<PathBuf> {
  for var in ["AETHERVK_VCPKG_ROOT", "VCPKG_ROOT"] {
    let Some(root) = env::var_os(var).filter(|v| !v.is_empty()).map(PathBuf::from) else {
      continue;
    };
    let port = root.join("ports").join("cfitsio").join("vcpkg.json");
    let ok = fs::read_to_string(&port)
      .map(|s| port_has_version(&s, CFITSIO_PORT_VERSION))
      .unwrap_or(false);
    if ok {
      return Some(root);
    }
    println!(
      "cargo:warning=aethervk-cfitsio-sys: ignoring {var}={} (no ports/cfitsio at version {CFITSIO_PORT_VERSION}); using the pinned vcpkg {VCPKG_TAG}",
      root.display()
    );
  }
  None
}

fn port_has_version(vcpkg_json: &str, version: &str) -> bool {
  let compact: String = vcpkg_json.chars().filter(|c| !c.is_whitespace()).collect();
  compact.contains(&format!("\"version\":\"{version}\""))
}

/// `<repo>/.vcpkg/vcpkg-<tag>`: download + verify + extract when missing. The extraction goes to a
/// temporary directory and is renamed into place, so the final path only ever holds a complete tree.
fn ensure_vcpkg_tree(vcpkg_dir: &Path) -> PathBuf {
  let root = vcpkg_dir.join(format!("vcpkg-{VCPKG_TAG}"));
  if root.join("bootstrap-vcpkg.sh").is_file() || root.join("bootstrap-vcpkg.bat").is_file() {
    return root;
  }
  let tarball = vcpkg_dir.join(format!("vcpkg-{VCPKG_TAG}.tar.gz"));
  if !tarball_is_good(&tarball) {
    let part = vcpkg_dir.join(format!(
      "vcpkg-{VCPKG_TAG}.tar.gz.part-{}",
      std::process::id()
    ));
    println!("cargo:warning=aethervk-cfitsio-sys: downloading {VCPKG_TARBALL_URL}");
    run(
      Command::new("curl")
        .args(["--retry", "3", "-fsSL", "-o"])
        .arg(&part)
        .arg(VCPKG_TARBALL_URL),
      "curl (vcpkg tarball)",
    );
    let got = sha256_file(&part);
    if got != VCPKG_TARBALL_SHA256 {
      let _ = fs::remove_file(&part);
      panic!(
        "aethervk-cfitsio-sys: vcpkg tarball sha256 mismatch: expected {VCPKG_TARBALL_SHA256}, got {got}; {HINT}"
      );
    }
    if fs::rename(&part, &tarball).is_err() {
      // another process finished first: keep theirs if it verifies, else ours
      if tarball_is_good(&tarball) {
        let _ = fs::remove_file(&part);
      } else {
        let _ = fs::remove_file(&tarball);
        fs::rename(&part, &tarball)
          .unwrap_or_else(|e| panic!("rename {} -> {}: {e}", part.display(), tarball.display()));
      }
    }
  }

  let scratch = vcpkg_dir.join(format!("extract-{}", std::process::id()));
  let _ = fs::remove_dir_all(&scratch);
  fs::create_dir_all(&scratch).unwrap_or_else(|e| panic!("create {}: {e}", scratch.display()));
  run(
    Command::new("tar").arg("-xzf").arg(&tarball).arg("-C").arg(&scratch),
    "tar (vcpkg tarball)",
  );
  let extracted = scratch.join(format!("vcpkg-{VCPKG_TAG}"));
  if !extracted.is_dir() {
    panic!(
      "aethervk-cfitsio-sys: {} did not contain vcpkg-{VCPKG_TAG}/",
      tarball.display()
    );
  }
  // a failed rename onto an existing complete tree means another build won the race
  if let Err(e) = fs::rename(&extracted, &root)
    && !root.is_dir()
  {
    panic!("rename {} -> {}: {e}", extracted.display(), root.display());
  }
  let _ = fs::remove_dir_all(&scratch);
  root
}

fn tarball_is_good(tarball: &Path) -> bool {
  tarball.is_file() && sha256_file(tarball) == VCPKG_TARBALL_SHA256
}

/// Returns the vcpkg executable, running the bootstrap script once (marker file) when it is missing.
fn ensure_bootstrapped(root: &Path, windows_host: bool) -> PathBuf {
  let exe = root.join(if windows_host { "vcpkg.exe" } else { "vcpkg" });
  let marker = root.join(BOOTSTRAP_MARKER);
  if exe.is_file() && marker.is_file() {
    return exe;
  }
  println!(
    "cargo:warning=aethervk-cfitsio-sys: bootstrapping vcpkg in {}",
    root.display()
  );
  if windows_host {
    run(
      Command::new("cmd")
        .arg("/C")
        .arg(root.join("bootstrap-vcpkg.bat"))
        .arg("-disableMetrics")
        .current_dir(root),
      "bootstrap-vcpkg.bat",
    );
  } else {
    run(
      Command::new("sh")
        .arg(root.join("bootstrap-vcpkg.sh"))
        .arg("-disableMetrics")
        .current_dir(root),
      "bootstrap-vcpkg.sh",
    );
  }
  if !exe.is_file() {
    panic!(
      "aethervk-cfitsio-sys: bootstrap finished but {} is missing; {HINT}",
      exe.display()
    );
  }
  fs::write(
    &marker,
    format!("vcpkg {VCPKG_TAG} bootstrapped by aethervk-cfitsio-sys\n"),
  )
  .unwrap_or_else(|e| panic!("write {}: {e}", marker.display()));
  exe
}

// ───────────────────────────── helpers ─────────────────────────────

fn run(cmd: &mut Command, what: &str) -> Output {
  let out = cmd.output().unwrap_or_else(|e| {
    let program = cmd.get_program().to_string_lossy().into_owned();
    panic!("aethervk-cfitsio-sys: cannot start `{program}` for {what}: {e}; {HINT}")
  });
  if !out.status.success() {
    panic!(
      "aethervk-cfitsio-sys: {what} failed ({}) — {}\n--- stdout ---\n{}\n--- stderr ---\n{}\n{HINT}",
      out.status,
      render(cmd),
      String::from_utf8_lossy(&out.stdout),
      String::from_utf8_lossy(&out.stderr)
    );
  }
  out
}

fn render(cmd: &Command) -> String {
  let mut s = cmd.get_program().to_string_lossy().into_owned();
  for a in cmd.get_args() {
    s.push(' ');
    s.push_str(&a.to_string_lossy());
  }
  s
}

/// A directory used as a mutex between concurrent build scripts (`create_dir` is atomic on every
/// platform). A lock older than `LOCK_STALE_AFTER` is treated as abandoned.
struct DirLock(PathBuf);

impl DirLock {
  fn acquire(path: &Path) -> Self {
    let mut warned = false;
    loop {
      match fs::create_dir(path) {
        Ok(()) => {
          let _ = fs::write(path.join("pid"), std::process::id().to_string());
          return DirLock(path.to_path_buf());
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
          let age = fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| SystemTime::now().duration_since(t).ok());
          if age.is_some_and(|a| a > LOCK_STALE_AFTER) {
            println!(
              "cargo:warning=aethervk-cfitsio-sys: removing stale lock {}",
              path.display()
            );
            let _ = fs::remove_dir_all(path);
            continue;
          }
          if !warned {
            println!(
              "cargo:warning=aethervk-cfitsio-sys: waiting for another build holding {}",
              path.display()
            );
            warned = true;
          }
          std::thread::sleep(Duration::from_millis(500));
        }
        Err(e) => panic!(
          "aethervk-cfitsio-sys: cannot create lock {}: {e}",
          path.display()
        ),
      }
    }
  }
}

impl Drop for DirLock {
  fn drop(&mut self) {
    let _ = fs::remove_dir_all(&self.0);
  }
}

fn sha256_file(path: &Path) -> String {
  let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
  let digest = sha256(&bytes);
  let mut hex = String::with_capacity(64);
  for b in digest {
    hex.push_str(&format!("{b:02x}"));
  }
  hex
}

/// FIPS 180-4 SHA-256, std-only, so the pinned tarball hash is checked the same way on every host.
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
