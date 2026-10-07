use super::*;
use std::path::{Path, PathBuf};

/// Fresh, unique scratch directory per test (removed on drop).
struct Scratch(PathBuf);

impl Scratch {
  fn new(tag: &str) -> Self {
    static N: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("avk_asset_lib_{tag}_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    Self(dir)
  }

  fn path(&self, name: &str) -> PathBuf {
    self.0.join(name)
  }

  fn cache(&self) -> String {
    self.0.join("asset_cache").to_str().unwrap().to_string()
  }

  fn library(&self) -> AssetLibrary {
    let mut lib = AssetLibrary::new();
    lib.set_cache_dir(&self.cache()).unwrap();
    lib
  }
}

impl Drop for Scratch {
  fn drop(&mut self) {
    let _ = std::fs::remove_dir_all(&self.0);
  }
}

fn png_bytes(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
  let img = image::RgbaImage::from_fn(w, h, |x, y| image::Rgba(f(x, y)));
  let mut out = std::io::Cursor::new(Vec::new());
  img.write_to(&mut out, image::ImageFormat::Png).unwrap();
  out.into_inner()
}

fn write_png(path: &Path, w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) {
  std::fs::write(path, png_bytes(w, h, f)).unwrap();
}

const TETRA_OBJ: &str = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
vt 0 0
vt 1 0
vt 0 1
f 1/1 3/3 2/2
f 1/1 2/2 4/1
f 1/1 4/1 3/3
f 2/2 3/3 4/1
";

/// Builds a GLB containing one tetrahedron (POSITION + indices only, so normals / UVs /
/// tangents must be synthesised) and one embedded PNG used as baseColorTexture.
fn tetra_glb(image_name: Option<&str>, png: &[u8]) -> Vec<u8> {
  let positions: [[f32; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ];
  let indices: [u16; 12] = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
  let mut bin: Vec<u8> = Vec::new();
  for p in positions {
    for c in p {
      bin.extend_from_slice(&c.to_le_bytes());
    }
  }
  let idx_off = bin.len();
  for i in indices {
    bin.extend_from_slice(&i.to_le_bytes());
  }
  while bin.len() % 4 != 0 {
    bin.push(0);
  }
  let img_off = bin.len();
  bin.extend_from_slice(png);
  while bin.len() % 4 != 0 {
    bin.push(0);
  }

  let name = image_name.map(|n| format!(r#","name":"{n}""#)).unwrap_or_default();
  let json = format!(
    r#"{{"asset":{{"version":"2.0"}},
"buffers":[{{"byteLength":{bl}}}],
"bufferViews":[
 {{"buffer":0,"byteOffset":0,"byteLength":48}},
 {{"buffer":0,"byteOffset":{idx_off},"byteLength":24}},
 {{"buffer":0,"byteOffset":{img_off},"byteLength":{img_len}}}],
"accessors":[
 {{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[0,0,0],"max":[1,1,1]}},
 {{"bufferView":1,"componentType":5123,"count":12,"type":"SCALAR"}}],
"images":[{{"bufferView":2,"mimeType":"image/png"{name}}}],
"textures":[{{"source":0}}],
"materials":[{{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}}}}}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"material":0}}]}}],
"nodes":[{{"mesh":0}}],"scenes":[{{"nodes":[0]}}],"scene":0}}"#,
    bl = bin.len(),
    img_len = png.len(),
  );
  let mut json = json.into_bytes();
  while json.len() % 4 != 0 {
    json.push(b' ');
  }

  let total = 12 + 8 + json.len() + 8 + bin.len();
  let mut glb = Vec::with_capacity(total);
  glb.extend_from_slice(b"glTF");
  glb.extend_from_slice(&2u32.to_le_bytes());
  glb.extend_from_slice(&(total as u32).to_le_bytes());
  glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
  glb.extend_from_slice(b"JSON");
  glb.extend_from_slice(&json);
  glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
  glb.extend_from_slice(b"BIN\0");
  glb.extend_from_slice(&bin);
  glb
}

fn red(_: u32, _: u32) -> [u8; 4] {
  [255, 0, 0, 255]
}

#[test]
fn standalone_png_import_is_mapped_and_thumbnailed() {
  let s = Scratch::new("png");
  let png = s.path("rock.png");
  write_png(&png, 64, 32, red);
  let mut lib = s.library();

  let out = lib.import_file(png.to_str().unwrap()).unwrap();
  assert_eq!(out.mesh, None);
  assert_eq!(out.textures.len(), 1);
  assert_eq!(out.added, out.textures);

  let t = lib.texture(out.textures[0]).unwrap();
  assert_eq!(t.label, "rock");
  assert_eq!((t.width(), t.height()), (64, 32));
  assert_eq!(t.format(), TexelFormat::R8G8B8A8_UNORM);
  assert_eq!(&t.texture().data[..4], &[255, 0, 0, 255]);
  assert_eq!((t.thumbnail.width, t.thumbnail.height), (64, 32));

  // A cache file exists for it.
  let files: Vec<_> = std::fs::read_dir(s.cache()).unwrap().collect();
  assert_eq!(files.len(), 1);
}

#[test]
fn reimport_and_content_duplicates_do_not_create_assets() {
  let s = Scratch::new("dedup");
  let a = s.path("a.png");
  let b = s.path("copy_of_a.png");
  write_png(&a, 8, 8, red);
  std::fs::copy(&a, &b).unwrap();
  let mut lib = s.library();

  let first = lib.import_file(a.to_str().unwrap()).unwrap();
  let again = lib.import_file(a.to_str().unwrap()).unwrap();
  let copy = lib.import_file(b.to_str().unwrap()).unwrap();
  assert_eq!(first.textures, again.textures);
  assert!(again.added.is_empty());
  assert_eq!(first.textures, copy.textures, "same texels → same asset");
  assert!(copy.added.is_empty());
  assert_eq!(lib.assets().len(), 1);

  // Path spelling variants resolve to the same key.
  let dotted = format!("{}/./a.png", s.0.to_str().unwrap());
  assert_eq!(lib.lookup_key(&dotted), Some(first.textures[0]));
}

#[test]
fn glb_import_splits_mesh_and_embedded_texture() {
  let s = Scratch::new("glb");
  let glb = s.path("tetra.glb");
  std::fs::write(
    &glb,
    tetra_glb(Some("surface_color"), &png_bytes(4, 4, red)),
  )
  .unwrap();
  let mut lib = s.library();

  let out = lib.import_file(glb.to_str().unwrap()).unwrap();
  let mesh_id = out.mesh.expect("mesh asset");
  assert_eq!(out.textures.len(), 1);
  assert_eq!(out.added.len(), 2);

  let mesh = lib.mesh(mesh_id).unwrap();
  assert_eq!(mesh.label, "tetra");
  assert_eq!(mesh.vertex_count, 4);
  assert_eq!(mesh.index_count, 12);
  assert_eq!(
    mesh.bundled_textures[TextureChannel::Albedo as usize],
    Some(out.textures[0])
  );
  assert!(mesh.thumbnail.opaque_pixel_count() > 0);

  let tex = lib.texture(out.textures[0]).unwrap();
  assert_eq!(tex.label, "surface_color");
  assert_eq!(tex.channel_hint, Some(TextureChannel::Albedo));
  assert!(tex.key.ends_with("tetra.glb#image0"));

  // Synthesised attributes are sane.
  let comet = lib.materialize_mesh(mesh_id).unwrap();
  assert_eq!(comet.vertices.len(), 4);
  assert!(
    comet.albedo_map.is_none(),
    "materialised meshes carry no textures"
  );
  for v in &comet.vertices {
    let n = v.normal;
    assert!(((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() - 1.0).abs() < 1e-4);
  }

  // Re-import: nothing new.
  let again = lib.import_file(glb.to_str().unwrap()).unwrap();
  assert_eq!(again.mesh, Some(mesh_id));
  assert_eq!(again.textures, out.textures);
  assert!(again.added.is_empty());
}

#[test]
fn unnamed_embedded_image_gets_stem_channel_label() {
  let s = Scratch::new("glb_unnamed");
  let glb = s.path("rock.glb");
  std::fs::write(&glb, tetra_glb(None, &png_bytes(2, 2, red))).unwrap();
  let mut lib = s.library();
  let out = lib.import_file(glb.to_str().unwrap()).unwrap();
  assert_eq!(lib.texture(out.textures[0]).unwrap().label, "rock_albedo");
}

#[test]
fn glb_embedded_texture_dedups_with_identical_standalone_png() {
  let s = Scratch::new("glb_png");
  let bytes = png_bytes(4, 4, red);
  std::fs::write(s.path("t.glb"), tetra_glb(None, &bytes)).unwrap();
  std::fs::write(s.path("same.png"), &bytes).unwrap();
  let mut lib = s.library();
  let a = lib.import_file(s.path("t.glb").to_str().unwrap()).unwrap();
  let b = lib.import_file(s.path("same.png").to_str().unwrap()).unwrap();
  assert_eq!(a.textures, b.textures);
  assert_eq!(lib.assets().len(), 2, "1 mesh + 1 texture");
}

#[test]
fn obj_with_mtl_imports_bundled_textures_once() {
  let s = Scratch::new("obj");
  write_png(&s.path("albedo.png"), 4, 4, red);
  write_png(&s.path("nrm.png"), 4, 4, |_, _| [128, 128, 255, 255]);
  std::fs::write(
    s.path("rock.mtl"),
    "newmtl rock\nmap_Kd albedo.png\nmap_Bump -bm 1.0 nrm.png\n",
  )
  .unwrap();
  std::fs::write(s.path("rock.obj"), format!("mtllib rock.mtl\n{TETRA_OBJ}")).unwrap();
  let mut lib = s.library();

  let out = lib.import_file(s.path("rock.obj").to_str().unwrap()).unwrap();
  let mesh = lib.mesh(out.mesh.unwrap()).unwrap();
  let albedo = mesh.bundled_textures[TextureChannel::Albedo as usize].unwrap();
  let normal = mesh.bundled_textures[TextureChannel::Normal as usize].unwrap();
  assert_ne!(albedo, normal);
  assert_eq!(lib.texture(albedo).unwrap().label, "albedo");
  assert_eq!(lib.texture(normal).unwrap().label, "nrm");

  // Later standalone import of the same file deduplicates by key.
  let again = lib.import_file(s.path("albedo.png").to_str().unwrap()).unwrap();
  assert_eq!(again.textures, vec![albedo]);
  assert!(again.added.is_empty());
  assert_eq!(lib.assets().len(), 3);
}

#[test]
fn obj_with_missing_mtl_still_imports_mesh() {
  let s = Scratch::new("obj_nomtl");
  std::fs::write(s.path("m.obj"), format!("mtllib missing.mtl\n{TETRA_OBJ}")).unwrap();
  let mut lib = s.library();
  let out = lib.import_file(s.path("m.obj").to_str().unwrap()).unwrap();
  assert!(out.mesh.is_some());
  assert!(out.textures.is_empty());
}

#[test]
fn ply_texture_file_comment_is_honoured() {
  let s = Scratch::new("ply");
  write_png(&s.path("skin.png"), 2, 2, red);
  let ply = "ply
format ascii 1.0
comment TextureFile skin.png
element vertex 4
property float x
property float y
property float z
element face 4
property list uchar uint vertex_indices
end_header
0 0 0
1 0 0
0 1 0
0 0 1
3 0 2 1
3 0 1 3
3 0 3 2
3 1 2 3
";
  std::fs::write(s.path("m.ply"), ply).unwrap();
  let mut lib = s.library();
  let out = lib.import_file(s.path("m.ply").to_str().unwrap()).unwrap();
  let mesh = lib.mesh(out.mesh.unwrap()).unwrap();
  assert!(mesh.bundled_textures[TextureChannel::Albedo as usize].is_some());
}

#[test]
fn materialized_mesh_matches_loader_output() {
  let s = Scratch::new("roundtrip");
  std::fs::write(s.path("m.obj"), TETRA_OBJ).unwrap();
  let direct = comet::load_comet_from_obj(s.path("m.obj").to_str().unwrap(), false, None).unwrap();
  let mut lib = s.library();
  let id = lib.import_file(s.path("m.obj").to_str().unwrap()).unwrap().mesh.unwrap();
  let a = lib.materialize_mesh(id).unwrap();
  let b = lib.materialize_mesh(id).unwrap();
  assert_eq!(a.vertices, direct.vertices);
  assert_eq!(a.indices, direct.indices);
  assert_ne!(
    a.id, b.id,
    "every materialisation gets a fresh GPU cache key"
  );
}

#[test]
fn errors_are_reported() {
  let s = Scratch::new("errors");
  let mut no_cache = AssetLibrary::new();
  assert!(matches!(
    no_cache.import_file("/x.png"),
    Err(AssetError::NoCacheDir)
  ));

  let mut lib = s.library();
  std::fs::write(s.path("notes.txt"), "hi").unwrap();
  assert!(matches!(
    lib.import_file(s.path("notes.txt").to_str().unwrap()),
    Err(AssetError::UnsupportedExtension(e)) if e == "txt"
  ));
  assert!(lib.import_file(s.path("missing.png").to_str().unwrap()).is_err());

  write_png(&s.path("big.png"), 16, 16, red);
  lib.set_max_texture_dimension(8);
  assert!(matches!(
    lib.import_file(s.path("big.png").to_str().unwrap()),
    Err(AssetError::TextureTooLarge {
      width: 16,
      height: 16,
      max: 8
    })
  ));
  assert!(lib.assets().is_empty());
}

#[test]
fn clear_unmaps_and_deletes_cache_files() {
  let s = Scratch::new("clear");
  write_png(&s.path("a.png"), 4, 4, red);
  std::fs::write(s.path("m.obj"), TETRA_OBJ).unwrap();
  let mut lib = s.library();
  lib.import_file(s.path("a.png").to_str().unwrap()).unwrap();
  lib.import_file(s.path("m.obj").to_str().unwrap()).unwrap();
  assert_eq!(std::fs::read_dir(s.cache()).unwrap().count(), 2);
  lib.clear();
  assert_eq!(std::fs::read_dir(s.cache()).unwrap().count(), 0);
  assert_eq!(lib.stats(), AssetLibraryStats::default());
}

#[test]
fn cache_file_roundtrips_and_rejects_wrong_kind() {
  let s = Scratch::new("cachefmt");
  let p = s.path("x.avkt");
  let header = CacheHeader {
    kind: AssetKind::Texture,
    format: 37,
    a: 2,
    b: 3,
    payload_len: 5,
    content_hash: 0xdead_beef,
  };
  let bytes = write_and_map(p.to_str().unwrap(), &header, &[&[1, 2], &[3, 4, 5]]).unwrap();
  assert_eq!(&bytes[..], &[1, 2, 3, 4, 5]);
  assert!(map_cache_file(p.to_str().unwrap(), AssetKind::Mesh).is_err());
  assert_eq!(
    &map_cache_file(p.to_str().unwrap(), AssetKind::Texture).unwrap()[..],
    &[1, 2, 3, 4, 5]
  );
}

/// The point of the mmap design: after importing a large texture the library holds almost
/// nothing on the heap, the texels are file-backed.
#[test]
fn large_texture_is_file_backed_not_heap_resident() {
  #[cfg(target_os = "linux")]
  pin_mmap_threshold();
  let s = Scratch::new("memory");
  const N: u32 = 2048;
  write_png(&s.path("big.png"), N, N, |x, y| {
    [(x % 251) as u8, (y % 241) as u8, 7, 255]
  });
  let mut lib = s.library();

  #[cfg(any(target_os = "linux", windows))]
  let anon_before = {
    trim_allocator();
    private_memory_kib()
  };
  let id = lib.import_file(s.path("big.png").to_str().unwrap()).unwrap().textures[0];
  // Freed decode buffers may be retained by the test's allocator (glibc raises its dynamic mmap
  // threshold after large frees); hand them back so only *live* anonymous memory is measured.
  // The shipped cdylib uses jemalloc/mimalloc, which return freed pages on their own.
  #[cfg(any(target_os = "linux", windows))]
  let anon_after = {
    trim_allocator();
    private_memory_kib()
  };

  let decoded = (N * N * 4) as u64;
  let stats = lib.stats();
  assert_eq!(stats.mapped_bytes, decoded + CACHE_PAYLOAD_OFFSET as u64);
  assert!(
    stats.heap_bytes < 128 * 1024,
    "heap {} bytes",
    stats.heap_bytes
  );

  // Texels are readable through the mapping.
  let tex = lib.texture(id).unwrap().texture();
  assert_eq!(tex.data.len() as u64, decoded);
  let px = |x: u32, y: u32| {
    let i = ((y * N + x) * 4) as usize;
    [
      tex.data[i],
      tex.data[i + 1],
      tex.data[i + 2],
      tex.data[i + 3],
    ]
  };
  assert_eq!(px(300, 500), [(300 % 251) as u8, (500 % 241) as u8, 7, 255]);

  // Process-wide private memory is only meaningful when this test owns the process
  // (cargo-nextest runs one test per process).
  #[cfg(any(target_os = "linux", windows))]
  if std::env::var_os("NEXTEST").is_some() {
    if let (Some(b), Some(a)) = (anon_before, anon_after) {
      let grown = a.saturating_sub(b) * 1024;
      if grown >= decoded / 2 {
        // what holds the memory: every large private region, plus the process counters
        dump_large_private_regions();
        eprintln!("[memory] private before {b} KiB, after {a} KiB");
      }
      assert!(
        grown < decoded / 2,
        "private memory grew by {grown} bytes for a {decoded}-byte texture"
      );
    }
  }
}

// ── Process memory probes (Linux: /proc + glibc, Windows: psapi + VirtualQuery) ──────────────
//
// "Private" memory is what the decoded texels would occupy if they stayed on the heap; a mapped
// cache file is not private (Linux: file-backed RSS, Windows: MEM_MAPPED), so it is excluded.

/// glibc raises its mmap threshold dynamically after large frees, after which big buffers are
/// served from (and retained in) arena heaps. Pinning it makes every large buffer an
/// individual mapping released on `free`, so RSS reflects live memory only. (The Windows heap
/// always serves blocks above ~512 KiB with `VirtualAlloc` and releases them on free: nothing to
/// pin there.)
#[cfg(target_os = "linux")]
fn pin_mmap_threshold() {
  #[cfg(target_env = "gnu")]
  unsafe {
    libc::mallopt(libc::M_MMAP_THRESHOLD, 1 << 20);
  }
}

/// Hands freed allocator memory back to the OS.
#[cfg(target_os = "linux")]
fn trim_allocator() {
  #[cfg(target_env = "gnu")]
  unsafe {
    libc::malloc_trim(0);
  }
}

#[cfg(windows)]
fn trim_allocator() {
  use windows::Win32::System::Memory::{GetProcessHeap, HEAP_FLAGS, HeapCompact};
  unsafe {
    if let Ok(heap) = GetProcessHeap() {
      HeapCompact(heap, HEAP_FLAGS(0));
    }
  }
}

/// Private (anonymous) memory of the process in KiB: resident `RssAnon` on Linux, committed
/// `PrivateUsage` on Windows.
#[cfg(target_os = "linux")]
fn private_memory_kib() -> Option<u64> {
  let status = std::fs::read_to_string("/proc/self/status").ok()?;
  status
    .lines()
    .find_map(|l| l.strip_prefix("RssAnon:"))
    .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
}

#[cfg(windows)]
fn private_memory_kib() -> Option<u64> {
  use windows::Win32::System::{
    ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX},
    Threading::GetCurrentProcess,
  };
  let mut c = PROCESS_MEMORY_COUNTERS_EX {
    cb: core::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
    ..Default::default()
  };
  unsafe {
    GetProcessMemoryInfo(
      GetCurrentProcess(),
      (&mut c as *mut PROCESS_MEMORY_COUNTERS_EX).cast::<PROCESS_MEMORY_COUNTERS>(),
      c.cb,
    )
  }
  .ok()?;
  Some(c.PrivateUsage as u64 / 1024)
}

/// Diagnostic: prints every private region above 1 MiB (address range and size).
#[cfg(target_os = "linux")]
fn dump_large_private_regions() {
  let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap_or_default();
  // a mapping header looks like `7f12..-7f34.. rw-p 00000000 00:00 0  [heap]`
  let is_header = |l: &str| {
    let mut it = l.split_whitespace();
    matches!((it.next(), it.next()), (Some(range), Some(perms)) if range.contains('-') && perms.len() == 4)
  };
  let mut head = String::new();
  for line in smaps.lines() {
    if is_header(line) {
      head = line.to_string();
    } else if let Some(v) = line.strip_prefix("Anonymous:") {
      let kib: u64 = v.trim().trim_end_matches("kB").trim().parse().unwrap_or(0);
      if kib > 1024 {
        eprintln!("[memory] {kib} KiB anonymous in {head}");
      }
    }
  }
  let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
  for l in status.lines().filter(|l| l.starts_with("Rss") || l.starts_with("Threads")) {
    eprintln!("[memory] {l}");
  }
}

#[cfg(windows)]
fn dump_large_private_regions() {
  use windows::Win32::System::Memory::{
    MEM_COMMIT, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
  };
  let mut addr = 0usize;
  let mut info = MEMORY_BASIC_INFORMATION::default();
  let size = core::mem::size_of::<MEMORY_BASIC_INFORMATION>();
  // walks the user address space region by region
  while unsafe { VirtualQuery(Some(addr as *const core::ffi::c_void), &mut info, size) } == size {
    if info.State == MEM_COMMIT && info.Type == MEM_PRIVATE && info.RegionSize > 1 << 20 {
      eprintln!(
        "[memory] {} KiB private at {:#x}..{:#x}",
        info.RegionSize / 1024,
        info.BaseAddress as usize,
        info.BaseAddress as usize + info.RegionSize
      );
    }
    let next = info.BaseAddress as usize + info.RegionSize;
    if next <= addr {
      break;
    }
    addr = next;
  }
  if let Some(kib) = private_memory_kib() {
    eprintln!("[memory] PrivateUsage {kib} KiB");
  }
}

#[test]
fn bounding_sphere_contains_all_vertices_and_is_tight() {
  let mut sphere = comet::generate_uv_sphere(2.0, 24, 24, 1.0, false);
  for v in &mut sphere.vertices {
    v.position[0] += 10.0;
    v.position[2] -= 3.0;
  }
  let (c, r) = bounding_sphere(&sphere.vertices);
  assert!(
    (c[0] - 10.0).abs() < 0.1 && c[1].abs() < 0.1 && (c[2] + 3.0).abs() < 0.1,
    "{c:?}"
  );
  assert!(r >= 2.0 - 1e-4 && r < 2.0 * 1.05, "radius {r}");
  for v in &sphere.vertices {
    let d = ((v.position[0] - c[0]).powi(2)
      + (v.position[1] - c[1]).powi(2)
      + (v.position[2] - c[2]).powi(2))
    .sqrt();
    assert!(d <= r + 1e-5);
  }
}

#[test]
fn key_normalisation() {
  assert_eq!(
    normalize_key("/a/b/../c/./d.png"),
    if cfg!(windows) {
      "/a/c/d.png"
    } else {
      "/a/c/d.png"
    }
  );
  assert_eq!(normalize_key("a\\b\\c.png"), "a/b/c.png");
  assert_eq!(normalize_key("../x.png"), "../x.png");
  assert_eq!(file_stem("/dir/name.with.dots.glb"), "name.with.dots");
  assert_eq!(extension_of("/dir/A.PNG"), "png");
  assert_eq!(extension_of("/dir.d/noext"), "");
}

// ---------------------------------------------------------------------------------------------
// Multi-texture GLBs: synthetic (all PBR slots + a shared packed image) and real-world fixtures
// ---------------------------------------------------------------------------------------------

/// GLB with one tetrahedron and `images` embedded as PNG buffer views. `material` is the JSON of
/// the single material, referencing textures `0..images.len()` (texture *i* → image *i*).
fn tetra_glb_with_images(images: &[(Option<&str>, Vec<u8>)], material: &str) -> Vec<u8> {
  let positions: [[f32; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ];
  let indices: [u16; 12] = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
  let mut bin: Vec<u8> = Vec::new();
  for p in positions {
    for c in p {
      bin.extend_from_slice(&c.to_le_bytes());
    }
  }
  let idx_off = bin.len();
  for i in indices {
    bin.extend_from_slice(&i.to_le_bytes());
  }
  let mut views = vec![
    "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":48}".to_string(),
    format!("{{\"buffer\":0,\"byteOffset\":{idx_off},\"byteLength\":24}}"),
  ];
  let mut image_json = Vec::new();
  let mut texture_json = Vec::new();
  for (i, (name, png)) in images.iter().enumerate() {
    while bin.len() % 4 != 0 {
      bin.push(0);
    }
    views.push(format!(
      "{{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{}}}",
      bin.len(),
      png.len()
    ));
    bin.extend_from_slice(png);
    let name = name.map(|n| format!(",\"name\":\"{n}\"")).unwrap_or_default();
    image_json.push(format!(
      "{{\"bufferView\":{},\"mimeType\":\"image/png\"{name}}}",
      views.len() - 1
    ));
    texture_json.push(format!("{{\"source\":{i}}}"));
  }
  while bin.len() % 4 != 0 {
    bin.push(0);
  }
  let json = format!(
    r#"{{"asset":{{"version":"2.0"}},"buffers":[{{"byteLength":{}}}],"bufferViews":[{}],
"accessors":[{{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[0,0,0],"max":[1,1,1]}},
{{"bufferView":1,"componentType":5123,"count":12,"type":"SCALAR"}}],
"images":[{}],"textures":[{}],"materials":[{material}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"material":0}}]}}],
"nodes":[{{"mesh":0}}],"scenes":[{{"nodes":[0]}}],"scene":0}}"#,
    bin.len(),
    views.join(","),
    image_json.join(","),
    texture_json.join(","),
  );
  let mut json = json.into_bytes();
  while json.len() % 4 != 0 {
    json.push(b' ');
  }
  let total = 12 + 8 + json.len() + 8 + bin.len();
  let mut glb = Vec::with_capacity(total);
  glb.extend_from_slice(b"glTF");
  glb.extend_from_slice(&2u32.to_le_bytes());
  glb.extend_from_slice(&(total as u32).to_le_bytes());
  glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
  glb.extend_from_slice(b"JSON");
  glb.extend_from_slice(&json);
  glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
  glb.extend_from_slice(b"BIN\0");
  glb.extend_from_slice(&bin);
  glb
}

fn solid(c: [u8; 4]) -> impl Fn(u32, u32) -> [u8; 4] {
  move |_, _| c
}

/// Asserts the first texel of a texture asset (texels are RGBA8 for PNG/JPEG sources).
fn assert_first_texel(lib: &AssetLibrary, id: AssetId, expected: [u8; 4]) {
  let t = lib.texture(id).unwrap().texture();
  assert_eq!(t.format, TexelFormat::R8G8B8A8_UNORM);
  assert_eq!(&t.data[..4], &expected, "texture {id}");
}

/// All four PBR slots, with the metallic-roughness and occlusion slots sharing one packed (ORM)
/// image, as exporters commonly do: 1 mesh + 3 texture assets, every slot wired to the right one.
#[test]
fn glb_with_all_pbr_slots_and_shared_orm_image() {
  let s = Scratch::new("glb_pbr");
  let images = vec![
    (Some("base"), png_bytes(8, 8, solid([200, 10, 10, 255]))),
    (Some("orm"), png_bytes(8, 8, solid([255, 128, 0, 255]))),
    (Some("nrm"), png_bytes(8, 8, solid([128, 128, 255, 255]))),
  ];
  let material = r#"{"name":"rock","pbrMetallicRoughness":{"baseColorTexture":{"index":0},"metallicRoughnessTexture":{"index":1}},"occlusionTexture":{"index":1},"normalTexture":{"index":2}}"#;
  std::fs::write(s.path("rock.glb"), tetra_glb_with_images(&images, material)).unwrap();
  let mut lib = s.library();

  let out = lib.import_file(s.path("rock.glb").to_str().unwrap()).unwrap();
  let mesh = lib.mesh(out.mesh.expect("mesh asset")).unwrap();
  assert_eq!(out.textures.len(), 3, "shared ORM image imported once");
  assert_eq!(out.added.len(), 4, "1 mesh + 3 textures");

  let b = mesh.bundled_textures;
  let (albedo, normal, rough, ao) = (b[0].unwrap(), b[1].unwrap(), b[2].unwrap(), b[3].unwrap());
  assert_eq!(
    rough, ao,
    "roughness and AO slots reference the same packed image"
  );
  assert_ne!(albedo, normal);
  assert_ne!(albedo, rough);

  assert_eq!(lib.texture(albedo).unwrap().label, "base");
  assert_eq!(lib.texture(normal).unwrap().label, "nrm");
  assert_eq!(lib.texture(rough).unwrap().label, "orm");
  assert_eq!(
    lib.texture(albedo).unwrap().channel_hint,
    Some(TextureChannel::Albedo)
  );
  assert_eq!(
    lib.texture(normal).unwrap().channel_hint,
    Some(TextureChannel::Normal)
  );
  assert_first_texel(&lib, albedo, [200, 10, 10, 255]);
  assert_first_texel(&lib, normal, [128, 128, 255, 255]);
  assert_first_texel(&lib, rough, [255, 128, 0, 255]);

  // previews: mesh silhouette and a solid-colour texture preview per texture
  assert!(mesh.thumbnail.opaque_pixel_count() > 0);
  for id in [albedo, normal, rough] {
    let th = lib.thumbnail(id).unwrap();
    assert_eq!((th.width, th.height), (8, 8));
    assert_eq!(th.rgba[..4], lib.texture(id).unwrap().texture().data[..4]);
  }

  // registry sizes and the cache: one mapped file per asset
  let stats = lib.stats();
  assert_eq!((stats.mesh_count, stats.texture_count), (1, 3));
  assert_eq!(std::fs::read_dir(s.cache()).unwrap().count(), 4);

  // importing again changes nothing
  let again = lib.import_file(s.path("rock.glb").to_str().unwrap()).unwrap();
  assert!(again.added.is_empty());
  assert_eq!(again.mesh, out.mesh);
  assert_eq!(lib.stats(), stats);
  assert_eq!(std::fs::read_dir(s.cache()).unwrap().count(), 4);
}

/// The same image under two names, in two different GLBs: one texture asset; two meshes with
/// identical geometry from different files: one mesh asset (content dedup).
#[test]
fn two_glbs_sharing_geometry_and_texture_produce_no_duplicates() {
  let s = Scratch::new("glb_twice");
  let img = png_bytes(4, 4, solid([1, 2, 3, 255]));
  let mat = r#"{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}"#;
  std::fs::write(
    s.path("a.glb"),
    tetra_glb_with_images(&[(Some("a"), img.clone())], mat),
  )
  .unwrap();
  std::fs::write(
    s.path("b.glb"),
    tetra_glb_with_images(&[(Some("b"), img)], mat),
  )
  .unwrap();
  let mut lib = s.library();
  let a = lib.import_file(s.path("a.glb").to_str().unwrap()).unwrap();
  let b = lib.import_file(s.path("b.glb").to_str().unwrap()).unwrap();
  assert_eq!(a.mesh, b.mesh);
  assert_eq!(a.textures, b.textures);
  assert!(b.added.is_empty());
  assert_eq!(lib.assets().len(), 2);
}

fn repo_test_asset(name: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_assets").join(name)
}

/// Khronos `BoxTextured.glb` (CC-BY-4.0, committed in `test_assets/`): palette PNG base colour,
/// no TANGENT attribute.
#[test]
fn khronos_box_textured_glb() {
  let path = repo_test_asset("BoxTextured.glb");
  assert!(path.is_file(), "missing fixture {}", path.display());
  let s = Scratch::new("box_textured");
  let mut lib = s.library();

  let out = lib.import_file(path.to_str().unwrap()).unwrap();
  let mesh = lib.mesh(out.mesh.unwrap()).unwrap();
  assert_eq!((mesh.vertex_count, mesh.index_count), (24, 36));
  assert_eq!(mesh.label, "BoxTextured");
  assert_eq!(
    mesh.orientation_fix,
    OrientationFix::None,
    "well-formed file left untouched"
  );
  let albedo = mesh.bundled_textures[TextureChannel::Albedo as usize].expect("base colour");
  assert_eq!(mesh.bundled_textures[1..], [None, None, None]);
  assert_eq!(out.textures, vec![albedo]);

  let tex = lib.texture(albedo).unwrap();
  assert_eq!((tex.width(), tex.height()), (256, 256));
  assert_eq!(tex.label, "BoxTextured_albedo");
  assert!(tex.key.ends_with("BoxTextured.glb#image0"));

  // tangents were synthesised (unit xyz, handedness ±1) and the cube is centred
  let comet = lib.materialize_mesh(mesh.id).unwrap();
  for v in &comet.vertices {
    let t = v.tangent;
    assert!(
      ((t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt() - 1.0).abs() < 1e-3,
      "{t:?}"
    );
    assert!(t[3].abs() == 1.0);
  }
  assert!(mesh.bounding_center.iter().all(|c| c.abs() < 1e-3));
  assert!(
    (mesh.bounding_radius - 0.75f32.sqrt()).abs() < 1e-3,
    "{}",
    mesh.bounding_radius
  );

  // previews: the cube silhouette and a non-uniform (logo) texture preview
  assert!(mesh.thumbnail.opaque_pixel_count() > 1000);
  let th = &tex.thumbnail;
  assert_eq!((th.width, th.height), (128, 128));
  let distinct: std::collections::BTreeSet<[u8; 4]> =
    th.rgba.chunks_exact(4).map(|p| [p[0], p[1], p[2], p[3]]).collect();
  assert!(distinct.len() > 8, "logo preview should not be flat");

  // twice: no duplication anywhere
  let stats = lib.stats();
  let again = lib.import_file(path.to_str().unwrap()).unwrap();
  assert!(again.added.is_empty());
  assert_eq!(lib.stats(), stats);
}

const DAMAGED_HELMET_URL: &str = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb";
const DAMAGED_HELMET_SIZE: u64 = 3_773_916;

/// Khronos `DamagedHelmet.glb` (partly CC-BY-NC-4.0, hence never committed). Looked up in
/// `$AVK_TEST_DAMAGED_HELMET`, then in `target/test_downloads/`, else downloaded there.
/// `None` (test skipped) when offline.
fn damaged_helmet() -> Option<PathBuf> {
  let valid = |p: &Path| {
    std::fs::metadata(p).map(|m| m.len() == DAMAGED_HELMET_SIZE).unwrap_or(false)
      && std::fs::read(p).map(|d| d.starts_with(b"glTF")).unwrap_or(false)
  };
  if let Some(p) = std::env::var_os("AVK_TEST_DAMAGED_HELMET").map(PathBuf::from) {
    if valid(&p) {
      return Some(p);
    }
  }
  let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test_downloads");
  let path = dir.join("DamagedHelmet.glb");
  if valid(&path) {
    return Some(path);
  }
  let bytes = match reqwest::blocking::get(DAMAGED_HELMET_URL).and_then(|r| r.error_for_status()) {
    Ok(r) => r.bytes().ok()?,
    Err(e) => {
      println!("Skipping test: could not download DamagedHelmet.glb ({e})");
      return None;
    }
  };
  if bytes.len() as u64 != DAMAGED_HELMET_SIZE {
    println!(
      "Skipping test: unexpected DamagedHelmet.glb size {}",
      bytes.len()
    );
    return None;
  }
  std::fs::create_dir_all(&dir).ok()?;
  let tmp = dir.join(format!("DamagedHelmet.glb.{}", std::process::id()));
  std::fs::write(&tmp, &bytes).ok()?;
  std::fs::rename(&tmp, &path).ok()?;
  Some(path)
}

/// Real-world PBR asset: 5 embedded 2048² JPEGs (base colour, metallic-roughness — progressive
/// JPEG —, emissive, occlusion, normal). The 4 supported channels become 4 texture assets, the
/// emissive map is ignored, every preview is generated.
#[test]
fn khronos_damaged_helmet_glb_imports_all_supported_channels() {
  let Some(path) = damaged_helmet() else {
    return;
  };
  let s = Scratch::new("helmet");
  let mut lib = s.library();

  let out = lib.import_file(path.to_str().unwrap()).unwrap();
  let mesh = lib.mesh(out.mesh.unwrap()).unwrap();
  assert_eq!(mesh.vertex_count, 14_556);
  assert_eq!(mesh.index_count, 46_356);
  assert_eq!(
    out.textures.len(),
    4,
    "emissive texture is not a supported channel"
  );
  assert_eq!(out.added.len(), 5);
  assert_eq!(
    mesh.orientation_fix,
    OrientationFix::None,
    "well-formed file left untouched"
  );

  // material: base 0, MR 1, emissive 2, occlusion 3, normal 4
  let image_of = |c: TextureChannel| {
    let id = mesh.bundled_textures[c as usize].unwrap_or_else(|| panic!("{c:?} not wired"));
    lib.texture(id).unwrap()
  };
  for (channel, image) in [
    (TextureChannel::Albedo, 0),
    (TextureChannel::Roughness, 1),
    (TextureChannel::Ao, 3),
    (TextureChannel::Normal, 4),
  ] {
    let t = image_of(channel);
    assert!(
      t.key.ends_with(&format!("DamagedHelmet.glb#image{image}")),
      "{channel:?}: {}",
      t.key
    );
    assert_eq!(t.channel_hint, Some(channel));
    assert_eq!((t.width(), t.height()), (2048, 2048));
    assert_eq!(
      t.format(),
      TexelFormat::R8G8B8A8_UNORM,
      "3-channel JPEG expanded to RGBA"
    );
    assert_eq!(t.label, format!("DamagedHelmet_{}", channel.suffix()));
    assert_eq!((t.thumbnail.width, t.thumbnail.height), (128, 128));
    assert!(
      t.thumbnail.rgba.chunks_exact(4).any(|p| p[..3] != [0, 0, 0]),
      "{channel:?} preview is black"
    );
  }
  assert!(mesh.thumbnail.opaque_pixel_count() > 3000);

  // memory: 4 × 16 MiB of texels are file-backed, the heap only holds previews + names
  let stats = lib.stats();
  assert!(stats.mapped_bytes >= 4 * 2048 * 2048 * 4);
  assert!(stats.heap_bytes < 512 * 1024, "heap {}", stats.heap_bytes);

  let again = lib.import_file(path.to_str().unwrap()).unwrap();
  assert!(again.added.is_empty());
  assert_eq!(lib.stats(), stats);
}

// ---------------------------------------------------------------------------------------------
// Orientation auto-fix (front faces must point outwards: the mesh pipeline culls back faces)
// ---------------------------------------------------------------------------------------------

/// (fraction of faces whose CCW normal agrees with their vertex normals, signed volume about
/// the bounding-box centre) of a materialised mesh.
fn orientation_stats(comet: &comet::Comet) -> (f64, f64) {
  let mut lo = [f32::MAX; 3];
  let mut hi = [f32::MIN; 3];
  for v in &comet.vertices {
    for k in 0..3 {
      lo[k] = lo[k].min(v.position[k]);
      hi[k] = hi[k].max(v.position[k]);
    }
  }
  let o = [
    (lo[0] + hi[0]) * 0.5,
    (lo[1] + hi[1]) * 0.5,
    (lo[2] + hi[2]) * 0.5,
  ];
  let p = |i: u32| {
    let q = comet.vertices[i as usize].position;
    [
      (q[0] - o[0]) as f64,
      (q[1] - o[1]) as f64,
      (q[2] - o[2]) as f64,
    ]
  };
  let (mut agree, mut total, mut vol) = (0usize, 0usize, 0.0f64);
  for t in comet.indices.chunks_exact(3) {
    let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let f = [
      e1[1] * e2[2] - e1[2] * e2[1],
      e1[2] * e2[0] - e1[0] * e2[2],
      e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let n: [f64; 3] = core::array::from_fn(|k| {
      t.iter().map(|&i| comet.vertices[i as usize].normal[k] as f64).sum()
    });
    let d = f[0] * n[0] + f[1] * n[1] + f[2] * n[2];
    if d != 0.0 {
      total += 1;
      if d > 0.0 {
        agree += 1;
      }
    }
    vol += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
      + a[2] * (b[0] * c[1] - b[1] * c[0]);
  }
  (agree as f64 / total.max(1) as f64, vol / 6.0)
}

/// Tetrahedron with outward per-face normals (`vn`), faces given in `faces` order.
fn tetra_obj_with_normals(faces: &str) -> String {
  format!(
    "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nvn 0 0 -1\nvn 0 -1 0\nvn -1 0 0\nvn 0.57735 0.57735 0.57735\n{faces}"
  )
}

const TETRA_FACES_OUTWARD: &str =
  "f 1//1 3//1 2//1\nf 1//2 2//2 4//2\nf 1//3 4//3 3//3\nf 2//4 3//4 4//4\n";
const TETRA_FACES_REVERSED: &str =
  "f 1//1 2//1 3//1\nf 1//2 4//2 2//2\nf 1//3 3//3 4//3\nf 2//4 4//4 3//4\n";

#[test]
fn correctly_wound_mesh_is_not_modified() {
  let s = Scratch::new("orient_ok");
  std::fs::write(
    s.path("ok.obj"),
    tetra_obj_with_normals(TETRA_FACES_OUTWARD),
  )
  .unwrap();
  let path = s.path("ok.obj");
  let direct = comet::load_comet_from_obj(path.to_str().unwrap(), false, None).unwrap();
  let mut lib = s.library();
  let out = lib.import_file(path.to_str().unwrap()).unwrap();
  assert_eq!(out.orientation_fix, OrientationFix::None);
  let m = lib.materialize_mesh(out.mesh.unwrap()).unwrap();
  assert_eq!(
    m.vertices, direct.vertices,
    "bytes identical to the loader output"
  );
  assert_eq!(m.indices, direct.indices);
}

/// Rule 1: winding opposite to (correct, outward) vertex normals — the Comet2.glb case.
#[test]
fn winding_opposite_to_normals_is_flipped() {
  let s = Scratch::new("orient_rule1");
  std::fs::write(
    s.path("rev.obj"),
    tetra_obj_with_normals(TETRA_FACES_REVERSED),
  )
  .unwrap();
  let mut lib = s.library();
  let out = lib.import_file(s.path("rev.obj").to_str().unwrap()).unwrap();
  assert_eq!(out.orientation_fix, OrientationFix::WindingFlipped);
  let mesh_id = out.mesh.unwrap();
  assert_eq!(
    lib.mesh(mesh_id).unwrap().orientation_fix,
    OrientationFix::WindingFlipped
  );
  assert_eq!(
    lib.assets()[0].orientation_fix,
    OrientationFix::WindingFlipped
  );
  let m = lib.materialize_mesh(mesh_id).unwrap();
  let (agree, vol) = orientation_stats(&m);
  assert_eq!(agree, 1.0);
  assert!(vol > 0.0, "volume {vol}");
  // normals are authoritative: untouched
  assert!(m.vertices.iter().any(|v| v.normal == [0.0, 0.0, -1.0]));
}

/// Rule 2: closed mesh whose winding *and* normals point inwards (normals generated from the
/// inverted winding, as for a GLB without NORMAL).
#[test]
fn closed_inside_out_mesh_is_turned_outward() {
  let s = Scratch::new("orient_rule2");
  let mat = r#"{"pbrMetallicRoughness":{}}"#;
  let mut glb = tetra_glb_with_images(&[], mat);
  // reverse each triangle of the u16 index buffer (bytes 48..72 of the BIN chunk)
  let bin_start = {
    let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    20 + json_len + 8
  };
  for t in 0..4 {
    let o = bin_start + 48 + t * 6;
    let (b, c) = (glb[o + 2..o + 4].to_vec(), glb[o + 4..o + 6].to_vec());
    glb[o + 2..o + 4].copy_from_slice(&c);
    glb[o + 4..o + 6].copy_from_slice(&b);
  }
  std::fs::write(s.path("inside_out.glb"), glb).unwrap();
  let mut lib = s.library();
  let out = lib.import_file(s.path("inside_out.glb").to_str().unwrap()).unwrap();
  assert_eq!(out.orientation_fix, OrientationFix::InsideOutFixed);
  let m = lib.materialize_mesh(out.mesh.unwrap()).unwrap();
  let (agree, vol) = orientation_stats(&m);
  assert_eq!(agree, 1.0);
  assert!(vol > 0.0, "volume {vol}");
}

/// The file the user tested: Blender export whose winding is inverted w.r.t. its normals
/// (0 % agreement, negative volume). After import every face is outward and agrees.
#[test]
fn comet2_glb_winding_is_fixed_at_import() {
  let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/Comet2.glb");
  if !path.is_file() {
    println!("Skipping: {} not found", path.display());
    return;
  }
  let raw = comet::load_comet_from_gltf(path.to_str().unwrap(), false, None).unwrap();
  let (raw_agree, raw_vol) = orientation_stats(&raw);
  assert!(
    raw_agree < 0.01 && raw_vol < 0.0,
    "fixture assumption: ({raw_agree}, {raw_vol})"
  );

  let s = Scratch::new("comet2");
  let mut lib = s.library();
  let out = lib.import_file(path.to_str().unwrap()).unwrap();
  assert!(
    out.textures.is_empty(),
    "Comet2.glb has no images / materials"
  );
  assert_eq!(out.orientation_fix, OrientationFix::WindingFlipped);
  let m = lib.materialize_mesh(out.mesh.unwrap()).unwrap();
  let (agree, vol) = orientation_stats(&m);
  assert!(agree > 0.99, "agreement {agree}");
  assert!(vol > 0.0, "volume {vol}");
}

// ---------------------------------------------------------------------------------------------
// Unloading
// ---------------------------------------------------------------------------------------------

#[test]
fn removing_a_mesh_keeps_its_textures_and_frees_its_key() {
  let s = Scratch::new("remove_mesh");
  let images = vec![(Some("base"), png_bytes(4, 4, solid([9, 9, 9, 255])))];
  let mat = r#"{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}"#;
  std::fs::write(s.path("m.glb"), tetra_glb_with_images(&images, mat)).unwrap();
  let path = s.path("m.glb");
  let mut lib = s.library();
  let first = lib.import_file(path.to_str().unwrap()).unwrap();
  let (mesh, tex) = (first.mesh.unwrap(), first.textures[0]);
  assert_eq!(std::fs::read_dir(s.cache()).unwrap().count(), 2);

  assert_eq!(lib.remove(mesh).unwrap(), AssetKind::Mesh);
  assert!(lib.mesh(mesh).is_none());
  assert!(
    lib.texture(tex).is_some(),
    "bundled textures are independent assets"
  );
  assert_eq!(lib.lookup_key(path.to_str().unwrap()), None);
  assert_eq!(
    std::fs::read_dir(s.cache()).unwrap().count(),
    1,
    "mesh cache file deleted"
  );
  assert_eq!((lib.stats().mesh_count, lib.stats().texture_count), (0, 1));

  // the source can be imported again: new mesh, the texture is reused
  let again = lib.import_file(path.to_str().unwrap()).unwrap();
  let new_mesh = again.mesh.unwrap();
  assert_ne!(new_mesh, mesh);
  assert_eq!(again.added, vec![new_mesh]);
  assert_eq!(again.textures, vec![tex]);
  assert_eq!(lib.mesh(new_mesh).unwrap().bundled_textures[0], Some(tex));
}

#[test]
fn removing_a_texture_unbundles_it_and_forgets_its_aliases() {
  let s = Scratch::new("remove_tex");
  let bytes = png_bytes(4, 4, solid([1, 2, 3, 255]));
  let mat = r#"{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}"#;
  std::fs::write(
    s.path("m.glb"),
    tetra_glb_with_images(&[(None, bytes.clone())], mat),
  )
  .unwrap();
  std::fs::write(s.path("same.png"), &bytes).unwrap();
  let mut lib = s.library();
  let out = lib.import_file(s.path("m.glb").to_str().unwrap()).unwrap();
  let tex = out.textures[0];
  // alias: same texels under another key
  assert_eq!(
    lib.import_file(s.path("same.png").to_str().unwrap()).unwrap().textures,
    vec![tex]
  );

  assert_eq!(lib.remove(tex).unwrap(), AssetKind::Texture);
  assert_eq!(
    lib.mesh(out.mesh.unwrap()).unwrap().bundled_textures,
    [None; 4]
  );
  assert_eq!(lib.lookup_key(s.path("same.png").to_str().unwrap()), None);
  let re = lib.import_file(s.path("same.png").to_str().unwrap()).unwrap();
  assert_ne!(re.textures[0], tex);
  assert_eq!(re.added, re.textures);

  assert!(matches!(lib.remove(tex), Err(AssetError::UnknownAsset)));
}
