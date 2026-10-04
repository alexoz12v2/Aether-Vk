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

  #[cfg(target_os = "linux")]
  let anon_before = {
    trim_allocator();
    rss_anon_kib()
  };
  let id = lib.import_file(s.path("big.png").to_str().unwrap()).unwrap().textures[0];
  // Freed decode buffers may be retained by the test's allocator (glibc raises its dynamic mmap
  // threshold after large frees); hand them back so only *live* anonymous memory is measured.
  // The shipped cdylib uses jemalloc/mimalloc, which return freed pages on their own.
  #[cfg(target_os = "linux")]
  let anon_after = {
    trim_allocator();
    rss_anon_kib()
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

  // Process-wide anonymous RSS is only meaningful when this test owns the process
  // (cargo-nextest runs one test per process).
  #[cfg(target_os = "linux")]
  if std::env::var_os("NEXTEST").is_some() {
    if let (Some(b), Some(a)) = (anon_before, anon_after) {
      let grown = a.saturating_sub(b) * 1024;
      assert!(
        grown < decoded / 2,
        "anonymous RSS grew by {grown} bytes for a {decoded}-byte texture"
      );
    }
  }
}

/// glibc raises its mmap threshold dynamically after large frees, after which big buffers are
/// served from (and retained in) arena heaps. Pinning it makes every large buffer an
/// individual mapping released on `free`, so RSS reflects live memory only.
#[cfg(target_os = "linux")]
fn pin_mmap_threshold() {
  #[cfg(target_env = "gnu")]
  unsafe {
    libc::mallopt(libc::M_MMAP_THRESHOLD, 1 << 20);
  }
}

#[cfg(target_os = "linux")]
fn trim_allocator() {
  #[cfg(target_env = "gnu")]
  unsafe {
    libc::malloc_trim(0);
  }
}

#[cfg(target_os = "linux")]
fn rss_anon_kib() -> Option<u64> {
  let status = std::fs::read_to_string("/proc/self/status").ok()?;
  status
    .lines()
    .find_map(|l| l.strip_prefix("RssAnon:"))
    .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
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

