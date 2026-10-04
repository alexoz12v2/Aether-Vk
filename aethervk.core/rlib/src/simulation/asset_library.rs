//! Imported mesh / texture assets (Imports tab) backed by memory-mapped cache files.
//!
//! # Memory residency
//!
//! Decoded asset data never stays on the heap. On import every texture is decoded once, written
//! to `<cache_dir>/<content hash>.avkt` (meshes: `.avkm`) and the heap buffer is dropped; the
//! library then keeps a **read-only file mapping** of that cache file. Mapped pages are backed
//! by the file itself rather than by swap / the commit charge, so the OS can drop them under
//! memory pressure and fault them back in on access (Linux `PROT_READ`+`MAP_PRIVATE`, Windows
//! `PAGE_READONLY` views, macOS likewise). Peak heap usage during an import is therefore about
//! one decoded texture.
//!
//! [`Texture::data`] is a `bytes::Bytes`, so a mapped texture is handed to the renderer as
//! `Bytes::from_owner(mapping).slice(payload)` — zero copies and no type changes downstream.
//! Only the mesh that is currently wired to the comet is copied into a heap `Comet`
//! ([`AssetLibrary::materialize_mesh`]).
//!
//! We only ever map our *own* cache copies (never user files), so nobody can truncate a mapping
//! underneath us (SIGBUS / `EXCEPTION_IN_PAGE_ERROR`). Windows refuses to delete a mapped file,
//! hence [`AssetLibrary::clear`] drops all mappings before removing cache files; stale session
//! folders are swept by the .NET `LocalStorageService`.
//!
//! # Deduplication
//!
//! - every asset has a *key*: the normalised path of its source file, or
//!   `<normalised glTF path>#image<N>` for images embedded in a glTF/GLB;
//! - importing a key twice returns the existing asset;
//! - textures (and meshes) are additionally deduplicated by content, so the same PNG reached via
//!   an OBJ material and via a standalone import produces a single asset.

use alloc::{
  collections::BTreeMap,
  format,
  string::{String, ToString},
  vec::Vec,
};

use aethervk_oshal_rlib::{
  self as oshal,
  os::{files::MappedFile, fs},
};

use crate::{
  simulation::{
    comet::{
      self, Comet, CometLoadError, EncodedImageKind, GltfTextureSlot, TexelFormat, Texture, Vertex,
    },
    thumbnail::{self, THUMBNAIL_SIZE, Thumbnail},
  },
  types::EngineError,
};

/// Identifier of an imported asset. Unique across meshes and textures, never reused.
pub type AssetId = u64;

/// Kind of an imported asset.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetKind {
  Mesh = 1,
  Texture = 2,
}

/// Texture channels supported by `physical_mesh2.frag` (bindings 0..=3).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureChannel {
  Albedo = 0,
  Normal = 1,
  Roughness = 2,
  Ao = 3,
}

impl TextureChannel {
  pub const ALL: [TextureChannel; 4] = [
    TextureChannel::Albedo,
    TextureChannel::Normal,
    TextureChannel::Roughness,
    TextureChannel::Ao,
  ];

  pub fn from_u32(v: u32) -> Option<Self> {
    match v {
      0 => Some(Self::Albedo),
      1 => Some(Self::Normal),
      2 => Some(Self::Roughness),
      3 => Some(Self::Ao),
      _ => None,
    }
  }

  fn suffix(self) -> &'static str {
    match self {
      Self::Albedo => "albedo",
      Self::Normal => "normal",
      Self::Roughness => "roughness",
      Self::Ao => "ao",
    }
  }
}

impl From<GltfTextureSlot> for TextureChannel {
  fn from(slot: GltfTextureSlot) -> Self {
    match slot {
      GltfTextureSlot::BaseColor => Self::Albedo,
      GltfTextureSlot::Normal => Self::Normal,
      GltfTextureSlot::MetallicRoughness => Self::Roughness,
      GltfTextureSlot::Occlusion => Self::Ao,
    }
  }
}

/// Errors produced while importing an asset.
#[derive(Debug)]
pub enum AssetError {
  /// The source could not be parsed / decoded.
  Load(CometLoadError),
  /// The file extension is not one of obj, ply, gltf, glb, png, jpg, jpeg, ktx2.
  UnsupportedExtension(String),
  /// Texture larger than the device's `maxImageDimension2D`.
  TextureTooLarge { width: u32, height: u32, max: u32 },
  /// Texture payload size does not match its declared format / extent.
  MalformedTexture,
  /// [`AssetLibrary::set_cache_dir`] was never called.
  NoCacheDir,
  /// Cache file could not be written or mapped.
  CacheIo,
  /// The mesh has no triangles.
  EmptyMesh,
}

impl From<CometLoadError> for AssetError {
  fn from(e: CometLoadError) -> Self {
    Self::Load(e)
  }
}

impl AssetError {
  /// Human readable description (surfaced to the UI through the import task result).
  pub fn message(&self) -> String {
    match self {
      AssetError::Load(e) => format!("could not load file: {e:?}"),
      AssetError::UnsupportedExtension(ext) => format!("unsupported file type '.{ext}'"),
      AssetError::TextureTooLarge { width, height, max } => {
        format!("texture {width}x{height} exceeds the device limit of {max}")
      }
      AssetError::MalformedTexture => "texture data does not match its format".to_string(),
      AssetError::NoCacheDir => "asset cache directory not configured".to_string(),
      AssetError::CacheIo => "could not write or map the asset cache file".to_string(),
      AssetError::EmptyMesh => "mesh has no triangles".to_string(),
    }
  }
}

impl From<AssetError> for EngineError {
  fn from(value: AssetError) -> Self {
    match value {
      AssetError::Load(e) => EngineError::from(e),
      AssetError::UnsupportedExtension(_) => {
        EngineError::InvalidOperation("AssetError::UnsupportedExtension")
      }
      AssetError::TextureTooLarge { .. } => {
        EngineError::InvalidOperation("AssetError::TextureTooLarge")
      }
      AssetError::MalformedTexture => EngineError::InvalidOperation("AssetError::MalformedTexture"),
      AssetError::NoCacheDir => EngineError::InvalidOperation("AssetError::NoCacheDir"),
      AssetError::CacheIo => EngineError::InvalidOperation("AssetError::CacheIo"),
      AssetError::EmptyMesh => EngineError::InvalidOperation("AssetError::EmptyMesh"),
    }
  }
}

/// An imported mesh. Geometry lives in the mapped cache file only.
pub struct MeshAsset {
  pub id: AssetId,
  pub key: String,
  pub label: String,
  pub source_path: String,
  pub vertex_count: u64,
  pub index_count: u64,
  /// Bounding sphere (Ritter) in mesh space; used to recenter + rescale to the nucleus radius.
  pub bounding_center: [f32; 3],
  pub bounding_radius: f32,
  /// Textures that came bundled with the mesh (glTF material / OBJ mtl), per channel.
  pub bundled_textures: [Option<AssetId>; 4],
  pub thumbnail: Thumbnail,
  content_hash: u64,
  /// `Vertex[vertex_count]` followed by `u32[index_count]`.
  payload: bytes::Bytes,
}

/// An imported texture. Texels live in the mapped cache file only.
pub struct TextureAsset {
  pub id: AssetId,
  pub key: String,
  pub label: String,
  pub source_path: String,
  /// Channel suggested by the source (glTF material slot / mtl map), if any.
  pub channel_hint: Option<TextureChannel>,
  pub thumbnail: Thumbnail,
  content_hash: u64,
  /// `data` is a slice of the mapped cache file.
  texture: Texture,
}

impl TextureAsset {
  /// Zero-copy handle on the mapped texels (cloning only bumps a refcount).
  pub fn texture(&self) -> Texture {
    self.texture.clone()
  }

  pub fn width(&self) -> u32 {
    self.texture.width
  }

  pub fn height(&self) -> u32 {
    self.texture.height
  }

  pub fn format(&self) -> TexelFormat {
    self.texture.format
  }
}

/// Flat description of an asset (FFI / UI listing).
#[derive(Debug, Clone, PartialEq)]
pub struct AssetInfo {
  pub id: AssetId,
  pub kind: AssetKind,
  pub key: String,
  pub label: String,
  pub source_path: String,
  /// Texture: width/height. Mesh: vertex/index count.
  pub a: u64,
  pub b: u64,
  pub channel_hint: Option<TextureChannel>,
  pub bundled_textures: [Option<AssetId>; 4],
}

/// Result of a single [`AssetLibrary::import_file`] call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImportOutcome {
  pub mesh: Option<AssetId>,
  /// Every texture the file resolved to (bundled or standalone), new or pre-existing.
  pub textures: Vec<AssetId>,
  /// Subset of the above ids that did not exist before this import.
  pub added: Vec<AssetId>,
}

/// Memory accounting of the library itself (not of the GPU copies).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AssetLibraryStats {
  /// Bytes of cache files currently mapped (file-backed, reclaimable).
  pub mapped_bytes: u64,
  /// Heap bytes held by the library (thumbnails + strings), i.e. anonymous memory.
  pub heap_bytes: u64,
  pub mesh_count: u32,
  pub texture_count: u32,
}

/// Default upper bound for texture extents until the device reports `maxImageDimension2D`.
pub const DEFAULT_MAX_TEXTURE_DIMENSION: u32 = 16384;

/// See the module documentation.
pub struct AssetLibrary {
  cache_dir: Option<String>,
  max_texture_dimension: u32,
  next_id: AssetId,
  meshes: BTreeMap<AssetId, MeshAsset>,
  textures: BTreeMap<AssetId, TextureAsset>,
  by_key: BTreeMap<String, AssetId>,
  by_hash: BTreeMap<u64, Vec<AssetId>>,
  /// Cache files written by this library (deleted on [`Self::clear`]).
  cache_files: Vec<String>,
}

impl Default for AssetLibrary {
  fn default() -> Self {
    Self::new()
  }
}

impl AssetLibrary {
  pub fn new() -> Self {
    Self {
      cache_dir: None,
      max_texture_dimension: DEFAULT_MAX_TEXTURE_DIMENSION,
      next_id: 1,
      meshes: BTreeMap::new(),
      textures: BTreeMap::new(),
      by_key: BTreeMap::new(),
      by_hash: BTreeMap::new(),
      cache_files: Vec::new(),
    }
  }

  /// Directory receiving the decoded cache files (created if missing). Typically
  /// `<.NET session dir>/asset_cache`, so the files share the session's lifetime.
  pub fn set_cache_dir(&mut self, dir: &str) -> Result<(), AssetError> {
    fs::create_dir_all(dir).map_err(|_| AssetError::CacheIo)?;
    self.cache_dir = Some(dir.trim_end_matches(['/', '\\']).to_string());
    Ok(())
  }

  pub fn cache_dir(&self) -> Option<&str> {
    self.cache_dir.as_deref()
  }

  pub fn set_max_texture_dimension(&mut self, max: u32) {
    self.max_texture_dimension = max.max(1);
  }

  pub fn mesh(&self, id: AssetId) -> Option<&MeshAsset> {
    self.meshes.get(&id)
  }

  pub fn texture(&self, id: AssetId) -> Option<&TextureAsset> {
    self.textures.get(&id)
  }

  pub fn kind_of(&self, id: AssetId) -> Option<AssetKind> {
    if self.meshes.contains_key(&id) {
      Some(AssetKind::Mesh)
    } else if self.textures.contains_key(&id) {
      Some(AssetKind::Texture)
    } else {
      None
    }
  }

  pub fn thumbnail(&self, id: AssetId) -> Option<&Thumbnail> {
    self
      .meshes
      .get(&id)
      .map(|m| &m.thumbnail)
      .or_else(|| self.textures.get(&id).map(|t| &t.thumbnail))
  }

  pub fn lookup_key(&self, key: &str) -> Option<AssetId> {
    self.by_key.get(&normalize_key(key)).copied()
  }

  /// All assets, meshes first, each group in import order.
  pub fn assets(&self) -> Vec<AssetInfo> {
    let meshes = self.meshes.values().map(|m| AssetInfo {
      id: m.id,
      kind: AssetKind::Mesh,
      key: m.key.clone(),
      label: m.label.clone(),
      source_path: m.source_path.clone(),
      a: m.vertex_count,
      b: m.index_count,
      channel_hint: None,
      bundled_textures: m.bundled_textures,
    });
    let textures = self.textures.values().map(|t| AssetInfo {
      id: t.id,
      kind: AssetKind::Texture,
      key: t.key.clone(),
      label: t.label.clone(),
      source_path: t.source_path.clone(),
      a: t.texture.width as u64,
      b: t.texture.height as u64,
      channel_hint: t.channel_hint,
      bundled_textures: [None; 4],
    });
    meshes.chain(textures).collect()
  }

  pub fn stats(&self) -> AssetLibraryStats {
    let mut s = AssetLibraryStats {
      mesh_count: self.meshes.len() as u32,
      texture_count: self.textures.len() as u32,
      ..Default::default()
    };
    for m in self.meshes.values() {
      s.mapped_bytes += (CACHE_PAYLOAD_OFFSET + m.payload.len()) as u64;
      s.heap_bytes +=
        (m.thumbnail.rgba.len() + m.key.len() + m.label.len() + m.source_path.len()) as u64;
    }
    for t in self.textures.values() {
      s.mapped_bytes += (CACHE_PAYLOAD_OFFSET + t.texture.data.len()) as u64;
      s.heap_bytes +=
        (t.thumbnail.rgba.len() + t.key.len() + t.label.len() + t.source_path.len()) as u64;
    }
    for k in self.by_key.keys() {
      s.heap_bytes += k.len() as u64;
    }
    s
  }

  /// Copies the mapped geometry of mesh `id` into a fresh heap [`Comet`] (new `Comet::id`, no
  /// texture maps). Only the mesh currently displayed should be materialised.
  pub fn materialize_mesh(&self, id: AssetId) -> Option<Comet> {
    let m = self.meshes.get(&id)?;
    let vbytes = m.vertex_count as usize * core::mem::size_of::<Vertex>();
    let (vpart, ipart) = m.payload.split_at(vbytes);
    let vertices: Vec<Vertex> = match bytemuck::try_cast_slice::<u8, Vertex>(vpart) {
      Ok(v) => v.to_vec(),
      Err(_) => vpart
        .chunks_exact(core::mem::size_of::<Vertex>())
        .map(bytemuck::pod_read_unaligned)
        .collect(),
    };
    let indices: Vec<u32> = ipart
      .chunks_exact(4)
      .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
      .collect();
    Some(Comet {
      id: comet::next_comet_id(),
      vertices,
      indices,
      albedo_map: None,
      normal_map: None,
      roughness_map: None,
      ao_map: None,
    })
  }

  /// Imports a mesh (obj, ply, gltf, glb) or a texture (png, jpg, jpeg, ktx2).
  ///
  /// Meshes are split from their textures: bundled images become separate texture assets,
  /// recorded in [`MeshAsset::bundled_textures`]. Re-importing a known key is a no-op that
  /// returns the existing ids.
  pub fn import_file(&mut self, path: &str) -> Result<ImportOutcome, AssetError> {
    if self.cache_dir.is_none() {
      return Err(AssetError::NoCacheDir);
    }
    let key = normalize_key(path);
    let ext = extension_of(path);

    if let Some(&id) = self.by_key.get(&key) {
      let mut outcome = ImportOutcome::default();
      if let Some(m) = self.meshes.get(&id) {
        outcome.mesh = Some(id);
        outcome.textures = m.bundled_textures.iter().flatten().copied().collect();
      } else {
        outcome.textures.push(id);
      }
      return Ok(outcome);
    }

    match ext.as_str() {
      "obj" | "ply" | "gltf" | "glb" => self.import_mesh_file(path, &key, &ext),
      "png" | "jpg" | "jpeg" | "ktx2" => {
        let mut outcome = ImportOutcome::default();
        let (id, added) = self.import_texture_file(path, None)?;
        outcome.textures.push(id);
        if added {
          outcome.added.push(id);
        }
        Ok(outcome)
      }
      _ => Err(AssetError::UnsupportedExtension(ext)),
    }
  }

  /// Drops every asset, unmaps the cache files and deletes them (best effort: a file still
  /// mapped through a `Texture` clone held elsewhere cannot be deleted on Windows).
  pub fn clear(&mut self) {
    self.meshes.clear();
    self.textures.clear();
    self.by_key.clear();
    self.by_hash.clear();
    for f in self.cache_files.drain(..) {
      let _ = fs::remove_file(f.as_str());
    }
  }

  // ------------------------------------------------------------------------------------------

  fn alloc_id(&mut self) -> AssetId {
    let id = self.next_id;
    self.next_id += 1;
    id
  }

  fn cache_path(&self, hash: u64, ext: &str) -> Result<String, AssetError> {
    let dir = self.cache_dir.as_ref().ok_or(AssetError::NoCacheDir)?;
    Ok(format!("{dir}/{hash:016x}.{ext}"))
  }

  fn import_mesh_file(
    &mut self,
    path: &str,
    key: &str,
    ext: &str,
  ) -> Result<ImportOutcome, AssetError> {
    let mut outcome = ImportOutcome::default();
    let mut bundled: [Option<AssetId>; 4] = [None; 4];
    let mut note_texture = |outcome: &mut ImportOutcome, channel: TextureChannel, id, added| {
      if bundled[channel as usize].is_none() {
        bundled[channel as usize] = Some(id);
      }
      if !outcome.textures.contains(&id) {
        outcome.textures.push(id);
      }
      if added && !outcome.added.contains(&id) {
        outcome.added.push(id);
      }
    };

    let (vertices, indices) = match ext {
      "gltf" | "glb" => {
        let file = comet::GltfFile::open(path)?;
        let geometry = comet::read_gltf_geometry(&file)?;
        let stem = file_stem(path);
        for (slot, image_index) in &geometry.texture_refs {
          let channel = TextureChannel::from(*slot);
          let image = file.document.images().nth(*image_index);
          // External image files are keyed by their own path so that a later standalone import
          // of the same file deduplicates by key.
          let uri = image.as_ref().and_then(|img| match img.source() {
            gltf::image::Source::Uri { uri, .. } if !uri.starts_with("data:") => Some(uri),
            _ => None,
          });
          let (tex_key, label, source) = match uri {
            Some(uri) => {
              let p = join_path(&parent_dir(path), uri);
              (normalize_key(&p), file_stem(&p), p)
            }
            None => {
              let label = image
                .as_ref()
                .and_then(|i| i.name())
                .filter(|n| !n.is_empty())
                .map(|n| n.to_string())
                .unwrap_or_else(|| format!("{stem}_{}", channel.suffix()));
              (format!("{key}#image{image_index}"), label, path.to_string())
            }
          };
          if let Some(&id) = self.by_key.get(&tex_key) {
            note_texture(&mut outcome, channel, id, false);
            continue;
          }
          let Some(tex) = comet::decode_gltf_image(&file, *image_index)? else {
            continue;
          };
          let (id, added) = self.insert_texture(tex, tex_key, label, source, Some(channel))?;
          note_texture(&mut outcome, channel, id, added);
        }
        (geometry.vertices, geometry.indices)
      }
      "obj" => {
        let mesh = comet::load_comet_from_obj(path, false, None)?;
        for (channel, tex_path) in obj_material_textures(path) {
          match self.import_texture_file(&tex_path, Some(channel)) {
            Ok((id, added)) => note_texture(&mut outcome, channel, id, added),
            Err(e) => oshal::log!("OBJ material texture '{}' skipped: {:?}", tex_path, e),
          }
        }
        (mesh.vertices, mesh.indices)
      }
      _ => {
        let mesh = comet::load_comet_from_ply(path, false, None)?;
        if let Some(tex_path) = ply_texture_file(path) {
          match self.import_texture_file(&tex_path, Some(TextureChannel::Albedo)) {
            Ok((id, added)) => note_texture(&mut outcome, TextureChannel::Albedo, id, added),
            Err(e) => oshal::log!("PLY TextureFile '{}' skipped: {:?}", tex_path, e),
          }
        }
        (mesh.vertices, mesh.indices)
      }
    };

    if vertices.is_empty() || indices.len() < 3 {
      return Err(AssetError::EmptyMesh);
    }

    let (bounding_center, bounding_radius) = bounding_sphere(&vertices);
    let thumbnail = thumbnail::mesh_thumbnail(
      &vertices,
      &indices,
      bounding_center,
      bounding_radius,
      THUMBNAIL_SIZE,
    );

    let vbytes: &[u8] = bytemuck::cast_slice(&vertices);
    let ibytes: &[u8] = bytemuck::cast_slice(&indices);
    let hash = content_hash(
      &[vbytes, ibytes],
      [vertices.len() as u64, indices.len() as u64, 0],
    );

    // Content dedup: the same geometry under another path is the same asset.
    if let Some(existing) = self.find_by_hash(hash, |lib, id| {
      lib
        .meshes
        .get(&id)
        .map(|m| {
          m.payload.len() == vbytes.len() + ibytes.len()
            && &m.payload[..vbytes.len()] == vbytes
            && &m.payload[vbytes.len()..] == ibytes
        })
        .unwrap_or(false)
    }) {
      self.by_key.insert(key.to_string(), existing);
      outcome.mesh = Some(existing);
      return Ok(outcome);
    }

    let header = CacheHeader {
      kind: AssetKind::Mesh,
      format: 0,
      a: vertices.len() as u64,
      b: indices.len() as u64,
      payload_len: (vbytes.len() + ibytes.len()) as u64,
      content_hash: hash,
    };
    let cache_path = self.cache_path(hash, "avkm")?;
    let payload = write_and_map(&cache_path, &header, &[vbytes, ibytes])?;
    self.cache_files.push(cache_path);
    let vertex_count = vertices.len() as u64;
    let index_count = indices.len() as u64;
    drop(vertices);
    drop(indices);

    let id = self.alloc_id();
    self.meshes.insert(
      id,
      MeshAsset {
        id,
        key: key.to_string(),
        label: file_stem(path),
        source_path: path.to_string(),
        vertex_count,
        index_count,
        bounding_center,
        bounding_radius,
        bundled_textures: bundled,
        thumbnail,
        content_hash: hash,
        payload,
      },
    );
    self.by_key.insert(key.to_string(), id);
    self.by_hash.entry(hash).or_default().push(id);
    outcome.mesh = Some(id);
    outcome.added.insert(0, id);
    Ok(outcome)
  }

  /// Returns `(id, newly_added)`.
  fn import_texture_file(
    &mut self,
    path: &str,
    hint: Option<TextureChannel>,
  ) -> Result<(AssetId, bool), AssetError> {
    let key = normalize_key(path);
    if let Some(&id) = self.by_key.get(&key) {
      return Ok((id, false));
    }
    let kind = EncodedImageKind::from_path(path)
      .ok_or_else(|| AssetError::UnsupportedExtension(extension_of(path)))?;
    let encoded =
      MappedFile::new(path).map_err(|_| AssetError::Load(CometLoadError::TextureNotFound))?;
    let tex = comet::decode_encoded_image(&encoded, kind)?;
    drop(encoded);
    self.insert_texture(tex, key, file_stem(path), path.to_string(), hint)
  }

  /// Validates, thumbnails, caches + maps a decoded texture. Consumes (and frees) `tex`.
  fn insert_texture(
    &mut self,
    tex: Texture,
    key: String,
    label: String,
    source_path: String,
    hint: Option<TextureChannel>,
  ) -> Result<(AssetId, bool), AssetError> {
    if tex.width > self.max_texture_dimension || tex.height > self.max_texture_dimension {
      return Err(AssetError::TextureTooLarge {
        width: tex.width,
        height: tex.height,
        max: self.max_texture_dimension,
      });
    }
    if let Some(bpp) = bytes_per_texel(tex.format) {
      if tex.width == 0
        || tex.height == 0
        || tex.data.len() != tex.width as usize * tex.height as usize * bpp
      {
        return Err(AssetError::MalformedTexture);
      }
    }

    let format_raw = tex.format.to_vk_format().as_raw() as u32;
    let hash = content_hash(
      &[&tex.data[..]],
      [tex.width as u64, tex.height as u64, format_raw as u64],
    );
    if let Some(existing) = self.find_by_hash(hash, |lib, id| {
      lib
        .textures
        .get(&id)
        .map(|t| {
          t.texture.width == tex.width
            && t.texture.height == tex.height
            && t.texture.format == tex.format
            && t.texture.data[..] == tex.data[..]
        })
        .unwrap_or(false)
    }) {
      self.by_key.insert(key, existing);
      return Ok((existing, false));
    }

    let thumbnail = thumbnail::texture_thumbnail(&tex, THUMBNAIL_SIZE);
    let header = CacheHeader {
      kind: AssetKind::Texture,
      format: format_raw,
      a: tex.width as u64,
      b: tex.height as u64,
      payload_len: tex.data.len() as u64,
      content_hash: hash,
    };
    let cache_path = self.cache_path(hash, "avkt")?;
    let data = write_and_map(&cache_path, &header, &[&tex.data[..]])?;
    self.cache_files.push(cache_path);
    let mapped = Texture { data, ..tex };

    let id = self.alloc_id();
    self.textures.insert(
      id,
      TextureAsset {
        id,
        key: key.clone(),
        label,
        source_path,
        channel_hint: hint,
        thumbnail,
        content_hash: hash,
        texture: mapped,
      },
    );
    self.by_key.insert(key, id);
    self.by_hash.entry(hash).or_default().push(id);
    Ok((id, true))
  }

  fn find_by_hash(&self, hash: u64, same: impl Fn(&Self, AssetId) -> bool) -> Option<AssetId> {
    self.by_hash.get(&hash)?.iter().copied().find(|&id| same(self, id))
  }
}

impl Drop for AssetLibrary {
  fn drop(&mut self) {
    self.clear();
  }
}

// ---------------------------------------------------------------------------------------------
// Cache file format
// ---------------------------------------------------------------------------------------------

const CACHE_MAGIC: [u8; 4] = *b"AVKC";
const CACHE_VERSION: u32 = 1;
/// Payload offset: one page, so the payload of a mapping is page (hence `Vertex`/`u32`) aligned.
pub(crate) const CACHE_PAYLOAD_OFFSET: usize = 4096;

struct CacheHeader {
  kind: AssetKind,
  /// `vk::Format` raw value for textures, 0 for meshes.
  format: u32,
  /// Texture: width. Mesh: vertex count.
  a: u64,
  /// Texture: height. Mesh: index count.
  b: u64,
  payload_len: u64,
  content_hash: u64,
}

impl CacheHeader {
  fn encode(&self) -> Vec<u8> {
    let mut out = alloc::vec![0u8; CACHE_PAYLOAD_OFFSET];
    out[0..4].copy_from_slice(&CACHE_MAGIC);
    out[4..8].copy_from_slice(&CACHE_VERSION.to_le_bytes());
    out[8..12].copy_from_slice(&(self.kind as u32).to_le_bytes());
    out[12..16].copy_from_slice(&self.format.to_le_bytes());
    out[16..24].copy_from_slice(&self.a.to_le_bytes());
    out[24..32].copy_from_slice(&self.b.to_le_bytes());
    out[32..40].copy_from_slice(&self.payload_len.to_le_bytes());
    out[40..48].copy_from_slice(&self.content_hash.to_le_bytes());
    out
  }

  fn decode(bytes: &[u8]) -> Option<Self> {
    if bytes.len() < 48 || bytes[0..4] != CACHE_MAGIC {
      return None;
    }
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    if u32_at(4) != CACHE_VERSION {
      return None;
    }
    let kind = match u32_at(8) {
      1 => AssetKind::Mesh,
      2 => AssetKind::Texture,
      _ => return None,
    };
    Some(Self {
      kind,
      format: u32_at(12),
      a: u64_at(16),
      b: u64_at(24),
      payload_len: u64_at(32),
      content_hash: u64_at(40),
    })
  }
}

/// Writes `header ‖ parts` to `path`, maps the file read-only and returns the payload as
/// file-backed `Bytes` (the mapping lives as long as any clone of the returned `Bytes`).
fn write_and_map(
  path: &str,
  header: &CacheHeader,
  parts: &[&[u8]],
) -> Result<bytes::Bytes, AssetError> {
  let encoded = header.encode();
  let mut all: Vec<&[u8]> = Vec::with_capacity(parts.len() + 1);
  all.push(&encoded);
  all.extend_from_slice(parts);
  fs::write_parts(path, &all).map_err(|_| AssetError::CacheIo)?;
  map_cache_file(path, header.kind)
}

/// Maps an existing cache file and returns its payload, validating the header.
pub(crate) fn map_cache_file(path: &str, expected: AssetKind) -> Result<bytes::Bytes, AssetError> {
  let mapped = MappedFile::new(path).map_err(|_| AssetError::CacheIo)?;
  let header = CacheHeader::decode(&mapped).ok_or(AssetError::CacheIo)?;
  if header.kind != expected {
    return Err(AssetError::CacheIo);
  }
  let end = CACHE_PAYLOAD_OFFSET + header.payload_len as usize;
  if mapped.len() < end {
    return Err(AssetError::CacheIo);
  }
  Ok(bytes::Bytes::from_owner(mapped).slice(CACHE_PAYLOAD_OFFSET..end))
}

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn bytes_per_texel(format: TexelFormat) -> Option<usize> {
  match format {
    TexelFormat::R8_UNORM => Some(1),
    TexelFormat::R8G8_UNORM => Some(2),
    TexelFormat::R8G8B8_UNORM => Some(3),
    TexelFormat::R8G8B8A8_UNORM => Some(4),
    _ => None,
  }
}

/// 64-bit content hash (word-wise multiply/xor + splitmix finaliser). Collisions are guarded by
/// a full byte comparison wherever the hash is used for deduplication.
fn content_hash(parts: &[&[u8]], extra: [u64; 3]) -> u64 {
  const K: u64 = 0x9E37_79B9_7F4A_7C15;
  let mut h: u64 = 0xcbf2_9ce4_8422_2325;
  let mut mix = |w: u64| {
    h = (h ^ w).wrapping_mul(K).rotate_left(31);
  };
  for e in extra {
    mix(e);
  }
  for part in parts {
    mix(part.len() as u64);
    let mut chunks = part.chunks_exact(8);
    for c in &mut chunks {
      mix(u64::from_le_bytes(c.try_into().unwrap()));
    }
    let mut tail = [0u8; 8];
    let rem = chunks.remainder();
    tail[..rem.len()].copy_from_slice(rem);
    mix(u64::from_le_bytes(tail));
  }
  // splitmix64 finaliser
  let mut z = h;
  z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
  z ^ (z >> 31)
}

/// Ritter bounding sphere followed by a growth pass, so every vertex is contained.
pub fn bounding_sphere(vertices: &[Vertex]) -> ([f32; 3], f32) {
  if vertices.is_empty() {
    return ([0.0; 3], 0.0);
  }
  let d2 = |a: [f32; 3], b: [f32; 3]| {
    let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    x * x + y * y + z * z
  };
  let p0 = vertices[0].position;
  let far = |from: [f32; 3]| {
    vertices
      .iter()
      .map(|v| v.position)
      .fold((from, 0.0f32), |(best, bd), p| {
        let d = d2(from, p);
        if d > bd { (p, d) } else { (best, bd) }
      })
      .0
  };
  let a = far(p0);
  let b = far(a);
  let mut c = [
    (a[0] + b[0]) * 0.5,
    (a[1] + b[1]) * 0.5,
    (a[2] + b[2]) * 0.5,
  ];
  let mut r = d2(a, b).sqrt() * 0.5;
  for v in vertices {
    let p = v.position;
    let d = d2(c, p).sqrt();
    if d > r {
      let new_r = (r + d) * 0.5;
      let k = (new_r - r) / d;
      c = [
        c[0] + (p[0] - c[0]) * k,
        c[1] + (p[1] - c[1]) * k,
        c[2] + (p[2] - c[2]) * k,
      ];
      r = new_r;
    }
  }
  // Floating-point slack: make containment exact.
  let max_d = vertices.iter().map(|v| d2(c, v.position)).fold(0.0f32, f32::max).sqrt();
  (c, r.max(max_d))
}

/// Lexically normalised path used as asset key: `/` separators, `.`/`..` resolved and,
/// on Windows, case folded.
pub fn normalize_key(path: &str) -> String {
  let unified = path.replace('\\', "/");
  let absolute = unified.starts_with('/');
  let mut parts: Vec<&str> = Vec::new();
  for seg in unified.split('/') {
    match seg {
      "" | "." => {}
      ".." => {
        if matches!(parts.last(), Some(p) if *p != "..") {
          parts.pop();
        } else if !absolute {
          parts.push("..");
        }
      }
      s => parts.push(s),
    }
  }
  let mut out = parts.join("/");
  if absolute {
    out.insert(0, '/');
  }
  if cfg!(windows) {
    out = out.to_lowercase();
  }
  out
}

fn extension_of(path: &str) -> String {
  let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
  match name.rfind('.') {
    Some(i) if i > 0 => name[i + 1..].to_ascii_lowercase(),
    _ => String::new(),
  }
}

/// File name without directory and extension (asset label).
pub fn file_stem(path: &str) -> String {
  let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
  match name.rfind('.') {
    Some(i) if i > 0 => name[..i].to_string(),
    _ => name.to_string(),
  }
}

fn parent_dir(path: &str) -> String {
  match path.rfind(['/', '\\']) {
    Some(i) => path[..i].to_string(),
    None => String::new(),
  }
}

fn join_path(dir: &str, rel: &str) -> String {
  if rel.starts_with('/') || rel.contains(":\\") || rel.contains(":/") || dir.is_empty() {
    rel.to_string()
  } else {
    format!("{dir}/{rel}")
  }
}

/// Texture maps referenced by the `.mtl` libraries of an OBJ file.
///
/// `map_Kd` → albedo, `map_Bump`/`bump`/`norm` → normal, `map_Pr` → roughness,
/// `map_ao`/`map_AO` → ambient occlusion. Map options (`-bm 1.0 …`) are skipped by taking the
/// last token as file name.
pub(crate) fn obj_material_textures(obj_path: &str) -> Vec<(TextureChannel, String)> {
  let mut out: Vec<(TextureChannel, String)> = Vec::new();
  let Ok(data) = fs::read(obj_path) else {
    return out;
  };
  let Ok(text) = core::str::from_utf8(&data) else {
    return out;
  };
  let dir = parent_dir(obj_path);
  for line in text.lines() {
    let line = line.trim();
    let Some(rest) = line.strip_prefix("mtllib") else {
      continue;
    };
    let mtl_path = join_path(&dir, rest.trim());
    let Ok(mtl) = fs::read(mtl_path.as_str()) else {
      oshal::log!("OBJ mtllib '{}' not readable", mtl_path);
      continue;
    };
    let Ok(mtl) = core::str::from_utf8(&mtl) else {
      continue;
    };
    let mtl_dir = parent_dir(&mtl_path);
    for l in mtl.lines() {
      let mut tokens = l.split_whitespace();
      let Some(directive) = tokens.next() else {
        continue;
      };
      let channel = match directive {
        "map_Kd" => TextureChannel::Albedo,
        "map_Bump" | "map_bump" | "bump" | "norm" => TextureChannel::Normal,
        "map_Pr" => TextureChannel::Roughness,
        "map_ao" | "map_AO" => TextureChannel::Ao,
        _ => continue,
      };
      let Some(file) = tokens.last() else {
        continue;
      };
      if !out.iter().any(|(c, _)| *c == channel) {
        out.push((channel, join_path(&mtl_dir, file)));
      }
    }
  }
  out
}

/// `comment TextureFile <name>` header line used by several PLY exporters (MeshLab, Blender).
pub(crate) fn ply_texture_file(ply_path: &str) -> Option<String> {
  let data = fs::read(ply_path).ok()?;
  // The header is ASCII even for binary PLY; stop at end_header.
  let end = data
    .windows(10)
    .position(|w| w == b"end_header")
    .unwrap_or(data.len().min(64 * 1024));
  let header = core::str::from_utf8(&data[..end]).ok()?;
  header.lines().find_map(|l| {
    let rest = l.trim().strip_prefix("comment")?.trim();
    let name = rest.strip_prefix("TextureFile")?.trim();
    (!name.is_empty()).then(|| join_path(&parent_dir(ply_path), name))
  })
}

#[cfg(test)]
mod tests;
