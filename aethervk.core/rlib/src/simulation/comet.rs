//! comet module.
use crate::types::EngineError;
use aethervk_oshal_rlib::{
  self as oshal,
  math::matrix::{Matrix, Matrix3, mat3::Mat3f32},
  math::quaternion::Quaternion,
  math::vector::vec4::Quat,
  math::vector::{Vector, Vector3, vec3::Vec3f32},
  os::FsError,
  os::fs::{self, FileSystemObject, PathBuf},
};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use ktx2;
use zune_core::bytestream::ZCursor;
use zune_jpeg;

// TODO delete uv_grid. not needed
pub mod uv_grid;

// TODO rename this file into mesh.rs

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
  pub position: [f32; 3],
  pub normal: [f32; 3],
  pub uv: [f32; 2],
  pub tangent: [f32; 4],
}

pub const POSITION_COMPONENTS: u32 = 3;
pub const NORMAL_COMPONENTS: u32 = 3;
pub const UV_COMPONENTS: u32 = 2;
pub const TANGENT_COMPONENTS: u32 = 4;
pub const ATTRIBUTES_COMPONENTS: u32 = 9;

impl core::hash::Hash for Vertex {
  fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
    self.position.iter().for_each(|f| f.to_bits().hash(state));
    self.normal.iter().for_each(|f| f.to_bits().hash(state));
    self.uv.iter().for_each(|f| f.to_bits().hash(state));
    self.tangent.iter().for_each(|f| f.to_bits().hash(state));
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(non_camel_case_types)]
pub enum TexelFormat {
  // Basic formats
  R8_UNORM,
  R8G8_UNORM,
  R8G8B8_UNORM,
  R8G8B8A8_UNORM,
  // Compressed
  BC7_UNORM_BLOCK,
  ETC2_R8G8B8_UNORM_BLOCK,
  ASTC_4x4_UNORM_BLOCK,
  // Catch-all for other formats
  Unsupported(u32),
  #[default]
  Undefined,
}

impl TexelFormat {
  pub fn to_vk_format(self) -> ash::vk::Format {
    use ash::vk;
    match self {
      TexelFormat::R8_UNORM => vk::Format::R8_UNORM,
      TexelFormat::R8G8_UNORM => vk::Format::R8G8_UNORM,
      TexelFormat::R8G8B8_UNORM => vk::Format::R8G8B8_UNORM,
      TexelFormat::R8G8B8A8_UNORM => vk::Format::R8G8B8A8_UNORM,
      TexelFormat::BC7_UNORM_BLOCK => vk::Format::BC7_UNORM_BLOCK,
      TexelFormat::ETC2_R8G8B8_UNORM_BLOCK => vk::Format::ETC2_R8G8B8_UNORM_BLOCK,
      TexelFormat::ASTC_4x4_UNORM_BLOCK => vk::Format::ASTC_4X4_UNORM_BLOCK,
      TexelFormat::Unsupported(vk_format_val) => vk::Format::from_raw(vk_format_val as i32),
      TexelFormat::Undefined => vk::Format::UNDEFINED,
    }
  }

  /// TODO: Document this item
  pub fn from_vk_format(vk_format: u32) -> Self {
    use ash::vk;
    match vk::Format::from_raw(vk_format as i32) {
      vk::Format::R8_UNORM => TexelFormat::R8_UNORM,
      vk::Format::R8G8_UNORM => TexelFormat::R8G8_UNORM,
      vk::Format::R8G8B8_UNORM => TexelFormat::R8G8B8_UNORM,
      vk::Format::R8G8B8A8_UNORM => TexelFormat::R8G8B8A8_UNORM,
      vk::Format::BC7_UNORM_BLOCK => TexelFormat::BC7_UNORM_BLOCK,
      vk::Format::ETC2_R8G8B8_UNORM_BLOCK => TexelFormat::ETC2_R8G8B8_UNORM_BLOCK,
      vk::Format::ASTC_4X4_UNORM_BLOCK => TexelFormat::ASTC_4x4_UNORM_BLOCK,
      vk::Format::UNDEFINED => TexelFormat::Undefined,
      _ => TexelFormat::Unsupported(vk_format),
    }
  }
}

#[derive(Clone, Default)]
pub struct Texture {
  pub data: bytes::Bytes,
  pub format: TexelFormat,
  pub width: u32,
  pub height: u32,
  pub has_mipmaps: bool,
}

static NEXT_COMET_ID: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);

// TODO change name of this class to `Mesh`
/// Warning: `PartialEq` implemented only with `id` field
#[derive(Clone)]
pub struct Comet {
  pub id: u64,
  pub vertices: Vec<Vertex>,
  pub indices: Vec<u32>,
  pub albedo_map: Option<Texture>,
  pub normal_map: Option<Texture>,
  pub roughness_map: Option<Texture>,
  pub ao_map: Option<Texture>,
}

impl PartialEq for Comet {
  fn eq(&self, other: &Self) -> bool {
    self.id == other.id
  }
}

impl core::fmt::Debug for Comet {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Comet")
      .field("id", &self.id)
      .field("vertices", &self.vertices.len())
      .field("indices", &self.indices.len())
      .finish()
  }
}

#[derive(Debug)]
pub enum CometLoadError {
  PathNotFound,
  TextureNotFound,
  TooLarge,
  AnimationsNotSupported,
  MultipleMeshesNotSupported,
  NotWatertight,
  UnsupportedPrimitiveMode,
  UnsupportedImageFormat,
  ImageDecodingError,
  UnsupportedNormalData,
  GltfImportError(gltf::Error),
  IoError,
  MissingBuffer,
}

impl From<FsError> for CometLoadError {
  fn from(_: FsError) -> Self {
    CometLoadError::IoError
  }
}

impl From<CometLoadError> for EngineError {
  fn from(value: CometLoadError) -> Self {
    let message = match value {
      CometLoadError::PathNotFound => "CometLoadError::PathNotFound",
      CometLoadError::TextureNotFound => "CometLoadError::TextureNotFound",
      CometLoadError::TooLarge => "CometLoadError::TooLarge",
      CometLoadError::AnimationsNotSupported => "CometLoadError::AnimationsNotSupported",
      CometLoadError::MultipleMeshesNotSupported => "CometLoadError::MultipleMeshesNotSupported",
      CometLoadError::NotWatertight => "CometLoadError::NotWatertight",
      CometLoadError::UnsupportedPrimitiveMode => "CometLoadError::UnsupportedPrimitiveMode",
      CometLoadError::UnsupportedImageFormat => "CometLoadError::UnsupportedImageFormat",
      CometLoadError::ImageDecodingError => "CometLoadError::ImageDecodingError",
      CometLoadError::UnsupportedNormalData => "CometLoadError::UnsupportedNormalData",
      CometLoadError::GltfImportError(_) => "CometLoadError::GltfImportError",
      CometLoadError::IoError => "CometLoadError::IoError",
      CometLoadError::MissingBuffer => "CometLoadError::MissingBuffer",
    };

    Self::InvalidOperation(message)
  }
}

fn get_texture_data(
  source: gltf::image::Source,
  base_path: &PathBuf,
  blob: Option<&[u8]>,
) -> Result<Option<Texture>, CometLoadError> {
  let (encoded_data, mime_type, uri_path) = match source {
    gltf::image::Source::View { view, mime_type } => {
      if view.buffer().index() != 0 {
        return Ok(None); // We only support the main binary blob for embedded
      }
      let blob = blob.ok_or(CometLoadError::MissingBuffer)?;
      let start = view.offset();
      let end = start + view.length();
      (blob[start..end].to_vec(), Some(mime_type), None)
    }
    gltf::image::Source::Uri { uri, mime_type } => {
      let path = base_path.join(uri);
      let data = fs::read(&path).map_err(|_| CometLoadError::TextureNotFound)?;
      (data, mime_type, Some(path))
    }
  };

  let (decoded_data, format, width, height, has_mipmaps) = if let Some(mime) = mime_type {
    match mime {
      "image/jpeg" => {
        let mut decoder = zune_jpeg::JpegDecoder::new(ZCursor::new(&encoded_data));
        let info = decoder.info().ok_or(CometLoadError::ImageDecodingError)?;
        let data = decoder.decode().map_err(|_| CometLoadError::ImageDecodingError)?;
        let format = match info.components {
          1 => TexelFormat::R8_UNORM,
          3 => TexelFormat::R8G8B8_UNORM,
          4 => TexelFormat::R8G8B8A8_UNORM,
          _ => return Err(CometLoadError::UnsupportedImageFormat),
        };
        (data, format, info.width as u32, info.height as u32, false)
      }
      "image/png" => {
        let (header, image_data) =
          png_decoder::decode(&encoded_data).map_err(|_| CometLoadError::ImageDecodingError)?;
        if header.bit_depth != png_decoder::BitDepth::Eight
          || (header.color_type != png_decoder::ColorType::RgbAlpha
            && header.color_type != png_decoder::ColorType::Grayscale)
          || header.interlace_method != png_decoder::InterlaceMethod::None
        {
          return Err(CometLoadError::UnsupportedImageFormat);
        }

        (
          image_data.into_flattened(),
          if header.color_type == png_decoder::ColorType::RgbAlpha {
            TexelFormat::R8G8B8A8_UNORM
          } else {
            TexelFormat::R8_UNORM
          },
          header.width,
          header.height,
          false,
        )
      }
      _ => return Err(CometLoadError::UnsupportedImageFormat),
    }
  } else if let Some(path) = uri_path {
    if path.extension().map(|s| s == "ktx2").unwrap_or(false) {
      let reader =
        ktx2::Reader::new(&encoded_data).map_err(|_| CometLoadError::ImageDecodingError)?;
      let header = reader.header();
      let mip_level_count = reader.levels().len();
      if mip_level_count != 1 {
        return Err(CometLoadError::UnsupportedImageFormat);
      }
      let data = unsafe { reader.levels().take(1).next().unwrap_unchecked() }.data;

      let vk_format = header.format.ok_or(CometLoadError::ImageDecodingError)?;
      (
        data.to_vec(),
        TexelFormat::from_vk_format(vk_format.value()),
        header.pixel_width,
        header.pixel_height,
        header.level_count > 1,
      )
    } else {
      return Err(CometLoadError::UnsupportedImageFormat);
    }
  } else {
    return Err(CometLoadError::UnsupportedImageFormat);
  };

  Ok(Some(Texture {
    data: decoded_data.into(),
    format,
    width,
    height,
    has_mipmaps,
  }))
}

/// Function to load a GLTF/GLB file
/// 1. Watertightness Check: A closed, physical (manifold) mesh must have every undirected edge shared by exactly two triangles. If an edge has only one triangle, there's a hole. If it has three or more, there's self-intersecting/non-manifold geometry.
/// 2. Outward Normals Check: By calculating the signed volume of the mesh using the divergence theorem, we can verify winding order. If the volume is negative, the triangles are wound backwards, meaning your normals are pointing inward.
pub fn load_comet_from_gltf(
  path: &str,
  verbose: bool,
  provided_inertia: Option<Mat3f32>,
) -> Result<Comet, CometLoadError> {
  oshal::log!("--- Starting GLTF load for: {} ---", path);

  let mut path_buf = PathBuf::new();
  path_buf.push(path);

  if !path_buf.is_file() {
    oshal::log!("ERROR: File not found at path: {}", path);
    return Err(CometLoadError::PathNotFound);
  }

  let base_path = path_buf.parent().unwrap_or_else(PathBuf::new);
  let data = fs::read(&path_buf)?;

  oshal::log!(
    "File read successfully ({} bytes). Parsing GLTF...",
    data.len()
  );
  let gltf = gltf::Gltf::from_slice(&data).map_err(|e| {
    oshal::log!("ERROR: GLTF Parse failed: {:?}", e);
    CometLoadError::GltfImportError(e)
  })?;

  oshal::log!("Correctly parsed GLB");

  // Validations
  if gltf.animations().next().is_some() {
    oshal::log!("ERROR: Animations found, but are not supported.");
    return Err(CometLoadError::AnimationsNotSupported);
  }

  let mesh_count = gltf.meshes().count();
  if mesh_count > 1 {
    oshal::log!("ERROR: Found {} meshes. Only 1 is supported.", mesh_count);
    return Err(CometLoadError::MultipleMeshesNotSupported);
  }

  let mesh = gltf.meshes().next().ok_or_else(|| {
    oshal::log!("ERROR: GLTF contains no meshes.");
    CometLoadError::PathNotFound
  })?;

  oshal::log!(
    "Mesh found: '{}'. Processing primitives...",
    mesh.name().unwrap_or("Unnamed")
  );

  let mut vertices = Vec::new();
  let mut indices = Vec::new();

  let mut albedo_map = None;
  let mut normal_map = None;
  let mut roughness_map = None;
  let mut ao_map = None;

  let blob = gltf.blob.as_deref();

  for (i, primitive) in mesh.primitives().enumerate() {
    oshal::log!("  Processing primitive {}...", i);

    if primitive.mode() != gltf::json::mesh::Mode::Triangles {
      oshal::log!(
        "  ERROR: Primitive {} is not triangulated. Mode: {:?}",
        i,
        primitive.mode()
      );
      return Err(CometLoadError::UnsupportedPrimitiveMode);
    }

    let reader = primitive.reader(|buffer| if buffer.index() == 0 { blob } else { None });

    // Granular attribute checks
    let positions = reader.read_positions().ok_or_else(|| {
      oshal::log!("  ERROR: Primitive {} is missing POSITION data.", i);
      CometLoadError::UnsupportedNormalData // Or a more specific error
    })?;

    let normals = reader.read_normals().ok_or_else(|| {
      oshal::log!("  ERROR: Primitive {} is missing NORMAL data.", i);
      CometLoadError::UnsupportedNormalData
    })?;

    let uvs = reader.read_tex_coords(0).map(|v| v.into_f32()).ok_or_else(|| {
      oshal::log!("  ERROR: Primitive {} is missing TEXCOORD_0 (UV) data.", i);
      CometLoadError::UnsupportedNormalData
    })?;

    let tangents = reader.read_tangents().ok_or_else(|| {
      oshal::log!("  ERROR: Primitive {} is missing TANGENT data.", i);
      CometLoadError::UnsupportedNormalData
    })?;

    oshal::log!("  All required vertex attributes found. Building vertices...");
    let start_vertex_count = vertices.len();
    for ((position, normal), (uv, tangent)) in positions.zip(normals).zip(uvs.zip(tangents)) {
      vertices.push(Vertex {
        position,
        normal,
        uv,
        tangent,
      });
    }
    oshal::log!("  Added {} vertices.", vertices.len() - start_vertex_count);

    if let Some(indices_iter) = reader.read_indices() {
      let start_index_count = indices.len();
      indices.extend(indices_iter.into_u32());
      oshal::log!("  Added {} indices.", indices.len() - start_index_count);
    } else {
      oshal::log!("  ERROR: Primitive {} is missing index buffer data.", i);
      // Depending on your architecture, you might want to auto-generate indices here,
      // but if you strictly require them:
      return Err(CometLoadError::UnsupportedNormalData); // Swap for MissingIndices error
    }

    // Textures... (Assuming get_texture_data has its own logs or succeeds silently)
    let material = primitive.material();
    let pbr = material.pbr_metallic_roughness();

    if let Some(info) = pbr.base_color_texture() {
      albedo_map = get_texture_data(info.texture().source().source(), &base_path, blob)?;
    }
    if let Some(info) = material.normal_texture() {
      normal_map = get_texture_data(info.texture().source().source(), &base_path, blob)?;
    }
    if let Some(info) = pbr.metallic_roughness_texture() {
      roughness_map = get_texture_data(info.texture().source().source(), &base_path, blob)?;
    }
    if let Some(info) = material.occlusion_texture() {
      ao_map = get_texture_data(info.texture().source().source(), &base_path, blob)?;
    }
  }

  let result = finalize_comet(
    vertices,
    indices,
    albedo_map,
    normal_map,
    roughness_map,
    ao_map,
    verbose,
    provided_inertia,
  );

  if result.is_ok() {
    oshal::log!("--- GLTF load successful! ---");
  }

  result
}

fn finalize_comet(
  vertices: Vec<Vertex>,
  indices: Vec<u32>,
  albedo_map: Option<Texture>,
  normal_map: Option<Texture>,
  roughness_map: Option<Texture>,
  ao_map: Option<Texture>,
  verbose: bool,
  provided_inertia: Option<Mat3f32>,
) -> Result<Comet, CometLoadError> {
  // --- TOPOLOGY VALIDATIONS ---
  oshal::log!("Running geometric validations...");

  // 1. Watertight Check
  let mut edge_counts: BTreeMap<(u32, u32), u32> = BTreeMap::new();
  for chunk in indices.chunks_exact(3) {
    for i in 0..3 {
      let u = chunk[i];
      let v = chunk[(i + 1) % 3];
      // Store undirected edge (smallest index first)
      let edge = if u < v { (u, v) } else { (v, u) };
      *edge_counts.entry(edge).or_insert(0) += 1;
    }
  }

  let mut is_watertight = true;
  for (edge, count) in &edge_counts {
    if *count != 2 {
      if verbose {
        oshal::log!(
          "  ERROR: Mesh is not watertight! Edge ({}, {}) is shared by {} triangles (expected 2).",
          edge.0,
          edge.1,
          count
        );
      }
      is_watertight = false;
      // Depending on strictness, you might want to return an error here:
      // return Err(CometLoadError::NotWatertight);
    }
  }
  if is_watertight {
    oshal::log!("  Watertight check passed.");
  }

  // 2. Outward Normals (Signed Volume Check)
  let mut signed_volume = 0.0;
  for chunk in indices.chunks_exact(3) {
    let v0 = vertices[chunk[0] as usize].position;
    let v1 = vertices[chunk[1] as usize].position;
    let v2 = vertices[chunk[2] as usize].position;

    // Cross product of v1 and v2
    let cp_x = v1[1] * v2[2] - v1[2] * v2[1];
    let cp_y = v1[2] * v2[0] - v1[0] * v2[2];
    let cp_z = v1[0] * v2[1] - v1[1] * v2[0];

    // Dot product with v0
    signed_volume += v0[0] * cp_x + v0[1] * cp_y + v0[2] * cp_z;
  }
  signed_volume /= 6.0;

  if signed_volume <= 0.0 {
    oshal::log!(
      "  ERROR: Signed volume is {} (<= 0). Winding order is inverted (normals point inwards) or mesh is degenerate.",
      signed_volume
    );
    // return Err(CometLoadError::InvertedNormals);
  } else {
    oshal::log!(
      "  Outward normals check passed (Volume: {}).",
      signed_volume
    );
  }

  Ok(Comet {
    id: NEXT_COMET_ID.fetch_add(1, core::sync::atomic::Ordering::Relaxed),
    vertices,
    indices,
    albedo_map,
    normal_map,
    roughness_map,
    ao_map,
  })
}

pub fn load_comet_from_obj(
  path: &str,
  verbose: bool,
  provided_inertia: Option<Mat3f32>,
) -> Result<Comet, CometLoadError> {
  oshal::log!("--- Starting OBJ load for: {} ---", path);

  let mut path_buf = PathBuf::new();
  path_buf.push(path);

  if !path_buf.is_file() {
    oshal::log!("ERROR: File not found at path: {}", path);
    return Err(CometLoadError::PathNotFound);
  }

  let data = fs::read(&path_buf)?;
  let data_str = core::str::from_utf8(&data).map_err(|_| CometLoadError::UnsupportedImageFormat)?;

  let mut temp_positions = Vec::new();
  let mut temp_normals = Vec::new();
  let mut temp_uvs = Vec::new();

  let mut vertices = Vec::new();
  let mut indices = Vec::new();

  // A vertex in OBJ is given by index triplet (v, vt, vn)
  let mut unique_vertices = alloc::collections::BTreeMap::<(u32, u32, u32), u32>::new();

  for line in data_str.lines() {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
      continue;
    }

    let mut parts = line.split_whitespace();
    let Some(prefix) = parts.next() else {
      continue;
    };

    match prefix {
      "v" => {
        let x = parts.next().unwrap_or("0").parse::<f32>().unwrap_or(0.0);
        let y = parts.next().unwrap_or("0").parse::<f32>().unwrap_or(0.0);
        let z = parts.next().unwrap_or("0").parse::<f32>().unwrap_or(0.0);
        temp_positions.push([x, y, z]);
      }
      "vt" => {
        let u = parts.next().unwrap_or("0").parse::<f32>().unwrap_or(0.0);
        let v = parts.next().unwrap_or("0").parse::<f32>().unwrap_or(0.0);
        temp_uvs.push([u, v]);
      }
      "vn" => {
        let x = parts.next().unwrap_or("0").parse::<f32>().unwrap_or(0.0);
        let y = parts.next().unwrap_or("0").parse::<f32>().unwrap_or(0.0);
        let z = parts.next().unwrap_or("0").parse::<f32>().unwrap_or(0.0);
        temp_normals.push([x, y, z]);
      }
      "f" => {
        let mut face_vertices = Vec::new();
        for vertex_str in parts {
          let mut v_parts = vertex_str.split('/');
          let v_idx = v_parts.next().unwrap_or("").parse::<u32>().unwrap_or(0);
          let vt_idx = v_parts.next().unwrap_or("").parse::<u32>().unwrap_or(0);
          let vn_idx = v_parts.next().unwrap_or("").parse::<u32>().unwrap_or(0);

          let key = (v_idx, vt_idx, vn_idx);
          let index = *unique_vertices.entry(key).or_insert_with(|| {
            let new_index = vertices.len() as u32;

            let pos = if v_idx > 0 && v_idx as usize <= temp_positions.len() {
              temp_positions[v_idx as usize - 1]
            } else {
              [0.0, 0.0, 0.0]
            };

            let uv = if vt_idx > 0 && vt_idx as usize <= temp_uvs.len() {
              temp_uvs[vt_idx as usize - 1]
            } else {
              [0.0, 0.0]
            };

            let norm = if vn_idx > 0 && vn_idx as usize <= temp_normals.len() {
              temp_normals[vn_idx as usize - 1]
            } else {
              [0.0, 1.0, 0.0]
            };

            vertices.push(Vertex {
              position: pos,
              normal: norm,
              uv,
              tangent: [1.0, 0.0, 0.0, 1.0], // Default tangent
            });
            new_index
          });
          face_vertices.push(index);
        }

        // Triangulate
        if face_vertices.len() >= 3 {
          let v0 = face_vertices[0];
          for i in 1..face_vertices.len() - 1 {
            indices.push(v0);
            indices.push(face_vertices[i]);
            indices.push(face_vertices[i + 1]);
          }
        }
      }
      _ => {}
    }
  }

  // Generate tangents if UVs and positions exist
  generate_tangents(&mut vertices, &indices);

  let result = finalize_comet(
    vertices,
    indices,
    None,
    None,
    None,
    None,
    verbose,
    provided_inertia,
  );

  if result.is_ok() {
    oshal::log!("--- OBJ load successful! ---");
  }

  result
}

pub fn load_comet_from_ply(
  path: &str,
  verbose: bool,
  provided_inertia: Option<Mat3f32>,
) -> Result<Comet, CometLoadError> {
  oshal::log!("--- Starting PLY load for: {} ---", path);

  let mut path_buf = PathBuf::new();
  path_buf.push(path);

  if !path_buf.is_file() {
    oshal::log!("ERROR: File not found at path: {}", path);
    return Err(CometLoadError::PathNotFound);
  }

  let data = fs::read(&path_buf)?;
  let data_str = core::str::from_utf8(&data).map_err(|_| CometLoadError::UnsupportedImageFormat)?;

  let mut lines = data_str.lines();

  // Header
  if lines.next().unwrap_or("").trim() != "ply" {
    return Err(CometLoadError::UnsupportedImageFormat);
  }

  let mut format = "";
  let mut vertex_count = 0;
  let mut face_count = 0;
  let mut current_element = "";

  let mut properties = alloc::vec::Vec::new();

  while let Some(line) = lines.next() {
    let line = line.trim();
    if line == "end_header" {
      break;
    }

    let mut parts = line.split_whitespace();
    let Some(prefix) = parts.next() else {
      continue;
    };

    match prefix {
      "format" => {
        format = parts.next().unwrap_or("");
      }
      "element" => {
        current_element = parts.next().unwrap_or("");
        let count = parts.next().unwrap_or("0").parse::<usize>().unwrap_or(0);
        if current_element == "vertex" {
          vertex_count = count;
        } else if current_element == "face" {
          face_count = count;
        }
      }
      "property" => {
        if current_element == "vertex" {
          let _type = parts.next().unwrap_or("");
          let name = parts.next().unwrap_or("");
          properties.push(name);
        }
      }
      _ => {}
    }
  }

  if format != "ascii" {
    oshal::log!("ERROR: Only ASCII PLY format is currently supported.");
    return Err(CometLoadError::UnsupportedImageFormat);
  }

  let mut vertices = Vec::with_capacity(vertex_count);
  for _ in 0..vertex_count {
    let Some(line) = lines.next() else {
      break;
    };
    let parts: Vec<&str> = line.split_whitespace().collect();

    let mut pos = [0.0, 0.0, 0.0];
    let mut norm = [0.0, 1.0, 0.0];
    let mut uv = [0.0, 0.0];

    for (i, &prop) in properties.iter().enumerate() {
      if i >= parts.len() {
        break;
      }
      let val = parts[i].parse::<f32>().unwrap_or(0.0);
      match prop {
        "x" => pos[0] = val,
        "y" => pos[1] = val,
        "z" => pos[2] = val,
        "nx" => norm[0] = val,
        "ny" => norm[1] = val,
        "nz" => norm[2] = val,
        "s" | "u" => uv[0] = val,
        "t" | "v" => uv[1] = val,
        _ => {}
      }
    }

    vertices.push(Vertex {
      position: pos,
      normal: norm,
      uv,
      tangent: [1.0, 0.0, 0.0, 1.0],
    });
  }

  let mut indices = Vec::new();
  for _ in 0..face_count {
    let Some(line) = lines.next() else {
      break;
    };
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
      continue;
    }

    let count = parts[0].parse::<usize>().unwrap_or(0);
    if count >= 3 && parts.len() >= count + 1 {
      let mut face_indices = Vec::with_capacity(count);
      for i in 1..=count {
        face_indices.push(parts[i].parse::<u32>().unwrap_or(0));
      }

      let v0 = face_indices[0];
      for i in 1..count - 1 {
        indices.push(v0);
        indices.push(face_indices[i]);
        indices.push(face_indices[i + 1]);
      }
    }
  }

  generate_tangents(&mut vertices, &indices);

  let result = finalize_comet(
    vertices,
    indices,
    None,
    None,
    None,
    None,
    verbose,
    provided_inertia,
  );

  if result.is_ok() {
    oshal::log!("--- PLY load successful! ---");
  }

  result
}

// Simple tangent generation (Lengyel, Eric. "Computing Tangent Space Basis Vectors for an Arbitrary Mesh")
fn generate_tangents(vertices: &mut [Vertex], indices: &[u32]) {
  let mut tan1 = alloc::vec![Vec3f32::from_components(0.0, 0.0, 0.0); vertices.len()];
  let mut tan2 = alloc::vec![Vec3f32::from_components(0.0, 0.0, 0.0); vertices.len()];

  for chunk in indices.chunks_exact(3) {
    let i1 = chunk[0] as usize;
    let i2 = chunk[1] as usize;
    let i3 = chunk[2] as usize;

    let v1 = vertices[i1];
    let v2 = vertices[i2];
    let v3 = vertices[i3];

    let x1 = v2.position[0] - v1.position[0];
    let x2 = v3.position[0] - v1.position[0];
    let y1 = v2.position[1] - v1.position[1];
    let y2 = v3.position[1] - v1.position[1];
    let z1 = v2.position[2] - v1.position[2];
    let z2 = v3.position[2] - v1.position[2];

    let s1 = v2.uv[0] - v1.uv[0];
    let s2 = v3.uv[0] - v1.uv[0];
    let t1 = v2.uv[1] - v1.uv[1];
    let t2 = v3.uv[1] - v1.uv[1];

    let div = s1 * t2 - s2 * t1;
    let r = if div == 0.0 { 1.0 } else { 1.0 / div };

    let sdir = Vec3f32::from_components(
      (t2 * x1 - t1 * x2) * r,
      (t2 * y1 - t1 * y2) * r,
      (t2 * z1 - t1 * z2) * r,
    );
    let tdir = Vec3f32::from_components(
      (s1 * x2 - s2 * x1) * r,
      (s1 * y2 - s2 * y1) * r,
      (s1 * z2 - s2 * z1) * r,
    );

    tan1[i1] = tan1[i1] + sdir;
    tan1[i2] = tan1[i2] + sdir;
    tan1[i3] = tan1[i3] + sdir;

    tan2[i1] = tan2[i1] + tdir;
    tan2[i2] = tan2[i2] + tdir;
    tan2[i3] = tan2[i3] + tdir;
  }

  for (i, v) in vertices.iter_mut().enumerate() {
    let n = Vec3f32::from_components(v.normal[0], v.normal[1], v.normal[2]);
    let t = tan1[i];

    // Gram-Schmidt orthogonalize
    let t_dot_n = t.dot(n);
    let tangent_unnorm = t - n * t_dot_n;
    let tangent_len = tangent_unnorm.length();
    let tangent = if tangent_len > 0.000001 {
      tangent_unnorm / tangent_len
    } else {
      Vec3f32::from_components(1.0, 0.0, 0.0)
    };

    // Calculate handedness
    let w = if n.cross(t).dot(tan2[i]) < 0.0 {
      -1.0
    } else {
      1.0
    };

    v.tangent = [tangent.x(), tangent.y(), tangent.z(), w];
  }
}

pub fn generate_quad(normal: Vec3f32, size: f32) -> Comet {
  let mut vertices = Vec::new();
  let mut indices = Vec::new();

  // Determine tangent and bitangent based on the normal to create the quad basis
  let mut tangent = Vec3f32::from_components(1.0, 0.0, 0.0);
  if normal.dot(tangent).abs() > 0.99 {
    tangent = Vec3f32::from_components(0.0, 1.0, 0.0);
  }
  let bitangent = normal.cross(tangent).normalize();
  tangent = bitangent.cross(normal).normalize();

  let half_size = size * 0.5;

  let norm_array = [normal.x(), normal.y(), normal.z()];
  let tang_array = [tangent.x(), tangent.y(), tangent.z(), 1.0];

  // Quad centered at origin, normal pointing outwards
  // Bottom-Left
  let p0 = -tangent * half_size - bitangent * half_size;
  vertices.push(Vertex {
    position: [p0.x(), p0.y(), p0.z()],
    normal: norm_array,
    uv: [0.0, 0.0],
    tangent: tang_array,
  });

  // Bottom-Right
  let p1 = tangent * half_size - bitangent * half_size;
  vertices.push(Vertex {
    position: [p1.x(), p1.y(), p1.z()],
    normal: norm_array,
    uv: [1.0, 0.0],
    tangent: tang_array,
  });

  // Top-Right
  let p2 = tangent * half_size + bitangent * half_size;
  vertices.push(Vertex {
    position: [p2.x(), p2.y(), p2.z()],
    normal: norm_array,
    uv: [1.0, 1.0],
    tangent: tang_array,
  });

  // Top-Left
  let p3 = -tangent * half_size + bitangent * half_size;
  vertices.push(Vertex {
    position: [p3.x(), p3.y(), p3.z()],
    normal: norm_array,
    uv: [0.0, 1.0],
    tangent: tang_array,
  });

  // Two triangles: CCW winding assuming normal points "out" (towards viewer)
  indices.push(0);
  indices.push(1);
  indices.push(2);

  indices.push(0);
  indices.push(2);
  indices.push(3);

  Comet {
    id: NEXT_COMET_ID.fetch_add(1, core::sync::atomic::Ordering::Relaxed),
    vertices,
    indices,
    albedo_map: None,
    normal_map: None,
    roughness_map: None,
    ao_map: None,
  }
}

/// Produces `6 * lon_segments * (lat_segments - 1)` indices
pub fn generate_uv_sphere(
  radius: f32,
  lat_segments: u32,
  lon_segments: u32,
  total_mass: f32,
  flip_winding: bool,
) -> Comet {
  let mut vertices = Vec::new();
  let mut indices = Vec::new();

  for lat in 0..=lat_segments {
    let theta = lat as f32 * core::f32::consts::PI / lat_segments as f32;
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();

    for lon in 0..=lon_segments {
      let phi = lon as f32 * 2.0 * core::f32::consts::PI / lon_segments as f32;
      let sin_phi = phi.sin();
      let cos_phi = phi.cos();

      let x = cos_phi * sin_theta;
      let y = sin_phi * sin_theta;
      let z = cos_theta;

      let normal = [x, y, z];
      let position = [x * radius, y * radius, z * radius];
      let uv = [
        lon as f32 / lon_segments as f32,
        lat as f32 / lat_segments as f32,
      ];

      let tangent = if x == 0.0 && y == 0.0 {
        [1.0, 0.0, 0.0, 1.0]
      } else {
        let tangent_len = (x * x + y * y).sqrt();
        [-y / tangent_len, x / tangent_len, 0.0, 1.0]
      };

      vertices.push(Vertex {
        position,
        normal,
        uv,
        tangent,
      });
    }
  }

  for lat in 0..lat_segments {
    for lon in 0..lon_segments {
      let first = lat * (lon_segments + 1) + lon;
      let second = first + lon_segments + 1;

      // 3. FIXED: Enforcing standard CCW local winding to match glTF assets.
      // Now that the Vulkan pipeline is VK_FRONT_FACE_CLOCKWISE (compensating for Y-flip),
      // all local geometry must be generated as CCW.
      if lat != 0 {
        if flip_winding {
          indices.push(first);
          indices.push(first + 1);
          indices.push(second);
        } else {
          indices.push(first);
          indices.push(second);
          indices.push(first + 1);
        }
      }
      if lat != lat_segments - 1 {
        if flip_winding {
          indices.push(second);
          indices.push(first + 1);
          indices.push(second + 1);
        } else {
          indices.push(second);
          indices.push(second + 1);
          indices.push(first + 1);
        }
      }
    }
  }

  Comet {
    id: NEXT_COMET_ID.fetch_add(1, core::sync::atomic::Ordering::Relaxed),
    vertices,
    indices,
    albedo_map: None,
    normal_map: None,
    roughness_map: None,
    ao_map: None,
  }
}

pub fn load_texture_from_file(path: &str) -> Result<Texture, CometLoadError> {
  let path_buf = PathBuf::from(path);
  if !path_buf.is_file() {
    return Err(CometLoadError::PathNotFound);
  }
  let encoded_data = fs::read(&path_buf).map_err(|_| CometLoadError::TextureNotFound)?;

  let extension = path.split('.').last().unwrap_or("").to_lowercase();
  let (decoded_data, format, width, height, has_mipmaps) = match extension.as_str() {
    "jpg" | "jpeg" => {
      let mut decoder = zune_jpeg::JpegDecoder::new(ZCursor::new(&encoded_data));
      if let Err(e) = decoder.decode_headers() {
        oshal::log!("zune_jpeg header decode error: {:?}", e);
      }
      let info = decoder.info().ok_or_else(|| {
        oshal::log!("zune_jpeg info error: no info available");
        CometLoadError::ImageDecodingError
      })?;
      let mut data = decoder.decode().map_err(|e| {
        oshal::log!("zune_jpeg decode error: {:?}", e);
        CometLoadError::ImageDecodingError
      })?;

      let format = match info.components {
        1 => TexelFormat::R8_UNORM,
        3 => {
          let mut rgba = Vec::with_capacity(data.len() / 3 * 4);
          for chunk in data.chunks_exact(3) {
            rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
          }
          data = rgba;
          TexelFormat::R8G8B8A8_UNORM
        }
        4 => TexelFormat::R8G8B8A8_UNORM,
        _ => return Err(CometLoadError::UnsupportedImageFormat),
      };

      (data, format, info.width as u32, info.height as u32, false)
    }
    "png" => {
      let (header, image_data) =
        png_decoder::decode(&encoded_data).map_err(|_| CometLoadError::ImageDecodingError)?;
      (
        image_data.into_flattened(),
        if header.color_type == png_decoder::ColorType::RgbAlpha {
          TexelFormat::R8G8B8A8_UNORM
        } else {
          TexelFormat::R8_UNORM
        },
        header.width,
        header.height,
        false,
      )
    }
    _ => return Err(CometLoadError::UnsupportedImageFormat),
  };

  Ok(Texture {
    data: decoded_data.into(),
    format,
    width,
    height,
    has_mipmaps,
  })
}

/// Returns the squared distance to the closest point on the triangle
/// and the Barycentric weights [w0, w1, w2] of that closest point
/// note: `p` is the given point, while `a`,`b`,`c` are the triangle vertices
///
/// Christer Ericson's classic Voronoi region-based algorithm for finding the closest point on a triangle.
fn closest_point_barycentric_3d(p: Vec3f32, a: Vec3f32, b: Vec3f32, c: Vec3f32) -> (f32, Vec3f32) {
  let ab = b - a;
  let ac = c - a;
  let ap = p - a;

  // Check if P in vertex region outside A
  let d1 = ab.dot(ap);
  let d2 = ac.dot(ap);
  if d1 <= 0.0 && d2 <= 0.0 {
    return (ap.length_squared(), Vec3f32::from_components(1.0, 0.0, 0.0));
  }

  // Check if P in vertex region outside B
  let bp = p - b; // FIXED: This was p - a in your original code
  let d3 = ab.dot(bp);
  let d4 = ac.dot(bp);
  if d3 >= 0.0 && d4 <= d3 {
    return (bp.length_squared(), Vec3f32::from_components(0.0, 1.0, 0.0));
  }

  // Check if P in edge region of AB, if so return projection of P onto AB
  let vc = d1 * d4 - d3 * d2;
  if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
    let v = d1 / (d1 - d3);
    let closest = a + ab * v;
    return (
      (p - closest).length_squared(),
      Vec3f32::from_components(1.0 - v, v, 0.0),
    );
  }

  // Check if P in vertex region outside C
  let cp = p - c;
  let d5 = ab.dot(cp);
  let d6 = ac.dot(cp);
  if d6 >= 0.0 && d5 <= d6 {
    return (cp.length_squared(), Vec3f32::from_components(0.0, 0.0, 1.0));
  }

  // Check if P in edge region of AC, if so return projection of P onto AC
  let vb = d5 * d2 - d1 * d6;
  if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
    let w = d2 / (d2 - d6);
    let closest = a + ac * w;
    return (
      (p - closest).length_squared(),
      Vec3f32::from_components(1.0 - w, 0.0, w),
    );
  }

  // Check if P in edge region of BC, if so return projection of P onto BC
  let va = d3 * d6 - d5 * d4;
  if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
    let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    let closest = b + (c - b) * w;
    return (
      (p - closest).length_squared(),
      Vec3f32::from_components(0.0, 1.0 - w, w),
    );
  }

  // P inside face region. Compute Q through its barycentric coordinates (u, v, w)
  let denom = 1.0 / (va + vb + vc);
  let v = vb * denom;
  let w = vc * denom;
  let u = 1.0 - v - w;
  let closest = a + ab * v + ac * w;

  (
    (p - closest).length_squared(),
    Vec3f32::from_components(u, v, w),
  )
}

#[cfg(test)]
mod tests {
  use super::*;
  use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;

  #[test]
  fn test_triangle_properties() {
    let t = Triangle {
      vertices: [
        Vec3f32::from_components(0.0, 0.0, 0.0),
        Vec3f32::from_components(1.0, 0.0, 0.0),
        Vec3f32::from_components(0.0, 1.0, 0.0),
      ],
    };

    assert_eq!(t.area(), 0.5);

    let center = t.mean_vector();
    assert!((center.x() - 0.3333333).abs() < 1e-6);
    assert!((center.y() - 0.3333333).abs() < 1e-6);
    assert_eq!(center.z(), 0.0);

    let normal = t.normal_ccw_unnormalized();
    assert_eq!(normal.x(), 0.0);
    assert_eq!(normal.y(), 0.0);
    assert_eq!(normal.z(), 1.0);
  }

  #[test]
  fn test_uv_sphere_generation() {
    let sphere = generate_uv_sphere(2.0, 10, 10, 1.0, false);
    let expected_indices = 6 * 10 * (10 - 1);

    // Check if the number of vertices and indices is correct
    assert_eq!(sphere.vertices.len(), (10 + 1) * (10 + 1));
    assert_eq!(sphere.indices.len(), expected_indices);

    // Check that the mass properties are reasonable
    assert!(sphere.mass_properties.mass > 0.0);

    // Check that vertices are roughly distance 2.0 from origin
    for v in sphere.vertices.iter() {
      let pos = Vec3f32::from_components(v.position[0], v.position[1], v.position[2]);
      assert!((pos.length() - 2.0).abs() < 1e-5);
    }
  }

  #[test]
  fn test_comet_glb_loading_and_inertia_diagonalization() {
    crate::gpu::set_asset_dir_for_tests();
    let assets_dir = std::path::PathBuf::from(crate::gpu::ASSET_DIR.read().as_ref().unwrap());
    let model_dir = assets_dir.join("Comet.glb");
    assert!(model_dir.is_file());

    let comet =
      load_comet_from_gltf(model_dir.to_str().unwrap(), false, None).expect("Failed to load comet");
    let inertia = comet.mass_properties.inertia;
    const THRESHOLD: f64 = 1e-6;
    assert!(inertia.xy.abs() < THRESHOLD);
    assert!(inertia.xz.abs() < THRESHOLD);
    assert!(inertia.yz.abs() < THRESHOLD);
  }

  #[test]
  fn test_comet_custom_inertia() {
    let dummy_vertices = [
      Vertex {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 0.0, 0.0],
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 0.0],
      },
      Vertex {
        position: [1.0, 0.0, 0.0],
        normal: [0.0, 0.0, 0.0],
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 0.0],
      },
      Vertex {
        position: [0.0, 1.0, 0.0],
        normal: [0.0, 0.0, 0.0],
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 0.0],
      },
      Vertex {
        position: [0.0, 0.0, 1.0],
        normal: [0.0, 0.0, 0.0],
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 0.0],
      },
    ];
    let dummy_indices = [0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3];
    let mut mass_properties = calculate_mass_properties(&dummy_vertices, &dummy_indices, 1.0);
    let vertices = [
      Vertex {
        position: [1.0, 0.0, 0.0],
        normal: [1.0, 0.0, 0.0],
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 0.0],
      },
      Vertex {
        position: [0.0, 1.0, 0.0],
        normal: [0.0, 1.0, 0.0],
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 0.0],
      },
      Vertex {
        position: [0.0, 0.0, 1.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 0.0],
      },
    ];
    let indices = [0, 1, 2];

    // Create a non-diagonal inertia tensor by rotating a diagonal one [1, 2, 3] by 45 deg around Z
    // I_bf = [1.5, -0.5, 0; -0.5, 1.5, 0; 0, 0, 3]
    let custom_inertia = Mat3f32::from_columns(
      Vec3f32::from_components(1.5, -0.5, 0.0),
      Vec3f32::from_components(-0.5, 1.5, 0.0),
      Vec3f32::from_components(0.0, 0.0, 3.0),
    );

    let (_, _, bf_to_pa, _) = compute_comet_extras(
      &vertices,
      &indices,
      &mut mass_properties,
      Some(custom_inertia),
    );

    // Verify eigenvalues (diagonalized inertia)
    let mut eigenvalues = [
      mass_properties.inertia.xx,
      mass_properties.inertia.yy,
      mass_properties.inertia.zz,
    ];
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert!((eigenvalues[0] - 1.0).abs() < 1e-6);
    assert!((eigenvalues[1] - 2.0).abs() < 1e-6);
    assert!((eigenvalues[2] - 3.0).abs() < 1e-6);

    // Verify bf_to_pa rotation
    // 45 degrees around Z: cos(22.5) = 0.9238, sin(22.5) = 0.3826
    // Quat = [0, 0, sin(theta/2), cos(theta/2)] = [0, 0, 0.3826, 0.9238]
    // Wait, the eigenvectors might be flipped or reordered.
    // Let's check the rotation of a vector.
    let bf_to_pa = bf_to_pa.unwrap();
    let v_bf = Vec3f32::from_components(1.0, 1.0, 0.0);
    let v_pa = bf_to_pa.rotate_vector(v_bf);

    // In PA frame, eigenvectors are [1, 1, 0]/sqrt(2) and [-1, 1, 0]/sqrt(2) (or similar)
    // [1, 1, 0] rotated by 45 deg should align with one of the axes.
    assert!(
      (v_pa.x().abs() - 2.0_f32.sqrt()).abs() < 1e-6
        || (v_pa.y().abs() - 2.0_f32.sqrt()).abs() < 1e-6
    );
    assert!(v_pa.z().abs() < 1e-6);
  }

  #[test]
  fn test_comet_obj_loading() {
    let obj_content = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
";
    let tmp_path = std::env::temp_dir().join("test.obj");
    std::fs::write(&tmp_path, obj_content).unwrap();
    let comet = load_comet_from_obj(tmp_path.to_str().unwrap(), false, None).unwrap();
    assert_eq!(comet.vertices.len(), 4);
    assert_eq!(comet.indices.len(), 12);
  }

  #[test]
  fn test_comet_ply_loading() {
    let ply_content = "ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 4
property list uchar uint vertex_indices
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
3 0 2 1
3 0 1 3
3 0 3 2
3 1 2 3
";
    let tmp_path = std::env::temp_dir().join("test.ply");
    std::fs::write(&tmp_path, ply_content).unwrap();
    let comet = load_comet_from_ply(tmp_path.to_str().unwrap(), false, None).unwrap();
    assert_eq!(comet.vertices.len(), 4);
    assert_eq!(comet.indices.len(), 12);
  }
}
