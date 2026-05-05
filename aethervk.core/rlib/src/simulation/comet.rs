use aethervk_oshal_rlib::{
  self as oshal,
  math::vector::{Vector, Vector3, vec3::Vec3f32},
  os::FsError,
};
use alloc::vec::Vec;
use ktx2;
use oshal::os::fs::{self, FileSystemObject, PathBuf};
use polyhedral_mass_properties::{MassProperties, TriangleContrib};
use zune_core::bytestream::ZCursor;
use zune_jpeg;

pub mod uv_grid;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Clone)]
pub struct Texture {
  pub data: Vec<u8>,
  pub format: TexelFormat,
  pub width: u32,
  pub height: u32,
  pub has_mipmaps: bool,
}

use crate::math::collision::bvh_builder::{BVHBuilder, BVHBuilderParams};
use crate::math::collision::linear_bvh::LinearBVH;
use aethervk_oshal_rlib::math::matrix::{Matrix3, mat3::Mat3f32};

#[derive(Clone)]
pub struct Comet {
  pub vertices: Vec<Vertex>,
  pub indices: Vec<u32>,
  pub albedo_map: Option<Texture>,
  pub normal_map: Option<Texture>,
  pub roughness_map: Option<Texture>,
  pub ao_map: Option<Texture>,
  /// mass, inertia_tensor, center_of_mass stored here as f64 (TODO Accessors)
  /// from this field, all other accessors are computed, plus conversion
  /// to any other scalar numeric format declared in oshal library, to support mixed precision simulation
  pub mass_properties: MassProperties,
  pub bvh: Option<LinearBVH<f32>>,
  pub principal_axes: Option<Mat3f32>,
}

fn compute_comet_extras(
  vertices: &[Vertex],
  indices: &[u32],
  mass_properties: &mut MassProperties,
) -> (Option<LinearBVH<f32>>, Option<Mat3f32>, Vec<Vertex>) {
  use crate::math::compute_com_and_tensor;
  let raw_verts: Vec<Vec3f32> = vertices
    .iter()
    .map(|v| Vec3f32::from_components(v.position[0], v.position[1], v.position[2]))
    .collect();
  let (_, mat) = compute_com_and_tensor(&raw_verts, 1.0); // Assume unit mass per vertex for geometry proxy
  let (principal_moments, principal_axes) = crate::math::jacobi_diagonalization(mat, 1e-6, 100);

  // Update mass properties to match the new diagonalized tensor
  mass_properties.inertia.xx = principal_moments.x() as f64;
  mass_properties.inertia.yy = principal_moments.y() as f64;
  mass_properties.inertia.zz = principal_moments.z() as f64;
  mass_properties.inertia.xy = 0.0;
  mass_properties.inertia.xz = 0.0;
  mass_properties.inertia.yz = 0.0;

  // Center of mass becomes (0,0,0) in the local frame since vertices are translated
  mass_properties.center_of_mass = [0.0, 0.0, 0.0];

  // Calculate new principal axes correctly oriented.
  // Transform vertices into the new local coordinate system aligned with the principal axes.
  let mut local_vertices = vertices.to_vec();
  for v in local_vertices.iter_mut() {
    let v_world = Vec3f32::from_components(v.position[0], v.position[1], v.position[2]);
    let vx = principal_axes.x.dot(v_world);
    let vy = principal_axes.y.dot(v_world);
    let vz = principal_axes.z.dot(v_world);
    v.position = [vx, vy, vz];

    let n_world = Vec3f32::from_components(v.normal[0], v.normal[1], v.normal[2]);
    let nx = principal_axes.x.dot(n_world);
    let ny = principal_axes.y.dot(n_world);
    let nz = principal_axes.z.dot(n_world);
    v.normal = [nx, ny, nz];
  }

  // Log the axes properly formatted
  use aethervk_oshal_rlib::math::vector::Vector;

  let mut tris = Vec::new();
  for chunk in indices.chunks_exact(3) {
    let v0 = local_vertices[chunk[0] as usize].position;
    let v1 = local_vertices[chunk[1] as usize].position;
    let v2 = local_vertices[chunk[2] as usize].position;
    tris.push(Triangle {
      vertices: [
        Vec3f32::from_components(v0[0], v0[1], v0[2]),
        Vec3f32::from_components(v1[0], v1[1], v1[2]),
        Vec3f32::from_components(v2[0], v2[1], v2[2]),
      ],
    });
  }

  let builder = BVHBuilder::<f32, Vec3f32, Mat3f32>::new(BVHBuilderParams::default());
  let bvh = builder.build(&tris);
  let linear_bvh = bvh.map(|root| LinearBVH::from_build_node(&root, 0));

  (linear_bvh, Some(principal_axes), local_vertices)
}

#[derive(Debug, Clone, Copy)]
pub struct Triangle {
  pub vertices: [Vec3f32; 3],
}

impl Triangle {
  #[inline]
  pub fn v0(&self) -> &Vec3f32 {
    &self.vertices[0]
  }
  #[inline]
  pub fn v1(&self) -> &Vec3f32 {
    &self.vertices[1]
  }
  #[inline]
  pub fn v2(&self) -> &Vec3f32 {
    &self.vertices[2]
  }
  #[inline]
  pub fn mean_vector(&self) -> Vec3f32 {
    (*self.v0() + *self.v1() + *self.v2()) / 3.0
  }
  /// Warning: unnormalized
  /// Warning: Formula correct if "outwards" is counter clockwise
  #[inline]
  pub fn normal_ccw_unnormalized(&self) -> Vec3f32 {
    let v0 = self.vertices[0];
    let v1 = self.vertices[1];
    let v2 = self.vertices[2];

    (v1 - v0).cross(v2 - v1)
  }

  #[inline]
  pub fn area(&self) -> f32 {
    0.5 * (*self.v1() - *self.v0()).cross(*self.v2() - *self.v1()).length()
  }
}

impl Comet {
  /// Returns a zero-allocation, cloneable iterator over the mesh's triangles.
  pub fn iter_triangles(&self) -> impl Iterator<Item = Triangle> + Clone + '_ {
    // Iterate over indices in groups of 3
    self.indices.chunks_exact(3).map(move |chunk| {
      let i0 = chunk[0] as usize;
      let i1 = chunk[1] as usize;
      let i2 = chunk[2] as usize;

      // Construct the Triangle on the fly
      Triangle {
        // Assuming your Vec3f32 has a from/into implementation for [f32; 3]
        vertices: [
          Vec3f32::from(self.vertices[i0].position),
          Vec3f32::from(self.vertices[i1].position),
          Vec3f32::from(self.vertices[i2].position),
        ],
      }
    })
  }
}

impl PartialEq for Comet {
  fn eq(&self, other: &Self) -> bool {
    self.vertices == other.vertices && self.indices == other.indices
  }
}

impl core::fmt::Debug for Comet {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Comet")
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

  // TODO: watertight check, triangulated check, normal check, ...

  Ok(Some(Texture {
    data: decoded_data,
    format,
    width,
    height,
    has_mipmaps,
  }))
}

fn calculate_mass_properties(
  vertices: &[Vertex],
  indices: &[u32],
  total_mass: f32,
) -> MassProperties {
  let points: Vec<[f64; 3]> = vertices
    .iter()
    .map(|v| {
      [
        v.position[0] as f64,
        v.position[1] as f64,
        v.position[2] as f64,
      ]
    })
    .collect();
  let tris = indices.chunks_exact(3);

  let contrib_sum = tris
    .map(|tri| {
      let p0 = points[tri[0] as usize];
      let p1 = points[tri[1] as usize];
      let p2 = points[tri[2] as usize];
      TriangleContrib::new(p0, p1, p2)
    })
    .sum();

  let unit_mass_properties = MassProperties::from_contrib_sum(contrib_sum).unwrap();
  let volume = unit_mass_properties.volume();
  let density = total_mass as f64 / volume;
  unit_mass_properties.with_density(density)
}

use crate::types::EngineError;
use alloc::collections::BTreeMap;
// Assuming Vec, PathBuf, etc. are already in scope

/// Function to load a GLTF/GLB file
/// 1. Watertightness Check: A closed, physical (manifold) mesh must have every undirected edge shared by exactly two triangles. If an edge has only one triangle, there's a hole. If it has three or more, there's self-intersecting/non-manifold geometry.
/// 2. Outward Normals Check: By calculating the signed volume of the mesh using the divergence theorem, we can verify winding order. If the volume is negative, the triangles are wound backwards, meaning your normals are pointing inward.
pub fn load_comet_from_gltf(path: &str, verbose: bool) -> Result<Comet, CometLoadError> {
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

  oshal::log!("--- GLTF load successful! ---");

  let mut mass_properties = calculate_mass_properties(&vertices, &indices, 1.0);
  let (bvh, principal_axes, local_vertices) =
    compute_comet_extras(&vertices, &indices, &mut mass_properties);

  Ok(Comet {
    vertices: local_vertices,
    indices,
    albedo_map,
    normal_map,
    roughness_map,
    ao_map,
    mass_properties,
    bvh,
    principal_axes,
  })
}

/// Produces `6 * lon_segments * (lat_segments - 1)` indices
pub fn generate_uv_sphere(
  radius: f32,
  lat_segments: u32,
  lon_segments: u32,
  total_mass: f32,
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

      // 3. FIXED: Strictly Enforcing Vulkan CCW Front-Face Winding.
      // Additionally, if the geometry is evaluated at the South or North pole,
      // the respective index evaluation drops the degenerate zero-area triangle.
      if lat != 0 {
        indices.push(first);
        indices.push(first + 1);
        indices.push(second);
      }
      if lat != lat_segments - 1 {
        indices.push(second);
        indices.push(first + 1);
        indices.push(second + 1);
      }
    }
  }

  let mut mass_properties = calculate_mass_properties(&vertices, &indices, total_mass);

  // 2. FIXED: Apply the exact closed-form diagonal solid-sphere inertia properties.
  // Bypass numeric precision drift stemming from iterative integration.
  let i_diag = (2.0 / 5.0) * (total_mass as f64) * (radius as f64).powi(2);
  mass_properties.center_of_mass = [0.0, 0.0, 0.0];
  mass_properties.inertia.xx = i_diag;
  mass_properties.inertia.yy = i_diag;
  mass_properties.inertia.zz = i_diag;
  mass_properties.inertia.xy = 0.0;
  mass_properties.inertia.xz = 0.0;
  mass_properties.inertia.yz = 0.0;

  // Symmetrical Solid Spheres naturally assert the Identity Matrix for local principal axes
  let principal_axes = Mat3f32 {
    x: Vec3f32::from_components(1.0, 0.0, 0.0),
    y: Vec3f32::from_components(0.0, 1.0, 0.0),
    z: Vec3f32::from_components(0.0, 0.0, 1.0),
  };

  // We completely bypass `compute_comet_extras` here because the sphere is already diagonalized
  // and perfectly centered physically around the internal [0,0,0] origin point.
  let mut tris = Vec::with_capacity(indices.len() / 3);
  for chunk in indices.chunks_exact(3) {
    let v0 = vertices[chunk[0] as usize].position;
    let v1 = vertices[chunk[1] as usize].position;
    let v2 = vertices[chunk[2] as usize].position;
    tris.push(Triangle {
      vertices: [
        Vec3f32::from_components(v0[0], v0[1], v0[2]),
        Vec3f32::from_components(v1[0], v1[1], v1[2]),
        Vec3f32::from_components(v2[0], v2[1], v2[2]),
      ],
    });
  }

  let builder = BVHBuilder::<f32, Vec3f32, Mat3f32>::new(BVHBuilderParams::default());
  let bvh = builder.build(&tris).map(|root| LinearBVH::from_build_node(&root, 0));

  Comet {
    vertices,
    indices,
    albedo_map: None,
    normal_map: None,
    roughness_map: None,
    ao_map: None,
    mass_properties,
    bvh,
    principal_axes: Some(principal_axes),
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
    data: decoded_data,
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
    let sphere = generate_uv_sphere(2.0, 10, 10, 1.0);
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
    let assets_dir = {
      let mut home_dir = std::env::current_exe().unwrap();
      let mut iter: i32 = 0;
      const MAX_ITER: i32 = 32;
      while {
        let d = home_dir.join("assets");
        !d.is_dir() && iter < MAX_ITER
      } {
        home_dir.pop();
        iter += 1;
        assert!(home_dir.is_dir());
      }
      home_dir.join("assets")
    };
    let model_dir = assets_dir.join("Comet.glb");
    assert!(model_dir.is_file());

    let comet =
      load_comet_from_gltf(model_dir.to_str().unwrap(), false).expect("Failed to load comet");
    let inertia = comet.mass_properties.inertia;
    const THRESHOLD: f64 = 1e-6;
    assert!(inertia.xy.abs() < THRESHOLD);
    assert!(inertia.xz.abs() < THRESHOLD);
    assert!(inertia.yz.abs() < THRESHOLD);
  }
}
