// this is a no_std module. if we need to use os-specific stuff, eg objc2-* libraries, windows crate, libc crate,
// we can go through the oshal rlib, expose a function/struct, and then come back here.
// since oshal and core are then exposed to cdylibs, each with their own instance of an Allocator,
// we should never take ownership or return allocated stuff to cdylib interface.
use alloc::vec::Vec;
use aethervk_oshal_rlib as oshal;
use oshal::os::{
  fs::{self, FileSystemObject, PathBuf},
};
use bitflags::bitflags;
use ktx2;
use polyhedral_mass_properties::{MassProperties, TriangleContrib};
use zune_core::bytestream::ZCursor;
use zune_jpeg;

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

pub struct Texture {
  pub data: Vec<u8>,
  pub format: TexelFormat,
  pub width: u32,
  pub height: u32,
  pub has_mipmaps: bool,
}

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
  mass_properties: MassProperties,
}

impl Comet {
  pub fn texture_flags(&self) -> TextureFlags {
    let mut flags = TextureFlags::empty();
    if self.albedo_map.is_some() {
      flags |= TextureFlags::ALBEDO;
    }
    if self.normal_map.is_some() {
      flags |= TextureFlags::NORMAL;
    }
    if self.roughness_map.is_some() {
      flags |= TextureFlags::ROUGHNESS;
    }
    if self.ao_map.is_some() {
      flags |= TextureFlags::AO;
    }
    flags
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

impl From<fs::FsError> for CometLoadError {
  fn from(_: fs::FsError) -> Self {
    CometLoadError::IoError
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
      let data = fs::read(path.as_ref()).map_err(|_| CometLoadError::TextureNotFound)?;
      (data, mime_type, Some(path))
    }
  };

  let (decoded_data, format, width, height, has_mipmaps) = if let Some(mime) = mime_type {
    match mime {
      "image/jpeg" => {
        let mut decoder = zune_jpeg::JpegDecoder::new(ZCursor::new(&encoded_data));
        let info = decoder.info().ok_or(CometLoadError::ImageDecodingError)?;
        let data = decoder
          .decode()
          .map_err(|_| CometLoadError::ImageDecodingError)?;
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
    if path
      .extension()
      .map(|s| s.as_ref() == "ktx2")
      .unwrap_or(false)
    {
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

use alloc::collections::BTreeMap;
// Assuming Vec, PathBuf, etc. are already in scope

/// Function to load a GLTF/GLB file
/// 1. Watertightness Check: A closed, physical (manifold) mesh must have every undirected edge shared by exactly two triangles. If an edge has only one triangle, there's a hole. If it has three or more, there's self-intersecting/non-manifold geometry.
/// 2. Outward Normals Check: By calculating the signed volume of the mesh using the divergence theorem, we can verify winding order. If the volume is negative, the triangles are wound backwards, meaning your normals are pointing inward.
pub fn load_comet_from_gltf(path: &str) -> Result<Comet, CometLoadError> {
  oshal::log!("--- Starting GLTF load for: {} ---", path);

  let mut path_buf = PathBuf::new();
  path_buf.push(path);

  if !path_buf.is_file() {
    oshal::log!("ERROR: File not found at path: {}", path);
    return Err(CometLoadError::PathNotFound);
  }

  let base_path = path_buf.parent().unwrap_or_else(PathBuf::new);
  let data = fs::read(path_buf.as_ref())?;

  oshal::log!(
    "File read successfully ({} bytes). Parsing GLTF...",
    data.len()
  );
  let gltf = gltf::Gltf::from_slice(&data).map_err(|e| {
    oshal::log!("ERROR: GLTF Parse failed: {:?}", e);
    CometLoadError::GltfImportError(e)
  })?;

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

    let uvs = reader
      .read_tex_coords(0)
      .map(|v| v.into_f32())
      .ok_or_else(|| {
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
      oshal::log!(
        "  ERROR: Mesh is not watertight! Edge ({}, {}) is shared by {} triangles (expected 2).",
        edge.0,
        edge.1,
        count
      );
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

  let mass_properties = calculate_mass_properties(&vertices, &indices, 1.0);

  Ok(Comet {
    vertices,
    indices,
    albedo_map,
    normal_map,
    roughness_map,
    ao_map,
    mass_properties,
  })
}

bitflags! {
  #[repr(C)]
  #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
  pub struct TextureFlags: u32 {
    const ALBEDO    = 1 << 0;
    const NORMAL    = 1 << 1;
    const ROUGHNESS = 1 << 2;
    const AO        = 1 << 3;
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PushConstants {
  pub model_view_proj: [[f32; 4]; 4],
  pub model: [[f32; 4]; 4],
  pub sun_dir: [f32; 3],
  pub texture_flags: TextureFlags,
  pub sun_color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CometSpecializationConstants {
  pub base_albedo_r: f32,
  pub base_albedo_g: f32,
  pub base_albedo_b: f32,
  pub base_roughness: f32,
  pub base_ao: f32,
}

impl Default for CometSpecializationConstants {
  fn default() -> Self {
    Self {
      base_albedo_r: 0.04,
      base_albedo_g: 0.04,
      base_albedo_b: 0.04,
      base_roughness: 0.9,
      base_ao: 1.0,
    }
  }
}

// this is a no_std module. if we need to use os-specific stuff, eg objc2-* libraries, windows crate, libc crate,
// we can go through the oshal rlib, expose a function/struct, and then come back here.
// since oshal and core are then exposed to cdylibs, each with their own instance of an Allocator,
// we should never take ownership or return allocated stuff to cdylib interface.

/*
 *  Mesh reading:
- take a gltf directory path
- ensure path exists and that associated texture resources exist
- reject file if

-     1. Size mesh + textures too big
-     2. Any animation/morph targets present
-     3. More than one mesh present
-     4. Mesh is not watertight
-     4.1 Triangularization (or reject if not triangularized)
-     5. Any geometric normal, whether for faces or for vertex, is converted by corner (meaning attributes are stored per domain like blender)
-     textures should be associated to a given channel, which is remapped to a semantic meaning according to Oren Nayar model
- compute, given dimension of spherical bounding box and total mass, per vertex mass and inertia tensor
  - https://www.cs.upc.edu/~virtual/SGI/docs/3.%20Further%20Reading/Fast%20and%20accurate%20computation%20of%20polyhedral%20mass%20properties.pdf
- diagonalize inertia tensor and transform object space accordingly

We also need a Render pass abstraction to work with  frame+render path

Render pass is created and cached given a render operation description. It doesn't match if an output attachment format (swapchain image) changes.
Since we suppose that mesh fits as a whole, we can write a shared host side struct immediately shared by all render backends
- Windows: IOCP via windows crate
- Linux: libc + epoll / io_uring
- MacOS: ?

Once mesh parsing and transformation of object space is done, next is
- Compute AABB and Bounding Volume Hierarchy in object space through the compute engine interface,
  in such a way that each node stores the primitive span it references, such that, when the rigidbody is
  transformed, we can cheaply recompute AABB for each node (or a better approach I'm not aware of)

- the resulting struct is the "host side" backing, over which we'll compute VkBuffers and VkImages with their
associated descriptor sets,
- this means that there should be a trait "Renderable" which outputs an associated type which is the Backend specific
- this means that both ComputeFrontend and RenderFrontend should have a AcceptRenderable trait/function to register/update their renderable state (eg transform or VkBuffer, or change texture)

The following shaders will be used to render the rigidbody/comet body loaded through GLTF with this algorithm:

// --- comet.frag ---
DO NOT DELETE THESE!
 */
