//! CPU-side preview generation for imported assets (Imports tab / Settings tab pickers).
//!
//! Everything here runs on the logic thread during import and never touches the GPU, so it is
//! deterministic and unit-testable without Vulkan. Output is always tightly packed RGBA8.

use alloc::vec::Vec;

use crate::simulation::comet::{TexelFormat, Texture, Vertex};

/// Default edge length (pixels) of generated thumbnails.
pub const THUMBNAIL_SIZE: u32 = 128;

/// Tightly packed RGBA8 preview image.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Thumbnail {
  pub width: u32,
  pub height: u32,
  pub rgba: Vec<u8>,
}

impl Thumbnail {
  fn transparent(width: u32, height: u32) -> Self {
    Self {
      width,
      height,
      rgba: alloc::vec![0; (width * height * 4) as usize],
    }
  }

  /// Pixel accessor (tests / diagnostics).
  pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * self.width + x) * 4) as usize;
    [
      self.rgba[i],
      self.rgba[i + 1],
      self.rgba[i + 2],
      self.rgba[i + 3],
    ]
  }

  /// Number of pixels with non-zero alpha.
  pub fn opaque_pixel_count(&self) -> usize {
    self.rgba.chunks_exact(4).filter(|p| p[3] != 0).count()
  }
}

/// Renders a shaded silhouette of a mesh.
///
/// Orthographic camera at a fixed 3/4 view, framed by the bounding sphere (`center`, `radius`),
/// z-buffered rasterisation, Lambert shading with face normals. Triangles are not culled and
/// normals are flipped towards the viewer, so inverted winding still produces a sensible
/// preview. Background stays fully transparent.
pub fn mesh_thumbnail(
  vertices: &[Vertex],
  indices: &[u32],
  center: [f32; 3],
  radius: f32,
  size: u32,
) -> Thumbnail {
  let mut out = Thumbnail::transparent(size, size);
  if vertices.is_empty() || indices.len() < 3 || !(radius > 0.0) || size == 0 {
    return out;
  }

  // View rotation: yaw around +Z (the comet body frame's pole), then pitch around +X.
  let (sy, cy) = 35.0_f32.to_radians().sin_cos();
  let (sp, cp) = (-60.0_f32).to_radians().sin_cos();
  let to_view = |p: [f32; 3]| -> [f32; 3] {
    let x = p[0] - center[0];
    let y = p[1] - center[1];
    let z = p[2] - center[2];
    // yaw
    let x1 = cy * x - sy * y;
    let y1 = sy * x + cy * y;
    // pitch
    let y2 = cp * y1 - sp * z;
    let z2 = sp * y1 + cp * z;
    [x1, y2, z2]
  };

  let half = size as f32 * 0.5;
  let scale = half * 0.92 / radius;
  // Screen space: x right, y down; view-space +z points towards the viewer.
  let projected: Vec<[f32; 3]> = vertices
    .iter()
    .map(|v| {
      let p = to_view(v.position);
      [half + p[0] * scale, half - p[1] * scale, p[2]]
    })
    .collect();

  let light = normalize([-0.4, 0.5, 0.75]);
  let mut depth = alloc::vec![f32::NEG_INFINITY; (size * size) as usize];

  for tri in indices.chunks_exact(3) {
    let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
    if ia >= projected.len() || ib >= projected.len() || ic >= projected.len() {
      continue;
    }
    let (a, b, c) = (projected[ia], projected[ib], projected[ic]);

    // Face normal in view space (screen y is flipped, so recompute from view positions).
    let va = to_view(vertices[ia].position);
    let vb = to_view(vertices[ib].position);
    let vc = to_view(vertices[ic].position);
    let mut n = cross(sub(vb, va), sub(vc, va));
    if n[2] < 0.0 {
      n = [-n[0], -n[1], -n[2]];
    }
    let n = normalize(n);
    let lambert = (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]).max(0.0);
    let shade = 0.18 + 0.82 * lambert;
    let color = [
      (196.0 * shade) as u8,
      (192.0 * shade) as u8,
      (186.0 * shade) as u8,
    ];

    let area = edge(a, b, c);
    if area.abs() < 1e-12 {
      continue;
    }
    let min_x = a[0].min(b[0]).min(c[0]).floor().max(0.0) as i32;
    let max_x = a[0].max(b[0]).max(c[0]).ceil().min(size as f32 - 1.0) as i32;
    let min_y = a[1].min(b[1]).min(c[1]).floor().max(0.0) as i32;
    let max_y = a[1].max(b[1]).max(c[1]).ceil().min(size as f32 - 1.0) as i32;

    for py in min_y..=max_y {
      for px in min_x..=max_x {
        let p = [px as f32 + 0.5, py as f32 + 0.5, 0.0];
        let w0 = edge(b, c, p) / area;
        let w1 = edge(c, a, p) / area;
        let w2 = edge(a, b, p) / area;
        if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
          continue;
        }
        let z = w0 * a[2] + w1 * b[2] + w2 * c[2];
        let di = (py as u32 * size + px as u32) as usize;
        if z <= depth[di] {
          continue;
        }
        depth[di] = z;
        let i = di * 4;
        out.rgba[i] = color[0];
        out.rgba[i + 1] = color[1];
        out.rgba[i + 2] = color[2];
        out.rgba[i + 3] = 255;
      }
    }
  }
  out
}

/// Box-filtered downscale of a texture so that its longest edge is at most `max_size`
/// (aspect ratio preserved, never upscaled). Uncompressed 8-bit formats only; compressed or
/// unknown formats yield a checkerboard placeholder of `max_size × max_size`.
pub fn texture_thumbnail(texture: &Texture, max_size: u32) -> Thumbnail {
  let channels = match texture.format {
    TexelFormat::R8_UNORM => 1usize,
    TexelFormat::R8G8_UNORM => 2,
    TexelFormat::R8G8B8_UNORM => 3,
    TexelFormat::R8G8B8A8_UNORM => 4,
    _ => return placeholder_thumbnail(max_size),
  };
  let (w, h) = (texture.width as usize, texture.height as usize);
  if w == 0 || h == 0 || texture.data.len() < w * h * channels || max_size == 0 {
    return placeholder_thumbnail(max_size);
  }

  let longest = w.max(h);
  let (tw, th) = if longest <= max_size as usize {
    (w, h)
  } else {
    (
      ((w * max_size as usize) / longest).max(1),
      ((h * max_size as usize) / longest).max(1),
    )
  };

  let src = &texture.data[..];
  let mut out = Thumbnail::transparent(tw as u32, th as u32);
  for ty in 0..th {
    let y0 = ty * h / th;
    let y1 = ((ty + 1) * h / th).max(y0 + 1);
    for tx in 0..tw {
      let x0 = tx * w / tw;
      let x1 = ((tx + 1) * w / tw).max(x0 + 1);
      let mut acc = [0u32; 4];
      for y in y0..y1 {
        let row = y * w;
        for x in x0..x1 {
          let i = (row + x) * channels;
          let px = match channels {
            1 => [src[i], src[i], src[i], 255],
            2 => [src[i], src[i + 1], 0, 255],
            3 => [src[i], src[i + 1], src[i + 2], 255],
            _ => [src[i], src[i + 1], src[i + 2], src[i + 3]],
          };
          for c in 0..4 {
            acc[c] += px[c] as u32;
          }
        }
      }
      let n = ((y1 - y0) * (x1 - x0)) as u32;
      let o = (ty * tw + tx) * 4;
      for c in 0..4 {
        out.rgba[o + c] = (acc[c] / n) as u8;
      }
    }
  }
  out
}

/// Grey checkerboard used for textures whose texels cannot be previewed on the CPU
/// (block-compressed formats).
pub fn placeholder_thumbnail(size: u32) -> Thumbnail {
  let mut out = Thumbnail::transparent(size, size);
  let cell = (size / 8).max(1);
  for y in 0..size {
    for x in 0..size {
      let dark = ((x / cell) + (y / cell)) % 2 == 0;
      let v = if dark { 90 } else { 150 };
      let i = ((y * size + x) * 4) as usize;
      out.rgba[i..i + 4].copy_from_slice(&[v, v, v, 255]);
    }
  }
  out
}

#[inline(always)]
fn edge(a: [f32; 3], b: [f32; 3], p: [f32; 3]) -> f32 {
  (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

#[inline(always)]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
  [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

#[inline(always)]
fn normalize(v: [f32; 3]) -> [f32; 3] {
  let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
  if l > 0.0 {
    [v[0] / l, v[1] / l, v[2] / l]
  } else {
    [0.0, 0.0, 1.0]
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::simulation::comet::generate_uv_sphere;

  #[test]
  fn sphere_thumbnail_is_centered_round_and_transparent_outside() {
    let sphere = generate_uv_sphere(3.0, 32, 32, 1.0, false);
    let t = mesh_thumbnail(&sphere.vertices, &sphere.indices, [0.0; 3], 3.0, 128);
    assert_eq!((t.width, t.height), (128, 128));
    assert_eq!(t.rgba.len(), 128 * 128 * 4);

    // Corners are outside the inscribed disc.
    for (x, y) in [(0, 0), (127, 0), (0, 127), (127, 127)] {
      assert_eq!(t.pixel(x, y)[3], 0, "corner ({x},{y}) must be transparent");
    }
    // Center is covered and lit.
    let c = t.pixel(64, 64);
    assert_eq!(c[3], 255);
    assert!(c[0] > 30, "center should not be black: {c:?}");

    // Disc of radius 0.92*64 ≈ 58.9 px → area ≈ 10 900 px; allow tessellation slack.
    let covered = t.opaque_pixel_count() as f32;
    let expected = core::f32::consts::PI * (64.0 * 0.92f32).powi(2);
    assert!(
      (covered - expected).abs() / expected < 0.06,
      "coverage {covered} vs expected {expected}"
    );
  }

  #[test]
  fn thumbnail_is_framed_by_bounding_sphere_not_origin() {
    let mut sphere = generate_uv_sphere(1.0, 16, 16, 1.0, false);
    for v in &mut sphere.vertices {
      v.position[0] += 100.0;
    }
    let t = mesh_thumbnail(
      &sphere.vertices,
      &sphere.indices,
      [100.0, 0.0, 0.0],
      1.0,
      64,
    );
    assert_eq!(t.pixel(32, 32)[3], 255);
  }

  #[test]
  fn inverted_winding_still_renders() {
    let sphere = generate_uv_sphere(1.0, 16, 16, 1.0, true);
    let t = mesh_thumbnail(&sphere.vertices, &sphere.indices, [0.0; 3], 1.0, 64);
    assert!(t.opaque_pixel_count() > 1000);
    assert!(t.pixel(32, 32)[0] > 30);
  }

  #[test]
  fn degenerate_inputs_give_transparent_thumbnail() {
    let t = mesh_thumbnail(&[], &[], [0.0; 3], 1.0, 16);
    assert_eq!(t.opaque_pixel_count(), 0);
  }

  fn rgba_texture(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Texture {
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
      for x in 0..w {
        data.extend_from_slice(&f(x, y));
      }
    }
    Texture {
      data: data.into(),
      format: TexelFormat::R8G8B8A8_UNORM,
      width: w,
      height: h,
      has_mipmaps: false,
    }
  }

  #[test]
  fn texture_downscale_preserves_aspect_and_averages() {
    let tex = rgba_texture(512, 256, |x, _| {
      if x < 256 {
        [255, 0, 0, 255]
      } else {
        [0, 0, 255, 255]
      }
    });
    let t = texture_thumbnail(&tex, 128);
    assert_eq!((t.width, t.height), (128, 64));
    assert_eq!(t.pixel(10, 10), [255, 0, 0, 255]);
    assert_eq!(t.pixel(120, 50), [0, 0, 255, 255]);
  }

  #[test]
  fn small_texture_is_not_upscaled() {
    let tex = rgba_texture(16, 8, |_, _| [1, 2, 3, 4]);
    let t = texture_thumbnail(&tex, 128);
    assert_eq!((t.width, t.height), (16, 8));
    assert_eq!(t.pixel(3, 3), [1, 2, 3, 4]);
  }

  #[test]
  fn grayscale_texture_expands_to_rgba() {
    let tex = Texture {
      data: alloc::vec![200u8; 4 * 4].into(),
      format: TexelFormat::R8_UNORM,
      width: 4,
      height: 4,
      has_mipmaps: false,
    };
    assert_eq!(
      texture_thumbnail(&tex, 128).pixel(0, 0),
      [200, 200, 200, 255]
    );
  }

  #[test]
  fn compressed_texture_uses_placeholder() {
    let tex = Texture {
      data: alloc::vec![0u8; 16].into(),
      format: TexelFormat::BC7_UNORM_BLOCK,
      width: 4,
      height: 4,
      has_mipmaps: false,
    };
    let t = texture_thumbnail(&tex, 64);
    assert_eq!((t.width, t.height), (64, 64));
    assert_eq!(t.opaque_pixel_count(), 64 * 64);
    assert_eq!(t, placeholder_thumbnail(64));
  }
}
