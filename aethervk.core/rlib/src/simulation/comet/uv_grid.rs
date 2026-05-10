//! uv_grid module.

extern crate alloc;
use crate::simulation::comet::Vertex;
use alloc::vec;
use alloc::vec::Vec;

/// Uniform Spatial Grid backed by a flat CSR (Compressed Sparse Row) array.
#[derive(Clone)]
pub struct UvGrid {
  /// uv space is split into a grid of resolution.x * resolution.y cells
  pub resolution: usize,
  pub min_uv: [f32; 2],
  pub max_uv: [f32; 2],
  /// Prefix sums dictating where a specific cell's triangle list starts
  /// each element here corresponds to a cell
  pub offsets: Vec<usize>,
  /// A flattened continuous array of triangle indices mapping to the Mesh indices
  pub triangle_ids: Vec<u32>,
}

impl UvGrid {
  /// Builds the O(1) lookup grid. A resolution of 64 to 256 is optimal.
  pub fn new(vertices: &[Vertex], indices: &[u32], resolution: usize) -> Self {
    let num_triangles = indices.len() / 3;
    let num_cells = resolution * resolution;

    if num_triangles == 0 || vertices.is_empty() {
      return Self {
        resolution,
        min_uv: [0.0, 0.0],
        max_uv: [1.0, 1.0],
        offsets: vec![0; num_cells + 1],
        triangle_ids: vec![],
      };
    }

    // Measure true UV bounds without assuming standard [0, 1] wrapping
    let mut min_u = f32::MAX;
    let mut max_u = f32::MIN;
    let mut min_v = f32::MAX;
    let mut max_v = f32::MIN;

    for v in vertices {
      min_u = min_u.min(v.uv[0]);
      max_u = max_u.max(v.uv[0]);
      min_v = min_v.min(v.uv[1]);
      max_v = max_v.max(v.uv[1]);
    }

    if max_u <= min_u {
      max_u = min_u + 1.0;
    }
    if max_v <= min_v {
      max_v = min_v + 1.0;
    }

    let inv_w = resolution as f32 / (max_u - min_u);
    let inv_h = resolution as f32 / (max_v - min_v);

    let get_cell = |u: f32, v: f32| -> (usize, usize) {
      let mut x = if u < min_u {
        0
      } else {
        ((u - min_u) * inv_w) as usize
      };
      let mut y = if v < min_v {
        0
      } else {
        ((v - min_v) * inv_h) as usize
      };
      if x >= resolution {
        x = resolution - 1;
      }
      if y >= resolution {
        y = resolution - 1;
      }
      (x, y)
    };

    let get_tri_bounds = |v0: &Vertex, v1: &Vertex, v2: &Vertex| -> (usize, usize, usize, usize) {
      let u_min = v0.uv[0].min(v1.uv[0].min(v2.uv[0]));
      let u_max = v0.uv[0].max(v1.uv[0].max(v2.uv[0]));
      let v_min = v0.uv[1].min(v1.uv[1].min(v2.uv[1]));
      let v_max = v0.uv[1].max(v1.uv[1].max(v2.uv[1]));
      let (min_x, min_y) = get_cell(u_min, v_min);
      let (max_x, max_y) = get_cell(u_max, v_max);
      (min_x, max_x, min_y, max_y)
    };

    // Pass 1: Count overlaps per cell
    let mut counts = vec![0; num_cells];
    for tri_idx in 0..num_triangles {
      let i0 = indices[tri_idx * 3] as usize;
      let i1 = indices[tri_idx * 3 + 1] as usize;
      let i2 = indices[tri_idx * 3 + 2] as usize;
      let (min_x, max_x, min_y, max_y) =
        get_tri_bounds(&vertices[i0], &vertices[i1], &vertices[i2]);

      for y in min_y..=max_y {
        for x in min_x..=max_x {
          counts[y * resolution + x] += 1;
        }
      }
    }

    // Pass 2: Compute exact memory layout via prefix sum
    let mut offsets = vec![0; num_cells + 1];
    for i in 0..num_cells {
      offsets[i + 1] = offsets[i] + counts[i];
    }

    // Pass 3: Fill contiguous index array
    let mut triangle_ids = vec![0; offsets[num_cells]];
    let mut current_offsets = offsets.clone();

    for tri_idx in 0..num_triangles {
      let i0 = indices[tri_idx * 3] as usize;
      let i1 = indices[tri_idx * 3 + 1] as usize;
      let i2 = indices[tri_idx * 3 + 2] as usize;
      let (min_x, max_x, min_y, max_y) =
        get_tri_bounds(&vertices[i0], &vertices[i1], &vertices[i2]);

      for y in min_y..=max_y {
        for x in min_x..=max_x {
          let cell_idx = y * resolution + x;
          triangle_ids[current_offsets[cell_idx]] = tri_idx as u32;
          current_offsets[cell_idx] += 1;
        }
      }
    }

    Self {
      resolution,
      min_uv: [min_u, min_v],
      max_uv: [max_u, max_v],
      offsets,
      triangle_ids,
    }
  }

  /// Queries the mapper for an O(1) interpolated 3D object space position and normal.
  pub fn query(
    &self,
    uv: [f32; 2],
    vertices: &[Vertex],
    indices: &[u32],
  ) -> Option<([f32; 3], [f32; 3])> {
    if uv[0] < self.min_uv[0]
      || uv[0] > self.max_uv[0]
      || uv[1] < self.min_uv[1]
      || uv[1] > self.max_uv[1]
    {
      return None;
    }

    let inv_w = self.resolution as f32 / (self.max_uv[0] - self.min_uv[0]);
    let inv_h = self.resolution as f32 / (self.max_uv[1] - self.min_uv[1]);

    let mut x = if uv[0] < self.min_uv[0] {
      0
    } else {
      ((uv[0] - self.min_uv[0]) * inv_w) as usize
    };
    let mut y = if uv[1] < self.min_uv[1] {
      0
    } else {
      ((uv[1] - self.min_uv[1]) * inv_h) as usize
    };
    if x >= self.resolution {
      x = self.resolution - 1;
    }
    if y >= self.resolution {
      y = self.resolution - 1;
    }

    let cell_idx = y * self.resolution + x;

    // Loop purely over the 1 to ~4 triangles physically present in this UV square
    for i in self.offsets[cell_idx]..self.offsets[cell_idx + 1] {
      let tri_idx = self.triangle_ids[i] as usize;
      let i0 = indices[tri_idx * 3] as usize;
      let i1 = indices[tri_idx * 3 + 1] as usize;
      let i2 = indices[tri_idx * 3 + 2] as usize;

      let v0 = &vertices[i0];
      let v1 = &vertices[i1];
      let v2 = &vertices[i2];

      if let Some([w0, w1, w2]) = barycentric_2d(uv, v0.uv, v1.uv, v2.uv) {
        return Some((
          [
            w0 * v0.position[0] + w1 * v1.position[0] + w2 * v2.position[0],
            w0 * v0.position[1] + w1 * v1.position[1] + w2 * v2.position[1],
            w0 * v0.position[2] + w1 * v1.position[2] + w2 * v2.position[2],
          ],
          [
            w0 * v0.normal[0] + w1 * v1.normal[0] + w2 * v2.normal[0],
            w0 * v0.normal[1] + w1 * v1.normal[1] + w2 * v2.normal[1],
            w0 * v0.normal[2] + w1 * v1.normal[2] + w2 * v2.normal[2],
          ],
        ));
      }
    }
    None
  }
}

/// Computes Barycentric coordinates (w0, w1, w2) for point `p` inside triangle `a, b, c`.
#[inline]
fn barycentric_2d(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> Option<[f32; 3]> {
  let det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);

  // Reject degenerate (zero-area) triangles seamlessly
  if det > -1e-8 && det < 1e-8 {
    return None;
  }

  let inv_det = 1.0 / det;
  let w1 = ((p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])) * inv_det;
  let w2 = ((b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])) * inv_det;
  let w0 = 1.0 - w1 - w2;

  // A small negative epsilon safely captures floating point hits exactly on boundaries
  if w0 >= -1e-5 && w1 >= -1e-5 && w2 >= -1e-5 {
    Some([w0, w1, w2])
  } else {
    None
  }
}
