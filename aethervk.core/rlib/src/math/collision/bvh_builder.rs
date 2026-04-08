//! BVH Builder module
//!
//! Provides algorithms to build Bounding Volume Hierarchies using Top-down approach
//! with Surface Area Heuristic (SAH), supporting multiple bounding types.

use alloc::{boxed::Box, vec::Vec};
use aethervk_oshal_rlib::math::{
  floating::{FloatBits, FloatOps},
  matrix::{Matrix3, mat3::Mat3f32},
  vector::{Vector, Vector3, vec3::Vec3f32},
  FloatLike,
};

use crate::{
  math::collision::bounds::{AABB, BS, OBB},
  simulation::comet::Triangle,
};

#[derive(Debug, Clone)]
pub enum BoundNode<S, V, M>
where
  M: Matrix3<Scalar = S, Vector = V>,
  V: Vector3<Scalar = S> + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  S: FloatLike
    + FloatOps
    + From<f32>
    + From<i32>
    + core::ops::Mul<V, Output = V>
    + core::ops::Mul<M, Output = M>,
{
  AABB(AABB<V>),
  OBB(OBB<S, V, M>),
  BS(BS<V>),
}

#[derive(Debug, Clone)]
pub struct BVHNode<S, V, M>
where
  M: Matrix3<Scalar = S, Vector = V>,
  V: Vector3<Scalar = S> + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  S: FloatLike
    + FloatOps
    + From<f32>
    + From<i32>
    + core::ops::Mul<V, Output = V>
    + core::ops::Mul<M, Output = M>,
{
  pub bound: BoundNode<S, V, M>,
  pub left: Option<Box<BVHNode<S, V, M>>>,
  pub right: Option<Box<BVHNode<S, V, M>>>,
  // Indices into the original triangle slice
  pub primitive_indices: Vec<usize>,
}

pub struct BVHBuilderParams {
  pub aabb_levels: usize,
  pub max_primitives_per_node: usize,
  pub bin_count: usize,
}

impl BVHBuilderParams {
  pub fn aabb_levels(&self, value: usize) -> Self {
    Self {
      aabb_levels: value,
      ..*self
    }
  }

  pub fn max_primitives_per_node(&self, value: usize) -> Self {
    Self {
      max_primitives_per_node: value,
      ..*self
    }
  }

  pub fn bin_count(&self, value: usize) -> Self {
    Self {
      bin_count: value,
      ..*self
    }
  }
}

impl Default for BVHBuilderParams {
  fn default() -> Self {
    Self {
      aabb_levels: 4,
      max_primitives_per_node: 4,
      bin_count: 16,
    }
  }
}

pub struct BVHBuilder<S, V, M>
where
  M: Matrix3<Scalar = S, Vector = V> + From<Mat3f32>,
  V: Vector3<Scalar = S>
    + From<Vec3f32>
    + Into<Vec3f32>
    + From<[V::Scalar; 3]>
    + Into<[V::Scalar; 3]>,
  S: FloatLike
    + FloatOps
    + FloatBits
    + From<f32>
    + From<i32>
    + core::ops::Mul<V, Output = V>
    + core::ops::Mul<M, Output = M>,
{
  params: BVHBuilderParams,
  _phantom: core::marker::PhantomData<(S, V, M)>,
}

impl<S, V, M> BVHBuilder<S, V, M>
where
  M: Matrix3<Scalar = S, Vector = V> + From<Mat3f32>,
  V: Vector3<Scalar = S>
    + From<Vec3f32>
    + Into<Vec3f32>
    + From<[V::Scalar; 3]>
    + Into<[V::Scalar; 3]>,
  S: FloatLike
    + FloatOps
    + FloatBits
    + From<f32>
    + From<i32>
    + core::ops::Mul<V, Output = V>
    + core::ops::Mul<M, Output = M>,
{
  pub fn new(params: BVHBuilderParams) -> Self {
    Self {
      params,
      _phantom: core::marker::PhantomData,
    }
  }

  pub fn build(&self, triangles: &[Triangle]) -> Option<Box<BVHNode<S, V, M>>> {
    if triangles.is_empty() {
      return None;
    }

    let mut indices: Vec<usize> = (0..triangles.len()).collect();
    Some(self.build_recursive(triangles, &mut indices, 0))
  }

  fn build_recursive(
    &self,
    triangles: &[Triangle],
    indices: &mut [usize],
    depth: usize,
  ) -> Box<BVHNode<S, V, M>> {
    let count = indices.len();
    let current_tris = indices.iter().map(|&i| triangles[i]);

    // Create bound based on depth
    let bound = if depth < self.params.aabb_levels {
      BoundNode::AABB(AABB::from_tris(current_tris))
    } else {
      BoundNode::OBB(OBB::from_tris(current_tris))
    };

    if count <= self.params.max_primitives_per_node {
      return Box::new(BVHNode {
        bound,
        left: None,
        right: None,
        primitive_indices: indices.to_vec(),
      });
    }

    // Centroid bounds to find the split axes
    let mut min_centroid = Vec3f32::splat(core::f32::INFINITY);
    let mut max_centroid = Vec3f32::splat(core::f32::NEG_INFINITY);

    for &i in indices.iter() {
      let tri = &triangles[i];
      let centroid = tri.mean_vector();
      min_centroid = min_centroid.min(centroid);
      max_centroid = max_centroid.max(centroid);
    }

    let extents = max_centroid - min_centroid;
    let mut axes = [0, 1, 2];

    // Sort axes by longest to shortest extents
    axes.sort_by(|&a, &b| {
      extents
        .component(b)
        .unwrap()
        .partial_cmp(&extents.component(a).unwrap())
        .unwrap_or(core::cmp::Ordering::Equal)
    });

    let mut split_index = 0;
    let mut found_split = false;

    // 1. Try SAH binning on axes starting from longest
    for &axis in &axes {
      if let Some(idx) = self.try_sah_split(triangles, indices, axis, min_centroid, max_centroid) {
        split_index = idx;
        found_split = true;
        break;
      }
    }

    // 2. If SAH fails, try median split on axes starting from longest
    if !found_split {
      for &axis in &axes {
        indices.sort_by(|&a, &b| {
          triangles[a]
            .mean_vector()
            .component(axis)
            .unwrap()
            .partial_cmp(&triangles[b].mean_vector().component(axis).unwrap())
            .unwrap_or(core::cmp::Ordering::Equal)
        });

        let mid = count / 2;
        // Verify it actually splits things (in case all centroids overlap on this axis)
        if mid > 0 && mid < count {
          split_index = mid;
          found_split = true;
          break;
        }
      }
    }

    // 3. If median fails, just split array in half physically (sorted by longest axis)
    if !found_split {
      let axis = axes[0];
      indices.sort_by(|&a, &b| {
        triangles[a]
          .mean_vector()
          .component(axis)
          .unwrap()
          .partial_cmp(&triangles[b].mean_vector().component(axis).unwrap())
          .unwrap_or(core::cmp::Ordering::Equal)
      });
      split_index = count / 2;
    }

    if split_index == 0 || split_index == count {
      // Should be impossible due to fallback 3, but just in case
      return Box::new(BVHNode {
        bound,
        left: None,
        right: None,
        primitive_indices: indices.to_vec(),
      });
    }

    let (left_indices, right_indices) = indices.split_at_mut(split_index);

    Box::new(BVHNode {
      bound,
      left: Some(self.build_recursive(triangles, left_indices, depth + 1)),
      right: Some(self.build_recursive(triangles, right_indices, depth + 1)),
      primitive_indices: Vec::new(), // inner nodes don't store primitives directly
    })
  }

  fn try_sah_split(
    &self,
    triangles: &[Triangle],
    indices: &mut [usize],
    axis: usize,
    min_centroid: Vec3f32,
    max_centroid: Vec3f32,
  ) -> Option<usize> {
    let min_c = min_centroid.component(axis).unwrap();
    let max_c = max_centroid.component(axis).unwrap();

    if min_c >= max_c {
      return None; // Cannot split on this axis (zero extent)
    }

    let bin_count = self.params.bin_count;
    let mut bins = alloc::vec![(0, AABB::<V>::new(V::splat(V::Scalar::from_f32(core::f32::INFINITY)), V::splat(V::Scalar::from_f32(core::f32::NEG_INFINITY)))); bin_count];

    let extent = max_c - min_c;

    // Populate bins
    for &i in indices.iter() {
      let tri = &triangles[i];
      let centroid = tri.mean_vector().component(axis).unwrap();

      let mut bin_idx = (((centroid - min_c) / extent) * (bin_count as f32)) as usize;
      if bin_idx == bin_count {
        bin_idx = bin_count - 1;
      }

      bins[bin_idx].0 += 1;

      let tri_aabb = AABB::<V>::from_tris(core::iter::once(*tri));

      let b_min = bins[bin_idx].1.min().min(tri_aabb.min());
      let b_max = bins[bin_idx].1.max().max(tri_aabb.max());
      bins[bin_idx].1 = AABB::new(b_min, b_max);
    }

    // SAH arrays
    let mut left_area = alloc::vec![S::from_f32(0.0); bin_count - 1];
    let mut right_area = alloc::vec![S::from_f32(0.0); bin_count - 1];
    let mut left_count = alloc::vec![0; bin_count - 1];
    let mut right_count = alloc::vec![0; bin_count - 1];

    let mut left_box = AABB::<V>::new(
      V::splat(V::Scalar::from_f32(core::f32::INFINITY)),
      V::splat(V::Scalar::from_f32(core::f32::NEG_INFINITY)),
    );
    let mut left_sum = 0;
    for i in 0..bin_count - 1 {
      left_sum += bins[i].0;
      left_count[i] = left_sum;
      left_box = AABB::new(
        left_box.min().min(bins[i].1.min()),
        left_box.max().max(bins[i].1.max()),
      );
      left_area[i] = self.aabb_surface_area(&left_box);
    }

    let mut right_box = AABB::<V>::new(
      V::splat(V::Scalar::from_f32(core::f32::INFINITY)),
      V::splat(V::Scalar::from_f32(core::f32::NEG_INFINITY)),
    );
    let mut right_sum = 0;
    for i in (1..bin_count).rev() {
      right_sum += bins[i].0;
      right_count[i - 1] = right_sum;
      right_box = AABB::new(
        right_box.min().min(bins[i].1.min()),
        right_box.max().max(bins[i].1.max()),
      );
      right_area[i - 1] = self.aabb_surface_area(&right_box);
    }

    // Find best split
    let mut min_cost = S::from_f32(core::f32::INFINITY);
    let mut best_bin = 0;

    for i in 0..bin_count - 1 {
      let cost = S::from_f32(left_count[i] as f32) * left_area[i]
        + S::from_f32(right_count[i] as f32) * right_area[i];
      if cost < min_cost {
        min_cost = cost;
        best_bin = i;
      }
    }

    // Leaf cost logic (simplistic)
    let leaf_cost = S::from_f32(indices.len() as f32)
      * self.aabb_surface_area(&AABB::new(
        left_box.min().min(right_box.min()),
        left_box.max().max(right_box.max()),
      ));

    if min_cost >= leaf_cost {
      // Split cost is worse than not splitting
      return None;
    }

    // Partition the indices based on the best bin
    let mut left_ptr = 0;
    let mut right_ptr = indices.len();
    while left_ptr < right_ptr {
      let tri = &triangles[indices[left_ptr]];
      let centroid = tri.mean_vector().component(axis).unwrap();
      let mut bin_idx = (((centroid - min_c) / extent) * (bin_count as f32)) as usize;
      if bin_idx == bin_count {
        bin_idx = bin_count - 1;
      }

      if bin_idx <= best_bin {
        left_ptr += 1;
      } else {
        indices.swap(left_ptr, right_ptr - 1);
        right_ptr -= 1;
      }
    }

    if left_ptr == 0 || left_ptr == indices.len() {
      return None; // Empty side
    }

    Some(left_ptr)
  }

  fn aabb_surface_area(&self, aabb: &AABB<V>) -> S {
    let d = aabb.max() - aabb.min();
    let dx = d.x();
    let dy = d.y();
    let dz = d.z();
    let _0 = S::from_f32(0.0);
    // Use positive dimensions only (empty boxes have 0 surface area)
    if dx < _0 || dy < _0 || dz < _0 {
      return _0;
    }
    S::from_f32(2.0) * (dx * dy + dy * dz + dz * dx)
  }
}
