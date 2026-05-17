//! BVH Builder module
//!
//! Provides algorithms to build Bounding Volume Hierarchies using Top-down approach
//! with Surface Area Heuristic (SAH), supporting multiple bounding types.

use aethervk_oshal_rlib::math::{
  FloatLike,
  floating::{FloatBits, FloatOps},
  matrix::{Matrix3, MatrixVectorMul, mat3::Mat3f32},
  vector::{Vector, Vector3, vec3::Vec3f32},
};
use alloc::{boxed::Box, vec::Vec};

use crate::{
  math::collision::bounds::{AABB, BS, OBB},
  simulation::comet::Triangle,
};

#[derive(Debug, Clone)]
/// TODO: Document this item
pub enum BoundNode<S>
where
  S: FloatLike + FloatOps + FloatBits + From<f32>,
{
  AABB(AABB<S>),
  OBB(OBB<S>),
  BS(BS<S>),
}

impl<S> BoundNode<S>
where
  S: FloatLike + FloatOps + FloatBits + From<f32>,
{
  /// TODO: Document this item
  pub fn contains<V, M>(&self, other: &Self) -> bool
  where
    V: Vector3<Scalar = S> + From<[S; 3]>,
    M: Matrix3<Scalar = S, Vector = V> + MatrixVectorMul,
  {
    match (self, other) {
      (BoundNode::AABB(a), BoundNode::AABB(b)) => a.contains_aabb::<V>(b),
      (BoundNode::AABB(a), BoundNode::OBB(b)) => a.contains_obb::<V>(b),
      (BoundNode::OBB(a), BoundNode::AABB(b)) => a.contains_aabb::<V, M>(b),
      (BoundNode::OBB(a), BoundNode::OBB(b)) => a.contains_obb::<V, M>(b),
      _ => true,
    }
  }

  /// TODO: Document this item
  pub fn encapsulate_bound<V, M>(&mut self, other: &Self)
  where
    V: Vector3<Scalar = S> + From<[S; 3]>,
    M: Matrix3<Scalar = S, Vector = V> + MatrixVectorMul,
  {
    match (self, other) {
      (BoundNode::AABB(a), BoundNode::AABB(b)) => a.encapsulate_aabb::<V>(b),
      (BoundNode::AABB(a), BoundNode::OBB(b)) => a.encapsulate_obb::<V>(b),
      (BoundNode::OBB(a), BoundNode::AABB(b)) => a.encapsulate_aabb::<V, M>(b),
      (BoundNode::OBB(a), BoundNode::OBB(b)) => a.encapsulate_obb::<V, M>(b),
      _ => {}
    }
  }
}

#[derive(Debug, Clone)]
/// TODO: Document this item
pub struct BVHNode<S>
where
  S: FloatLike + FloatOps + FloatBits + From<f32>,
{
  pub bound: BoundNode<S>,
  pub left: Option<Box<BVHNode<S>>>,
  pub right: Option<Box<BVHNode<S>>>,
  // Indices into the original triangle slice
  pub primitive_indices: Vec<usize>,
}

/// TODO: Document this item
pub struct BVHBuilderParams {
  pub aabb_levels: usize,
  pub max_primitives_per_node: usize,
  pub bin_count: usize,
}

impl BVHBuilderParams {
  /// TODO: Document this item
  pub fn aabb_levels(&self, value: usize) -> Self {
    Self {
      aabb_levels: value,
      ..*self
    }
  }

  /// TODO: Document this item
  pub fn max_primitives_per_node(&self, value: usize) -> Self {
    Self {
      max_primitives_per_node: value,
      ..*self
    }
  }

  /// TODO: Document this item
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

/// TODO: Document this item
pub struct BVHBuilder<S, V, M>
where
  M: Matrix3<Scalar = S, Vector = V> + From<Mat3f32> + MatrixVectorMul,
  V: Vector3<Scalar = S> + From<Vec3f32> + Into<Vec3f32> + From<[S; 3]> + Into<[S; 3]>,
  S: FloatLike
    + FloatOps
    + FloatBits
    + From<f32>
    + core::ops::Mul<V, Output = V>
    + core::ops::Mul<M, Output = M>,
{
  params: BVHBuilderParams,
  _phantom: core::marker::PhantomData<(S, V, M)>,
}

impl<S, V, M> BVHBuilder<S, V, M>
where
  M: Matrix3<Scalar = S, Vector = V> + From<Mat3f32> + MatrixVectorMul,
  V: Vector3<Scalar = S> + From<Vec3f32> + Into<Vec3f32> + From<[S; 3]> + Into<[S; 3]>,
  S: FloatLike
    + FloatOps
    + FloatBits
    + From<f32>
    + core::ops::Mul<V, Output = V>
    + core::ops::Mul<M, Output = M>,
{
  /// TODO: Document this item
  pub fn new(params: BVHBuilderParams) -> Self {
    Self {
      params,
      _phantom: core::marker::PhantomData,
    }
  }

  /// TODO: Document this item
  pub fn build(&self, triangles: &[Triangle]) -> Option<Box<BVHNode<S>>> {
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
  ) -> Box<BVHNode<S>> {
    let count = indices.len();
    let current_tris = indices.iter().map(|&i| triangles[i]);

    // Create bound based on depth
    let bound = if depth < self.params.aabb_levels {
      BoundNode::AABB(AABB::from_tris::<V, _>(current_tris))
    } else {
      BoundNode::OBB(OBB::from_tris::<V, M, _>(current_tris))
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
      // TODO: Split by using AABB if current level is AABB, otherwise use OBB
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

    let left = self.build_recursive(triangles, left_indices, depth + 1);
    let right = self.build_recursive(triangles, right_indices, depth + 1);

    Box::new(BVHNode {
      bound,
      left: Some(left),
      right: Some(right),
      primitive_indices: Vec::new(),
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
      return None;
    }

    let bin_count = self.params.bin_count;
    let mut bins = alloc::vec![(0, AABB::<S>::new(V::splat(V::Scalar::from_f32(core::f32::INFINITY)), V::splat(V::Scalar::from_f32(core::f32::NEG_INFINITY)))); bin_count];

    let extent = max_c - min_c;

    // Populate bins
    for &i in indices.iter() {
      let tri = &triangles[i];
      let centroid = tri.mean_vector().component(axis).unwrap();

      let mut bin_idx = (((centroid - min_c) / extent) * (bin_count as f32)) as usize;
      if bin_idx >= bin_count {
        bin_idx = bin_count - 1;
      }

      bins[bin_idx].0 += 1;

      let tri_aabb = AABB::<S>::from_tris::<V, _>(core::iter::once(*tri));

      let b_min: V = bins[bin_idx].1.min::<V>().min(tri_aabb.min());
      let b_max: V = bins[bin_idx].1.max::<V>().max(tri_aabb.max());
      bins[bin_idx].1 = AABB::new(b_min, b_max);
    }

    // SAH arrays
    let mut left_area = alloc::vec![S::from_f32(0.0); bin_count - 1];
    let mut right_area = alloc::vec![S::from_f32(0.0); bin_count - 1];
    let mut left_count = alloc::vec![0; bin_count - 1];
    let mut right_count = alloc::vec![0; bin_count - 1];

    let mut left_box = AABB::<S>::new(
      V::splat(V::Scalar::from_f32(core::f32::INFINITY)),
      V::splat(V::Scalar::from_f32(core::f32::NEG_INFINITY)),
    );
    let mut left_sum = 0;
    for i in 0..bin_count - 1 {
      left_sum += bins[i].0;
      left_count[i] = left_sum;
      left_box.encapsulate_aabb::<V>(&bins[i].1);
      left_area[i] = self.aabb_surface_area(&left_box);
    }

    let mut right_box = AABB::<S>::new(
      V::splat(V::Scalar::from_f32(core::f32::INFINITY)),
      V::splat(V::Scalar::from_f32(core::f32::NEG_INFINITY)),
    );
    let mut right_sum = 0;
    for i in (1..bin_count).rev() {
      right_sum += bins[i].0;
      right_count[i - 1] = right_sum;
      right_box.encapsulate_aabb::<V>(&bins[i].1);
      right_area[i - 1] = self.aabb_surface_area(&right_box);
    }

    let mut min_cost = S::from_f32(core::f32::INFINITY);
    let mut best_bin: usize = 0;

    for i in 0..bin_count - 1 {
      let cost = S::from_f32(left_count[i] as f32) * left_area[i]
        + S::from_f32(right_count[i] as f32) * right_area[i];
      if cost < min_cost {
        min_cost = cost;
        best_bin = i;
      }
    }

    let leaf_cost = S::from_f32(indices.len() as f32)
      * self.aabb_surface_area(&AABB::new(
        left_box.min::<V>().min(right_box.min()),
        left_box.max::<V>().max(right_box.max()),
      ));

    if min_cost >= leaf_cost {
      return None;
    }

    let mut left_ptr = 0;
    let mut right_ptr = indices.len();
    while left_ptr < right_ptr {
      let tri = &triangles[indices[left_ptr]];
      let centroid = tri.mean_vector().component(axis).unwrap();
      let mut bin_idx = (((centroid - min_c) / extent) * (bin_count as f32)) as usize;
      if bin_idx >= bin_count {
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
      return None;
    }

    Some(left_ptr)
  }

  fn aabb_surface_area(&self, aabb: &AABB<S>) -> S {
    let d: V = aabb.max::<V>() - aabb.min();
    let dx = d.x();
    let dy = d.y();
    let dz = d.z();
    let _0 = S::from_f32(0.0);
    if dx < _0 || dy < _0 || dz < _0 {
      return _0;
    }
    S::from_f32(2.0) * (dx * dy + dy * dz + dz * dx)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use aethervk_oshal_rlib::math::{matrix::mat3::Mat3f32, vector::vec3::Vec3f32};

  #[test]
  fn test_bvh_builder_empty() {
    let builder = BVHBuilder::<f32, Vec3f32, Mat3f32>::new(BVHBuilderParams::default());
    let tris = vec![];
    let root = builder.build(&tris);
    assert!(root.is_none());
  }

  #[test]
  fn test_bvh_builder_single_triangle() {
    let builder = BVHBuilder::<f32, Vec3f32, Mat3f32>::new(BVHBuilderParams::default());
    let t = Triangle {
      vertices: [
        Vec3f32::from_components(0.0, 0.0, 0.0),
        Vec3f32::from_components(1.0, 0.0, 0.0),
        Vec3f32::from_components(0.0, 1.0, 0.0),
      ],
    };
    let tris = vec![t];
    let root = builder.build(&tris).expect("Should build root");

    assert!(root.left.is_none());
    assert!(root.right.is_none());
    assert_eq!(root.primitive_indices.len(), 1);
    assert_eq!(root.primitive_indices[0], 0);

    // Bounds should encapsulate the triangle
    match &root.bound {
      BoundNode::AABB(aabb) => {
        assert_eq!(aabb.min::<Vec3f32>().x(), 0.0);
        assert_eq!(aabb.max::<Vec3f32>().x(), 1.0);
        assert_eq!(aabb.min::<Vec3f32>().y(), 0.0);
        assert_eq!(aabb.max::<Vec3f32>().y(), 1.0);
      }
      _ => panic!("Expected AABB"),
    }
  }

  #[test]
  fn test_bvh_builder_split() {
    let builder = BVHBuilder::<f32, Vec3f32, Mat3f32>::new(BVHBuilderParams {
      aabb_levels: 10,
      max_primitives_per_node: 1, // Force split
      bin_count: 4,
    });
    let t1 = Triangle {
      vertices: [
        Vec3f32::from_components(0.0, 0.0, 0.0),
        Vec3f32::from_components(1.0, 0.0, 0.0),
        Vec3f32::from_components(0.0, 1.0, 0.0),
      ],
    };
    let t2 = Triangle {
      vertices: [
        Vec3f32::from_components(10.0, 10.0, 10.0),
        Vec3f32::from_components(11.0, 10.0, 10.0),
        Vec3f32::from_components(10.0, 11.0, 10.0),
      ],
    };
    let tris = vec![t1, t2];
    let root = builder.build(&tris).expect("Should build root");

    assert!(root.left.is_some());
    assert!(root.right.is_some());
    assert!(root.primitive_indices.is_empty()); // Parent has no primitives

    // Parent should encapsulate both
    match &root.bound {
      BoundNode::AABB(aabb) => {
        assert_eq!(aabb.min::<Vec3f32>().x(), 0.0);
        assert_eq!(aabb.max::<Vec3f32>().x(), 11.0);
      }
      _ => panic!("Expected AABB"),
    }
  }
}
