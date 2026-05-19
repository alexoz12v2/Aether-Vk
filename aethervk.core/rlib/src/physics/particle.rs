//! particle module.

use aethervk_oshal_rlib::math::{
  FloatLike,
  floating::{FloatBits, FloatOps},
  matrix::{Matrix3, Matrix4, mat3::Mat3f32, mat4::Mat4x4f32},
  vector::{Vector, Vector3, Vector4, vec3::Vec3f32, vec4::Vec4f32},
};
use alloc::{boxed::Box, vec::Vec};

use crate::math::collision::{
  bounds::{AABB, OBB},
  bvh_builder::{BVHBuilderParams, BVHNode, BoundNode},
};

#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
pub struct Particle<V>
where
  V: Vector3,
  V::Scalar: FloatLike + FloatOps + FloatBits,
{
  pub position: V,
  pub radius: V::Scalar,
}

impl<V> Particle<V>
where
  V: Vector3,
  V::Scalar: FloatLike + FloatOps + FloatBits,
{
  /// TODO: Document this item
  pub fn aabb(&self) -> AABB<V::Scalar>
  where
    V: Into<[V::Scalar; 3]>,
  {
    let r = V::splat(self.radius);
    AABB::new(self.position - r, self.position + r)
  }
}

/// TODO: Document this item
pub struct ParticleBVHBuilder {
  params: BVHBuilderParams,
}

impl ParticleBVHBuilder {
  /// TODO: Document this item
  pub fn new(params: BVHBuilderParams) -> Self {
    Self { params }
  }

  /// TODO: Document this item
  pub fn build<'a, I, V, M>(&self, particles: I) -> Option<Box<BVHNode<V::Scalar>>>
  where
    I: IntoIterator<Item = &'a Particle<V>>,
    V: Vector3 + 'a + Into<[V::Scalar; 3]> + From<[V::Scalar; 3]> + core::ops::Mul<f32, Output = V>,
    M: Matrix3<Vector = V, Scalar = V::Scalar> + core::fmt::Debug,
    V::Scalar: FloatLike + FloatOps + FloatBits + From<f32>,
    I::IntoIter: ExactSizeIterator + Clone,
  {
    let particles_vec: Vec<Particle<V>> = particles.into_iter().copied().collect();
    if particles_vec.is_empty() {
      return None;
    }

    let mut indices: Vec<usize> = (0..particles_vec.len()).collect();
    Some(self.build_recursive::<V, M>(&particles_vec, &mut indices, 0))
  }

  fn build_recursive<V, M>(
    &self,
    particles: &[Particle<V>],
    indices: &mut [usize],
    depth: usize,
  ) -> Box<BVHNode<V::Scalar>>
  where
    V: Vector3 + Into<[V::Scalar; 3]> + From<[V::Scalar; 3]> + core::ops::Mul<f32, Output = V>,
    M: Matrix3<Vector = V, Scalar = V::Scalar> + core::fmt::Debug,
    V::Scalar: FloatLike + FloatOps + FloatBits + From<f32>,
  {
    let count = indices.len();

    let mut min_bound = V::splat(<V::Scalar as FloatOps>::INFINITY);
    let mut max_bound = V::splat(<V::Scalar as FloatOps>::NEG_INFINITY);

    for &i in indices.iter() {
      let p = &particles[i];
      let p_min = p.position - V::splat(p.radius);
      let p_max = p.position + V::splat(p.radius);
      min_bound = min_bound.min(p_min);
      max_bound = max_bound.max(p_max);
    }
    let aabb = AABB::<V::Scalar>::new(min_bound, max_bound);

    let bound = if depth < self.params.aabb_levels {
      BoundNode::AABB(aabb)
    } else {
      // OBB for particles is typically just a bounding box if we don't do PCA.
      // We will fallback to AABB converted to OBB representation for simplicity here.
      let min: V = aabb.min();
      let max: V = aabb.max();
      let center = (min + max) * 0.5;
      let extents = (max - min) * 0.5;
      let obb = OBB::new(center, M::identity(), extents);
      BoundNode::OBB(obb)
    };

    if count <= self.params.max_primitives_per_node {
      return Box::new(BVHNode {
        bound,
        left: None,
        right: None,
        primitive_indices: indices.to_vec(),
      });
    }

    let mut min_centroid = V::splat(<V::Scalar as FloatOps>::INFINITY);
    let mut max_centroid = V::splat(<V::Scalar as FloatOps>::NEG_INFINITY);

    for &i in indices.iter() {
      let centroid = particles[i].position;
      min_centroid = min_centroid.min(centroid);
      max_centroid = max_centroid.max(centroid);
    }

    let extents = max_centroid - min_centroid;
    let mut axes = [0, 1, 2];
    axes.sort_by(|&a, &b| {
      extents
        .component(b)
        .unwrap()
        .partial_cmp(&extents.component(a).unwrap())
        .unwrap_or(core::cmp::Ordering::Equal)
    });

    let mut split_index = count / 2;
    let axis = axes[0];
    indices.sort_by(|&a, &b| {
      particles[a]
        .position
        .component(axis)
        .unwrap()
        .partial_cmp(&particles[b].position.component(axis).unwrap())
        .unwrap_or(core::cmp::Ordering::Equal)
    });

    if split_index == 0 || split_index == count {
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
      left: Some(self.build_recursive::<V, M>(particles, left_indices, depth + 1)),
      right: Some(self.build_recursive::<V, M>(particles, right_indices, depth + 1)),
      primitive_indices: alloc::vec::Vec::new(),
    })
  }
}

/// Refits an existing BVH tree to encompass motion bounds between `old_particles` and `new_particles`.
/// Assuming the shape of the input matches.
pub fn refit_motion_bounds<V, M>(
  node: &mut BVHNode<V::Scalar>,
  old_particles: &[Particle<V>],
  new_particles: &[Particle<V>],
) where
  V: Vector3 + Into<[V::Scalar; 3]> + From<[V::Scalar; 3]> + core::ops::Mul<f32, Output = V>,
  M: Matrix3<Vector = V, Scalar = V::Scalar>,
  V::Scalar: FloatLike + FloatOps + FloatBits + From<f32>,
{
  if let Some(left) = &mut node.left {
    refit_motion_bounds::<V, M>(left, old_particles, new_particles);
  }
  if let Some(right) = &mut node.right {
    refit_motion_bounds::<V, M>(right, old_particles, new_particles);
  }

  let mut min_bound = V::splat(<V::Scalar as FloatOps>::INFINITY);
  let mut max_bound = V::splat(<V::Scalar as FloatOps>::NEG_INFINITY);

  if node.left.is_none() && node.right.is_none() {
    for &i in &node.primitive_indices {
      let p_old = &old_particles[i];
      let p_new = &new_particles[i];

      let p_old_min = p_old.position - V::splat(p_old.radius);
      let p_old_max = p_old.position + V::splat(p_old.radius);

      let p_new_min = p_new.position - V::splat(p_new.radius);
      let p_new_max = p_new.position + V::splat(p_new.radius);

      min_bound = min_bound.min(p_old_min).min(p_new_min);
      max_bound = max_bound.max(p_old_max).max(p_new_max);
    }
  } else {
    // Combine children AABBs
    let combine = |child: &Option<Box<BVHNode<V::Scalar>>>, min_b: &mut V, max_b: &mut V| {
      if let Some(c) = child {
        match &c.bound {
          BoundNode::AABB(aabb) => {
            *min_b = min_b.min(aabb.min());
            *max_b = max_b.max(aabb.max());
          }
          BoundNode::OBB(obb) => {
            // Very conservative AABB of OBB
            let center: V = obb.translation();
            let extents: V = obb.half_extent();
            let axes: [V; 3] = obb.axes();
            let ex = V::from_components(axes[0].x().abs(), axes[0].y().abs(), axes[0].z().abs())
              * extents.x();
            let ey = V::from_components(axes[1].x().abs(), axes[1].y().abs(), axes[1].z().abs())
              * extents.y();
            let ez = V::from_components(axes[2].x().abs(), axes[2].y().abs(), axes[2].z().abs())
              * extents.z();
            let aabb_ext = ex + ey + ez;
            *min_b = min_b.min(center - aabb_ext);
            *max_b = max_b.max(center + aabb_ext);
          }
          _ => {}
        }
      }
    };
    combine(&node.left, &mut min_bound, &mut max_bound);
    combine(&node.right, &mut min_bound, &mut max_bound);
  }

  // Set this node's bound to the newly computed motion bounds AABB.
  // Wait, refit means we only build a BVH of AABB motion bounds.
  // The user says "a BVH of AABB motion bounds"
  node.bound = BoundNode::AABB(AABB::new(min_bound, max_bound));
}

#[cfg(test)]
mod tests {
  use super::*;
  use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;

  #[test]
  fn test_particle_bvh_build_and_refit() {
    let p1 = Particle {
      position: Vec3f32::from_components(0.0, 0.0, 0.0),
      radius: 1.0,
    };
    let p2 = Particle {
      position: Vec3f32::from_components(5.0, 0.0, 0.0),
      radius: 1.0,
    };
    let p3 = Particle {
      position: Vec3f32::from_components(0.0, 5.0, 0.0),
      radius: 1.0,
    };

    let old_particles = vec![p1, p2, p3];
    let builder = ParticleBVHBuilder::new(BVHBuilderParams::default());
    let mut bvh = builder.build::<_, Vec3f32, Mat3f32>(&old_particles).unwrap();

    let mut new_particles = old_particles.clone();
    new_particles[0].position = Vec3f32::from_components(-1.0, 0.0, 0.0);

    refit_motion_bounds::<Vec3f32, Mat3f32>(&mut bvh, &old_particles, &new_particles);

    if let BoundNode::AABB(aabb) = &bvh.bound {
      assert!(aabb.min::<Vec3f32>().x() <= -2.0); // p1 old min was -1, new min is -2
      assert!(aabb.max::<Vec3f32>().x() >= 6.0); // p2 max is 6
    } else {
      panic!("Expected AABB root after refit");
    }
  }
}
