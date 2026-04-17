use alloc::boxed::Box;
use alloc::vec::Vec;
use aethervk_oshal_rlib::math::{
  floating::{FloatBits, FloatOps},
  matrix::{Matrix3, mat3::Mat3f32},
  vector::{Vector, Vector3, vec3::Vec3f32},
  FloatLike,
};

use crate::math::collision::bounds::{AABB, OBB};
use crate::math::collision::bvh_builder::{BVHNode, BoundNode, BVHBuilderParams};

#[derive(Debug, Clone, Copy)]
pub struct Particle {
  pub position: Vec3f32,
  pub radius: f32,
}

impl Particle {
  pub fn aabb(&self) -> AABB<Vec3f32> {
    let r = Vec3f32::splat(self.radius);
    AABB::new(self.position - r, self.position + r)
  }
}

pub struct ParticleBVHBuilder {
  params: BVHBuilderParams,
}

impl ParticleBVHBuilder {
  pub fn new(params: BVHBuilderParams) -> Self {
    Self { params }
  }

  pub fn build<'a, I>(&self, particles: I) -> Option<Box<BVHNode<f32, Vec3f32, Mat3f32>>>
  where
    I: IntoIterator<Item = &'a Particle>,
    I::IntoIter: ExactSizeIterator + Clone,
  {
    let particles_vec: Vec<Particle> = particles.into_iter().copied().collect();
    if particles_vec.is_empty() {
      return None;
    }

    let mut indices: Vec<usize> = (0..particles_vec.len()).collect();
    Some(self.build_recursive(&particles_vec, &mut indices, 0))
  }

  fn build_recursive(
    &self,
    particles: &[Particle],
    indices: &mut [usize],
    depth: usize,
  ) -> Box<BVHNode<f32, Vec3f32, Mat3f32>> {
    let count = indices.len();

    let mut min_bound = Vec3f32::splat(core::f32::INFINITY);
    let mut max_bound = Vec3f32::splat(core::f32::NEG_INFINITY);

    for &i in indices.iter() {
      let p = &particles[i];
      let p_min = p.position - Vec3f32::splat(p.radius);
      let p_max = p.position + Vec3f32::splat(p.radius);
      min_bound = min_bound.min(p_min);
      max_bound = max_bound.max(p_max);
    }
    let aabb = AABB::new(min_bound, max_bound);

    let bound = if depth < self.params.aabb_levels {
      BoundNode::AABB(aabb)
    } else {
      // OBB for particles is typically just a bounding box if we don't do PCA.
      // We will fallback to AABB converted to OBB representation for simplicity here.
      let min = aabb.min();
      let max = aabb.max();
      let center = (min + max) * 0.5;
      let extents = (max - min) * 0.5;
      let obb = OBB::new(center, Mat3f32::identity(), extents);
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

    let mut min_centroid = Vec3f32::splat(core::f32::INFINITY);
    let mut max_centroid = Vec3f32::splat(core::f32::NEG_INFINITY);

    for &i in indices.iter() {
      let centroid = particles[i].position;
      min_centroid = min_centroid.min(centroid);
      max_centroid = max_centroid.max(centroid);
    }

    let extents = max_centroid - min_centroid;
    let mut axes = [0, 1, 2];
    axes.sort_by(|&a, &b| {
      extents.component(b).unwrap().partial_cmp(&extents.component(a).unwrap()).unwrap_or(core::cmp::Ordering::Equal)
    });

    let mut split_index = count / 2;
    let axis = axes[0];
    indices.sort_by(|&a, &b| {
      particles[a].position.component(axis).unwrap()
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
      left: Some(self.build_recursive(particles, left_indices, depth + 1)),
      right: Some(self.build_recursive(particles, right_indices, depth + 1)),
      primitive_indices: alloc::vec::Vec::new(),
    })
  }
}

/// Refits an existing BVH tree to encompass motion bounds between `old_particles` and `new_particles`.
/// Assuming the shape of the input matches.
pub fn refit_motion_bounds(
  node: &mut BVHNode<f32, Vec3f32, Mat3f32>,
  old_particles: &[Particle],
  new_particles: &[Particle],
) {
  if let Some(left) = &mut node.left {
    refit_motion_bounds(left, old_particles, new_particles);
  }
  if let Some(right) = &mut node.right {
    refit_motion_bounds(right, old_particles, new_particles);
  }

  let mut min_bound = Vec3f32::splat(core::f32::INFINITY);
  let mut max_bound = Vec3f32::splat(core::f32::NEG_INFINITY);

  if node.left.is_none() && node.right.is_none() {
    for &i in &node.primitive_indices {
      let p_old = &old_particles[i];
      let p_new = &new_particles[i];

      let p_old_min = p_old.position - Vec3f32::splat(p_old.radius);
      let p_old_max = p_old.position + Vec3f32::splat(p_old.radius);

      let p_new_min = p_new.position - Vec3f32::splat(p_new.radius);
      let p_new_max = p_new.position + Vec3f32::splat(p_new.radius);

      min_bound = min_bound.min(p_old_min).min(p_new_min);
      max_bound = max_bound.max(p_old_max).max(p_new_max);
    }
  } else {
    // Combine children AABBs
    let combine = |child: &Option<Box<BVHNode<f32, Vec3f32, Mat3f32>>>, min_b: &mut Vec3f32, max_b: &mut Vec3f32| {
      if let Some(c) = child {
        match &c.bound {
          BoundNode::AABB(aabb) => {
            *min_b = min_b.min(aabb.min());
            *max_b = max_b.max(aabb.max());
          }
          BoundNode::OBB(obb) => {
            // Very conservative AABB of OBB
            let center = obb.translation();
            let extents = obb.half_extent();
            let axes = obb.axes();
            let ex = Vec3f32::from_components(axes[0].x().abs(), axes[0].y().abs(), axes[0].z().abs()) * extents.x();
            let ey = Vec3f32::from_components(axes[1].x().abs(), axes[1].y().abs(), axes[1].z().abs()) * extents.y();
            let ez = Vec3f32::from_components(axes[2].x().abs(), axes[2].y().abs(), axes[2].z().abs()) * extents.z();
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
    let p1 = Particle { position: Vec3f32::from_components(0.0, 0.0, 0.0), radius: 1.0 };
    let p2 = Particle { position: Vec3f32::from_components(5.0, 0.0, 0.0), radius: 1.0 };
    let p3 = Particle { position: Vec3f32::from_components(0.0, 5.0, 0.0), radius: 1.0 };

    let old_particles = vec![p1, p2, p3];
    let builder = ParticleBVHBuilder::new(BVHBuilderParams::default());
    let mut bvh = builder.build(&old_particles).unwrap();

    let mut new_particles = old_particles.clone();
    new_particles[0].position = Vec3f32::from_components(-1.0, 0.0, 0.0);

    refit_motion_bounds(&mut bvh, &old_particles, &new_particles);

    if let BoundNode::AABB(aabb) = &bvh.bound {
      assert!(aabb.min().x() <= -2.0); // p1 old min was -1, new min is -2
      assert!(aabb.max().x() >= 6.0);  // p2 max is 6
    } else {
      panic!("Expected AABB root after refit");
    }
  }
}
