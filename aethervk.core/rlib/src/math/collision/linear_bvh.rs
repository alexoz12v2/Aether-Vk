use alloc::vec::Vec;
use aethervk_oshal_rlib::math::{
  floating::{FloatBits, FloatOps},
  matrix::{Matrix3, mat3::Mat3f32},
  vector::{Vector, Vector3, vec3::Vec3f32},
  FloatLike,
};

use crate::math::collision::bounds::{AABB, OBB};
use crate::math::collision::bvh_builder::{BVHNode, BoundNode};

#[derive(Debug, Clone)]
pub struct LinearBVHHeader {
  /// How many levels from the bottom of the tree use OBB instead of AABB.
  pub preciseness: u32,
  /// Total number of nodes in this BVH.
  pub node_count: u32,
  /// Total number of primitives referenced by this BVH.
  pub primitive_count: u32,
}

#[derive(Debug, Clone)]
pub enum LinearBound<S, V, M>
where
  M: Matrix3<Scalar = S, Vector = V>,
  V: Vector3<Scalar = S> + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  S: FloatLike
    + FloatOps
    + From<f32>
    
    + core::ops::Mul<V, Output = V>
    + core::ops::Mul<M, Output = M>,
{
  AABB(AABB<V>),
  OBB(OBB<S, V, M>),
}

#[derive(Debug, Clone)]
pub struct LinearBVHNode<S, V, M>
where
  M: Matrix3<Scalar = S, Vector = V>,
  V: Vector3<Scalar = S> + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  S: FloatLike
    + FloatOps
    + From<f32>
    
    + core::ops::Mul<V, Output = V>
    + core::ops::Mul<M, Output = M>,
{
  pub bound: LinearBound<S, V, M>,
  /// Index of the left child, or `u32::MAX` if leaf. Right child is always `left_child_or_primitive_offset + 1` if not leaf.
  /// If leaf, this is the offset into the primitives array.
  pub left_child_or_primitive_offset: u32,
  /// Number of primitives in this leaf. 0 if not a leaf.
  pub primitive_count: u32,
}

#[derive(Debug, Clone)]
pub struct LinearBVH<S, V, M>
where
  M: Matrix3<Scalar = S, Vector = V>,
  V: Vector3<Scalar = S> + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  S: FloatLike
    + FloatOps
    + From<f32>
    
    + core::ops::Mul<V, Output = V>
    + core::ops::Mul<M, Output = M>,
{
  pub header: LinearBVHHeader,
  pub nodes: Vec<LinearBVHNode<S, V, M>>,
  pub primitives: Vec<usize>,
}

impl<S, V, M> LinearBVH<S, V, M>
where
  M: Matrix3<Scalar = S, Vector = V>,
  V: Vector3<Scalar = S> + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  S: FloatLike
    + FloatOps
    + From<f32>
    
    + core::ops::Mul<V, Output = V>
    + core::ops::Mul<M, Output = M>,
{
  pub fn from_build_node(root: &BVHNode<S, V, M>, preciseness: u32) -> Self {
    let mut nodes = Vec::new();
    let mut primitives = Vec::new();

    // Flatten in depth-first order first, we can do Morton order sorting later if needed
    // The typical GPU linear layout is left child = next node, right child = current + skip offset.
    Self::flatten_node(root, &mut nodes, &mut primitives);

    Self {
      header: LinearBVHHeader {
        preciseness,
        node_count: nodes.len() as u32,
        primitive_count: primitives.len() as u32,
      },
      nodes,
      primitives,
    }
  }

  fn flatten_node(
    node: &BVHNode<S, V, M>,
    nodes: &mut Vec<LinearBVHNode<S, V, M>>,
    primitives: &mut Vec<usize>,
  ) -> u32 {
    let bound = match &node.bound {
      BoundNode::AABB(aabb) => LinearBound::AABB(aabb.clone()),
      BoundNode::OBB(obb) => LinearBound::OBB(obb.clone()),
      BoundNode::BS(_) => panic!("BS not supported in LinearBVH yet"),
    };

    let current_idx = nodes.len() as u32;

    // Placeholder node, will be updated later
    nodes.push(LinearBVHNode {
      bound,
      left_child_or_primitive_offset: 0,
      primitive_count: 0,
    });

    if node.left.is_none() && node.right.is_none() {
      // Leaf
      let primitive_offset = primitives.len() as u32;
      primitives.extend(&node.primitive_indices);
      nodes[current_idx as usize].left_child_or_primitive_offset = primitive_offset;
      nodes[current_idx as usize].primitive_count = node.primitive_indices.len() as u32;
    } else {
      // Inner node
      let mut left_idx = u32::MAX;
      if let Some(left) = &node.left {
        left_idx = Self::flatten_node(left, nodes, primitives);
      }
      if let Some(right) = &node.right {
        Self::flatten_node(right, nodes, primitives);
      }
      
      nodes[current_idx as usize].left_child_or_primitive_offset = left_idx;
      nodes[current_idx as usize].primitive_count = 0;
    }

    current_idx
  }
}
