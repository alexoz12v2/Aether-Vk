//! linear_bvh module.

use aethervk_oshal_rlib::math::{
  FloatLike,
  floating::{FloatBits, FloatOps},
  matrix::mat3::Mat3f32,
  vector::vec3::Vec3f32,
  vector::Vector,
  vector::Vector3,
};
use alloc::{vec, vec::Vec};

use crate::math::collision::{
  bounds::{AABB, OBB},
  bvh_builder::{BVHNode, BoundNode},
};

#[derive(Debug, Clone)]
/// TODO: Document this item
pub struct LinearBVHHeader {
  /// How many levels from the bottom of the tree use OBB instead of AABB.
  pub preciseness: u32,
  /// Total number of nodes in this BVH.
  pub node_count: u32,
  /// Total number of primitives referenced by this BVH.
  pub primitive_count: u32,
}

#[derive(Debug, Clone)]
/// TODO: Document this item
pub enum LinearBound<S>
where
  S: FloatLike + FloatOps + FloatBits + From<f32>,
{
  AABB(AABB<S>),
  OBB(OBB<S>),
}

#[derive(Debug, Clone)]
/// TODO: Document this item
pub struct LinearBVHNode<S>
where
  S: FloatLike + FloatOps + FloatBits + From<f32>,
{
  pub bound: LinearBound<S>,
  /// Index of the left child, or `u32::MAX` if leaf. Right child is always `left_child_or_primitive_offset + 1` if not leaf.
  /// If leaf, this is the offset into the primitives array.
  pub left_child_or_primitive_offset: u32,
  /// Index of the right child, or `u32::MAX` if leaf.
  pub right_child_offset: u32,
  /// Number of primitives in this leaf. 0 if not a leaf.
  pub primitive_count: u32,
  pub mass: f32,
  pub center_of_mass: [f32; 3],
}

#[derive(Debug, Clone)]
/// TODO: Document this item
pub struct LinearBVH<S>
where
  S: FloatLike + FloatOps + FloatBits + From<f32>,
{
  pub header: LinearBVHHeader,
  pub nodes: Vec<LinearBVHNode<S>>,
  pub primitives: Vec<usize>,
}

impl<S> LinearBVH<S>
where
  S: FloatLike + FloatOps + FloatBits + From<f32>,
{
  /// TODO: Document this item
  pub fn from_build_node(root: &BVHNode<S>, preciseness: u32) -> Self {
    let mut nodes = Vec::new();
    let mut primitives = Vec::new();

    // TODO do Morton order sorting
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

  pub fn to_multi_bvh<const N: usize>(
    &self,
  ) -> crate::math::collision::multi_bvh::MultiBvh<LinearBound<S>, usize, N>
  where
    LinearBound<S>: Clone,
  {
    crate::math::collision::multi_bvh::MultiBvh::build(self)
  }

  fn flatten_node(
    node: &BVHNode<S>,
    nodes: &mut Vec<LinearBVHNode<S>>,
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
      right_child_offset: u32::MAX,
      primitive_count: 0,
      mass: 0.0,
      center_of_mass: [0.0; 3],
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
      let mut right_idx = u32::MAX;
      if let Some(left) = &node.left {
        left_idx = Self::flatten_node(left, nodes, primitives);
      }
      if let Some(right) = &node.right {
        right_idx = Self::flatten_node(right, nodes, primitives);
      }

      nodes[current_idx as usize].left_child_or_primitive_offset = left_idx;
      nodes[current_idx as usize].right_child_offset = right_idx;
      nodes[current_idx as usize].primitive_count = 0;
    }

    current_idx
  }
}

impl<S> crate::math::collision::multi_bvh::BinaryBvh for LinearBVH<S>
where
  S: FloatLike + FloatOps + FloatBits + From<f32> + Clone,
  LinearBound<S>: Clone,
{
  type Bound = LinearBound<S>;
  type Primitive = usize;

  fn root(&self) -> Option<u32> {
    if self.nodes.is_empty() { None } else { Some(0) }
  }

  fn bound(&self, node_idx: u32) -> Self::Bound {
    self.nodes[node_idx as usize].bound.clone()
  }

  fn is_leaf(&self, node_idx: u32) -> bool {
    self.nodes[node_idx as usize].primitive_count > 0
  }

  fn children(&self, node_idx: u32) -> (Option<u32>, Option<u32>) {
    if self.is_leaf(node_idx) {
      return (None, None);
    }
    let n = &self.nodes[node_idx as usize];

    let l = if n.left_child_or_primitive_offset != u32::MAX {
      Some(n.left_child_or_primitive_offset)
    } else {
      None
    };
    let r = if n.right_child_offset != u32::MAX {
      Some(n.right_child_offset)
    } else {
      None
    };
    (l, r)
  }

  fn extract_primitives(&self, node_idx: u32, out: &mut Vec<Self::Primitive>) -> u32 {
    let n = &self.nodes[node_idx as usize];
    if n.primitive_count > 0 {
      let start = n.left_child_or_primitive_offset as usize;
      let end = start + n.primitive_count as usize;
      out.extend_from_slice(&self.primitives[start..end]);
      n.primitive_count
    } else {
      0
    }
  }
}

impl LinearBound<f32> {
  /// TODO: Document this item
  pub fn intersects(&self, other: &Self) -> bool {
    match (self, other) {
      (LinearBound::AABB(a), LinearBound::AABB(b)) => {
        crate::math::collision::intersection::intersect_aabb_aabb::<Vec3f32>(
          unsafe { core::mem::transmute(a) },
          unsafe { core::mem::transmute(b) },
        )
      }
      (LinearBound::OBB(a), LinearBound::OBB(b)) => {
        crate::math::collision::intersection::intersect_obb_obb::<f32, Vec3f32, Mat3f32>(
          unsafe { core::mem::transmute(a) },
          unsafe { core::mem::transmute(b) },
        )
      }
      (LinearBound::AABB(aabb), LinearBound::OBB(obb)) => {
        let aabb_as_obb = OBB::new(
          aabb.center::<Vec3f32>(),
          Mat3f32::identity(),
          aabb.half_extents::<Vec3f32>(),
        );
        crate::math::collision::intersection::intersect_obb_obb::<f32, Vec3f32, Mat3f32>(
          unsafe { core::mem::transmute(&aabb_as_obb) },
          unsafe { core::mem::transmute(obb) },
        )
      }
      (LinearBound::OBB(obb), LinearBound::AABB(aabb)) => {
        let aabb_as_obb = OBB::new(
          aabb.center::<Vec3f32>(),
          Mat3f32::identity(),
          aabb.half_extents::<Vec3f32>(),
        );
        crate::math::collision::intersection::intersect_obb_obb::<f32, Vec3f32, Mat3f32>(
          unsafe { core::mem::transmute(obb) },
          unsafe { core::mem::transmute(&aabb_as_obb) },
        )
      }
    }
  }
}

impl LinearBVH<f32> {
  /// Perform self-collision detection. Returns pairs of primitive indices that potentially intersect.
  pub fn get_self_collisions(&self) -> Vec<(usize, usize)> {
    let mut collisions = Vec::new();
    if self.nodes.is_empty() {
      return collisions;
    }

    let mut stack = Vec::new();
    // Compare root against its children, then traverse down
    if self.nodes[0].left_child_or_primitive_offset != u32::MAX
      && self.nodes[0].right_child_offset != u32::MAX
      && self.nodes[0].primitive_count == 0
    {
      stack.push((
        self.nodes[0].left_child_or_primitive_offset as usize,
        self.nodes[0].right_child_offset as usize,
      ));
    }

    while let Some((node_a_idx, node_b_idx)) = stack.pop() {
      let node_a = &self.nodes[node_a_idx];
      let node_b = &self.nodes[node_b_idx];

      if node_a.bound.intersects(&node_b.bound) {
        let a_is_leaf = node_a.primitive_count > 0;
        let b_is_leaf = node_b.primitive_count > 0;

        if a_is_leaf && b_is_leaf {
          let a_start = node_a.left_child_or_primitive_offset as usize;
          let a_end = a_start + node_a.primitive_count as usize;
          let b_start = node_b.left_child_or_primitive_offset as usize;
          let b_end = b_start + node_b.primitive_count as usize;

          for i in a_start..a_end {
            for j in b_start..b_end {
              // Usually you wouldn't return identical pairs if the BVH leaves can contain the same primitives,
              // but since it's a hierarchy, leaves don't overlap their primitive contents.
              collisions.push((self.primitives[i], self.primitives[j]));
            }
          }
        } else if a_is_leaf {
          if node_b.left_child_or_primitive_offset != u32::MAX {
            stack.push((node_a_idx, node_b.left_child_or_primitive_offset as usize));
          }
          if node_b.right_child_offset != u32::MAX {
            stack.push((node_a_idx, node_b.right_child_offset as usize));
          }
        } else if b_is_leaf {
          if node_a.left_child_or_primitive_offset != u32::MAX {
            stack.push((node_a.left_child_or_primitive_offset as usize, node_b_idx));
          }
          if node_a.right_child_offset != u32::MAX {
            stack.push((node_a.right_child_offset as usize, node_b_idx));
          }
        } else {
          // split the largest node, or just both
          if node_a.left_child_or_primitive_offset != u32::MAX
            && node_b.left_child_or_primitive_offset != u32::MAX
          {
            stack.push((
              node_a.left_child_or_primitive_offset as usize,
              node_b.left_child_or_primitive_offset as usize,
            ));
          }
          if node_a.left_child_or_primitive_offset != u32::MAX
            && node_b.right_child_offset != u32::MAX
          {
            stack.push((
              node_a.left_child_or_primitive_offset as usize,
              node_b.right_child_offset as usize,
            ));
          }
          if node_a.right_child_offset != u32::MAX
            && node_b.left_child_or_primitive_offset != u32::MAX
          {
            stack.push((
              node_a.right_child_offset as usize,
              node_b.left_child_or_primitive_offset as usize,
            ));
          }
          if node_a.right_child_offset != u32::MAX && node_b.right_child_offset != u32::MAX {
            stack.push((
              node_a.right_child_offset as usize,
              node_b.right_child_offset as usize,
            ));
          }
        }
      }

      // We also must traverse down within node A and B individually if they are not leaves!
      // Actually, standard self collision requires enqueuing the internal children pairs of each node!
      // But we only want to do that once per internal node.
      // So if node_a was just popped, we shouldn't enqueue its internal children here, unless we are traversing the tree systematically.
    }

    // Proper self-collision traversal of a single tree:
    let mut self_colls = Vec::new();
    let mut self_stack = Vec::new();
    if self.nodes.len() > 0 && self.nodes[0].primitive_count == 0 {
      self_stack.push(0usize);
    }

    while let Some(node_idx) = self_stack.pop() {
      let n = &self.nodes[node_idx];
      let left = n.left_child_or_primitive_offset as usize;
      let right = n.right_child_offset as usize;

      if left != usize::MAX && right != usize::MAX {
        // Enqueue children to check against each other
        let mut pair_stack = vec![(left, right)];
        while let Some((a_idx, b_idx)) = pair_stack.pop() {
          let a = &self.nodes[a_idx];
          let b = &self.nodes[b_idx];
          if a.bound.intersects(&b.bound) {
            let a_leaf = a.primitive_count > 0;
            let b_leaf = b.primitive_count > 0;

            if a_leaf && b_leaf {
              for i in 0..a.primitive_count as usize {
                for j in 0..b.primitive_count as usize {
                  self_colls.push((
                    self.primitives[a.left_child_or_primitive_offset as usize + i],
                    self.primitives[b.left_child_or_primitive_offset as usize + j],
                  ));
                }
              }
            } else if a_leaf {
              pair_stack.push((a_idx, b.left_child_or_primitive_offset as usize));
              pair_stack.push((a_idx, b.right_child_offset as usize));
            } else if b_leaf {
              pair_stack.push((a.left_child_or_primitive_offset as usize, b_idx));
              pair_stack.push((a.right_child_offset as usize, b_idx));
            } else {
              pair_stack.push((
                a.left_child_or_primitive_offset as usize,
                b.left_child_or_primitive_offset as usize,
              ));
              pair_stack.push((
                a.left_child_or_primitive_offset as usize,
                b.right_child_offset as usize,
              ));
              pair_stack.push((
                a.right_child_offset as usize,
                b.left_child_or_primitive_offset as usize,
              ));
              pair_stack.push((a.right_child_offset as usize, b.right_child_offset as usize));
            }
          }
        }

        // Traverse down the tree
        if self.nodes[left].primitive_count == 0 {
          self_stack.push(left);
        }
        if self.nodes[right].primitive_count == 0 {
          self_stack.push(right);
        }
      }
    }

    self_colls
  }

  /// Perform collision detection against another BVH.
  pub fn get_collisions_with(&self, other: &Self) -> Vec<(usize, usize)> {
    let mut collisions = Vec::new();
    if self.nodes.is_empty() || other.nodes.is_empty() {
      return collisions;
    }

    let mut stack = vec![(0, 0)];

    while let Some((node_a_idx, node_b_idx)) = stack.pop() {
      let node_a = &self.nodes[node_a_idx];
      let node_b = &other.nodes[node_b_idx];

      if node_a.bound.intersects(&node_b.bound) {
        let a_is_leaf = node_a.primitive_count > 0;
        let b_is_leaf = node_b.primitive_count > 0;

        if a_is_leaf && b_is_leaf {
          let a_start = node_a.left_child_or_primitive_offset as usize;
          let a_end = a_start + node_a.primitive_count as usize;
          let b_start = node_b.left_child_or_primitive_offset as usize;
          let b_end = b_start + node_b.primitive_count as usize;

          for i in a_start..a_end {
            for j in b_start..b_end {
              collisions.push((self.primitives[i], other.primitives[j]));
            }
          }
        } else if a_is_leaf {
          if node_b.left_child_or_primitive_offset != u32::MAX {
            stack.push((node_a_idx, node_b.left_child_or_primitive_offset as usize));
          }
          if node_b.right_child_offset != u32::MAX {
            stack.push((node_a_idx, node_b.right_child_offset as usize));
          }
        } else if b_is_leaf {
          if node_a.left_child_or_primitive_offset != u32::MAX {
            stack.push((node_a.left_child_or_primitive_offset as usize, node_b_idx));
          }
          if node_a.right_child_offset != u32::MAX {
            stack.push((node_a.right_child_offset as usize, node_b_idx));
          }
        } else {
          if node_a.left_child_or_primitive_offset != u32::MAX
            && node_b.left_child_or_primitive_offset != u32::MAX
          {
            stack.push((
              node_a.left_child_or_primitive_offset as usize,
              node_b.left_child_or_primitive_offset as usize,
            ));
          }
          if node_a.left_child_or_primitive_offset != u32::MAX
            && node_b.right_child_offset != u32::MAX
          {
            stack.push((
              node_a.left_child_or_primitive_offset as usize,
              node_b.right_child_offset as usize,
            ));
          }
          if node_a.right_child_offset != u32::MAX
            && node_b.left_child_or_primitive_offset != u32::MAX
          {
            stack.push((
              node_a.right_child_offset as usize,
              node_b.left_child_or_primitive_offset as usize,
            ));
          }
          if node_a.right_child_offset != u32::MAX && node_b.right_child_offset != u32::MAX {
            stack.push((
              node_a.right_child_offset as usize,
              node_b.right_child_offset as usize,
            ));
          }
        }
      }
    }

    collisions
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use alloc::vec;

  fn make_aabb(
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
  ) -> AABB<f32> {
    let mut min = [min_x, min_y, min_z];
    let mut max = [max_x, max_y, max_z];
    unsafe {
      core::mem::transmute::<[f32; 6], AABB<f32>>([min_x, min_y, min_z, max_x, max_y, max_z])
    }
  }

  fn make_obb(cx: f32, cy: f32, cz: f32, hx: f32, hy: f32, hz: f32) -> OBB<f32> {
    OBB::new(
      Vec3f32::from_components(cx, cy, cz),
      Mat3f32::identity(),
      Vec3f32::from_components(hx, hy, hz),
    )
  }

  #[test]
  fn test_bvh_self_collision() {
    // Manually construct a small BVH with intersecting leaves
    let mut bvh: LinearBVH<f32> = LinearBVH {
      header: LinearBVHHeader {
        preciseness: 0,
        node_count: 3,
        primitive_count: 2,
      },
      nodes: vec![
        LinearBVHNode {
          center_of_mass: [0.0, 0.0, 0.0],
          mass: 0.0,
          bound: LinearBound::AABB(make_aabb(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)),
          left_child_or_primitive_offset: 1,
          right_child_offset: 2,
          primitive_count: 0,
        },
        LinearBVHNode {
          center_of_mass: [0.0, 0.0, 0.0],
          mass: 0.0,
          bound: LinearBound::AABB(make_aabb(0.0, 0.0, 0.0, 6.0, 6.0, 6.0)),
          left_child_or_primitive_offset: 0,
          right_child_offset: u32::MAX,
          primitive_count: 1,
        },
        LinearBVHNode {
          center_of_mass: [0.0, 0.0, 0.0],
          mass: 0.0,
          bound: LinearBound::AABB(make_aabb(5.0, 5.0, 5.0, 10.0, 10.0, 10.0)),
          left_child_or_primitive_offset: 1,
          right_child_offset: u32::MAX,
          primitive_count: 1,
        },
      ],
      primitives: vec![100, 101],
    };

    let colls = bvh.get_self_collisions();
    assert_eq!(colls.len(), 1);
    assert!(colls.contains(&(100, 101)) || colls.contains(&(101, 100)));

    // No intersection
    bvh.nodes[1].bound = LinearBound::AABB(make_aabb(0.0, 0.0, 0.0, 4.0, 4.0, 4.0));
    let colls = bvh.get_self_collisions();
    assert!(colls.is_empty());
  }

  #[test]
  fn test_bvh_with_other() {
    let bvh1: LinearBVH<f32> = LinearBVH {
      header: LinearBVHHeader {
        preciseness: 0,
        node_count: 1,
        primitive_count: 1,
      },
      nodes: vec![LinearBVHNode {
        center_of_mass: [0.0, 0.0, 0.0],
        mass: 0.0,
        bound: LinearBound::OBB(make_obb(0.0, 0.0, 0.0, 5.0, 5.0, 5.0)),
        left_child_or_primitive_offset: 0,
        right_child_offset: u32::MAX,
        primitive_count: 1,
      }],
      primitives: vec![42],
    };

    let bvh2: LinearBVH<f32> = LinearBVH {
      header: LinearBVHHeader {
        preciseness: 0,
        node_count: 1,
        primitive_count: 1,
      },
      nodes: vec![LinearBVHNode {
        center_of_mass: [0.0, 0.0, 0.0],
        mass: 0.0,
        bound: LinearBound::AABB(make_aabb(3.0, 0.0, 0.0, 12.0, 5.0, 5.0)),
        left_child_or_primitive_offset: 0,
        right_child_offset: u32::MAX,
        primitive_count: 1,
      }],
      primitives: vec![99],
    };

    let colls = bvh1.get_collisions_with(&bvh2);
    assert_eq!(colls.len(), 1);
    assert_eq!(colls[0], (42, 99));

    let bvh3: LinearBVH<f32> = LinearBVH {
      header: LinearBVHHeader {
        preciseness: 0,
        node_count: 1,
        primitive_count: 1,
      },
      nodes: vec![LinearBVHNode {
        center_of_mass: [0.0, 0.0, 0.0],
        mass: 0.0,
        bound: LinearBound::OBB(make_obb(12.0, 0.0, 0.0, 1.0, 1.0, 1.0)),
        left_child_or_primitive_offset: 0,
        right_child_offset: u32::MAX,
        primitive_count: 1,
      }],
      primitives: vec![88],
    };

    let colls2 = bvh1.get_collisions_with(&bvh3);
    assert!(colls2.is_empty());
  }
}
