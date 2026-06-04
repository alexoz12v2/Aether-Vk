//! linear_bvh module.

use crate::{
  math::collision::{
    bounds::{AABB, OBB},
    bvh_builder::{BVHNode, BoundNode},
    multi_bvh::TlasMultiNode,
  },
  scene::EntityId,
  simulation::comet::Vertex,
  simulation_api::structs::SceneContext,
};
use aethervk_oshal_rlib::math::{
  FloatLike,
  floating::{FloatBits, FloatOps},
  matrix::{Matrix4, MatrixVectorMul, SquareMatrix, mat3::Mat3f32, mat4::Mat4x4f32},
  quaternion::Quaternion,
  vector::{Vector, Vector3, Vector4, vec3::Vec3f32, vec4::Vec4f32},
};
use alloc::{vec, vec::Vec};

/// TODO: Document this item
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
  S: FloatLike + FloatOps + FloatBits + From<f32> + core::fmt::Debug,
{
  pub header: LinearBVHHeader,
  pub nodes: Vec<LinearBVHNode<S>>,
  pub primitives: Vec<usize>,
}

impl<S> LinearBVH<S>
where
  S: FloatLike + FloatOps + FloatBits + From<f32> + core::fmt::Debug,
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

  pub fn find_path_to_primitive(&self, target_prim_idx: usize) -> Option<alloc::vec::Vec<u32>> {
    if self.nodes.is_empty() {
      return None;
    }

    let mut stack = alloc::vec::Vec::new();
    stack.push((0u32, false)); // (node_idx, visited_children)
    let mut path = alloc::vec::Vec::new();

    while let Some(&(node_idx, visited_children)) = stack.last() {
      if !visited_children {
        path.push(node_idx);
        let node = &self.nodes[node_idx as usize];

        if node.primitive_count > 0 {
          let start = node.left_child_or_primitive_offset as usize;
          let end = start + node.primitive_count as usize;
          let mut found = false;
          for i in start..end {
            if self.primitives[i] == target_prim_idx {
              found = true;
              break;
            }
          }
          if found {
            return Some(path);
          }
          path.pop();
          stack.pop();
        } else {
          stack.last_mut().unwrap().1 = true;
          let right = node.right_child_offset;
          if right != u32::MAX {
            stack.push((right, false));
          }
          let left = node.left_child_or_primitive_offset;
          if left != u32::MAX {
            stack.push((left, false));
          }
        }
      } else {
        path.pop();
        stack.pop();
      }
    }
    None
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
  S: FloatLike + FloatOps + FloatBits + From<f32> + Clone + core::fmt::Debug,
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

  #[inline(always)]
  fn expand_bits(mut v: u32) -> u32 {
    v = (v.wrapping_mul(0x00010001)) & 0xFF0000FF;
    v = (v.wrapping_mul(0x00000101)) & 0x0F00F00F;
    v = (v.wrapping_mul(0x00000011)) & 0xC30C30C3;
    v = (v.wrapping_mul(0x00000005)) & 0x49249249;
    v
  }

  fn morton_3d(normalized_pos: Vec3f32) -> u32 {
    let x = (normalized_pos.x() * 1023.0).clamp(0.0, 1023.0) as u32;
    let y = (normalized_pos.y() * 1023.0).clamp(0.0, 1023.0) as u32;
    let z = (normalized_pos.z() * 1023.0).clamp(0.0, 1023.0) as u32;

    let xx = Self::expand_bits(x);
    let yy = Self::expand_bits(y);
    let zz = Self::expand_bits(z);

    (xx << 2) | (yy << 1) | zz
  }

  fn common_prefix(morton_codes: &[(u32, usize)], i: isize, j: isize) -> i32 {
    if j < 0 || j >= morton_codes.len() as isize {
      return -1;
    }
    let key1 = morton_codes[i as usize].0;
    let key2 = morton_codes[j as usize].0;

    if key1 == key2 {
      let idx1 = morton_codes[i as usize].1 as u32;
      let idx2 = morton_codes[j as usize].1 as u32;
      return 32 + (idx1 ^ idx2).leading_zeros() as i32;
    }

    (key1 ^ key2).leading_zeros() as i32
  }

  fn determine_range(morton_codes: &[(u32, usize)], i: usize) -> (usize, usize) {
    let i_isize = i as isize;
    let d = (Self::common_prefix(morton_codes, i_isize, i_isize + 1)
      - Self::common_prefix(morton_codes, i_isize, i_isize - 1))
    .signum() as isize;
    let min_prefix = Self::common_prefix(morton_codes, i_isize, i_isize - d);

    let mut l_max = 2;
    while Self::common_prefix(morton_codes, i_isize, i_isize + l_max * d) > min_prefix {
      l_max *= 2;
    }

    let mut l = 0;
    let mut t = l_max / 2;
    while t >= 1 {
      if Self::common_prefix(morton_codes, i_isize, i_isize + (l + t) * d) > min_prefix {
        l += t;
      }
      t /= 2;
    }

    let j = i_isize + l * d;
    let min_idx = i_isize.min(j) as usize;
    let max_idx = i_isize.max(j) as usize;

    (min_idx, max_idx)
  }

  fn find_split(morton_codes: &[(u32, usize)], first: usize, last: usize) -> usize {
    let first_isize = first as isize;
    let last_isize = last as isize;
    let common_prefix_node = Self::common_prefix(morton_codes, first_isize, last_isize);

    let mut split = first_isize;
    let mut step = last_isize - first_isize;

    loop {
      step = (step + 1) >> 1;
      let new_split = split + step;

      if new_split < last_isize {
        let split_prefix = Self::common_prefix(morton_codes, first_isize, new_split);
        if split_prefix > common_prefix_node {
          split = new_split;
        }
      }
      if step <= 1 {
        break;
      }
    }
    split as usize
  }

  pub fn build_mesh_lbvh(vertices: &[Vertex], indices: &[u32]) -> Option<LinearBVH<f32>> {
    let num_triangles = indices.len() / 3;
    if num_triangles == 0 {
      return None;
    }

    let get_vertex_pos = |idx: u32| {
      let v = &vertices[idx as usize];
      Vec3f32::from_components(v.position[0], v.position[1], v.position[2])
    };

    let mut min_b = get_vertex_pos(indices[0]);
    let mut max_b = min_b;
    for &idx in indices {
      let pos = get_vertex_pos(idx);
      min_b = min_b.min(pos);
      max_b = max_b.max(pos);
    }
    let extent = max_b - min_b;

    let mut morton_entries = Vec::with_capacity(num_triangles);
    for i in 0..num_triangles {
      let v0 = get_vertex_pos(indices[i * 3]);
      let v1 = get_vertex_pos(indices[i * 3 + 1]);
      let v2 = get_vertex_pos(indices[i * 3 + 2]);
      let centroid = (v0 + v1 + v2) * (1.0 / 3.0);

      let normalized = if extent.length_squared() > 1e-8 {
        Vec3f32::from_components(
          (centroid.x() - min_b.x()) / extent.x(),
          (centroid.y() - min_b.y()) / extent.y(),
          (centroid.z() - min_b.z()) / extent.z(),
        )
      } else {
        Vec3f32::zero()
      };

      let code = Self::morton_3d(normalized);
      morton_entries.push((code, i));
    }

    morton_entries.sort_unstable_by_key(|&(code, _)| code);

    let n = num_triangles;
    let num_internal_nodes = n.saturating_sub(1);
    let num_total_nodes = if n == 1 { 1 } else { 2 * n - 1 };

    let dummy_node = LinearBVHNode {
      center_of_mass: [0.0, 0.0, 0.0],
      mass: 0.0,
      bound: LinearBound::AABB(AABB::new(Vec3f32::zero(), Vec3f32::zero())),
      left_child_or_primitive_offset: u32::MAX,
      right_child_offset: u32::MAX,
      primitive_count: 0,
    };

    let mut nodes = vec![dummy_node; num_total_nodes];

    if n == 1 {
      let orig = morton_entries[0].1;
      let v0 = get_vertex_pos(indices[orig * 3]);
      let v1 = get_vertex_pos(indices[orig * 3 + 1]);
      let v2 = get_vertex_pos(indices[orig * 3 + 2]);
      let min_bound = v0.min(v1).min(v2);
      let max_bound = v0.max(v1).max(v2);

      nodes[0].bound = LinearBound::AABB(AABB::new(min_bound, max_bound));
      nodes[0].left_child_or_primitive_offset = 0;
      nodes[0].primitive_count = 1;

      return Some(LinearBVH {
        header: LinearBVHHeader {
          preciseness: 0,
          node_count: 1,
          primitive_count: 1,
        },
        nodes,
        primitives: vec![orig],
      });
    }

    for i in 0..num_internal_nodes {
      let (first, last) = Self::determine_range(&morton_entries, i);
      let split = Self::find_split(&morton_entries, first, last);

      let left_child = if split == first {
        num_internal_nodes + split
      } else {
        split
      };
      let right_child = if split + 1 == last {
        num_internal_nodes + split + 1
      } else {
        split + 1
      };

      nodes[i].left_child_or_primitive_offset = left_child as u32;
      nodes[i].right_child_offset = right_child as u32;
    }

    let mut primitives = Vec::with_capacity(n);
    for i in 0..n {
      let leaf_idx = num_internal_nodes + i;
      let orig = morton_entries[i].1;

      let v0 = get_vertex_pos(indices[orig * 3]);
      let v1 = get_vertex_pos(indices[orig * 3 + 1]);
      let v2 = get_vertex_pos(indices[orig * 3 + 2]);
      let min_bound = v0.min(v1).min(v2);
      let max_bound = v0.max(v1).max(v2);

      nodes[leaf_idx].bound = LinearBound::AABB(AABB::new(min_bound, max_bound));
      nodes[leaf_idx].left_child_or_primitive_offset = i as u32;
      nodes[leaf_idx].primitive_count = 1;

      primitives.push(orig);
    }

    for i in (0..num_internal_nodes).rev() {
      let left_idx = nodes[i].left_child_or_primitive_offset as usize;
      let right_idx = nodes[i].right_child_offset as usize;

      let mut aabb = match &nodes[left_idx].bound {
        LinearBound::AABB(b) => b.clone(),
        _ => unreachable!(),
      };

      match &nodes[right_idx].bound {
        LinearBound::AABB(b) => aabb.encapsulate_aabb::<Vec3f32>(b),
        _ => unreachable!(),
      };

      nodes[i].bound = LinearBound::AABB(aabb);
    }

    Some(LinearBVH {
      header: LinearBVHHeader {
        preciseness: 0,
        node_count: num_total_nodes as u32,
        primitive_count: n as u32,
      },
      nodes,
      primitives,
    })
  }

  pub fn raycast(
    &self,
    ray_origin: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    ray_dir: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    mesh_vertices: &[Vertex],
    mesh_indices: &[u32],
  ) -> Option<(
    f32,
    aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
  )> {
    if self.nodes.is_empty() {
      return None;
    }

    let inv_dir = Vec3f32::from_components(
      if ray_dir.x() != 0.0 {
        1.0 / ray_dir.x()
      } else {
        f32::MAX
      },
      if ray_dir.y() != 0.0 {
        1.0 / ray_dir.y()
      } else {
        f32::MAX
      },
      if ray_dir.z() != 0.0 {
        1.0 / ray_dir.z()
      } else {
        f32::MAX
      },
    );

    let mut stack = vec![0usize];
    let mut closest_t = f32::MAX;
    let mut closest_point = Vec3f32::zero();
    let mut closest_normal = Vec3f32::zero();
    let mut hit_anything = false;

    while let Some(node_idx) = stack.pop() {
      let node = &self.nodes[node_idx];

      let mut tmin = 0.0f32;
      let mut tmax = f32::MAX;
      if let LinearBound::AABB(aabb) = &node.bound {
        let bmin = aabb.min::<Vec3f32>();
        let bmax = aabb.max::<Vec3f32>();

        let tx1 = (bmin.x() - ray_origin.x()) * inv_dir.x();
        let tx2 = (bmax.x() - ray_origin.x()) * inv_dir.x();
        tmin = tmin.max(tx1.min(tx2));
        tmax = tmax.min(tx1.max(tx2));

        let ty1 = (bmin.y() - ray_origin.y()) * inv_dir.y();
        let ty2 = (bmax.y() - ray_origin.y()) * inv_dir.y();
        tmin = tmin.max(ty1.min(ty2));
        tmax = tmax.min(ty1.max(ty2));

        let tz1 = (bmin.z() - ray_origin.z()) * inv_dir.z();
        let tz2 = (bmax.z() - ray_origin.z()) * inv_dir.z();
        tmin = tmin.max(tz1.min(tz2));
        tmax = tmax.min(tz1.max(tz2));
      }

      if tmax < tmin || tmin >= closest_t {
        continue;
      }

      if node.primitive_count > 0 {
        for i in 0..node.primitive_count as usize {
          let offset = self.primitives[node.left_child_or_primitive_offset as usize + i];
          let v0_idx = mesh_indices[offset * 3] as usize;
          let v1_idx = mesh_indices[offset * 3 + 1] as usize;
          let v2_idx = mesh_indices[offset * 3 + 2] as usize;

          let v0_pos = mesh_vertices[v0_idx].position;
          let v1_pos = mesh_vertices[v1_idx].position;
          let v2_pos = mesh_vertices[v2_idx].position;

          let v0 = Vec3f32::from_components(v0_pos[0], v0_pos[1], v0_pos[2]);
          let v1 = Vec3f32::from_components(v1_pos[0], v1_pos[1], v1_pos[2]);
          let v2 = Vec3f32::from_components(v2_pos[0], v2_pos[1], v2_pos[2]);

          let edge1 = v1 - v0;
          let edge2 = v2 - v0;
          let h = ray_dir.cross(edge2);
          let a = edge1.dot(h);

          if a.abs() > 1e-8 {
            let f = 1.0 / a;
            let s = ray_origin - v0;
            let u = f * s.dot(h);
            if u >= 0.0 && u <= 1.0 {
              let q = s.cross(edge1);
              let v = f * ray_dir.dot(q);
              if v >= 0.0 && u + v <= 1.0 {
                let t = f * edge2.dot(q);
                if t > 0.0 && t < closest_t {
                  closest_t = t;
                  closest_point = ray_origin + ray_dir * t;
                  closest_normal = edge1.cross(edge2).normalize();
                  hit_anything = true;
                }
              }
            }
          }
        }
      } else {
        stack.push(node.left_child_or_primitive_offset as usize);
        stack.push(node.right_child_offset as usize);
      }
    }

    if hit_anything {
      Some((closest_t, closest_point, closest_normal))
    } else {
      None
    }
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

pub fn raycast_scene(
  scene: &SceneContext,
  tlas: &[TlasMultiNode<32>],
  ray_origin: Vec3f32,
  ray_dir: Vec3f32,
) -> Option<(crate::scene::EntityId, f32, Vec3f32, Vec3f32)> {
  if tlas.is_empty() {
    return None;
  }

  let inv_dir = Vec3f32::from_components(
    if ray_dir.x() != 0.0 {
      1.0 / ray_dir.x()
    } else {
      f32::MAX
    },
    if ray_dir.y() != 0.0 {
      1.0 / ray_dir.y()
    } else {
      f32::MAX
    },
    if ray_dir.z() != 0.0 {
      1.0 / ray_dir.z()
    } else {
      f32::MAX
    },
  );

  let mut stack = vec![0u32];
  let mut closest_t = f32::MAX;
  let mut closest_point = Vec3f32::zero();
  let mut closest_normal = Vec3f32::zero();
  let mut closest_entity = None;

  while let Some(node_idx) = stack.pop() {
    let node = &tlas[node_idx as usize];

    for i in 0..32 {
      let meta = node.metadata[i];
      if meta == 0 {
        continue; // Empty slot
      }

      let tx1 = (node.min_x[i] - ray_origin.x()) * inv_dir.x();
      let tx2 = (node.max_x[i] - ray_origin.x()) * inv_dir.x();
      let mut tmin = tx1.min(tx2);
      let mut tmax = tx1.max(tx2);

      let ty1 = (node.min_y[i] - ray_origin.y()) * inv_dir.y();
      let ty2 = (node.max_y[i] - ray_origin.y()) * inv_dir.y();
      tmin = tmin.max(ty1.min(ty2));
      tmax = tmax.min(ty1.max(ty2));

      let tz1 = (node.min_z[i] - ray_origin.z()) * inv_dir.z();
      let tz2 = (node.max_z[i] - ray_origin.z()) * inv_dir.z();
      tmin = tmin.max(tz1.min(tz2));
      tmax = tmax.min(tz1.max(tz2));

      if tmax >= tmin && tmin < closest_t && tmax > 0.0 {
        if (meta & 0x8000_0000) != 0 {
          // Leaf node
          let entity_id_val =
            (node.child_indices[i] as u64) | (((meta & 0x7FFF_FFFF) as u64) << 32);
          let key_data = slotmap::KeyData::from_ffi(entity_id_val);
          let entity_id = crate::scene::EntityId::from(key_data);

          if let Some(transform) = scene.scene.global_transform(entity_id) {
            scene
              .scene
              .with_component(entity_id, |mesh: &crate::scene::PhysicalMeshComponent| {
                use aethervk_oshal_rlib::math::{
                  matrix::{Matrix4, MatrixVectorMul, SquareMatrix},
                  vector::{Vector, Vector3, Vector4},
                };

                let model_matrix = <Mat4x4f32 as Matrix4>::translation(transform.position)
                  * <Mat4x4f32 as Matrix4>::from_quat_custom_frame(transform.rotation)
                  * <Mat4x4f32 as Matrix4>::from_scale(transform.scale);

                let inv_model = model_matrix
                  .inverse()
                  .unwrap_or_else(|| <Mat4x4f32 as SquareMatrix>::identity());

                let ro_local_v4 = inv_model.mul_vector(<Vec4f32 as Vector4>::from_components(
                  ray_origin.x(),
                  ray_origin.y(),
                  ray_origin.z(),
                  1.0,
                ));
                let ro_local =
                  Vec3f32::from_components(ro_local_v4.x(), ro_local_v4.y(), ro_local_v4.z());

                let rd_local_v4 = inv_model.mul_vector(<Vec4f32 as Vector4>::from_components(
                  ray_dir.x(),
                  ray_dir.y(),
                  ray_dir.z(),
                  0.0,
                ));
                let rd_local =
                  Vec3f32::from_components(rd_local_v4.x(), rd_local_v4.y(), rd_local_v4.z())
                    .normalize();

                if let Some(bvh) = &mesh.mesh.bvh {
                  if let Some((_, p_loc, n_loc)) =
                    bvh.raycast(ro_local, rd_local, &mesh.mesh.vertices, &mesh.mesh.indices)
                  {
                    let p_world_v4 = model_matrix.mul_vector(
                      <Vec4f32 as Vector4>::from_components(p_loc.x(), p_loc.y(), p_loc.z(), 1.0),
                    );
                    let p_world =
                      Vec3f32::from_components(p_world_v4.x(), p_world_v4.y(), p_world_v4.z());

                    let n_world_v4 = model_matrix.mul_vector(
                      <Vec4f32 as Vector4>::from_components(n_loc.x(), n_loc.y(), n_loc.z(), 0.0),
                    );
                    let n_world =
                      Vec3f32::from_components(n_world_v4.x(), n_world_v4.y(), n_world_v4.z())
                        .normalize();

                    let t_world = (p_world - ray_origin).length();

                    if t_world < closest_t {
                      closest_t = t_world;
                      closest_point = p_world;
                      closest_normal = n_world;
                      closest_entity = Some(entity_id);
                    }
                  }
                }
              });
          }
        } else {
          // Internal node
          stack.push(node.child_indices[i]);
        }
      }
    }
  }

  if closest_t < f32::MAX {
    Some((
      closest_entity.unwrap(),
      closest_t,
      closest_point,
      closest_normal,
    ))
  } else {
    None
  }
}