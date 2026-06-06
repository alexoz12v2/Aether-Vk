//! Physics Scene Top-Level Acceleration Structure (TLAS) Multi-BVH
//!
//! The TLAS is an N-ary hierarchy optimized for GPU subgroup traversal, where
//! the branching factor `N` equals the hardware `SUBGROUP_SIZE` (e.g., 32 or 64).
//!
//! ## Coordinate Systems & Scale Domains
//! - **Macro Frame**: World space (+x=right, -y=forward, +z=up).
//!   - Length: Astronomical Units (AU) | Mass: Earth Masses ($M_\oplus$).
//! - **Micro Frame**: Nested Local Coordinate Area (LCA).
//!   - Length: kilometers (km) | Mass: kilograms (kg).
//!
//! ## Node Data Layout (SoA)
//! The node uses a Structure of Arrays (SoA) layout so each lane `i` in the subgroup
//! fetches the bounds of `child[i]` with perfectly coalesced memory access.
//!
//! ```text
//! +----------------------------------------------------------------------------------+
//! | TlasMultiNode<const N: usize>                                                    |
//! +----------------------------------------------------------------------------------+
//! | [AABB Bounds SoA]                                                                |
//! | - min_x: [f32; N]     - min_y: [f32; N]     - min_z: [f32; N]                    |
//! | - max_x: [f32; N]     - max_y: [f32; N]     - max_z: [f32; N]                    |
//! +----------------------------------------------------------------------------------+
//! | [Connectivity & Metadata]                                                        |
//! | - child_indices: [u32; N] (Index to child TLAS node OR BLAS Instance Array)      |
//! | - metadata:      [u32; N] (Packed bitfield describing the leaf payload)          |
//! +----------------------------------------------------------------------------------+
//! | [Traversal Optimization]                                                         |
//! | - valid_mask: u64 (Bitmask of active children, 1 bit per child up to N=64)       |
//! | - permutations: [u32; 8] (Precomputed orderings for 8 ray sign combinations)     |
//! +----------------------------------------------------------------------------------+
//! | Optional Parallel Buffer: Instance Descriptor Array (indexed if IsLeaf == 1)     |
//! | - transform: Mat4x4 (Affine matrix mapping Object Space -> Macro Space)          |
//! | - blas_root_ptr: u32 (Pointer to the BLAS in device memory)                      |
//! | - shape_data: Vec4 (Extents/Radius if OBB/Sphere)                                |
//! +----------------------------------------------------------------------------------+
//! ```
//!
//! ## Metadata Bitfield Specification (`metadata[i]`)
//! ```text
//! 31       30       28       26                                                0
//! +--------+--------+--------+-------------------------------------------------+
//! | IsLeaf |  Frame | Shape  | Child Node Index / Instance Descriptor Index    |
//! |  (1b)  |  (2b)  |  (2b)  |                      (27b)                      |
//! +--------+--------+--------+-------------------------------------------------+
//! ```

use alloc::{vec, vec::Vec};
use core::array;

// --- MULTI-BRANCH BVH STRUCTURES ---

/// A node in a Multi-Branch BVH tree that stores up to N children.
/// Designed natively for SIMD processing (Structure of Arrays).
#[derive(Debug, Clone)]
pub struct MultiBvhNode<B, const N: usize> {
  /// Bounds for each child. Unused slots gracefully duplicate the parent's bound.
  pub bounds: [B; N],
  /// If `is_leaf[i]` is false, this is the index of the internal child `MultiBvhNode`.
  /// If `is_leaf[i]` is true, this is the offset into the `primitives` array.
  pub child_or_primitive_offsets: [u32; N],
  /// The number of primitives in the child leaf. 0 means it's an internal node or unused.
  pub primitive_counts: [u32; N],
  /// Indicates whether the corresponding child is a leaf.
  pub is_leaf: [bool; N],
  /// The actual number of valid children in this node (0 to N).
  pub valid_count: u32,
  /// Mass of the node
  pub masses: [f32; N],
  /// Center of mass of the node
  pub centers_of_mass: [[f32; 3]; N],
}

/// A generic Multi-Branch BVH constructed from any `BinaryBvh`.
#[derive(Debug, Clone)]
pub struct MultiBvh<B, P, const N: usize> {
  pub nodes: Vec<MultiBvhNode<B, N>>,
  pub primitives: Vec<P>,
  pub root: Option<u32>,
}

/// Dynamic Subgroup-Size Multi-BVH representation.
#[derive(Debug, Clone)]
pub enum MeshMultiBvh {
  Bvh4(MultiBvh<crate::math::collision::linear_bvh::LinearBound<f32>, usize, 4>),
  Bvh8(MultiBvh<crate::math::collision::linear_bvh::LinearBound<f32>, usize, 8>),
  Bvh16(MultiBvh<crate::math::collision::linear_bvh::LinearBound<f32>, usize, 16>),
  Bvh32(MultiBvh<crate::math::collision::linear_bvh::LinearBound<f32>, usize, 32>),
  Bvh64(MultiBvh<crate::math::collision::linear_bvh::LinearBound<f32>, usize, 64>),
  Bvh128(MultiBvh<crate::math::collision::linear_bvh::LinearBound<f32>, usize, 128>),
}

// --- BINARY BVH ABSTRACTION TRAIT ---

/// A unified trait to inspect any binary BVH natively.
pub trait BinaryBvh: core::fmt::Debug {
  type Bound: Clone;
  type Primitive: Clone;

  fn root(&self) -> Option<u32>;
  fn bound(&self, node_idx: u32) -> Self::Bound;
  fn is_leaf(&self, node_idx: u32) -> bool;
  fn children(&self, node_idx: u32) -> (Option<u32>, Option<u32>);
  fn mass(&self, _node_idx: u32) -> f32 {
    0.0
  }
  fn center_of_mass(&self, _node_idx: u32) -> [f32; 3] {
    [0.0; 3]
  }

  /// Returns the split axis (0 for X, 1 for Y, 2 for Z) used to partition the children.
  /// Used for generating sign heuristic permutations in Multi-BVH conversion.
  fn split_axis(&self, _node_idx: u32) -> u32 {
    0
  }

  /// Optional metadata for leaves
  fn leaf_meta(&self, _node_idx: u32) -> Option<u32> {
    None
  }

  /// Extracts primitives of the leaf at `node_idx` and appends them to `out`.
  /// Returns the number of primitives appended.
  fn extract_primitives(&self, node_idx: u32, out: &mut Vec<Self::Primitive>) -> u32;
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TlasMultiNode<const N: usize> {
  pub min_x: [f32; N],
  pub max_x: [f32; N],
  pub min_y: [f32; N],
  pub max_y: [f32; N],
  pub min_z: [f32; N],
  pub max_z: [f32; N],
  pub child_indices: [u32; N],
  pub metadata: [u32; N],
  pub masses: [f32; N],
  pub com_x: [f32; N],
  pub com_y: [f32; N],
  pub com_z: [f32; N],
  pub particle_start: [u32; N],
  pub particle_count: [u32; N],
  pub valid_mask: [u32; 2],
  pub parent_idx: u32,
  pub _pad: u32,
  /// Precomputed traversal orderings for the 8 ray-sign combinations.
  /// `permutations[sign_mask][i]` = local child index to visit i-th.
  /// Stored as `u32` (upper 24 bits unused) so the struct is `bytemuck::Pod`.
  pub permutations: [[u32; N]; 8],
}

impl<const N: usize> Default for TlasMultiNode<N> {
  fn default() -> Self {
    Self {
      min_x: [0.0; N],
      max_x: [0.0; N],
      min_y: [0.0; N],
      max_y: [0.0; N],
      min_z: [0.0; N],
      max_z: [0.0; N],
      child_indices: [u32::MAX; N],
      metadata: [0; N],
      masses: [0.0; N],
      com_x: [0.0; N],
      com_y: [0.0; N],
      com_z: [0.0; N],
      particle_start: [0; N],
      particle_count: [0; N],
      valid_mask: [0; 2],
      parent_idx: u32::MAX,
      _pad: 0,
      permutations: [[0; N]; 8],
    }
  }
}

// ── `Into<[f32; 6]>` for concrete bound types ────────────────────────────────
// Required by `convert_binary_to_multi_bvh`'s `T::Bound: Into<[f32; 6]>` bound.

use crate::math::collision::{bounds::AABB, linear_bvh::LinearBound};
use aethervk_oshal_rlib::math::vector::{Vector, Vector3, vec3::Vec3f32 as Vec3f32Alias};

impl From<AABB<f32>> for [f32; 6] {
  fn from(val: AABB<f32>) -> Self {
    let mn: Vec3f32Alias = val.min();
    let mx: Vec3f32Alias = val.max();
    [mn.x(), mn.y(), mn.z(), mx.x(), mx.y(), mx.z()]
  }
}

impl From<LinearBound<f32>> for [f32; 6] {
  fn from(val: LinearBound<f32>) -> Self {
    match val {
      LinearBound::AABB(a) => a.into(),
      LinearBound::OBB(o) => {
        let a = o.to_aabb::<Vec3f32Alias>();
        let mn: Vec3f32Alias = a.min();
        let mx: Vec3f32Alias = a.max();
        [mn.x(), mn.y(), mn.z(), mx.x(), mx.y(), mx.z()]
      }
    }
  }
}

// ── bytemuck::Pod + Zeroable for TlasMultiNode at the three used sizes ───────
// Safety: TlasMultiNode<N> is #[repr(C)], all fields are f32/u32 (Pod), and the
// permutations field was changed from [[u8;N];8] to [[u32;N];8] to ensure no
// unexpected padding or alignment issues.

// SAFETY: TlasMultiNode<N> is #[repr(C)] with only Pod fields, for N ∈ {4,8,16,32,64,128}.
unsafe impl bytemuck::Zeroable for TlasMultiNode<4> {}
unsafe impl bytemuck::Zeroable for TlasMultiNode<8> {}
unsafe impl bytemuck::Zeroable for TlasMultiNode<16> {}
unsafe impl bytemuck::Zeroable for TlasMultiNode<32> {}
unsafe impl bytemuck::Zeroable for TlasMultiNode<64> {}
unsafe impl bytemuck::Zeroable for TlasMultiNode<128> {}
unsafe impl bytemuck::Pod for TlasMultiNode<4> {}
unsafe impl bytemuck::Pod for TlasMultiNode<8> {}
unsafe impl bytemuck::Pod for TlasMultiNode<16> {}
unsafe impl bytemuck::Pod for TlasMultiNode<32> {}
unsafe impl bytemuck::Pod for TlasMultiNode<64> {}
unsafe impl bytemuck::Pod for TlasMultiNode<128> {}

///
/// # Sign Heuristic & Permutation Ordering
/// When collapsing a binary treelet into a wide node (e.g., N=32), traversing the children
/// front-to-back dynamically based on distances is expensive ($O(N \log N)$ sorting).
/// Instead, we use the **Ray Direction Sign Heuristic**:
/// 1. A binary node was split along a specific axis (X, Y, or Z).
/// 2. If a ray's direction sign for that axis is positive, the left child should be visited before the right.
/// 3. If negative, the right child should be visited first.
///
/// By evaluating this across the 8 possible ray sign combinations (`[+/-X, +/-Y, +/-Z]`),
/// we precompute 8 exact front-to-back traversal sequences for the `N` children of this wide node.
/// At runtime, the GPU simply selects the precomputed permutation array using the ray's signs in $O(1)$ time.
pub fn convert_binary_to_multi_bvh<const N: usize, T: BinaryBvh>(
  binary_bvh: &T,
) -> Vec<TlasMultiNode<N>>
where
  T::Bound: Into<[f32; 6]>, // Assuming Bound can be converted to [min_x, min_y, min_z, max_x, max_y, max_z]
{
  assert!(
    N > 1 && N.is_power_of_two() && N <= 128,
    "N must be a power of two between 2 and 128"
  );

  let mut multi_nodes = Vec::new();

  if let Some(binary_root) = binary_bvh.root() {
    collapse_binary_to_multi_recursive::<N, T>(binary_root, binary_bvh, &mut multi_nodes, 0);
  }

  multi_nodes
}

fn collapse_binary_to_multi_recursive<const N: usize, T: BinaryBvh>(
  binary_idx: u32,
  binary_bvh: &T,
  multi_nodes: &mut Vec<TlasMultiNode<N>>,
  depth: u32,
) -> u32
where
  T::Bound: Into<[f32; 6]>,
{
  let multi_idx = multi_nodes.len() as u32;
  multi_nodes.push(TlasMultiNode::default());

  #[derive(Clone, Copy)]
  enum Treelet {
    Leaf(u32),
    Node {
      left: usize,
      right: usize,
      axis: u32,
    },
  }

  let mut treelet_nodes = vec![Treelet::Leaf(binary_idx)];

  // Expand the treelet using Breadth-First Search until we hit N leaves or run out of internal nodes
  loop {
    let leaf_count = treelet_nodes.iter().filter(|n| matches!(n, Treelet::Leaf(_))).count();
    if leaf_count >= N {
      break;
    }

    let mut expanded = false;
    for i in 0..treelet_nodes.len() {
      if let Treelet::Leaf(b_idx) = treelet_nodes[i] {
        if b_idx != u32::MAX && !binary_bvh.is_leaf(b_idx) {
          let (l_opt, r_opt) = binary_bvh.children(b_idx);
          if l_opt.is_some() || r_opt.is_some() {
            let axis = binary_bvh.split_axis(b_idx);

            let l_idx = treelet_nodes.len();
            treelet_nodes.push(Treelet::Leaf(l_opt.unwrap_or(u32::MAX)));

            let r_idx = treelet_nodes.len();
            treelet_nodes.push(Treelet::Leaf(r_opt.unwrap_or(u32::MAX)));

            treelet_nodes[i] = Treelet::Node {
              left: l_idx,
              right: r_idx,
              axis,
            };
            expanded = true;
            break; // Expand one per iteration to keep breadth-first expansion
          }
        }
      }
    }
    if !expanded {
      break;
    }
  }

  // Collect leaves in base order
  let mut frontier = Vec::new();
  fn collect_leaves(idx: usize, tree: &[Treelet], frontier: &mut Vec<u32>) {
    match tree[idx] {
      Treelet::Leaf(b_idx) => {
        if b_idx != u32::MAX {
          frontier.push(b_idx);
        }
      }
      Treelet::Node { left, right, .. } => {
        collect_leaves(left, tree, frontier);
        collect_leaves(right, tree, frontier);
      }
    }
  }
  collect_leaves(0, &treelet_nodes, &mut frontier);

  let mut node = TlasMultiNode::<N>::default();
  let mut valid_mask_0 = 0u32;
  let mut valid_mask_1 = 0u32;

  for (i, &child_idx) in frontier.iter().enumerate() {
    if i < 32 {
      valid_mask_0 |= 1 << i;
    } else {
      valid_mask_1 |= 1 << (i - 32);
    }

    let bound: [f32; 6] = binary_bvh.bound(child_idx).into();
    node.min_x[i] = bound[0];
    node.min_y[i] = bound[1];
    node.min_z[i] = bound[2];
    node.max_x[i] = bound[3];
    node.max_y[i] = bound[4];
    node.max_z[i] = bound[5];

    node.masses[i] = binary_bvh.mass(child_idx);
    let com = binary_bvh.center_of_mass(child_idx);
    node.com_x[i] = com[0];
    node.com_y[i] = com[1];
    node.com_z[i] = com[2];
    node.particle_start[i] = 0; // TODO: Fetch from binary_bvh if supported
    node.particle_count[i] = u32::MAX;

    if binary_bvh.is_leaf(child_idx) {
      if let Some(meta) = binary_bvh.leaf_meta(child_idx) {
        node.metadata[i] = meta;
      } else {
        node.metadata[i] = (1 << 31) | child_idx;
      }
    } else if depth > 64 {
      aethervk_oshal_rlib::log!(
        "Warning: max depth exceeded in collapse_binary_to_multi_recursive, depth={}, child_idx={}",
        depth,
        child_idx
      );
      if let Some(meta) = binary_bvh.leaf_meta(child_idx) {
        node.metadata[i] = meta;
      } else {
        node.metadata[i] = (1 << 31) | child_idx;
      }
    } else {
      let child_multi_idx =
        collapse_binary_to_multi_recursive::<N, T>(child_idx, binary_bvh, multi_nodes, depth + 1);
      node.child_indices[i] = child_multi_idx;
      node.metadata[i] = 0;
    }
  }
  node.valid_mask = [valid_mask_0, valid_mask_1];

  // Generate permutations
  for sign_mask in 0..8 {
    let mut perm = Vec::new();
    fn traverse_perm(
      idx: usize,
      tree: &[Treelet],
      sign_mask: u8,
      frontier: &[u32],
      perm: &mut Vec<u8>,
    ) {
      match tree[idx] {
        Treelet::Leaf(b_idx) => {
          if b_idx != u32::MAX {
            if let Some(pos) = frontier.iter().position(|&x| x == b_idx) {
              perm.push(pos as u8);
            }
          }
        }
        Treelet::Node { left, right, axis } => {
          let is_negative = (sign_mask & (1 << axis)) != 0;
          if is_negative {
            traverse_perm(right, tree, sign_mask, frontier, perm);
            traverse_perm(left, tree, sign_mask, frontier, perm);
          } else {
            traverse_perm(left, tree, sign_mask, frontier, perm);
            traverse_perm(right, tree, sign_mask, frontier, perm);
          }
        }
      }
    }
    traverse_perm(0, &treelet_nodes, sign_mask, &frontier, &mut perm);
    for i in 0..perm.len() {
      node.permutations[sign_mask as usize][i] = perm[i] as u32;
    }
  }

  multi_nodes[multi_idx as usize] = node;
  multi_idx
}

// --- CONVERSION BUILDER ---

impl<B: Clone, P: Clone, const N: usize> MultiBvh<B, P, N> {
  /// Builds a multi-branch BVH from any binary BVH implementing `BinaryBvh`.
  pub fn build<T: BinaryBvh<Bound = B, Primitive = P>>(binary_bvh: &T) -> Self {
    assert!(N > 1 && N.is_power_of_two(), "N must be a power of two > 1");

    let mut nodes = Vec::new();
    let mut primitives = Vec::new();
    let mut root = None;

    if let Some(binary_root) = binary_bvh.root() {
      root = Some(Self::collapse_recursive(
        binary_root,
        binary_bvh,
        &mut nodes,
        &mut primitives,
        0,
      ));
    }

    Self {
      nodes,
      primitives,
      root,
    }
  }

  fn collapse_recursive<T: BinaryBvh<Bound = B, Primitive = P>>(
    binary_idx: u32,
    binary_bvh: &T,
    multi_nodes: &mut Vec<MultiBvhNode<B, N>>,
    multi_primitives: &mut Vec<P>,
    depth: u32,
  ) -> u32 {
    let multi_idx = multi_nodes.len() as u32;
    let parent_bound = binary_bvh.bound(binary_idx);

    // Push a placeholder node so downstream allocations maintain accurate indexing maps
    multi_nodes.push(MultiBvhNode {
      bounds: array::from_fn(|_| parent_bound.clone()),
      child_or_primitive_offsets: [u32::MAX; N],
      primitive_counts: [0; N],
      is_leaf: [false; N],
      valid_count: 0,
      masses: [0.0; N],
      centers_of_mass: [[0.0; 3]; N],
    });

    // Use Breadth-First-Search local clustering to gather up to `N` children
    let mut frontier = vec![binary_idx];

    while frontier.len() < N {
      let mut split_pos = None;
      for (i, &idx) in frontier.iter().enumerate() {
        if !binary_bvh.is_leaf(idx) {
          let (l, r) = binary_bvh.children(idx);
          if l.is_some() || r.is_some() {
            split_pos = Some((i, l, r));
            break;
          }
        }
      }

      if let Some((pos, left, right)) = split_pos {
        frontier.remove(pos);
        if let Some(l) = left {
          frontier.push(l);
        }
        if let Some(r) = right {
          frontier.push(r);
        }
      } else {
        break; // Met completely fully packed leaf boundaries early
      }
    }

    let valid_count = frontier.len() as u32;
    let mut bounds = array::from_fn(|_| parent_bound.clone());
    let mut child_or_primitive_offsets = [u32::MAX; N];
    let mut primitive_counts = [0; N];
    let mut is_leaf = [false; N];
    let mut masses = [0.0; N];
    let mut centers_of_mass = [[0.0; 3]; N];

    for (i, &child_idx) in frontier.iter().enumerate() {
      bounds[i] = binary_bvh.bound(child_idx);
      masses[i] = binary_bvh.mass(child_idx);
      centers_of_mass[i] = binary_bvh.center_of_mass(child_idx);

      if binary_bvh.is_leaf(child_idx) {
        is_leaf[i] = true;
        let offset = multi_primitives.len() as u32;
        let count = binary_bvh.extract_primitives(child_idx, multi_primitives);
        child_or_primitive_offsets[i] = offset;
        primitive_counts[i] = count;
      } else if depth > 64 {
        aethervk_oshal_rlib::log!("Warning: max depth exceeded in collapse_recursive");
        is_leaf[i] = true;
        let offset = multi_primitives.len() as u32;
        let count = binary_bvh.extract_primitives(child_idx, multi_primitives);
        child_or_primitive_offsets[i] = offset;
        primitive_counts[i] = count;
      } else {
        is_leaf[i] = false;
        primitive_counts[i] = 0;
        let child_multi_idx = Self::collapse_recursive(
          child_idx,
          binary_bvh,
          multi_nodes,
          multi_primitives,
          depth + 1,
        );
        child_or_primitive_offsets[i] = child_multi_idx;
      }
    }

    // Overwrite initial placeholder with correctly mapped properties
    multi_nodes[multi_idx as usize] = MultiBvhNode {
      bounds,
      child_or_primitive_offsets,
      primitive_counts,
      is_leaf,
      valid_count,
      masses,
      centers_of_mass,
    };

    multi_idx
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use alloc::vec;

  #[derive(Debug)]
  struct MockBinaryBvh {
    nodes: Vec<MockNode>,
  }

  #[derive(Debug)]
  struct MockNode {
    bound: f32,
    children: (Option<u32>, Option<u32>),
    primitives: Vec<u32>,
  }

  impl BinaryBvh for MockBinaryBvh {
    type Bound = f32;
    type Primitive = u32;

    fn root(&self) -> Option<u32> {
      if self.nodes.is_empty() { None } else { Some(0) }
    }

    fn bound(&self, node_idx: u32) -> Self::Bound {
      self.nodes[node_idx as usize].bound
    }

    fn is_leaf(&self, node_idx: u32) -> bool {
      !self.nodes[node_idx as usize].primitives.is_empty()
    }

    fn children(&self, node_idx: u32) -> (Option<u32>, Option<u32>) {
      self.nodes[node_idx as usize].children
    }

    fn extract_primitives(&self, node_idx: u32, out: &mut Vec<Self::Primitive>) -> u32 {
      let count = self.nodes[node_idx as usize].primitives.len() as u32;
      out.extend_from_slice(&self.nodes[node_idx as usize].primitives);
      count
    }
  }

  #[test]
  fn test_multi_bvh_build_quad() {
    // Build a simple binary tree:
    //      0 (b:10)
    //     / \
    //    1   2
    //   / \ / \
    //  3  4 5  6 (leaves)

    let mock = MockBinaryBvh {
      nodes: vec![
        MockNode {
          bound: 10.0,
          children: (Some(1), Some(2)),
          primitives: vec![],
        },
        MockNode {
          bound: 5.0,
          children: (Some(3), Some(4)),
          primitives: vec![],
        },
        MockNode {
          bound: 5.0,
          children: (Some(5), Some(6)),
          primitives: vec![],
        },
        MockNode {
          bound: 2.0,
          children: (None, None),
          primitives: vec![100],
        },
        MockNode {
          bound: 2.0,
          children: (None, None),
          primitives: vec![101],
        },
        MockNode {
          bound: 2.0,
          children: (None, None),
          primitives: vec![102],
        },
        MockNode {
          bound: 2.0,
          children: (None, None),
          primitives: vec![103],
        },
      ],
    };

    let multi = MultiBvh::<f32, u32, 4>::build(&mock);

    // The BFS should collapse [1, 2] and then [3, 4, 5, 6].
    // Since N=4, it should be able to fit all 4 leaves in the root's children.
    assert_eq!(multi.nodes.len(), 1);
    let root = &multi.nodes[0];
    assert_eq!(root.valid_count, 4);
    assert_eq!(root.is_leaf, [true, true, true, true]);
    assert_eq!(multi.primitives, vec![100, 101, 102, 103]);
  }

  #[test]
  fn test_multi_bvh_unbalanced() {
    // Unbalanced tree:
    //    0
    //   / \
    //  1   2
    //     / \
    //    3   4
    //       / \
    //      5   6

    let mock = MockBinaryBvh {
      nodes: vec![
        MockNode {
          bound: 10.0,
          children: (Some(1), Some(2)),
          primitives: vec![],
        }, // 0
        MockNode {
          bound: 1.0,
          children: (None, None),
          primitives: vec![1],
        }, // 1 (leaf)
        MockNode {
          bound: 9.0,
          children: (Some(3), Some(4)),
          primitives: vec![],
        }, // 2
        MockNode {
          bound: 1.0,
          children: (None, None),
          primitives: vec![3],
        }, // 3 (leaf)
        MockNode {
          bound: 8.0,
          children: (Some(5), Some(6)),
          primitives: vec![],
        }, // 4
        MockNode {
          bound: 1.0,
          children: (None, None),
          primitives: vec![5],
        }, // 5 (leaf)
        MockNode {
          bound: 1.0,
          children: (None, None),
          primitives: vec![6],
        }, // 6 (leaf)
      ],
    };

    let multi = MultiBvh::<f32, u32, 4>::build(&mock);

    // With N=4, it should eventually gather 4 leaves: 1, 3, 5, 6.
    assert_eq!(multi.nodes.len(), 1);
    assert_eq!(multi.nodes[0].valid_count, 4);
    assert_eq!(multi.nodes[0].is_leaf, [true, true, true, true]);
  }
}
