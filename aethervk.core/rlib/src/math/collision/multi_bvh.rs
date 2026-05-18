//! multi_bvh module

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

// --- BINARY BVH ABSTRACTION TRAIT ---

/// A unified trait to inspect any binary BVH natively.
pub trait BinaryBvh {
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

  /// Extracts primitives of the leaf at `node_idx` and appends them to `out`.
  /// Returns the number of primitives appended.
  fn extract_primitives(&self, node_idx: u32, out: &mut Vec<Self::Primitive>) -> u32;
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
      } else {
        is_leaf[i] = false;
        primitive_counts[i] = 0;
        let child_multi_idx =
          Self::collapse_recursive(child_idx, binary_bvh, multi_nodes, multi_primitives);
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

  struct MockBinaryBvh {
    nodes: Vec<MockNode>,
  }

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