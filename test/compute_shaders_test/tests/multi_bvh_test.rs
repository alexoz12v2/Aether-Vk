use aethervk_core_rlib::gpu::compute_push_constants::BvhNodeAABBGpu;
use aethervk_core_rlib::math::collision::multi_bvh::{BinaryBvh, MultiBvh};
use std::vec;

struct MockBinaryBvh {
  nodes: vec::Vec<MockNode>,
}

struct MockNode {
  bound: [f32; 6], // min_x, min_y, min_z, max_x, max_y, max_z
  children: (Option<u32>, Option<u32>),
  primitives: vec::Vec<u32>,
}

impl BinaryBvh for MockBinaryBvh {
  type Bound = [f32; 6];
  type Primitive = BvhNodeAABBGpu; // Just a placeholder for primitive type

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

  fn extract_primitives(&self, node_idx: u32, out: &mut vec::Vec<Self::Primitive>) -> u32 {
    let count = self.nodes[node_idx as usize].primitives.len() as u32;
    // In a real scenario we'd map primitives from the binary tree to GPU-ready primitives
    for _ in &self.nodes[node_idx as usize].primitives {
      out.push(BvhNodeAABBGpu {
        min_bounds: [0.0; 3],
        max_bounds: [1.0; 3],
        left_child_or_primitive_offset: 0,
        right_child_offset: 0,
        primitive_count: 1,
        parent_idx: 0,
        node_type: 0,
        mass: 1.0,
        center_of_mass: [0.5; 3],
        });    }
    count
  }
}

#[test]
fn test_multi_bvh_gpu_conversion() {
  let mock = MockBinaryBvh {
    nodes: vec![
      MockNode {
        bound: [-10.0, -10.0, -10.0, 10.0, 10.0, 10.0],
        children: (Some(1), Some(2)),
        primitives: vec![],
      }, // 0
      MockNode {
        bound: [-5.0, -5.0, -5.0, 0.0, 0.0, 0.0],
        children: (None, None),
        primitives: vec![1],
      }, // 1
      MockNode {
        bound: [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
        children: (None, None),
        primitives: vec![2],
      }, // 2
    ],
  };

  // Assuming we want a 4-way MultiBvh which maps nicely to GPU subgroups or specific tree layouts
  let multi_bvh = MultiBvh::<[f32; 6], BvhNodeAABBGpu, 4>::build(&mock);

  assert_eq!(multi_bvh.nodes.len(), 1);
  let root = &multi_bvh.nodes[0];

  assert_eq!(root.valid_count, 2); // 2 leaves collapsed into the root
  assert_eq!(root.is_leaf[0], true);
  assert_eq!(root.is_leaf[1], true);
  assert_eq!(multi_bvh.primitives.len(), 2);
}
