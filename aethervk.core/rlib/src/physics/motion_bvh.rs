use aethervk_oshal_rlib::math::vector::{Vector, Vector3, vec3::Vec3f32};
use alloc::vec::Vec;

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
  pub min: Vec3f32,
  pub max: Vec3f32,
}

impl Aabb {
  pub fn new(min: Vec3f32, max: Vec3f32) -> Self {
    Self { min, max }
  }

  pub fn merge(&self, other: &Aabb) -> Self {
    Self {
      min: Vec3f32::from_array([
        self.min.x().min(other.min.x()),
        self.min.y().min(other.min.y()),
        self.min.z().min(other.min.z()),
      ]),
      max: Vec3f32::from_array([
        self.max.x().max(other.max.x()),
        self.max.y().max(other.max.y()),
        self.max.z().max(other.max.z()),
      ]),
    }
  }

  pub fn intersects(&self, other: &Aabb) -> bool {
    self.min.x() <= other.max.x()
      && self.max.x() >= other.min.x()
      && self.min.y() <= other.max.y()
      && self.max.y() >= other.min.y()
      && self.min.z() <= other.max.z()
      && self.max.z() >= other.min.z()
  }

  pub fn center(&self) -> Vec3f32 {
    (self.min + self.max) * 0.5
  }
}

#[derive(Clone, Copy, Debug)]
pub enum CpuBvhItem {
  Primitive(u32, f32, Vec3f32),
  SubTree(u32),
}

pub struct MotionBvhNode {
  pub bounds: Aabb,
  pub left_child: Option<u32>,
  pub right_child: Option<u32>,
  pub data_index: Option<u32>,
  pub mass: f32,
  pub center_of_mass: Vec3f32,
}

pub struct MotionBvhTree {
  pub nodes: Vec<MotionBvhNode>,
  pub root: Option<u32>,
}

impl MotionBvhTree {
  pub fn build(aabbs: &[(Aabb, u32, f32, Vec3f32)]) -> Self {
    let mut nodes = Vec::new();
    if aabbs.is_empty() {
      return Self { nodes, root: None };
    }

    let mut items: Vec<(Aabb, CpuBvhItem)> = aabbs
      .iter()
      .map(|&(aabb, idx, mass, com)| (aabb, CpuBvhItem::Primitive(idx, mass, com)))
      .collect();
    let root = Self::build_recursive(&mut items, &mut nodes);
    Self {
      nodes,
      root: Some(root),
    }
  }

  pub fn build_into(
    items: &mut [(Aabb, CpuBvhItem)],
    nodes: &mut Vec<MotionBvhNode>,
  ) -> Option<u32> {
    if items.is_empty() {
      return None;
    }
    Some(Self::build_recursive(items, nodes))
  }

  fn build_recursive(items: &mut [(Aabb, CpuBvhItem)], nodes: &mut Vec<MotionBvhNode>) -> u32 {
    if items.len() == 1 {
      let idx = nodes.len() as u32;
      match items[0].1 {
        CpuBvhItem::Primitive(data_idx, mass, com) => {
          nodes.push(MotionBvhNode {
            bounds: items[0].0,
            left_child: None,
            right_child: None,
            data_index: Some(data_idx),
            mass,
            center_of_mass: com,
          });
        }
        CpuBvhItem::SubTree(root_idx) => {
          let mass = nodes[root_idx as usize].mass;
          let center_of_mass = nodes[root_idx as usize].center_of_mass;
          nodes.push(MotionBvhNode {
            bounds: items[0].0,
            left_child: Some(root_idx),
            right_child: None,
            data_index: None,
            mass,
            center_of_mass,
          });
        }
      }
      return idx;
    }

    let mut total_bounds = items[0].0;
    for (aabb, _) in &items[1..] {
      total_bounds = total_bounds.merge(aabb);
    }

    let extents = total_bounds.max - total_bounds.min;
    let axis = if extents.x() > extents.y() && extents.x() > extents.z() {
      0
    } else if extents.y() > extents.z() {
      1
    } else {
      2
    };

    items.sort_by(|a, b| {
      let ca = a.0.center();
      let cb = b.0.center();
      let va = if axis == 0 {
        ca.x()
      } else if axis == 1 {
        ca.y()
      } else {
        ca.z()
      };
      let vb = if axis == 0 {
        cb.x()
      } else if axis == 1 {
        cb.y()
      } else {
        cb.z()
      };
      va.partial_cmp(&vb).unwrap_or(core::cmp::Ordering::Equal)
    });

    let mid = items.len() / 2;
    let (left_items, right_items) = items.split_at_mut(mid);

    let node_idx = nodes.len() as u32;
    nodes.push(MotionBvhNode {
      bounds: total_bounds,
      left_child: None,
      right_child: None,
      data_index: None,
      mass: 0.0,
      center_of_mass: Vec3f32::zero(),
    });

    let left_child = Self::build_recursive(left_items, nodes);
    let right_child = Self::build_recursive(right_items, nodes);

    let left_mass = nodes[left_child as usize].mass;
    let left_com = nodes[left_child as usize].center_of_mass;
    let right_mass = nodes[right_child as usize].mass;
    let right_com = nodes[right_child as usize].center_of_mass;

    let total_mass = left_mass + right_mass;
    let mut total_com = Vec3f32::zero();
    if total_mass > 0.0 {
      total_com = (left_com * left_mass + right_com * right_mass) / total_mass;
    }

    nodes[node_idx as usize].left_child = Some(left_child);
    nodes[node_idx as usize].right_child = Some(right_child);
    nodes[node_idx as usize].mass = total_mass;
    nodes[node_idx as usize].center_of_mass = total_com;

    node_idx
  }

  pub fn query_intersections(&self, query_aabb: &Aabb, out: &mut Vec<u32>) {
    if let Some(root) = self.root {
      self.query_recursive(root, query_aabb, out);
    }
  }

  fn query_recursive(&self, node_idx: u32, query_aabb: &Aabb, out: &mut Vec<u32>) {
    let node = &self.nodes[node_idx as usize];
    if !node.bounds.intersects(query_aabb) {
      return;
    }

    if let Some(data_idx) = node.data_index {
      out.push(data_idx);
    } else {
      if let Some(left) = node.left_child {
        self.query_recursive(left, query_aabb, out);
      }
      if let Some(right) = node.right_child {
        self.query_recursive(right, query_aabb, out);
      }
    }
  }
}

impl crate::math::collision::multi_bvh::BinaryBvh for MotionBvhTree {
  type Bound = Aabb;
  type Primitive = u32;

  fn root(&self) -> Option<u32> {
    self.root
  }

  fn bound(&self, node_idx: u32) -> Self::Bound {
    self.nodes[node_idx as usize].bounds
  }

  fn is_leaf(&self, node_idx: u32) -> bool {
    self.nodes[node_idx as usize].data_index.is_some()
  }

  fn children(&self, node_idx: u32) -> (Option<u32>, Option<u32>) {
    let n = &self.nodes[node_idx as usize];
    (n.left_child, n.right_child)
  }

  fn extract_primitives(&self, node_idx: u32, out: &mut Vec<Self::Primitive>) -> u32 {
    if let Some(idx) = self.nodes[node_idx as usize].data_index {
      out.push(idx);
      1
    } else {
      0
    }
  }
}
impl MotionBvhTree {
  pub fn generate_depth_sorted_indices(&self) -> alloc::vec::Vec<u32> {
    let mut depths = alloc::vec::Vec::new();
    depths.resize(self.nodes.len(), -1i32);
    
    // Compute depth for all nodes (post-order traversal implicitly done via recursive helper)
    let mut max_depth = -1i32;
    if let Some(root) = self.root {
      max_depth = self.compute_depth(root, &mut depths);
    }
    
    if max_depth < 0 {
      return alloc::vec::Vec::new(); // No internal nodes
    }
    
    let mut depth_groups = alloc::vec::Vec::new();
    depth_groups.resize((max_depth + 1) as usize, alloc::vec::Vec::new());
    
    for (idx, &d) in depths.iter().enumerate() {
      if d >= 0 { // Only internal nodes
        depth_groups[d as usize].push(idx as u32);
      }
    }
    
    let mut sorted_indices = alloc::vec::Vec::new();
    for group in depth_groups {
      sorted_indices.extend(group);
    }
    
    sorted_indices
  }
  
  fn compute_depth(&self, node_idx: u32, depths: &mut [i32]) -> i32 {
    let node = &self.nodes[node_idx as usize];
    if node.data_index.is_some() {
      return -1; // Leaf
    }
    
    let mut left_depth = -1;
    if let Some(left) = node.left_child {
      left_depth = self.compute_depth(left, depths);
    }
    
    let mut right_depth = -1;
    if let Some(right) = node.right_child {
      right_depth = self.compute_depth(right, depths);
    }
    
    let d = core::cmp::max(left_depth, right_depth) + 1;
    depths[node_idx as usize] = d;
    d
  }
}
