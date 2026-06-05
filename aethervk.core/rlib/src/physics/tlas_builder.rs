//! Per-tick motion TLAS/BLAS hierarchy builder.
//!
//! Hierarchy:
//!   Scene Root TLAS (macro, motion, multi-branch N = subgroup_size)
//!     ├── [BVH_SHAPE_AABB]   Body BLAS roots   (mesh entities, LCA-local swept)
//!     ├── [BVH_SHAPE_SPHERE] Particle BLAS roots (particle systems)
//!     └── [BVH_SHAPE_OBB]    Micro-Frame sub-TLAS roots (one per ReferenceFrameComponent)
//!           ├── Body BLAS roots   (inside the LCA)
//!           └── Particle BLAS roots (inside the LCA)

use crate::{
  math::collision::{
    linear_bvh::LinearBVH,
    multi_bvh::{TlasMultiNode, convert_binary_to_multi_bvh},
  },
  physics::physics_scene::{
    BVH_SHAPE_OBB, BVH_SHAPE_SPHERE, BVH_SHAPE_SUB_TLAS, PhysicsScene, RootBoundsBvh,
  },
};
use aethervk_oshal_rlib::math::{
  matrix::{
    Matrix, Matrix3, Matrix4, MatrixVectorMul, SquareMatrix, mat3::Mat3f32, mat4::Mat4x4f32,
  },
  quaternion::Quaternion,
  vector::{
    Vector, Vector3, Vector4,
    vec3::Vec3f32,
    vec4::{Quat, Vec4f32},
  },
};
use alloc::vec::Vec;

pub const PARTICLE_BLAS_SENTINEL: u32 = u32::MAX;

// ── Flat multi-branch node buffer ──────────────────────────────────────────

struct FlatNodes<const N: usize> {
  nodes: Vec<TlasMultiNode<N>>,
}

impl<const N: usize> FlatNodes<N> {
  fn new() -> Self {
    Self { nodes: Vec::new() }
  }

  fn append(&mut self, src: &[TlasMultiNode<N>]) -> u32 {
    let base = self.nodes.len() as u32;
    for n in src {
      let mut node = n.clone();
      for i in 0..N {
        let meta = node.metadata[i];
        if meta != 0 && (meta & 0x8000_0000) == 0 {
          let idx = meta & 0x07FF_FFFF;
          node.metadata[i] = (meta & 0xFF80_0000) | ((idx + base) & 0x07FF_FFFF);
          node.child_indices[i] += base;
        }
      }
      self.nodes.push(node);
    }
    base
  }
}

pub fn build_scene_motion_tlas<const N: usize>(physical_scene: &mut PhysicsScene) -> (Vec<u8>, u32)
where
  TlasMultiNode<N>: bytemuck::Pod,
{
  let mut flat = FlatNodes::<N>::new();

  // 1. Build and append all mesh BLASes
  let mut mesh_root_indices = hashbrown::HashMap::new();
  for (idx, blas_opt) in physical_scene.mesh_blases.iter().enumerate() {
    if let Some(blas) = blas_opt {
      let multi_nodes = convert_binary_to_multi_bvh::<N, LinearBVH<f32>>(blas);
      let root_idx = flat.append(&multi_nodes);
      mesh_root_indices.insert(physical_scene.mesh_entity_map[idx], root_idx);
    }
  }

  // 2. Build and append all particle BLASes (CPU path)
  let mut particle_root_indices = hashbrown::HashMap::new();
  for (idx, blas_opt) in physical_scene.particle_blases.iter().enumerate() {
    if let Some(blas) = blas_opt {
      let multi_nodes = convert_binary_to_multi_bvh::<N, LinearBVH<f32>>(blas);
      let root_idx = flat.append(&multi_nodes);
      mark_particle_sentinels(&mut flat.nodes, root_idx, N);
      particle_root_indices.insert(physical_scene.particle_entity_map[idx], root_idx);
    }
  }

  // 3. Process micro TLASes
  let mut sub_tlas_root_indices = hashbrown::HashMap::new();
  for (frame_idx, rbvh) in &physical_scene.micro_tlases {
    let mut sub_multi = convert_binary_to_multi_bvh::<N, RootBoundsBvh>(rbvh);
    patch_tlas_leaves(
      &mut sub_multi,
      N,
      &mesh_root_indices,
      &particle_root_indices,
      &sub_tlas_root_indices,
    );
    let sub_root = flat.append(&sub_multi);
    sub_tlas_root_indices.insert(*frame_idx, sub_root);
    if let Some(gf) = physical_scene.gpu_frames.get_mut(*frame_idx as usize) {
      gf.bvh_root_index = sub_root;
    }
  }

  // 4. Process macro TLAS
  let mut macro_multi = convert_binary_to_multi_bvh::<N, RootBoundsBvh>(&physical_scene.macro_tlas);
  patch_tlas_leaves(
    &mut macro_multi,
    N,
    &mesh_root_indices,
    &particle_root_indices,
    &sub_tlas_root_indices,
  );
  let macro_root = flat.append(&macro_multi);

  if let Some(macro_frame_idx) = physical_scene.gpu_frames.iter().position(|f| f.frame_type == 0) {
    physical_scene.gpu_frames[macro_frame_idx].bvh_root_index = macro_root;
  }

  (bytemuck::cast_slice(&flat.nodes).to_vec(), macro_root)
}

fn patch_tlas_leaves<const N: usize>(
  nodes: &mut [TlasMultiNode<N>],
  n: usize,
  mesh_root_indices: &hashbrown::HashMap<u32, u32>,
  particle_root_indices: &hashbrown::HashMap<u32, u32>,
  sub_tlas_root_indices: &hashbrown::HashMap<u32, u32>,
) {
  for node in nodes.iter_mut() {
    for i in 0..n {
      let meta = node.metadata[i];
      if meta == 0 {
        continue;
      }
      let is_leaf = (meta & 0x8000_0000) != 0;
      if is_leaf {
        let dense_idx = meta & 0x07FF_FFFF;
        let shape = (meta >> 27) & 0x3;
        if shape == BVH_SHAPE_SUB_TLAS {
          if let Some(&sub_root) = sub_tlas_root_indices.get(&dense_idx) {
            node.child_indices[i] = sub_root;
          }
        } else if shape == BVH_SHAPE_SPHERE {
          if let Some(&sub_root) = particle_root_indices.get(&dense_idx) {
            node.child_indices[i] = sub_root;
          } else {
            node.child_indices[i] = PARTICLE_BLAS_SENTINEL; // Vulcan GPU LBVH path
          }
        } else {
          if let Some(&sub_root) = mesh_root_indices.get(&dense_idx) {
            node.child_indices[i] = sub_root;
          }
        }
      }
    }
  }
}

fn mark_particle_sentinels<const N: usize>(
  nodes: &mut Vec<TlasMultiNode<N>>,
  root_idx: u32,
  n: usize,
) {
  let mut stack = alloc::vec![root_idx];
  while let Some(idx) = stack.pop() {
    let node = &mut nodes[idx as usize];
    for i in 0..n {
      let meta = node.metadata[i];
      if meta == 0 {
        continue;
      }
      let is_leaf = (meta & 0x8000_0000) != 0;
      if is_leaf {
        node.child_indices[i] = PARTICLE_BLAS_SENTINEL;
      } else {
        stack.push(node.child_indices[i]);
      }
    }
  }
}

pub fn trace_particle_bvh_path(
  multi_nodes: &[crate::math::collision::multi_bvh::TlasMultiNode<32>],
  root_idx: u32,
  target_primitive_idx: u32,
) -> Option<alloc::vec::Vec<u32>> {
  let mut stack = alloc::vec![(root_idx, alloc::vec![root_idx])];

  let target_meta = 0x8000_0000 | target_primitive_idx;

  while let Some((node_idx, current_path)) = stack.pop() {
    if (node_idx as usize) >= multi_nodes.len() {
      continue;
    }
    let node = &multi_nodes[node_idx as usize];

    for i in 0..32 {
      let meta = node.metadata[i];
      if meta == 0 {
        continue;
      }

      let is_leaf = (meta & 0x8000_0000) != 0;
      if is_leaf {
        if meta == target_meta {
          return Some(current_path);
        }
      } else {
        let child_idx = node.child_indices[i];
        if child_idx != PARTICLE_BLAS_SENTINEL {
          let mut next_path = current_path.clone();
          next_path.push(child_idx);
          stack.push((child_idx, next_path));
        }
      }
    }
  }
  None
}

pub fn build_selection_tlas(
  scene: &crate::scene::Scene,
) -> alloc::vec::Vec<crate::math::collision::multi_bvh::TlasMultiNode<32>> {
  use crate::{
    math::collision::multi_bvh::convert_binary_to_multi_bvh, physics::physics_scene::RootBoundsBvh,
  };

  let mut leaves = alloc::vec::Vec::new();

  let mut mesh_entities = alloc::vec::Vec::new();

  scene.query2::<crate::scene::PhysicalMeshComponent, crate::scene::TransformComponent, _>(
    |entity, mesh, transform| {
      if let Some(ref bvh) = mesh.mesh.bvh {
        if let Some(bvh_root) = bvh.nodes.first() {
          if let crate::math::collision::linear_bvh::LinearBound::AABB(aabb) = &bvh_root.bound {
            let min = aabb.min::<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>();
            let max = aabb.max::<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>();
            mesh_entities.push((entity, *transform, min, max));
          }
        }
      }
    },
  );

  for (entity, transform, min, max) in mesh_entities {
    let global_transform = scene.global_transform(entity).unwrap_or(transform);
    let model_matrix =
      <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as Matrix4>::translation(
        global_transform.position,
      ) * <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as Matrix4>::from_quat_custom_frame(
        global_transform.rotation,
      ) * <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as Matrix4>::from_scale(
        global_transform.scale,
      );

    let mut bmin = <aethervk_oshal_rlib::math::vector::vec3::Vec3f32 as Vector3>::from_components(
      f32::MAX,
      f32::MAX,
      f32::MAX,
    );
    let mut bmax = <aethervk_oshal_rlib::math::vector::vec3::Vec3f32 as Vector3>::from_components(
      f32::MIN,
      f32::MIN,
      f32::MIN,
    );

    for i in 0..8 {
      let corner = <aethervk_oshal_rlib::math::vector::vec3::Vec3f32 as Vector3>::from_components(
        if i & 1 == 0 { min.x() } else { max.x() },
        if i & 2 == 0 { min.y() } else { max.y() },
        if i & 4 == 0 { min.z() } else { max.z() },
      );
      let corner_v4 =
        <aethervk_oshal_rlib::math::vector::vec4::Vec4f32 as Vector4>::from_components(
          corner.x(),
          corner.y(),
          corner.z(),
          1.0,
        );
      let transformed =
        <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as MatrixVectorMul>::mul_vector(
          model_matrix,
          corner_v4,
        );
      bmin = <aethervk_oshal_rlib::math::vector::vec3::Vec3f32 as Vector3>::from_components(
        bmin.x().min(transformed.x()),
        bmin.y().min(transformed.y()),
        bmin.z().min(transformed.z()),
      );
      bmax = <aethervk_oshal_rlib::math::vector::vec3::Vec3f32 as Vector3>::from_components(
        bmax.x().max(transformed.x()),
        bmax.y().max(transformed.y()),
        bmax.z().max(transformed.z()),
      );
    }

    use slotmap::Key;
    let entity_id = entity.data().as_ffi();
    let index = leaves.len() as u32;
    leaves.push((index, bmin, bmax, entity_id));
  }

  if leaves.is_empty() {
    return alloc::vec::Vec::new();
  }

  let binary_leaves: alloc::vec::Vec<(
    u32,
    aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    u32,
    u32,
  )> = leaves
    .iter()
    .map(|&(idx, mn, mx, _)| {
      (
        idx,
        mn,
        mx,
        crate::physics::physics_scene::BVH_SHAPE_AABB,
        crate::physics::physics_scene::BVH_FRAME_MACRO,
      )
    })
    .collect();

  let bvh = RootBoundsBvh::build(&binary_leaves);
  let mut multi_nodes = convert_binary_to_multi_bvh::<32, RootBoundsBvh>(&bvh);

  for node in multi_nodes.iter_mut() {
    for i in 0..32 {
      let meta = node.metadata[i];
      if meta != 0 && (meta & 0x8000_0000) != 0 {
        let binary_node_id = (meta & 0x7FFF_FFFF) as usize;
        let index = bvh.nodes[binary_node_id].leaf_child_idx as usize;
        let entity_id = leaves[index].3;

        node.child_indices[i] = (entity_id & 0xFFFF_FFFF) as u32;
        node.metadata[i] = ((entity_id >> 32) as u32) | 0x8000_0000;
      }
    }
  }

  multi_nodes
}
