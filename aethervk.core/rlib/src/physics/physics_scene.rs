//! physics_scene module.

use crate::math::collision::linear_bvh::LinearBound;
use crate::scene::particles::{ParticleData, ParticleSystemComponent};
use crate::scene::{Component, EntityId, Scene, TransformComponent};
use aethervk_oshal_rlib::math::matrix::Matrix4;
use aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::{Vector, Vector3};

pub mod math;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
/// TODO: Document this item
pub struct GpuBvhNode {
  pub aabb_min: [f32; 3],
  pub left_child_or_prim: u32,
  pub aabb_max: [f32; 3],
  pub right_child_offset: u32,
  pub prim_count: u32,
  pub _pad0: u32,
  pub _pad1: u32,
  pub _pad2: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
/// TODO: Document this item
pub struct GpuReferenceFrame {
  pub center_pos: [f32; 3],
  pub scale: f32,
  pub center_vel: [f32; 3],
  pub soi_radius: f32,
  pub frame_type: u32,
  pub parent_frame_idx: u32,
  pub bvh_root_index: u32,
  pub entity_id_raw: u64,
  pub _pad0: u32,
  pub _pad1: u32,
}

#[derive(Clone, Debug)]
/// TODO: Document this item
pub struct PhysicsScene {
  pub gpu_frames: alloc::vec::Vec<GpuReferenceFrame>,
  pub gpu_bvh_nodes: alloc::vec::Vec<GpuBvhNode>,
  pub gpu_primitives: alloc::vec::Vec<u32>,
  pub gpu_entity_mappings: alloc::vec::Vec<EntityId>,
}

impl PhysicsScene {
  /// TODO: Document this item
  pub fn build_from_scene(scene: &Scene) -> Self {
    use crate::scene::{KinematicComponent, ReferenceFrameComponent, ReferenceFrameType};
    let mut frame_map: hashbrown::HashMap<EntityId, u32> = hashbrown::HashMap::new();
    let mut gpu_frames = alloc::vec::Vec::new();

    scene.query2::<TransformComponent, ReferenceFrameComponent, _>(|e, t, f| {
      let vel =
        scene.with_component(e, |k: &KinematicComponent| k.velocity).unwrap_or(Vec3f32::zero());

      let gpu_frame = GpuReferenceFrame {
        center_pos: [t.position.x(), t.position.y(), t.position.z()],
        scale: f.scale,
        center_vel: [vel.x(), vel.y(), vel.z()],
        soi_radius: f.soi_radius,
        frame_type: f.frame_type as u32,
        parent_frame_idx: u32::MAX,
        bvh_root_index: u32::MAX,
        entity_id_raw: slotmap::Key::data(&e).as_ffi(),
        ..Default::default()
      };

      frame_map.insert(e, gpu_frames.len() as u32);
      gpu_frames.push(gpu_frame);
    });

    for (entity, idx) in frame_map.iter() {
      let mut curr = *entity;
      while let Some(parent_id) = scene.get_parent(curr) {
        if let Some(&parent_idx) = frame_map.get(&parent_id) {
          gpu_frames[*idx as usize].parent_frame_idx = parent_idx;
          break;
        }
        curr = parent_id;
      }
    }

    if gpu_frames.is_empty() {
      gpu_frames.push(GpuReferenceFrame {
        center_pos: [0.0, 0.0, 0.0],
        scale: 1.0,
        // center_rot: [0.0, 0.0, 0.0, 1.0],
        center_vel: [0.0, 0.0, 0.0],
        soi_radius: f32::MAX,
        frame_type: crate::scene::ReferenceFrameType::Macro as u32,
        parent_frame_idx: u32::MAX,
        bvh_root_index: u32::MAX,
        entity_id_raw: 0,
        ..Default::default()
      });
    }

    let mut frame_instances: hashbrown::HashMap<
      u32,
      alloc::vec::Vec<(EntityId, LinearBound<f32>)>,
    > = hashbrown::HashMap::new();
    let default_parent = gpu_frames
      .iter()
      .position(|f| f.frame_type == crate::scene::ReferenceFrameType::Macro as u32)
      .unwrap_or(0) as u32;

    scene.query2::<TransformComponent, crate::scene::PhysicalMeshComponent, _>(
      |entity, transform, mesh| {
        if let Some(bvh) = &mesh.mesh.bvh {
          if !bvh.nodes.is_empty() {
            let root_bound = &bvh.nodes[0].bound;
            let mat = Mat4x4f32::translation(transform.position)
              * <Mat4x4f32 as Matrix4>::from_quat_custom_frame(transform.rotation)
              * Mat4x4f32::from_scale(transform.scale);

            let transformed_aabb = match root_bound {
              LinearBound::AABB(aabb) => aabb.transform_f32(&mat),
              LinearBound::OBB(obb) => obb.transform_f32(&mat).to_aabb::<Vec3f32>(),
            };

            let mut target_frame_idx = default_parent;
            let mut curr = entity;
            loop {
              if let Some(&idx) = frame_map.get(&curr) {
                target_frame_idx = idx;
                break;
              }
              if let Some(parent) = scene.get_parent(curr) {
                curr = parent;
              } else {
                break;
              }
            }

            frame_instances
              .entry(target_frame_idx)
              .or_default()
              .push((entity, LinearBound::AABB(transformed_aabb)));
          }
        }
      },
    );

    let mut gpu_bvh_nodes = alloc::vec::Vec::new();
    let mut gpu_primitives = alloc::vec::Vec::new();
    let mut gpu_entity_mappings = alloc::vec::Vec::new();

    for (frame_idx, mut instances) in frame_instances {
      if !instances.is_empty() {
        gpu_frames[frame_idx as usize].bvh_root_index = gpu_bvh_nodes.len() as u32;

        Self::build_recursive_flat(
          &mut instances,
          &mut gpu_bvh_nodes,
          &mut gpu_primitives,
          &mut gpu_entity_mappings,
        );
      }
    }

    Self {
      gpu_frames,
      gpu_bvh_nodes,
      gpu_primitives,
      gpu_entity_mappings,
    }
  }

  fn build_recursive_flat(
    instances: &mut [(EntityId, LinearBound<f32>)],
    nodes: &mut alloc::vec::Vec<GpuBvhNode>,
    primitives: &mut alloc::vec::Vec<u32>,
    entity_mappings: &mut alloc::vec::Vec<EntityId>,
  ) -> u32 {
    let node_idx = nodes.len() as u32;
    nodes.push(GpuBvhNode::default());

    let mut aggregate_aabb = match &instances[0].1 {
      LinearBound::AABB(aabb) => aabb.clone(),
      LinearBound::OBB(obb) => obb.to_aabb::<Vec3f32>(),
    };
    for (_, bound) in &instances[1..] {
      let b_aabb = match bound {
        LinearBound::AABB(aabb) => aabb.clone(),
        LinearBound::OBB(obb) => obb.to_aabb::<Vec3f32>(),
      };
      aggregate_aabb.encapsulate_aabb::<Vec3f32>(&b_aabb);
    }

    let min: Vec3f32 = aggregate_aabb.min();
    let max: Vec3f32 = aggregate_aabb.max();

    if instances.len() == 1 {
      let prim_idx = entity_mappings.len() as u32;
      entity_mappings.push(instances[0].0);
      primitives.push(prim_idx);

      nodes[node_idx as usize] = GpuBvhNode {
        aabb_min: [min.x(), min.y(), min.z()],
        left_child_or_prim: prim_idx,
        aabb_max: [max.x(), max.y(), max.z()],
        right_child_offset: u32::MAX,
        prim_count: 1,
        ..Default::default()
      };
    } else {
      let mut min_c = Vec3f32::from_components(f32::INFINITY, f32::INFINITY, f32::INFINITY);
      let mut max_c =
        Vec3f32::from_components(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
      for (_, bound) in instances.iter() {
        let c = match bound {
          LinearBound::AABB(aabb) => aabb.center(),
          LinearBound::OBB(obb) => obb.to_aabb::<Vec3f32>().center(),
        };
        min_c = min_c.min(c);
        max_c = max_c.max(c);
      }

      let extents = max_c - min_c;
      let axis = if extents.x() > extents.y() && extents.x() > extents.z() {
        0
      } else if extents.y() > extents.z() {
        1
      } else {
        2
      };

      instances.sort_by(|(_, a), (_, b)| {
        let ca: Vec3f32 = match a {
          LinearBound::AABB(aabb) => aabb.center(),
          LinearBound::OBB(obb) => obb.to_aabb::<Vec3f32>().center(),
        };
        let cb: Vec3f32 = match b {
          LinearBound::AABB(aabb) => aabb.center(),
          LinearBound::OBB(obb) => obb.to_aabb::<Vec3f32>().center(),
        };
        ca[axis].partial_cmp(&cb[axis]).unwrap_or(core::cmp::Ordering::Equal)
      });

      let mid = instances.len() / 2;
      let left =
        Self::build_recursive_flat(&mut instances[..mid], nodes, primitives, entity_mappings);
      let right =
        Self::build_recursive_flat(&mut instances[mid..], nodes, primitives, entity_mappings);

      nodes[node_idx as usize] = GpuBvhNode {
        aabb_min: [min.x(), min.y(), min.z()],
        left_child_or_prim: left,
        aabb_max: [max.x(), max.y(), max.z()],
        right_child_offset: right,
        prim_count: 0,
        ..Default::default()
      };
    }

    node_idx
  }
}
