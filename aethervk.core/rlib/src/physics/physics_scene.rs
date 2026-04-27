use crate::scene::{Scene, EntityId};
use crate::math::collision::linear_bvh::{LinearBVH, LinearBVHHeader, LinearBVHNode, LinearBound};
use aethervk_oshal_rlib::math::{
  matrix::{Matrix4, mat4::Mat4x4f32},
  vector::{vec3::Vec3f32, Vector, Vector3},
};

pub mod math;

/// PhysicsScene is a "throwaway projection" of the scene hierarchy
/// (rebuilt every update, not fixed update).
/// It aggregates the components from `Scene` and builds a world-wide BVH.
#[derive(Clone, Debug)]
pub struct PhysicsScene {
  // TODO variable precision
  /// World-wide BVH, composed of leaf nodes that are roots of each instance's BVH.
  pub world_bvh: LinearBVH<f32>,
  /// Mapping from leaf index in the world BVH back to the entity it represents.
  pub entity_mappings: alloc::vec::Vec<EntityId>,
}

impl PhysicsScene {
  /// Builds the throwaway physics projection from the given ECS `Scene`.
  pub fn build_from_scene(scene: &Scene) -> Self {
    let mut instances = alloc::vec::Vec::new();

    scene.query2::<crate::scene::TransformComponent, crate::scene::PhysicalMeshComponent, _>(
      |entity, transform, mesh| {
        if let Some(bvh) = &mesh.mesh.bvh {
          if !bvh.nodes.is_empty() {
            let root_bound = &bvh.nodes[0].bound;

            let mat = Mat4x4f32::translation(transform.position)
              * Mat4x4f32::from_quat_custom_frame(transform.rotation)
              * Mat4x4f32::from_scale(transform.scale);

            let transformed_bound = match root_bound {
              LinearBound::AABB(aabb) => LinearBound::AABB(aabb.transform_f32(&mat)),
              LinearBound::OBB(obb) => LinearBound::OBB(obb.transform_f32(&mat)),
            };
            instances.push((entity, transformed_bound));
          }
        }
      },
    );

    let mut entity_mappings = alloc::vec::Vec::new();
    let mut nodes = alloc::vec::Vec::new();
    let mut primitives = alloc::vec::Vec::new();

    if !instances.is_empty() {
      Self::build_recursive(
        &mut instances,
        &mut nodes,
        &mut primitives,
        &mut entity_mappings,
      );
    }

    let world_bvh = LinearBVH {
      header: LinearBVHHeader {
        preciseness: 0,
        node_count: nodes.len() as u32,
        primitive_count: primitives.len() as u32,
      },
      nodes,
      primitives,
    };

    Self {
      world_bvh,
      entity_mappings,
    }
  }

  fn build_recursive(
    instances: &mut [(EntityId, LinearBound<f32>)],
    nodes: &mut alloc::vec::Vec<LinearBVHNode<f32>>,
    primitives: &mut alloc::vec::Vec<usize>,
    entity_mappings: &mut alloc::vec::Vec<EntityId>,
  ) -> u32 {
    let node_idx = nodes.len() as u32;

    // Compute aggregate bound
    let mut aggregate_aabb = match &instances[0].1 {
      LinearBound::AABB(aabb) => aabb.clone(),
      LinearBound::OBB(obb) => obb.to_aabb::<Vec3f32>(),
    };
    for (_, bound) in &instances[1..] {
      match bound {
        LinearBound::AABB(aabb) => aggregate_aabb.encapsulate_aabb::<Vec3f32>(aabb),
        LinearBound::OBB(obb) => aggregate_aabb.encapsulate_aabb::<Vec3f32>(&obb.to_aabb::<Vec3f32>()),
      }
    }


    // Placeholder
    nodes.push(LinearBVHNode {
      bound: LinearBound::AABB(aggregate_aabb.clone()),
      left_child_or_primitive_offset: 0,
      right_child_offset: u32::MAX,
      primitive_count: 0,
    });

    if instances.len() == 1 {
      // Leaf
      let prim_idx = entity_mappings.len() as u32;
      entity_mappings.push(instances[0].0);
      primitives.push(prim_idx as usize);

      nodes[node_idx as usize].bound = instances[0].1.clone();
      nodes[node_idx as usize].left_child_or_primitive_offset = prim_idx;
      nodes[node_idx as usize].primitive_count = 1;
    } else {
      // Split
      // Find longest axis of centroids
      let mut min_c = Vec3f32::from_components(f32::INFINITY, f32::INFINITY, f32::INFINITY);
      let mut max_c =
        Vec3f32::from_components(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
      for (_, bound) in instances.iter() {
        let c: Vec3f32 = match bound {
          LinearBound::AABB(aabb) => aabb.center(),
          LinearBound::OBB(obb) => obb.translation(),
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
          LinearBound::OBB(obb) => obb.translation(),
        };
        let cb: Vec3f32 = match b {
          LinearBound::AABB(aabb) => aabb.center(),
          LinearBound::OBB(obb) => obb.translation(),
        };
        ca[axis]
          .partial_cmp(&cb[axis])
          .unwrap_or(core::cmp::Ordering::Equal)
      });

      let mid = instances.len() / 2;
      let left = Self::build_recursive(&mut instances[..mid], nodes, primitives, entity_mappings);
      let right = Self::build_recursive(&mut instances[mid..], nodes, primitives, entity_mappings);

      nodes[node_idx as usize].left_child_or_primitive_offset = left;
      nodes[node_idx as usize].right_child_offset = right;
    }

    node_idx
  }
}
