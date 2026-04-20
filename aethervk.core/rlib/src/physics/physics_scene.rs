use crate::scene::{Scene, EntityId};
use crate::math::collision::linear_bvh::{LinearBVH, LinearBVHHeader, LinearBVHNode, LinearBound};
use aethervk_oshal_rlib::math::{
  matrix::{Matrix4, mat4::Mat4x4f32, mat3::Mat3f32},
  vector::vec3::Vec3f32,
};

/// PhysicsScene is a "throwaway projection" of the scene hierarchy
/// (rebuilt every update, not fixed update).
/// It aggregates the components from `Scene` and builds a world-wide BVH.
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
    let mut entity_mappings = alloc::vec::Vec::new();
    let mut nodes = alloc::vec::Vec::new();
    let mut primitives = alloc::vec::Vec::new();

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

            let primitive_offset = entity_mappings.len() as u32;
            entity_mappings.push(entity);
            primitives.push(primitive_offset as usize);

            nodes.push(LinearBVHNode {
              bound: transformed_bound,
              left_child_or_primitive_offset: primitive_offset,
              right_child_offset: u32::MAX,
              primitive_count: 1,
            });
          }
        }
      },
    );

    // TODO: Implement actual bottom-up aggregation of instances to form `world_bvh`
    // using Morton codes or a simple SAH builder over the instance bounds.
    // For now, this is just a flat array of leaf nodes.

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
}
