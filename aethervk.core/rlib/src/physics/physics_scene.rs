use crate::scene::{Scene, EntityId};
use crate::math::collision::linear_bvh::LinearBVH;
use aethervk_oshal_rlib::math::{
  matrix::mat3::Mat3f32,
  vector::vec3::Vec3f32,
};

/// PhysicsScene is a "throwaway projection" of the scene hierarchy
/// (rebuilt every update, not fixed update).
/// It aggregates the components from `Scene` and builds a world-wide BVH.
pub struct PhysicsScene {
  /// World-wide BVH, composed of leaf nodes that are roots of each instance's BVH.
  pub world_bvh: LinearBVH<f32, Vec3f32, Mat3f32>,
  /// Mapping from leaf index in the world BVH back to the entity it represents.
  pub entity_mappings: alloc::vec::Vec<EntityId>,
}

impl PhysicsScene {
  /// Builds the throwaway physics projection from the given ECS `Scene`.
  pub fn build_from_scene(scene: &Scene) -> Self {
    // 1. Query the scene for all entities with a TransformComponent and a PhysicalMeshComponent.
    // 2. Extract their local BVHs (which should be precomputed in PhysicalMeshComponent).
    // 3. Create a top-level BVH where the leaves are the transformed AABBs/OBBs of each entity's local BVH root.
    // 4. Return the new PhysicsScene with the combined world_bvh and entity mappings.
    //
    // TODO: Implement the actual bottom-up aggregation of instances to form `world_bvh`
    // using Morton codes or a simple SAH builder over the instance bounds.

    let world_bvh = LinearBVH {
      header: crate::math::collision::linear_bvh::LinearBVHHeader {
        preciseness: 0,
        node_count: 0,
        primitive_count: 0,
      },
      nodes: alloc::vec::Vec::new(),
      primitives: alloc::vec::Vec::new(),
    };

    Self {
      world_bvh,
      entity_mappings: alloc::vec::Vec::new(),
    }
  }
}
