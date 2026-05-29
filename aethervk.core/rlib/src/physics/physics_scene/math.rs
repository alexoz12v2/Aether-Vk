//! math module.

use crate::{
  math::collision::{intersection, intersection::Ray, linear_bvh},
  physics::physics_scene::PhysicsScene,
  scene::{EntityId, PhysicalMeshComponent},
};
use aethervk_oshal_rlib::math::{
  matrix::{MatrixVectorMul, SquareMatrix, mat3::Mat3f32, mat4::Mat4x4f32},
  vector::{Vector, Vector3, Vector4, vec3::Vec3f32, vec4::Vec4f32},
};
use alloc::vec::Vec;

// TODO unit tests

/// TODO: Document this item
pub trait PhysicsSceneMathExt {
  /// Broadphase: Traverses the top-level World BVH using a stack to find all
  /// entities whose bounding volumes intersect the ray.
  // fn intersect_world_bvh_math(&self, ray: &Ray<Vec3f32>) -> Vec<EntityId>; // Removed as requested

  /// Narrowphase: Purely mathematical extraction of Ray/Mesh local BVH and Möller–Trumbore intersection.
  /// Traverses the local BVH and returns `Some((t, global_hit_point))` if a hit is found closer than `max_t`.
  fn intersect_mesh_bvh_math(
    &self,
    ro: Vec3f32,
    rd: Vec3f32,
    model_matrix: Mat4x4f32,
    mesh_comp: &PhysicalMeshComponent,
    max_t: f32,
  ) -> Option<(f32, Vec3f32, [f32; 2])>;
}

impl PhysicsSceneMathExt for PhysicsScene {
  fn intersect_mesh_bvh_math(
    &self,
    ro: Vec3f32,
    rd: Vec3f32,
    model_matrix: Mat4x4f32,
    mesh_comp: &PhysicalMeshComponent,
    max_t: f32,
  ) -> Option<(f32, Vec3f32, [f32; 2])> {
    let bvh = mesh_comp.mesh.bvh.as_ref()?; // Early exit if there is no BVH

    // 1. Transform Ray into Local Space
    let inv_model = model_matrix.inverse().unwrap_or(Mat4x4f32::identity());

    let local_ro = inv_model.mul_vector(Vec4f32::from_components(ro.x(), ro.y(), ro.z(), 1.0));
    let local_rd = inv_model.mul_vector(Vec4f32::from_components(rd.x(), rd.y(), rd.z(), 0.0));

    let local_ro = Vec3f32::from_components(local_ro.x(), local_ro.y(), local_ro.z());
    let local_rd_vec = Vec3f32::from_components(local_rd.x(), local_rd.y(), local_rd.z());

    if local_rd_vec.dot(local_rd_vec) < 1e-6 {
      aethervk_oshal_rlib::log!("DEBUG: intersect_mesh_bvh_math aborted, local_rd is 0");
      return None;
    }

    let local_rd = local_rd_vec.normalize();
    let local_ray = Ray {
      origin: local_ro,
      direction: local_rd,
      length: f32::MAX,
    };

    // 2. BVH Traversal & Möller–Trumbore Math
    let mut closest_t = max_t;
    let mut hit_point = None;
    let mut hit_uv = [0.0, 0.0];

    let mut stack = Vec::new();
    if !bvh.nodes.is_empty() {
      stack.push(0);
    }

    let mut tri_tests = 0;

    while let Some(node_idx) = stack.pop() {
      let local_node = &bvh.nodes[node_idx];

      let hit_local_node = match &local_node.bound {
        linear_bvh::LinearBound::AABB(aabb) => intersection::intersect_ray_aabb(&local_ray, &aabb),
        linear_bvh::LinearBound::OBB(obb) => {
          intersection::intersect_ray_obb::<_, _, Mat3f32>(&local_ray, &obb)
        }
      };

      if hit_local_node {
        if local_node.primitive_count > 0 {
          let prim_start = local_node.left_child_or_primitive_offset as usize;
          let prim_end = prim_start + local_node.primitive_count as usize;

          for j in prim_start..prim_end {
            tri_tests += 1;
            let tri_idx = bvh.primitives[j];
            let idx0 = mesh_comp.mesh.indices[tri_idx * 3] as usize;
            let idx1 = mesh_comp.mesh.indices[tri_idx * 3 + 1] as usize;
            let idx2 = mesh_comp.mesh.indices[tri_idx * 3 + 2] as usize;

            let v0 = mesh_comp.mesh.vertices[idx0].position;
            let v1 = mesh_comp.mesh.vertices[idx1].position;
            let v2 = mesh_comp.mesh.vertices[idx2].position;

            let v0 = Vec3f32::from_components(v0[0], v0[1], v0[2]);
            let v1 = Vec3f32::from_components(v1[0], v1[1], v1[2]);
            let v2 = Vec3f32::from_components(v2[0], v2[1], v2[2]);

            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let h = local_rd.cross(edge2);
            let a = edge1.dot(h);

            if a < 1e-6 {
              continue;
            }

            let f = 1.0 / a;
            let s = local_ro - v0;
            let u = f * s.dot(h);
            if u < 0.0 || u > 1.0 {
              continue;
            }

            let q = s.cross(edge1);
            let v = f * local_rd.dot(q);
            if v < 0.0 || u + v > 1.0 {
              continue;
            }

            let t = f * edge2.dot(q);

            // We track against max_t directly inside the math loop to continuously prune distant hits early
            if t > 1e-5 && t < closest_t {
              closest_t = t;
              let local_hit = local_ro + local_rd * t;

              let global_hit = model_matrix.mul_vector(Vec4f32::from_components(
                local_hit.x(),
                local_hit.y(),
                local_hit.z(),
                1.0,
              ));

              hit_point = Some(Vec3f32::from_components(
                global_hit.x(),
                global_hit.y(),
                global_hit.z(),
              ));

              let uv0 = mesh_comp.mesh.vertices[idx0].uv;
              let uv1 = mesh_comp.mesh.vertices[idx1].uv;
              let uv2 = mesh_comp.mesh.vertices[idx2].uv;
              let w = 1.0 - u - v;
              hit_uv = [
                w * uv0[0] + u * uv1[0] + v * uv2[0],
                w * uv0[1] + u * uv1[1] + v * uv2[1],
              ];
            }
          }
        } else {
          if local_node.right_child_offset != u32::MAX {
            stack.push(local_node.right_child_offset as usize);
          }
          if local_node.left_child_or_primitive_offset != u32::MAX {
            stack.push(local_node.left_child_or_primitive_offset as usize);
          }
        }
      }
    }

    hit_point.map(|point| (closest_t, point, hit_uv))
  }
}

/// TODO: Document this item
pub fn closest_intersection(
  intersections: impl AsRef<[((f32, Vec3f32, [f32; 2]), EntityId)]>,
) -> Option<(f32, Vec3f32, [f32; 2], EntityId)> {
  intersections
    .as_ref() // Converts the input into a slice: &[(...)]
    .iter() // Iterates over references to the items
    .filter(|item| item.0.0 > 0.0)
    .min_by(|a, b| a.0.0.partial_cmp(&b.0.0).unwrap_or(core::cmp::Ordering::Equal))
    .map(|&((t, p, uv), e_id)| (t, p, uv, e_id))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    math::collision::{
      bounds::AABB,
      linear_bvh::{LinearBVH, LinearBVHNode, LinearBound},
    },
    physics::physics_scene::{GpuBvhNode, GpuReferenceFrame, PhysicsScene},
    scene::PhysicalMeshComponent,
    simulation::comet::{Comet, Vertex},
  };
  use aethervk_oshal_rlib::math::{
    matrix::{SquareMatrix, mat4::Mat4x4f32},
    vector::vec3::Vec3f32,
  };
  use alloc::{sync::Arc, vec};
  use polyhedral_mass_properties::MassProperties;
  use slotmap::Key;

  fn create_dummy_physics_scene() -> PhysicsScene {
    let mut gpu_frames = vec![GpuReferenceFrame {
      center_pos: [0.0, 0.0, 0.0],
      scale: 1.0,
      center_vel: [0.0, 0.0, 0.0],
      soi_radius: 100.0,
      frame_type: 0,
      parent_frame_idx: u32::MAX,
      bvh_root_index: 0,
      _pad0: 0,
      entity_id_raw: 1,
      frame_bda: 0,
    }];

    let gpu_bvh_nodes = vec![GpuBvhNode {
      aabb_min: [-10.0, -10.0, -10.0],
      left_child_or_prim: 0,
      aabb_max: [10.0, 10.0, 10.0],
      right_child_offset: u32::MAX,
      prim_count: 1,
      _pad0: 0,
      _pad1: 0,
      _pad2: 0,
    }];

    PhysicsScene {
      gpu_frames,
      macro_tlas: crate::physics::physics_scene::RootBoundsBvh { nodes: vec![] },
      micro_tlases: hashbrown::HashMap::new(),
      mesh_blases: vec![],
      particle_blases: vec![],
      mesh_entity_map: vec![],
      particle_entity_map: vec![],
      dt_s: 0.016,
      recent_collisions: vec![],
    }
  }

  fn create_dummy_mesh_component() -> PhysicalMeshComponent {
    let vertices = vec![
      Vertex {
        position: [-1.0, -1.0, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        tangent: [1.0, 0.0, 0.0, 1.0],
      },
      Vertex {
        position: [1.0, -1.0, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [1.0, 0.0],
        tangent: [1.0, 0.0, 0.0, 1.0],
      },
      Vertex {
        position: [0.0, 1.0, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.5, 1.0],
        tangent: [1.0, 0.0, 0.0, 1.0],
      },
    ];
    let indices = vec![0, 1, 2];

    let bvh_nodes = vec![LinearBVHNode {
      center_of_mass: [0.0, 0.0, 0.0],
      mass: 0.0,
      bound: LinearBound::AABB(AABB::new(
        Vec3f32::from_components(-1.0, -1.0, -0.1),
        Vec3f32::from_components(1.0, 1.0, 0.1),
      )),
      left_child_or_primitive_offset: 0,
      right_child_offset: u32::MAX,
      primitive_count: 1,
    }];

    let bvh = LinearBVH {
      header: crate::math::collision::linear_bvh::LinearBVHHeader {
        preciseness: 0,
        node_count: bvh_nodes.len() as u32,
        primitive_count: 1,
      },
      nodes: bvh_nodes,
      primitives: vec![0],
    };

    let comet = Comet {
      id: 1,
      vertices,
      indices,
      albedo_map: None,
      normal_map: None,
      roughness_map: None,
      ao_map: None,
      mass_properties: unsafe { core::mem::zeroed() },
      bvh: Some(bvh),
      pa_basis_bf: None,
      bf_to_pa: None,
    };

    PhysicalMeshComponent {
      asset_path: "".into(),
      mesh: Arc::new(comet),
      emissive_intensity: 0.0,
      emissive_color: [0.0, 0.0, 0.0],
      use_new_path: false,
      paint_display_mode: 0,
      sphere_center: [0.0, 0.0, 0.0],
      sphere_radius: 1.0,
      grid_color: [0.0, 0.0, 0.0],
      grid_density: 1.0,
    }
  }

  #[test]
  fn perf_test_intersect_mesh_bvh_math() {
    let scene = create_dummy_physics_scene();
    let mesh_comp = create_dummy_mesh_component();
    let model_matrix = Mat4x4f32::identity();

    let start = std::time::Instant::now();
    let iters = 100_000;
    for _ in 0..iters {
      let _ = scene.intersect_mesh_bvh_math(
        Vec3f32::from_components(0.0, 0.0, 5.0),
        Vec3f32::from_components(0.0, 0.0, -1.0),
        model_matrix,
        &mesh_comp,
        100.0,
      );
    }
    let elapsed = start.elapsed();
    println!(
      "intersect_mesh_bvh_math took {:?} for {} iterations",
      elapsed, iters
    );
  }
}
