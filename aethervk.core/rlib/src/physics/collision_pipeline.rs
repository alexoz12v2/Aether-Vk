use aethervk_oshal_rlib::math::vector::Vector4;
use aethervk_oshal_rlib::math::matrix::Matrix4;
use crate::gpu::{ColliderId, CollisionPair};
use crate::math::collision::cta::{CtaBody, compute_toi};
use crate::math::collision::gjk::Support;
use crate::physics::cpu_kernels::CpuMotionBvh;
use crate::physics::lca::resolve_lca_transform;
use aethervk_oshal_rlib::math::matrix::MatrixVectorMul;
use aethervk_oshal_rlib::math::matrix::SquareMatrix;
use aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::Vector;
use alloc::vec::Vec;

struct SphereShape {
  center: Vec3f32,
  radius: f32,
  velocity: Vec3f32,
}

impl Support for SphereShape {
  fn support(&self, dir: Vec3f32) -> Vec3f32 {
    let dir_normalized = if dir.length_squared() > 1e-6 {
      dir.normalize()
    } else {
      Vec3f32::from_array([1.0, 0.0, 0.0])
    };
    self.center + dir_normalized * self.radius
  }
}

impl CtaBody for SphereShape {
  fn linear_velocity(&self) -> Vec3f32 {
    self.velocity
  }
  fn angular_velocity(&self) -> Vec3f32 {
    Vec3f32::zero()
  }
  fn max_radius(&self) -> f32 {
    self.radius
  }
}

pub fn detect_collisions_cpu(bvh: &CpuMotionBvh) -> Vec<CollisionPair> {
  let mut pairs = Vec::new();
  let dynamics = &bvh.rigid_bodies_copy;
  let kinematics = &bvh.kinematics_copy;

  // Build LCA frames map
  let mut frames_map = hashbrown::HashMap::new();

  // FIX: Explicitly insert the root identity node so LCA traversal chains can converge securely
  frames_map.insert(0, (0, Mat4x4f32::identity()));

  for kin in kinematics {
    let mat: Mat4x4f32 = kin.transform.to_mat4();
    frames_map.insert(kin.own_frame_id, (kin.parent_frame_id, mat));
  }

  let get_global_pos = |frame_id: u32, local_pos: Vec3f32| -> Vec3f32 {
    let mut pos = local_pos;
    let mut curr = frame_id;
    while let Some(&(parent, ref transform)) = frames_map.get(&curr) {
      if curr == 0 {
        break; // Crucial check to prevent traversing the sentinel Identity root infinitely
      }
      let pt = transform.mul_vector(pos.to_point());
      pos = Vec3f32::from_array([pt.x(), pt.y(), pt.z()]);
      curr = parent;
    }
    pos
  };

  for (i, p1) in dynamics.iter().enumerate() {
    let global_pos = get_global_pos(p1.parent_frame_id, aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(p1.position));
    let max_travel = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(p1.linear_velocity).length();
    let r = 0.1 + max_travel;
    let query_aabb = crate::physics::motion_bvh::Aabb::new(
      global_pos - Vec3f32::from_array([r, r, r]),
      global_pos + Vec3f32::from_array([r, r, r]),
    );

    let mut overlaps = Vec::new();
    bvh.bvh_tree.query_intersections(&query_aabb, &mut overlaps);

    for &data_idx in &overlaps {
      let is_kinematic = (data_idx & (1 << 31)) != 0;
      let j = (data_idx & !(1 << 31)) as usize;

      if !is_kinematic {
        // Dynamic vs Dynamic
        if i >= j {
          continue;
        } // avoid duplicates
        let p2 = &dynamics[j];

        let mut p1_pos = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(p1.position);
        let mut p2_pos = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(p2.position);

        if p1.parent_frame_id != p2.parent_frame_id {
          let p1_mat = aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::from_columns(aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(p1.rotation[0][0], p1.rotation[0][1], p1.rotation[0][2], 0.0), aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(p1.rotation[1][0], p1.rotation[1][1], p1.rotation[1][2], 0.0), aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(p1.rotation[2][0], p1.rotation[2][1], p1.rotation[2][2], 0.0), aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(p1.position[0], p1.position[1], p1.position[2], 1.0));
          let p2_mat = aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::from_columns(aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(p2.rotation[0][0], p2.rotation[0][1], p2.rotation[0][2], 0.0), aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(p2.rotation[1][0], p2.rotation[1][1], p2.rotation[1][2], 0.0), aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(p2.rotation[2][0], p2.rotation[2][1], p2.rotation[2][2], 0.0), aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(p2.position[0], p2.position[1], p2.position[2], 1.0));

          if let Some(m_b_lca) = resolve_lca_transform(
            p1.parent_frame_id,
            p1_mat,
            p2.parent_frame_id,
            p2_mat,
            &frames_map,
          ) {
            // FIX: Evaluate relative position mapped into Body 1's local coordinate space
            let p2_in_lca = m_b_lca.mul_vector(Vec3f32::zero().to_point());
            p2_pos = Vec3f32::from_array([p2_in_lca.x(), p2_in_lca.y(), p2_in_lca.z()]);
            p1_pos = Vec3f32::zero();
          }
        }

        let dist_sq = (p1_pos - p2_pos).length_squared();
        if dist_sq < 0.04 {
          // Assuming radius = 0.1
          pairs.push(CollisionPair {
            a: ColliderId {
              entity_id: slotmap::Key::data(&p1.entity_id).as_ffi() as u32,
              primitive_index: i as u32,
            },
            b: ColliderId {
              entity_id: slotmap::Key::data(&p2.entity_id).as_ffi() as u32,
              primitive_index: j as u32,
            },
            time_of_impact: 0.0,
            contact_normal: [0.0; 3],
            contact_point: [0.0; 3],
            penetration_depth: 0.0,
          });
        }
      } else {
        // Dynamic vs Kinematic (Particle vs Extended Body)
        let k2 = &kinematics[j];

        let mut s1 = SphereShape {
          center: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(p1.position),
          radius: 0.1,
          velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(p1.linear_velocity),
        };
        let mut s2 = SphereShape {
          center: k2.transform.position,
          radius: k2.scale,
          velocity: k2.velocity,
        };

        if p1.parent_frame_id != k2.parent_frame_id {
          let p1_mat = aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::from_columns(aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(p1.rotation[0][0], p1.rotation[0][1], p1.rotation[0][2], 0.0), aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(p1.rotation[1][0], p1.rotation[1][1], p1.rotation[1][2], 0.0), aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(p1.rotation[2][0], p1.rotation[2][1], p1.rotation[2][2], 0.0), aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(p1.position[0], p1.position[1], p1.position[2], 1.0));
          let k2_mat: Mat4x4f32 = k2.transform.to_mat4();

          if let Some(m_b_lca) = resolve_lca_transform(
            p1.parent_frame_id,
            p1_mat,
            k2.parent_frame_id,
            k2_mat,
            &frames_map,
          ) {
            let k2_in_lca = m_b_lca.mul_vector(Vec3f32::zero().to_point());
            s2.center = Vec3f32::from_array([k2_in_lca.x(), k2_in_lca.y(), k2_in_lca.z()]);
            s1.center = Vec3f32::zero();
          }
        }

        let dist_centers = (s1.center - s2.center).length();
        let sum_radii = s1.radius + s2.radius;

        if dist_centers <= sum_radii {
          pairs.push(CollisionPair {
            a: ColliderId {
              entity_id: slotmap::Key::data(&p1.entity_id).as_ffi() as u32,
              primitive_index: i as u32,
            },
            b: ColliderId {
              entity_id: slotmap::Key::data(&k2.entity_id).as_ffi() as u32,
              primitive_index: j as u32,
            },
            time_of_impact: 0.0,
            contact_normal: [0.0; 3],
            contact_point: [0.0; 3],
            penetration_depth: 0.0,
          });
          continue;
        }

        let max_travel = s1.velocity.length() + s2.velocity.length();
        if dist_centers > sum_radii + max_travel {
          continue;
        }

        if let Some(toi) = compute_toi(&s1, &s2, 1e-3, 10) {
          pairs.push(CollisionPair {
            a: ColliderId {
              entity_id: slotmap::Key::data(&p1.entity_id).as_ffi() as u32,
              primitive_index: i as u32,
            },
            b: ColliderId {
              entity_id: slotmap::Key::data(&k2.entity_id).as_ffi() as u32,
              primitive_index: j as u32,
            },
            time_of_impact: toi,
            contact_normal: [0.0; 3],
            contact_point: [0.0; 3],
            penetration_depth: 0.0,
          });
        }
      }
    }
  }

  for (i, k1) in kinematics.iter().enumerate() {
    let global_pos = get_global_pos(k1.parent_frame_id, k1.transform.position);
    let max_travel = k1.velocity.length();
    let r = k1.scale + max_travel;
    let query_aabb = crate::physics::motion_bvh::Aabb::new(
      global_pos - Vec3f32::from_array([r, r, r]),
      global_pos + Vec3f32::from_array([r, r, r]),
    );

    let mut overlaps = Vec::new();
    bvh.bvh_tree.query_intersections(&query_aabb, &mut overlaps);

    for &data_idx in &overlaps {
      let is_kinematic = (data_idx & (1 << 31)) != 0;
      if !is_kinematic {
        // Dynamic vs Kinematic: Already handled above in Dynamic loop.
        continue;
      }

      let j = (data_idx & !(1 << 31)) as usize;
      if i >= j {
        continue;
      } // avoid duplicates

      let k2 = &kinematics[j];

      let mut s1 = SphereShape {
        center: k1.transform.position,
        radius: k1.scale,
        velocity: k1.velocity,
      };
      let mut s2 = SphereShape {
        center: k2.transform.position,
        radius: k2.scale,
        velocity: k2.velocity,
      };

      if k1.parent_frame_id != k2.parent_frame_id {
        let k1_mat: Mat4x4f32 = k1.transform.to_mat4();
        let k2_mat: Mat4x4f32 = k2.transform.to_mat4();

        if let Some(m_b_lca) = resolve_lca_transform(
          k1.parent_frame_id,
          k1_mat,
          k2.parent_frame_id,
          k2_mat,
          &frames_map,
        ) {
          // FIX: Evaluate relative position mapped into Body 1's local coordinate space
          let k2_in_lca = m_b_lca.mul_vector(Vec3f32::zero().to_point());
          s2.center = Vec3f32::from_array([k2_in_lca.x(), k2_in_lca.y(), k2_in_lca.z()]);
          s1.center = Vec3f32::zero();
        }
      }

      let dist_centers = (s1.center - s2.center).length();
      let sum_radii = s1.radius + s2.radius;

      if dist_centers <= sum_radii {
        pairs.push(CollisionPair {
          a: ColliderId {
            entity_id: slotmap::Key::data(&k1.entity_id).as_ffi() as u32,
            primitive_index: i as u32,
          },
          b: ColliderId {
            entity_id: slotmap::Key::data(&k2.entity_id).as_ffi() as u32,
            primitive_index: j as u32,
          },
          time_of_impact: 0.0,
          contact_normal: [0.0; 3],
          contact_point: [0.0; 3],
          penetration_depth: 0.0,
        });
        continue;
      }

      let max_travel = s1.velocity.length() + s2.velocity.length();
      if dist_centers > sum_radii + max_travel {
        continue;
      }

      if let Some(toi) = compute_toi(&s1, &s2, 1e-3, 10) {
        pairs.push(CollisionPair {
          a: ColliderId {
            entity_id: slotmap::Key::data(&k1.entity_id).as_ffi() as u32,
            primitive_index: i as u32,
          },
          b: ColliderId {
            entity_id: slotmap::Key::data(&k2.entity_id).as_ffi() as u32,
            primitive_index: j as u32,
          },
          time_of_impact: toi,
          contact_normal: [0.0; 3],
          contact_point: [0.0; 3],
          penetration_depth: 0.0,
        });
      }
    }
  }

  pairs
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::gpu::KinematicBody;
  use crate::scene::TransformComponent;
  use aethervk_oshal_rlib::math::quaternion::Quaternion;
  use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
  use aethervk_oshal_rlib::math::vector::vec4::Quat;

  fn create_dummy_dynamic(id: u64, pos: [f32; 3], vel: [f32; 3], parent_frame: u32) -> crate::gpu::RigidBodyGpu {
    crate::gpu::RigidBodyGpu {
      position: pos,
      mass: 1.0,
      rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
      linear_velocity: vel,
      _pad0: 0.0,
      angular_velocity: [0.0; 3],
      _pad1: 0.0,
      inertia_tensor: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
      force: [0.0; 3],
      torque: [0.0; 3],
      entity_id: slotmap::KeyData::from_ffi(id).into(),
      parent_frame_id: parent_frame,
      shape_type: 0,
      shape_data: [1.0, 0.0, 0.0],
    }
  }

  fn create_dummy_kinematic(
    id: u64,
    pos: [f32; 3],
    vel: [f32; 3],
    own_frame: u32,
    parent_frame: u32,
    scale: f32,
  ) -> KinematicBody {
    KinematicBody {
      entity_id: slotmap::KeyData::from_ffi(id).into(),
      transform: TransformComponent {
        position: Vec3f32::from_array(pos),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
      velocity: Vec3f32::from_array(vel),
      parent_frame_id: parent_frame,
      mu: 1.0,
      own_frame_id: own_frame,
      frame_type: 0,
      scale,
      shape_type: 0,
      shape_data: [1.0, 0.0, 0.0],
    }
  }

  fn build_test_bvh(kinematics: Vec<KinematicBody>, dynamics: Vec<crate::gpu::RigidBodyGpu>) -> CpuMotionBvh {
    let mut frames_map = hashbrown::HashMap::new();

    // FIX: Insert Identity for Root
    frames_map.insert(0, (0, Mat4x4f32::identity()));

    for kin in &kinematics {
      let mat: Mat4x4f32 = kin.transform.to_mat4();
      frames_map.insert(kin.own_frame_id, (kin.parent_frame_id, mat));
    }

    let get_global_pos = |frame_id: u32, local_pos: Vec3f32| -> Vec3f32 {
      let mut pos = local_pos;
      let mut curr = frame_id;
      while let Some(&(parent, ref transform)) = frames_map.get(&curr) {
        if curr == 0 {
          break;
        } // Safe bail to prevent infinite loops mapping the root
        let pt = transform.mul_vector(pos.to_point());
        pos = Vec3f32::from_array([pt.x(), pt.y(), pt.z()]);
        curr = parent;
      }
      pos
    };

    let mut aabbs = alloc::vec::Vec::new();

    for (i, kin) in kinematics.iter().enumerate() {
      let global_pos = get_global_pos(kin.parent_frame_id, kin.transform.position);
      let max_travel = kin.velocity.length();
      let r = kin.scale + max_travel;
      let bounds = crate::physics::motion_bvh::Aabb::new(
        global_pos - Vec3f32::from_array([r, r, r]),
        global_pos + Vec3f32::from_array([r, r, r]),
      );
      aabbs.push((bounds, (1 << 31) | (i as u32)));
    }

    for (i, dyn_body) in dynamics.iter().enumerate() {
      let global_pos = get_global_pos(dyn_body.parent_frame_id, aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(dyn_body.position));
      let max_travel = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(dyn_body.linear_velocity).length();
      let r = 0.1 + max_travel;
      let bounds = crate::physics::motion_bvh::Aabb::new(
        global_pos - Vec3f32::from_array([r, r, r]),
        global_pos + Vec3f32::from_array([r, r, r]),
      );
      aabbs.push((bounds, i as u32));
    }
    let bvh_tree = crate::physics::motion_bvh::MotionBvhTree::build(&aabbs);

    CpuMotionBvh {
      kinematics_copy: kinematics,
      rigid_bodies_copy: dynamics,
      particles_copy: alloc::vec::Vec::new(),
      bvh_tree,
    }
  }

  // 1. Particle - Particle (Self Collision)
  #[test]
  fn test_particle_particle_overlap() {
    let p1 = create_dummy_dynamic(1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0);
    let p2 = create_dummy_dynamic(2, [0.1, 0.0, 0.0], [0.0, 0.0, 0.0], 0);
    let bvh = build_test_bvh(vec![], vec![p1, p2]);
    let pairs = detect_collisions_cpu(&bvh);
    assert_eq!(pairs.len(), 1);
  }

  #[test]
  fn test_particle_particle_separate_clusters() {
    let p1 = create_dummy_dynamic(1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0);
    let p2 = create_dummy_dynamic(2, [0.1, 0.0, 0.0], [0.0, 0.0, 0.0], 0);
    let p3 = create_dummy_dynamic(3, [10.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0);
    let p4 = create_dummy_dynamic(4, [10.1, 0.0, 0.0], [0.0, 0.0, 0.0], 0);
    let bvh = build_test_bvh(vec![], vec![p1, p2, p3, p4]);
    let pairs = detect_collisions_cpu(&bvh);
    assert_eq!(pairs.len(), 2);
  }

  #[test]
  fn test_particle_particle_edge_case() {
    let p1 = create_dummy_dynamic(1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0);
    // boundary is dist_sq < 0.04 -> dist < 0.2
    let p2 = create_dummy_dynamic(2, [0.199, 0.0, 0.0], [0.0, 0.0, 0.0], 0);
    let p3 = create_dummy_dynamic(3, [-0.201, 0.0, 0.0], [0.0, 0.0, 0.0], 0);
    let bvh = build_test_bvh(vec![], vec![p1, p2, p3]);
    let pairs = detect_collisions_cpu(&bvh);
    assert_eq!(pairs.len(), 1); // Only p1 and p2
  }

  // 2. Particle - Particle (Cross-System)
  #[test]
  fn test_particle_cross_system_match_lca() {
    let k1 = create_dummy_kinematic(10, [5.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1, 0, 0.001); // small scale
    let k2 = create_dummy_kinematic(20, [10.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2, 0, 0.001); // far from k1
    let p1 = create_dummy_dynamic(1, [2.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1); // global: [7,0,0]
    let p2 = create_dummy_dynamic(2, [-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2); // global: [7,0,0]
    let bvh = build_test_bvh(vec![k1, k2], vec![p1, p2]);
    let pairs = detect_collisions_cpu(&bvh);
    assert_eq!(pairs.len(), 1);
  }

  #[test]
  fn test_particle_cross_system_scaled() {
    let mut k1 = create_dummy_kinematic(10, [10.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1, 0, 0.1);
    // We manually attach the scale here to test cross-system matrix mapping behavior
    k1.transform.scale = Vec3f32::from_array([2.0, 2.0, 2.0]);

    // local pos [1,0,0] -> global [12,0,0]
    let p1 = create_dummy_dynamic(1, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1);
    let p2 = create_dummy_dynamic(2, [12.05, 0.0, 0.0], [0.0, 0.0, 0.0], 0);
    let bvh = build_test_bvh(vec![k1], vec![p1, p2]);
    let pairs = detect_collisions_cpu(&bvh);
    assert_eq!(pairs.len(), 1);
  }

  #[test]
  fn test_particle_cross_system_pruning() {
    let k1 = create_dummy_kinematic(10, [100.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1, 0, 1.0);
    // Offset this slightly so P1 does not sit directly inside K1 to prevent false overlap flagging
    let p1 = create_dummy_dynamic(1, [2.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1); // global 102
    let p2 = create_dummy_dynamic(2, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0); // global 0
    let bvh = build_test_bvh(vec![k1], vec![p1, p2]);
    let pairs = detect_collisions_cpu(&bvh);
    assert_eq!(pairs.len(), 0);
  }

  // 3. Particle - Extended Body
  #[test]
  fn test_particle_extended_direct_hit() {
    let k1 = create_dummy_kinematic(10, [5.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1, 0, 1.0); // radius 1.0
    let p1 = create_dummy_dynamic(1, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], 0); // moving right, will hit around t=0.39
    let bvh = build_test_bvh(vec![k1], vec![p1]);
    let pairs = detect_collisions_cpu(&bvh);
    assert_eq!(pairs.len(), 1);
    assert!(pairs[0].time_of_impact > 0.0 && pairs[0].time_of_impact <= 1.0);
  }

  #[test]
  fn test_particle_extended_miss() {
    let k1 = create_dummy_kinematic(10, [5.0, 5.0, 0.0], [0.0, 0.0, 0.0], 1, 0, 1.0);
    let p1 = create_dummy_dynamic(1, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], 0);
    let bvh = build_test_bvh(vec![k1], vec![p1]);
    let pairs = detect_collisions_cpu(&bvh);
    assert_eq!(pairs.len(), 0);
  }

  #[test]
  fn test_particle_extended_parallel() {
    let k1 = create_dummy_kinematic(10, [0.0, 5.0, 0.0], [10.0, 0.0, 0.0], 1, 0, 1.0);
    let p1 = create_dummy_dynamic(1, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], 0);
    let bvh = build_test_bvh(vec![k1], vec![p1]);
    let pairs = detect_collisions_cpu(&bvh);
    assert_eq!(pairs.len(), 0);
  }

  // 4. Extended Body - Extended Body
  #[test]
  fn test_extended_extended_head_on() {
    let k1 = create_dummy_kinematic(10, [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], 1, 0, 1.0);
    let k2 = create_dummy_kinematic(20, [10.0, 0.0, 0.0], [-5.0, 0.0, 0.0], 2, 0, 1.0);
    let bvh = build_test_bvh(vec![k1, k2], vec![]);
    let pairs = detect_collisions_cpu(&bvh);
    assert_eq!(pairs.len(), 1);
    // Dist = 10, radii = 2. Rel vel = 10. They meet at (10-2)/10 = 0.8
    assert!((pairs[0].time_of_impact - 0.8).abs() < 0.05);
  }

  #[test]
  fn test_extended_extended_narrow_phase_overlap() {
    let k1 = create_dummy_kinematic(10, [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], 1, 0, 1.0);
    let k2 = create_dummy_kinematic(20, [0.5, 0.0, 0.0], [-5.0, 0.0, 0.0], 2, 0, 1.0);
    let bvh = build_test_bvh(vec![k1, k2], vec![]);
    let pairs = detect_collisions_cpu(&bvh);
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].time_of_impact, 0.0); // Pre-CCD overlap fires
  }

  #[test]
  fn test_extended_extended_motion_bounds_early_out() {
    let k1 = create_dummy_kinematic(10, [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], 1, 0, 1.0);
    let k2 = create_dummy_kinematic(20, [10.0, 0.0, 0.0], [-0.1, 0.0, 0.0], 2, 0, 1.0);
    // Radii sum = 2. Max travel sum = 0.2. Total reach = 2.2. Dist = 10.
    // Bounds test will fail immediately.
    let bvh = build_test_bvh(vec![k1, k2], vec![]);
    let pairs = detect_collisions_cpu(&bvh);
    assert_eq!(pairs.len(), 0);
  }
}
