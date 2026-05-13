use crate::gpu::{ColliderId, CollisionPair};
use crate::math::collision::cta::{CtaBody, compute_toi};
use crate::math::collision::gjk::{Support, gjk_distance};
use crate::physics::cpu_kernels::CpuMotionBvh;
use crate::physics::lca::resolve_lca_transform;
use aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32;
use aethervk_oshal_rlib::math::matrix::MatrixVectorMul;
use aethervk_oshal_rlib::math::vector::{Vector, Vector4};
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
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
  let dynamics = &bvh.dynamics_copy;
  let kinematics = &bvh.kinematics_copy;

  // Build LCA frames map
  let mut frames_map = hashbrown::HashMap::new();
  for kin in kinematics {
    let mat: Mat4x4f32 = kin.transform.to_mat4();
    frames_map.insert(kin.own_frame_id, (kin.parent_frame_id, mat));
  }

  // 1 & 2: Particle-Particle (Internal & Cross-system)
  for i in 0..dynamics.len() {
    for j in (i + 1)..dynamics.len() {
      let p1 = &dynamics[i];
      let p2 = &dynamics[j];

      let mut p1_pos = p1.transform.position;
      let mut p2_pos = p2.transform.position;

      // Resolve LCA if in different frames
      if p1.parent_frame_id != p2.parent_frame_id {
        let p1_mat: Mat4x4f32 = p1.transform.to_mat4();
        let p2_mat: Mat4x4f32 = p2.transform.to_mat4();

        if let Some(m_b_lca) = resolve_lca_transform(
          p1.parent_frame_id,
          p1_mat,
          p2.parent_frame_id,
          p2_mat,
          &frames_map,
        ) {
          // Transform p2 into p1's local space (or their LCA)
          let p2_in_lca = m_b_lca.mul_vector(p2_pos.to_point());
          p2_pos = Vec3f32::from_array([p2_in_lca.x(), p2_in_lca.y(), p2_in_lca.z()]);
        }
      }

      let dist_sq = (p1_pos - p2_pos).length_squared();
      if dist_sq < 4.0 {
        // Assuming radius = 1.0
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
        });
      }
    }
  }

  // 3: Rigidbody-Rigidbody (Kinematic-Kinematic)
  // For simplicity, model kinematic bodies as spheres for the GJK/CTA demonstration
  for i in 0..kinematics.len() {
    for j in (i + 1)..kinematics.len() {
      let k1 = &kinematics[i];
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

      // LCA for rigidbodies
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
          let k2_in_lca = m_b_lca.mul_vector(k2.transform.position.to_point());
          s2.center = Vec3f32::from_array([k2_in_lca.x(), k2_in_lca.y(), k2_in_lca.z()]);
        }
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
        });
      }
    }
  }

  // 4: Rigidbody-Particle (Kinematic-Dynamic)
  for i in 0..kinematics.len() {
    for j in 0..dynamics.len() {
      let k1 = &kinematics[i];
      let p2 = &dynamics[j];

      let mut s1 = SphereShape {
        center: k1.transform.position,
        radius: k1.scale,
        velocity: k1.velocity,
      };
      let mut s2 = SphereShape {
        center: p2.transform.position,
        radius: 1.0,
        velocity: p2.velocity,
      };

      // LCA for mixed
      if k1.parent_frame_id != p2.parent_frame_id {
        let k1_mat: Mat4x4f32 = k1.transform.to_mat4();
        let p2_mat: Mat4x4f32 = p2.transform.to_mat4();

        if let Some(m_b_lca) = resolve_lca_transform(
          k1.parent_frame_id,
          k1_mat,
          p2.parent_frame_id,
          p2_mat,
          &frames_map,
        ) {
          let p2_in_lca = m_b_lca.mul_vector(p2.transform.position.to_point());
          s2.center = Vec3f32::from_array([p2_in_lca.x(), p2_in_lca.y(), p2_in_lca.z()]);
        }
      }

      if let Some(toi) = compute_toi(&s1, &s2, 1e-3, 10) {
        pairs.push(CollisionPair {
          a: ColliderId {
            entity_id: slotmap::Key::data(&k1.entity_id).as_ffi() as u32,
            primitive_index: i as u32,
          },
          b: ColliderId {
            entity_id: slotmap::Key::data(&p2.entity_id).as_ffi() as u32,
            primitive_index: j as u32,
          },
          time_of_impact: toi,
        });
      }
    }
  }

  pairs
}
