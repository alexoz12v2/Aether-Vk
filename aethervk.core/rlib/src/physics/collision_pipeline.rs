use crate::{
  gpu::{ColliderId, CollisionPair},
  physics::cpu_kernels::CpuMotionBvh,
};
use aethervk_oshal_rlib::math::{
  matrix::{Matrix4, MatrixVectorMul, SquareMatrix, mat4::Mat4x4f32},
  vector::{Vector, Vector3, Vector4, vec3::Vec3f32},
};
use alloc::vec::Vec;

pub fn detect_collisions_cpu(bvh: &CpuMotionBvh) -> Vec<CollisionPair> {
  let mut pairs = Vec::new();
  let dynamics = &bvh.rigid_bodies_copy;
  let kinematics = &bvh.kinematics_copy;
  let particles = &bvh.particles_copy;

  // Build LCA frames map
  let mut frames_map = hashbrown::HashMap::new();
  frames_map.insert(0, (0, Mat4x4f32::identity()));
  for kin in kinematics {
    frames_map.insert(
      kin.own_frame_id,
      (kin.parent_frame_id, kin.transform.to_mat4()),
    );
  }

  let get_global_pos = |frame_id: u32, local_pos: Vec3f32| -> Vec3f32 {
    let mut pos = local_pos;
    let mut curr = frame_id;
    while let Some(&(parent, ref transform)) = frames_map.get(&curr) {
      if curr == 0 {
        break;
      }
      let pt = transform.mul_vector(pos.to_point());
      pos = Vec3f32::from_array([pt.x(), pt.y(), pt.z()]);
      curr = parent;
    }
    pos
  };

  let mut query_and_push = |idx: u32,
                            entity_id_as_u32: u32,
                            parent_id: u32,
                            pos: [f32; 3],
                            vel: [f32; 3],
                            radius: f32| {
    let global_pos = get_global_pos(parent_id, Vec3f32::from_array(pos));
    let max_travel = Vec3f32::from_array(vel).length();
    let r = radius + max_travel;
    let query_aabb = crate::physics::motion_bvh::Aabb::new(
      global_pos - Vec3f32::from_array([r, r, r]),
      global_pos + Vec3f32::from_array([r, r, r]),
    );
    let mut overlaps = Vec::new();
    bvh.bvh_tree.query_intersections(&query_aabb, &mut overlaps);

    for &data_idx in &overlaps {
      if data_idx > idx {
        // Determine entity_b
        let is_kin = (data_idx & (1 << 31)) != 0;
        let is_par = (data_idx & (1 << 30)) != 0;
        let j = data_idx & 0x3FFF_FFFF;

        let ent_b_as_u32 = if is_kin {
          slotmap::Key::data(&kinematics[j as usize].entity_id).as_ffi() as u32
        } else if is_par {
          slotmap::Key::data(&particles[j as usize].entity_id).as_ffi() as u32
        } else {
          dynamics[j as usize].wrench_idx
        };

        pairs.push(CollisionPair {
          a: ColliderId {
            entity_id: entity_id_as_u32,
            primitive_index: idx,
          },
          b: ColliderId {
            entity_id: ent_b_as_u32,
            primitive_index: data_idx,
          },
          time_of_impact: 0.0,
          contact_normal: [0.0; 3],
          contact_point: [0.0; 3],
          penetration_depth: 0.0,
        });
      }
    }
  };

  for (i, p) in dynamics.iter().enumerate() {
    query_and_push(
      i as u32,
      p.wrench_idx,
      0,
      [p.position_mass[0], p.position_mass[1], p.position_mass[2]],
      [p.linear_vel_drag[0], p.linear_vel_drag[1], p.linear_vel_drag[2]],
      0.5,
    );
  }
  for (i, p) in particles.iter().enumerate() {
    query_and_push(
      (1 << 30) | (i as u32),
      slotmap::Key::data(&p.entity_id).as_ffi() as u32,
      p.parent_frame_id,
      p.position,
      p.velocity,
      1.0,
    );
  }
  for (i, k) in kinematics.iter().enumerate() {
    query_and_push(
      (1 << 31) | (i as u32),
      slotmap::Key::data(&k.entity_id).as_ffi() as u32,
      k.parent_frame_id,
      k.transform.position.into(),
      k.velocity.into(),
      k.scale,
    );
  }

  pairs
}
