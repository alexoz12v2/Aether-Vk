//! compute_push_constants module.

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RigidBodyGpu {
  pub position: [f32; 3],
  pub mass: f32,
  pub rotation: [f32; 9],
  pub linear_velocity: [f32; 3],
  pub _pad0: f32,
  pub angular_velocity: [f32; 3],
  pub _pad1: f32,
  pub inertia_tensor: [f32; 9],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BvhNodeAABBGpu {
  pub min_bounds: [f32; 3],
  pub max_bounds: [f32; 3],
  pub left_child_or_primitive_offset: u32,
  pub right_child_offset: u32,
  pub primitive_count: u32,
  pub node_type: u32,
  pub parent_idx: u32,
  pub mass: f32,
  pub center_of_mass: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct P1_2PushConstants {
  pub particles_addr: u64,
  pub dt: f32,
  pub total_particles: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct P3_4PushConstants {
  pub rigid_bodies_addr: u64,
  pub emitters_addr: u64,
  pub dt: f32,
  pub total_bodies: u32,
  pub num_emitters: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct P5PushConstants {
  pub particles_addr: u64,
  pub emitters_addr: u64,
  pub dt: f32,
  pub total_particles: u32,
  pub num_emitters: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct EntityGpu {
  pub bvh: u64,
  pub transform: [[f32; 4]; 4],
  pub inv_transform: [[f32; 4]; 4],
  pub linear_velocity: [f32; 3],
  pub root_index: u32,
  pub angular_velocity: [f32; 3],
  pub entity_type: u32,
  pub primitive_offset: u32,
  pub total_primitives: u32,
  pub frame_scale_type: u32,
  pub scale_factor: f32,
  pub shape_type: u32,
  pub shape_data: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BroadPhasePushConstants {
  pub tlas_bvh_addr: u64,
  pub scene_entities_addr: u64,
  pub overlapping_pairs_addr: u64,
  pub tlas_root_index: u32,
  pub total_entities: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CcdPushConstants {
  pub particle_bvh: u64,
  pub output_list: u64,
  pub root_index: u32,
  pub total_particles: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NarrowPhaseParticlesPushConstants {
  pub scene_entities_addr: u64,
  pub output_list_addr: u64,
  pub particles_addr: u64,
  pub entity_a_idx: u32,
  pub entity_b_idx: u32,
  pub dt: f32,
  pub particle_radius: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NarrowPhaseRigidBodyPushConstants {
  pub scene_entities_addr: u64,
  pub output_list_addr: u64,
  pub particles_addr: u64,
  pub entity_a_idx: u32,
  pub entity_b_idx: u32,
  pub dt: f32,
  pub particle_radius: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct LbvhPushConstants {
  pub bvh_addr: u64,
  pub sorted_morton_addr: u64,
  pub counters_addr: u64,
  pub particles_addr: u64,
  pub num_primitives: u32,
  pub particle_radius: f32,
  pub dt: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct StreamCompactPushConstants {
  pub sparse_in_addr: u64,
  pub packed_out_addr: u64,
  pub total_elements: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ReduceToiPushConstants {
  pub particles_addr: u64,
  pub collisions_addr: u64,
  pub out_toi_addr: u64,
  pub particle_radius: f32,
  pub dt: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct LcpPushConstants {
  pub particles_addr: u64,
  pub collisions_addr: u64,
  pub impulses_addr: u64,
  pub total_clusters: u32,
  pub rigid_bodies_addr: u64,
  pub dt: f32,
  pub restitution: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ApplyImpulsesPushConstants {
  pub particles_addr: u64,
  pub collisions_addr: u64,
  pub impulses_addr: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BarnesHutPushConstants {
  pub particles_addr: u64,
  pub bvh_addr: u64,
  pub root_index: u32,
  pub total_particles: u32,
  pub theta: f32,
  pub g: f32,
}
