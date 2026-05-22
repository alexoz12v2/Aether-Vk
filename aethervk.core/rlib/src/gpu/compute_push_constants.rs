//! compute_push_constants module.

// ── Legacy format (rotation-matrix based) — used by old shaders only ──────────
/// Legacy rigid-body GPU layout. Kept for old `p3-4_imex_rigidbody_imr.comp` interop.
/// New code should use [`RigidBodyImex`].
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RigidBodyLegacyGpu {
  pub position: [f32; 3],
  pub mass: f32,
  /// Row-major 3×3 rotation matrix (9 floats).
  pub rotation: [f32; 9],
  pub linear_velocity: [f32; 3],
  pub _pad0: f32,
  pub angular_velocity: [f32; 3],
  pub _pad1: f32,
  pub inertia_tensor: [f32; 9],
}

/// Backward-compat alias so existing code compiles while migrating.
#[allow(deprecated)]
#[deprecated(since = "0.0.0", note = "Use `RigidBodyImex` for the new IMEX pipeline")]
pub type RigidBodyGpu = RigidBodyLegacyGpu;

// ── IMEX format (quaternion based) — matches `imex_math.glsl RigidBody` ───────
/// New rigid-body GPU layout consumed by `integrate_bodies_p3.comp` and the
/// full IMEX broad-phase suite.  Layout **must** exactly mirror the GLSL
/// `RigidBody` struct in `imex_math.glsl` (scalar block layout).
///
/// ```text
/// layout(scalar) struct RigidBody {
///   vec4  position_mass;          // xyz = CoM, w = mass
///   vec4  orientation;            // unit quaternion (x,y,z,w)
///   vec4  linear_vel_drag;        // xyz = v_lin, w = drag coeff
///   vec4  angular_vel_drag;       // xyz = ω, w = rotational drag
///   vec3  inertia_inv_diag;       // diagonal of I^{-1} in body frame
///   uint  wrench_idx;             // index into WrenchArray
/// };
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct RigidBodyImex {
  /// CoM position (xyz) + mass (w).
  pub position_mass: [f32; 4],
  /// Unit quaternion (x, y, z, w).
  pub orientation: [f32; 4],
  /// Linear velocity (xyz) + drag coefficient (w).
  pub linear_vel_drag: [f32; 4],
  /// Angular velocity (xyz) + rotational drag (w).
  pub angular_vel_drag: [f32; 4],
  /// Diagonal of I^{-1} in body frame (xyz) + unused (w for alignment).
  pub inertia_inv_diag: [f32; 4],
  /// Index into the per-frame `WrenchArray` buffer.
  pub wrench_idx: u32,
  pub _pad: [u32; 3],
}

// ── Wrench (force + torque) ────────────────────────────────────────────────────
/// 6-DOF wrench accumulated per rigid body. Matches `imex_math.glsl Wrench`.
/// Layout: `struct Wrench { vec3 force; vec3 torque; }` — 6 floats, 24 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Wrench {
  pub force: [f32; 3],
  pub _pad0: f32,
  pub torque: [f32; 3],
  pub _pad1: f32,
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
pub struct BpScenePushConstants {
  pub tlas_bvh: u64,
  pub query_leaves: u64,
  pub overlapping_pairs: u64,
  pub tlas_root_index: u32,
  pub total_queries: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BpParticleSelfPushConstants {
  pub bvh: u64,
  pub particles: u64,
  pub wrench_buffer: u64,
  pub root_index: u32,
  pub total_particles: u32,
  pub particle_radius: f32,
  pub stiffness: f32,
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

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct AabbGpu {
  pub min_bounds: [f32; 3],
  pub max_bounds: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MultiBvhNodeGpu {
  pub aabbs: [AabbGpu; 2],
  pub child_ptrs: [u32; 2],
  pub parent_idx: u32,
  pub is_leaf: u32,
  pub pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LbvhPrepassPushConstants {
  pub bvh: u64,
  pub counters_addr: u64,
  pub num_internal_nodes: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MotionBoundsPushConstants {
  pub bvh: u64,
  pub primitive_data_addr: u64,
  pub num_primitives: u32,
  pub dt: f32,
  pub particle_radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MotionRefitPushConstants {
  pub bvh: u64,
  pub depth_indices_addr: u64,
  pub total_nodes_at_depth: u32,
}
