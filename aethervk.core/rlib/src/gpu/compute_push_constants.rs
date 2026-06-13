//! compute_push_constants module.

// ── Legacy format (rotation-matrix based) — used by old shaders only ──────────
/// Legacy rigid-body GPU layout. Kept for old `p3-4_imex_rigidbody_imr.comp` interop.
/// New code should use [`RigidBodyImex`].
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RigidBodyLegacyGpu {
  pub position: [f32; 3],
  pub mass: f32,
  /// Row-major 3×3 rotation matrix (9 floats).
  pub rotation: [f32; 9],
  pub _pad_rot: [f32; 3],
  pub linear_velocity: [f32; 3],
  pub _pad0: f32,
  pub angular_velocity: [f32; 3],
  pub _pad1: f32,
  pub inertia_tensor: [f32; 9],
  pub _pad_inertia: [f32; 3],
}

/// Backward-compat alias so existing code compiles while migrating.
#[allow(deprecated)]
#[deprecated(
  since = "0.0.0",
  note = "Use `RigidBodyImex` for the new IMEX pipeline"
)]
pub type RigidBodyGpu = RigidBodyLegacyGpu;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleGpu {
  pub position: [f32; 3],
  pub mass: f32,
  pub velocity: [f32; 3],
  pub _pad0: f32,
  pub force: [f32; 3],
  pub _pad1: f32,
}

// ── IMEX format (quaternion based) — matches `imex_math.glsl RigidBody` ───────
/// New rigid-body GPU layout consumed by `integrate_bodies_p3.comp` and the
/// full IMEX broad-phase suite.  Layout **must** exactly mirror the GLSL
/// `RigidBody` struct in `imex_math.glsl`
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
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
  pub leaf_start_idx: u32,
  pub leaf_count: u32,
  pub shape_type: u32,
  pub shape_extents: [f32; 3],
  pub frame_idx: u32,
}

// ── Wrench (force + torque) ────────────────────────────────────────────────────
/// 6-DOF wrench accumulated per rigid body. Matches `imex_math.glsl Wrench`.
/// Layout: `struct Wrench { vec3 force; vec3 torque; }` — 6 floats, 24 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Wrench {
  pub force: [f32; 3],
  pub torque: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BvhNodeAABBGpu {
  pub min_bounds: [f32; 3],
  pub _pad_min: f32,
  pub max_bounds: [f32; 3],
  pub _pad_max: f32,
  pub left_child_or_primitive_offset: u32,
  pub right_child_offset: u32,
  pub primitive_count: u32,
  pub node_type: u32,
  pub parent_idx: u32,
  pub mass: f32,
  pub _pad2: [u32; 2],
  pub center_of_mass: [f32; 3],
  pub _pad3: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct P1_2PushConstants {
  pub particles_addr: u64,
  pub dt: f32,
  pub total_particles: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct P3_4PushConstants {
  pub rigid_bodies_addr: u64,
  pub emitters_addr: u64,
  pub dt: f32,
  pub total_bodies: u32,
  pub num_emitters: u32,
  pub _pad: u32,
}



#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EntityGpu {
  pub bvh: u64,
  pub _pad0: u64,
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
  pub _pad1: [u32; 3],
  pub shape_data: [f32; 3],
  pub _pad2: f32,
}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TlasLeaf {
  pub min_bound: [f32; 3],
  pub entity_idx: u32,
  pub max_bound: [f32; 3],
  pub metadata: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoundingBox {
  pub min_bound: [f32; 3],
  pub max_bound: [f32; 3],
}








#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
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
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
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
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LbvhBuildBottomUpPushConstants {
  pub bvh_addr: u64,
  pub sorted_morton_addr: u64,
  pub counters_addr: u64,
  pub particles_addr: u64,
  pub num_primitives: u32,
  pub particle_radius: f32,
  pub dt: f32,
  pub _pad: u32,
}







#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ApplyImpulsesPushConstants {
  pub particles_addr: u64,
  pub collisions_addr: u64,
  pub impulses_addr: u64,
  pub rigid_bodies_addr: u64,
  pub lca_entities: u64,
  pub num_rigid_bodies: u32,
  pub _pad: u32,
}



#[repr(C)]
#[derive(Clone, Copy, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AabbGpu {
  pub min_bounds: [f32; 3],
  pub _pad_min: f32,
  pub max_bounds: [f32; 3],
  pub _pad_max: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MultiBvhNodeGpu {
  pub aabbs: [AabbGpu; 2],
  pub child_ptrs: [u32; 2],
  pub parent_idx: u32,
  pub is_leaf: u32,
  pub pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MultiBvhNodeWideGpu<const N: usize> {
  pub min_x: [f32; N],
  pub max_x: [f32; N],
  pub min_y: [f32; N],
  pub max_y: [f32; N],
  pub min_z: [f32; N],
  pub max_z: [f32; N],
  pub child_indices: [u32; N],
  pub metadata: [u32; N],
  pub masses: [f32; N],
  pub com_x: [f32; N],
  pub com_y: [f32; N],
  pub com_z: [f32; N],
  pub particle_start: [u32; N],
  pub particle_count: [u32; N],
  pub force_x: [f32; N],
  pub force_y: [f32; N],
  pub force_z: [f32; N],
  pub valid_mask: [u32; 2],
  pub parent_idx: u32,
  pub _pad: u32,
}

// SAFETY: MultiBvhNodeWideGpu<N> is #[repr(C)] with only f32/u32 fields (all Pod),
// for the concrete sizes we actually use (powers of two 4..=128).
unsafe impl bytemuck::Zeroable for MultiBvhNodeWideGpu<4> {}
unsafe impl bytemuck::Zeroable for MultiBvhNodeWideGpu<8> {}
unsafe impl bytemuck::Zeroable for MultiBvhNodeWideGpu<16> {}
unsafe impl bytemuck::Zeroable for MultiBvhNodeWideGpu<32> {}
unsafe impl bytemuck::Zeroable for MultiBvhNodeWideGpu<64> {}
unsafe impl bytemuck::Zeroable for MultiBvhNodeWideGpu<128> {}
unsafe impl bytemuck::Pod for MultiBvhNodeWideGpu<4> {}
unsafe impl bytemuck::Pod for MultiBvhNodeWideGpu<8> {}
unsafe impl bytemuck::Pod for MultiBvhNodeWideGpu<16> {}
unsafe impl bytemuck::Pod for MultiBvhNodeWideGpu<32> {}
unsafe impl bytemuck::Pod for MultiBvhNodeWideGpu<64> {}
unsafe impl bytemuck::Pod for MultiBvhNodeWideGpu<128> {}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LbvhPrepassPushConstants {
  pub bvh: u64,
  pub counters_addr: u64,
  pub num_internal_nodes: u32,
  pub _pad: u32,
  /// BDA of a 1-element `u32` watchdog buffer. Shader writes a shader-ID bit via atomicOr.
  /// 0 if watchdog reporting is disabled (non-test builds).
  pub watchdog_out: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MotionBoundsPushConstants {
  pub bvh: u64,
  pub primitive_data_addr: u64,
  pub num_primitives: u32,
  pub dt: f32,
  pub particle_radius: f32,
  pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MotionRefitPushConstants {
  pub bvh: u64,
  pub depth_indices_addr: u64,
  pub total_nodes_at_depth: u32,
  pub _pad: u32,
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  #[should_panic]
  fn print_sizes() {
    println!(
      "MultiBvhNodeWideGpu<4>: {}",
      core::mem::size_of::<MultiBvhNodeWideGpu<4>>()
    );
    println!(
      "MultiBvhNodeWideGpu<8>: {}",
      core::mem::size_of::<MultiBvhNodeWideGpu<8>>()
    );
    println!(
      "MultiBvhNodeWideGpu<16>: {}",
      core::mem::size_of::<MultiBvhNodeWideGpu<16>>()
    );
    println!(
      "MultiBvhNodeWideGpu<32>: {}",
      core::mem::size_of::<MultiBvhNodeWideGpu<32>>()
    );
    println!(
      "MultiBvhNodeWideGpu<64>: {}",
      core::mem::size_of::<MultiBvhNodeWideGpu<64>>()
    );
    println!(
      "MultiBvhNodeWideGpu<128>: {}",
      core::mem::size_of::<MultiBvhNodeWideGpu<128>>()
    );
    panic!("Show me the sizes");
  }
}/// `narrow_ccd.comp` push constants — **56 bytes** (matches SPIR-V layout exactly).
///
/// GLSL offsets (std430):
/// ```text
///  0  RigidBodyArray scene_entities   (u64 BDA)
///  8  uint64_t       particles        (u64 BDA)
/// 16  uint64_t       output_list      (u64 BDA)
/// 24  uint64_t       pair_buffer      (u64 BDA)
/// 32  float          dt
/// 36  float          particle_radius
/// 40  GpuReferenceFrameArray lca_entities  (u64 BDA)
/// 48  uint           space_type  (0 = rb-rb PairBuffer, 1 = CrossPairBuffer)
/// 52  uint           _pad
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NarrowCcdPushConstants {
  pub scene_entities: u64,
  pub particles: u64,
  pub output_list: u64,
  pub pair_buffer: u64,
  pub dt: f32,
  pub particle_radius: f32,
  pub lca_entities: u64, // frames / lca BDA
  pub space_type: u32,
  pub _pad: u32,
}

/// `narrow_ccd_cross_lca.comp` push constants — **56 bytes**.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NarrowCcdCrossLcaPushConstants {
  pub scene_entities: u64,
  pub particles: u64,
  pub cross_output_list: u64,
  pub cross_pair_buffer: u64,
  pub dt: f32,
  pub particle_radius: f32,
  pub lca_entities: u64, // frames / lca BDA
  pub space_type: u32,
  pub _pad: u32,
}

/// Push constants for `narrow_ccd_particles.comp`
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NarrowCcdParticlesPushConstants {
  pub scene_entities: u64,
  pub output_list: u64,
  pub particles: u64,
  pub num_rigid_bodies: u32,
  pub num_particles: u32,
  pub dt: f32,
  pub particle_radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Push constants for `integrate_particles_p1_p2.comp` (legacy)
pub struct P12PushConstants {
  pub particles: u64,
  pub dt: f32,
  pub total_particles: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Push constants for `lbvh_build.comp` or `lbvh_build_bottomup.comp`
pub struct LbvhPushConstants {
  pub bvh: u64,
  pub sorted_morton: u64,
  pub counters: u64,
  pub particles: u64,
  pub num_primitives: u32,
  pub particle_radius: f32,
  /// BDA of a 1-element `u32` watchdog buffer. Shader writes a shader-ID bit via atomicOr.
  /// 0 if watchdog reporting is disabled (non-test builds).
  pub watchdog_out: u64,
  pub dt: f32,
  pub _pad: u32,
  pub _pad_end: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Push constants for `lbvh_collapse.comp`: collapses binary BVH → wide multi-BVH.
pub struct LbvhCollapsePushConstants {
  /// BDA of the source binary BVH (MultiBvhNodeWideGpu<N> layout, Karras tree).
  pub binary_bvh: u64,
  /// BDA of the destination wide multi-BVH buffer.
  pub multi_bvh: u64,
  /// BDA of the collapse map buffer: u32 array of binary_root indices per multi-node.
  pub collapse_map: u64,
  /// Number of wide multi-nodes to produce (= dispatch groups).
  pub num_multi_nodes: u32,
  pub _pad: u32,
}


#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Push constants for `ccd.comp`
pub struct CcdPushConstants {
  pub particle_bvh: u64,
  pub output_list: u64,
  pub root_index: u32,
  pub total_particles: u32,
}


#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Push constants for `stream_compact.comp`
pub struct StreamCompactPushConstants {
  pub sparse_in: u64,
  pub packed_out: u64,
  pub total_elements: u32,
  pub _pad: u32,
  pub _pad_align16: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Push constants for `reduce_toi.comp`
pub struct ReduceToiPushConstants {
  pub particles: u64,
  pub collisions: u64,
  pub out_toi: u64,
  pub particle_radius: f32,
  pub dt: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Push constants for `lcp_solver.comp`
pub struct LcpPushConstants {
  pub particles: u64,
  pub collisions: u64,
  pub outputs: u64,
  pub total_clusters: u32,
  pub num_rigid_bodies: u32,
  pub rigid_bodies: u64,
  pub dt: f32,
  pub restitution: f32,
  pub lca_entities: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Push constants for `barnes_hut.comp`
pub struct BarnesHutPushConstants {
  pub particles: u64,
  pub bvh: u64,
  pub cluster_list: u64,
  pub wrenches: u64,
  pub num_clusters: u32,
  pub dt: f32,
  pub theta: f32,
  pub g: f32,
  pub softening_sq: f32,
  pub root_node_idx: u32,
  pub cluster_threshold: u32,
  pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Push constants for `integrate_particles_p4_5.comp` (legacy)
pub struct P5PushConstants {
  pub particles: u64,
  pub emitters: u64,
  pub kinematics: u64,
  pub dt: f32,
  pub total_particles: u32,
  pub num_emitters: u32,
  pub num_kinematics: u32,
  pub _pad_align16: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Push constants for `integrate_bodies_p3.comp` (legacy)
pub struct P34PushConstants {
  pub rigid_bodies: u64,
  pub emitters: u64,
  pub kinematics: u64,
  pub dt: f32,
  pub total_rigid_bodies: u32,
  pub num_emitters: u32,
  pub num_kinematics: u32,
  pub _pad_align16: [u32; 2],
}

/// `integrate_particles_p1_p2.comp` — 16 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImexParticlesP12PushConstants {
  /// BDA to AOSOA particle data (float[])
  pub particles: u64,
  /// Physical dt in seconds
  pub dt: f32,
  pub total_particles: u32,
}

/// `integrate_bodies_p3.comp` — 40 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImexBodiesP3PushConstants {
  /// BDA to RigidBodyArray (imex_math.glsl layout: quaternion + wrench_idx)
  pub rigid_bodies: u64,
  /// BDA to WrenchArray (6-float Wrench per entry)
  pub wrenches: u64,
  pub emitters: u64,
  /// BDA to GpuReferenceFrameArray — used for macro→micro position transform
  pub frames: u64,
  /// Physical dt in seconds
  pub dt: f32,
  pub n_bodies: u32,
  /// Picard iteration count (4 is sufficient for most scenes; 8–10 for high-spin)
  pub n_iterations: u32,
  pub num_emitters: u32,
}

/// `integrate_particles_p4_5.comp` — 40 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImexParticlesP45PushConstants {
  /// BDA to AOSOA particle data
  pub particles: u64,
  /// BDA to ClockBuffer (uvec2 global_time_us)
  pub clock: u64,
  /// Physical dt in seconds
  pub dt: f32,
  pub total_particles: u32,
  /// dt in microseconds — low 32 bits
  pub dt_us_lo: u32,
  /// dt in microseconds — high 32 bits
  pub dt_us_hi: u32,
  /// t_n (current frame start) in microseconds — low 32 bits
  pub current_time_lo: u32,
  /// t_n (current frame start) in microseconds — high 32 bits
  pub current_time_hi: u32,
  pub _pad_align16: [u32; 2],
}

/// `rb_force_assign.comp` — 24 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RbForceAssignPushConstants {
  /// BDA to RigidBodyArray (read-only; only leaf_start_idx / leaf_count / wrench_idx used)
  pub rigid_bodies: u64,
  /// BDA to WrenchArray (leaf wrenches AND CoM wrench; both in same buffer)
  pub wrenches: u64,
  pub n_bodies: u32,
  pub _pad: u32,
  pub _pad_align16: [u32; 2],
}

/// `bp_clear.comp` — 32 bytes  (4 × 8-byte BDAs)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BpClearPushConstants {
  pub raw_scene_pairs: u64,
  pub out_rb_rb: u64,
  pub out_rb_ps: u64,
  pub out_rb_lca: u64,
  pub out_internal: u64,
  pub out_sparse: u64,
}

/// `bp_bounds_gen.comp` — 28 bytes
///
/// Matches the existing `bp_bounds_gen.comp` push-constant block:
/// ```glsl
/// EntityArray scene_entities;  // BDA
/// LeafBuffer  tlas_leaves;     // BDA
/// uvec2       dt_us;
/// uint        total_entities;
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BpBoundsGenPushConstants {
  pub scene_entities: u64,
  pub tlas_leaves: u64,
  pub lca_entities: u64,
  pub dt_us_lo: u32,
  pub dt_us_hi: u32,
  pub total_entities: u32,
  pub _pad: u32,
}

/// `bp_scene.comp` — 40 bytes
///
/// Matches:
/// ```glsl
/// MultiBvhBuffer tlas_bvh;          // BDA (8 bytes — treated as u64 opaque)
/// LeafBuffer     query_leaves;       // BDA
/// PairBuffer     overlapping_pairs;  // BDA
/// uint           tlas_root_index;
/// uint           total_queries;
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BpScenePushConstants {
  pub tlas_bvh: u64,
  pub query_leaves: u64,
  pub overlapping_pairs: u64,
  pub tlas_root_index: u32,
  pub total_queries: u32,
}

/// `bp_classify.comp` — 40 bytes
///
/// ```glsl
/// EntityArray scene_entities;  // BDA
/// RawPairs    raw_pairs;       // BDA
/// QueueBuf    out_rb_rb;       // BDA
/// QueueBuf    out_rb_ps;       // BDA
/// QueueBuf    out_rb_lca;      // BDA
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BpClassifyPushConstants {
  pub raw_pairs: u64,
  pub out_rb_rb: u64,
  pub out_rb_ps: u64,
  pub out_ps_ps: u64,
  pub out_macro_lca: u64,
  pub out_lca_lca: u64,
  pub max_pairs: u32,
  pub num_rigid_bodies: u32,
  pub rigid_bodies: u64,
}

/// `bp_cross_lca.comp` — 88 bytes
///
/// ```glsl
/// MultiBvhBuffer      tlas_bvh;
/// GpuReferenceFrameArray lca_entities;
/// LeafBuffer          macro_leaves;
/// RigidBodyArray      rigid_bodies;      // replaces old EntityHeaderArray
/// PairBuffer          lca_query_pairs;
/// PairBuffer          out_rb_rb;
/// PairBuffer          out_rb_ps;
/// PairBuffer          out_ps_ps;
/// CrossPairBuffer     out_cross_pairs;
/// uint                total_queries;
/// uint                max_pairs;
/// uint                num_rigid_bodies;  // threshold for entity type detection
/// uint                _pad;
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BpCrossLcaPushConstants {
  pub tlas_bvh_addr: u64,
  pub lca_entities: u64,
  pub macro_leaves: u64,
  /// BDA of the rigid-body array — used by the shader to detect entity type
  /// by index comparison (`id < num_rigid_bodies → TYPE_RIGID_BODY`).
  pub rigid_bodies: u64,
  pub lca_query_pairs: u64,
  pub out_rb_rb: u64,
  pub out_rb_ps: u64,
  pub out_ps_ps: u64,
  pub out_cross_pairs: u64,
  pub total_queries: u32,
  pub max_pairs: u32,
  pub num_rigid_bodies: u32,
  pub _pad: u32,
}

/// `bp_particle_self.comp` — 40 bytes
///
/// ```glsl
/// MultiBvhBuffer bvh;            // BDA
/// ParticleData   particles;      // BDA
/// WrenchArray    wrench_buffer;  // BDA (writes into AOSOA force slots via atomicAdd)
/// uint           root_index;
/// uint           total_particles;
/// float          particle_radius;
/// float          stiffness;
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BpParticleSelfPushConstants {
  pub bvh: u64,
  pub particles: u64,
  pub wrench_buffer: u64,
  pub root_index: u32,
  pub total_particles: u32,
  pub particle_radius: f32,
  pub stiffness: f32,
  pub _pad_align16: [u32; 2],
}

/// `apply_emitters_to_particles.comp` — 40 bytes
///
/// Applies macro-frame gravity emitters to microframe particles, performing
/// GPU-inline macro (AU) → micro (km) coordinate transform per particle frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ApplyEmittersPushConstants {
  pub particles: u64,
  pub emitters: u64,
  pub frames: u64,
  pub particle_frame_ids: u64,
  pub bvh: u64,
  pub num_emitters: u32,
  pub total_particles: u32,
  pub root_node_idx: u32,
  pub _pad: [u32; 3],
}

/// `apply_emitters_direct.comp` — 48 bytes
///
/// Direct per-particle external-gravity pass for test-particle systems
/// (comet dust, tracers) where inter-particle self-gravity is disabled.
/// No BVH is needed: each thread reads its own AOSOA position, computes
/// the gravitational acceleration from all emitters, and atomically adds
/// the force into AOSOA slots 7-9.
///
/// glslc rounds push-constant blocks up to a multiple of 16 bytes,
/// so 40 bytes of data → 48-byte block (3 × 16).  The trailing `_pad`
/// keeps the Rust struct in sync with the SPIR-V reflection.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ApplyEmittersDirectPushConstants {
  pub particles: u64,
  pub emitters: u64,
  pub frames: u64,
  pub particle_frame_ids: u64,
  pub num_emitters: u32,
  pub total_particles: u32,
  pub _pad: [u32; 2], // align block to 48 bytes (3 × 16)
}

/// Push constants for `accumulate_bvh_forces_to_particles.comp` (Phase B).
/// Splats per-cluster BVH forces (self-gravity + external) back to per-particle
/// AOSOA slots 7-9 via a leaf→root traversal.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AccumBvhForcesPushConstants {
  pub particles: u64,      // ParticleData BDA
  pub bvh: u64,            // MultiBvhBuffer BDA
  pub total_particles: u32,
  pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RadixSortPushConstants {
  pub input_keys: u64,
  pub output_keys: u64,
  pub histograms: u64,
  pub num_particles: u32,
  pub shift: u32,
  pub stage: u32,
  pub num_blocks: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MortonEncodePushConstants {
  pub morton_out: u64,       // offset 0,  8 bytes
  pub particles: u64,        // offset 8,  8 bytes
  pub num_particles: u32,    // offset 16, 4 bytes
  pub _pad0: [u32; 3],       // offset 20, 12 bytes → scene_min at offset 32
  pub scene_min: [f32; 3],   // offset 32, 12 bytes
  pub _pad1: u32,            // offset 44, 4 bytes  → scene_max at offset 48
  pub scene_max: [f32; 3],   // offset 48, 12 bytes
  pub _pad2: u32,            // offset 60, 4 bytes  → total = 64 bytes
}

const _ASSERT_MORTON_PC_SIZE: () =
  assert!(core::mem::size_of::<MortonEncodePushConstants>() == 64);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PermuteParticlesPushConstants {
  pub particles_in: u64,
  pub particles_out: u64,
  pub frame_ids_in: u64,
  pub frame_ids_out: u64,
  pub sorted_morton: u64,
  pub num_particles: u32,
  pub _pad: [u32; 1],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Push constants for `emit_particles.comp`
pub struct EmitParticlesPushConstants {
  pub particles: u64,
  pub candidates: u64,
  pub bvh: u64,
  pub counter: u64,
  pub root_index: u32,
  pub num_candidates: u32,
  pub _pad0: [u32; 2],
  pub sun_pos: [f32; 3],
  pub _pad1: u32,
}

