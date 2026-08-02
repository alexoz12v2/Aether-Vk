//! compute_push_constants module.

/// Push constants for shader `apply_emitters_direct_new.comp`
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ApplyEmittersDirectNewPushConstants {
  pub global_particle_buffer_address: u64,
  pub particle_page_table: u64,
  pub emitter_array: u64, // BDA
  pub emitter_count: u32,
  pub _pad: u32,
}

/// Push constants for shader `integrate_particles_p1_p2_new.comp`
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IntegrateParticlesP1P2NewPushConstants {
  pub global_particle_buffer_address: u64,
  pub particle_page_table: u64,
  pub delta_time: f32,
  pub _pad: u32,
}

/// Push constants for shader `integrate_particles_p4_5_new.comp`
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IntegrateParticlesP45NewPushConstants {
  pub global_particle_buffer_address: u64,
  pub particle_page_table: u64,
  pub delta_time: f32,
  pub _pad: u32,
}

/// Push constants for shader `new_particles_compact_reset.comp`
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NewParticlesCompactResetPushConstants {
  pub particle_page_table: u64,
  pub max_chunks: u32,
  pub _pad: u32,
}

/// Push constants for shader `new_particles_emit.comp`
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NewParticlesEmitPushConstants {
  pub global_particle_buffer: u64,
  pub particle_page_table: u64,
  pub free_list: u64,

  pub cone_dir_aperture: [f32; 4],
  pub mass_vel_mean_std: [f32; 4],

  pub emit_count: u32,
  pub current_time: u32, // 1/300 s, scaled time
  pub seed: u32,
  pub radius: f32, // m

  pub sun_dir_and_beta: [f32; 4],
}

/// Push constants for shader `new_particles_compact.comp`
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NewParticlesCompactPushConstants {
  pub global_particle_buffer_address: u64,
  pub particle_page_table: u64,
  pub free_list: u64,
  pub doomsday: u32, // 1/300 s. Scaled
  pub now: u32,      // 1/300 s. Scaled
  pub max_chunks: u32,
  pub _pad: u32,
}

/// Push constants for shader `new_particles_offset_particles.comp`
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NewParticlesOffsetParticlesPushConstants {
  pub global_particle_buffer: u64,
  pub particle_page_table: u64,
  pub delta_rot: [f32; 4],
  pub delta_pos: [f32; 4],
}
