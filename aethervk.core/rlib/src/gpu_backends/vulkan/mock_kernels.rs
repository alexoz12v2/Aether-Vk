#![cfg(ignore)]
use crate::{
  gpu::{
    CollisionPair, CommandBufferSyncInfo, DeviceBuffer, ForceEmitter, Kernels, KinematicBody,
    ParticleMetadata, RigidBodyImex, Wrench,
  },
  gpu_backends::vulkan::{
    device::Device,
    physics::{VulkanBuffer, VulkanCommandBuffer},
  },
  physics::physics_scene::{GpuReferenceFrame, PhysicsScene},
  scene::Scene,
  simulation_api::structs::{MockTargetShader, SHADER_MOCK_RESULTS},
  types::EngineResult,
};
use aethervk_oshal_rlib::{math::vector::vec3::Vec3f32, os::time::timeus_t};

pub struct MockVulkanKernels<'a> {
  pub base: &'a Device,
  pub target: MockTargetShader,
  pub scene_id: u64,
}

impl<'a> Kernels for MockVulkanKernels<'a> {
  type Cmd = VulkanCommandBuffer;
  type Buffer<T: Copy + Send + Sync> = VulkanBuffer<T>;
  type List<T: Copy + Send + Sync> = VulkanBuffer<T>;
  type MotionBvh = VulkanBuffer<()>;
  type MotionTlas = VulkanBuffer<()>;

  fn toggle_particle_self_gravity(&self, _enable: bool) {}

  fn discard_buffer<T: Copy + Send + Sync>(&self, buffer: VulkanBuffer<T>) {
    self.base.discard_buffer(buffer)
  }
  fn discard_list<T: Copy + Send + Sync>(&self, list: VulkanBuffer<T>) {
    self.base.discard_list(list)
  }
  fn discard_bvh(&self, bvh: VulkanBuffer<()>) {
    self.base.discard_bvh(bvh)
  }
  fn discard_tlas(&self, tlas: VulkanBuffer<()>) {
    self.base.discard_tlas(tlas)
  }
  fn read_buffer_u32_first(&self, buf: &VulkanBuffer<u32>) -> EngineResult<u32> {
    self.base.read_buffer_u32_first(buf)
  }
  fn subgroup_size(&self) -> Option<crate::gpu::SubgroupSize> {
    self.base.subgroup_size()
  }
  fn wait_sync(&self, sync: &CommandBufferSyncInfo) -> EngineResult<()> {
    self.base.wait_sync(sync)
  }

  fn wait_idle(&self) -> EngineResult<()> {
    self.base.wait_idle()
  }

  fn is_cpu_device(&self) -> bool {
    self.base.is_cpu_device()
  }
  fn refit_motion_blas(
    &self,
    cmd: &mut VulkanCommandBuffer,
    bvh: &VulkanBuffer<()>,
    depth_indices: &VulkanBuffer<u32>,
    total_nodes: u32,
  ) -> EngineResult<()> {
    self.base.refit_motion_blas(cmd, bvh, depth_indices, total_nodes)
  }
  fn upload_motion_tlas(
    &self,
    cmd: &mut VulkanCommandBuffer,
    node_bytes: &[u8],
  ) -> EngineResult<VulkanBuffer<()>> {
    self.base.upload_motion_tlas(cmd, node_bytes)
  }
  fn create_command_buffer(&self) -> EngineResult<VulkanCommandBuffer> {
    self.base.create_command_buffer()
  }
  fn build_list<T: Copy + Send + Sync>(
    &self,
    cmd: &mut VulkanCommandBuffer,
    capacity: usize,
  ) -> EngineResult<VulkanBuffer<T>> {
    self.base.build_list(cmd, capacity)
  }
  fn build_leaves(
    &self,
    cmd: &mut VulkanCommandBuffer,
    capacity: usize,
  ) -> EngineResult<VulkanBuffer<[u32; 8]>> {
    self.base.build_leaves(cmd, capacity)
  }
  fn build_kinematic_bodies(
    &self,
    cmd: &mut VulkanCommandBuffer,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<VulkanBuffer<KinematicBody>> {
    self.base.build_kinematic_bodies(cmd, scene, scene0)
  }
  fn build_rigid_bodies(
    &self,
    cmd: &mut VulkanCommandBuffer,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<(VulkanBuffer<RigidBodyImex>, VulkanBuffer<Wrench>, u32)> {
    self.base.build_rigid_bodies(cmd, scene, scene0)
  }
  fn build_frames(
    &self,
    cmd: &mut VulkanCommandBuffer,
    scene: &PhysicsScene,
  ) -> EngineResult<VulkanBuffer<GpuReferenceFrame>> {
    self.base.build_frames(cmd, scene)
  }
  fn build_particles(
    &self,
    cmd: &mut VulkanCommandBuffer,
    scene: &Scene,
  ) -> EngineResult<
    alloc::vec::Vec<(
      crate::scene::EntityId,
      VulkanBuffer<f32>,
      alloc::vec::Vec<ParticleMetadata>,
      bool,
    )>,
  > {
    self.base.build_particles(cmd, scene)
  }
  fn build_particle_frame_ids(
    &self,
    cmd: &mut VulkanCommandBuffer,
    particle_metadata: &[ParticleMetadata],
  ) -> EngineResult<VulkanBuffer<u32>> {
    self.base.build_particle_frame_ids(cmd, particle_metadata)
  }
  fn build_emitters(
    &self,
    cmd: &mut VulkanCommandBuffer,
    scene: &Scene,
  ) -> EngineResult<(VulkanBuffer<ForceEmitter>, u32)> {
    self.base.build_emitters(cmd, scene)
  }
  fn build_emission_candidates(
    &self,
    cmd: &mut VulkanCommandBuffer,
    scene: &Scene,
  ) -> EngineResult<VulkanBuffer<f32>> {
    self.base.build_emission_candidates(cmd, scene)
  }

  // --- MOCKED DISPATCH METHODS ---

  fn emit_particles(
    &self,
    cmd: &mut VulkanCommandBuffer,
    particles: &mut VulkanBuffer<f32>,
    physical_scene: &PhysicsScene,
    scene: &Scene,
    sun_pos: Vec3f32,
    dt: timeus_t,
  ) -> EngineResult<()> {
    if self.target == MockTargetShader::EmitParticles {
      self.base.emit_particles(cmd, particles, physical_scene, scene, sun_pos, dt)?;
    }
    Ok(())
  }

  fn step_ode_p1_p2(
    &self,
    _cmd: &mut VulkanCommandBuffer,
    _particles: &mut VulkanBuffer<f32>,
    _dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn step_ode_p3_p4(
    &self,
    _cmd: &mut VulkanCommandBuffer,
    _kinematics: &VulkanBuffer<KinematicBody>,
    _rigid_bodies: &mut VulkanBuffer<crate::gpu::RigidBodyGpu>,
    _emitters: &VulkanBuffer<ForceEmitter>,
    _dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn compute_self_gravity(
    &self,
    _cmd: &mut VulkanCommandBuffer,
    _bvh: &VulkanBuffer<()>,
    _particles: &mut VulkanBuffer<f32>,
  ) -> EngineResult<()> {
    if self.target == MockTargetShader::BarnesHut {
      self.base.compute_self_gravity(_cmd, _bvh, _particles)?;
    }
    Ok(())
  }

  fn step_ode_p5(
    &self,
    _cmd: &mut VulkanCommandBuffer,
    _kinematics: &VulkanBuffer<KinematicBody>,
    _particles: &mut VulkanBuffer<f32>,
    _emitters: &VulkanBuffer<ForceEmitter>,
    _dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn imex_integrate_particles_p1_p2(
    &self,
    cmd: &mut VulkanCommandBuffer,
    particles: &mut VulkanBuffer<f32>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    if self.target == MockTargetShader::IntegrateParticlesP1P2 {
      self.base.imex_integrate_particles_p1_p2(cmd, particles, dt)?;
    }
    Ok(())
  }

  fn imex_integrate_bodies_p3(
    &self,
    cmd: &mut VulkanCommandBuffer,
    bodies: &mut VulkanBuffer<RigidBodyImex>,
    wrenches: &mut VulkanBuffer<Wrench>,
    emitters: &VulkanBuffer<ForceEmitter>,
    frames: &VulkanBuffer<crate::physics::physics_scene::GpuReferenceFrame>,
    n_bodies: u32,
    n_emitters: u32,
    dt: timeus_t,
  ) -> EngineResult<()> {
    if self.target == MockTargetShader::IntegrateBodiesP3 {
      self.base.imex_integrate_bodies_p3(
        cmd, bodies, wrenches, emitters, frames, n_bodies, n_emitters, dt,
      )?;
    }
    Ok(())
  }

  fn imex_rb_force_assign(
    &self,
    cmd: &mut VulkanCommandBuffer,
    bodies: &VulkanBuffer<RigidBodyImex>,
    wrenches: &mut VulkanBuffer<Wrench>,
    n_bodies: u32,
  ) -> EngineResult<()> {
    if self.target == MockTargetShader::RbForceAssign {
      self.base.imex_rb_force_assign(cmd, bodies, wrenches, n_bodies)?;
    }
    Ok(())
  }

  fn imex_integrate_particles_p4_5(
    &self,
    cmd: &mut VulkanCommandBuffer,
    particles: &mut VulkanBuffer<f32>,
    dt: timeus_t,
    current_time_us: timeus_t,
  ) -> EngineResult<()> {
    if self.target == MockTargetShader::IntegrateParticlesP4P5 {
      self.base.imex_integrate_particles_p4_5(cmd, particles, dt, current_time_us)?;
    }
    Ok(())
  }

  fn apply_emitters_to_particles(
    &self,
    cmd: &mut VulkanCommandBuffer,
    particles: &mut VulkanBuffer<f32>,
    emitters: &VulkanBuffer<ForceEmitter>,
    frames: &VulkanBuffer<GpuReferenceFrame>,
    particle_frame_ids: &VulkanBuffer<u32>,
    bvh: &VulkanBuffer<()>,
    num_emitters: u32,
  ) -> EngineResult<()> {
    if self.target == MockTargetShader::ApplyEmitters {
      self.base.apply_emitters_to_particles(
        cmd,
        particles,
        emitters,
        frames,
        particle_frame_ids,
        bvh,
        num_emitters,
      )?;
    }
    Ok(())
  }

  fn accumulate_bvh_forces_to_particles(
    &self,
    cmd: &mut VulkanCommandBuffer,
    particles: &mut VulkanBuffer<f32>,
    bvh: &VulkanBuffer<()>,
  ) -> EngineResult<()> {
    self.base.accumulate_bvh_forces_to_particles(cmd, particles, bvh)
  }

  fn apply_emitters_direct(
    &self,
    _cmd: &mut VulkanCommandBuffer,
    _particles: &mut VulkanBuffer<f32>,
    _emitters: &VulkanBuffer<ForceEmitter>,
    _frames: &VulkanBuffer<GpuReferenceFrame>,
    _particle_frame_ids: &VulkanBuffer<u32>,
    _num_emitters: u32,
  ) -> EngineResult<()> {
    Ok(()) // no-op in mock
  }

  fn bp_clear(
    &self,
    cmd: &mut VulkanCommandBuffer,
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_rb_lca_addr: u64,
    out_internal: u64,
    out_sparse: u64,
  ) -> EngineResult<()> {
    if self.target == MockTargetShader::BpClear {
      self.base.bp_clear(
        cmd,
        raw_pairs_addr,
        out_rb_rb_addr,
        out_rb_ps_addr,
        out_rb_lca_addr,
        out_internal,
        out_sparse,
      )?;
    }
    Ok(())
  }

  fn bp_bounds_gen(
    &self,
    cmd: &mut VulkanCommandBuffer,
    bodies: &VulkanBuffer<RigidBodyImex>,
    leaves_addr: u64,
    lca_entities_addr: u64,
    total_entities: u32,
    dt: timeus_t,
  ) -> EngineResult<()> {
    if self.target == MockTargetShader::BpBoundsGen {
      self.base.bp_bounds_gen(
        cmd,
        bodies,
        leaves_addr,
        lca_entities_addr,
        total_entities,
        dt,
      )?;
    }
    Ok(())
  }

  fn bp_scene(
    &self,
    cmd: &mut VulkanCommandBuffer,
    tlas_bvh_addr: u64,
    query_leaves_addr: u64,
    overlapping_pairs_addr: u64,
    tlas_root_index: u32,
    total_queries: u32,
  ) -> EngineResult<()> {
    if self.target == MockTargetShader::BpScene {
      self.base.bp_scene(
        cmd,
        tlas_bvh_addr,
        query_leaves_addr,
        overlapping_pairs_addr,
        tlas_root_index,
        total_queries,
      )?;
    }
    Ok(())
  }

  fn bp_classify(
    &self,
    cmd: &mut VulkanCommandBuffer,
    bodies: &VulkanBuffer<RigidBodyImex>,
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_ps_ps_addr: u64,
    out_macro_lca_addr: u64,
    out_lca_lca_addr: u64,
    total_raw_pairs: u32,
  ) -> EngineResult<()> {
    self.base.bp_classify(
      cmd,
      bodies,
      raw_pairs_addr,
      out_rb_rb_addr,
      out_rb_ps_addr,
      out_ps_ps_addr,
      out_macro_lca_addr,
      out_lca_lca_addr,
      total_raw_pairs,
    )?;
    Ok(())
  }

  fn bp_cross_lca(
    &self,
    _cmd: &mut VulkanCommandBuffer,
    _tlas_bvh_addr: u64,
    _lca_entities_addr: u64,
    _macro_leaves_addr: u64,
    _entity_headers_addr: u64,
    _lca_query_pairs_addr: u64,
    _out_rb_rb_addr: u64,
    _out_rb_ps_addr: u64,
    _out_ps_ps_addr: u64,
    _out_cross_pairs_addr: u64,
    _total_queries: u32,
    _max_pairs: u32,
    _num_rigid_bodies: u32,
  ) -> EngineResult<()> {
    self.base.bp_cross_lca(
      _cmd,
      _tlas_bvh_addr,
      _lca_entities_addr,
      _macro_leaves_addr,
      _entity_headers_addr,
      _lca_query_pairs_addr,
      _out_rb_rb_addr,
      _out_rb_ps_addr,
      _out_ps_ps_addr,
      _out_cross_pairs_addr,
      _total_queries,
      _max_pairs,
      _num_rigid_bodies,
    )?;
    Ok(())
  }

  fn bp_particle_self(
    &self,
    cmd: &mut VulkanCommandBuffer,
    bvh_addr: u64,
    particles: &mut VulkanBuffer<f32>,
    wrench_buffer_addr: u64,
    total_particles: u32,
    root_index: u32,
    particle_radius: f32,
    stiffness: f32,
  ) -> EngineResult<()> {
    if self.target == MockTargetShader::BpParticleSelf {
      self.base.bp_particle_self(
        cmd,
        bvh_addr,
        particles,
        wrench_buffer_addr,
        total_particles,
        root_index,
        particle_radius,
        stiffness,
      )?;
    }
    Ok(())
  }

  fn build_motion_bvh(
    &self,
    cmd: &mut VulkanCommandBuffer,
    kinematics: &VulkanBuffer<KinematicBody>,
    rigid_bodies: &VulkanBuffer<RigidBodyImex>,
    particles: &mut VulkanBuffer<f32>,
    particle_frame_ids: &mut VulkanBuffer<u32>,
    dt: timeus_t,
    entity_id: crate::scene::EntityId,
    particle_aabb: Option<([f32; 3], [f32; 3])>,
  ) -> EngineResult<VulkanBuffer<()>> {
    self.base.build_motion_bvh(
      cmd,
      kinematics,
      rigid_bodies,
      particles,
      particle_frame_ids,
      dt,
      entity_id,
      particle_aabb,
    )
  }

  fn self_intersect_scene(
    &self,
    cmd: &mut VulkanCommandBuffer,
    bvh: &VulkanBuffer<()>,
  ) -> EngineResult<VulkanBuffer<CollisionPair>> {
    self.base.self_intersect_scene(cmd, bvh)
  }

  fn intersect_instances(
    &self,
    cmd: &mut VulkanCommandBuffer,
    potentials: &VulkanBuffer<CollisionPair>,
    kinematics: &VulkanBuffer<KinematicBody>,
    rigid_bodies: &VulkanBuffer<RigidBodyImex>,
    particles: &VulkanBuffer<f32>,
  ) -> EngineResult<VulkanBuffer<CollisionPair>> {
    self
      .base
      .intersect_instances(cmd, potentials, kinematics, rigid_bodies, particles)
  }

  fn narrow_ccd(
    &self,
    cmd: &mut VulkanCommandBuffer,
    broadphase_pairs: &VulkanBuffer<CollisionPair>,
    rigid_bodies: &VulkanBuffer<RigidBodyImex>,
    particles: &VulkanBuffer<f32>,
    lca_entities: u64,
    space_type: u32,
    dt: f32,
    output_list: &VulkanBuffer<CollisionPair>,
  ) -> EngineResult<()> {
    if self.target == MockTargetShader::Ccd {
      self.base.narrow_ccd(
        cmd,
        broadphase_pairs,
        rigid_bodies,
        particles,
        lca_entities,
        space_type,
        dt,
        output_list,
      )
    } else {
      Ok(())
    }
  }

  fn narrow_ccd_cross_lca(
    &self,
    cmd: &mut VulkanCommandBuffer,
    broadphase_pairs: &VulkanBuffer<crate::gpu::CrossPair>,
    rigid_bodies: &VulkanBuffer<RigidBodyImex>,
    particles: &VulkanBuffer<f32>,
    lca_entities: u64,
    space_type: u32,
    dt: f32,
    output_list: &VulkanBuffer<CollisionPair>,
  ) -> EngineResult<()> {
    self.base.narrow_ccd_cross_lca(
      cmd,
      broadphase_pairs,
      rigid_bodies,
      particles,
      lca_entities,
      space_type,
      dt,
      output_list,
    )
  }

  fn compact_collisions(
    &self,
    cmd: &mut VulkanCommandBuffer,
    globals: &VulkanBuffer<CollisionPair>,
    time_delta: timeus_t,
  ) -> EngineResult<VulkanBuffer<CollisionPair>> {
    if self.target == MockTargetShader::StreamCompact {
      self.base.compact_collisions(cmd, globals, time_delta)
    } else {
      self.base.build_list(cmd, globals.capacity())
    }
  }

  fn find_earliest_collision(
    &self,
    cmd: &mut VulkanCommandBuffer,
    compacted: &VulkanBuffer<CollisionPair>,
    dt: f32,
  ) -> EngineResult<VulkanBuffer<u32>> {
    if self.target == MockTargetShader::ReduceToi {
      self.base.find_earliest_collision(cmd, compacted, dt)
    } else {
      // Need a valid 1-element buffer to return
      self.base.build_list(cmd, 1)
    }
  }

  fn apply_collision_responses(
    &self,
    cmd: &mut VulkanCommandBuffer,
    kinematics: &VulkanBuffer<KinematicBody>,
    rigid_bodies: &mut VulkanBuffer<RigidBodyImex>,
    particles: &mut VulkanBuffer<f32>,
    collisions: &VulkanBuffer<CollisionPair>,
    lca_entities_addr: u64,
    force_inelastic: bool,
  ) -> EngineResult<()> {
    if self.target == MockTargetShader::ApplyImpulses {
      self.base.apply_collision_responses(
        cmd,
        kinematics,
        rigid_bodies,
        particles,
        collisions,
        lca_entities_addr,
        force_inelastic,
      )?;
    }
    Ok(())
  }

  fn snapshot_dynamics(
    &self,
    cmd: &mut VulkanCommandBuffer,
    rigid_bodies: &VulkanBuffer<RigidBodyImex>,
    particles: Option<&VulkanBuffer<f32>>,
  ) -> EngineResult<(VulkanBuffer<RigidBodyImex>, Option<VulkanBuffer<f32>>)> {
    self.base.snapshot_dynamics(cmd, rigid_bodies, particles)
  }

  fn restore_dynamics(
    &self,
    cmd: &mut VulkanCommandBuffer,
    rigid_bodies: &mut VulkanBuffer<RigidBodyImex>,
    particles: Option<&mut VulkanBuffer<f32>>,
    snapshot: &(VulkanBuffer<RigidBodyImex>, Option<VulkanBuffer<f32>>),
  ) -> EngineResult<()> {
    self.base.restore_dynamics(cmd, rigid_bodies, particles, snapshot)
  }

  fn write_back_to_scene(
    &self,
    cmd: &mut VulkanCommandBuffer,
    rigid_bodies: &VulkanBuffer<RigidBodyImex>,
    particle_systems: &[(
      crate::scene::EntityId,
      VulkanBuffer<f32>,
      alloc::vec::Vec<ParticleMetadata>,
      bool,
    )],
    physical_scene: &mut PhysicsScene,
    scene: &Scene,
  ) -> EngineResult<Option<CommandBufferSyncInfo>> {
    self
      .base
      .write_back_to_scene(cmd, rigid_bodies, particle_systems, physical_scene, scene)
  }
}
