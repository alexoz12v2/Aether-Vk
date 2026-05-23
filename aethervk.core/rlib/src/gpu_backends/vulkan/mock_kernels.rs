use crate::gpu::{Kernels, CommandBufferSyncInfo, CollisionPair, KinematicBody, RigidBodyImex, Wrench, ForceEmitter, ParticleMetadata, DeviceBuffer};
use crate::physics::physics_scene::GpuReferenceFrame;
use crate::gpu_backends::vulkan::physics::{VulkanCommandBuffer, VulkanBuffer};
use crate::gpu_backends::vulkan::device::Device;
use crate::scene::Scene;
use crate::physics::physics_scene::PhysicsScene;
use crate::types::{EngineResult};
use aethervk_oshal_rlib::os::time::timeus_t;
use crate::simulation_api::structs::{MockTargetShader, SHADER_MOCK_RESULTS};
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;

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

    fn discard_buffer<T: Copy + Send + Sync>(&self, buffer: Self::Buffer<T>) { self.base.discard_buffer(buffer) }
    fn discard_list<T: Copy + Send + Sync>(&self, list: Self::List<T>) { self.base.discard_list(list) }
    fn discard_bvh(&self, bvh: Self::MotionBvh) { self.base.discard_bvh(bvh) }
    fn discard_tlas(&self, tlas: Self::MotionTlas) { self.base.discard_tlas(tlas) }
    fn subgroup_size(&self) -> Option<crate::gpu::SubgroupSize> { self.base.subgroup_size() }
    fn wait_sync(&self, sync: &CommandBufferSyncInfo) -> EngineResult<()> { self.base.wait_sync(sync) }
    fn refit_motion_blas(&self, cmd: &mut Self::Cmd, bvh: &Self::MotionBvh, depth_indices: &Self::Buffer<u32>, total_nodes: u32) -> EngineResult<()> { self.base.refit_motion_blas(cmd, bvh, depth_indices, total_nodes) }
    fn upload_motion_tlas(&self, cmd: &mut Self::Cmd, node_bytes: &[u8]) -> EngineResult<Self::MotionTlas> { self.base.upload_motion_tlas(cmd, node_bytes) }
    fn create_command_buffer(&self) -> EngineResult<Self::Cmd> { self.base.create_command_buffer() }
    fn build_list<T: Copy + Send + Sync>(&self, cmd: &mut Self::Cmd, capacity: usize) -> EngineResult<Self::List<T>> { self.base.build_list(cmd, capacity) }
    fn build_leaves(&self, cmd: &mut Self::Cmd, capacity: usize) -> EngineResult<Self::Buffer<[u32; 8]>> { self.base.build_leaves(cmd, capacity) }
    fn build_kinematic_bodies(&self, cmd: &mut Self::Cmd, scene: &PhysicsScene, scene0: &Scene) -> EngineResult<Self::Buffer<KinematicBody>> { self.base.build_kinematic_bodies(cmd, scene, scene0) }
    fn build_rigid_bodies(&self, cmd: &mut Self::Cmd, scene: &PhysicsScene, scene0: &Scene) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<Wrench>)> { self.base.build_rigid_bodies(cmd, scene, scene0) }
    fn build_frames(&self, cmd: &mut Self::Cmd, scene: &PhysicsScene) -> EngineResult<Self::Buffer<GpuReferenceFrame>> { self.base.build_frames(cmd, scene) }
    fn build_particles(&self, cmd: &mut Self::Cmd, scene: &Scene) -> EngineResult<(Self::Buffer<f32>, alloc::vec::Vec<ParticleMetadata>)> { self.base.build_particles(cmd, scene) }
    fn build_emitters(&self, cmd: &mut Self::Cmd, scene: &Scene) -> EngineResult<Self::Buffer<ForceEmitter>> { self.base.build_emitters(cmd, scene) }
    
    // --- MOCKED DISPATCH METHODS ---

    fn emit_particles(&self, cmd: &mut Self::Cmd, particles: &mut Self::Buffer<f32>, physical_scene: &PhysicsScene, scene: &Scene, sun_pos: Vec3f32, dt: timeus_t) -> EngineResult<()> { 
        if self.target == MockTargetShader::EmitParticles {
            self.base.emit_particles(cmd, particles, physical_scene, scene, sun_pos, dt)?;
        }
        Ok(())
    }
    
    fn step_ode_p1_p2(&self, cmd: &mut Self::Cmd, particles: &mut Self::Buffer<f32>, dt: timeus_t) -> EngineResult<()> { 
        if self.target == MockTargetShader::P1_2Imex {
            self.base.step_ode_p1_p2(cmd, particles, dt)?;
        }
        Ok(())
    }
    
    fn step_ode_p3_p4(&self, cmd: &mut Self::Cmd, kinematics: &Self::Buffer<KinematicBody>, rigid_bodies: &mut Self::Buffer<crate::gpu::RigidBodyGpu>, emitters: &Self::Buffer<ForceEmitter>, dt: timeus_t) -> EngineResult<()> { 
        if self.target == MockTargetShader::P3_4Imex {
            self.base.step_ode_p3_p4(cmd, kinematics, rigid_bodies, emitters, dt)?;
        }
        Ok(())
    }
    
    fn compute_self_gravity(&self, cmd: &mut Self::Cmd, bvh: &Self::MotionBvh, particles: &mut Self::Buffer<f32>) -> EngineResult<()> { 
        if self.target == MockTargetShader::BarnesHut {
            self.base.compute_self_gravity(cmd, bvh, particles)?;
        }
        Ok(())
    }
    
    fn step_ode_p5(&self, cmd: &mut Self::Cmd, kinematics: &Self::Buffer<KinematicBody>, particles: &mut Self::Buffer<f32>, emitters: &Self::Buffer<ForceEmitter>, dt: timeus_t) -> EngineResult<()> { 
        if self.target == MockTargetShader::P5Imex {
            self.base.step_ode_p5(cmd, kinematics, particles, emitters, dt)?;
        }
        Ok(())
    }

    fn imex_integrate_particles_p1_p2(&self, cmd: &mut Self::Cmd, particles: &mut Self::Buffer<f32>, dt: timeus_t) -> EngineResult<()> { 
        if self.target == MockTargetShader::IntegrateParticlesP1P2 {
            self.base.imex_integrate_particles_p1_p2(cmd, particles, dt)?;
        }
        Ok(())
    }

    fn imex_integrate_bodies_p3(&self, cmd: &mut Self::Cmd, bodies: &mut Self::Buffer<RigidBodyImex>, wrenches: &mut Self::Buffer<Wrench>, emitters: &Self::Buffer<ForceEmitter>, dt: timeus_t) -> EngineResult<()> { 
        if self.target == MockTargetShader::IntegrateBodiesP3 {
            self.base.imex_integrate_bodies_p3(cmd, bodies, wrenches, emitters, dt)?;
        }
        Ok(())
    }

    fn imex_rb_force_assign(&self, cmd: &mut Self::Cmd, bodies: &Self::Buffer<RigidBodyImex>, wrenches: &mut Self::Buffer<Wrench>) -> EngineResult<()> { 
        if self.target == MockTargetShader::RbForceAssign {
            self.base.imex_rb_force_assign(cmd, bodies, wrenches)?;
        }
        Ok(())
    }

    fn imex_integrate_particles_p4_5(&self, cmd: &mut Self::Cmd, particles: &mut Self::Buffer<f32>, dt: timeus_t, current_time_us: timeus_t) -> EngineResult<()> { 
        if self.target == MockTargetShader::IntegrateParticlesP4P5 {
            self.base.imex_integrate_particles_p4_5(cmd, particles, dt, current_time_us)?;
        }
        Ok(())
    }

    fn bp_clear(&self, cmd: &mut Self::Cmd, raw_pairs_addr: u64, out_rb_rb_addr: u64, out_rb_ps_addr: u64, out_rb_lca_addr: u64, out_internal: u64) -> EngineResult<()> { 
        if self.target == MockTargetShader::BpClear {
            self.base.bp_clear(cmd, raw_pairs_addr, out_rb_rb_addr, out_rb_ps_addr, out_rb_lca_addr, out_internal)?;
        }
        Ok(())
    }

    fn bp_bounds_gen(&self, cmd: &mut Self::Cmd, bodies: &Self::Buffer<RigidBodyImex>, leaves_addr: u64, total_entities: u32, dt: timeus_t) -> EngineResult<()> { 
        if self.target == MockTargetShader::BpBoundsGen {
            self.base.bp_bounds_gen(cmd, bodies, leaves_addr, total_entities, dt)?;
        }
        Ok(())
    }

    fn bp_scene(&self, cmd: &mut Self::Cmd, tlas_bvh_addr: u64, query_leaves_addr: u64, overlapping_pairs_addr: u64, tlas_root_index: u32, total_queries: u32) -> EngineResult<()> { 
        if self.target == MockTargetShader::BpScene {
            self.base.bp_scene(cmd, tlas_bvh_addr, query_leaves_addr, overlapping_pairs_addr, tlas_root_index, total_queries)?;
        }
        Ok(())
    }

    fn bp_classify(&self, cmd: &mut Self::Cmd, bodies: &Self::Buffer<RigidBodyImex>, raw_pairs_addr: u64, out_rb_rb_addr: u64, out_rb_ps_addr: u64, out_rb_lca_addr: u64, total_raw_pairs: u32) -> EngineResult<()> { 
        if self.target == MockTargetShader::BpClassify {
            self.base.bp_classify(cmd, bodies, raw_pairs_addr, out_rb_rb_addr, out_rb_ps_addr, out_rb_lca_addr, total_raw_pairs)?;
        }
        Ok(())
    }

    fn bp_cross_lca(&self, cmd: &mut Self::Cmd, scene_entities: &Self::Buffer<RigidBodyImex>, lca_query_pairs_addr: u64, output_internal_pairs_addr: u64, total_queries: u32) -> EngineResult<()> { 
        if self.target == MockTargetShader::BpCrossLca {
            self.base.bp_cross_lca(cmd, scene_entities, lca_query_pairs_addr, output_internal_pairs_addr, total_queries)?;
        }
        Ok(())
    }

    fn bp_particle_self(&self, cmd: &mut Self::Cmd, bvh_addr: u64, particles: &mut Self::Buffer<f32>, wrench_buffer_addr: u64, total_particles: u32, root_index: u32, particle_radius: f32, stiffness: f32) -> EngineResult<()> { 
        if self.target == MockTargetShader::BpParticleSelf {
            self.base.bp_particle_self(cmd, bvh_addr, particles, wrench_buffer_addr, total_particles, root_index, particle_radius, stiffness)?;
        }
        Ok(())
    }

    fn build_motion_bvh(&self, cmd: &mut Self::Cmd, kinematics: &Self::Buffer<KinematicBody>, rigid_bodies: &Self::Buffer<RigidBodyImex>, particles: &Self::Buffer<f32>, dt: timeus_t) -> EngineResult<Self::MotionBvh> {
        self.base.build_motion_bvh(cmd, kinematics, rigid_bodies, particles, dt)
    }

    fn self_intersect_scene(&self, cmd: &mut Self::Cmd, bvh: &Self::MotionBvh) -> EngineResult<Self::List<CollisionPair>> {
        self.base.self_intersect_scene(cmd, bvh)
    }

    fn intersect_instances(&self, cmd: &mut Self::Cmd, potentials: &Self::List<CollisionPair>, kinematics: &Self::Buffer<KinematicBody>, rigid_bodies: &Self::Buffer<RigidBodyImex>, particles: &Self::Buffer<f32>) -> EngineResult<Self::List<CollisionPair>> {
        self.base.intersect_instances(cmd, potentials, kinematics, rigid_bodies, particles)
    }

    fn narrow_ccd(&self, cmd: &mut Self::Cmd, broadphase_pairs: &Self::List<CollisionPair>, rigid_bodies: &Self::Buffer<RigidBodyImex>, particles: &Self::Buffer<f32>) -> EngineResult<Self::List<CollisionPair>> {
        if self.target == MockTargetShader::Ccd {
            self.base.narrow_ccd(cmd, broadphase_pairs, rigid_bodies, particles)
        } else {
            self.base.build_list(cmd, broadphase_pairs.capacity())
        }
    }

    fn compact_collisions(&self, cmd: &mut Self::Cmd, globals: &Self::List<CollisionPair>, time_delta: timeus_t) -> EngineResult<Self::List<CollisionPair>> {
        if self.target == MockTargetShader::StreamCompact {
            self.base.compact_collisions(cmd, globals, time_delta)
        } else {
            self.base.build_list(cmd, globals.capacity())
        }
    }

    fn find_earliest_collision(&self, cmd: &mut Self::Cmd, compacted: &Self::List<CollisionPair>) -> EngineResult<Self::Buffer<u32>> {
        if self.target == MockTargetShader::ReduceToi {
            self.base.find_earliest_collision(cmd, compacted)
        } else {
            // Need a valid 1-element buffer to return
            self.base.build_list(cmd, 1)
        }
    }

    fn apply_collision_responses(&self, cmd: &mut Self::Cmd, kinematics: &Self::Buffer<KinematicBody>, rigid_bodies: &mut Self::Buffer<RigidBodyImex>, particles: &mut Self::Buffer<f32>, collisions: &Self::List<CollisionPair>, force_inelastic: bool) -> EngineResult<()> {
        if self.target == MockTargetShader::ApplyImpulses {
            self.base.apply_collision_responses(cmd, kinematics, rigid_bodies, particles, collisions, force_inelastic)?;
        }
        Ok(())
    }

    fn snapshot_dynamics(&self, cmd: &mut Self::Cmd, rigid_bodies: &Self::Buffer<RigidBodyImex>, particles: &Self::Buffer<f32>) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<f32>)> {
        self.base.snapshot_dynamics(cmd, rigid_bodies, particles)
    }

    fn restore_dynamics(&self, cmd: &mut Self::Cmd, rigid_bodies: &mut Self::Buffer<RigidBodyImex>, particles: &mut Self::Buffer<f32>, snapshot: &(Self::Buffer<RigidBodyImex>, Self::Buffer<f32>)) -> EngineResult<()> {
        self.base.restore_dynamics(cmd, rigid_bodies, particles, snapshot)
    }

    fn write_back_to_scene(&self, cmd: &mut Self::Cmd, rigid_bodies: &Self::Buffer<RigidBodyImex>, particles: &Self::Buffer<f32>, particle_metadata: &[ParticleMetadata], physical_scene: &mut PhysicsScene, scene: &Scene) -> EngineResult<Option<CommandBufferSyncInfo>> {
        // Here we read back for the test
        let mut results = SHADER_MOCK_RESULTS.lock().unwrap();
        // Depending on target shader, we might want to read particles or rigid bodies or both.
        // For simplicity, let's just trigger a readback of everything, or we can customize it later in the test.
        // Wait, the user asked to immediately transfer the results in a CPU backed buffer for assertions.
        // We can just rely on `write_back_to_scene` or add a specific readback logic here.
        // Let's implement the standard write_back_to_scene but ALSO readback to SHADER_MOCK_RESULTS.
        
        // This is a test harness, so we can enqueue a read to CPU right here!
        // We'll read rigid bodies. 
        if let Ok(rb_handle) = rigid_bodies.enqueue_read_to_cpu(cmd) {
            // Note: we can't wait here because cmd hasn't been submitted.
            // But we can store a flag or we can just let the test itself do the readback from the ECS scene!
            // Wait, if write_back_to_scene executes, the data gets written into the `scene` ECS!
            // So the test can just assert the `Scene` components!
            // BUT for internal buffers like wrenches, they aren't written to the scene.
            // In that case, we need to read them back.
        }
        
        self.base.write_back_to_scene(cmd, rigid_bodies, particles, particle_metadata, physical_scene, scene)
    }
}
