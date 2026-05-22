//! cpu_kernels module.

use crate::{
  gpu::{
    CollisionPair, CommandBuffer, DeviceBuffer, DeviceBvh, DeviceList, ForceEmitter, Kernels,
    KinematicBody, ParticleGpu, RigidBodyGpu, RigidBodyImex, Wrench, ParticleMetadata, WaitHandle,
  },
  physics::physics_scene::{PhysicsScene, GpuReferenceFrame},
  scene::{KinematicComponent, Scene, TransformComponent},
  simulation_api::structs::SendPtr,
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib::{
  math::{
    matrix::{Matrix, Matrix4, MatrixVectorMul, SquareMatrix, mat3::Mat3f32, mat4::Mat4x4f32},
    quaternion::Quaternion,
    vector::{
      Vector, Vector3, Vector4,
      vec3::Vec3f32,
      vec4::{Quat, Vec4f32},
    },
  },
  os::time::timeus_t,
};
use alloc::{boxed::Box, vec::Vec};

/// TODO: Document this item
pub struct CpuCommandBuffer {
  tasks: Vec<Box<dyn FnOnce() + Send + Sync>>,
}

impl CommandBuffer for CpuCommandBuffer {
  fn submit(&mut self) -> EngineResult<Option<crate::gpu::CommandBufferSyncInfo>> {
    for task in self.tasks.drain(..) {
      task();
    }
    Ok(None)
  }
}

/// TODO: Document this item
pub struct CpuWaitHandle<T> {
  data: Option<T>,
}

impl<T: Send + Sync> WaitHandle<T> for CpuWaitHandle<T> {
  fn wait(mut self) -> EngineResult<T> {
    self.data.take().ok_or(EngineError::InvalidOperation("WaitHandle already consumed"))
  }
}

/// TODO: Document this item
pub struct CpuBuffer<T> {
  pub data: Vec<T>,
}

impl<T: Copy + Send + Sync> DeviceBuffer<T> for CpuBuffer<T> {
  type Cmd = CpuCommandBuffer;
  type ReadHandle<'a>
    = CpuWaitHandle<Vec<T>>
  where
    Self: 'a,
    T: 'a;

  fn capacity(&self) -> usize {
    self.data.capacity()
  }

  fn address(&self) -> u64 { 0 }
  fn enqueue_read_to_cpu(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'_>>
  {
    Ok(CpuWaitHandle {
      data: Some(self.data.clone()),
    })
  }
}

/// TODO: Document this item
#[derive(Clone)]
pub struct CpuList<T> {
  pub data: Vec<T>,
}

impl<T: Copy + Send + Sync> DeviceBuffer<T> for CpuList<T> {
  type Cmd = CpuCommandBuffer;
  type ReadHandle<'a>
    = CpuWaitHandle<Vec<T>>
  where
    Self: 'a,
    T: 'a;

  fn capacity(&self) -> usize {
    self.data.capacity()
  }

  fn address(&self) -> u64 { 0 }
  fn enqueue_read_to_cpu(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'_>>
  {
    Ok(CpuWaitHandle {
      data: Some(self.data.clone()),
    })
  }
}

impl<T: Copy + Send + Sync> DeviceList<T> for CpuList<T> {
  fn clear(&mut self, _cmd: &mut Self::Cmd) -> EngineResult<()> {
    self.data.clear();
    Ok(())
  }
}

/// TODO: Document this item
pub struct CpuMotionBvh {
  pub kinematics_copy: Vec<KinematicBody>,
  pub rigid_bodies_copy: Vec<RigidBodyImex>,
  pub particles_copy: Vec<f32>,
  pub particle_metadata_copy: Vec<crate::gpu::ParticleMetadata>,
  pub bvh_tree: crate::physics::motion_bvh::MotionBvhTree,
}

impl DeviceBvh for CpuMotionBvh {
  type Cmd = CpuCommandBuffer;
  fn address(&self) -> u64 { 0 }
}

/// Zero-cost CPU stand-in for the per-tick TLAS GPU buffer.
/// `address()` returns 0 — broadphase shaders are no-ops in the CPU kernel path.
pub struct CpuTlasHandle;

impl DeviceBvh for CpuTlasHandle {
  type Cmd = CpuCommandBuffer;
  fn address(&self) -> u64 { 0 }
}

/// TODO: Document this item
pub struct CpuScalarKernels {}

impl Kernels for CpuScalarKernels {
  type Cmd = CpuCommandBuffer;
  type Buffer<T: Copy + Send + Sync> = CpuBuffer<T>;
  type List<T: Copy + Send + Sync> = CpuList<T>;
  type MotionBvh = CpuMotionBvh;
  /// CPU path: TLAS is a no-op; broadphase shaders are not dispatched.
  type MotionTlas = CpuTlasHandle;

  fn discard_buffer<T: Copy + Send + Sync>(&self, _buffer: Self::Buffer<T>) {}
  fn discard_list<T: Copy + Send + Sync>(&self, _list: Self::List<T>) {}
  fn discard_bvh(&self, _bvh: Self::MotionBvh) {}
  fn discard_tlas(&self, _tlas: Self::MotionTlas) {}

  fn subgroup_size(&self) -> Option<crate::gpu::SubgroupSize> { Some(crate::gpu::SubgroupSize::Size32) }
  fn wait_sync(&self, _sync: &crate::gpu::CommandBufferSyncInfo) -> EngineResult<()> { Ok(()) }
  fn refit_motion_blas(&self, _cmd: &mut Self::Cmd, _bvh: &Self::MotionBvh, _depth_indices: &Self::Buffer<u32>, _total_nodes: u32) -> EngineResult<()> { Ok(()) }

  fn upload_motion_tlas(
    &self,
    _cmd: &mut Self::Cmd,
    _node_bytes: &[u8],
  ) -> EngineResult<Self::MotionTlas> {
    Ok(CpuTlasHandle)
  }


  fn create_command_buffer(&self) -> EngineResult<Self::Cmd> {
    Ok(CpuCommandBuffer { tasks: alloc::vec::Vec::new() })
  }

  fn build_list<T: Copy + Send + Sync>(
    &self,
    cmd: &mut Self::Cmd,
    capacity: usize,
  ) -> EngineResult<Self::List<T>> {
    Ok(CpuList { data: alloc::vec::Vec::new() })
  }

  fn build_leaves(
    &self,
    _cmd: &mut Self::Cmd,
    _capacity: usize,
  ) -> EngineResult<Self::Buffer<[u32; 8]>> {
    Ok(CpuBuffer { data: alloc::vec::Vec::new() })
  }

  fn build_kinematic_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<KinematicBody>> {
    Ok(CpuBuffer { data: alloc::vec::Vec::new() })
  }

  fn build_rigid_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<Wrench>)> {
    Ok((CpuBuffer { data: alloc::vec::Vec::new() }, CpuBuffer { data: alloc::vec::Vec::new() }))
  }

  fn build_frames(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
  ) -> EngineResult<Self::Buffer<GpuReferenceFrame>> {
    Ok(CpuBuffer { data: alloc::vec::Vec::new() })
  }

  fn build_particles(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<(Self::Buffer<f32>, alloc::vec::Vec<ParticleMetadata>)> {
    Ok((CpuBuffer { data: alloc::vec::Vec::new() }, alloc::vec::Vec::new()))
  }

  fn build_emitters(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<Self::Buffer<ForceEmitter>> {
    Ok(CpuBuffer { data: alloc::vec::Vec::new() })
  }

  fn emit_particles(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    physical_scene: &PhysicsScene,
    scene: &Scene,
    sun_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn step_ode_p1_p2(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn step_ode_p3_p4(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
    emitters: &Self::Buffer<ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn compute_self_gravity(
    &self,
    cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
    particles: &mut Self::Buffer<f32>,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn step_ode_p5(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    particles: &mut Self::Buffer<f32>,
    emitters: &Self::Buffer<ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn imex_integrate_particles_p1_p2(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn imex_integrate_bodies_p3(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &mut Self::Buffer<RigidBodyImex>,
    wrenches: &mut Self::Buffer<Wrench>,
    emitters: &Self::Buffer<ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn imex_rb_force_assign(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &Self::Buffer<RigidBodyImex>,
    wrenches: &mut Self::Buffer<Wrench>,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn imex_integrate_particles_p4_5(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    dt: timeus_t,
    current_time_us: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn bp_clear(
    &self,
    cmd: &mut Self::Cmd,
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_rb_lca_addr: u64,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn bp_bounds_gen(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &Self::Buffer<RigidBodyImex>,
    leaves_addr: u64,
    total_entities: u32,
    dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn bp_scene(
    &self,
    cmd: &mut Self::Cmd,
    tlas_bvh_addr: u64,
    query_leaves_addr: u64,
    overlapping_pairs_addr: u64,
    tlas_root_index: u32,
    total_queries: u32,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn bp_classify(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &Self::Buffer<RigidBodyImex>,
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_rb_lca_addr: u64,
    total_raw_pairs: u32,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn bp_cross_lca(
    &self,
    cmd: &mut Self::Cmd,
    frames: &Self::Buffer<GpuReferenceFrame>,
    lca_query_pairs_addr: u64,
    output_internal_pairs_addr: u64,
    total_queries: u32,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn bp_particle_self(
    &self,
    cmd: &mut Self::Cmd,
    bvh_addr: u64,
    particles: &mut Self::Buffer<f32>,
    wrench_buffer_addr: u64,
    total_particles: u32,
    root_index: u32,
    particle_radius: f32,
    stiffness: f32,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn build_motion_bvh(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
    dt: timeus_t,
  ) -> EngineResult<Self::MotionBvh> {
    Ok(CpuMotionBvh { kinematics_copy: alloc::vec::Vec::new(), rigid_bodies_copy: alloc::vec::Vec::new(), particles_copy: alloc::vec::Vec::new(), particle_metadata_copy: alloc::vec::Vec::new(), bvh_tree: crate::physics::motion_bvh::MotionBvhTree { nodes: alloc::vec::Vec::new(), root: None } })
  }

  fn self_intersect_scene(
    &self,
    cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList { data: alloc::vec::Vec::new() })
  }

  fn intersect_instances(
    &self,
    cmd: &mut Self::Cmd,
    potentials: &Self::List<CollisionPair>,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList { data: alloc::vec::Vec::new() })
  }

  fn narrow_ccd(
    &self,
    _cmd: &mut Self::Cmd,
    broadphase_pairs: &Self::List<CollisionPair>,
    _rigid_bodies: &Self::Buffer<RigidBodyImex>,
    _particles: &Self::Buffer<f32>,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList { data: alloc::vec::Vec::new() })
  }

  fn compact_collisions(
    &self,
    cmd: &mut Self::Cmd,
    globals: &Self::List<CollisionPair>,
    time_delta: timeus_t,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList { data: alloc::vec::Vec::new() })
  }

  fn find_earliest_collision(
    &self,
    cmd: &mut Self::Cmd,
    compacted: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::Buffer<u32>> {
    Ok(CpuBuffer { data: alloc::vec::Vec::new() })
  }

  fn apply_collision_responses(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &mut Self::Buffer<RigidBodyImex>,
    particles: &mut Self::Buffer<f32>,
    collisions: &Self::List<CollisionPair>,
    force_inelastic: bool,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn snapshot_dynamics(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
  ) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<f32>)> {
    Ok((CpuBuffer { data: alloc::vec::Vec::new() }, CpuBuffer { data: alloc::vec::Vec::new() }))
  }

  fn restore_dynamics(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &mut Self::Buffer<RigidBodyImex>,
    particles: &mut Self::Buffer<f32>,
    snapshot: &(Self::Buffer<RigidBodyImex>, Self::Buffer<f32>),
  ) -> EngineResult<()> {
    Ok(())
  }

  fn write_back_to_scene(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
    particle_metadata: &[ParticleMetadata],
    physical_scene: &mut PhysicsScene,
    scene: &Scene,
  ) -> EngineResult<Option<crate::gpu::CommandBufferSyncInfo>> {
    Ok(None)
  }
}

/// TODO: Document this item
pub struct CpuSimdKernels {
  pub thread_pool: alloc::sync::Arc<aethervk_oshal_rlib::os::pool::ThreadPool>,
}

impl Kernels for CpuSimdKernels {
  type Cmd = CpuCommandBuffer;
  type Buffer<T: Copy + Send + Sync> = CpuBuffer<T>;
  type List<T: Copy + Send + Sync> = CpuList<T>;
  type MotionBvh = CpuMotionBvh;
  /// CPU path: TLAS is a no-op; broadphase shaders are not dispatched.
  type MotionTlas = CpuTlasHandle;

  fn discard_buffer<T: Copy + Send + Sync>(&self, _buffer: Self::Buffer<T>) {}
  fn discard_list<T: Copy + Send + Sync>(&self, _list: Self::List<T>) {}
  fn discard_bvh(&self, _bvh: Self::MotionBvh) {}
  fn discard_tlas(&self, _tlas: Self::MotionTlas) {}

  fn subgroup_size(&self) -> Option<crate::gpu::SubgroupSize> { Some(crate::gpu::SubgroupSize::Size32) }
  fn wait_sync(&self, _sync: &crate::gpu::CommandBufferSyncInfo) -> EngineResult<()> { Ok(()) }
  fn refit_motion_blas(&self, _cmd: &mut Self::Cmd, _bvh: &Self::MotionBvh, _depth_indices: &Self::Buffer<u32>, _total_nodes: u32) -> EngineResult<()> { Ok(()) }

  fn upload_motion_tlas(
    &self,
    _cmd: &mut Self::Cmd,
    _node_bytes: &[u8],
  ) -> EngineResult<Self::MotionTlas> {
    Ok(CpuTlasHandle)
  }


  fn create_command_buffer(&self) -> EngineResult<Self::Cmd> {
    Ok(CpuCommandBuffer { tasks: alloc::vec::Vec::new() })
  }

  fn build_list<T: Copy + Send + Sync>(
    &self,
    cmd: &mut Self::Cmd,
    capacity: usize,
  ) -> EngineResult<Self::List<T>> {
    Ok(CpuList { data: alloc::vec::Vec::new() })
  }

  fn build_leaves(
    &self,
    _cmd: &mut Self::Cmd,
    _capacity: usize,
  ) -> EngineResult<Self::Buffer<[u32; 8]>> {
    Ok(CpuBuffer { data: alloc::vec::Vec::new() })
  }

  fn build_kinematic_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<KinematicBody>> {
    Ok(CpuBuffer { data: alloc::vec::Vec::new() })
  }

  fn build_rigid_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<Wrench>)> {
    Ok((CpuBuffer { data: alloc::vec::Vec::new() }, CpuBuffer { data: alloc::vec::Vec::new() }))
  }

  fn build_frames(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
  ) -> EngineResult<Self::Buffer<GpuReferenceFrame>> {
    Ok(CpuBuffer { data: alloc::vec::Vec::new() })
  }

  fn build_particles(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<(Self::Buffer<f32>, alloc::vec::Vec<ParticleMetadata>)> {
    Ok((CpuBuffer { data: alloc::vec::Vec::new() }, alloc::vec::Vec::new()))
  }

  fn build_emitters(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<Self::Buffer<ForceEmitter>> {
    Ok(CpuBuffer { data: alloc::vec::Vec::new() })
  }

  fn emit_particles(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    physical_scene: &PhysicsScene,
    scene: &Scene,
    sun_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn step_ode_p1_p2(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn step_ode_p3_p4(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
    emitters: &Self::Buffer<ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn compute_self_gravity(
    &self,
    cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
    particles: &mut Self::Buffer<f32>,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn step_ode_p5(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    particles: &mut Self::Buffer<f32>,
    emitters: &Self::Buffer<ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn imex_integrate_particles_p1_p2(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn imex_integrate_bodies_p3(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &mut Self::Buffer<RigidBodyImex>,
    wrenches: &mut Self::Buffer<Wrench>,
    emitters: &Self::Buffer<ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn imex_rb_force_assign(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &Self::Buffer<RigidBodyImex>,
    wrenches: &mut Self::Buffer<Wrench>,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn imex_integrate_particles_p4_5(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    dt: timeus_t,
    current_time_us: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn bp_clear(
    &self,
    cmd: &mut Self::Cmd,
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_rb_lca_addr: u64,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn bp_bounds_gen(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &Self::Buffer<RigidBodyImex>,
    leaves_addr: u64,
    total_entities: u32,
    dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn bp_scene(
    &self,
    cmd: &mut Self::Cmd,
    tlas_bvh_addr: u64,
    query_leaves_addr: u64,
    overlapping_pairs_addr: u64,
    tlas_root_index: u32,
    total_queries: u32,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn bp_classify(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &Self::Buffer<RigidBodyImex>,
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_rb_lca_addr: u64,
    total_raw_pairs: u32,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn bp_cross_lca(
    &self,
    cmd: &mut Self::Cmd,
    frames: &Self::Buffer<GpuReferenceFrame>,
    lca_query_pairs_addr: u64,
    output_internal_pairs_addr: u64,
    total_queries: u32,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn bp_particle_self(
    &self,
    cmd: &mut Self::Cmd,
    bvh_addr: u64,
    particles: &mut Self::Buffer<f32>,
    wrench_buffer_addr: u64,
    total_particles: u32,
    root_index: u32,
    particle_radius: f32,
    stiffness: f32,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn build_motion_bvh(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
    dt: timeus_t,
  ) -> EngineResult<Self::MotionBvh> {
    Ok(CpuMotionBvh { kinematics_copy: alloc::vec::Vec::new(), rigid_bodies_copy: alloc::vec::Vec::new(), particles_copy: alloc::vec::Vec::new(), particle_metadata_copy: alloc::vec::Vec::new(), bvh_tree: crate::physics::motion_bvh::MotionBvhTree { nodes: alloc::vec::Vec::new(), root: None } })
  }

  fn self_intersect_scene(
    &self,
    cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList { data: alloc::vec::Vec::new() })
  }

  fn intersect_instances(
    &self,
    cmd: &mut Self::Cmd,
    potentials: &Self::List<CollisionPair>,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList { data: alloc::vec::Vec::new() })
  }

  fn narrow_ccd(
    &self,
    _cmd: &mut Self::Cmd,
    broadphase_pairs: &Self::List<CollisionPair>,
    _rigid_bodies: &Self::Buffer<RigidBodyImex>,
    _particles: &Self::Buffer<f32>,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList { data: alloc::vec::Vec::new() })
  }

  fn compact_collisions(
    &self,
    cmd: &mut Self::Cmd,
    globals: &Self::List<CollisionPair>,
    time_delta: timeus_t,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList { data: alloc::vec::Vec::new() })
  }

  fn find_earliest_collision(
    &self,
    cmd: &mut Self::Cmd,
    compacted: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::Buffer<u32>> {
    Ok(CpuBuffer { data: alloc::vec::Vec::new() })
  }

  fn apply_collision_responses(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &mut Self::Buffer<RigidBodyImex>,
    particles: &mut Self::Buffer<f32>,
    collisions: &Self::List<CollisionPair>,
    force_inelastic: bool,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn snapshot_dynamics(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
  ) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<f32>)> {
    Ok((CpuBuffer { data: alloc::vec::Vec::new() }, CpuBuffer { data: alloc::vec::Vec::new() }))
  }

  fn restore_dynamics(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &mut Self::Buffer<RigidBodyImex>,
    particles: &mut Self::Buffer<f32>,
    snapshot: &(Self::Buffer<RigidBodyImex>, Self::Buffer<f32>),
  ) -> EngineResult<()> {
    Ok(())
  }

  fn write_back_to_scene(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
    particle_metadata: &[ParticleMetadata],
    physical_scene: &mut PhysicsScene,
    scene: &Scene,
  ) -> EngineResult<Option<crate::gpu::CommandBufferSyncInfo>> {
    Ok(None)
  }
}

pub fn group_and_cluster_collisions(
  mut collisions: alloc::vec::Vec<CollisionPair>,
  time_tolerance: f32,
) -> alloc::vec::Vec<alloc::vec::Vec<CollisionPair>> {
  collisions.sort_by(|a, b| {
    a.time_of_impact.partial_cmp(&b.time_of_impact).unwrap_or(core::cmp::Ordering::Equal)
  });

  let mut collided_entities: hashbrown::HashSet<u32> = hashbrown::HashSet::new();
  let mut resolved_clusters = alloc::vec::Vec::new();

  let mut current_group = alloc::vec::Vec::<CollisionPair>::new();
  let mut current_time = -1.0;

  for col in collisions {
    if collided_entities.contains(&col.a.primitive_index)
      || collided_entities.contains(&col.b.primitive_index)
    {
      continue;
    }

    if current_group.is_empty() {
      current_group.push(col.clone());
      current_time = col.time_of_impact;
      continue;
    }

    if (col.time_of_impact - current_time).abs() <= time_tolerance {
      current_group.push(col);
    } else {
      let mut clusters = form_clusters(current_group.as_ref());
      resolved_clusters.append(&mut clusters);

      for c in &current_group {
        collided_entities.insert(c.a.primitive_index);
        collided_entities.insert(c.b.primitive_index);
      }

      current_group.clear();

      if collided_entities.contains(&col.a.primitive_index)
        || collided_entities.contains(&col.b.primitive_index)
      {
        continue;
      }

      current_time = col.time_of_impact;
      current_group.push(col);
    }
  }

  if !current_group.is_empty() {
    let mut clusters = form_clusters(current_group.as_ref());
    resolved_clusters.append(&mut clusters);
  }

  resolved_clusters
}

fn form_clusters(group: &[CollisionPair]) -> alloc::vec::Vec<alloc::vec::Vec<CollisionPair>> {
  let mut adj_list: hashbrown::HashMap<u32, alloc::vec::Vec<usize>> = hashbrown::HashMap::new();
  for (i, col) in group.iter().enumerate() {
    adj_list.entry(col.a.primitive_index).or_default().push(i);
    adj_list.entry(col.b.primitive_index).or_default().push(i);
  }

  let mut visited_collisions = alloc::vec![false; group.len()];
  let mut clusters = alloc::vec::Vec::new();

  for i in 0..group.len() {
    if !visited_collisions[i] {
      let mut cluster = alloc::vec::Vec::new();
      let mut stack = alloc::vec![i];
      visited_collisions[i] = true;

      while let Some(idx) = stack.pop() {
        let col = &group[idx];
        cluster.push(col.clone());

        if let Some(neighbors) = adj_list.get(&col.a.primitive_index) {
          for &n_idx in neighbors {
            if !visited_collisions[n_idx] {
              visited_collisions[n_idx] = true;
              stack.push(n_idx);
            }
          }
        }

        if let Some(neighbors) = adj_list.get(&col.b.primitive_index) {
          for &n_idx in neighbors {
            if !visited_collisions[n_idx] {
              visited_collisions[n_idx] = true;
              stack.push(n_idx);
            }
          }
        }
      }
      clusters.push(cluster);
    }
  }

  clusters
}

#[cfg(test)]
mod tests {
  use super::*;
  use alloc::vec;

  #[test]
  fn test_group_and_cluster_collisions() {
    let c1 = CollisionPair {
      a: crate::gpu::ColliderId {
        entity_id: 0,
        primitive_index: 0,
      },
      b: crate::gpu::ColliderId {
        entity_id: 0,
        primitive_index: 1,
      },
      time_of_impact: 0.05,
      contact_normal: [0.0; 3],
      contact_point: [0.0; 3],
      penetration_depth: 0.0,
    };
    let c2 = CollisionPair {
      a: crate::gpu::ColliderId {
        entity_id: 0,
        primitive_index: 1,
      },
      b: crate::gpu::ColliderId {
        entity_id: 0,
        primitive_index: 2,
      },
      time_of_impact: 0.051,
      contact_normal: [0.0; 3],
      contact_point: [0.0; 3],
      penetration_depth: 0.0,
    };
    let c3 = CollisionPair {
      a: crate::gpu::ColliderId {
        entity_id: 0,
        primitive_index: 3,
      },
      b: crate::gpu::ColliderId {
        entity_id: 0,
        primitive_index: 4,
      },
      time_of_impact: 0.052,
      contact_normal: [0.0; 3],
      contact_point: [0.0; 3],
      penetration_depth: 0.0,
    };
    let c4 = CollisionPair {
      a: crate::gpu::ColliderId {
        entity_id: 0,
        primitive_index: 5,
      },
      b: crate::gpu::ColliderId {
        entity_id: 0,
        primitive_index: 6,
      },
      time_of_impact: 0.1,
      contact_normal: [0.0; 3],
      contact_point: [0.0; 3],
      penetration_depth: 0.0,
    };

    let pairs = vec![c1, c2, c3, c4];
    let clusters = group_and_cluster_collisions(pairs, 0.01);

    assert_eq!(clusters.len(), 3);

    // The first cluster should have c1 and c2
    assert_eq!(clusters[0].len(), 2);
    // The second cluster should have c3
    assert_eq!(clusters[1].len(), 1);
    // The third cluster should have c4
    assert_eq!(clusters[2].len(), 1);
  }
}