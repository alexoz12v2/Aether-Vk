extern crate alloc;
use alloc::vec::Vec;

use aethervk_core_rlib::{
  gpu::{
    CollisionPair, CommandBuffer, CommandBufferSyncInfo, DeviceBuffer, DeviceBvh, DeviceList,
    ForceEmitter, Kernels, KinematicBody, ParticleMetadata, RigidBodyGpu, RigidBodyImex,
    SubgroupSize, WaitHandle, Wrench,
  },
  physics::physics_scene::{GpuReferenceFrame, PhysicsScene},
  scene::Scene,
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib::os::time::timeus_t;

pub struct CpuCommandBuffer {}
impl CommandBuffer for CpuCommandBuffer {
  fn submit(&mut self) -> EngineResult<Option<CommandBufferSyncInfo>> {
    Ok(None)
  }
}

pub struct CpuDeviceBuffer<T> {
  data: core::marker::PhantomData<T>,
}
impl<T: Copy + Send + Sync> DeviceBuffer<T> for CpuDeviceBuffer<T> {
  type Cmd = CpuCommandBuffer;
  type ReadHandle<'a>
    = CpuWaitHandle<'a, Vec<T>>
  where
    Self: 'a,
    T: 'a;
  fn capacity(&self) -> usize {
    0
  }
  fn enqueue_read_to_cpu(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'_>> {
    todo!()
  }
  fn address(&self) -> u64 {
    todo!()
  }
}

pub struct CpuDeviceList<T> {
  data: core::marker::PhantomData<T>,
}
impl<T: Copy + Send + Sync> DeviceBuffer<T> for CpuDeviceList<T> {
  type Cmd = CpuCommandBuffer;
  type ReadHandle<'a>
    = CpuWaitHandle<'a, Vec<T>>
  where
    Self: 'a,
    T: 'a;
  fn capacity(&self) -> usize {
    0
  }
  fn enqueue_read_to_cpu(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'_>> {
    todo!()
  }
  fn address(&self) -> u64 {
    todo!()
  }
}
impl<T: Copy + Send + Sync> DeviceList<T> for CpuDeviceList<T> {
  fn clear(&mut self, _cmd: &mut Self::Cmd) -> EngineResult<()> {
    Ok(())
  }
}

pub struct CpuWaitHandle<'a, T> {
  _phantom: core::marker::PhantomData<&'a T>,
}
impl<'a, T: Send + Sync> WaitHandle<T> for CpuWaitHandle<'a, T> {
  fn wait(self) -> EngineResult<T> {
    todo!()
  }
}

pub struct CpuMotionBvh {}
impl DeviceBvh for CpuMotionBvh {
  type Cmd = CpuCommandBuffer;
  fn address(&self) -> u64 {
    todo!()
  }
}

pub struct CpuKernels {}
impl CpuKernels {
  pub fn new() -> Self {
    Self {}
  }
}

impl Kernels for CpuKernels {
  type Cmd = CpuCommandBuffer;
  type Buffer<T: Copy + Send + Sync> = CpuDeviceBuffer<T>;
  type List<T: Copy + Send + Sync> = CpuDeviceList<T>;
  type MotionBvh = CpuMotionBvh;
  type MotionTlas = CpuMotionBvh;

  fn discard_buffer<T: Copy + Send + Sync>(&self, _buffer: Self::Buffer<T>) {}
  fn discard_list<T: Copy + Send + Sync>(&self, _list: Self::List<T>) {}
  fn discard_bvh(&self, _bvh: Self::MotionBvh) {}
  fn discard_tlas(&self, _: <Self as Kernels>::MotionTlas) {}

  fn subgroup_size(&self) -> std::option::Option<SubgroupSize> {
    todo!()
  }
  fn wait_sync(&self, _: &CommandBufferSyncInfo) -> Result<(), EngineError> {
    todo!()
  }
  fn refit_motion_blas(
    &self,
    _: &mut Self::Cmd,
    _: &Self::MotionBvh,
    _: &Self::Buffer<u32>,
    _: u32,
  ) -> Result<(), EngineError> {
    todo!()
  }
  fn upload_motion_tlas(
    &self,
    _: &mut Self::Cmd,
    _: &[u8],
  ) -> Result<Self::MotionTlas, EngineError> {
    todo!()
  }
  fn build_list<T: Copy + Send + Sync>(
    &self,
    _: &mut Self::Cmd,
    _: usize,
  ) -> Result<Self::List<T>, EngineError> {
    todo!()
  }
  fn build_leaves(
    &self,
    _: &mut Self::Cmd,
    _: usize,
  ) -> Result<Self::Buffer<[u32; 8]>, EngineError> {
    todo!()
  }
  fn build_frames(
    &self,
    _: &mut Self::Cmd,
    _: &PhysicsScene,
  ) -> Result<Self::Buffer<GpuReferenceFrame>, EngineError> {
    todo!()
  }
  fn build_particle_frame_ids(
    &self,
    _: &mut Self::Cmd,
    _: &[ParticleMetadata],
  ) -> Result<Self::Buffer<u32>, EngineError> {
    todo!()
  }
  fn imex_integrate_particles_p1_p2(
    &self,
    _: &mut Self::Cmd,
    _: &mut Self::Buffer<f32>,
    _: i64,
  ) -> Result<(), EngineError> {
    todo!()
  }
  fn imex_integrate_bodies_p3(
    &self,
    _: &mut Self::Cmd,
    _: &mut Self::Buffer<RigidBodyImex>,
    _: &mut Self::Buffer<Wrench>,
    _: &Self::Buffer<ForceEmitter>,
    _: &Self::Buffer<GpuReferenceFrame>,
    _: i64,
  ) -> Result<(), EngineError> {
    todo!()
  }
  fn imex_rb_force_assign(
    &self,
    _: &mut Self::Cmd,
    _: &Self::Buffer<RigidBodyImex>,
    _: &mut Self::Buffer<Wrench>,
  ) -> Result<(), EngineError> {
    todo!()
  }
  fn imex_integrate_particles_p4_5(
    &self,
    _: &mut Self::Cmd,
    _: &mut Self::Buffer<f32>,
    _: i64,
    _: i64,
  ) -> Result<(), EngineError> {
    todo!()
  }
  fn apply_emitters_to_particles(
    &self,
    _: &mut Self::Cmd,
    _: &mut Self::Buffer<f32>,
    _: &Self::Buffer<ForceEmitter>,
    _: &Self::Buffer<GpuReferenceFrame>,
    _: &Self::Buffer<u32>,
    _: u32,
  ) -> Result<(), EngineError> {
    todo!()
  }

  fn create_command_buffer(&self) -> EngineResult<Self::Cmd> {
    todo!()
  }
  fn build_kinematic_bodies(
    &self,
    _cmd: &mut Self::Cmd,
    _scene: &PhysicsScene,
    _scene0: &Scene,
  ) -> EngineResult<Self::Buffer<KinematicBody>> {
    todo!()
  }
  fn build_rigid_bodies(
    &self,
    _cmd: &mut Self::Cmd,
    _scene: &PhysicsScene,
    _scene0: &Scene,
  ) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<Wrench>)> {
    todo!()
  }
  fn build_particles(
    &self,
    _cmd: &mut Self::Cmd,
    _scene0: &Scene,
  ) -> EngineResult<(Self::Buffer<f32>, Vec<ParticleMetadata>)> {
    todo!()
  }
  fn build_emitters(
    &self,
    _cmd: &mut Self::Cmd,
    _scene: &Scene,
  ) -> EngineResult<Self::Buffer<ForceEmitter>> {
    todo!()
  }
  fn build_emission_candidates(
    &self,
    _cmd: &mut Self::Cmd,
    _scene: &Scene,
  ) -> EngineResult<Self::Buffer<f32>> {
    todo!()
  }
  fn emit_particles(
    &self,
    _cmd: &mut Self::Cmd,
    _particles: &mut Self::Buffer<f32>,
    _physical_scene: &PhysicsScene,
    _scene: &Scene,
    _sun_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    _dt: timeus_t,
  ) -> EngineResult<()> {
    todo!()
  }
  fn step_ode_p1_p2(
    &self,
    _cmd: &mut Self::Cmd,
    _particles: &mut Self::Buffer<f32>,
    _dt: timeus_t,
  ) -> EngineResult<()> {
    todo!()
  }
  fn step_ode_p3_p4(
    &self,
    _cmd: &mut Self::Cmd,
    _kinematics: &Self::Buffer<KinematicBody>,
    _rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
    _emitters: &Self::Buffer<ForceEmitter>,
    _dt: timeus_t,
  ) -> EngineResult<()> {
    todo!()
  }
  fn compute_self_gravity(
    &self,
    _cmd: &mut Self::Cmd,
    _bvh: &Self::MotionBvh,
    _particles: &mut Self::Buffer<f32>,
  ) -> EngineResult<()> {
    todo!()
  }
  fn step_ode_p5(
    &self,
    _cmd: &mut Self::Cmd,
    _kinematics: &Self::Buffer<KinematicBody>,
    _particles: &mut Self::Buffer<f32>,
    _emitters: &Self::Buffer<ForceEmitter>,
    _dt: timeus_t,
  ) -> EngineResult<()> {
    todo!()
  }
  fn build_motion_bvh(
    &self,
    _cmd: &mut Self::Cmd,
    _kinematics: &Self::Buffer<KinematicBody>,
    _rigid_bodies: &Self::Buffer<RigidBodyImex>,
    _particles: &Self::Buffer<f32>,
    _dt: timeus_t,
  ) -> EngineResult<Self::MotionBvh> {
    todo!()
  }
  fn write_back_to_scene(
    &self,
    _cmd: &mut Self::Cmd,
    _rigid_bodies: &Self::Buffer<RigidBodyImex>,
    _particles: &Self::Buffer<f32>,
    _metadata: &[ParticleMetadata],
    _physical_scene: &mut PhysicsScene,
    _scene: &Scene,
  ) -> EngineResult<Option<CommandBufferSyncInfo>> {
    todo!()
  }
}
