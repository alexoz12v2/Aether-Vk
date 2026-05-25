//! cpu_kernels module.

use crate::{
  gpu::{
    CollisionPair, CommandBuffer, DeviceBuffer, DeviceBvh, DeviceList, ForceEmitter, Kernels,
    KinematicBody, ParticleGpu, ParticleMetadata, RigidBodyGpu, RigidBodyImex, WaitHandle, Wrench,
  },
  physics::physics_scene::{GpuReferenceFrame, PhysicsScene},
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

  fn address(&self) -> u64 {
    0
  }
  fn enqueue_read_to_cpu(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'_>> {
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

  fn address(&self) -> u64 {
    0
  }
  fn enqueue_read_to_cpu(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'_>> {
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
  fn address(&self) -> u64 {
    0
  }
}

/// Zero-cost CPU stand-in for the per-tick TLAS GPU buffer.
/// `address()` returns 0 — broadphase shaders are no-ops in the CPU kernel path.
pub struct CpuTlasHandle;

impl DeviceBvh for CpuTlasHandle {
  type Cmd = CpuCommandBuffer;
  fn address(&self) -> u64 {
    0
  }
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

  fn subgroup_size(&self) -> Option<crate::gpu::SubgroupSize> {
    Some(crate::gpu::SubgroupSize::Size32)
  }
  fn wait_sync(&self, _sync: &crate::gpu::CommandBufferSyncInfo) -> EngineResult<()> {
    Ok(())
  }
  fn refit_motion_blas(
    &self,
    _cmd: &mut Self::Cmd,
    _bvh: &Self::MotionBvh,
    _depth_indices: &Self::Buffer<u32>,
    _total_nodes: u32,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn upload_motion_tlas(
    &self,
    _cmd: &mut Self::Cmd,
    _node_bytes: &[u8],
  ) -> EngineResult<Self::MotionTlas> {
    Ok(CpuTlasHandle)
  }

  fn create_command_buffer(&self) -> EngineResult<Self::Cmd> {
    Ok(CpuCommandBuffer {
      tasks: alloc::vec::Vec::new(),
    })
  }

  fn build_list<T: Copy + Send + Sync>(
    &self,
    cmd: &mut Self::Cmd,
    capacity: usize,
  ) -> EngineResult<Self::List<T>> { Ok(CpuList { data: alloc::vec![unsafe { core::mem::zeroed() }; capacity] })
  }

  fn build_leaves(
    &self,
    _cmd: &mut Self::Cmd,
    _capacity: usize,
  ) -> EngineResult<Self::Buffer<[u32; 8]>> {
    Ok(CpuBuffer {
      data: alloc::vec::Vec::new(),
    })
  }

  fn build_kinematic_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<KinematicBody>> {
    Ok(CpuBuffer {
      data: alloc::vec::Vec::new(),
    })
  }

  fn build_rigid_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<Wrench>)> {
    Ok((
      CpuBuffer {
        data: alloc::vec::Vec::new(),
      },
      CpuBuffer {
        data: alloc::vec::Vec::new(),
      },
    ))
  }

  fn build_frames(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
  ) -> EngineResult<Self::Buffer<GpuReferenceFrame>> {
    Ok(CpuBuffer {
      data: alloc::vec::Vec::new(),
    })
  }

  fn build_particles(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<(Self::Buffer<f32>, alloc::vec::Vec<ParticleMetadata>)> {
    Ok((
      CpuBuffer {
        data: alloc::vec::Vec::new(),
      },
      alloc::vec::Vec::new(),
    ))
  }

  fn build_emitters(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<Self::Buffer<ForceEmitter>> {
    Ok(CpuBuffer {
      data: alloc::vec::Vec::new(),
    })
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
    _cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    let dt_sec = dt as f32 / 1_000_000.0;
    let half_dt = dt_sec * 0.5;
    let subgroup_size = 32;
    let stride = 10 * subgroup_size;

    // Total blocks based on buffer capacity
    let total_blocks = particles.data.len() / stride;

    for block_idx in 0..total_blocks {
      for lane_idx in 0..subgroup_size {
        let base = block_idx * stride + lane_idx;

        // Safety bounds check
        if base + 9 * subgroup_size >= particles.data.len() {
          continue;
        }

        let mass = particles.data[base + 6 * subgroup_size];
        if mass <= 0.0 {
          continue;
        }

        let inv_m = 1.0 / mass;

        // Load Velocity (slots 3, 4, 5)
        let vx = particles.data[base + 3 * subgroup_size];
        let vy = particles.data[base + 4 * subgroup_size];
        let vz = particles.data[base + 5 * subgroup_size];

        // Load Force (slots 7, 8, 9)
        let fx = particles.data[base + 7 * subgroup_size];
        let fy = particles.data[base + 8 * subgroup_size];
        let fz = particles.data[base + 9 * subgroup_size];

        // Half-kick: v_{n+1/2} = v_n + (dt/2) * M^-1 * F(x_n)
        let v_half_x = vx + fx * inv_m * half_dt;
        let v_half_y = vy + fy * inv_m * half_dt;
        let v_half_z = vz + fz * inv_m * half_dt;

        // Load Position (slots 0, 1, 2)
        let px = particles.data[base + 0 * subgroup_size];
        let py = particles.data[base + 1 * subgroup_size];
        let pz = particles.data[base + 2 * subgroup_size];

        // Full position leap
        particles.data[base + 0 * subgroup_size] = px + v_half_x * dt_sec;
        particles.data[base + 1 * subgroup_size] = py + v_half_y * dt_sec;
        particles.data[base + 2 * subgroup_size] = pz + v_half_z * dt_sec;

        // Store half-kick velocity for p4_5
        particles.data[base + 3 * subgroup_size] = v_half_x;
        particles.data[base + 4 * subgroup_size] = v_half_y;
        particles.data[base + 5 * subgroup_size] = v_half_z;

        // Clear Force for the next passes (bp_particle_self, barnes_hut, etc.)
        particles.data[base + 7 * subgroup_size] = 0.0;
        particles.data[base + 8 * subgroup_size] = 0.0;
        particles.data[base + 9 * subgroup_size] = 0.0;
      }
    }
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
    _cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    dt: timeus_t,
    _current_time_us: timeus_t,
  ) -> EngineResult<()> {
    let dt_sec = dt as f32 / 1_000_000.0;
    let half_dt = dt_sec * 0.5;
    let subgroup_size = 32;
    let stride = 10 * subgroup_size;
    let total_blocks = particles.data.len() / stride;

    for block_idx in 0..total_blocks {
      for lane_idx in 0..subgroup_size {
        let base = block_idx * stride + lane_idx;
        if base + 9 * subgroup_size >= particles.data.len() {
          continue;
        }

        let mass = particles.data[base + 6 * subgroup_size];
        if mass <= 0.0 {
          continue;
        }

        let inv_m = 1.0 / mass;

        // Load v_{n+1/2} from slots 3, 4, 5
        let v_half_x = particles.data[base + 3 * subgroup_size];
        let v_half_y = particles.data[base + 4 * subgroup_size];
        let v_half_z = particles.data[base + 5 * subgroup_size];

        // Load newly accumulated F(x_{n+1}) from slots 7, 8, 9
        let f_next_x = particles.data[base + 7 * subgroup_size];
        let f_next_y = particles.data[base + 8 * subgroup_size];
        let f_next_z = particles.data[base + 9 * subgroup_size];

        // VV Corrector
        particles.data[base + 3 * subgroup_size] = v_half_x + f_next_x * inv_m * half_dt;
        particles.data[base + 4 * subgroup_size] = v_half_y + f_next_y * inv_m * half_dt;
        particles.data[base + 5 * subgroup_size] = v_half_z + f_next_z * inv_m * half_dt;

        // Notice we DO NOT clear the force buffer here, just like in GLSL.
        // It persists into the start of the next frame.
      }
    }

    // In the CPU path, clock advancement is likely handled by the top-level
    // Engine/PhysicsScene manager rather than thread 0 of this kernel.
    Ok(())
  }


  #[cfg(any(test, feature = "collisions"))]
  fn bp_clear(
    &self,
    _cmd: &mut Self::Cmd,
    _raw_pairs_addr: u64,
    _out_rb_rb_addr: u64,
    _out_rb_ps_addr: u64,
    _out_rb_lca_addr: u64,
    _out_internal: u64,
    _out_sparse: u64,
  ) -> EngineResult<()> {
    // Equivalent of bp_clear.comp.
    // Because device lists/buffers are purely simulated by standard Rust Vecs
    // on the CPU, and your DeviceList trait already dictates clearing natively
    // via `.clear()`, this explicit GPU phase does not need any pointer dereferencing
    // and is a safe no-op on the CPU architecture.
    Ok(())
  }


  #[cfg(any(test, feature = "collisions"))]
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


  #[cfg(any(test, feature = "collisions"))]
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


  #[cfg(any(test, feature = "collisions"))]
  fn bp_classify(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &Self::Buffer<RigidBodyImex>,
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_ps_ps_addr: u64,
    out_macro_lca_addr: u64,
    out_lca_lca_addr: u64,
    total_raw_pairs: u32,
  ) -> EngineResult<()> {
    Ok(())
  }


  #[cfg(any(test, feature = "collisions"))]
  fn bp_cross_lca(
    &self,
    cmd: &mut Self::Cmd,
    tlas_bvh_addr: u64,
    lca_entities_addr: u64,
    macro_leaves_addr: u64,
    entity_headers_addr: u64,
    lca_query_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_ps_ps_addr: u64,
    out_cross_pairs_addr: u64,
    total_queries: u32,
    max_pairs: u32,
  ) -> EngineResult<()> {
    Ok(())
  }


  #[cfg(any(test, feature = "collisions"))]
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
    Ok(CpuMotionBvh {
      kinematics_copy: alloc::vec::Vec::new(),
      rigid_bodies_copy: alloc::vec::Vec::new(),
      particles_copy: alloc::vec::Vec::new(),
      particle_metadata_copy: alloc::vec::Vec::new(),
      bvh_tree: crate::physics::motion_bvh::MotionBvhTree {
        nodes: alloc::vec::Vec::new(),
        root: None,
      },
    })
  }


  #[cfg(any(test, feature = "collisions"))]
  fn self_intersect_scene(
    &self,
    cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
  ) -> EngineResult<Self::List<CollisionPair>> { Ok(CpuList { data: alloc::vec::Vec::new() }) }


  #[cfg(any(test, feature = "collisions"))]
  fn intersect_instances(
    &self,
    cmd: &mut Self::Cmd,
    potentials: &Self::List<CollisionPair>,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
  ) -> EngineResult<Self::List<CollisionPair>> { Ok(CpuList { data: alloc::vec::Vec::new() }) }


  #[cfg(any(test, feature = "collisions"))]
  fn narrow_ccd_cross_lca(
    &self,
    _cmd: &mut Self::Cmd,
    _broadphase_pairs: &Self::List<crate::gpu::CrossPair>,
    _rigid_bodies: &Self::Buffer<RigidBodyImex>,
    _particles: &Self::Buffer<f32>,
    _lca_entities: u64,
    _space_type: u32,
    _dt: f32,
    _output_list: &Self::List<CollisionPair>,
  ) -> EngineResult<()> { unimplemented!() }

  #[cfg(any(test, feature = "collisions"))]
  fn narrow_ccd(
    &self,
    _cmd: &mut Self::Cmd,
    _broadphase_pairs: &Self::List<CollisionPair>,
    _rigid_bodies: &Self::Buffer<RigidBodyImex>,
    _particles: &Self::Buffer<f32>,
    _lca_entities: u64,
    _space_type: u32, dt: f32,
    _output_list: &Self::List<CollisionPair>,
  ) -> EngineResult<()> { Ok(()) }


  #[cfg(any(test, feature = "collisions"))]
  fn compact_collisions(
    &self,
    cmd: &mut Self::Cmd,
    globals: &Self::List<CollisionPair>,
    time_delta: timeus_t,
  ) -> EngineResult<Self::List<CollisionPair>> { Ok(CpuList { data: alloc::vec![unsafe { core::mem::zeroed() }; globals.capacity().max(1)] }) }


  #[cfg(any(test, feature = "collisions"))]
  fn find_earliest_collision(
    &self,
    cmd: &mut Self::Cmd,
    compacted: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::Buffer<u32>> {
    Ok(CpuBuffer {
      data: alloc::vec::Vec::new(),
    })
  }


  #[cfg(any(test, feature = "collisions"))]
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


  #[cfg(any(test, feature = "collisions"))]
  fn snapshot_dynamics(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
  ) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<f32>)> {
    Ok((
      CpuBuffer {
        data: alloc::vec::Vec::new(),
      },
      CpuBuffer {
        data: alloc::vec::Vec::new(),
      },
    ))
  }


  #[cfg(any(test, feature = "collisions"))]
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

  fn subgroup_size(&self) -> Option<crate::gpu::SubgroupSize> {
    Some(crate::gpu::SubgroupSize::Size32)
  }
  fn wait_sync(&self, _sync: &crate::gpu::CommandBufferSyncInfo) -> EngineResult<()> {
    Ok(())
  }
  fn refit_motion_blas(
    &self,
    _cmd: &mut Self::Cmd,
    _bvh: &Self::MotionBvh,
    _depth_indices: &Self::Buffer<u32>,
    _total_nodes: u32,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn upload_motion_tlas(
    &self,
    _cmd: &mut Self::Cmd,
    _node_bytes: &[u8],
  ) -> EngineResult<Self::MotionTlas> {
    Ok(CpuTlasHandle)
  }

  fn create_command_buffer(&self) -> EngineResult<Self::Cmd> {
    Ok(CpuCommandBuffer {
      tasks: alloc::vec::Vec::new(),
    })
  }

  fn build_list<T: Copy + Send + Sync>(
    &self,
    cmd: &mut Self::Cmd,
    capacity: usize,
  ) -> EngineResult<Self::List<T>> { Ok(CpuList { data: alloc::vec![unsafe { core::mem::zeroed() }; capacity] })
  }

  fn build_leaves(
    &self,
    _cmd: &mut Self::Cmd,
    _capacity: usize,
  ) -> EngineResult<Self::Buffer<[u32; 8]>> {
    Ok(CpuBuffer {
      data: alloc::vec::Vec::new(),
    })
  }

  fn build_kinematic_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<KinematicBody>> {
    Ok(CpuBuffer {
      data: alloc::vec::Vec::new(),
    })
  }

  fn build_rigid_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<Wrench>)> {
    Ok((
      CpuBuffer {
        data: alloc::vec::Vec::new(),
      },
      CpuBuffer {
        data: alloc::vec::Vec::new(),
      },
    ))
  }

  fn build_frames(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
  ) -> EngineResult<Self::Buffer<GpuReferenceFrame>> {
    Ok(CpuBuffer {
      data: alloc::vec::Vec::new(),
    })
  }

  fn build_particles(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<(Self::Buffer<f32>, alloc::vec::Vec<ParticleMetadata>)> {
    Ok((
      CpuBuffer {
        data: alloc::vec::Vec::new(),
      },
      alloc::vec::Vec::new(),
    ))
  }

  fn build_emitters(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<Self::Buffer<ForceEmitter>> {
    Ok(CpuBuffer {
      data: alloc::vec::Vec::new(),
    })
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


  #[cfg(any(test, feature = "collisions"))]
  fn bp_clear(
    &self,
    cmd: &mut Self::Cmd,
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_rb_lca_addr: u64,
    out_internal: u64, out_sparse: u64,
  ) -> EngineResult<()> {
    Ok(())
  }


  #[cfg(any(test, feature = "collisions"))]
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


  #[cfg(any(test, feature = "collisions"))]
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


  #[cfg(any(test, feature = "collisions"))]
  fn bp_classify(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &Self::Buffer<RigidBodyImex>,
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_ps_ps_addr: u64,
    out_macro_lca_addr: u64,
    out_lca_lca_addr: u64,
    total_raw_pairs: u32,
  ) -> EngineResult<()> {
    Ok(())
  }


  #[cfg(any(test, feature = "collisions"))]
  fn bp_cross_lca(
    &self,
    cmd: &mut Self::Cmd,
    tlas_bvh_addr: u64,
    lca_entities_addr: u64,
    macro_leaves_addr: u64,
    entity_headers_addr: u64,
    lca_query_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_ps_ps_addr: u64,
    out_cross_pairs_addr: u64,
    total_queries: u32,
    max_pairs: u32,
  ) -> EngineResult<()> {
    Ok(())
  }


  #[cfg(any(test, feature = "collisions"))]
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
    Ok(CpuMotionBvh {
      kinematics_copy: alloc::vec::Vec::new(),
      rigid_bodies_copy: alloc::vec::Vec::new(),
      particles_copy: alloc::vec::Vec::new(),
      particle_metadata_copy: alloc::vec::Vec::new(),
      bvh_tree: crate::physics::motion_bvh::MotionBvhTree {
        nodes: alloc::vec::Vec::new(),
        root: None,
      },
    })
  }


  #[cfg(any(test, feature = "collisions"))]
  fn self_intersect_scene(
    &self,
    cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
  ) -> EngineResult<Self::List<CollisionPair>> { Ok(CpuList { data: alloc::vec::Vec::new() }) }


  #[cfg(any(test, feature = "collisions"))]
  fn intersect_instances(
    &self,
    cmd: &mut Self::Cmd,
    potentials: &Self::List<CollisionPair>,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
  ) -> EngineResult<Self::List<CollisionPair>> { Ok(CpuList { data: alloc::vec::Vec::new() }) }


  #[cfg(any(test, feature = "collisions"))]
  fn narrow_ccd_cross_lca(
    &self,
    _cmd: &mut Self::Cmd,
    _broadphase_pairs: &Self::List<crate::gpu::CrossPair>,
    _rigid_bodies: &Self::Buffer<RigidBodyImex>,
    _particles: &Self::Buffer<f32>,
    _lca_entities: u64,
    _space_type: u32,
    _dt: f32,
    _output_list: &Self::List<CollisionPair>,
  ) -> EngineResult<()> { unimplemented!() }

  #[cfg(any(test, feature = "collisions"))]
  fn narrow_ccd(
    &self,
    _cmd: &mut Self::Cmd,
    _broadphase_pairs: &Self::List<CollisionPair>,
    _rigid_bodies: &Self::Buffer<RigidBodyImex>,
    _particles: &Self::Buffer<f32>,
    _lca_entities: u64,
    _space_type: u32, dt: f32,
    _output_list: &Self::List<CollisionPair>,
  ) -> EngineResult<()> { Ok(()) }


  #[cfg(any(test, feature = "collisions"))]
  fn compact_collisions(
    &self,
    cmd: &mut Self::Cmd,
    globals: &Self::List<CollisionPair>,
    time_delta: timeus_t,
  ) -> EngineResult<Self::List<CollisionPair>> { Ok(CpuList { data: alloc::vec![unsafe { core::mem::zeroed() }; globals.capacity().max(1)] }) }


  #[cfg(any(test, feature = "collisions"))]
  fn find_earliest_collision(
    &self,
    cmd: &mut Self::Cmd,
    compacted: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::Buffer<u32>> {
    Ok(CpuBuffer {
      data: alloc::vec::Vec::new(),
    })
  }


  #[cfg(any(test, feature = "collisions"))]
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


  #[cfg(any(test, feature = "collisions"))]
  fn snapshot_dynamics(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
  ) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<f32>)> {
    Ok((
      CpuBuffer {
        data: alloc::vec::Vec::new(),
      },
      CpuBuffer {
        data: alloc::vec::Vec::new(),
      },
    ))
  }


  #[cfg(any(test, feature = "collisions"))]
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
    let norm = Vec3f32::from_array([1.0, 0.0, 0.0]);
    // FIX: Changed from 0.5_f32 to 0.05_f32 so it falls within the
    // time_tolerance (0.01) of c2 (0.051) and c3 (0.052).
    let t = 0.05_f32;
    let pt = Vec3f32::from_array([1.0, 1.0, 1.0]);
    let depth = 1_f32;
    let c1 = CollisionPair {
      a: crate::gpu::ColliderId {
        entity_id: 0,
        primitive_index: 0,
      },
      b: crate::gpu::ColliderId {
        entity_id: 0,
        primitive_index: 1,
      },
      time_of_impact: t,
      contact_normal: norm.into(),
      contact_point: pt.into(),
      penetration_depth: depth,
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