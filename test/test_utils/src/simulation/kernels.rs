extern crate alloc;
use alloc::vec::Vec;

use aethervk_core_rlib::{
  gpu::{
    CollisionPair, CommandBuffer, DeviceBuffer, DeviceBvh, DeviceList, ForceEmitter, Kernels,
    KinematicBody, ParticleGpu, RigidBodyGpu, WaitHandle,
  },
  physics::physics_scene::PhysicsScene,
  scene::{ParticleSystemComponent, Scene},
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib::{
  math::vector::{Vector3, vec3::Vec3f32},
  os::time::timeus_t,
};

pub struct CpuCommandBuffer {
  pub submitted: bool,
}

impl CommandBuffer for CpuCommandBuffer {
  fn submit(&mut self) -> EngineResult<()> {
    self.submitted = true;
    Ok(())
  }
}

pub struct CpuDeviceBuffer<T> {
  pub data: Vec<T>,
}

impl<T: Copy + Send + Sync> DeviceBuffer<T> for CpuDeviceBuffer<T> {
  type Cmd = CpuCommandBuffer;
  type ReadHandle<'a>
    = CpuWaitHandle<'a, Vec<T>>
  where
    Self: 'a,
    T: 'a;

  fn capacity(&self) -> usize {
    self.data.capacity()
  }

  fn enqueue_read_to_cpu(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'_>> {
    Ok(CpuWaitHandle {
      data: self.data.clone(),
      _phantom: core::marker::PhantomData,
    })
  }
}

pub struct CpuDeviceList<T> {
  pub buffer: CpuDeviceBuffer<T>,
}

impl<T: Copy + Send + Sync> DeviceBuffer<T> for CpuDeviceList<T> {
  type Cmd = CpuCommandBuffer;
  type ReadHandle<'a>
    = CpuWaitHandle<'a, Vec<T>>
  where
    Self: 'a,
    T: 'a;

  fn capacity(&self) -> usize {
    self.buffer.capacity()
  }

  fn enqueue_read_to_cpu(&self, cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'_>> {
    self.buffer.enqueue_read_to_cpu(cmd)
  }
}

impl<T: Copy + Send + Sync> DeviceList<T> for CpuDeviceList<T> {
  fn clear(&mut self, _cmd: &mut Self::Cmd) -> EngineResult<()> {
    self.buffer.data.clear();
    Ok(())
  }
}

pub struct CpuWaitHandle<'a, T> {
  data: T,
  _phantom: core::marker::PhantomData<&'a ()>,
}

impl<'a, T: Send + Sync> WaitHandle<T> for CpuWaitHandle<'a, T> {
  fn wait(self) -> EngineResult<T> {
    Ok(self.data)
  }
}

pub struct CpuMotionBvh {}

impl DeviceBvh for CpuMotionBvh {
  type Cmd = CpuCommandBuffer;
}

pub struct CpuKernels {
  pub kinematic_masses: std::sync::RwLock<Vec<f32>>,
  pub kinematic_is_sun: std::sync::RwLock<Vec<bool>>,
  pub dynamic_betas: std::sync::RwLock<Vec<f32>>,
  pub dynamic_mapping: std::sync::RwLock<Vec<(aethervk_core_rlib::scene::EntityId, usize)>>,
  pub dynamic_accelerations:
    std::sync::RwLock<Vec<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>>,
}

impl CpuKernels {
  pub fn new() -> Self {
    Self {
      kinematic_masses: std::sync::RwLock::new(Vec::new()),
      kinematic_is_sun: std::sync::RwLock::new(Vec::new()),
      dynamic_betas: std::sync::RwLock::new(Vec::new()),
      dynamic_mapping: std::sync::RwLock::new(Vec::new()),
      dynamic_accelerations: std::sync::RwLock::new(Vec::new()),
    }
  }
}

impl Kernels for CpuKernels {
  type Cmd = CpuCommandBuffer;
  type Buffer<T: Copy + Send + Sync> = CpuDeviceBuffer<T>;
  type List<T: Copy + Send + Sync> = CpuDeviceList<T>;
  type MotionBvh = CpuMotionBvh;

  fn discard_buffer<T: Copy + Send + Sync>(&self, _buffer: Self::Buffer<T>) {}
  fn discard_list<T: Copy + Send + Sync>(&self, _list: Self::List<T>) {}
  fn discard_bvh(&self, _bvh: Self::MotionBvh) {}

  fn create_command_buffer(&self) -> EngineResult<Self::Cmd> {
    Ok(CpuCommandBuffer { submitted: false })
  }

  fn build_kinematic_bodies(
    &self,
    _cmd: &mut Self::Cmd,
    _scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<KinematicBody>> {
    let mut bodies = Vec::new();
    let mut masses = Vec::new();
    let mut is_sun = Vec::new();

    scene0.query1::<aethervk_core_rlib::scene::SunComponent, _>(|entity, _| {
      if let Some(transform) = scene0.global_transform(entity) {
        bodies.push(KinematicBody {
          entity_id: entity,
          transform,
          velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]),
          parent_frame_id: 0,
          mu: 0.0,
          own_frame_id: 0,
          frame_type: 0,
          scale: 0.0,
          shape_type: 0,
          shape_data: [1.0, 0.0, 0.0],
        });
        masses.push(100000000.0);
        is_sun.push(true);
      }
    });

    scene0.query1::<aethervk_core_rlib::scene::PhysicalMeshComponent, _>(|entity, mesh_comp| {
      if let Some(transform) = scene0.global_transform(entity) {
        bodies.push(KinematicBody {
          entity_id: entity,
          transform,
          velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]),
          parent_frame_id: 0,
          mu: 0.0,
          own_frame_id: 0,
          frame_type: 0,
          scale: 0.0,
          shape_type: 0,
          shape_data: [1.0, 0.0, 0.0],
        });
        masses.push(mesh_comp.mesh.mass_properties.mass as f32 * 5000000.0);
        is_sun.push(false);
      }
    });

    *self.kinematic_masses.write().unwrap() = masses;
    *self.kinematic_is_sun.write().unwrap() = is_sun;

    Ok(CpuDeviceBuffer { data: bodies })
  }

  fn build_rigid_bodies(
    &self,
    _cmd: &mut Self::Cmd,
    _scene: &PhysicsScene,
    _scene0: &Scene,
  ) -> EngineResult<Self::Buffer<RigidBodyGpu>> {
    Ok(CpuDeviceBuffer { data: Vec::new() })
  }

  fn build_particles(
    &self,
    _cmd: &mut Self::Cmd,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<ParticleGpu>> {
    let mut bodies = Vec::new();
    let mut betas = Vec::new();
    let mut mapping = Vec::new();

    scene0.query2::<ParticleSystemComponent, aethervk_core_rlib::scene::particles::ParticleEmitterComponent, _>(|entity, sys, config| {
      for (i, p) in sys.particles.read().iter().enumerate() {
        if p.active != 0 {
          bodies.push(ParticleGpu {
            position: p.position,
            velocity: p.velocity,
            mass: p.mass,
            force: [0.0, 0.0, 0.0],
            entity_id: entity,
            parent_frame_id: 0,
            original_index: i as u32,
          });
          betas.push(config.beta);
          mapping.push((entity, i));
        }
      }
    });

    *self.dynamic_betas.write().unwrap() = betas;
    *self.dynamic_mapping.write().unwrap() = mapping;
    self.dynamic_accelerations.write().unwrap().resize(
      bodies.len(),
      aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]),
    );

    Ok(CpuDeviceBuffer { data: bodies })
  }

  fn build_emitters(
    &self,
    _cmd: &mut Self::Cmd,
    _scene: &Scene,
  ) -> EngineResult<Self::Buffer<ForceEmitter>> {
    Ok(CpuDeviceBuffer { data: Vec::new() })
  }

  fn emit_particles(
    &self,
    _cmd: &mut Self::Cmd,
    _particles: &mut Self::Buffer<ParticleGpu>,
    _physical_scene: &PhysicsScene,
    _scene: &Scene,
    _sun_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    _dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn step_ode_p1_p2(
    &self,
    _cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<ParticleGpu>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    let dt_sec = dt as f32 / 1_000_000.0;
    let half_dt = dt_sec * 0.5;
    let accels =
      self.dynamic_accelerations.read().map_err(|_| EngineError::InvalidOperation("fail lock"))?;

    for (i, p) in particles.data.iter_mut().enumerate() {
      p.velocity[0] += accels[i].x() * half_dt;
      p.velocity[1] += accels[i].y() * half_dt;
      p.velocity[2] += accels[i].z() * half_dt;

      p.position[0] += p.velocity[0] * half_dt;
      p.position[1] += p.velocity[1] * half_dt;
      p.position[2] += p.velocity[2] * half_dt;
    }
    Ok(())
  }

  fn step_ode_p3_p4(
    &self,
    _cmd: &mut Self::Cmd,
    _kinematics: &Self::Buffer<KinematicBody>,
    _rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
    _emitters: &Self::Buffer<ForceEmitter>,
    _dt: timeus_t,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn compute_self_gravity(
    &self,
    _cmd: &mut Self::Cmd,
    _bvh: &Self::MotionBvh,
    _particles: &mut Self::Buffer<ParticleGpu>,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn step_ode_p5(
    &self,
    _cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    particles: &mut Self::Buffer<ParticleGpu>,
    _emitters: &Self::Buffer<ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    let dt_sec = dt as f32 / 1_000_000.0;
    let half_dt = dt_sec * 0.5;

    let masses = self.kinematic_masses.read().unwrap();
    let is_sun = self.kinematic_is_sun.read().unwrap();
    let betas = self.dynamic_betas.read().unwrap();
    let mut accels = self.dynamic_accelerations.write().unwrap();

    for (i, p) in particles.data.iter_mut().enumerate() {
      p.position[0] += p.velocity[0] * half_dt;
      p.position[1] += p.velocity[1] * half_dt;
      p.position[2] += p.velocity[2] * half_dt;

      let mut total_force = Vec3f32::from_array([0.0, 0.0, 0.0]);
      let beta = betas[i];
      let p_pos = Vec3f32::from_array(p.position);

      for (j, kin_body) in kinematics.data.iter().enumerate() {
        let to_kin = kin_body.transform.position - p_pos;
        use aethervk_oshal_rlib::math::vector::Vector;
        let dist_sq = to_kin.length_squared().max(1e-4);

        let force_mag = aethervk_core_rlib::physics::cpu::G * masses[j] * p.mass / dist_sq;
        let mut force = to_kin.normalize() * force_mag;

        if is_sun[j] {
          force = force * (1.0 - beta);
        }

        total_force += force;
      }

      accels[i] = total_force / p.mass;
      p.velocity[0] += accels[i].x() * half_dt;
      p.velocity[1] += accels[i].y() * half_dt;
      p.velocity[2] += accels[i].z() * half_dt;
    }
    Ok(())
  }

  fn build_motion_bvh(
    &self,
    _cmd: &mut Self::Cmd,
    _kinematics: &Self::Buffer<KinematicBody>,
    _rigid_bodies: &Self::Buffer<RigidBodyGpu>,
    _particles: &Self::Buffer<ParticleGpu>,
    _dt: timeus_t,
  ) -> EngineResult<Self::MotionBvh> {
    Ok(CpuMotionBvh {})
  }

  fn self_intersect_scene(
    &self,
    _cmd: &mut Self::Cmd,
    _bvh: &Self::MotionBvh,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuDeviceList {
      buffer: CpuDeviceBuffer { data: Vec::new() },
    })
  }

  fn intersect_instances(
    &self,
    _cmd: &mut Self::Cmd,
    _potentials: &Self::List<CollisionPair>,
    _kinematics: &Self::Buffer<KinematicBody>,
    _rigid_bodies: &Self::Buffer<RigidBodyGpu>,
    _particles: &Self::Buffer<ParticleGpu>,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuDeviceList {
      buffer: CpuDeviceBuffer { data: Vec::new() },
    })
  }

  fn compact_collisions(
    &self,
    _cmd: &mut Self::Cmd,
    _globals: &Self::List<CollisionPair>,
    _time_delta: timeus_t,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuDeviceList {
      buffer: CpuDeviceBuffer { data: Vec::new() },
    })
  }

  fn find_earliest_collision(
    &self,
    _cmd: &mut Self::Cmd,
    _compacted: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::Buffer<timeus_t>> {
    Ok(CpuDeviceBuffer {
      data: alloc::vec![aethervk_oshal_rlib::os::time::timeus_t::MAX],
    })
  }

  fn apply_collision_responses(
    &self,
    _cmd: &mut Self::Cmd,
    _kinematics: &Self::Buffer<KinematicBody>,
    _rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
    _particles: &mut Self::Buffer<ParticleGpu>,
    _collisions: &Self::List<CollisionPair>,
    _force_inelastic: bool,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn snapshot_dynamics(
    &self,
    _cmd: &mut Self::Cmd,
    _rigid_bodies: &Self::Buffer<RigidBodyGpu>,
    particles: &Self::Buffer<ParticleGpu>,
  ) -> EngineResult<(Self::Buffer<RigidBodyGpu>, Self::Buffer<ParticleGpu>)> {
    Ok((
      CpuDeviceBuffer { data: Vec::new() },
      CpuDeviceBuffer {
        data: particles.data.clone(),
      },
    ))
  }

  fn restore_dynamics(
    &self,
    _cmd: &mut Self::Cmd,
    _rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
    particles: &mut Self::Buffer<ParticleGpu>,
    snapshot: &(Self::Buffer<RigidBodyGpu>, Self::Buffer<ParticleGpu>),
  ) -> EngineResult<()> {
    particles.data = snapshot.1.data.clone();
    Ok(())
  }

  fn write_back_to_scene(
    &self,
    _cmd: &mut Self::Cmd,
    _rigid_bodies: &Self::Buffer<RigidBodyGpu>,
    particles: &Self::Buffer<ParticleGpu>,
    _physical_scene: &mut PhysicsScene,
    scene: &Scene,
  ) -> EngineResult<()> {
    let mapping = self.dynamic_mapping.read().unwrap();
    scene.query1_mut::<ParticleSystemComponent, _>(|entity, sys| {
      for (i, p_gpu) in particles.data.iter().enumerate() {
        let (e_id, particle_idx) = mapping[i];
        if e_id == entity {
          if let Some(p) = sys.particles.write().get_mut(particle_idx) {
            p.position = p_gpu.position;
            p.velocity = p_gpu.velocity;
          }
        }
      }
    });
    Ok(())
  }
}
