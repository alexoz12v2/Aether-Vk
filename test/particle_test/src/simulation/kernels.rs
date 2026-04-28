extern crate alloc;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

use aethervk_core_rlib::gpu::{
  CommandBuffer, DeviceBuffer, DeviceBvh, DeviceList, Kernels, DynamicBody, KinematicBody,
  CollisionPair, WaitHandle,
};
use aethervk_core_rlib::types::{EngineError, EngineResult};
use aethervk_oshal_rlib::os::time::timeus_t;
use aethervk_core_rlib::physics::physics_scene::PhysicsScene;
use aethervk_core_rlib::scene::Scene;

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

  fn enqueue_read_to_cpu<'a>(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'a>>
  where
    T: 'a,
  {
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

  fn enqueue_read_to_cpu<'a>(&self, cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'a>>
  where
    T: 'a,
  {
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
        });
        masses.push(mesh_comp.mesh.mass_properties.mass as f32 * 5000000.0);
        is_sun.push(false);
      }
    });

    *self.kinematic_masses.write().unwrap() = masses;
    *self.kinematic_is_sun.write().unwrap() = is_sun;

    Ok(CpuDeviceBuffer { data: bodies })
  }

  fn build_dynamic_bodies(
    &self,
    _cmd: &mut Self::Cmd,
    _scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<DynamicBody>> {
    let mut bodies = Vec::new();
    let mut betas = Vec::new();
    let mut mapping = Vec::new();

    scene0.query1::<super::components::ParticleSystemComponent, _>(|entity, sys| {
      for (i, p) in sys.particles.iter().enumerate() {
        if p.active != 0 {
          bodies.push(DynamicBody {
            entity_id: entity,
            transform: aethervk_core_rlib::scene::TransformComponent {
              position: p.position,
              rotation: Default::default(),
              scale: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([1.0, 1.0, 1.0]),
            },
            velocity: p.velocity,
            mass: p.mass,
          });
          betas.push(sys.config.beta);
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

  fn compute_forces(
    &self,
    _cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    dynamics: &mut Self::Buffer<DynamicBody>,
  ) -> EngineResult<()> {
    let masses = self.kinematic_masses.read().unwrap();
    let is_sun = self.kinematic_is_sun.read().unwrap();
    let betas = self.dynamic_betas.read().unwrap();
    let mut accels = self.dynamic_accelerations.write().unwrap();

    for (i, dyn_body) in dynamics.data.iter().enumerate() {
      let mut total_force =
        aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]);
      let beta = betas[i];

      for (j, kin_body) in kinematics.data.iter().enumerate() {
        let to_kin = kin_body.transform.position - dyn_body.transform.position;
        use aethervk_oshal_rlib::math::vector::Vector;
        let dist_sq = to_kin.length_squared().max(1e-4);

        let force_mag = aethervk_core_rlib::physics::cpu::G * masses[j] * dyn_body.mass / dist_sq;
        let mut force = to_kin.normalize() * force_mag;

        if is_sun[j] {
          force = force * (1.0 - beta);
        }

        total_force += force;
      }

      accels[i] = total_force / dyn_body.mass;
    }
    Ok(())
  }

  fn step_ode(
    &self,
    _cmd: &mut Self::Cmd,
    dynamics: &mut Self::Buffer<DynamicBody>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    let dt_sec = dt as f32 / 1_000_000.0;
    let accels = self
      .dynamic_accelerations
      .read()
      .map_err(|_| EngineError::InvalidOperation("fail lock"))?;

    for (i, dyn_body) in dynamics.data.iter_mut().enumerate() {
      dyn_body.velocity += accels[i] * dt_sec;
      dyn_body.transform.position += dyn_body.velocity * dt_sec;
    }

    Ok(())
  }

  fn build_motion_bvh(
    &self,
    _cmd: &mut Self::Cmd,
    _dynamics: &Self::Buffer<DynamicBody>,
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
    _dynamics: &mut Self::Buffer<DynamicBody>,
    _collisions: &Self::List<CollisionPair>,
    _force_inelastic: bool,
  ) -> EngineResult<()> {
    Ok(())
  }

  fn snapshot_dynamics(
    &self,
    _cmd: &mut Self::Cmd,
    dynamics: &Self::Buffer<DynamicBody>,
  ) -> EngineResult<Self::Buffer<DynamicBody>> {
    Ok(CpuDeviceBuffer {
      data: dynamics.data.clone(),
    })
  }

  fn restore_dynamics(
    &self,
    _cmd: &mut Self::Cmd,
    dynamics: &mut Self::Buffer<DynamicBody>,
    snapshot: &Self::Buffer<DynamicBody>,
  ) -> EngineResult<()> {
    dynamics.data = snapshot.data.clone();
    Ok(())
  }

  fn write_back_to_scene(
    &self,
    _cmd: &mut Self::Cmd,
    dynamics: &Self::Buffer<DynamicBody>,
    _physical_scene: &mut PhysicsScene,
    scene: &Scene,
  ) -> EngineResult<()> {
    let mapping = self.dynamic_mapping.read().unwrap();

    scene.query1_mut::<super::components::ParticleSystemComponent, _>(|entity, sys| {
      for (i, dyn_body) in dynamics.data.iter().enumerate() {
        let (e_id, particle_idx) = mapping[i];
        if e_id == entity {
          if let Some(p) = sys.particles.get_mut(particle_idx) {
            p.position = dyn_body.transform.position;
            p.velocity = dyn_body.velocity;
          }
        }
      }
    });

    Ok(())
  }
}
