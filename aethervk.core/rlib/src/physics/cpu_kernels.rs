use crate::gpu::{
  CollisionPair, CommandBuffer, DeviceBuffer, DeviceBvh, DeviceList, DynamicBody, Kernels,
  KinematicBody, WaitHandle,
};
use crate::physics::physics_scene::PhysicsScene;
use crate::scene::Scene;
use crate::types::{EngineError, EngineResult};
use aethervk_oshal_rlib::os::time::timeus_t;
use aethervk_oshal_rlib::math::vector::{Vector, Vector3};
use aethervk_oshal_rlib::math::floating::FloatOps;
use alloc::boxed::Box;
use alloc::vec::Vec;

pub struct CpuCommandBuffer {
  tasks: Vec<Box<dyn FnOnce() + Send + Sync>>,
}

impl CommandBuffer for CpuCommandBuffer {
  fn submit(&mut self) -> EngineResult<()> {
    for task in self.tasks.drain(..) {
      task();
    }
    Ok(())
  }
}

pub struct CpuWaitHandle<T> {
  data: Option<T>,
}

impl<T: Send + Sync> WaitHandle<T> for CpuWaitHandle<T> {
  fn wait(mut self) -> EngineResult<T> {
    self.data.take().ok_or(EngineError::InvalidOperation("WaitHandle already consumed"))
  }
}

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

  fn enqueue_read_to_cpu<'a>(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'a>>
  where
    T: 'a,
  {
    Ok(CpuWaitHandle {
      data: Some(self.data.clone()),
    })
  }
}

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

  fn enqueue_read_to_cpu<'a>(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'a>>
  where
    T: 'a,
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

pub struct CpuMotionBvh {}

impl DeviceBvh for CpuMotionBvh {
  type Cmd = CpuCommandBuffer;
}

pub struct CpuScalarKernels {}

impl Kernels for CpuScalarKernels {
  type Cmd = CpuCommandBuffer;
  type Buffer<T: Copy + Send + Sync> = CpuBuffer<T>;
  type List<T: Copy + Send + Sync> = CpuList<T>;
  type MotionBvh = CpuMotionBvh;

  fn create_command_buffer(&self) -> EngineResult<Self::Cmd> {
    Ok(CpuCommandBuffer { tasks: Vec::new() })
  }

  fn build_kinematic_bodies(
    &self,
    _cmd: &mut Self::Cmd,
    _scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<KinematicBody>> {
    let mut bodies = Vec::new();
    scene0.query2::<crate::scene::TransformComponent, crate::scene::AlmanacPlanet, _>(
      |entity, transform, planet| {
        let parent_id =
          scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let own_id = slotmap::Key::data(&entity).as_ffi() as u32;
        let (frame_type, scale) = scene0.with_component(entity, |f: &crate::scene::ReferenceFrameComponent| (f.frame_type as u32, f.scale)).unwrap_or((crate::scene::ReferenceFrameType::Macro as u32, 1.0));
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: transform.clone(),
          parent_frame_id: parent_id,
          mu: planet.mu,
          own_frame_id: own_id,
          frame_type,
          scale,
        });
      },
    );
    scene0.query2::<crate::scene::TransformComponent, crate::scene::SunComponent, _>(
      |entity, transform, _sun| {
        let parent_id =
          scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let own_id = slotmap::Key::data(&entity).as_ffi() as u32;
        let (frame_type, scale) = scene0.with_component(entity, |f: &crate::scene::ReferenceFrameComponent| (f.frame_type as u32, f.scale)).unwrap_or((crate::scene::ReferenceFrameType::Macro as u32, 1.0));
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: transform.clone(),
          parent_frame_id: parent_id,
          mu: 1.3271244e11, // Example Sun mu
          own_frame_id: own_id,
          frame_type,
          scale,
        });
      },
    );
    Ok(CpuBuffer { data: bodies })
  }

  fn build_dynamic_bodies(
    &self,
    _cmd: &mut Self::Cmd,
    _scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<DynamicBody>> {
    let mut bodies = Vec::new();
    scene0.query2::<crate::scene::TransformComponent, crate::scene::particles::ParticleSystemComponent, _>(
      |entity, transform, sys| {
        let parent_id = scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let particles = sys.particles.read();
        for p in particles.iter().filter(|p| p.active != 0) {
          let mut t = transform.clone();
          t.position = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(p.position);
          bodies.push(DynamicBody {
            entity_id: entity,
            transform: t,
            velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(p.velocity),
            mass: p.mass,
            parent_frame_id: parent_id,
            force: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]),
          });
        }
      }
    );
    Ok(CpuBuffer { data: bodies })
  }

  fn compute_forces(
    &self,
    _cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    dynamics: &mut Self::Buffer<DynamicBody>,
  ) -> EngineResult<()> {
    for dyn_body in dynamics.data.iter_mut() {
      let mut f_grav = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]);
      for kin_body in kinematics.data.iter() {
        if dyn_body.parent_frame_id == kin_body.parent_frame_id {
          let r = kin_body.transform.position - dyn_body.transform.position;
          let dist_sq = aethervk_oshal_rlib::math::vector::Vector::dot(r, r);
          if dist_sq > 1e-6 {
            let dist = dist_sq.sqrt();
            f_grav = f_grav + r * (kin_body.mu * dyn_body.mass / (dist_sq * dist));
          }
        }
      }
      dyn_body.force = f_grav;
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
    let half_dt = dt_sec * 0.5;

    for dyn_body in dynamics.data.iter_mut() {
      if dyn_body.mass > 0.0 {
        let inv_mass = 1.0 / dyn_body.mass;
        // Kick
        dyn_body.velocity = dyn_body.velocity + dyn_body.force * (inv_mass * half_dt);
        // Drift
        dyn_body.transform.position = dyn_body.transform.position + dyn_body.velocity * dt_sec;
      }
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
    Ok(CpuList { data: Vec::new() })
  }

  fn intersect_instances(
    &self,
    _cmd: &mut Self::Cmd,
    _potentials: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList { data: Vec::new() })
  }

  fn compact_collisions(
    &self,
    _cmd: &mut Self::Cmd,
    _globals: &Self::List<CollisionPair>,
    _time_delta: timeus_t,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList { data: Vec::new() })
  }

  fn find_earliest_collision(
    &self,
    _cmd: &mut Self::Cmd,
    _compacted: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::Buffer<timeus_t>> {
    Ok(CpuBuffer {
      data: alloc::vec![timeus_t::MAX],
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
    Ok(CpuBuffer {
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
    // Write the updated positions and velocities back to the particle components.
    scene.query2_mut::<crate::scene::TransformComponent, crate::scene::particles::ParticleSystemComponent, _>(
      |entity, _transform, sys| {
        let mut particles = sys.particles.write();
        let mut p_idx = 0;
        // Optimization: since we map them linearly, we can just consume dynamics sequentially
        // For robustness, we check the entity ID.
        for dyn_body in dynamics.data.iter() {
          if dyn_body.entity_id == entity {
            // Find next active particle
            while p_idx < particles.len() && particles[p_idx].active == 0 {
                p_idx += 1;
            }
            if p_idx < particles.len() {
              particles[p_idx].position = [dyn_body.transform.position.x(), dyn_body.transform.position.y(), dyn_body.transform.position.z()];
              particles[p_idx].velocity = [dyn_body.velocity.x(), dyn_body.velocity.y(), dyn_body.velocity.z()];
              p_idx += 1;
            }
          }
        }
      }
    );
    Ok(())
  }
}

pub struct CpuSimdKernels {
  pub thread_pool: alloc::sync::Arc<aethervk_oshal_rlib::os::pool::ThreadPool>,
}

impl Kernels for CpuSimdKernels {
  type Cmd = CpuCommandBuffer;
  type Buffer<T: Copy + Send + Sync> = CpuBuffer<T>;
  type List<T: Copy + Send + Sync> = CpuList<T>;
  type MotionBvh = CpuMotionBvh;

  fn create_command_buffer(&self) -> EngineResult<Self::Cmd> {
    Ok(CpuCommandBuffer { tasks: Vec::new() })
  }

  fn build_kinematic_bodies(
    &self,
    _cmd: &mut Self::Cmd,
    _scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<KinematicBody>> {
    let mut bodies = Vec::new();
    scene0.query2::<crate::scene::TransformComponent, crate::scene::AlmanacPlanet, _>(
      |entity, transform, planet| {
        let parent_id =
          scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let own_id = slotmap::Key::data(&entity).as_ffi() as u32;
        let (frame_type, scale) = scene0.with_component(entity, |f: &crate::scene::ReferenceFrameComponent| (f.frame_type as u32, f.scale)).unwrap_or((crate::scene::ReferenceFrameType::Macro as u32, 1.0));
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: transform.clone(),
          parent_frame_id: parent_id,
          mu: planet.mu,
          own_frame_id: own_id,
          frame_type,
          scale,
        });
      },
    );
    scene0.query2::<crate::scene::TransformComponent, crate::scene::SunComponent, _>(
      |entity, transform, _sun| {
        let parent_id =
          scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let own_id = slotmap::Key::data(&entity).as_ffi() as u32;
        let (frame_type, scale) = scene0.with_component(entity, |f: &crate::scene::ReferenceFrameComponent| (f.frame_type as u32, f.scale)).unwrap_or((crate::scene::ReferenceFrameType::Macro as u32, 1.0));
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: transform.clone(),
          parent_frame_id: parent_id,
          mu: 1.3271244e11, // Example Sun mu
          own_frame_id: own_id,
          frame_type,
          scale,
        });
      },
    );
    Ok(CpuBuffer { data: bodies })
  }

  fn build_dynamic_bodies(
    &self,
    _cmd: &mut Self::Cmd,
    _scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<DynamicBody>> {
    let mut bodies = Vec::new();
    scene0.query2::<crate::scene::TransformComponent, crate::scene::particles::ParticleSystemComponent, _>(
      |entity, transform, sys| {
        let parent_id = scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let particles = sys.particles.read();
        for p in particles.iter().filter(|p| p.active != 0) {
          let mut t = transform.clone();
          t.position = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(p.position);
          bodies.push(DynamicBody {
            entity_id: entity,
            transform: t,
            velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(p.velocity),
            mass: p.mass,
            parent_frame_id: parent_id,
            force: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]),
          });
        }
      }
    );
    Ok(CpuBuffer { data: bodies })
  }

  fn compute_forces(
    &self,
    _cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    dynamics: &mut Self::Buffer<DynamicBody>,
  ) -> EngineResult<()> {
    if dynamics.data.is_empty() { return Ok(()); }
    
    use aethervk_oshal_rlib::os::pool::chunked::ThreadPoolChunkedExt;
    use crate::scene::{ErasedPtr, ErasedMutPtr};

    let num_particles = dynamics.data.len();
    let chunk_size = 2048;
    let num_chunks = (num_particles + chunk_size - 1) / chunk_size;

    let dyn_ptr = ErasedMutPtr::new(dynamics.data.as_mut_ptr());
    let kin_ptr = ErasedPtr::new(kinematics.data.as_ptr());
    let num_kin = kinematics.data.len();

    let _ = self.thread_pool.spawn_chunked(num_chunks, move |chunk_id| {
      let start = chunk_id * chunk_size;
      let end = (start + chunk_size).min(num_particles);
      
      let dyn_array = unsafe { core::slice::from_raw_parts_mut(dyn_ptr.get::<DynamicBody>(), num_particles) };
      let kin_array = unsafe { core::slice::from_raw_parts(kin_ptr.get::<KinematicBody>(), num_kin) };

      for i in start..end {
        let dyn_body = &mut dyn_array[i];
        let mut f_grav = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]);
        
        let mut parent_scale = 1.0;
        let mut parent_macro_pos = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]);
        for kin_body in kin_array {
            if kin_body.own_frame_id == dyn_body.parent_frame_id {
                parent_scale = kin_body.scale;
                parent_macro_pos = kin_body.transform.position;
            }
        }

        for kin_body in kin_array {
          if dyn_body.parent_frame_id == kin_body.own_frame_id {
            // Local central body gravity
            let r = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]) - dyn_body.transform.position;
            let dist_sq = aethervk_oshal_rlib::math::vector::Vector::dot(r, r);
            if dist_sq > 1e-6 {
              let dist = dist_sq.sqrt();
              let local_mu = if kin_body.frame_type == 1 { kin_body.mu / (parent_scale * parent_scale * parent_scale) } else { kin_body.mu };
              f_grav = f_grav + r * (local_mu * dyn_body.mass / (dist_sq * dist));
            }
          } else if kin_body.frame_type == 0 {
            // Macro body (e.g. Sun) gravity on particle
            if dyn_body.parent_frame_id != kin_body.own_frame_id {
               // Macro body's position in Micro frame
               let macro_pos_in_micro = (kin_body.transform.position - parent_macro_pos) / parent_scale;
               let r = macro_pos_in_micro - dyn_body.transform.position;
               let dist_sq = aethervk_oshal_rlib::math::vector::Vector::dot(r, r);
               if dist_sq > 1e-6 {
                 let dist = dist_sq.sqrt();
                 let local_mu = kin_body.mu / (parent_scale * parent_scale * parent_scale);
                 f_grav = f_grav + r * (local_mu * dyn_body.mass / (dist_sq * dist));
               }
            }
          }
        }
        dyn_body.force = f_grav;
      }
    });

    Ok(())
  }

  fn step_ode(
    &self,
    _cmd: &mut Self::Cmd,
    dynamics: &mut Self::Buffer<DynamicBody>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    if dynamics.data.is_empty() { return Ok(()); }
    
    use aethervk_oshal_rlib::os::pool::chunked::ThreadPoolChunkedExt;
    use crate::scene::ErasedMutPtr;

    let dt_sec = dt as f32 / 1_000_000.0;
    let half_dt = dt_sec * 0.5;

    let num_particles = dynamics.data.len();
    let chunk_size = 2048;
    let num_chunks = (num_particles + chunk_size - 1) / chunk_size;

    let dyn_ptr = ErasedMutPtr::new(dynamics.data.as_mut_ptr());

    let _ = self.thread_pool.spawn_chunked(num_chunks, move |chunk_id| {
      let start = chunk_id * chunk_size;
      let end = (start + chunk_size).min(num_particles);
      
      let dyn_array = unsafe { core::slice::from_raw_parts_mut(dyn_ptr.get::<DynamicBody>(), num_particles) };

      for i in start..end {
        let dyn_body = &mut dyn_array[i];
        if dyn_body.mass > 0.0 {
          let inv_mass = 1.0 / dyn_body.mass;
          // Kick
          dyn_body.velocity = dyn_body.velocity + dyn_body.force * (inv_mass * half_dt);
          // Drift
          dyn_body.transform.position = dyn_body.transform.position + dyn_body.velocity * dt_sec;
        }
      }
    });

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
    Ok(CpuList { data: Vec::new() })
  }

  fn intersect_instances(
    &self,
    _cmd: &mut Self::Cmd,
    _potentials: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList { data: Vec::new() })
  }

  fn compact_collisions(
    &self,
    _cmd: &mut Self::Cmd,
    _globals: &Self::List<CollisionPair>,
    _time_delta: timeus_t,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList { data: Vec::new() })
  }

  fn find_earliest_collision(
    &self,
    _cmd: &mut Self::Cmd,
    _compacted: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::Buffer<timeus_t>> {
    Ok(CpuBuffer {
      data: alloc::vec![timeus_t::MAX],
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
    Ok(CpuBuffer {
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
    // Write the updated positions and velocities back to the particle components.
    scene.query2_mut::<crate::scene::TransformComponent, crate::scene::particles::ParticleSystemComponent, _>(
      |entity, _transform, sys| {
        let mut particles = sys.particles.write();
        let mut p_idx = 0;
        // Optimization: since we map them linearly, we can just consume dynamics sequentially
        // For robustness, we check the entity ID.
        for dyn_body in dynamics.data.iter() {
          if dyn_body.entity_id == entity {
            // Find next active particle
            while p_idx < particles.len() && particles[p_idx].active == 0 {
                p_idx += 1;
            }
            if p_idx < particles.len() {
              particles[p_idx].position = [dyn_body.transform.position.x(), dyn_body.transform.position.y(), dyn_body.transform.position.z()];
              particles[p_idx].velocity = [dyn_body.velocity.x(), dyn_body.velocity.y(), dyn_body.velocity.z()];
              p_idx += 1;
            }
          }
        }
      }
    );
    Ok(())
  }
}
