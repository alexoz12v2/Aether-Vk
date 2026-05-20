//! cpu_kernels module.

use crate::{
  gpu::{
    CollisionPair, CommandBuffer, DeviceBuffer, DeviceBvh, DeviceList, ForceEmitter, Kernels,
    KinematicBody, ParticleGpu, RigidBodyGpu, WaitHandle,
  },
  physics::physics_scene::PhysicsScene,
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
  fn submit(&mut self) -> EngineResult<()> {
    for task in self.tasks.drain(..) {
      task();
    }
    Ok(())
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

  fn enqueue_read_to_cpu<'a>(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'a>>
  where
    T: 'a,
  {
    Ok(CpuWaitHandle {
      data: Some(self.data.clone()),
    })
  }
}

/// TODO: Document this item
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

/// TODO: Document this item
pub struct CpuMotionBvh {
  pub kinematics_copy: Vec<KinematicBody>,
  pub rigid_bodies_copy: Vec<RigidBodyGpu>,
  pub particles_copy: Vec<ParticleGpu>,
  pub bvh_tree: crate::physics::motion_bvh::MotionBvhTree,
}

impl DeviceBvh for CpuMotionBvh {
  type Cmd = CpuCommandBuffer;
}

/// TODO: Document this item
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

    let get_shape_info = |entity| {
      scene0
        .with_component(entity, |c: &crate::scene::ColliderComponent| {
          match c.shape {
            crate::scene::ColliderShape::Sphere { radius } => (0, [radius, 0.0, 0.0]),
            crate::scene::ColliderShape::OBB { half_extents } => {
              (1, [half_extents.x(), half_extents.y(), half_extents.z()])
            }
          }
        })
        .unwrap_or((0, [0.0, 0.0, 0.0]))
    };

    scene0.query2::<crate::scene::TransformComponent, crate::scene::AlmanacPlanet, _>(
      |entity, transform, planet| {
        let t = scene0.global_transform(entity).unwrap_or(transform.clone());
        let vel = scene0
          .with_component(entity, |k: &crate::scene::KinematicComponent| k.velocity)
          .unwrap_or(aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero());
        let parent_id =
          scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let own_id = slotmap::Key::data(&entity).as_ffi() as u32;
        let (frame_type, scale) = scene0
          .with_component(entity, |f: &crate::scene::ReferenceFrameComponent| {
            (f.frame_type as u32, f.scale)
          })
          .unwrap_or((crate::scene::ReferenceFrameType::Macro as u32, 1.0));
        let (shape_type, shape_data) = get_shape_info(entity);
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: t.clone(),
          velocity: vel,
          parent_frame_id: parent_id,
          mu: planet.mu,
          own_frame_id: own_id,
          frame_type,
          scale: scale * t.scale.x(),
          shape_type,
          shape_data,
        });
      },
    );
    scene0.query2::<crate::scene::TransformComponent, crate::scene::SunComponent, _>(
      |entity, transform, _sun| {
        let t = scene0.global_transform(entity).unwrap_or(transform.clone());
        let vel = scene0
          .with_component(entity, |k: &crate::scene::KinematicComponent| k.velocity)
          .unwrap_or(aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero());
        let parent_id =
          scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let own_id = slotmap::Key::data(&entity).as_ffi() as u32;
        let (frame_type, scale) = scene0
          .with_component(entity, |f: &crate::scene::ReferenceFrameComponent| {
            (f.frame_type as u32, f.scale)
          })
          .unwrap_or((crate::scene::ReferenceFrameType::Macro as u32, 1.0));
        let (shape_type, shape_data) = get_shape_info(entity);
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: t.clone(),
          velocity: vel,
          parent_frame_id: parent_id,
          mu: 3.964e-14, // Example Sun mu in AU^3 / s^2
          own_frame_id: own_id,
          frame_type,
          scale: scale * t.scale.x(),
          shape_type,
          shape_data,
        });
      },
    );

    scene0.query2_without::<crate::scene::TransformComponent, crate::scene::KinematicComponent, crate::scene::AlmanacPlanet, _>(
      |entity, transform, kin| {
        if scene0.has_component::<crate::scene::SunComponent>(entity).into() {
          return; // Already processed
        }
        if scene0.has_component::<crate::scene::ColliderComponent>(entity).into() {
          return; // Handled by build_rigid_bodies
        }
        let t = scene0.global_transform(entity).unwrap_or(transform.clone());
        let parent_id =
          scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let own_id = slotmap::Key::data(&entity).as_ffi() as u32;
        let (frame_type, scale) = scene0
          .with_component(entity, |f: &crate::scene::ReferenceFrameComponent| {
            (f.frame_type as u32, f.scale)
          })
          .unwrap_or((crate::scene::ReferenceFrameType::Macro as u32, 1.0));
        let (shape_type, shape_data) = get_shape_info(entity);
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: t.clone(),
          velocity: kin.velocity,
          parent_frame_id: parent_id,
          mu: 0.0,
          own_frame_id: own_id,
          frame_type,
          scale: scale * t.scale.x() * 10.0, // Scale up quad bounds slightly
          shape_type,
          shape_data,
        });
      },
    );

    scene0.query2::<crate::scene::TransformComponent, crate::scene::ReferenceFrameComponent, _>(
      |entity, transform, f| {
        if scene0.has_component::<crate::scene::AlmanacPlanet>(entity).into()
          || scene0.has_component::<crate::scene::SunComponent>(entity).into()
          || scene0.has_component::<crate::scene::KinematicComponent>(entity).into()
          || scene0.has_component::<crate::scene::ColliderComponent>(entity).into()
        {
          return;
        }
        let t = scene0.global_transform(entity).unwrap_or(transform.clone());
        let parent_id =
          scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let own_id = slotmap::Key::data(&entity).as_ffi() as u32;
        let (shape_type, shape_data) = get_shape_info(entity);
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: t.clone(),
          velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero(),
          parent_frame_id: parent_id,
          mu: 0.0,
          own_frame_id: own_id,
          frame_type: f.frame_type as u32,
          scale: f.scale * t.scale.x(),
          shape_type,
          shape_data,
        });
      },
    );

    Ok(CpuBuffer { data: bodies })
  }

  fn build_rigid_bodies(
    &self,
    _cmd: &mut Self::Cmd,
    _scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<RigidBodyGpu>> {
    let mut bodies = Vec::new();

    // Query generic Dynamic Bodies (entities with ColliderComponent but NO ParticleSystemComponent)
    scene0.query2_without::<crate::scene::TransformComponent, crate::scene::ColliderComponent, crate::scene::particles::ParticleSystemComponent, _>(
      |entity, transform, collider| {
        let parent_id = scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let velocity = scene0.with_component(entity, |k: &crate::scene::KinematicComponent| k.velocity)
          .unwrap_or(Vec3f32::zero());
        let angular_velocity = scene0.with_component(entity, |k: &crate::scene::KinematicComponent| k.angular_velocity)
          .unwrap_or(Vec3f32::zero());

        let mass = collider.mass;

        let (shape_type, shape_data, inertia_tensor) = match collider.shape {
          crate::scene::ColliderShape::Sphere { radius } => {
            let i = 0.4 * mass * radius * radius;
            (0, [radius, 0.0, 0.0], [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]])
          }
          crate::scene::ColliderShape::OBB { half_extents } => {
            let dx = half_extents.x() * 2.0;
            let dy = half_extents.y() * 2.0;
            let dz = half_extents.z() * 2.0;
            let ix = (1.0 / 12.0) * mass * (dy * dy + dz * dz);
            let iy = (1.0 / 12.0) * mass * (dx * dx + dz * dz);
            let iz = (1.0 / 12.0) * mass * (dx * dx + dy * dy);
            (1, [half_extents.x(), half_extents.y(), half_extents.z()], [[ix, 0.0, 0.0], [0.0, iy, 0.0], [0.0, 0.0, iz]])
          }
        };

        let rot_mat = Mat4x4f32::from_quat_custom_frame(transform.rotation);
        let rot_arr = [
          [rot_mat.x.x(), rot_mat.x.y(), rot_mat.x.z()],
          [rot_mat.y.x(), rot_mat.y.y(), rot_mat.y.z()],
          [rot_mat.z.x(), rot_mat.z.y(), rot_mat.z.z()],
        ];

        bodies.push(RigidBodyGpu {
          position: [transform.position.x(), transform.position.y(), transform.position.z()],
          mass,
          rotation: rot_arr,
          linear_velocity: [velocity.x(), velocity.y(), velocity.z()],
          _pad0: 0.0,
          angular_velocity: [angular_velocity.x(), angular_velocity.y(), angular_velocity.z()],
          _pad1: 0.0,
          inertia_tensor,
          force: [0.0, 0.0, 0.0],
          torque: [0.0, 0.0, 0.0],
          entity_id: entity,
          parent_frame_id: parent_id,
          shape_type,
          shape_data,
        });
      }
    );

    Ok(CpuBuffer { data: bodies })
  }

  fn build_particles(
    &self,
    _cmd: &mut Self::Cmd,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<ParticleGpu>> {
    let mut bodies = Vec::new();
    scene0.query2::<crate::scene::TransformComponent, crate::scene::particles::ParticleSystemComponent, _>(
      |entity, _transform, sys| {
        let parent_id = scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let particles = sys.particles.read();
        for (i, p) in particles.iter().enumerate().filter(|(_, p)| p.active != 0) {
          bodies.push(ParticleGpu {
            position: p.position,
            velocity: p.velocity,
            mass: p.mass,
            force: [0.0, 0.0, 0.0],
            entity_id: entity,
            parent_frame_id: parent_id,
            original_index: i as u32,
          });
        }
      }
    );
    Ok(CpuBuffer { data: bodies })
  }

  fn build_emitters(
    &self,
    _cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<Self::Buffer<ForceEmitter>> {
    let mut emitters = Vec::new();
    scene.query2::<crate::scene::TransformComponent, crate::scene::ForceEmitterComponent, _>(
      |_, t, emitter| match emitter {
        crate::scene::ForceEmitterComponent::Gravity { mu } => {
          emitters.push(ForceEmitter {
            position: [t.position.x(), t.position.y(), t.position.z()],
            mu: *mu,
            normal: [0.0, 0.0, 0.0],
            type_id: 0,
            trunc_distance: 0.0,
            scale_factor: 1.0,
            _pad: [0, 0],
          });
        }
        crate::scene::ForceEmitterComponent::Planar {
          normal,
          base_force,
          trunc_distance,
        } => {
          emitters.push(ForceEmitter {
            position: [t.position.x(), t.position.y(), t.position.z()],
            mu: *base_force,
            normal: [normal.x(), normal.y(), normal.z()],
            type_id: 1,
            trunc_distance: *trunc_distance,
            scale_factor: 1.0,
            _pad: [0, 0],
          });
        }
      },
    );
    Ok(CpuBuffer { data: emitters })
  }

  fn emit_particles(
    &self,
    _cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<ParticleGpu>,
    physical_scene: &PhysicsScene,
    scene: &Scene,
    sun_pos: Vec3f32,
    dt: timeus_t,
  ) -> EngineResult<()> {
    let dt_us_scaled = dt as i64;
    let emitter_entities =
      scene.query1_res(|id, _: &crate::scene::particles::ParticleEmitterComponent| Some(id));

    for (id, _) in emitter_entities {
      let t = scene.global_transform(id).unwrap_or_default();
      let config = scene.with_component(
        id,
        |c: &crate::scene::particles::ParticleEmitterComponent| c.clone(),
      );
      let mesh = scene.with_component(id, |c: &crate::scene::PhysicalMeshComponent| c.mesh.clone());

      if let (Some(config), Some(mesh)) = (config, mesh) {
        let _ = scene.with_component_mut(
          id,
          |ps: &mut crate::scene::particles::ParticleSystemComponent| {
            let mut sys_particles = ps.particles.write();

            for p in sys_particles.iter_mut() {
              if p.active != 0 {
                let new_age = p.get_age() as i64 + dt_us_scaled;
                p.set_age(new_age as timeus_t);
                if new_age > config.lifetime as i64 {
                  p.active = 0;
                }
              }
            }
            sys_particles.retain(|p| p.active != 0);

            ps.accumulator += dt_us_scaled;
            if ps.accumulator >= config.delta {
              let events = (ps.accumulator / config.delta).min(100);
              ps.accumulator %= config.delta;

              let mut rng = rand::thread_rng();
              let mut u_emission = [0.0; 2];
              u_emission[0] = rand::Rng::r#gen(&mut rng);
              u_emission[1] = rand::Rng::r#gen(&mut rng);

              let burst_size = config.emission_count.mean as usize * events as usize + 1000;
              let mut u_particles = alloc::vec::Vec::with_capacity(burst_size);
              for _ in 0..burst_size {
                u_particles.push([
                  rand::Rng::r#gen(&mut rng),
                  rand::Rng::r#gen(&mut rng),
                  rand::Rng::r#gen(&mut rng),
                  rand::Rng::r#gen(&mut rng),
                ]);
              }

              let uv_grid =
                crate::simulation::comet::uv_grid::UvGrid::new(&mesh.vertices, &mesh.indices, 2);

              let mut temp_ps = crate::scene::particles::ParticleSystemComponent::new(10000);
              for _ in 0..events {
                temp_ps.emit_particles(
                  &config,
                  &mesh,
                  &uv_grid,
                  t.position,
                  t.rotation,
                  t.scale,
                  &u_emission,
                  (&u_particles).as_ref(),
                );
              }

              use crate::physics::physics_scene::math::PhysicsSceneMathExt;
              let parent_id =
                scene.get_parent(id).map(|k| slotmap::Key::data(&k).as_ffi() as u32).unwrap_or(0);
              for p in temp_ps.particles.read().iter() {
                if p.active != 0 {
                  let p_pos = Vec3f32::from_array(p.position);
                  let mut dir = sun_pos - p_pos;
                  let dist_to_sun = dir.length();
                  if dist_to_sun > 1e-5 {
                    dir = dir / dist_to_sun;
                    let ray = crate::math::collision::intersection::Ray {
                      origin: p_pos + dir * 0.1,
                      direction: dir,
                      length: dist_to_sun,
                    };

                    let hits = physical_scene.intersect_world_bvh_math(&ray);
                    if hits.is_empty() {
                      sys_particles.push(p.clone());
                      let new_idx = (sys_particles.len() - 1) as u32;
                      particles.data.push(ParticleGpu {
                        position: p.position,
                        velocity: p.velocity,
                        mass: p.mass,
                        force: [0.0, 0.0, 0.0],
                        entity_id: id,
                        parent_frame_id: parent_id,
                        original_index: new_idx,
                      });
                    }
                  }
                }
              }
            }
          },
        );
      }
    }
    Ok(())
  }

  fn step_ode_p1_p2(
    &self,
    _cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<ParticleGpu>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    // --- PHASE 1 & 2: Particle Explicit Velocity Half-Kick and Drift ---

    let dt_sec = dt as f32 / 1_000_000.0;
    let half_dt = dt_sec * 0.5;

    for p in particles.data.iter_mut() {
      if p.mass > 0.0 {
        let inv_mass = 1.0 / p.mass;
        let v_half_x = p.velocity[0] + p.force[0] * inv_mass * half_dt;
        let v_half_y = p.velocity[1] + p.force[1] * inv_mass * half_dt;
        let v_half_z = p.velocity[2] + p.force[2] * inv_mass * half_dt;

        p.position[0] += v_half_x * half_dt;
        p.position[1] += v_half_y * half_dt;
        p.position[2] += v_half_z * half_dt;

        p.velocity[0] = v_half_x;
        p.velocity[1] = v_half_y;
        p.velocity[2] = v_half_z;
      }
    }

    Ok(())
  }

  fn step_ode_p3_p4(
    &self,
    _cmd: &mut Self::Cmd,
    _kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
    emitters: &Self::Buffer<ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    // --- PHASE 3 & 4: Rigid Body Implicit Midpoint Rule (IMR) solve ---

    let dt_sec = dt as f32 / 1_000_000.0;
    for rb in rigid_bodies.data.iter_mut() {
      if rb.mass <= 0.0 {
        continue;
      }

      let temp_rb = crate::math::physics::RigidBody {
        position: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.position),
        rotation: aethervk_oshal_rlib::math::matrix::mat3::Mat3f32 {
          x: Vec3f32::from_components(rb.rotation[0][0], rb.rotation[0][1], rb.rotation[0][2]),
          y: Vec3f32::from_components(rb.rotation[1][0], rb.rotation[1][1], rb.rotation[1][2]),
          z: Vec3f32::from_components(rb.rotation[2][0], rb.rotation[2][1], rb.rotation[2][2]),
        },
        linear_velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(
          rb.linear_velocity,
        ),
        angular_velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(
          rb.angular_velocity,
        ),
        mass: rb.mass,
        inertia_tensor: aethervk_oshal_rlib::math::matrix::mat3::Mat3f32 {
          x: Vec3f32::from_components(
            rb.inertia_tensor[0][0],
            rb.inertia_tensor[0][1],
            rb.inertia_tensor[0][2],
          ),
          y: Vec3f32::from_components(
            rb.inertia_tensor[1][0],
            rb.inertia_tensor[1][1],
            rb.inertia_tensor[1][2],
          ),
          z: Vec3f32::from_components(
            rb.inertia_tensor[2][0],
            rb.inertia_tensor[2][1],
            rb.inertia_tensor[2][2],
          ),
        },
      };

      let force_eval = |x_mid: Vec3f32,
                        _r_mid: aethervk_oshal_rlib::math::matrix::mat3::Mat3f32|
       -> crate::math::physics::RigidBodyForceEval {
        let mut f_world = Vec3f32::zero();
        let mut k_translation = aethervk_oshal_rlib::math::matrix::mat3::Mat3f32 {
          x: Vec3f32::zero(),
          y: Vec3f32::zero(),
          z: Vec3f32::zero(),
        };

        for em in emitters.data.iter() {
          if em.type_id == 0 {
            let em_pos = Vec3f32::from_array(em.position);
            let r = em_pos - x_mid;
            let dist_sq = r.length_squared() * em.scale_factor * em.scale_factor;
            if dist_sq > 1e-6 {
              let dist = dist_sq.sqrt();
              let dist3 = dist_sq * dist;
              let dist5 = dist3 * dist_sq;

              let coeff = em.mu * temp_rb.mass / dist3;
              f_world += (r / dist) * (em.mu * temp_rb.mass / dist_sq);

              let term1 = aethervk_oshal_rlib::math::matrix::mat3::Mat3f32::identity() * (-coeff);
              let rr_t = aethervk_oshal_rlib::math::matrix::mat3::Mat3f32 {
                x: r * r.x(),
                y: r * r.y(),
                z: r * r.z(),
              };
              let term2 = rr_t * (3.0 * em.mu * temp_rb.mass / dist5);
              k_translation = k_translation + term1 + term2;
            }
          } else if em.type_id == 1 {
            let em_pos = Vec3f32::from_array(em.position);
            let em_norm = Vec3f32::from_array(em.normal);
            let r = x_mid - em_pos;
            let dist = r.dot(em_norm);
            if dist >= 0.0 && dist < em.trunc_distance {
              let denom = 1.0 + dist * dist;
              let force_mag = em.mu / denom;
              f_world += em_norm * force_mag;

              let dF_ddist = -2.0 * em.mu * dist / (denom * denom);
              let nn_t = aethervk_oshal_rlib::math::matrix::mat3::Mat3f32 {
                x: em_norm * em_norm.x(),
                y: em_norm * em_norm.y(),
                z: em_norm * em_norm.z(),
              };
              k_translation = k_translation + nn_t * dF_ddist;
            }
          }
        }

        crate::math::physics::RigidBodyForceEval {
          f_world,
          tau_body: Vec3f32::zero(),
          k_translation,
          p_body: Vec3f32::zero(),
        }
      };

      let (v_mid, w_mid) =
        crate::math::physics::rigid_body_implicit_solve(&temp_rb, dt_sec, force_eval);

      rb.position = [
        rb.position[0] + v_mid.x() * dt_sec,
        rb.position[1] + v_mid.y() * dt_sec,
        rb.position[2] + v_mid.z() * dt_sec,
      ];
      let r_next = temp_rb.rotation * crate::math::expm_hat(w_mid * dt_sec);
      rb.rotation = [
        [r_next.x.x(), r_next.x.y(), r_next.x.z()],
        [r_next.y.x(), r_next.y.y(), r_next.y.z()],
        [r_next.z.x(), r_next.z.y(), r_next.z.z()],
      ];
      let new_v = v_mid * 2.0
        - aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.linear_velocity);
      rb.linear_velocity = [new_v.x(), new_v.y(), new_v.z()];
      let new_w = w_mid * 2.0
        - aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.angular_velocity);
      rb.angular_velocity = [new_w.x(), new_w.y(), new_w.z()];
    }

    Ok(())
  }

  fn compute_self_gravity(
    &self,
    _cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
    particles: &mut Self::Buffer<ParticleGpu>,
  ) -> EngineResult<()> {
    if particles.data.is_empty() {
      return Ok(());
    }

    let g = 1.1906e-19 as f32; // G in AU^3 / (EarthMass * s^2)
    let n = particles.data.len();
    let theta = 0.5;

    for i in 0..n {
      let my_mass = particles.data[i].mass;
      if my_mass <= 0.0 {
        continue;
      }

      let my_pos = Vec3f32::from_array(particles.data[i].position);
      let mut my_global_pos = my_pos;
      let mut local_to_global = aethervk_oshal_rlib::math::matrix::mat3::Mat3f32::identity();
      let mut curr = particles.data[i].parent_frame_id;

      while curr != 0 {
        let mut found = false;
        for kin in bvh.kinematics_copy.iter() {
          if kin.own_frame_id == curr {
            let mat: Mat4x4f32 = kin.transform.to_mat4();
            let pt = mat.mul_vector(my_global_pos.to_point());
            my_global_pos = Vec3f32::from_array([pt.x(), pt.y(), pt.z()]);
            local_to_global = mat.into_linear::<Mat3f32>() * local_to_global;
            curr = kin.parent_frame_id;
            found = true;
            break;
          }
        }
        if !found {
          break;
        }
      }

      let mut total_force = Vec3f32::zero();
      let my_p_id = (1 << 30) | (i as u32);

      if let Some(root_idx) = bvh.bvh_tree.root {
        let mut stack = alloc::vec![root_idx];
        while let Some(node_idx) = stack.pop() {
          let node = &bvh.bvh_tree.nodes[node_idx as usize];
          if let Some(data_idx) = node.data_index {
            if my_p_id != data_idx {
              let r = node.center_of_mass - my_global_pos;
              let dist_sq = r.length_squared();
              if dist_sq > 1e-6 {
                let dist = dist_sq.sqrt();
                total_force += r * (g * my_mass * node.mass / (dist_sq * dist));
              }
            }
          } else {
            let r = node.center_of_mass - my_global_pos;
            let dist_sq = r.length_squared();
            let dist = dist_sq.max(1e-6).sqrt();
            let extents = node.bounds.max - node.bounds.min;
            let size = extents.x().max(extents.y()).max(extents.z());

            if size / dist < theta {
              total_force += r * (g * my_mass * node.mass / (dist_sq * dist));
            } else {
              if let Some(left) = node.left_child {
                stack.push(left);
              }
              if let Some(right) = node.right_child {
                stack.push(right);
              }
            }
          }
        }
      }

      let global_to_local = local_to_global
        .inverse()
        .unwrap_or(aethervk_oshal_rlib::math::matrix::mat3::Mat3f32::identity());
      let local_force = global_to_local.mul_vector(total_force);

      particles.data[i].force[0] += local_force.x();
      particles.data[i].force[1] += local_force.y();
      particles.data[i].force[2] += local_force.z();
    }

    Ok(())
  }

  fn step_ode_p5(
    &self,
    _cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    particles: &mut Self::Buffer<ParticleGpu>,
    emitters: &Self::Buffer<ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    use aethervk_oshal_rlib::math::matrix::MatrixVectorMul;

    let dt_sec = dt as f32 / 1_000_000.0;
    let half_dt = dt_sec * 0.5;

    let mut frames_map = hashbrown::HashMap::new();
    for kin in kinematics.data.iter() {
      frames_map.insert(
        kin.own_frame_id,
        (kin.parent_frame_id, kin.transform.to_mat4::<Mat4x4f32>()),
      );
    }

    let get_global_pos = |frame_id: u32, local_pos: Vec3f32| -> Vec3f32 {
      let mut pos = local_pos;
      let mut curr = frame_id;
      while let Some(&(parent, ref mat)) = frames_map.get(&curr) {
        let pt = mat.mul_vector(pos.to_point());
        pos = Vec3f32::from_array([pt.x(), pt.y(), pt.z()]);
        curr = parent;
        if curr == 0 {
          break;
        }
      }
      pos
    };

    let get_global_to_local = |frame_id: u32| -> aethervk_oshal_rlib::math::matrix::mat3::Mat3f32 {
      let mut local_to_global = aethervk_oshal_rlib::math::matrix::mat3::Mat3f32::identity();
      let mut curr = frame_id;
      while let Some(&(parent, ref mat)) = frames_map.get(&curr) {
        local_to_global =
          mat.into_linear::<aethervk_oshal_rlib::math::matrix::mat3::Mat3f32>() * local_to_global;
        curr = parent;
        if curr == 0 {
          break;
        }
      }
      local_to_global
        .inverse()
        .unwrap_or(aethervk_oshal_rlib::math::matrix::mat3::Mat3f32::identity())
    };

    for p in particles.data.iter_mut() {
      if p.mass <= 0.0 {
        continue;
      }
      let inv_mass = 1.0 / p.mass;

      p.position[0] += p.velocity[0] * half_dt;
      p.position[1] += p.velocity[1] * half_dt;
      p.position[2] += p.velocity[2] * half_dt;

      let mut f_next = Vec3f32::zero();
      let p_pos = Vec3f32::from_array(p.position);

      for em in emitters.data.iter() {
        if em.type_id == 0 {
          let em_pos = Vec3f32::from_array(em.position);
          let r = em_pos - p_pos;
          let scaled_dist_sq = r.length_squared() * em.scale_factor * em.scale_factor;
          if scaled_dist_sq > 1e-6 {
            let force_dir = r / r.length();
            f_next += force_dir * (em.mu * p.mass / scaled_dist_sq);
          }
        } else if em.type_id == 1 {
          let em_pos = Vec3f32::from_array(em.position);
          let em_norm = Vec3f32::from_array(em.normal);
          let r = p_pos - em_pos;
          let dist = r.dot(em_norm);
          if dist >= 0.0 && dist < em.trunc_distance {
            let force_mag = em.mu / (1.0 + dist * dist);
            f_next += em_norm * force_mag;
          }
        }
      }

      // FIX: Robust Gravity Evaluation in Global Space
      let p_global_pos = get_global_pos(p.parent_frame_id, p_pos);
      let mut global_f_next = Vec3f32::zero();

      for kin in kinematics.data.iter() {
        if kin.mu > 0.0 {
          let kin_global_pos = get_global_pos(kin.parent_frame_id, kin.transform.position);
          let r = kin_global_pos - p_global_pos;
          let dist_sq = r.length_squared();
          if dist_sq > 1e-6 {
            let dist = dist_sq.sqrt();
            global_f_next += r * (kin.mu * p.mass / (dist_sq * dist));
          }
        }
      }

      // Automatically scales local force bounds appropriately
      let global_to_local = get_global_to_local(p.parent_frame_id);
      let local_gravity = global_to_local.mul_vector(global_f_next);
      f_next += local_gravity;

      p.force[0] = f_next.x();
      p.force[1] = f_next.y();
      p.force[2] = f_next.z();

      p.velocity[0] += p.force[0] * inv_mass * half_dt;
      p.velocity[1] += p.force[1] * inv_mass * half_dt;
      p.velocity[2] += p.force[2] * inv_mass * half_dt;
    }

    Ok(())
  }

  fn build_motion_bvh(
    &self,
    _cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &Self::Buffer<RigidBodyGpu>,
    particles: &Self::Buffer<ParticleGpu>,
    _dt: timeus_t,
  ) -> EngineResult<Self::MotionBvh> {
    use crate::physics::motion_bvh::{Aabb, CpuBvhItem, MotionBvhTree};

    let mut frames_map = hashbrown::HashMap::new();
    let mut frame_types = hashbrown::HashMap::new();
    for kin in kinematics.data.iter() {
      let mat: Mat4x4f32 = kin.transform.to_mat4();
      frames_map.insert(kin.own_frame_id, (kin.parent_frame_id, mat));
      frame_types.insert(kin.own_frame_id, kin.frame_type);
    }

    let get_global_pos = |frame_id: u32, local_pos: Vec3f32| -> Vec3f32 {
      let mut pos = local_pos;
      let mut curr = frame_id;
      while let Some(&(parent, ref transform)) = frames_map.get(&curr) {
        let pt = transform.mul_vector(pos.to_point());
        pos = Vec3f32::from_array([pt.x(), pt.y(), pt.z()]);
        curr = parent;
        if curr == 0 {
          break;
        }
      }
      pos
    };

    let mut flat_items: alloc::vec::Vec<(Aabb, CpuBvhItem)> = alloc::vec::Vec::new();

    for (i, kin) in kinematics.data.iter().enumerate() {
      if kin.shape_data[0] == 0.0 && kin.shape_data[1] == 0.0 && kin.shape_data[2] == 0.0 {
        continue; // Don't add mathematical frames to the collision tree!
      }
      let global_pos = get_global_pos(kin.parent_frame_id, kin.transform.position);
      let max_travel = kin.velocity.length();
      let r = kin.scale + max_travel;
      let bounds = Aabb::new(
        global_pos - Vec3f32::from_array([r, r, r]),
        global_pos + Vec3f32::from_array([r, r, r]),
      );
      // data_index: top bit 1 means kinematic, lower 31 bits are index
      flat_items.push((
        bounds,
        CpuBvhItem::Primitive((1 << 31) | (i as u32), kin.mu, global_pos),
      ));
    }

    for (i, dyn_body) in rigid_bodies.data.iter().enumerate() {
      let global_pos = get_global_pos(
        dyn_body.parent_frame_id,
        aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(dyn_body.position),
      );
      let max_travel =
        aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(dyn_body.linear_velocity)
          .length();
      let r = dyn_body.shape_data[0] + max_travel;
      let bounds = Aabb::new(
        global_pos - Vec3f32::from_array([r, r, r]),
        global_pos + Vec3f32::from_array([r, r, r]),
      );
      // data_index: top bit 0 means dynamic, lower 31 bits are index
      flat_items.push((
        bounds,
        CpuBvhItem::Primitive(i as u32, dyn_body.mass, global_pos),
      ));
    }

    for (i, p) in particles.data.iter().enumerate() {
      let global_pos = get_global_pos(
        p.parent_frame_id,
        aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(p.position),
      );
      let max_travel =
        aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(p.velocity).length();
      let r = 1.0 + max_travel; // Particles default radius 1.0
      let bounds = Aabb::new(
        global_pos - Vec3f32::from_array([r, r, r]),
        global_pos + Vec3f32::from_array([r, r, r]),
      );
      // data_index: second top bit 1 means particle, lower 30 bits are index
      flat_items.push((
        bounds,
        CpuBvhItem::Primitive((1 << 30) | (i as u32), p.mass, global_pos),
      ));
    }

    let mut nodes = alloc::vec::Vec::new();
    let tlas_root = MotionBvhTree::build_into(&mut flat_items, &mut nodes);

    let bvh_tree = MotionBvhTree {
      nodes,
      root: tlas_root,
    };

    Ok(CpuMotionBvh {
      kinematics_copy: kinematics.data.clone(),
      rigid_bodies_copy: rigid_bodies.data.clone(),
      particles_copy: particles.data.clone(),
      bvh_tree,
    })
  }

  fn self_intersect_scene(
    &self,
    _cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
  ) -> EngineResult<Self::List<CollisionPair>> {
    let pairs = crate::physics::collision_pipeline::detect_collisions_cpu(bvh);
    Ok(CpuList { data: pairs })
  }

  fn intersect_instances(
    &self,
    _cmd: &mut Self::Cmd,
    potentials: &Self::List<CollisionPair>,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &Self::Buffer<RigidBodyGpu>,
    particles: &Self::Buffer<ParticleGpu>,
  ) -> EngineResult<Self::List<CollisionPair>> {
    use aethervk_oshal_rlib::math::{
      matrix::{MatrixVectorMul, SquareMatrix},
      vector::Vector,
    };
    let mut actual_collisions = alloc::vec::Vec::new();

    let mut frames_map = hashbrown::HashMap::new();
    for kin in kinematics.data.iter() {
      frames_map.insert(
        kin.own_frame_id,
        (kin.parent_frame_id, kin.transform.to_mat4::<Mat4x4f32>()),
      );
    }

    enum DynamicShape {
      Sphere(crate::math::collision::gjk::GjkSphere),
      Obb(crate::math::collision::bounds::OBB<f32>),
    }
    impl crate::math::collision::gjk::Support for DynamicShape {
      fn support(&self, dir: Vec3f32) -> Vec3f32 {
        match self {
          Self::Sphere(s) => crate::math::collision::gjk::Support::support(s, dir),
          Self::Obb(o) => crate::math::collision::gjk::Support::support(o, dir),
        }
      }
    }

    let get_properties = |idx: u32| -> Option<(u32, u32, [f32; 3], Mat4x4f32)> {
      if (idx & (1 << 31)) != 0 {
        let i = (idx & !(1 << 31)) as usize;
        if i >= kinematics.data.len() {
          return None;
        }
        let k = &kinematics.data[i];
        Some((
          k.parent_frame_id,
          k.shape_type,
          k.shape_data,
          k.transform.to_mat4(),
        ))
      } else if (idx & (1 << 30)) != 0 {
        let i = (idx & !(1 << 30)) as usize;
        if i >= particles.data.len() {
          return None;
        }
        let p = &particles.data[i];
        Some((
          p.parent_frame_id,
          0,
          [1.0, 0.0, 0.0],
          Mat4x4f32::translation(Vec3f32::from_array(p.position)),
        ))
      } else {
        let i = idx as usize;
        if i >= rigid_bodies.data.len() {
          return None;
        }
        let rb = &rigid_bodies.data[i];
        let rot = rb.rotation;
        let mat = Mat4x4f32::from_columns(
          Vec4f32::from_components(rot[0][0], rot[0][1], rot[0][2], 0.0),
          Vec4f32::from_components(rot[1][0], rot[1][1], rot[1][2], 0.0),
          Vec4f32::from_components(rot[2][0], rot[2][1], rot[2][2], 0.0),
          Vec4f32::from_components(rb.position[0], rb.position[1], rb.position[2], 1.0),
        );
        Some((rb.parent_frame_id, rb.shape_type, rb.shape_data, mat))
      }
    };

    let get_shape = |shape_type: u32, shape_data: [f32; 3], mat: Mat4x4f32| -> DynamicShape {
      if shape_type == 1 {
        let mut x = Vec3f32::from_components(
          mat.component(0).unwrap(),
          mat.component(1).unwrap(),
          mat.component(2).unwrap(),
        );
        let mut y = Vec3f32::from_components(
          mat.component(4).unwrap(),
          mat.component(5).unwrap(),
          mat.component(6).unwrap(),
        );
        let mut z = Vec3f32::from_components(
          mat.component(8).unwrap(),
          mat.component(9).unwrap(),
          mat.component(10).unwrap(),
        );

        let scale_x = x.length();
        let scale_y = y.length();
        let scale_z = z.length();

        if scale_x > 1e-6 {
          x = x / scale_x;
        }
        if scale_y > 1e-6 {
          y = y / scale_y;
        }
        if scale_z > 1e-6 {
          z = z / scale_z;
        }

        let rot3 = Mat3f32 { x, y, z };
        let pos = Vec3f32::from_components(
          mat.component(12).unwrap(),
          mat.component(13).unwrap(),
          mat.component(14).unwrap(),
        );
        let scaled_extents = [
          shape_data[0] * scale_x,
          shape_data[1] * scale_y,
          shape_data[2] * scale_z,
        ];
        DynamicShape::Obb(crate::math::collision::bounds::OBB::new(
          pos,
          rot3,
          Vec3f32::from_array(scaled_extents),
        ))
      } else {
        let x = Vec3f32::from_components(
          mat.component(0).unwrap(),
          mat.component(1).unwrap(),
          mat.component(2).unwrap(),
        );
        let scale_x = x.length(); // assuming uniform scale
        let pos = Vec3f32::from_components(
          mat.component(12).unwrap(),
          mat.component(13).unwrap(),
          mat.component(14).unwrap(),
        );
        DynamicShape::Sphere(crate::math::collision::gjk::GjkSphere {
          center: pos,
          radius: shape_data[0] * scale_x,
        })
      }
    };

    for pair in &potentials.data {
      let idx_a = pair.a.primitive_index;
      let idx_b = pair.b.primitive_index;

      let props_a = get_properties(idx_a);
      let props_b = get_properties(idx_b);

      if let (Some((parent_a, type_a, data_a, mat_a)), Some((parent_b, type_b, data_b, mat_b))) =
        (props_a, props_b)
      {
        let mat_a_transformed = if parent_a != parent_b {
          let mut m_a_lca = mat_a;
          let mut curr = parent_a;
          while curr != 0 {
            if let Some(&(p, ref t)) = frames_map.get(&curr) {
              m_a_lca = *t * m_a_lca;
              curr = p;
            } else {
              break;
            }
          }
          m_a_lca
        } else {
          mat_a
        };
        let mat_b_transformed = if parent_a != parent_b {
          let mut m_b_lca = mat_b;
          let mut curr = parent_b;
          while curr != 0 {
            if let Some(&(p, ref t)) = frames_map.get(&curr) {
              m_b_lca = *t * m_b_lca;
              curr = p;
            } else {
              break;
            }
          }
          m_b_lca
        } else {
          mat_b
        };

        let shape_a = get_shape(type_a, data_a, mat_a_transformed);
        let shape_b = get_shape(type_b, data_b, mat_b_transformed);

        let center_a = match &shape_a {
          DynamicShape::Sphere(s) => s.center,
          DynamicShape::Obb(o) => o.translation(),
        };
        let center_b = match &shape_b {
          DynamicShape::Sphere(s) => s.center,
          DynamicShape::Obb(o) => o.translation(),
        };

        #[cfg(test)]
        aethervk_oshal_rlib::log!(
          "GJK evaluating: idx_a={}, idx_b={}, A={:?}, B={:?}",
          idx_a,
          idx_b,
          center_a,
          center_b
        );

        let (dist, pt_a, pt_b) = crate::math::collision::gjk::gjk_distance(&shape_a, &shape_b);

        #[cfg(test)]
        aethervk_oshal_rlib::log!("GJK evaluated -> dist: {}", dist);

        if dist <= 0.0 {
          let mut new_pair = pair.clone();
          new_pair.penetration_depth = if dist < 0.0 { -dist } else { 0.0 };
          let mut normal = pt_a - pt_b;
          if dist < 0.0 {
            normal = -normal;
          }
          let len = normal.length();
          if len > 1e-6 {
            normal = normal / len;
          } else {
            let diff = center_a - center_b;
            let diff_len = diff.length();
            if diff_len > 1e-6 {
              normal = diff / diff_len;
            } else {
              normal =
                aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([1.0, 0.0, 0.0]);
            }
          }
          new_pair.contact_normal = [normal.x(), normal.y(), normal.z()];
          new_pair.contact_point = [
            (pt_a.x() + pt_b.x()) * 0.5,
            (pt_a.y() + pt_b.y()) * 0.5,
            (pt_a.z() + pt_b.z()) * 0.5,
          ];
          #[cfg(test)]
          aethervk_oshal_rlib::log!(
            "COLLISION DETECTED: idx_a={}, idx_b={}, dist={}, normal={:?}",
            idx_a,
            idx_b,
            dist,
            new_pair.contact_normal
          );
          actual_collisions.push(new_pair);
        }
      }
    }

    Ok(CpuList {
      data: actual_collisions,
    })
  }

  fn compact_collisions(
    &self,
    _cmd: &mut Self::Cmd,
    globals: &Self::List<CollisionPair>,
    _time_delta: timeus_t,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList {
      data: globals.data.clone(),
    })
  }

  fn find_earliest_collision(
    &self,
    _cmd: &mut Self::Cmd,
    compacted: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::Buffer<timeus_t>> {
    let mut min_tc = timeus_t::MAX;
    for pair in &compacted.data {
      if (pair.time_of_impact as timeus_t) < min_tc {
        min_tc = pair.time_of_impact as timeus_t;
      }
    }
    Ok(CpuBuffer {
      data: alloc::vec![min_tc],
    })
  }

  fn apply_collision_responses(
    &self,
    _cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
    particles: &mut Self::Buffer<ParticleGpu>,
    collisions: &Self::List<CollisionPair>,
    force_inelastic: bool,
  ) -> EngineResult<()> {
    if collisions.data.is_empty() {
      return Ok(());
    }

    #[cfg(test)]
    aethervk_oshal_rlib::log!(
      "apply_collision_responses: rigid_bodies.len={}, particles.len={}",
      rigid_bodies.data.len(),
      particles.data.len()
    );

    if rigid_bodies.data.len() > 0 {
      #[cfg(test)]
      aethervk_oshal_rlib::log!("rb 0 mass={}", rigid_bodies.data[0].mass);
    }

    let clusters =
      crate::physics::cpu_kernels::group_and_cluster_collisions(collisions.data.clone(), 0.01);
    let restitution = if force_inelastic { 0.0 } else { 0.5 };

    let dyn_array = rigid_bodies.data.as_mut_slice();
    let p_array = particles.data.as_mut_slice();

    for cluster in clusters {
      crate::physics::lcp_integration::resolve_cluster_lcp(
        &cluster,
        &kinematics.data,
        dyn_array,
        p_array,
        restitution,
      );
    }
    Ok(())
  }

  fn snapshot_dynamics(
    &self,
    _cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<RigidBodyGpu>,
    particles: &Self::Buffer<ParticleGpu>,
  ) -> EngineResult<(Self::Buffer<RigidBodyGpu>, Self::Buffer<ParticleGpu>)> {
    Ok((
      CpuBuffer {
        data: rigid_bodies.data.clone(),
      },
      CpuBuffer {
        data: particles.data.clone(),
      },
    ))
  }

  fn restore_dynamics(
    &self,
    _cmd: &mut Self::Cmd,
    rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
    particles: &mut Self::Buffer<ParticleGpu>,
    snapshot: &(Self::Buffer<RigidBodyGpu>, Self::Buffer<ParticleGpu>),
  ) -> EngineResult<()> {
    rigid_bodies.data = snapshot.0.data.clone();
    particles.data = snapshot.1.data.clone();
    Ok(())
  }

  fn write_back_to_scene(
    &self,
    _cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<RigidBodyGpu>,
    particles: &Self::Buffer<ParticleGpu>,
    _physical_scene: &mut PhysicsScene,
    scene: &Scene,
  ) -> EngineResult<()> {
    // Write the updated positions and velocities back to the particle components.
    scene.query2::<crate::scene::TransformComponent, crate::scene::particles::ParticleSystemComponent, _>(
      |entity, _transform, sys| {
        let mut sys_particles = sys.particles.write();
        for p_gpu in particles.data.iter() {
          if p_gpu.entity_id == entity {
            if (p_gpu.original_index as usize) < sys_particles.len() {
              sys_particles[p_gpu.original_index as usize].position = p_gpu.position;
              sys_particles[p_gpu.original_index as usize].velocity = p_gpu.velocity;
            }
          }
        }
      }
    );
    // Write back rigid bodies
    for rb in rigid_bodies.data.iter() {
      // TODO set error handling
      let _ = scene.with_component_mut(rb.entity_id, |trans: &mut TransformComponent| {
        trans.position = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.position);
        let mat = Mat4x4f32::from_columns(
          Vec4f32::from_components(rb.rotation[0][0], rb.rotation[0][1], rb.rotation[0][2], 0.0),
          Vec4f32::from_components(rb.rotation[1][0], rb.rotation[1][1], rb.rotation[1][2], 0.0),
          Vec4f32::from_components(rb.rotation[2][0], rb.rotation[2][1], rb.rotation[2][2], 0.0),
          Vec4f32::from_components(0.0, 0.0, 0.0, 1.0),
        );
        trans.rotation = Quat::from_mat4(&mat);
      });

      let _ = scene.with_component_mut(rb.entity_id, |kin: &mut KinematicComponent| {
        kin.velocity =
          aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.linear_velocity);
        kin.angular_velocity =
          aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.angular_velocity);
      });
    }
    Ok(())
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

    let get_shape_info = |entity| {
      scene0
        .with_component(entity, |c: &crate::scene::ColliderComponent| {
          match c.shape {
            crate::scene::ColliderShape::Sphere { radius } => (0, [radius, 0.0, 0.0]),
            crate::scene::ColliderShape::OBB { half_extents } => {
              (1, [half_extents.x(), half_extents.y(), half_extents.z()])
            }
          }
        })
        .unwrap_or((0, [0.0, 0.0, 0.0]))
    };

    scene0.query2::<crate::scene::TransformComponent, crate::scene::AlmanacPlanet, _>(
      |entity, transform, planet| {
        let t = scene0.global_transform(entity).unwrap_or(transform.clone());
        let vel = scene0
          .with_component(entity, |k: &crate::scene::KinematicComponent| k.velocity)
          .unwrap_or(aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero());
        let parent_id =
          scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let own_id = slotmap::Key::data(&entity).as_ffi() as u32;
        let (frame_type, scale) = scene0
          .with_component(entity, |f: &crate::scene::ReferenceFrameComponent| {
            (f.frame_type as u32, f.scale)
          })
          .unwrap_or((crate::scene::ReferenceFrameType::Macro as u32, 1.0));
        let (shape_type, shape_data) = get_shape_info(entity);
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: t.clone(),
          velocity: vel,
          parent_frame_id: parent_id,
          mu: planet.mu,
          own_frame_id: own_id,
          frame_type,
          scale: scale * t.scale.x(),
          shape_type,
          shape_data,
        });
      },
    );
    scene0.query2::<crate::scene::TransformComponent, crate::scene::SunComponent, _>(
      |entity, transform, _sun| {
        let t = scene0.global_transform(entity).unwrap_or(transform.clone());
        let vel = scene0
          .with_component(entity, |k: &crate::scene::KinematicComponent| k.velocity)
          .unwrap_or(aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero());
        let parent_id =
          scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let own_id = slotmap::Key::data(&entity).as_ffi() as u32;
        let (frame_type, scale) = scene0
          .with_component(entity, |f: &crate::scene::ReferenceFrameComponent| {
            (f.frame_type as u32, f.scale)
          })
          .unwrap_or((crate::scene::ReferenceFrameType::Macro as u32, 1.0));
        let (shape_type, shape_data) = get_shape_info(entity);
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: t.clone(),
          velocity: vel,
          parent_frame_id: parent_id,
          mu: 3.964e-14, // Example Sun mu in AU^3 / s^2
          own_frame_id: own_id,
          frame_type,
          scale: scale * t.scale.x(),
          shape_type,
          shape_data,
        });
      },
    );

    scene0.query2_without::<crate::scene::TransformComponent, crate::scene::KinematicComponent, crate::scene::AlmanacPlanet, _>(
      |entity, transform, kin| {
        if scene0.has_component::<crate::scene::SunComponent>(entity).into() {
          return; // Already processed
        }
        if scene0.has_component::<crate::scene::ColliderComponent>(entity).into() {
          return; // Handled by build_rigid_bodies
        }
        let t = scene0.global_transform(entity).unwrap_or(transform.clone());
        let parent_id =
          scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let own_id = slotmap::Key::data(&entity).as_ffi() as u32;
        let (frame_type, scale) = scene0
          .with_component(entity, |f: &crate::scene::ReferenceFrameComponent| {
            (f.frame_type as u32, f.scale)
          })
          .unwrap_or((crate::scene::ReferenceFrameType::Macro as u32, 1.0));
        let (shape_type, shape_data) = get_shape_info(entity);
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: t.clone(),
          velocity: kin.velocity,
          parent_frame_id: parent_id,
          mu: 0.0,
          own_frame_id: own_id,
          frame_type,
          scale: scale * t.scale.x() * 10.0, // Scale up quad bounds slightly
          shape_type,
          shape_data,
        });
      },
    );

    scene0.query2::<crate::scene::TransformComponent, crate::scene::ReferenceFrameComponent, _>(
      |entity, transform, f| {
        if scene0.has_component::<crate::scene::AlmanacPlanet>(entity).into()
          || scene0.has_component::<crate::scene::SunComponent>(entity).into()
          || scene0.has_component::<crate::scene::KinematicComponent>(entity).into()
          || scene0.has_component::<crate::scene::ColliderComponent>(entity).into()
        {
          return;
        }
        let t = scene0.global_transform(entity).unwrap_or(transform.clone());
        let parent_id =
          scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
        let own_id = slotmap::Key::data(&entity).as_ffi() as u32;
        let (shape_type, shape_data) = get_shape_info(entity);
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: t.clone(),
          velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero(),
          parent_frame_id: parent_id,
          mu: 0.0,
          own_frame_id: own_id,
          frame_type: f.frame_type as u32,
          scale: f.scale * t.scale.x(),
          shape_type,
          shape_data,
        });
      },
    );

    Ok(CpuBuffer { data: bodies })
  }

  fn build_rigid_bodies(
    &self,
    _cmd: &mut Self::Cmd,
    _scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<RigidBodyGpu>> {
    let mut bodies = Vec::new();

    // Query generic Dynamic Bodies (entities with ColliderComponent but NO ParticleSystemComponent)
    scene0.query2_without::<crate::scene::TransformComponent, crate::scene::ColliderComponent, crate::scene::particles::ParticleSystemComponent, _>(
        |entity, transform, collider| {
          let parent_id = scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
          let velocity = scene0.with_component(entity, |k: &crate::scene::KinematicComponent| k.velocity)
            .unwrap_or(Vec3f32::zero());
          let angular_velocity = scene0.with_component(entity, |k: &crate::scene::KinematicComponent| k.angular_velocity)
            .unwrap_or(Vec3f32::zero());

          let mass = collider.mass;

          let (shape_type, shape_data, inertia_tensor) = match collider.shape {
            crate::scene::ColliderShape::Sphere { radius } => {
              let i = 0.4 * mass * radius * radius;
              (0, [radius, 0.0, 0.0], [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]])
            }
            crate::scene::ColliderShape::OBB { half_extents } => {
              let dx = half_extents.x() * 2.0;
              let dy = half_extents.y() * 2.0;
              let dz = half_extents.z() * 2.0;
              let ix = (1.0 / 12.0) * mass * (dy * dy + dz * dz);
              let iy = (1.0 / 12.0) * mass * (dx * dx + dz * dz);
              let iz = (1.0 / 12.0) * mass * (dx * dx + dy * dy);
              (1, [half_extents.x(), half_extents.y(), half_extents.z()], [[ix, 0.0, 0.0], [0.0, iy, 0.0], [0.0, 0.0, iz]])
            }
          };

          let rot_mat = Mat4x4f32::from_quat_custom_frame(transform.rotation);
          #[rustfmt::skip]
          let rot_arr = [
            [rot_mat.component(0).unwrap(), rot_mat.component(1).unwrap(), rot_mat.component(2).unwrap()],
            [rot_mat.component(4).unwrap(), rot_mat.component(5).unwrap(), rot_mat.component(6).unwrap()],
            [rot_mat.component(8).unwrap(), rot_mat.component(9).unwrap(), rot_mat.component(10).unwrap()],
          ];

          bodies.push(RigidBodyGpu {
            position: [transform.position.x(), transform.position.y(), transform.position.z()],
            mass,
            rotation: rot_arr,
            linear_velocity: [velocity.x(), velocity.y(), velocity.z()],
            _pad0: 0.0,
            angular_velocity: [angular_velocity.x(), angular_velocity.y(), angular_velocity.z()],
            _pad1: 0.0,
            inertia_tensor,
            force: [0.0, 0.0, 0.0],
            torque: [0.0, 0.0, 0.0],
            entity_id: entity,
            parent_frame_id: parent_id,
            shape_type,
            shape_data,
          });
        }
      );

    Ok(CpuBuffer { data: bodies })
  }

  fn build_particles(
    &self,
    _cmd: &mut Self::Cmd,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<ParticleGpu>> {
    let mut bodies = Vec::new();
    scene0.query2::<crate::scene::TransformComponent, crate::scene::particles::ParticleSystemComponent, _>(
        |entity, transform, sys| {
          let parent_id = scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
          let particles = sys.particles.read();
          for (i, p) in particles.iter().enumerate().filter(|(_, p)| p.active != 0) {
            bodies.push(ParticleGpu {
              position: p.position,
              velocity: p.velocity,
              mass: p.mass,
              force: [0.0, 0.0, 0.0],
              entity_id: entity,
              parent_frame_id: parent_id,
              original_index: i as u32,
            });
          }
        }
      );
    Ok(CpuBuffer { data: bodies })
  }

  fn build_emitters(
    &self,
    _cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<Self::Buffer<ForceEmitter>> {
    let mut emitters = Vec::new();
    scene.query2::<crate::scene::TransformComponent, crate::scene::ForceEmitterComponent, _>(
      |_, t, emitter| match emitter {
        crate::scene::ForceEmitterComponent::Gravity { mu } => {
          emitters.push(ForceEmitter {
            position: [t.position.x(), t.position.y(), t.position.z()],
            mu: *mu,
            normal: [0.0, 0.0, 0.0],
            type_id: 0,
            trunc_distance: 0.0,
            scale_factor: 1.0,
            _pad: [0, 0],
          });
        }
        crate::scene::ForceEmitterComponent::Planar {
          normal,
          base_force,
          trunc_distance,
        } => {
          emitters.push(ForceEmitter {
            position: [t.position.x(), t.position.y(), t.position.z()],
            mu: *base_force,
            normal: [normal.x(), normal.y(), normal.z()],
            type_id: 1,
            trunc_distance: *trunc_distance,
            scale_factor: 1.0,
            _pad: [0, 0],
          });
        }
      },
    );
    Ok(CpuBuffer { data: emitters })
  }

  fn emit_particles(
    &self,
    _cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<ParticleGpu>,
    physical_scene: &PhysicsScene,
    scene: &Scene,
    sun_pos: Vec3f32,
    dt: timeus_t,
  ) -> EngineResult<()> {
    let dt_us_scaled = dt as i64;
    let emitter_entities =
      scene.query1_res(|id, _: &crate::scene::particles::ParticleEmitterComponent| Some(id));

    for (id, _) in emitter_entities {
      let t = scene.global_transform(id).unwrap_or_default();
      let config = scene.with_component(
        id,
        |c: &crate::scene::particles::ParticleEmitterComponent| c.clone(),
      );
      let mesh = scene.with_component(id, |c: &crate::scene::PhysicalMeshComponent| c.mesh.clone());

      if let (Some(config), Some(mesh)) = (config, mesh) {
        let res = scene.with_component_mut(
          id,
          |ps: &mut crate::scene::particles::ParticleSystemComponent| -> EngineResult<()> {
            let mut sys_particles = ps.particles.write();

            let orig_accumulator = ps.accumulator;
            let orig_particles = sys_particles.clone();

            for p in sys_particles.iter_mut() {
              if p.active != 0 {
                let new_age = p.get_age() as i64 + dt_us_scaled;
                p.set_age(new_age as timeus_t);
                if new_age > config.lifetime as i64 {
                  p.active = 0;
                }
              }
            }
            sys_particles.retain(|p| p.active != 0);

            ps.accumulator += dt_us_scaled;
            if ps.accumulator >= config.delta {
              let events = (ps.accumulator / config.delta).min(100);
              ps.accumulator %= config.delta;

              let mut rng = rand::thread_rng();
              let mut u_emission = [0.0; 2];
              u_emission[0] = rand::Rng::r#gen(&mut rng);
              u_emission[1] = rand::Rng::r#gen(&mut rng);

              let burst_size = config.emission_count.mean as usize * events as usize + 1000;
              let mut u_particles = alloc::vec::Vec::with_capacity(burst_size);
              for _ in 0..burst_size {
                u_particles.push([
                  rand::Rng::r#gen(&mut rng),
                  rand::Rng::r#gen(&mut rng),
                  rand::Rng::r#gen(&mut rng),
                  rand::Rng::r#gen(&mut rng),
                ]);
              }

              let uv_grid =
                crate::simulation::comet::uv_grid::UvGrid::new(&mesh.vertices, &mesh.indices, 2);

              let mut temp_ps = crate::scene::particles::ParticleSystemComponent::new(10000);
              for _ in 0..events {
                temp_ps.emit_particles(
                  &config,
                  &mesh,
                  &uv_grid,
                  t.position,
                  t.rotation,
                  t.scale,
                  &u_emission,
                  (&u_particles).as_ref(),
                );
              }

              use crate::physics::physics_scene::math::PhysicsSceneMathExt;
              let parent_id =
                scene.get_parent(id).map(|k| slotmap::Key::data(&k).as_ffi() as u32).unwrap_or(0);

              let temp_particles = temp_ps.particles.read().clone();
              let num_temp = temp_particles.len();

              if num_temp > 0 {
                let chunk_size = 64;
                let num_chunks = (num_temp + chunk_size - 1) / chunk_size;

                use aethervk_oshal_rlib::os::pool::chunked::ThreadPoolChunkedExt;
                let accepted_particles =
                  alloc::sync::Arc::new(spin::Mutex::new(alloc::vec::Vec::new()));

                let physical_scene_ref = SendPtr(physical_scene as *const PhysicsScene);
                let accepted_clone = accepted_particles.clone();

                let spawn_res = self.thread_pool.spawn_chunked(num_chunks, move |chunk_id| {
                  let start = chunk_id * chunk_size;
                  let end = (start + chunk_size).min(num_temp);
                  let mut local_accepted = alloc::vec::Vec::new();
                  let phys_scene = physical_scene_ref; // force rust >=2021 to move the struct
                  let phys_scene = unsafe { phys_scene.0.as_ref().unwrap_unchecked() };

                  for i in start..end {
                    let p = &temp_particles[i];
                    if p.active != 0 {
                      let p_pos = Vec3f32::from_array(p.position);
                      let mut dir = sun_pos - p_pos;
                      let dist_to_sun = dir.length();
                      if dist_to_sun > 1e-5 {
                        dir = dir / dist_to_sun;
                        let ray = crate::math::collision::intersection::Ray {
                          origin: p_pos + dir * 0.1,
                          direction: dir,
                          length: dist_to_sun,
                        };

                        // spawned particle shouldn't be already intersecting the scene
                        if phys_scene.intersect_world_bvh_math(&ray).is_empty() {
                          local_accepted.push(p.clone());
                        }
                      }
                    }
                  }

                  if !local_accepted.is_empty() {
                    accepted_clone.lock().extend(local_accepted);
                  }
                });

                let wait_res = match spawn_res {
                  Ok(task) => Ok(task.wait()),
                  Err(e) => Err(EngineError::from(e)),
                };

                if let Err(e) = wait_res {
                  *sys_particles = orig_particles;
                  ps.accumulator = orig_accumulator;
                  return Err(e);
                }

                let accepted = accepted_particles.lock();
                for p in accepted.iter() {
                  sys_particles.push(p.clone());
                  let new_idx = (sys_particles.len() - 1) as u32;
                  particles.data.push(ParticleGpu {
                    position: p.position,
                    velocity: p.velocity,
                    mass: p.mass,
                    force: [0.0, 0.0, 0.0],
                    entity_id: id,
                    parent_frame_id: parent_id,
                    original_index: new_idx,
                  });
                }
              }
            }
            Ok(())
          },
        );

        if let Some(Err(e)) = res {
          return Err(e);
        }
      }
    }
    Ok(())
  }

  fn step_ode_p1_p2(
    &self,
    _cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<ParticleGpu>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    // --- PHASE 1 & 2: Particle Explicit Velocity Half-Kick and Drift ---

    if particles.data.is_empty() {
      return Ok(());
    }
    let dt_sec = dt as f32 / 1_000_000.0;
    let half_dt = dt_sec * 0.5;
    let num_particles = particles.data.len();
    let chunk_size = 2048;
    let num_chunks = (num_particles + chunk_size - 1) / chunk_size;

    use crate::scene::ErasedMutPtr;
    use aethervk_oshal_rlib::os::pool::chunked::ThreadPoolChunkedExt;
    let dyn_ptr = ErasedMutPtr::new(particles.data.as_mut_ptr());

    self
      .thread_pool
      .spawn_chunked(num_chunks, move |chunk_id| {
        let start = chunk_id * chunk_size;
        let end = (start + chunk_size).min(num_particles);
        let dyn_array =
          unsafe { core::slice::from_raw_parts_mut(dyn_ptr.get::<ParticleGpu>(), num_particles) };
        for i in start..end {
          let p = &mut dyn_array[i];
          if p.mass > 0.0 {
            let inv_mass = 1.0 / p.mass;
            let v_half_x = p.velocity[0] + p.force[0] * inv_mass * half_dt;
            let v_half_y = p.velocity[1] + p.force[1] * inv_mass * half_dt;
            let v_half_z = p.velocity[2] + p.force[2] * inv_mass * half_dt;

            p.position[0] += v_half_x * half_dt;
            p.position[1] += v_half_y * half_dt;
            p.position[2] += v_half_z * half_dt;

            p.velocity[0] = v_half_x;
            p.velocity[1] = v_half_y;
            p.velocity[2] = v_half_z;
          }
        }
      })
      .map_err(|e| {
        <aethervk_oshal_rlib::os::NativeError as Into<crate::types::EngineError>>::into(e)
      })?
      .wait();

    Ok(())
  }

  fn step_ode_p3_p4(
    &self,
    _cmd: &mut Self::Cmd,
    _kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
    emitters: &Self::Buffer<ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    // --- PHASE 3 & 4: Rigid Body Implicit Midpoint Rule (IMR) solve ---

    let dt_sec = dt as f32 / 1_000_000.0;
    for rb in rigid_bodies.data.iter_mut() {
      if rb.mass <= 0.0 {
        continue;
      }

      let temp_rb = crate::math::physics::RigidBody {
        position: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.position),
        rotation: aethervk_oshal_rlib::math::matrix::mat3::Mat3f32 {
          x: Vec3f32::from_components(rb.rotation[0][0], rb.rotation[0][1], rb.rotation[0][2]),
          y: Vec3f32::from_components(rb.rotation[1][0], rb.rotation[1][1], rb.rotation[1][2]),
          z: Vec3f32::from_components(rb.rotation[2][0], rb.rotation[2][1], rb.rotation[2][2]),
        },
        linear_velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(
          rb.linear_velocity,
        ),
        angular_velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(
          rb.angular_velocity,
        ),
        mass: rb.mass,
        inertia_tensor: aethervk_oshal_rlib::math::matrix::mat3::Mat3f32 {
          x: Vec3f32::from_components(
            rb.inertia_tensor[0][0],
            rb.inertia_tensor[0][1],
            rb.inertia_tensor[0][2],
          ),
          y: Vec3f32::from_components(
            rb.inertia_tensor[1][0],
            rb.inertia_tensor[1][1],
            rb.inertia_tensor[1][2],
          ),
          z: Vec3f32::from_components(
            rb.inertia_tensor[2][0],
            rb.inertia_tensor[2][1],
            rb.inertia_tensor[2][2],
          ),
        },
      };

      let force_eval = |x_mid: Vec3f32,
                        _r_mid: aethervk_oshal_rlib::math::matrix::mat3::Mat3f32|
       -> crate::math::physics::RigidBodyForceEval {
        let mut f_world = Vec3f32::zero();
        let mut k_translation = aethervk_oshal_rlib::math::matrix::mat3::Mat3f32 {
          x: Vec3f32::zero(),
          y: Vec3f32::zero(),
          z: Vec3f32::zero(),
        };

        for em in emitters.data.iter() {
          if em.type_id == 0 {
            let em_pos = Vec3f32::from_array(em.position);
            let r = em_pos - x_mid;
            let dist_sq = r.length_squared() * em.scale_factor * em.scale_factor;
            if dist_sq > 1e-6 {
              let dist = dist_sq.sqrt();
              let dist3 = dist_sq * dist;
              let dist5 = dist3 * dist_sq;

              let coeff = em.mu * temp_rb.mass / dist3;
              f_world += (r / dist) * (em.mu * temp_rb.mass / dist_sq);

              let term1 = Mat3f32::identity() * (-coeff);
              let rr_t = Mat3f32 {
                x: r * r.x(),
                y: r * r.y(),
                z: r * r.z(),
              };
              let term2 = rr_t * (3.0 * em.mu * temp_rb.mass / dist5);
              k_translation = k_translation + term1 + term2;
            }
          } else if em.type_id == 1 {
            let em_pos = Vec3f32::from_array(em.position);
            let em_norm = Vec3f32::from_array(em.normal);
            let r = x_mid - em_pos;
            let dist = r.dot(em_norm);
            if dist >= 0.0 && dist < em.trunc_distance {
              let denom = 1.0 + dist * dist;
              let force_mag = em.mu / denom;
              f_world += em_norm * force_mag;

              let dF_ddist = -2.0 * em.mu * dist / (denom * denom);
              let nn_t = aethervk_oshal_rlib::math::matrix::mat3::Mat3f32 {
                x: em_norm * em_norm.x(),
                y: em_norm * em_norm.y(),
                z: em_norm * em_norm.z(),
              };
              k_translation = k_translation + nn_t * dF_ddist;
            }
          }
        }

        crate::math::physics::RigidBodyForceEval {
          f_world,
          tau_body: Vec3f32::zero(),
          k_translation,
          p_body: Vec3f32::zero(),
        }
      };

      let (v_mid, w_mid) =
        crate::math::physics::rigid_body_implicit_solve(&temp_rb, dt_sec, force_eval);

      rb.position = [
        rb.position[0] + v_mid.x() * dt_sec,
        rb.position[1] + v_mid.y() * dt_sec,
        rb.position[2] + v_mid.z() * dt_sec,
      ];
      let r_next = temp_rb.rotation * crate::math::expm_hat(w_mid * dt_sec);
      rb.rotation = [
        [r_next.x.x(), r_next.x.y(), r_next.x.z()],
        [r_next.y.x(), r_next.y.y(), r_next.y.z()],
        [r_next.z.x(), r_next.z.y(), r_next.z.z()],
      ];
      let new_v = v_mid * 2.0
        - aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.linear_velocity);
      rb.linear_velocity = [new_v.x(), new_v.y(), new_v.z()];
      let new_w = w_mid * 2.0
        - aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.angular_velocity);
      rb.angular_velocity = [new_w.x(), new_w.y(), new_w.z()];
    }

    Ok(())
  }

  fn compute_self_gravity(
    &self,
    _cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
    particles: &mut Self::Buffer<ParticleGpu>,
  ) -> EngineResult<()> {
    if particles.data.is_empty() {
      return Ok(());
    }

    let g = 1.1906e-19 as f32; // G in AU^3 / (EarthMass * s^2)
    let num_particles = particles.data.len();
    let chunk_size = 2048;
    let num_chunks = (num_particles + chunk_size - 1) / chunk_size;

    use crate::scene::ErasedMutPtr;
    use aethervk_oshal_rlib::os::pool::chunked::ThreadPoolChunkedExt;
    let dyn_ptr = ErasedMutPtr::new(particles.data.as_mut_ptr());
    let bvh_ptr = crate::scene::ErasedPtr::new(bvh as *const Self::MotionBvh);

    self
      .thread_pool
      .spawn_chunked(num_chunks, move |chunk_id| {
        let start = chunk_id * chunk_size;
        let end = (start + chunk_size).min(num_particles);
        let dyn_array =
          unsafe { core::slice::from_raw_parts_mut(dyn_ptr.get::<ParticleGpu>(), num_particles) };
        let bvh_ref = unsafe { &*bvh_ptr.get::<Self::MotionBvh>() };
        let theta = 0.5;

        for i in start..end {
          let my_mass = dyn_array[i].mass;
          if my_mass <= 0.0 {
            continue;
          }

          let my_pos = Vec3f32::from_array(dyn_array[i].position);
          let mut my_global_pos = my_pos;
          let mut local_to_global = aethervk_oshal_rlib::math::matrix::mat3::Mat3f32::identity();
          let mut curr = dyn_array[i].parent_frame_id;

          while curr != 0 {
            let mut found = false;
            for kin in bvh_ref.kinematics_copy.iter() {
              if kin.own_frame_id == curr {
                let mat: Mat4x4f32 = kin.transform.to_mat4();
                let pt = mat.mul_vector(my_global_pos.to_point());
                my_global_pos = Vec3f32::from_array([pt.x(), pt.y(), pt.z()]);
                local_to_global = mat.into_linear::<Mat3f32>() * local_to_global;
                curr = kin.parent_frame_id;
                found = true;
                break;
              }
            }
            if !found {
              break;
            }
          }

          let mut total_force = Vec3f32::zero();
          let my_p_id = (1 << 30) | (i as u32);

          if let Some(root_idx) = bvh_ref.bvh_tree.root {
            let mut stack = alloc::vec![root_idx];
            while let Some(node_idx) = stack.pop() {
              let node = &bvh_ref.bvh_tree.nodes[node_idx as usize];
              if let Some(data_idx) = node.data_index {
                if my_p_id != data_idx {
                  let r = node.center_of_mass - my_global_pos;
                  let dist_sq = r.length_squared();
                  if dist_sq > 1e-6 {
                    let dist = dist_sq.sqrt();
                    total_force += r * (g * my_mass * node.mass / (dist_sq * dist));
                  }
                }
              } else {
                let r = node.center_of_mass - my_global_pos;
                let dist_sq = r.length_squared();
                let dist = dist_sq.max(1e-6).sqrt();
                let extents = node.bounds.max - node.bounds.min;
                let size = extents.x().max(extents.y()).max(extents.z());

                if size / dist < theta {
                  total_force += r * (g * my_mass * node.mass / (dist_sq * dist));
                } else {
                  if let Some(left) = node.left_child {
                    stack.push(left);
                  }
                  if let Some(right) = node.right_child {
                    stack.push(right);
                  }
                }
              }
            }
          }

          let global_to_local = local_to_global
            .inverse()
            .unwrap_or(aethervk_oshal_rlib::math::matrix::mat3::Mat3f32::identity());
          let local_force = global_to_local.mul_vector(total_force);

          dyn_array[i].force[0] += local_force.x();
          dyn_array[i].force[1] += local_force.y();
          dyn_array[i].force[2] += local_force.z();
        }
      })
      .map_err(|e| {
        <aethervk_oshal_rlib::os::NativeError as Into<crate::types::EngineError>>::into(e)
      })?
      .wait();

    Ok(())
  }

  fn step_ode_p5(
    &self,
    _cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    particles: &mut Self::Buffer<ParticleGpu>,
    emitters: &Self::Buffer<ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    if particles.data.is_empty() {
      return Ok(());
    }
    let dt_sec = dt as f32 / 1_000_000.0;
    let half_dt = dt_sec * 0.5;
    let num_particles = particles.data.len();
    let chunk_size = 2048;
    let num_chunks = (num_particles + chunk_size - 1) / chunk_size;

    // Cache the transform hierarchy map outside of thread workers
    let mut frames_map = hashbrown::HashMap::new();
    for kin in kinematics.data.iter() {
      frames_map.insert(
        kin.own_frame_id,
        (kin.parent_frame_id, kin.transform.to_mat4::<Mat4x4f32>()),
      );
    }
    let frames_map = alloc::sync::Arc::new(frames_map);

    use crate::scene::{ErasedMutPtr, ErasedPtr};
    use aethervk_oshal_rlib::{
      math::matrix::MatrixVectorMul, os::pool::chunked::ThreadPoolChunkedExt,
    };

    let dyn_ptr = ErasedMutPtr::new(particles.data.as_mut_ptr());
    let kin_ptr = ErasedPtr::new(kinematics.data.as_ptr());
    let em_ptr = ErasedPtr::new(emitters.data.as_ptr());
    let num_kin = kinematics.data.len();
    let num_em = emitters.data.len();

    self
      .thread_pool
      .spawn_chunked(num_chunks, move |chunk_id| {
        let start = chunk_id * chunk_size;
        let end = (start + chunk_size).min(num_particles);

        let dyn_array =
          unsafe { core::slice::from_raw_parts_mut(dyn_ptr.get::<ParticleGpu>(), num_particles) };
        let kin_array =
          unsafe { core::slice::from_raw_parts(kin_ptr.get::<KinematicBody>(), num_kin) };
        let em_array = unsafe { core::slice::from_raw_parts(em_ptr.get::<ForceEmitter>(), num_em) };

        let get_global_pos = |frame_id: u32, local_pos: Vec3f32| -> Vec3f32 {
          let mut pos = local_pos;
          let mut curr = frame_id;
          while let Some(&(parent, ref mat)) = frames_map.get(&curr) {
            let pt = mat.mul_vector(pos.to_point());
            pos = Vec3f32::from_array([pt.x(), pt.y(), pt.z()]);
            curr = parent;
            if curr == 0 {
              break;
            }
          }
          pos
        };

        let get_global_to_local =
          |frame_id: u32| -> aethervk_oshal_rlib::math::matrix::mat3::Mat3f32 {
            let mut local_to_global = aethervk_oshal_rlib::math::matrix::mat3::Mat3f32::identity();
            let mut curr = frame_id;
            while let Some(&(parent, ref mat)) = frames_map.get(&curr) {
              local_to_global = mat
                .into_linear::<aethervk_oshal_rlib::math::matrix::mat3::Mat3f32>()
                * local_to_global;
              curr = parent;
              if curr == 0 {
                break;
              }
            }
            local_to_global
              .inverse()
              .unwrap_or(aethervk_oshal_rlib::math::matrix::mat3::Mat3f32::identity())
          };

        for i in start..end {
          let p = &mut dyn_array[i];
          if p.mass <= 0.0 {
            continue;
          }
          let inv_mass = 1.0 / p.mass;

          p.position[0] += p.velocity[0] * half_dt;
          p.position[1] += p.velocity[1] * half_dt;
          p.position[2] += p.velocity[2] * half_dt;

          let mut f_next = Vec3f32::zero();
          let p_pos = Vec3f32::from_array(p.position);

          for em in em_array.iter() {
            if em.type_id == 0 {
              let em_pos = Vec3f32::from_array(em.position);
              let r = em_pos - p_pos;
              let scaled_dist_sq = r.length_squared() * em.scale_factor * em.scale_factor;
              if scaled_dist_sq > 1e-6 {
                let force_dir = r / r.length();
                f_next += force_dir * (em.mu * p.mass / scaled_dist_sq);
              }
            } else if em.type_id == 1 {
              let em_pos = Vec3f32::from_array(em.position);
              let em_norm = Vec3f32::from_array(em.normal);
              let r = p_pos - em_pos;
              let dist = r.dot(em_norm);
              if dist >= 0.0 && dist < em.trunc_distance {
                let force_mag = em.mu / (1.0 + dist * dist);
                f_next += em_norm * force_mag;
              }
            }
          }

          let p_global_pos = get_global_pos(p.parent_frame_id, p_pos);
          let mut global_f_next = Vec3f32::zero();

          for kin in kin_array.iter() {
            if kin.mu > 0.0 {
              let kin_global_pos = get_global_pos(kin.parent_frame_id, kin.transform.position);
              let r = kin_global_pos - p_global_pos;
              let dist_sq = r.length_squared();
              if dist_sq > 1e-6 {
                let dist = dist_sq.sqrt();
                global_f_next += r * (kin.mu * p.mass / (dist_sq * dist));
              }
            }
          }

          let global_to_local = get_global_to_local(p.parent_frame_id);
          let local_gravity = global_to_local.mul_vector(global_f_next);
          f_next += local_gravity;

          p.force[0] = f_next.x();
          p.force[1] = f_next.y();
          p.force[2] = f_next.z();

          p.velocity[0] += p.force[0] * inv_mass * half_dt;
          p.velocity[1] += p.force[1] * inv_mass * half_dt;
          p.velocity[2] += p.force[2] * inv_mass * half_dt;
        }
      })
      .map_err(|e| {
        <aethervk_oshal_rlib::os::NativeError as Into<crate::types::EngineError>>::into(e)
      })?
      .wait();

    Ok(())
  }

  fn build_motion_bvh(
    &self,
    _cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &Self::Buffer<RigidBodyGpu>,
    particles: &Self::Buffer<ParticleGpu>,
    _dt: timeus_t,
  ) -> EngineResult<Self::MotionBvh> {
    use crate::physics::motion_bvh::{Aabb, CpuBvhItem, MotionBvhTree};
    use aethervk_oshal_rlib::math::{
      matrix::{MatrixVectorMul, mat4::Mat4x4f32},
      vector::{Vector, vec3::Vec3f32},
    };

    let mut frames_map = hashbrown::HashMap::new();
    let mut frame_types = hashbrown::HashMap::new();
    for kin in kinematics.data.iter() {
      let mat: Mat4x4f32 = kin.transform.to_mat4();
      frames_map.insert(kin.own_frame_id, (kin.parent_frame_id, mat));
      frame_types.insert(kin.own_frame_id, kin.frame_type);
    }

    let get_global_pos = |frame_id: u32, local_pos: Vec3f32| -> Vec3f32 {
      let mut pos = local_pos;
      let mut curr = frame_id;
      while let Some(&(parent, ref transform)) = frames_map.get(&curr) {
        let pt = transform.mul_vector(pos.to_point());
        pos = Vec3f32::from_array([pt.x(), pt.y(), pt.z()]);
        curr = parent;
        if curr == 0 {
          break;
        }
      }
      pos
    };

    let mut flat_items: alloc::vec::Vec<(Aabb, CpuBvhItem)> = alloc::vec::Vec::new();

    for (i, kin) in kinematics.data.iter().enumerate() {
      if kin.shape_data[0] == 0.0 && kin.shape_data[1] == 0.0 && kin.shape_data[2] == 0.0 {
        continue; // Don't add mathematical frames to the collision tree!
      }
      let global_pos = get_global_pos(kin.parent_frame_id, kin.transform.position);
      let max_travel = kin.velocity.length();
      let r = kin.scale + max_travel;
      let bounds = Aabb::new(
        global_pos - Vec3f32::from_array([r, r, r]),
        global_pos + Vec3f32::from_array([r, r, r]),
      );
      // data_index: top bit 1 means kinematic, lower 31 bits are index
      flat_items.push((
        bounds,
        CpuBvhItem::Primitive((1 << 31) | (i as u32), kin.mu, global_pos),
      ));
    }

    for (i, dyn_body) in rigid_bodies.data.iter().enumerate() {
      let global_pos = get_global_pos(
        dyn_body.parent_frame_id,
        aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(dyn_body.position),
      );
      let max_travel =
        aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(dyn_body.linear_velocity)
          .length();
      let r = dyn_body.shape_data[0] + max_travel;
      let bounds = Aabb::new(
        global_pos - Vec3f32::from_array([r, r, r]),
        global_pos + Vec3f32::from_array([r, r, r]),
      );
      // data_index: top bit 0 means dynamic, lower 31 bits are index
      flat_items.push((
        bounds,
        CpuBvhItem::Primitive(i as u32, dyn_body.mass, global_pos),
      ));
    }

    for (i, p) in particles.data.iter().enumerate() {
      let global_pos = get_global_pos(
        p.parent_frame_id,
        aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(p.position),
      );
      let max_travel =
        aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(p.velocity).length();
      let r = 1.0 + max_travel; // Particles default radius 1.0
      let bounds = Aabb::new(
        global_pos - Vec3f32::from_array([r, r, r]),
        global_pos + Vec3f32::from_array([r, r, r]),
      );
      // data_index: second top bit 1 means particle, lower 30 bits are index
      flat_items.push((
        bounds,
        CpuBvhItem::Primitive((1 << 30) | (i as u32), p.mass, global_pos),
      ));
    }

    let mut nodes = alloc::vec::Vec::new();
    let tlas_root = MotionBvhTree::build_into(&mut flat_items, &mut nodes);

    let bvh_tree = MotionBvhTree {
      nodes,
      root: tlas_root,
    };

    Ok(CpuMotionBvh {
      kinematics_copy: kinematics.data.clone(),
      rigid_bodies_copy: rigid_bodies.data.clone(),
      particles_copy: particles.data.clone(),
      bvh_tree,
    })
  }

  fn self_intersect_scene(
    &self,
    _cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
  ) -> EngineResult<Self::List<CollisionPair>> {
    let pairs = crate::physics::collision_pipeline::detect_collisions_cpu(bvh);
    Ok(CpuList { data: pairs })
  }

  fn intersect_instances(
    &self,
    _cmd: &mut Self::Cmd,
    potentials: &Self::List<CollisionPair>,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &Self::Buffer<RigidBodyGpu>,
    particles: &Self::Buffer<ParticleGpu>,
  ) -> EngineResult<Self::List<CollisionPair>> {
    use aethervk_oshal_rlib::math::{
      matrix::{MatrixVectorMul, SquareMatrix},
      vector::Vector,
    };
    let mut actual_collisions = alloc::vec::Vec::new();

    let mut frames_map = hashbrown::HashMap::new();
    for kin in kinematics.data.iter() {
      frames_map.insert(
        kin.own_frame_id,
        (kin.parent_frame_id, kin.transform.to_mat4::<Mat4x4f32>()),
      );
    }

    enum DynamicShape {
      Sphere(crate::math::collision::gjk::GjkSphere),
      Obb(crate::math::collision::bounds::OBB<f32>),
    }
    impl crate::math::collision::gjk::Support for DynamicShape {
      fn support(&self, dir: Vec3f32) -> Vec3f32 {
        match self {
          Self::Sphere(s) => crate::math::collision::gjk::Support::support(s, dir),
          Self::Obb(o) => crate::math::collision::gjk::Support::support(o, dir),
        }
      }
    }

    let get_properties = |idx: u32| -> Option<(u32, u32, [f32; 3], Mat4x4f32)> {
      if (idx & (1 << 31)) != 0 {
        let i = (idx & !(1 << 31)) as usize;
        if i >= kinematics.data.len() {
          return None;
        }
        let k = &kinematics.data[i];
        Some((
          k.parent_frame_id,
          k.shape_type,
          k.shape_data,
          k.transform.to_mat4(),
        ))
      } else if (idx & (1 << 30)) != 0 {
        let i = (idx & !(1 << 30)) as usize;
        if i >= particles.data.len() {
          return None;
        }
        let p = &particles.data[i];
        Some((
          p.parent_frame_id,
          0,
          [1.0, 0.0, 0.0],
          Mat4x4f32::translation(Vec3f32::from_array(p.position)),
        ))
      } else {
        let i = idx as usize;
        if i >= rigid_bodies.data.len() {
          return None;
        }
        let rb = &rigid_bodies.data[i];
        let rot = rb.rotation;
        let mat = Mat4x4f32::from_columns(
          Vec4f32::from_components(rot[0][0], rot[0][1], rot[0][2], 0.0),
          Vec4f32::from_components(rot[1][0], rot[1][1], rot[1][2], 0.0),
          Vec4f32::from_components(rot[2][0], rot[2][1], rot[2][2], 0.0),
          Vec4f32::from_components(rb.position[0], rb.position[1], rb.position[2], 1.0),
        );
        Some((rb.parent_frame_id, rb.shape_type, rb.shape_data, mat))
      }
    };

    let get_shape = |shape_type: u32, shape_data: [f32; 3], mat: Mat4x4f32| -> DynamicShape {
      if shape_type == 1 {
        let mut x = Vec3f32::from_components(
          mat.component(0).unwrap(),
          mat.component(1).unwrap(),
          mat.component(2).unwrap(),
        );
        let mut y = Vec3f32::from_components(
          mat.component(4).unwrap(),
          mat.component(5).unwrap(),
          mat.component(6).unwrap(),
        );
        let mut z = Vec3f32::from_components(
          mat.component(8).unwrap(),
          mat.component(9).unwrap(),
          mat.component(10).unwrap(),
        );

        let scale_x = x.length();
        let scale_y = y.length();
        let scale_z = z.length();

        if scale_x > 1e-6 {
          x = x / scale_x;
        }
        if scale_y > 1e-6 {
          y = y / scale_y;
        }
        if scale_z > 1e-6 {
          z = z / scale_z;
        }

        let rot3 = Mat3f32 { x, y, z };
        let pos = Vec3f32::from_components(
          mat.component(12).unwrap(),
          mat.component(13).unwrap(),
          mat.component(14).unwrap(),
        );
        let scaled_extents = [
          shape_data[0] * scale_x,
          shape_data[1] * scale_y,
          shape_data[2] * scale_z,
        ];
        DynamicShape::Obb(crate::math::collision::bounds::OBB::new(
          pos,
          rot3,
          Vec3f32::from_array(scaled_extents),
        ))
      } else {
        let x = Vec3f32::from_components(
          mat.component(0).unwrap(),
          mat.component(1).unwrap(),
          mat.component(2).unwrap(),
        );
        let scale_x = x.length(); // assuming uniform scale
        let pos = Vec3f32::from_components(
          mat.component(12).unwrap(),
          mat.component(13).unwrap(),
          mat.component(14).unwrap(),
        );
        DynamicShape::Sphere(crate::math::collision::gjk::GjkSphere {
          center: pos,
          radius: shape_data[0] * scale_x,
        })
      }
    };

    for pair in &potentials.data {
      let idx_a = pair.a.primitive_index;
      let idx_b = pair.b.primitive_index;

      let props_a = get_properties(idx_a);
      let props_b = get_properties(idx_b);

      if let (Some((parent_a, type_a, data_a, mat_a)), Some((parent_b, type_b, data_b, mat_b))) =
        (props_a, props_b)
      {
        let mat_a_transformed = if parent_a != parent_b {
          let mut m_a_lca = mat_a;
          let mut curr = parent_a;
          while curr != 0 {
            if let Some(&(p, ref t)) = frames_map.get(&curr) {
              m_a_lca = *t * m_a_lca;
              curr = p;
            } else {
              break;
            }
          }
          m_a_lca
        } else {
          mat_a
        };
        let mat_b_transformed = if parent_a != parent_b {
          let mut m_b_lca = mat_b;
          let mut curr = parent_b;
          while curr != 0 {
            if let Some(&(p, ref t)) = frames_map.get(&curr) {
              m_b_lca = *t * m_b_lca;
              curr = p;
            } else {
              break;
            }
          }
          m_b_lca
        } else {
          mat_b
        };

        let shape_a = get_shape(type_a, data_a, mat_a_transformed);
        let shape_b = get_shape(type_b, data_b, mat_b_transformed);

        let center_a = match &shape_a {
          DynamicShape::Sphere(s) => s.center,
          DynamicShape::Obb(o) => o.translation(),
        };
        let center_b = match &shape_b {
          DynamicShape::Sphere(s) => s.center,
          DynamicShape::Obb(o) => o.translation(),
        };

        #[cfg(test)]
        aethervk_oshal_rlib::log!(
          "GJK evaluating: idx_a={}, idx_b={}, A={:?}, B={:?}",
          idx_a,
          idx_b,
          center_a,
          center_b
        );

        let (dist, pt_a, pt_b) = crate::math::collision::gjk::gjk_distance(&shape_a, &shape_b);

        #[cfg(test)]
        aethervk_oshal_rlib::log!("GJK evaluated -> dist: {}", dist);

        if dist <= 0.0 {
          let mut new_pair = pair.clone();
          new_pair.penetration_depth = if dist < 0.0 { -dist } else { 0.0 };
          let mut normal = pt_a - pt_b;
          if dist < 0.0 {
            normal = -normal;
          }
          let len = normal.length();
          if len > 1e-6 {
            normal = normal / len;
          } else {
            let diff = center_a - center_b;
            let diff_len = diff.length();
            if diff_len > 1e-6 {
              normal = diff / diff_len;
            } else {
              normal =
                aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([1.0, 0.0, 0.0]);
            }
          }
          new_pair.contact_normal = [normal.x(), normal.y(), normal.z()];
          new_pair.contact_point = [
            (pt_a.x() + pt_b.x()) * 0.5,
            (pt_a.y() + pt_b.y()) * 0.5,
            (pt_a.z() + pt_b.z()) * 0.5,
          ];
          #[cfg(test)]
          aethervk_oshal_rlib::log!(
            "COLLISION DETECTED: idx_a={}, idx_b={}, dist={}, normal={:?}",
            idx_a,
            idx_b,
            dist,
            new_pair.contact_normal
          );
          actual_collisions.push(new_pair);
        }
      }
    }

    Ok(CpuList {
      data: actual_collisions,
    })
  }

  fn compact_collisions(
    &self,
    _cmd: &mut Self::Cmd,
    globals: &Self::List<CollisionPair>,
    _time_delta: timeus_t,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList {
      data: globals.data.clone(),
    })
  }

  fn find_earliest_collision(
    &self,
    _cmd: &mut Self::Cmd,
    compacted: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::Buffer<timeus_t>> {
    let mut min_tc = timeus_t::MAX;
    for pair in &compacted.data {
      if (pair.time_of_impact as timeus_t) < min_tc {
        min_tc = pair.time_of_impact as timeus_t;
      }
    }
    Ok(CpuBuffer {
      data: alloc::vec![min_tc],
    })
  }

  fn apply_collision_responses(
    &self,
    _cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
    particles: &mut Self::Buffer<ParticleGpu>,
    collisions: &Self::List<CollisionPair>,
    force_inelastic: bool,
  ) -> EngineResult<()> {
    if collisions.data.is_empty() {
      return Ok(());
    }

    #[cfg(test)]
    aethervk_oshal_rlib::log!(
      "apply_collision_responses: rigid_bodies.len={}, particles.len={}",
      rigid_bodies.data.len(),
      particles.data.len()
    );

    if rigid_bodies.data.len() > 0 {
      #[cfg(test)]
      aethervk_oshal_rlib::log!("rb 0 mass={}", rigid_bodies.data[0].mass);
    }

    let clusters =
      crate::physics::cpu_kernels::group_and_cluster_collisions(collisions.data.clone(), 0.01);
    let restitution = if force_inelastic { 0.0 } else { 0.5 };

    let dyn_array = rigid_bodies.data.as_mut_slice();
    let p_array = particles.data.as_mut_slice();

    for cluster in clusters {
      crate::physics::lcp_integration::resolve_cluster_lcp(
        &cluster,
        &kinematics.data,
        dyn_array,
        p_array,
        restitution,
      );
    }
    Ok(())
  }

  fn snapshot_dynamics(
    &self,
    _cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<RigidBodyGpu>,
    particles: &Self::Buffer<ParticleGpu>,
  ) -> EngineResult<(Self::Buffer<RigidBodyGpu>, Self::Buffer<ParticleGpu>)> {
    Ok((
      CpuBuffer {
        data: rigid_bodies.data.clone(),
      },
      CpuBuffer {
        data: particles.data.clone(),
      },
    ))
  }

  fn restore_dynamics(
    &self,
    _cmd: &mut Self::Cmd,
    rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
    particles: &mut Self::Buffer<ParticleGpu>,
    snapshot: &(Self::Buffer<RigidBodyGpu>, Self::Buffer<ParticleGpu>),
  ) -> EngineResult<()> {
    rigid_bodies.data = snapshot.0.data.clone();
    particles.data = snapshot.1.data.clone();
    Ok(())
  }

  fn write_back_to_scene(
    &self,
    _cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<RigidBodyGpu>,
    particles: &Self::Buffer<ParticleGpu>,
    _physical_scene: &mut PhysicsScene,
    scene: &Scene,
  ) -> EngineResult<()> {
    // Write the updated positions and velocities back to the particle components.
    scene.query2::<crate::scene::TransformComponent, crate::scene::particles::ParticleSystemComponent, _>(
        |entity, _transform, sys| {
          let mut sys_particles = sys.particles.write();
          for p_gpu in particles.data.iter() {
            if p_gpu.entity_id == entity {
              if (p_gpu.original_index as usize) < sys_particles.len() {
                sys_particles[p_gpu.original_index as usize].position = p_gpu.position;
                sys_particles[p_gpu.original_index as usize].velocity = p_gpu.velocity;
              }
            }
          }
        }
      );
    // Write back rigid bodies
    for rb in rigid_bodies.data.iter() {
      let _ = scene.with_component_mut(rb.entity_id, |trans: &mut TransformComponent| {
        trans.position = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.position);
        let mat = Mat4x4f32::from_columns(
          Vec4f32::from_components(rb.rotation[0][0], rb.rotation[0][1], rb.rotation[0][2], 0.0),
          Vec4f32::from_components(rb.rotation[1][0], rb.rotation[1][1], rb.rotation[1][2], 0.0),
          Vec4f32::from_components(rb.rotation[2][0], rb.rotation[2][1], rb.rotation[2][2], 0.0),
          Vec4f32::from_components(0.0, 0.0, 0.0, 1.0),
        );
        trans.rotation = Quat::from_mat4(&mat);
      });
      let _ = scene.with_component_mut(rb.entity_id, |kin: &mut KinematicComponent| {
        kin.velocity =
          aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.linear_velocity);
        kin.angular_velocity =
          aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.angular_velocity);
      });
    }
    Ok(())
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
