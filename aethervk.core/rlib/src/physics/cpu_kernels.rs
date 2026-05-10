//! cpu_kernels module.

use crate::gpu::{
  CollisionPair, CommandBuffer, DeviceBuffer, DeviceBvh, DeviceList, DynamicBody, Kernels,
  KinematicBody, WaitHandle,
};
use crate::physics::physics_scene::PhysicsScene;
use crate::scene::Scene;
use crate::types::{EngineError, EngineResult};
use aethervk_oshal_rlib::math::floating::FloatOps;
use aethervk_oshal_rlib::math::vector::{Vector, Vector3};
use aethervk_oshal_rlib::os::time::timeus_t;
use alloc::boxed::Box;
use alloc::vec::Vec;

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
  pub dynamics_copy: Vec<DynamicBody>,
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
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: t.clone(),
          velocity: vel,
          parent_frame_id: parent_id,
          mu: planet.mu,
          own_frame_id: own_id,
          frame_type,
          scale: scale * t.scale.x(),
        });
      },
    );
    scene0.query2::<crate::scene::TransformComponent, crate::scene::SunComponent, _>(
      |entity, transform, _sun| {
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
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: transform.clone(),
          velocity: vel,
          parent_frame_id: parent_id,
          mu: 1.3271244e11, // Example Sun mu
          own_frame_id: own_id,
          frame_type,
          scale: scale * transform.scale.x(),
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

  fn step_ode_p1_p2(
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
        dyn_body.velocity = dyn_body.velocity + dyn_body.force * (inv_mass * half_dt);
        dyn_body.transform.position = dyn_body.transform.position + dyn_body.velocity * half_dt;
      }
    }
    Ok(())
  }

  fn step_ode_p3_p4(
    &self,
    _cmd: &mut Self::Cmd,
    _kinematics: &mut Self::Buffer<KinematicBody>,
    _dynamics: &mut Self::Buffer<DynamicBody>,
    _dt: timeus_t,
  ) -> EngineResult<()> {
    // IMR solve for kinematic/rigid bodies goes here.
    // Currently treating kinematic bodies as driven by SPICE, so this is a no-op for now.
    Ok(())
  }

  fn step_ode_p5(
    &self,
    _cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    dynamics: &mut Self::Buffer<DynamicBody>,
    _bvh: &Self::MotionBvh,
    dt: timeus_t,
  ) -> EngineResult<()> {
    let dt_sec = dt as f32 / 1_000_000.0;
    let half_dt = dt_sec * 0.5;

    for dyn_body in dynamics.data.iter_mut() {
      if dyn_body.mass > 0.0 {
        dyn_body.transform.position = dyn_body.transform.position + dyn_body.velocity * half_dt;

        let mut f_grav =
          aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]);
        let mut parent_scale = 1.0;
        let mut parent_macro_pos =
          aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]);
        for kin_body in kinematics.data.iter() {
          if kin_body.own_frame_id == dyn_body.parent_frame_id {
            parent_scale = kin_body.scale;
            parent_macro_pos = kin_body.transform.position;
          }
        }

        for kin_body in kinematics.data.iter() {
          if dyn_body.parent_frame_id == kin_body.own_frame_id {
            let r = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0])
              - dyn_body.transform.position;
            let dist_sq = aethervk_oshal_rlib::math::vector::Vector::dot(r, r);
            if dist_sq > 1e-6 {
              let dist = dist_sq.sqrt();
              let local_mu = if kin_body.frame_type == 1 {
                kin_body.mu / (parent_scale * parent_scale * parent_scale)
              } else {
                kin_body.mu
              };
              f_grav = f_grav + r * (local_mu * dyn_body.mass / (dist_sq * dist));
            }
          } else if kin_body.frame_type == 0 {
            if dyn_body.parent_frame_id != kin_body.own_frame_id {
              let macro_pos_in_micro =
                (kin_body.transform.position - parent_macro_pos) / parent_scale;
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

        let inv_mass = 1.0 / dyn_body.mass;
        dyn_body.velocity = dyn_body.velocity + dyn_body.force * (inv_mass * half_dt);
      }
    }
    Ok(())
  }

  fn build_motion_bvh(
    &self,
    _cmd: &mut Self::Cmd,
    dynamics: &Self::Buffer<DynamicBody>,
  ) -> EngineResult<Self::MotionBvh> {
    Ok(CpuMotionBvh {
      dynamics_copy: dynamics.data.clone(),
    })
  }

  fn self_intersect_scene(
    &self,
    _cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
  ) -> EngineResult<Self::List<CollisionPair>> {
    let mut pairs = Vec::new();
    let dynamics = &bvh.dynamics_copy;
    if dynamics.is_empty() {
      return Ok(CpuList { data: pairs });
    }

    use crate::math::collision::bvh_builder::{BVHBuilderParams, BoundNode};
    use crate::physics::particle::{Particle, ParticleBVHBuilder};
    use aethervk_oshal_rlib::math::matrix::mat3::Mat3f32;

    // Group particles by parent_frame_id to build independent BVHs
    let mut frames_map: hashbrown::HashMap<
      u32,
      Vec<(
        usize,
        Particle<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>,
      )>,
    > = hashbrown::HashMap::new();
    for (i, b) in dynamics.iter().enumerate() {
      frames_map.entry(b.parent_frame_id).or_default().push((
        i,
        Particle {
          position: b.transform.position,
          radius: 1.0, // Assume 1.0 for now
        },
      ));
    }

    let builder = ParticleBVHBuilder::new(BVHBuilderParams::default());

    for (_frame_id, frame_particles) in frames_map {
      if frame_particles.len() < 2 {
        continue;
      }
      let just_particles: Vec<_> = frame_particles.iter().map(|(_, p)| *p).collect();
      if let Some(root) = builder.build::<_, _, Mat3f32>(&just_particles) {
        let mut stack = Vec::new();
        stack.push(&*root);

        while let Some(node) = stack.pop() {
          if let (Some(left), Some(right)) = (&node.left, &node.right) {
            let intersects = match (&left.bound, &right.bound) {
              (BoundNode::AABB(a), BoundNode::AABB(b)) => {
                crate::math::collision::intersection::intersect_aabb_aabb::<
                  aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
                >(a, b)
              }
              (BoundNode::OBB(a), BoundNode::OBB(b)) => {
                crate::math::collision::intersection::intersect_aabb_aabb::<
                  aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
                >(
                  &a.to_aabb::<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>(),
                  &b.to_aabb::<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>(),
                )
              }
              _ => false, // fallback
            };

            if intersects {
              // Gather primitives from both and cross-check
              let mut left_prims = Vec::new();
              let mut l_stack = alloc::vec![left.as_ref()];
              while let Some(l_node) = l_stack.pop() {
                if l_node.primitive_indices.is_empty() {
                  if let Some(ll) = &l_node.left {
                    l_stack.push(ll.as_ref());
                  }
                  if let Some(lr) = &l_node.right {
                    l_stack.push(lr.as_ref());
                  }
                } else {
                  left_prims.extend_from_slice(&l_node.primitive_indices);
                }
              }

              let mut right_prims = Vec::new();
              let mut r_stack = alloc::vec![right.as_ref()];
              while let Some(r_node) = r_stack.pop() {
                if r_node.primitive_indices.is_empty() {
                  if let Some(rl) = &r_node.left {
                    r_stack.push(rl.as_ref());
                  }
                  if let Some(rr) = &r_node.right {
                    r_stack.push(rr.as_ref());
                  }
                } else {
                  right_prims.extend_from_slice(&r_node.primitive_indices);
                }
              }

              for &l_idx in &left_prims {
                for &r_idx in &right_prims {
                  let orig_i = frame_particles[l_idx].0;
                  let orig_j = frame_particles[r_idx].0;
                  let b1 = &dynamics[orig_i];
                  let b2 = &dynamics[orig_j];
                  let dist_sq = (b1.transform.position - b2.transform.position).length_squared();
                  let radius = 1.0;
                  if dist_sq < (radius * 2.0) * (radius * 2.0) {
                    pairs.push(CollisionPair {
                      a: crate::gpu::ColliderId {
                        entity_id: slotmap::Key::data(&b1.entity_id).as_ffi() as u32,
                        primitive_index: orig_i as u32,
                      },
                      b: crate::gpu::ColliderId {
                        entity_id: slotmap::Key::data(&b2.entity_id).as_ffi() as u32,
                        primitive_index: orig_j as u32,
                      },
                      time_of_impact: 0.0,
                    });
                  }
                }
              }
            }

            stack.push(left.as_ref());
            stack.push(right.as_ref());
          }
        }
      }
    }

    Ok(CpuList { data: pairs })
  }

  fn intersect_instances(
    &self,
    _cmd: &mut Self::Cmd,
    potentials: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList {
      data: potentials.data.clone(),
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
    dynamics: &mut Self::Buffer<DynamicBody>,
    collisions: &Self::List<CollisionPair>,
    force_inelastic: bool,
  ) -> EngineResult<()> {
    if collisions.data.is_empty() { return Ok(()); }

    let clusters = group_and_cluster_collisions(collisions.data.clone(), 0.01);
    let restitution = if force_inelastic { 0.0 } else { 0.5 };

    let dyn_array = dynamics.data.as_mut_slice();
    let dyn_len = dyn_array.len();

    let max_iters = 20;

    for cluster in clusters {
      let mut impulses = alloc::vec::Vec::with_capacity(cluster.len());
      impulses.resize(cluster.len(), 0.0f32);

      for _iter in 0..max_iters {
        for (i, pair) in cluster.iter().enumerate() {
          let idx_a = pair.a.primitive_index as usize;
          let idx_b = pair.b.primitive_index as usize;
          if idx_a < dyn_len && idx_b < dyn_len {
            let pos_a = dyn_array[idx_a].transform.position;
            let pos_b = dyn_array[idx_b].transform.position;
            let mut normal = pos_a - pos_b;
            let dist = normal.length();
            if dist > 1e-6 {
              normal = normal / dist;
              let v_rel = dyn_array[idx_a].velocity - dyn_array[idx_b].velocity;
              let v_rel_n = aethervk_oshal_rlib::math::vector::Vector::dot(v_rel, normal);

              let m_a = dyn_array[idx_a].mass;
              let m_b = dyn_array[idx_b].mass;
              let inv_m_a = if m_a > 0.0 { 1.0 / m_a } else { 0.0 };
              let inv_m_b = if m_b > 0.0 { 1.0 / m_b } else { 0.0 };

              let a_ii = inv_m_a + inv_m_b;
              if a_ii > 1e-6 {
                let penetration = (2.0 - dist).max(0.0);
                let beta = 0.2;
                let slop = 0.01;
                let bias = (beta / 0.016) * (penetration - slop).max(0.0);

                let b_i = -(1.0 + restitution) * v_rel_n + bias;
                let lambda = b_i / a_ii;

                let old_impulse = impulses[i];
                let new_impulse = (old_impulse + lambda).max(0.0);
                let delta_lambda = new_impulse - old_impulse;
                impulses[i] = new_impulse;

                let impulse_vec = normal * delta_lambda;
                dyn_array[idx_a].velocity = dyn_array[idx_a].velocity + impulse_vec * inv_m_a;
                dyn_array[idx_b].velocity = dyn_array[idx_b].velocity - impulse_vec * inv_m_b;
              }
            }
          }
        }
      }
    }
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

pub fn group_and_cluster_collisions(
  mut collisions: alloc::vec::Vec<CollisionPair>,
  time_tolerance: f32,
) -> alloc::vec::Vec<alloc::vec::Vec<CollisionPair>> {
  collisions.sort_by(|a, b| {
    a.time_of_impact.partial_cmp(&b.time_of_impact).unwrap_or(core::cmp::Ordering::Equal)
  });

  let mut collided_entities: hashbrown::HashSet<u32> = hashbrown::HashSet::new();
  let mut resolved_clusters = alloc::vec::Vec::new();

  let mut current_group = alloc::vec::Vec::new();
  let mut current_time = -1.0;

  for col in collisions {
    if collided_entities.contains(&col.a.primitive_index) || collided_entities.contains(&col.b.primitive_index) {
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
      let mut clusters = form_clusters(&current_group);
      resolved_clusters.append(&mut clusters);

      for c in &current_group {
        collided_entities.insert(c.a.primitive_index);
        collided_entities.insert(c.b.primitive_index);
      }

      current_group.clear();

      if collided_entities.contains(&col.a.primitive_index) || collided_entities.contains(&col.b.primitive_index) {
        continue;
      }

      current_time = col.time_of_impact;
      current_group.push(col);
    }
  }

  if !current_group.is_empty() {
    let mut clusters = form_clusters(&current_group);
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
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: t.clone(),
          velocity: vel,
          parent_frame_id: parent_id,
          mu: planet.mu,
          own_frame_id: own_id,
          frame_type,
          scale: scale * t.scale.x(),
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
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: t.clone(),
          velocity: vel,
          parent_frame_id: parent_id,
          mu: 1.3271244e11, // Sun mu TODO constant
          own_frame_id: own_id,
          frame_type,
          scale: scale * t.scale.x(),
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

  fn step_ode_p1_p2(
    &self,
    _cmd: &mut Self::Cmd,
    dynamics: &mut Self::Buffer<DynamicBody>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    if dynamics.data.is_empty() {
      return Ok(());
    }

    use crate::scene::ErasedMutPtr;
    use aethervk_oshal_rlib::os::pool::chunked::ThreadPoolChunkedExt;

    let dt_sec = dt as f32 / 1_000_000.0;
    let half_dt = dt_sec * 0.5;

    let num_particles = dynamics.data.len();
    let chunk_size = 2048;
    let num_chunks = (num_particles + chunk_size - 1) / chunk_size;

    let dyn_ptr = ErasedMutPtr::new(dynamics.data.as_mut_ptr());

    let _ = self.thread_pool.spawn_chunked(num_chunks, move |chunk_id| {
      let start = chunk_id * chunk_size;
      let end = (start + chunk_size).min(num_particles);

      let dyn_array =
        unsafe { core::slice::from_raw_parts_mut(dyn_ptr.get::<DynamicBody>(), num_particles) };

      for i in start..end {
        let dyn_body = &mut dyn_array[i];
        if dyn_body.mass > 0.0 {
          let inv_mass = 1.0 / dyn_body.mass;
          dyn_body.velocity = dyn_body.velocity + dyn_body.force * (inv_mass * half_dt);
          dyn_body.transform.position = dyn_body.transform.position + dyn_body.velocity * half_dt;
        }
      }
    });

    Ok(())
  }

  fn step_ode_p3_p4(
    &self,
    _cmd: &mut Self::Cmd,
    _kinematics: &mut Self::Buffer<KinematicBody>,
    _dynamics: &mut Self::Buffer<DynamicBody>,
    _dt: timeus_t,
  ) -> EngineResult<()> {
    // IMR solve for kinematic/rigid bodies goes here.
    // Currently treating kinematic bodies as driven by SPICE, so this is a no-op for now.
    Ok(())
  }

  fn step_ode_p5(
    &self,
    _cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    dynamics: &mut Self::Buffer<DynamicBody>,
    _bvh: &Self::MotionBvh,
    dt: timeus_t,
  ) -> EngineResult<()> {
    if dynamics.data.is_empty() {
      return Ok(());
    }

    use crate::scene::{ErasedMutPtr, ErasedPtr};
    use aethervk_oshal_rlib::os::pool::chunked::ThreadPoolChunkedExt;

    let dt_sec = dt as f32 / 1_000_000.0;
    let half_dt = dt_sec * 0.5;

    let num_particles = dynamics.data.len();
    let chunk_size = 2048;
    let num_chunks = (num_particles + chunk_size - 1) / chunk_size;

    let dyn_ptr = ErasedMutPtr::new(dynamics.data.as_mut_ptr());
    let kin_ptr = ErasedPtr::new(kinematics.data.as_ptr());
    let num_kin = kinematics.data.len();

    let _ = self.thread_pool.spawn_chunked(num_chunks, move |chunk_id| {
      let start = chunk_id * chunk_size;
      let end = (start + chunk_size).min(num_particles);

      let dyn_array =
        unsafe { core::slice::from_raw_parts_mut(dyn_ptr.get::<DynamicBody>(), num_particles) };
      let kin_array =
        unsafe { core::slice::from_raw_parts(kin_ptr.get::<KinematicBody>(), num_kin) };

      for i in start..end {
        let dyn_body = &mut dyn_array[i];
        if dyn_body.mass > 0.0 {
          dyn_body.transform.position = dyn_body.transform.position + dyn_body.velocity * half_dt;

          let mut f_grav =
            aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]);
          let mut parent_scale = 1.0;
          let mut parent_macro_pos =
            aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]);
          for kin_body in kin_array {
            if kin_body.own_frame_id == dyn_body.parent_frame_id {
              parent_scale = kin_body.scale;
              parent_macro_pos = kin_body.transform.position;
            }
          }

          for kin_body in kin_array {
            if dyn_body.parent_frame_id == kin_body.own_frame_id {
              let r = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0])
                - dyn_body.transform.position;
              let dist_sq = aethervk_oshal_rlib::math::vector::Vector::dot(r, r);
              if dist_sq > 1e-6 {
                let dist = dist_sq.sqrt();
                let local_mu = if kin_body.frame_type == 1 {
                  kin_body.mu / (parent_scale * parent_scale * parent_scale)
                } else {
                  kin_body.mu
                };
                f_grav = f_grav + r * (local_mu * dyn_body.mass / (dist_sq * dist));
              }
            } else if kin_body.frame_type == 0 {
              if dyn_body.parent_frame_id != kin_body.own_frame_id {
                let macro_pos_in_micro =
                  (kin_body.transform.position - parent_macro_pos) / parent_scale;
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

          let inv_mass = 1.0 / dyn_body.mass;
          dyn_body.velocity = dyn_body.velocity + dyn_body.force * (inv_mass * half_dt);
        }
      }
    });

    Ok(())
  }

  fn build_motion_bvh(
    &self,
    _cmd: &mut Self::Cmd,
    dynamics: &Self::Buffer<DynamicBody>,
  ) -> EngineResult<Self::MotionBvh> {
    Ok(CpuMotionBvh {
      dynamics_copy: dynamics.data.clone(),
    })
  }

  fn self_intersect_scene(
    &self,
    _cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
  ) -> EngineResult<Self::List<CollisionPair>> {
    let mut pairs = Vec::new();
    let dynamics = &bvh.dynamics_copy;
    if dynamics.is_empty() {
      return Ok(CpuList { data: pairs });
    }

    use crate::math::collision::bvh_builder::{BVHBuilderParams, BoundNode};
    use crate::physics::particle::{Particle, ParticleBVHBuilder};
    use aethervk_oshal_rlib::math::matrix::mat3::Mat3f32;

    // Group particles by parent_frame_id to build independent BVHs
    let mut frames_map: hashbrown::HashMap<
      u32,
      Vec<(
        usize,
        Particle<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>,
      )>,
    > = hashbrown::HashMap::new();
    for (i, b) in dynamics.iter().enumerate() {
      frames_map.entry(b.parent_frame_id).or_default().push((
        i,
        Particle {
          position: b.transform.position,
          radius: 1.0, // Assume 1.0 for now
        },
      ));
    }

    let builder = ParticleBVHBuilder::new(BVHBuilderParams::default());

    for (_frame_id, frame_particles) in frames_map {
      if frame_particles.len() < 2 {
        continue;
      }
      let just_particles: Vec<_> = frame_particles.iter().map(|(_, p)| *p).collect();
      if let Some(root) = builder.build::<_, _, Mat3f32>(&just_particles) {
        let mut stack = Vec::new();
        stack.push(&*root);

        while let Some(node) = stack.pop() {
          if let (Some(left), Some(right)) = (&node.left, &node.right) {
            let intersects = match (&left.bound, &right.bound) {
              (BoundNode::AABB(a), BoundNode::AABB(b)) => {
                crate::math::collision::intersection::intersect_aabb_aabb::<
                  aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
                >(a, b)
              }
              (BoundNode::OBB(a), BoundNode::OBB(b)) => {
                crate::math::collision::intersection::intersect_aabb_aabb::<
                  aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
                >(
                  &a.to_aabb::<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>(),
                  &b.to_aabb::<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>(),
                )
              }
              _ => false, // fallback
            };

            if intersects {
              // Gather primitives from both and cross-check
              let mut left_prims = Vec::new();
              let mut l_stack = alloc::vec![left.as_ref()];
              while let Some(l_node) = l_stack.pop() {
                if l_node.primitive_indices.is_empty() {
                  if let Some(ll) = &l_node.left {
                    l_stack.push(ll.as_ref());
                  }
                  if let Some(lr) = &l_node.right {
                    l_stack.push(lr.as_ref());
                  }
                } else {
                  left_prims.extend_from_slice(&l_node.primitive_indices);
                }
              }

              let mut right_prims = Vec::new();
              let mut r_stack = alloc::vec![right.as_ref()];
              while let Some(r_node) = r_stack.pop() {
                if r_node.primitive_indices.is_empty() {
                  if let Some(rl) = &r_node.left {
                    r_stack.push(rl.as_ref());
                  }
                  if let Some(rr) = &r_node.right {
                    r_stack.push(rr.as_ref());
                  }
                } else {
                  right_prims.extend_from_slice(&r_node.primitive_indices);
                }
              }

              for &l_idx in &left_prims {
                for &r_idx in &right_prims {
                  let orig_i = frame_particles[l_idx].0;
                  let orig_j = frame_particles[r_idx].0;
                  let b1 = &dynamics[orig_i];
                  let b2 = &dynamics[orig_j];
                  let dist_sq = (b1.transform.position - b2.transform.position).length_squared();
                  let radius = 1.0;
                  if dist_sq < (radius * 2.0) * (radius * 2.0) {
                    pairs.push(CollisionPair {
                      a: crate::gpu::ColliderId {
                        entity_id: slotmap::Key::data(&b1.entity_id).as_ffi() as u32,
                        primitive_index: orig_i as u32,
                      },
                      b: crate::gpu::ColliderId {
                        entity_id: slotmap::Key::data(&b2.entity_id).as_ffi() as u32,
                        primitive_index: orig_j as u32,
                      },
                      time_of_impact: 0.0,
                    });
                  }
                }
              }
            }

            stack.push(left.as_ref());
            stack.push(right.as_ref());
          }
        }
      }
    }

    Ok(CpuList { data: pairs })
  }

  fn intersect_instances(
    &self,
    _cmd: &mut Self::Cmd,
    potentials: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::List<CollisionPair>> {
    Ok(CpuList {
      data: potentials.data.clone(),
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
    dynamics: &mut Self::Buffer<DynamicBody>,
    collisions: &Self::List<CollisionPair>,
    force_inelastic: bool,
  ) -> EngineResult<()> {
    if collisions.data.is_empty() { return Ok(()); }

    let clusters = group_and_cluster_collisions(collisions.data.clone(), 0.01);
    let restitution = if force_inelastic { 0.0 } else { 0.5 };

    let dyn_array = dynamics.data.as_mut_slice();
    let dyn_len = dyn_array.len();

    let max_iters = 20;

    for cluster in clusters {
      let mut impulses = alloc::vec::Vec::with_capacity(cluster.len());
      impulses.resize(cluster.len(), 0.0f32);

      for _iter in 0..max_iters {
        for (i, pair) in cluster.iter().enumerate() {
          let idx_a = pair.a.primitive_index as usize;
          let idx_b = pair.b.primitive_index as usize;
          if idx_a < dyn_len && idx_b < dyn_len {
            let pos_a = dyn_array[idx_a].transform.position;
            let pos_b = dyn_array[idx_b].transform.position;
            let mut normal = pos_a - pos_b;
            let dist = normal.length();
            if dist > 1e-6 {
              normal = normal / dist;
              let v_rel = dyn_array[idx_a].velocity - dyn_array[idx_b].velocity;
              let v_rel_n = aethervk_oshal_rlib::math::vector::Vector::dot(v_rel, normal);

              let m_a = dyn_array[idx_a].mass;
              let m_b = dyn_array[idx_b].mass;
              let inv_m_a = if m_a > 0.0 { 1.0 / m_a } else { 0.0 };
              let inv_m_b = if m_b > 0.0 { 1.0 / m_b } else { 0.0 };

              let a_ii = inv_m_a + inv_m_b;
              if a_ii > 1e-6 {
                let penetration = (2.0 - dist).max(0.0);
                let beta = 0.2;
                let slop = 0.01;
                let bias = (beta / 0.016) * (penetration - slop).max(0.0);

                let b_i = -(1.0 + restitution) * v_rel_n + bias;
                let lambda = b_i / a_ii;

                let old_impulse = impulses[i];
                let new_impulse = (old_impulse + lambda).max(0.0);
                let delta_lambda = new_impulse - old_impulse;
                impulses[i] = new_impulse;

                let impulse_vec = normal * delta_lambda;
                dyn_array[idx_a].velocity = dyn_array[idx_a].velocity + impulse_vec * inv_m_a;
                dyn_array[idx_b].velocity = dyn_array[idx_b].velocity - impulse_vec * inv_m_b;
              }
            }
          }
        }
      }
    }
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
