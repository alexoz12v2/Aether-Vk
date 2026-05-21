//! handoff module.

use crate::scene::{
  EntityId, ErasedMutPtr, ErasedPtr, KinematicComponent, ReferenceFrameComponent, Scene,
  TransformComponent,
  particles::{ParticleData, ParticleSystemComponent},
};
use aethervk_oshal_rlib::{
  math::vector::{Vector, Vector3, vec3::Vec3f32},
  os::pool::{ThreadPool, chunked::ThreadPoolChunkedExt},
};

#[derive(Clone, Debug)]
pub struct FrameData {
  pub entity_id: EntityId,
  pub parent_id: Option<EntityId>,
  pub transform: TransformComponent,
  pub velocity: Vec3f32,
  pub frame: ReferenceFrameComponent,
}

pub struct SpheresOfInfluenceSystem;

impl SpheresOfInfluenceSystem {
  pub fn process_handoffs_par(scene: &Scene, pool: &ThreadPool) {
    let mut all_frames = alloc::vec::Vec::new();
    let mut children_map: hashbrown::HashMap<EntityId, alloc::vec::Vec<usize>> =
      hashbrown::HashMap::new();

    // 1. Gather all frames natively pulling resolved Velocity momentum
    scene.query2::<TransformComponent, ReferenceFrameComponent, _>(|e, t, f| {
      let vel =
        scene.with_component(e, |k: &KinematicComponent| k.velocity).unwrap_or(Vec3f32::zero());
      let data = FrameData {
        entity_id: e,
        parent_id: scene.get_parent(e),
        transform: *t,
        velocity: vel,
        frame: f.clone(),
      };
      all_frames.push(data);
    });

    for (i, frame) in all_frames.iter().enumerate() {
      if let Some(pid) = frame.parent_id {
        children_map.entry(pid).or_default().push(i);
      }
    }

    if all_frames.is_empty() {
      return;
    }

    let mut to_add: alloc::vec::Vec<alloc::vec::Vec<ParticleData>> =
      core::iter::repeat_with(alloc::vec::Vec::new).take(all_frames.len()).collect();

    // 2. Process Escapes and Captures per frame
    for (_i, frame) in all_frames.iter().enumerate() {
      let parent_idx =
        frame.parent_id.and_then(|pid| all_frames.iter().position(|f| f.entity_id == pid));
      let children_indices = children_map.get(&frame.entity_id).cloned().unwrap_or_default();

      scene.with_component(frame.entity_id, |sys: &ParticleSystemComponent| {
        let mut particles = sys.particles.write();
        let num_particles = particles.len();
        if num_particles == 0 {
          return;
        }

        let chunk_size = 2048;
        let num_chunks = (num_particles + chunk_size - 1) / chunk_size;

        let mut escaped_chunks: alloc::vec::Vec<alloc::vec::Vec<ParticleData>> =
          core::iter::repeat_with(alloc::vec::Vec::new).take(num_chunks).collect();

        let mut captured_chunks: alloc::vec::Vec<alloc::vec::Vec<alloc::vec::Vec<ParticleData>>> =
          core::iter::repeat_with(|| {
            core::iter::repeat_with(alloc::vec::Vec::new).take(children_indices.len()).collect()
          })
          .take(num_chunks)
          .collect();

        let particles_ptr = ErasedMutPtr::new(particles.as_mut_ptr());
        let escaped_ptr = ErasedMutPtr::new(escaped_chunks.as_mut_ptr());
        let captured_ptr = ErasedMutPtr::new(captured_chunks.as_mut_ptr());
        let frame_ptr = ErasedPtr::new(frame as *const FrameData);

        let children_data: alloc::vec::Vec<FrameData> =
          children_indices.iter().map(|&idx| all_frames[idx].clone()).collect();
        let children_ptr = ErasedPtr::new(children_data.as_ptr());
        let num_children = children_data.len();
        let has_parent = parent_idx.is_some();

        let handle_res = pool.spawn_chunked(num_chunks, move |chunk_id| {
          let frame = unsafe { &*frame_ptr.get::<FrameData>() };
          let children_slice =
            unsafe { core::slice::from_raw_parts(children_ptr.get::<FrameData>(), num_children) };
          let start = chunk_id * chunk_size;
          let end = (start + chunk_size).min(num_particles);

          let p_array = unsafe {
            core::slice::from_raw_parts_mut(particles_ptr.get::<ParticleData>(), num_particles)
          };
          let escaped_local =
            unsafe { &mut *escaped_ptr.get::<alloc::vec::Vec<ParticleData>>().add(chunk_id) };
          let cap_local_arrays = unsafe {
            &mut *captured_ptr.get::<alloc::vec::Vec<alloc::vec::Vec<ParticleData>>>().add(chunk_id)
          };

          let soi_sq = frame.frame.soi_radius * frame.frame.soi_radius;

          for idx in start..end {
            let p = &mut p_array[idx];
            if p.active == 0 {
              continue;
            }

            let p_pos = Vec3f32::from_array(p.position);

            let mut captured = false;
            for (c_idx, child) in children_slice.iter().enumerate() {
              let dist_vec = p_pos - child.transform.position;
              let soi_child = child.frame.soi_radius * child.frame.scale;

              if dist_vec.length_squared() < (soi_child * soi_child) {
                let p_vel = Vec3f32::from_array(p.velocity);
                let (p_micro, v_micro) = ReferenceFrameComponent::macro_to_micro(
                  p_pos,
                  p_vel,
                  child.transform.position,
                  child.velocity,
                  child.frame.scale,
                );

                let mut captured_particle = p.clone();
                captured_particle.position = [p_micro.x(), p_micro.y(), p_micro.z()];
                captured_particle.velocity = [v_micro.x(), v_micro.y(), v_micro.z()];
                cap_local_arrays[c_idx].push(captured_particle);

                p.active = 0;
                captured = true;
                break;
              }
            }

            if captured {
              continue;
            }

            if has_parent && p_pos.length_squared() > soi_sq {
              let p_vel = Vec3f32::from_array(p.velocity);
              let (p_macro, v_macro) = ReferenceFrameComponent::micro_to_macro(
                p_pos,
                p_vel,
                frame.transform.position,
                frame.velocity,
                frame.frame.scale,
              );

              let mut escaped_particle = p.clone();
              escaped_particle.position = [p_macro.x(), p_macro.y(), p_macro.z()];
              escaped_particle.velocity = [v_macro.x(), v_macro.y(), v_macro.z()];
              escaped_local.push(escaped_particle);

              p.active = 0;
            }
          }
        });

        if let Ok(handle) = handle_res {
          handle.wait();
        }

        particles.retain(|p| p.active != 0);

        if let Some(p_idx) = parent_idx {
          for mut chunk in escaped_chunks {
            to_add[p_idx].append(&mut chunk);
          }
        }

        for cap_arrays in captured_chunks {
          for (c_idx, mut parts) in cap_arrays.into_iter().enumerate() {
            let child_global_idx = children_indices[c_idx];
            to_add[child_global_idx].append(&mut parts);
          }
        }
      });
    }

    // 3. Resolve Buffers back into ECS
    for (i, frame) in all_frames.iter().enumerate() {
      let parts = &mut to_add[i];
      if !parts.is_empty() {
        scene.with_component(frame.entity_id, |sys: &ParticleSystemComponent| {
          sys.particles.write().append(parts);
        });
      }
    }
  }
}
