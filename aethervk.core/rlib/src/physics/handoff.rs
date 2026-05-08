use crate::scene::{Scene, ReferenceFrameComponent, ReferenceFrameType, TransformComponent, KinematicComponent, EntityId, ErasedPtr, ErasedMutPtr};
use crate::scene::particles::{ParticleSystemComponent, ParticleData};
use aethervk_oshal_rlib::math::vector::{Vector, Vector3};
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::os::pool::ThreadPool;
use aethervk_oshal_rlib::os::pool::chunked::ThreadPoolChunkedExt;

#[derive(Clone, Debug)]
pub struct FrameData {
  pub entity_id: EntityId,
  pub transform: TransformComponent,
  pub velocity: Vec3f32,
  pub frame: ReferenceFrameComponent,
}

pub struct SpheresOfInfluenceSystem;

impl SpheresOfInfluenceSystem {
  pub fn process_handoffs_par(scene: &Scene, pool: &ThreadPool) {
    let mut macro_frame = None;
    let mut micro_frames = alloc::vec::Vec::new();

    // 1. Gather frames natively pulling resolved Velocity momentum
    scene.query2::<TransformComponent, ReferenceFrameComponent, _>(|e, t, f| {
      let vel = scene.with_component(e, |k: &KinematicComponent| k.velocity).unwrap_or(Vec3f32::zero());
      let data = FrameData { entity_id: e, transform: *t, velocity: vel, frame: f.clone() };
      
      if f.frame_type == ReferenceFrameType::Macro {
        macro_frame = Some(data);
      } else if f.frame_type == ReferenceFrameType::Micro {
        micro_frames.push(data);
      }
    });

    let macro_frame = match macro_frame {
      Some(f) => f,
      None => return,
    };

    if micro_frames.is_empty() { return; }

    let mut to_macro = alloc::vec::Vec::new();
    let num_micro = micro_frames.len();
    let mut to_micro: alloc::vec::Vec<alloc::vec::Vec<ParticleData>> = core::iter::repeat_with(alloc::vec::Vec::new).take(num_micro).collect();

    // 2. Micro -> Macro (Escaping Particles)
    for micro in &micro_frames {
      scene.with_component(micro.entity_id, |sys: &ParticleSystemComponent| {
        let mut particles = sys.particles.write();
        let num_particles = particles.len();
        if num_particles == 0 { return; }

        let chunk_size = 2048;
        let num_chunks = (num_particles + chunk_size - 1) / chunk_size;

        let mut escaped_chunks: alloc::vec::Vec<alloc::vec::Vec<ParticleData>> = core::iter::repeat_with(alloc::vec::Vec::new).take(num_chunks).collect();

        let particles_ptr = ErasedMutPtr::new(particles.as_mut_ptr());
        let escaped_ptr = ErasedMutPtr::new(escaped_chunks.as_mut_ptr());
        let micro_ptr = ErasedPtr::new(micro as *const FrameData);

        let handle_res = pool.spawn_chunked(num_chunks, move |chunk_id| {
          let micro = unsafe { &*micro_ptr.get::<FrameData>() };
          let start = chunk_id * chunk_size;
          let end = (start + chunk_size).min(num_particles);
          
          let p_array = unsafe { core::slice::from_raw_parts_mut(particles_ptr.get::<ParticleData>(), num_particles) };
          let escaped_local = unsafe { &mut *escaped_ptr.get::<alloc::vec::Vec<ParticleData>>().add(chunk_id) };

          let soi_sq = micro.frame.soi_radius * micro.frame.soi_radius;

          for i in start..end {
            let p = &mut p_array[i];
            if p.active == 0 { continue; }

            let p_pos = Vec3f32::from_array(p.position);
            
            // Squared length evaluation bypasses extremely slow CPU Square Roots!
            if p_pos.length_squared() > soi_sq {
              let p_vel = Vec3f32::from_array(p.velocity);
              let (p_macro, v_macro) = ReferenceFrameComponent::micro_to_macro(
                p_pos, p_vel, micro.transform.position, micro.velocity, micro.frame.scale,
              );

              let mut escaped_particle = p.clone();
              escaped_particle.position = [p_macro.x(), p_macro.y(), p_macro.z()];
              escaped_particle.velocity = [v_macro.x(), v_macro.y(), v_macro.z()];
              escaped_local.push(escaped_particle);

              p.active = 0; // Lock-free Tombstone allows memory disjoint modifications!
            }
          }
        });
        
        if let Ok(handle) = handle_res { handle.wait(); } 
        
        particles.retain(|p| p.active != 0); // Compress array instantly dropping dead particles
        for mut chunk in escaped_chunks { to_macro.append(&mut chunk); }
      });
    }

    // 3. Macro -> Micro (Captured Particles)
    scene.with_component(macro_frame.entity_id, |sys: &ParticleSystemComponent| {
      let mut particles = sys.particles.write();
      let num_particles = particles.len();
      if num_particles == 0 { return; }

      let chunk_size = 2048;
      let num_chunks = (num_particles + chunk_size - 1) / chunk_size;

      let mut captured_chunks: alloc::vec::Vec<alloc::vec::Vec<alloc::vec::Vec<ParticleData>>> = 
        core::iter::repeat_with(|| core::iter::repeat_with(alloc::vec::Vec::new).take(num_micro).collect()).take(num_chunks).collect();

      let particles_ptr = ErasedMutPtr::new(particles.as_mut_ptr());
      let captured_ptr = ErasedMutPtr::new(captured_chunks.as_mut_ptr());
      let micros_ptr = ErasedPtr::new(micro_frames.as_ptr());

      let handle_res = pool.spawn_chunked(num_chunks, move |chunk_id| {
        let start = chunk_id * chunk_size;
        let end = (start + chunk_size).min(num_particles);
        
        let p_array = unsafe { core::slice::from_raw_parts_mut(particles_ptr.get::<ParticleData>(), num_particles) };
        let micros_slice = unsafe { core::slice::from_raw_parts(micros_ptr.get::<FrameData>(), num_micro) };
        let cap_local_arrays = unsafe { &mut *captured_ptr.get::<alloc::vec::Vec<alloc::vec::Vec<ParticleData>>>().add(chunk_id) };

        for i in start..end {
          let p = &mut p_array[i];
          if p.active == 0 { continue; }

          let p_pos = Vec3f32::from_array(p.position);

          for (m_idx, micro) in micros_slice.iter().enumerate() {
            let dist_vec = p_pos - micro.transform.position;
            let soi_macro = micro.frame.soi_radius * micro.frame.scale;
            
            if dist_vec.length_squared() < (soi_macro * soi_macro) {
              let p_vel = Vec3f32::from_array(p.velocity);
              let (p_micro, v_micro) = ReferenceFrameComponent::macro_to_micro(
                p_pos, p_vel, micro.transform.position, micro.velocity, micro.frame.scale,
              );

              let mut captured_particle = p.clone();
              captured_particle.position = [p_micro.x(), p_micro.y(), p_micro.z()];
              captured_particle.velocity = [v_micro.x(), v_micro.y(), v_micro.z()];
              cap_local_arrays[m_idx].push(captured_particle);

              p.active = 0; // Terminate in macro!
              break; 
            }
          }
        }
      });

      if let Ok(handle) = handle_res { handle.wait(); } 

      particles.retain(|p| p.active != 0);

      // Extract thread matrices securely 
      for cap_arrays in captured_chunks {
        for (m_idx, mut parts) in cap_arrays.into_iter().enumerate() {
          to_micro[m_idx].append(&mut parts);
        }
      }
    });

    // 4. Resolve Buffers back into ECS
    if !to_macro.is_empty() {
      scene.with_component(macro_frame.entity_id, |sys: &ParticleSystemComponent| {
        sys.particles.write().extend(to_macro);
      });
    }

    for (idx, micro_particles) in to_micro.into_iter().enumerate() {
      if !micro_particles.is_empty() {
        scene.with_component(micro_frames[idx].entity_id, |sys: &ParticleSystemComponent| {
          sys.particles.write().extend(micro_particles);
        });
      }
    }
  }
}
