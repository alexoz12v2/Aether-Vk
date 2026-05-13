use aethervk_core_rlib::gpu::PresentationEngineHandle;
use aethervk_core_rlib::scene::{PhysicalMeshComponent, SunComponent, TransformComponent};
use aethervk_core_rlib::scene::{GaussianParams, ParticleEmitterConfig, ParticleSystemComponent};
use aethervk_core_rlib::simulation_api::SimulationContext;
use aethervk_core_rlib::types::EngineResult;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::{vec3::Vec3f32, vec4::Quat, Vector, Vector3};
use std::sync::Arc;
use test_utils::cycle_get_asset_path_from_exe;
use test_utils::sim_app::{run_simulation_app, SimulationDelegate};
use test_utils::simulation::kernels::CpuKernels;
use winit::window::Window;

struct ParticleTestDelegate {
  camera_ext_entity: u64,
  mesh_ext_entity: u64,
  sun_ext_entity: u64,
  particle_sys_1_entity: u64,
  particle_sys_2_entity: u64,
  kernels: CpuKernels,
}

impl SimulationDelegate for ParticleTestDelegate {
  fn create_scene(&mut self, ctx: &mut SimulationContext) -> EngineResult<u64> {
    ctx.create_empty_scene()
  }

  fn on_setup(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    pe_handle: PresentationEngineHandle,
    _window: &Window,
  ) -> EngineResult<()> {
    let root_entity = ctx.spawn_entity(scene_id, "root").unwrap();
    ctx
      .add_transform_component(
        scene_id,
        root_entity,
        Vec3f32::zero(),
        Quat::identity(),
        Vec3f32::one(),
      )
      .unwrap();

    let camera_entity = ctx
      .add_perspective_camera(
        scene_id,
        pe_handle,
        "camera",
        std::f32::consts::FRAC_PI_4,
        0.1,
        100.0,
      )
      .unwrap()
      .get();
    
    let cam_dist = 5.0;
    ctx.set_parent(scene_id, camera_entity, root_entity).unwrap();
    ctx
      .set_transform_component(
        scene_id,
        camera_entity,
        Vec3f32::from_array([0.0, -cam_dist, 0.0]),
        Quat::from_axis_angle(Vec3f32::from_array([0.0, 0.0, 1.0]), std::f32::consts::PI),
        Vec3f32::one(),
      )
      .unwrap();
    self.camera_ext_entity = camera_entity;

    let mesh_entity = ctx.spawn_entity(scene_id, "mesh").unwrap();
    ctx.set_parent(scene_id, mesh_entity, root_entity).unwrap();
    ctx
      .add_transform_component(
        scene_id,
        mesh_entity,
        Vec3f32::zero(),
        Quat::identity(),
        Vec3f32::one(),
      )
      .unwrap();
    self.mesh_ext_entity = mesh_entity;

    let asset_path = cycle_get_asset_path_from_exe(false);
    let loaded_mesh = aethervk_core_rlib::simulation::comet::load_comet_from_gltf(
      asset_path.join("Comet.glb").to_str().unwrap(),
      false,
    )
    .expect("Failed to load mesh");

    let uv_dist = build_uv_distribution(&loaded_mesh, 64);
    let uv_grid = aethervk_core_rlib::simulation::comet::uv_grid::UvGrid::new(
        &loaded_mesh.vertices,
        &loaded_mesh.indices,
        64,
    );

    let mesh_arc = Arc::from(loaded_mesh);

    if let Some(scene_ctx) = ctx.get_scene(scene_id) {
      let mut active_scene = scene_ctx.write();
      let mesh_internal_id = active_scene.get_entity(mesh_entity).unwrap();
      active_scene.scene.add_component(
        mesh_internal_id,
        PhysicalMeshComponent {
          asset_path: asset_path.join("Comet.glb").to_str().unwrap().to_string(),
          mesh: mesh_arc.clone(),
          emissive_intensity: 0.0,
          emissive_color: [0.0, 0.0, 0.0],
          use_new_path: false,
          paint_display_mode: 0,
        },
      ).unwrap();
    }

    let particle_sys_1 = ctx.spawn_entity(scene_id, "particle_sys_1").unwrap();
    ctx.set_parent(scene_id, particle_sys_1, root_entity).unwrap();
    ctx
      .add_transform_component(
        scene_id,
        particle_sys_1,
        Vec3f32::zero(),
        Quat::identity(),
        Vec3f32::one(),
      )
      .unwrap();
    self.particle_sys_1_entity = particle_sys_1;

    if let Some(scene_ctx) = ctx.get_scene(scene_id) {
      let mut active_scene = scene_ctx.write();
      let sys1_internal_id = active_scene.get_entity(particle_sys_1).unwrap();
      active_scene.scene.add_component(
        sys1_internal_id,
        ParticleSystemComponent::new(ParticleEmitterConfig {
          uv_distribution: uv_dist.clone(),
          delta: 100_000,
          max_particles: 1000,
          velocity_intensity: GaussianParams {
            mean: 5.0,
            std_dev: 1.0,
            min: 0.0,
            max: 10.0,
          },
          emission_count: GaussianParams {
            mean: 10.0,
            std_dev: 2.0,
            min: 1.0,
            max: 20.0,
          },
          particle_radius: 0.1,
          density: 1000.0,
          lifetime: 5_000_000,
          color: [1.0, 0.5, 0.0, 1.0],
          beta: 0.1,
          use_particle2: false,
        }),
      ).unwrap();
    }

    let particle_sys_2 = ctx.spawn_entity(scene_id, "particle_sys_2").unwrap();
    ctx.set_parent(scene_id, particle_sys_2, root_entity).unwrap();
    ctx
      .add_transform_component(
        scene_id,
        particle_sys_2,
        Vec3f32::zero(),
        Quat::identity(),
        Vec3f32::one(),
      )
      .unwrap();
    self.particle_sys_2_entity = particle_sys_2;

    if let Some(scene_ctx) = ctx.get_scene(scene_id) {
      let mut active_scene = scene_ctx.write();
      let sys2_internal_id = active_scene.get_entity(particle_sys_2).unwrap();
      active_scene.scene.add_component(
        sys2_internal_id,
        ParticleSystemComponent::new(ParticleEmitterConfig {
          uv_distribution: uv_dist.clone(),
          delta: 100_000,
          max_particles: 500,
          velocity_intensity: GaussianParams {
            mean: 3.0,
            std_dev: 0.5,
            min: 0.0,
            max: 5.0,
          },
          emission_count: GaussianParams {
            mean: 5.0,
            std_dev: 1.0,
            min: 1.0,
            max: 10.0,
          },
          particle_radius: 0.05,
          density: 500.0,
          lifetime: 3_000_000,
          color: [0.0, 0.5, 1.0, 1.0],
          beta: 0.5,
          use_particle2: false,
        }),
      ).unwrap();
    }

    let sun_entity = ctx.spawn_entity(scene_id, "sun").unwrap();
    ctx.set_parent(scene_id, sun_entity, root_entity).unwrap();
    ctx
      .add_transform_component(
        scene_id,
        sun_entity,
        Vec3f32::from_array([100.0, 0.0, 0.0]),
        Quat::identity(),
        Vec3f32::one(),
      )
      .unwrap();
    ctx.add_sun_component(scene_id, sun_entity, (64, 64, 64), 1.0).unwrap();
    self.sun_ext_entity = sun_entity;

    // ADD UPDATE COMPONENT
    let updater = ctx.spawn_entity(scene_id, "updater").unwrap();
    if let Some(scene_ctx) = ctx.get_scene(scene_id) {
        let mut active_scene = scene_ctx.write();
        let internal_updater = active_scene.get_entity(updater).unwrap();
        let internal_mesh = active_scene.get_entity(self.mesh_ext_entity).unwrap();
        let internal_sys1 = active_scene.get_entity(self.particle_sys_1_entity).unwrap();
        let internal_sys2 = active_scene.get_entity(self.particle_sys_2_entity).unwrap();

        active_scene.scene.add_component(internal_updater, aethervk_core_rlib::scene::script_components::UpdateComponent {
            entities: [Some(internal_mesh), Some(internal_sys1), Some(internal_sys2), None],
            arbitrary_data: [0.0; 4],
            callback: |_, scene, entities_arr, _data, dt| {
                // Decode ids
                let mesh_id = entities_arr[0].unwrap();
                let sys1_id = entities_arr[1].unwrap();
                let sys2_id = entities_arr[2].unwrap();

                let mut comet_pos = Vec3f32::zero();
                let mut comet_rot = Quat::identity();
                let mut mesh_arc_opt = None;

                scene.with_component(mesh_id, |t: &TransformComponent| {
                    comet_pos = t.position;
                    comet_rot = t.rotation;
                });
                mesh_arc_opt = scene.with_component(mesh_id, |m: &PhysicalMeshComponent| m.mesh.clone());

                if let Some(mesh_arc) = mesh_arc_opt {
                    let uv_grid = aethervk_core_rlib::simulation::comet::uv_grid::UvGrid::new(
                        &mesh_arc.vertices,
                        &mesh_arc.indices,
                        64,
                    );

                    let update_particles = |sys_entity: aethervk_core_rlib::scene::EntityId| {
                        scene.with_component_mut(
                            sys_entity,
                            |sys: &mut ParticleSystemComponent| {
                                sys.accumulator += (dt * 1_000_000.0) as i64;

                                // Emission
                                while sys.accumulator >= sys.config.delta {
                                    sys.accumulator -= sys.config.delta;

                                    let u_emission = [rand::random::<f32>(), rand::random::<f32>()];

                                    let count = sys.config.emission_count.sample(&u_emission) as usize;
                                    let mut u_particles = std::vec::Vec::with_capacity(count);
                                    for _ in 0..count {
                                        u_particles.push([
                                            rand::random::<f32>(),
                                            rand::random::<f32>(),
                                            rand::random::<f32>(),
                                            rand::random::<f32>(),
                                        ]);
                                    }

                                    sys.emit_particles(
                                        &mesh_arc,
                                        &uv_grid,
                                        comet_pos,
                                        comet_rot,
                                        Vec3f32::from_components(1.0, 1.0, 1.0),
                                        &u_emission,
                                        &u_particles,
                                    );
                                }

                                // Update
                                let mut u_roulette = std::vec::Vec::with_capacity(sys.particles.read().len());
                                for _ in 0..sys.particles.read().len() {
                                    u_roulette.push(rand::random::<f32>());
                                }

                                let mut roulette_idx = 0;
                                for p in sys.particles.write().iter_mut().filter(|p| p.active != 0) {
                                    let mut age = p.get_age();
                                    age += (dt * 1_000_000.0) as i64;
                                    p.set_age(age);

                                    // Russian roulette
                                    if age > sys.config.lifetime as i64 {
                                        let age_excess = (age - sys.config.lifetime as i64) as f32 / 1_000_000.0;
                                        let death_prob = 1.0 - (-age_excess).exp(); // Exponential decay

                                        let u = if roulette_idx < u_roulette.len() {
                                            u_roulette[roulette_idx]
                                        } else {
                                            0.5
                                        };
                                        roulette_idx += 1;

                                        if u < death_prob {
                                            p.active = 0;
                                            continue;
                                        }
                                    }
                                }

                                sys.particles.write().retain(|p| p.active != 0);
                                sys.update_bvh();
                            },
                        );
                    };

                    update_particles(sys1_id);
                    update_particles(sys2_id);
                }
            }
        }).unwrap();
    }


    let _ = ctx.threads.logic_thread.tx().try_send(
      aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id },
    );

    Ok(())
  }

  fn on_about_to_wait(&mut self, ctx: &mut SimulationContext, scene_id: u64, delta_time: f32) {
    if delta_time <= 0.0 {
      return;
    }

    if let Some(scene_ctx) = ctx.get_scene(scene_id) {
      let mut physics_scene =
        aethervk_core_rlib::physics::physics_scene::PhysicsScene::build_from_scene(
          &scene_ctx.read().scene,
        );
      // Faking time since this is simplified delegate
      let t0 = 0;
      let t1 = (delta_time * 1_000_000.0) as i64;

      aethervk_core_rlib::gpu_backends::simulation_step(
        &self.kernels,
        &mut physics_scene,
        &scene_ctx.read().scene,
        t0,
        t1,
      )
      .unwrap();
    }
  }

  fn on_mouse_motion(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    delta: (f64, f64),
    middle_mouse_down: bool,
    shift_down: bool,
    ctrl_down: bool,
  ) {
    let scene = ctx.get_scene(scene_id).unwrap();
    let camera_entity = scene.read().get_entity(self.camera_ext_entity).unwrap();
    if middle_mouse_down {
      if let Some(logic_command) = test_utils::command::process_mouse_motion_camera_commands(
        delta,
        middle_mouse_down,
        shift_down,
        ctrl_down,
        camera_entity,
        scene.clone(),
      ) {
        let _ = ctx.threads.logic_thread.tx().try_send(logic_command);
      }
    }
  }

  fn on_mouse_wheel(
    &mut self,
    _ctx: &mut SimulationContext,
    _scene_id: u64,
    _delta: winit::event::MouseScrollDelta,
  ) {
  }
}

fn build_uv_distribution(
  comet: &aethervk_core_rlib::simulation::comet::Comet,
  resolution: usize,
) -> aethervk_core_rlib::math::distribution::Distribution2D {
  let mut min_u = std::f32::MAX;
  let mut max_u = std::f32::MIN;
  let mut min_v = std::f32::MAX;
  let mut max_v = std::f32::MIN;

  for v in &comet.vertices {
    min_u = min_u.min(v.uv[0]);
    max_u = max_u.max(v.uv[0]);
    min_v = min_v.min(v.uv[1]);
    max_v = max_v.max(v.uv[1]);
  }
  if max_u <= min_u {
    max_u = min_u + 1.0;
  }
  if max_v <= min_v {
    max_v = min_v + 1.0;
  }

  let mut weights = vec![0.0; resolution * resolution];
  let inv_w = resolution as f32 / (max_u - min_u);
  let inv_h = resolution as f32 / (max_v - min_v);

  for chunk in comet.indices.chunks_exact(3) {
    let i0 = chunk[0] as usize;
    let i1 = chunk[1] as usize;
    let i2 = chunk[2] as usize;
    let v0 = &comet.vertices[i0];
    let v1 = &comet.vertices[i1];
    let v2 = &comet.vertices[i2];

    let p0 = Vec3f32::from_array(v0.position);
    let p1 = Vec3f32::from_array(v1.position);
    let p2 = Vec3f32::from_array(v2.position);
    let area = 0.5 * (p1 - p0).cross(p2 - p1).length();

    let cu = (v0.uv[0] + v1.uv[0] + v2.uv[0]) / 3.0;
    let cv = (v0.uv[1] + v1.uv[1] + v2.uv[1]) / 3.0;

    let mut x = if cu < min_u {
      0
    } else {
      ((cu - min_u) * inv_w) as usize
    };
    let mut y = if cv < min_v {
      0
    } else {
      ((cv - min_v) * inv_h) as usize
    };
    if x >= resolution {
      x = resolution - 1;
    }
    if y >= resolution {
      y = resolution - 1;
    }

    weights[y * resolution + x] += area;
  }

  aethervk_core_rlib::math::distribution::Distribution2D::new(&weights, resolution, resolution)
}

fn main() {
  let delegate = ParticleTestDelegate {
    camera_ext_entity: 0,
    mesh_ext_entity: 0,
    sun_ext_entity: 0,
    particle_sys_1_entity: 0,
    particle_sys_2_entity: 0,
    kernels: CpuKernels::new(),
  };

  run_simulation_app("Particle Test", delegate);
}