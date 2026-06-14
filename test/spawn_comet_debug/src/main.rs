use aethervk_core_rlib::{
  gpu::PresentationEngineHandle, scene::camera::QuatToEulerAngles,
  simulation_api::SimulationContext, types::EngineResult,
};
use aethervk_oshal_rlib::math::{
  matrix::SquareMatrix,
  quaternion::Quaternion,
  vector::{Vector, Vector3, Vector4, vec3::Vec3f32, vec4::Quat},
};
use std::sync::Arc;
use test_utils::{
  cycle_get_asset_path_from_exe,
  sim_app::{SimulationDelegate, run_simulation_app},
};
use winit::window::Window;

// To enable force_debug (GPU shader printf + low emission rate + infinite TTL):
//   cargo run -p spawn_comet_debug --release --features force_debug

struct SpawnCometDelegate {
  camera_ext_entity: u64,
  was_micro: Option<bool>,
  /// External ID of the comet entity (for jet manipulation).
  comet_ext: u64,
  /// Scene ID (for jet manipulation).
  scene_id: u64,
  /// Whether the jet is currently emitting particles.
  jet_emitting: bool,
  /// Whether the jet is on the sun-facing side (true) or dark side (false).
  jet_sunlit: bool,
  /// EMA of frame duration in milliseconds for a stable FPS display.
  ema_frame_ms: f32,
  /// Ring index into SUB_DT_RING — M key cycles through physics sub-step presets.
  sub_dt_idx: usize,
}

/// Physics sub-step presets cycled by the M key (seconds).
/// None = use TimeScale default (currently 100 s for OneDay).
const SUB_DT_RING: &[Option<f64>] = &[
  None,          // auto  (100 s for OneDay)
  Some(1440.0),  // 1 sub-step / day  — original blob behaviour
  Some(100.0),   // ~14 sub-steps / day
  Some(10.0),    // ~144 sub-steps / day
  Some(1.0),     // ~1440 sub-steps / day — finest grain
];

impl SimulationDelegate for SpawnCometDelegate {
  fn create_scene(&mut self, ctx: &mut SimulationContext) -> EngineResult<u64> {
    ctx.create_default_scene(true)
  }

  fn on_setup(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    pe_handle: PresentationEngineHandle,
    _window: &Window,
  ) -> EngineResult<()> {
    let assets_dir = cycle_get_asset_path_from_exe(true);
    let comet_path = assets_dir.join("Comet2.glb");
    aethervk_core_rlib::gpu::ASSET_DIR
      .write()
      .replace(assets_dir.to_string_lossy().to_string());

    let loaded_mesh = aethervk_core_rlib::simulation::comet::load_comet_from_gltf(
      comet_path.to_str().unwrap(),
      false,  // build BVH — required for sun-side occlusion check in emit_particles_from_circles
      None,
    )
    .expect("Failed to load comet mesh");

    let model_id = {
      let mut scenes = ctx.scenes.write();
      scenes.import_model_from_mesh("comet_model".to_string(), loaded_mesh)
    };

    let (_lca_ext, comet_ext) = ctx.spawn_comet_internal(
      scene_id,
      model_id,
      "comet",
      Vec3f32::from_components(1.0, 0.0, 0.0), // 1.0 AU along +x (realistic heliocentric distance)
      Quat::identity(),
      20.0,            // radius_km = 20 km (user-visible comet body)
      1000.0,          // mass_kg
      0,               // physics_type = static
      0,               // comet id (ignored when static)
      None,            // rotational_model
      Vec3f32::zero(), // angular_velocity
    )?;
    self.comet_ext = comet_ext;
    self.scene_id = scene_id;

    // ── Pre-configured jet at lat=60°, lon=30° ──────────────────────────────
    // Use a read lock for scene graph mutations (they use interior component
    // locks). Only register_entity() needs a write lock on SceneContext.
    let child_id_to_register;
    {
      let scene_lock = ctx.get_scene(scene_id).unwrap();
      let scene_ctx = scene_lock.read();
      let comet_int = scene_ctx.get_entity(comet_ext).expect("comet entity not found");

      // Get the mesh for raycasting
      let mesh_arc = scene_ctx
        .scene
        .with_component(
          comet_int,
          |p: &aethervk_core_rlib::scene::PhysicalMeshComponent| p.mesh.clone(),
        )
        .expect("comet missing PhysicalMeshComponent");

      let bounding_sphere = mesh_arc
        .vertices
        .iter()
        .map(|v| {
          (v.position[0] * v.position[0]
            + v.position[1] * v.position[1]
            + v.position[2] * v.position[2])
            .sqrt()
        })
        .fold(0.0_f32, f32::max)
        .max(0.01);
      println!(
        "[JET] BVH available: {}, bounding_sphere: {:.4}",
        mesh_arc.bvh.is_some(),
        bounding_sphere
      );

      // Comet transform scale
      let comet_scale = scene_ctx
        .scene
        .with_component(
          comet_int,
          |t: &aethervk_core_rlib::scene::TransformComponent| t.scale.x(),
        )
        .unwrap_or(1.0);

      // Jet parameters
      // lat=0°, lon=90° → emission direction = +y (equatorial, perpendicular to sun-comet +x axis).
      // Radiation pressure (+x) bends the trajectory from +y toward +x, giving a
      // clearly visible CURVED tail — the classic comet-tail geometry.
      // At lat=60°/lon=30° the emission was nearly parallel to radiation (+x), so
      // everything appeared straight.
      let lat_deg: f32 = 0.0;
      let lon_deg: f32 = 90.0;
      let lat_rad = lat_deg.to_radians();
      let lon_rad = lon_deg.to_radians();

      // Direction vector from center (spherical coordinates: Z=up, lat from XY plane)
      let dir_z = lat_rad.sin();
      let dir_x = lat_rad.cos() * lon_rad.cos();
      let dir_y = lat_rad.cos() * lon_rad.sin();
      let dir = Vec3f32::from_components(dir_x, dir_y, dir_z).normalize();

      // ── Jet surface placement ─────────────────────────────────────────────
      // USE_BVH_RAYCAST = false  → always place on the bounding sphere (fast,
      //                           deterministic, no miss case).
      // USE_BVH_RAYCAST = true   → raycast into the mesh BVH for exact surface
      //                           hit (useful for irregular shapes in Avalonia).
      const USE_BVH_RAYCAST: bool = false;

      let (hit_pt, hit_normal) = if USE_BVH_RAYCAST {
        // ── BVH raycast path ─────────────────────────────────────────────────
        let ray_orig = dir * (bounding_sphere * 2.0);
        let ray_dir = Vec3f32::from_components(-dir.x(), -dir.y(), -dir.z());
        if let Some(ref bvh) = mesh_arc.bvh {
          match bvh.raycast(ray_orig, ray_dir, &mesh_arc.vertices, &mesh_arc.indices) {
            Some((_t, pt, n)) => ([pt.x(), pt.y(), pt.z()], [n.x(), n.y(), n.z()]),
            None => {
              let pt = dir * bounding_sphere;
              ([pt.x(), pt.y(), pt.z()], [dir.x(), dir.y(), dir.z()])
            }
          }
        } else {
          let pt = dir * bounding_sphere;
          ([pt.x(), pt.y(), pt.z()], [dir.x(), dir.y(), dir.z()])
        }
      } else {
        // ── Bounding-sphere surface path (default) ───────────────────────────
        // Place the jet at the point on the bounding sphere in direction `dir`.
        // Normal is the outward radial direction (same as dir).
        let pt = dir * bounding_sphere;
        ([pt.x(), pt.y(), pt.z()], [dir.x(), dir.y(), dir.z()])
      };

      println!(
        "[JET] use_bvh={} surface=({:.4},{:.4},{:.4}) normal=({:.4},{:.4},{:.4})",
        USE_BVH_RAYCAST,
        hit_pt[0], hit_pt[1], hit_pt[2], hit_normal[0], hit_normal[1], hit_normal[2],
      );

      // Spawn child entity for the emission sphere
      let child_id = scene_ctx.scene.spawn_entity("JetEmissionSphere");
      scene_ctx.scene.set_parent(child_id, Some(comet_int));

      // 5 km sphere sits on the surface of the 20 km comet body:
      // sphere_scale = desired_radius_km / comet_scale (mesh units → km).
      let sphere_radius_km = 5.0;
      let sphere_scale = (sphere_radius_km / comet_scale).max(1e-4);
      println!(
        "[JET] comet_scale={:.6}, bounding_sphere={:.4}, sphere_scale={:.6}",
        comet_scale, bounding_sphere, sphere_scale
      );

      let _ = scene_ctx.scene.add_component(
        child_id,
        aethervk_core_rlib::scene::TransformComponent {
          position: Vec3f32::from_components(hit_pt[0], hit_pt[1], hit_pt[2]),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(sphere_scale, sphere_scale, sphere_scale),
        },
      );

      // StaticMeshComponent: emissive cyan sphere
      let sphere_mesh =
        aethervk_core_rlib::simulation::comet::generate_uv_sphere(1.0, 8, 8, 1.0, true);
      let _ = scene_ctx.scene.add_component(
        child_id,
        aethervk_core_rlib::scene::StaticMeshComponent {
          asset_path: "primitives/sphere.obj".into(),
          mesh: std::sync::Arc::from(sphere_mesh),
          emissive_color: [0.2, 0.8, 1.0, 1.0], // cyan
          is_visible: true,
        },
      );

      // ParticleSystemComponent for this jet
      let mut psc = aethervk_core_rlib::scene::particles::ParticleSystemComponent::new(1000000);
      psc.color = [0.2, 0.8, 1.0, 1.0]; // cyan
      psc.particle_radius = 0.2;          // 200m physics collision radius
      // Camera is 100 km from comet (sunward side, looking +x into the tail).
      // 10 km billboard → arctan(10/100)≈5.7° (≈7 px): clearly visible near comet.
      // With 166 sub-step compensation particles spanning 0–204 km at ~1.2 km
      // spacing, all 10 km-radius discs overlap → smooth, unbroken stream.
      psc.render_radius_km = 10.0;
      // beta=2.0 → net outward = (2-1)×GM/r² = 5.93e-6 km/s².
      // Per dispatch (166×49.96s = 8294 sim-s): particles spread from
      // 0 km (sub-step 165) to 204 km (sub-step 0) → strongly visible tail.
      psc.beta = 2.0;
      // Comet dust is a test-particle system: particles don't attract each other.
      // Skipping BVH construction + Barnes-Hut self-gravity saves ~95% GPU time at
      // high particle counts and avoids the GPU TDR watchdog hang.
      psc.disable_self_gravity = true;
      let _ = scene_ctx.scene.add_component(child_id, psc);

      // SphereGizmoComponent for visual debugging
      let _ = scene_ctx.scene.add_component(
        child_id,
        aethervk_core_rlib::scene::SphereGizmoComponent {
          radius: sphere_radius_km,
          subdivisions: 3.0,
          local_frame: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::identity(),
          is_visible: true,
        },
      );

      let emitter = aethervk_core_rlib::scene::particles::ParticleEmitterCirclesComponent {
        circles: vec![aethervk_core_rlib::scene::particles::EmissionCircle {
          latitude_rad: lat_rad,
          longitude_rad: lon_rad,
          circle_radius_km: sphere_radius_km,
          mass: 1e-12, // dust-like
          color: [0.2, 0.8, 1.0, 1.0],
          cached_point: Some(hit_pt),
          cached_normal: Some(hit_normal),
          particles_per_second: 0.0, // starts OFF, spacebar toggles
          // 1000 sub-steps × 49.96s/sub-step = 49960 sim-s ≈ 13.9 sim-hours lifetime.
          // Particle position at death: 0.5×5.93e-6×49960² ≈ 7400 km from comet.
          // Pool fills at 25 p/sub-step × 166 sub-steps/dispatch × 90fps = ~373k/real-s,
          // hitting 1M capacity in ~2.7 real-s then cycling at steady state.
          ttl: {
            #[cfg(feature = "force_debug")] { 0 }        // never expire: observe full trajectory
            #[cfg(not(feature = "force_debug"))] { 1000 } // normal: 1000 sub-step lifetime
          },
          // 0.1 km/s initial velocity perpendicular to radiation pressure (+x):
          // radiation pressure bends the trajectory from +y toward +x over time.
          // 0.05 km/s std-dev gives a wide fan that fills gaps between sub-step groups.
          mean_velocity: 0.1,
          velocity_std_dev: 0.05,
          child_entity: Some(child_id),
          // Must match psc.beta so the physics shader and CPU compensation agree.
          beta: 2.0,
          max_particles: 1000000,
          // Scatter spawn positions over a disc 2× the comet radius so the
          // tail origin looks like a diffuse surface patch, not a point.
          spawn_radius_km: sphere_radius_km * 2.0,
        }],
      };
      let _ = scene_ctx.scene.add_component(comet_int, emitter);

      println!(
        "[JET] Pre-configured jet at lat={}°, lon={}° | beta={} | Press SPACEBAR to toggle emission",
        lat_deg, lon_deg, 2.0
      );

      child_id_to_register = child_id;
    }

    // Brief write lock just for register_entity
    {
      let scene_lock = ctx.get_scene(scene_id).unwrap();
      let mut scene_ctx = scene_lock.write();
      let _child_ext = scene_ctx.register_entity(child_id_to_register);
    }

    // Spawn Earth Orbit for debugging
    let bsp_path = assets_dir.join("planets/de442.bsp");
    let bpc_path = assets_dir.join("earth_latest_high_prec.bpc");
    let _ = ctx.threads.logic_thread.tx().try_send(
      aethervk_core_rlib::simulation_api::structs::LogicCommand::LoadAlmanac {
        task_id: 0,
        path: bsp_path.to_string_lossy().into_owned(),
      },
    );
    let _ = ctx.threads.logic_thread.tx().try_send(
      aethervk_core_rlib::simulation_api::structs::LogicCommand::LoadAlmanac {
        task_id: 0,
        path: bpc_path.to_string_lossy().into_owned(),
      },
    );

    let earth_ext = ctx.spawn_entity(scene_id, "Earth Orbit").unwrap();

    {
      let scene_lock = ctx.get_scene(scene_id).unwrap();
      let scene_read = scene_lock.read();
      if let Some(earth_int) = scene_read.get_entity(earth_ext) {
        scene_read.scene.set_parent(earth_int, Some(scene_read.root_entity));
        if let Err(e) = scene_read.scene.add_component(
          earth_int,
          aethervk_core_rlib::scene::TransformComponent::default(),
        ) {
          println!("[DEBUG] Failed to add TransformComponent: {:?}", e);
        }
        if let Err(e) = scene_read.scene.add_component(
          earth_int,
          aethervk_core_rlib::scene::AlmanacPlanet::new(399, 0.0, 1.0),
        ) {
          println!("[DEBUG] Failed to add AlmanacPlanet: {:?}", e);
        }
      }
    }

    let year_in_sec = 31557600.0;
    let _ = ctx.threads.logic_thread.tx().try_send(
      aethervk_core_rlib::simulation_api::structs::LogicCommand::UpdateTrajectoryForSpk {
        task_id: 0,
        scene_id,
        entity_id: earth_ext,
        spk_id: 399,
        start_epoch_tai_sec: 0.0,
        end_epoch_tai_sec: year_in_sec,
        sample_step_days: 10.0,
      },
    );

    let scene_lock = ctx.get_scene(scene_id).unwrap();
    {
      let scene_read = scene_lock.read();

      let camera_int = {
        let mut found = None;
        for (_ext_id, &int_id) in scene_read.entity_map.iter() {
          if scene_read
            .scene
            .has_component::<aethervk_core_rlib::scene::CameraComponent>(int_id)
            .into()
          {
            found = Some(int_id);
            break;
          }
        }
        found.expect("No camera found in scene")
      };

      for (&ext_id, &int_id) in scene_read.entity_map.iter() {
        if int_id == camera_int {
          self.camera_ext_entity = ext_id;
          break;
        }
      }

      scene_read
        .presentation_engines
        .write()
        .get_mut(&pe_handle)
        .unwrap()
        .camera_entity = Some(camera_int);

      // Camera 100 km from the comet on the SUNWARD side, looking anti-sunward (+x).
      // Comet nucleus is +100 km ahead. Dust tail extends from +100 km to +15,000+ km
      // ahead — fully in the field of view.
      // (Previous bug: camera was on the anti-sunward side looking -x → tail was behind.)
      const KM_PER_AU: f64 = 149_597_870.7;
      let cam_offset_au = 100.0_f64 / KM_PER_AU; // ≈ 6.685e-7 AU
      // Sunward side: subtract the offset from the comet's x=1.0 AU position.
      let cam_pos = Vec3f32::from_components((1.0 - cam_offset_au) as f32, 0.0, 0.0);
      // Rotation: −90° around +y rotates camera default -z forward to +x.
      // Quaternion: (w=1/√2, x=0, y=−1/√2, z=0)
      let rot = <Quat as aethervk_oshal_rlib::math::quaternion::Quaternion>::from_vector_and_scalar(
        Vec3f32::from_components(0.0, -std::f32::consts::FRAC_1_SQRT_2, 0.0),
        std::f32::consts::FRAC_1_SQRT_2,
      );
      let _ = scene_read.scene.with_component_mut(
        camera_int,
        |t: &mut aethervk_core_rlib::scene::HighResTransformComponent| {
          t.position = aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
            cam_pos.x() as f64,
            cam_pos.y() as f64,
            cam_pos.z() as f64,
          );
          t.rotation = rot;
        },
      );
      let _ = scene_read.scene.with_component_mut(
        camera_int,
        |c: &mut aethervk_core_rlib::scene::CameraComponent| {
          // Focus at 100 km (where the comet nucleus is).
          c.focus_distance = cam_offset_au as f32;
          if let aethervk_core_rlib::scene::CameraProjection::Perspective {
            ref mut near,
            ref mut far,
            ..
          } = c.projection
          {
            // Near: 0.5 km = 3.34e-9 AU — safely inside the 100 km camera offset.
            *near = (0.5 / KM_PER_AU) as f32;
            *far = 10000.0;
          }
        },
      );

      let mut cursor_ent = None;
      if let Some((id, _)) = scene_read
        .scene
        .query1_first_res::<aethervk_core_rlib::scene::CursorComponent, _, _>(|id, _| Some(id))
      {
        cursor_ent = Some(id);
      }
      if let Some(id) = cursor_ent {
        let _ = scene_read.scene.with_component_mut(
          id,
          |t: &mut aethervk_core_rlib::scene::HighResTransformComponent| {
            t.position =
              aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(1.0, 0.0, 0.0);
          },
        );
      }
    }

    {
      let scene_read = scene_lock.read();
      let mut ts = scene_read.time_state.write();
      ts.is_playing = false; // starts paused — press P to play
      ts.current_scale = aethervk_core_rlib::simulation_api::structs::TimeScale::OneDay;
    }

    Ok(())
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
    let camera_entity = scene.read().get_entity(self.camera_ext_entity).expect(&format!(
      "No camera entity with id {} in scene {}",
      self.camera_ext_entity, scene_id
    ));

    let logic_command = test_utils::command::process_mouse_motion_camera_commands(
      delta,
      middle_mouse_down,
      shift_down,
      ctrl_down,
      camera_entity,
      Arc::clone(&scene),
    );

    if let Some(logic_command) = logic_command {
      let _ = ctx.threads.logic_thread.tx().try_send(logic_command);
    }
  }

  fn on_mouse_wheel(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    delta: winit::event::MouseScrollDelta,
  ) {
    let amount = match delta {
      winit::event::MouseScrollDelta::LineDelta(_, y) => y * 0.5,
      winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y * 0.01) as f32,
    };

    let scene = ctx.get_scene(scene_id).unwrap();
    let camera_entity = scene.read().get_entity(self.camera_ext_entity).unwrap();

    let _ = ctx.threads.logic_thread.tx().try_send(
      aethervk_core_rlib::simulation_api::structs::LogicCommand::ZoomCamera(
        aethervk_core_rlib::simulation_api::structs::ZoomCamera {
          camera_entity,
          scene,
          amount,
        },
      ),
    );
  }

  fn on_keyboard_input(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    event: &winit::event::KeyEvent,
    _modifiers: winit::keyboard::ModifiersState,
  ) {
    if event.state != winit::event::ElementState::Pressed {
      return;
    }

    // ── Spacebar: toggle jet emission ─────────────────────────────────────
    // On macOS, winit delivers space as Character(" ") rather than Named(Space)
    let is_space = matches!(
      event.logical_key.as_ref(),
      winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space)
        | winit::keyboard::Key::Character(" ")
    );
    if is_space {
      self.jet_emitting = !self.jet_emitting;
      // 0.5 p/sim-s: at OneDay/50s sub-steps → 0.5×49.96 ≈ 25 particles per GPU
      // sub-step. 166 sub-steps per dispatch × 25 = 4150 particles, spread 0–204 km
      // by the per-sub-step velocity compensation. render_radius=10 km means all
      // 25-particle bundles overlap → uniform density stream with no visible gaps.
      // force_debug: slow emission (10 p/s) so individual particle paths are visible.
      // Normal mode: 5000 p/s fills the tail quickly.
      let on_rate: f32 = if cfg!(feature = "force_debug") { 10.0 } else { 5000.0 };
      let new_rate: f32 = if self.jet_emitting { on_rate } else { 0.0 };
      let scene = ctx.get_scene(scene_id).unwrap();
      let scene_read = scene.read();
      if let Some(comet_int) = scene_read.get_entity(self.comet_ext) {
        let _ = scene_read.scene.with_component_mut(
          comet_int,
          |emitter: &mut aethervk_core_rlib::scene::particles::ParticleEmitterCirclesComponent| {
            if let Some(circle) = emitter.circles.first_mut() {
              circle.particles_per_second = new_rate;
            }
          },
        );
      }
      let state_str = if self.jet_emitting { "ON" } else { "OFF" };
      println!(
        "\x1b[1;36m[JET] Emission {} (particles_per_second={})\x1b[0m",
        state_str, new_rate
      );
      return;
    }

    // ── P key: toggle play/pause ──────────────────────────────────────────
    if matches!(
      event.logical_key.as_ref(),
      winit::keyboard::Key::Character("p") | winit::keyboard::Key::Character("P")
    ) {
      let scene = ctx.get_scene(scene_id).unwrap();
      let scene_read = scene.read();
      let was_playing = scene_read.time_state.read().is_playing;
      scene_read.time_state.write().is_playing = !was_playing;
      let state_str = if !was_playing { "PLAYING" } else { "PAUSED" };
      println!("\x1b[1;33m[SIM] {}\x1b[0m", state_str);
      return;
    }

    // ── M key: ring-toggle physics sub-step size ──────────────────────────
    // Cycles through SUB_DT_RING presets so you can verify whether clustering
    // and force magnitude are step-size artifacts.
    if matches!(
      event.logical_key.as_ref(),
      winit::keyboard::Key::Character("m") | winit::keyboard::Key::Character("M")
    ) {
      self.sub_dt_idx = (self.sub_dt_idx + 1) % SUB_DT_RING.len();
      let preset = SUB_DT_RING[self.sub_dt_idx];
      let scene = ctx.get_scene(scene_id).unwrap();
      let scene_read = scene.read();
      scene_read.time_state.write().max_sub_dt_override = preset;
      let label = match preset {
        None    => "auto (100 s)".to_string(),
        Some(s) => format!("{} s", s),
      };
      let n_steps = preset.map_or(
        (1440.0_f64 / 100.0).ceil() as u32,
        |s| (1440.0_f64 / s).ceil() as u32,
      );
      println!("\x1b[1;35m[SUB-DT] max_sub_dt = {}  ({} sub-steps/day)\x1b[0m",
        label, n_steps);
      return;
    }

    // ── J key: move jet between sun-facing and dark-side positions ───────
    // Comet at (1.0,0,0) AU, Sun at origin → sun direction = (-1,0,0)
    // Sun-facing: lat=20°, lon=170°  |  Dark side: lat=20°, lon=350°
    // Pressing J also clears all existing particles so the old position
    // doesn't linger (long TTL would otherwise show both positions emitting).
    if matches!(
      event.logical_key.as_ref(),
      winit::keyboard::Key::Character("j") | winit::keyboard::Key::Character("J")
    ) {
      self.jet_sunlit = !self.jet_sunlit;
      let (lat_deg, lon_deg): (f32, f32) = if self.jet_sunlit {
        (20.0, 170.0) // sun-facing
      } else {
        (20.0, 350.0) // dark side
      };
      let lat_rad = lat_deg.to_radians();
      let lon_rad = lon_deg.to_radians();

      let scene = ctx.get_scene(scene_id).unwrap();
      let scene_read = scene.read();
      let comet_int = match scene_read.get_entity(self.comet_ext) {
        Some(id) => id,
        None => return,
      };

      // Get mesh + scale for raycasting
      let mesh_arc = match scene_read.scene.with_component(
        comet_int,
        |p: &aethervk_core_rlib::scene::PhysicalMeshComponent| p.mesh.clone(),
      ) {
        Some(m) => m,
        None => return,
      };
      let comet_scale = scene_read
        .scene
        .with_component(
          comet_int,
          |t: &aethervk_core_rlib::scene::TransformComponent| t.scale.x(),
        )
        .unwrap_or(1.0);

      let bounding_sphere = mesh_arc
        .vertices
        .iter()
        .map(|v| {
          (v.position[0] * v.position[0]
            + v.position[1] * v.position[1]
            + v.position[2] * v.position[2])
            .sqrt()
        })
        .fold(0.0_f32, f32::max)
        .max(0.01);
      println!(
        "[JET-J] BVH available: {}, bounding_sphere: {:.4}",
        mesh_arc.bvh.is_some(),
        bounding_sphere
      );

      let dir_z = lat_rad.sin();
      let dir_x = lat_rad.cos() * lon_rad.cos();
      let dir_y = lat_rad.cos() * lon_rad.sin();
      let dir = Vec3f32::from_components(dir_x, dir_y, dir_z).normalize();

      // ── Jet surface placement (same switch as initial placement) ─────────
      const USE_BVH_RAYCAST_J: bool = false;

      let (hit_pt, hit_normal, bvh_hit) = if USE_BVH_RAYCAST_J {
        // ── BVH raycast path ─────────────────────────────────────────────────
        let ray_orig = dir * (bounding_sphere * 2.0);
        let ray_dir = Vec3f32::from_components(-dir.x(), -dir.y(), -dir.z());
        println!(
          "[JET-J] ray_orig=({:.4},{:.4},{:.4}) ray_dir=({:.4},{:.4},{:.4})",
          ray_orig.x(), ray_orig.y(), ray_orig.z(),
          ray_dir.x(), ray_dir.y(), ray_dir.z()
        );
        if let Some(ref bvh) = mesh_arc.bvh {
          match bvh.raycast(ray_orig, ray_dir, &mesh_arc.vertices, &mesh_arc.indices) {
            Some((_t, pt, n)) => ([pt.x(), pt.y(), pt.z()], [n.x(), n.y(), n.z()], true),
            None => {
              let pt = dir * bounding_sphere;
              ([pt.x(), pt.y(), pt.z()], [dir.x(), dir.y(), dir.z()], false)
            }
          }
        } else {
          let pt = dir * bounding_sphere;
          ([pt.x(), pt.y(), pt.z()], [dir.x(), dir.y(), dir.z()], false)
        }
      } else {
        // ── Bounding-sphere surface path (default) ───────────────────────────
        let pt = dir * bounding_sphere;
        ([pt.x(), pt.y(), pt.z()], [dir.x(), dir.y(), dir.z()], false)
      };
      println!(
        "[JET-J] use_bvh={} bvh_hit={} hit_pt=({:.6},{:.6},{:.6}) hit_normal=({:.6},{:.6},{:.6})",
        USE_BVH_RAYCAST_J, bvh_hit,
        hit_pt[0], hit_pt[1], hit_pt[2], hit_normal[0], hit_normal[1], hit_normal[2]
      );

      // Clear all existing particles so the old emission position doesn't persist.
      // (Long TTL means particles from lat=60°/lon=30° would otherwise linger
      //  alongside new particles, giving the "both positions emit" illusion.)
      let mut child_entity = None;
      let _ = scene_read.scene.with_component_mut(
        comet_int,
        |emitter: &mut aethervk_core_rlib::scene::particles::ParticleEmitterCirclesComponent| {
          if let Some(circle) = emitter.circles.first_mut() {
            circle.latitude_rad = lat_rad;
            circle.longitude_rad = lon_rad;
            circle.cached_point = Some(hit_pt);
            circle.cached_normal = Some(hit_normal);
            child_entity = circle.child_entity;
          }
        },
      );
      if let Some(child_id) = child_entity {
        let _ = scene_read.scene.with_component_mut(
          child_id,
          |psc: &mut aethervk_core_rlib::scene::particles::ParticleSystemComponent| {
            // Kill all alive particles immediately.
            let mut pool = psc.particles.write();
            for p in pool.iter_mut() { p.active = 0; }
            psc.gpu_alive_count = 0;
          },
        );
        println!("[JET-J] Cleared all particles from old emission position.");
      }

      // Update child entity position
      if let Some(child_id) = child_entity {
        let sphere_scale = (0.1 / comet_scale).max(1e-4);
        let updated = scene_read.scene.with_component_mut(
          child_id,
          |t: &mut aethervk_core_rlib::scene::TransformComponent| {
            t.position = Vec3f32::from_components(hit_pt[0], hit_pt[1], hit_pt[2]);
            t.scale = Vec3f32::from_components(sphere_scale, sphere_scale, sphere_scale);
          },
        );
        println!(
          "[JET-J] child_id={:?} updated={:?} sphere_scale={:.6}",
          child_id,
          updated.is_some(),
          sphere_scale
        );
      } else {
        println!("[JET-J] WARNING: no child_entity found in EmissionCircle!");
      }

      let side = if self.jet_sunlit {
        "SUN-FACING"
      } else {
        "DARK SIDE"
      };
      println!(
        "\x1b[1;35m[JET] Moved to {} (lat={}°, lon={}°) hit=({:.3},{:.3},{:.3})\x1b[0m",
        side, lat_deg, lon_deg, hit_pt[0], hit_pt[1], hit_pt[2]
      );
      return;
    }

    let (target_pos, offset, target_parent) = match event.logical_key.as_ref() {
      winit::keyboard::Key::Character("f") | winit::keyboard::Key::Character("F") => {
        // 50 km from comet nucleus, sunward side — same geometry as startup.
        let close_offset = 50.0_f32 / 149_597_870.7_f32;
        (
          Some(Vec3f32::from_components((1.0 - close_offset) as f32, 0.0, 0.0)),
          Some(close_offset as f64),  // will be overridden below; just moves position
          None,
        )
      }
      winit::keyboard::Key::Character("0") => {
        (
          Some(Vec3f32::from_components(0.0, 0.0, 0.0)),
          Some(0.02),
          None,
        ) // 0.02 AU away from sun
      }
      winit::keyboard::Key::Character("m") | winit::keyboard::Key::Character("M") => {
        let scene = ctx.get_scene(scene_id).unwrap();
        let scene_read = scene.read();
        if let Some(camera_ent) = scene_read.get_entity(self.camera_ext_entity) {
          let mut curr = camera_ent;
          print!("[DEBUG] Camera {:?} parent hierarchy: ", camera_ent);
          while let Some(parent) = scene_read.scene.get_parent(curr) {
            print!("{:?} -> ", parent);
            curr = parent;
          }
          println!("(root)");
        }
        (None, None, None)
      }
      winit::keyboard::Key::Named(winit::keyboard::NamedKey::Tab) => {
        let scene = ctx.get_scene(scene_id).unwrap();
        let scene_read = scene.read();
        if scene_read.time_state.read().is_playing {
          println!("[DEBUG] Cannot switch observer mode while simulation is playing. Pause first.");
          (None, None, None)
        } else {
          println!("[DEBUG] Switched to Earth Observer Mode.");
          let earth_int = scene_read.scene.get_entity_by_name("Earth Orbit").unwrap();
          (
            Some(Vec3f32::from_components(1.0, 0.0, 0.0)),
            None,
            Some(earth_int),
          )
        }
      }
      _ => (None, None, None),
    };

    if let Some(pos) = target_pos {
      let scene = ctx.get_scene(scene_id).unwrap();
      let scene_read = scene.read();

      let mut cursor_ent = None;
      if let Some((id, _)) = scene_read
        .scene
        .query1_first_res::<aethervk_core_rlib::scene::CursorComponent, _, _>(|id, _| Some(id))
      {
        cursor_ent = Some(id);
      }
      if let Some(id) = cursor_ent {
        let _ = scene_read.scene.with_component_mut(
          id,
          |t: &mut aethervk_core_rlib::scene::HighResTransformComponent| {
            t.position = aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
              pos.x() as f64,
              pos.y() as f64,
              pos.z() as f64,
            );
          },
        );
      }

      if let Some(camera_int) = scene_read.get_entity(self.camera_ext_entity) {
        // Reparent to root so that our local position is treated as macro scale (AU).
        // The logic thread will automatically reparent it back to the micro frame
        // if the new global position happens to fall inside its SOI.
        let new_parent = target_parent.unwrap_or(scene_read.root_entity);
        scene_read.scene.set_parent(camera_int, Some(new_parent));

        if let Some(off) = offset {
          let _ = scene_read.scene.with_component_mut(camera_int, |t: &mut aethervk_core_rlib::scene::HighResTransformComponent| {
            t.position = aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(pos.x() as f64 + off, pos.y() as f64, pos.z() as f64);
            // Look towards -X, Up is +Z
            t.rotation = <aethervk_oshal_rlib::math::vector::vec4::Quat as aethervk_oshal_rlib::math::quaternion::Quaternion>::from_vector_and_scalar(
              Vec3f32::from_components(0.0, 0.0, -std::f32::consts::FRAC_1_SQRT_2),
              std::f32::consts::FRAC_1_SQRT_2,
            );
          });
        }
        let _ = scene_read.scene.with_component_mut(
          camera_int,
          |c: &mut aethervk_core_rlib::scene::CameraComponent| {
            c.focus_distance = offset.unwrap_or(0.1) as f32;
          },
        );
      }
    }
  }

  fn on_about_to_wait(&mut self, ctx: &mut SimulationContext, scene_id: u64, delta_time: f32) {
    // ── Performance HUD ───────────────────────────────────────────────────────
    // Smooth the raw per-call delta_time with a slow EMA (α=0.05 ≈ 20-frame
    // window) so the display is readable even when on_about_to_wait fires at
    // very high rate now that physics no longer blocks the winit event loop.
    {
      let dt_ms = delta_time * 1000.0;
      if self.ema_frame_ms <= 0.0 {
        self.ema_frame_ms = dt_ms; // first frame bootstrap
      } else {
        self.ema_frame_ms = 0.05 * dt_ms + 0.95 * self.ema_frame_ms;
      }
      let fps = if self.ema_frame_ms > 0.0 { (1000.0 / self.ema_frame_ms).min(9999.0) } else { 0.0 };

      let scene = ctx.get_scene(scene_id).unwrap();
      let scene_read = scene.read();

      let sim_speed = scene_read.time_state.read().effective_sim_speed;

      // Count total alive particles + gather radiation-pressure diagnostic stats.
      // Comet is at [0.01, 0, 0] AU → sun-to-comet direction is +x in micro-frame km.
      // If radiation pressure (beta=1.5) is working, mean vel_x should grow > mean_velocity
      // and mean pos_x (centroid drift from comet) should grow outward each tick.
      let mut total_particles: u64 = 0;
      let mut sum_vel_x: f64 = 0.0;
      let mut sum_pos_x: f64 = 0.0;
      let mut max_speed: f64 = 0.0;
      let mut n_sampled: u64 = 0;
      scene_read.scene.query1::<
        aethervk_core_rlib::scene::particles::ParticleSystemComponent, _
      >(|_, psc| {
        total_particles += psc.gpu_alive_count as u64;
        let guard = psc.particles.read();
        for p in guard.iter().filter(|p| p.active != 0) {
          let vx = p.velocity[0] as f64;
          let vy = p.velocity[1] as f64;
          let vz = p.velocity[2] as f64;
          sum_vel_x += vx;
          sum_pos_x += p.position[0] as f64;
          let speed = (vx * vx + vy * vy + vz * vz).sqrt();
          if speed > max_speed { max_speed = speed; }
          n_sampled += 1;
        }
      });

      let mean_vel_x  = if n_sampled > 0 { sum_vel_x / n_sampled as f64 } else { 0.0 };
      let mean_pos_x  = if n_sampled > 0 { sum_pos_x / n_sampled as f64 } else { 0.0 };

      use std::io::Write;
      let sub_dt_label = {
        let scene = ctx.get_scene(scene_id).unwrap();
        let scene_read = scene.read();
        match scene_read.time_state.read().max_sub_dt_override {
          None    => "dt=auto".to_string(),
          Some(s) => format!("dt={:.0}s", s),
        }
      };
      print!(
        "\r\x1b[2K[PERF] FPS: {:5.1} | Sim: {:5.2}\u{00d7} | Particles: {:>9} | {} | \
         v\u{2093}={:+.3e} km/s | x\u{0304}={:+.3e} km | v_max={:.3e} km/s",
        fps, sim_speed, total_particles, sub_dt_label, mean_vel_x, mean_pos_x, max_speed
      );
      let _ = std::io::stdout().flush();

      // Every 240 frames print a full radiation-pressure proof line.
      // mean_vel_x should be growing positively (anti-sunward +x) if beta>1 is working.
      use core::sync::atomic::{AtomicU64, Ordering};
      static RAD_DIAG_COUNTER: AtomicU64 = AtomicU64::new(0);
      let diag_frame = RAD_DIAG_COUNTER.fetch_add(1, Ordering::Relaxed);
      if diag_frame % 240 == 0 && n_sampled > 0 {
        // ── Analytical expected values under pure radiation pressure at 1 AU ────
        // Net acceleration: a = (beta-1) × GM_sun / r²
        //   beta = 2.0, GM_sun = 1.327124e11 km³/s², r = 1 AU = 149597870.7 km
        //   a = 5.930e-6 km/s²  (anti-sunward, +x for comet at +1 AU)
        //
        // Within one dispatch (N sub-steps, T = N×dt ≈ 8294 s):
        //   x(i) = 0.5 × a × ((N-i)×dt)²   for sub-step i = 0 … N-1
        //   mean_x = a × T² / 6               (average over uniform i)
        //   mean_vx = a × T / 2               (mean particle velocity after 1 dispatch)
        // For a longer steady-state tail (multiple dispatches), values grow.
        const BETA: f64 = 2.0;
        const GM_SUN: f64 = 1.327_124e11;   // km³/s²
        const R_1AU: f64  = 149_597_870.7;  // km
        let a_net: f64 = (BETA - 1.0) * GM_SUN / (R_1AU * R_1AU); // ≈ 5.93e-6 km/s²
        // At OneDay scale, batched dispatch ≈ 6 frames × (1/60 s × 86400 s/day) = 8640 sim-s.
        // With max_sub_dt=50s: N≈66×2=166, T≈8294s (hardcoded approx, see [EMIT-DIAG]).
        let t_dispatch: f64 = 8_294.0; // sim-s per dispatch
        let exp_mean_x  = a_net * t_dispatch * t_dispatch / 6.0;  // km
        let exp_mean_vx = a_net * t_dispatch / 2.0;               // km/s
        println!(
          "\n[RAD-PROOF] frame={} alive={}",
          diag_frame, n_sampled,
        );
        println!(
          "  actual:   mean_vx={:+.4e} km/s  mean_x={:+.4e} km  v_max={:.4e} km/s",
          mean_vel_x, mean_pos_x, max_speed,
        );
        println!(
          "  expected: mean_vx≈{:+.4e} km/s  mean_x≈{:+.4e} km  (1 dispatch, beta=2, r=1AU)",
          exp_mean_vx, exp_mean_x,
        );
        println!(
          "  ratio:    vx_ratio={:.3}  x_ratio={:.3}  (expected ≈1.0 once tail fills)",
          if exp_mean_vx.abs() > 1e-12 { mean_vel_x / exp_mean_vx } else { 0.0 },
          if exp_mean_x.abs() > 1e-12 { mean_pos_x / exp_mean_x } else { 0.0 },
        );
      }

      // Every 120 frames: verbose particle state dump for the first 5 alive particles.
      // Prints position, velocity, and cluster spread (σ_x) to diagnose:
      //  - Are all particles at the same location? (spread ≈ 0 → still a blob)
      //  - Is velocity growing anti-sunward? (+vx with beta>1 = radiation push working)
      //  - Are particles clearly separated? (|pos_k - pos_0| > render_radius_km = 1 km)
      static VERBOSE_COUNTER: AtomicU64 = AtomicU64::new(0);
      let verbose_frame = VERBOSE_COUNTER.fetch_add(1, Ordering::Relaxed);
      if verbose_frame % 120 == 0 && n_sampled > 0 {
        // Collect first 5 alive particles from the PSC
        let scene_diag = ctx.get_scene(scene_id).unwrap();
        let scene_diag_read = scene_diag.read();
        let comet_int_diag = scene_diag_read.get_entity(self.comet_ext);
        if let Some(comet_int_d) = comet_int_diag {
          let child_entity = scene_diag_read
            .scene
            .with_component(
              comet_int_d,
              |em: &aethervk_core_rlib::scene::particles::ParticleEmitterCirclesComponent| {
                em.circles.first().and_then(|c| c.child_entity)
              },
            )
            .flatten();
          if let Some(child_id) = child_entity {
            scene_diag_read.scene.with_component(
              child_id,
              |psc: &aethervk_core_rlib::scene::particles::ParticleSystemComponent| {
                let pool = psc.particles.read();
                let alive: Vec<_> = pool
                  .iter()
                  .filter(|p| p.active != 0)
                  .take(5)
                  .collect();
                if alive.is_empty() {
                  println!("[PARTICLES] No alive particles yet.");
                  return;
                }
                // Compute position spread (σ_x) across sampled alive particles
                let all_alive_n = pool.iter().filter(|p| p.active != 0).count();
                let (sx, sy, sz) = pool.iter()
                  .filter(|p| p.active != 0)
                  .fold((0.0_f64, 0.0_f64, 0.0_f64), |(ax, ay, az), p| {
                    (ax + p.position[0] as f64, ay + p.position[1] as f64, az + p.position[2] as f64)
                  });
                let (mx, my, mz) = (sx / all_alive_n as f64, sy / all_alive_n as f64, sz / all_alive_n as f64);
                let var_x = pool.iter()
                  .filter(|p| p.active != 0)
                  .map(|p| (p.position[0] as f64 - mx).powi(2))
                  .sum::<f64>() / all_alive_n as f64;
                let sigma_x = var_x.sqrt();
                // Read render_radius_km from the PSC — was previously hardcoded as 1.0 (bug).
                let r_km = psc.render_radius_km;
                println!(
                  "\n[PARTICLE-DIAG] frame={} alive={} pos_mean=({:.2},{:.2},{:.2}) km  σ_x={:.3} km  render_r={:.1} km",
                  verbose_frame, all_alive_n, mx, my, mz, sigma_x, r_km
                );
                println!("  [PARTICLE-DIAG] First {} alive particles:", alive.len());
                // Overlap threshold = 2 × render_radius_km (discs touch when dist < 2r)
                let overlap_threshold_km = 2.0 * r_km as f64;
                for (i, p) in alive.iter().enumerate() {
                  println!(
                    "    p[{}]: pos=({:+.4e},{:+.4e},{:+.4e}) km  vel=({:+.4e},{:+.4e},{:+.4e}) km/s  active={}",
                    i, p.position[0], p.position[1], p.position[2],
                    p.velocity[0], p.velocity[1], p.velocity[2], p.active
                  );
                }
                // Separation between p[0] and p[1]
                if alive.len() >= 2 {
                  let dx = (alive[0].position[0] - alive[1].position[0]) as f64;
                  let dy = (alive[0].position[1] - alive[1].position[1]) as f64;
                  let dz = (alive[0].position[2] - alive[1].position[2]) as f64;
                  let sep = (dx*dx + dy*dy + dz*dz).sqrt();
                  let overlap = if sep < overlap_threshold_km {
                    format!("OVERLAP (sep={:.1} km < 2×{:.0} km)", sep, r_km)
                  } else {
                    format!("SEPARATED (gap={:.1} km)", sep - overlap_threshold_km)
                  };
                  println!("    [PARTICLE-DIAG] p[0]-p[1] separation: {:.4e} km  => {}", sep, overlap);
                }
              },
            );
          }
        }
      }

    }
    // ── End HUD ───────────────────────────────────────────────────────────────

    let scene = ctx.get_scene(scene_id).unwrap();
    let scene_read = scene.read();
    if let Some(camera_entity) = scene_read.get_entity(self.camera_ext_entity) {
      if let Some(t) = scene_read.scene.with_component(
        camera_entity,
        |t: &aethervk_core_rlib::scene::HighResTransformComponent| t.clone(),
      ) {
        let t_pos_f32 = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
          t.position.x() as f32,
          t.position.y() as f32,
          t.position.z() as f32,
        );
        let dist = (t_pos_f32 - Vec3f32::from_components(1.0, 0.0, 0.0)).length();
        let is_micro = dist < 0.5; // comet is at 1.0 AU; micro-frame SOI ≈ 0.5 AU radius

        if let Some(was_micro) = self.was_micro {
          if was_micro != is_micro {
            let color = "\x1b[1;31m"; // Bold Red
            let reset = "\x1b[0m";
            if is_micro {
              println!(
                "{}*** CAMERA ENTERED MICRO FRAME (dist to comet: {:.6} AU) ***{}",
                color, dist, reset
              );
            } else {
              println!(
                "{}*** CAMERA EXITED MICRO FRAME (dist to comet: {:.6} AU) ***{}",
                color, dist, reset
              );
            }
          }
        }
        self.was_micro = Some(is_micro);

        use core::sync::atomic::{AtomicU64, Ordering};
        static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);
        let frame = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        // if frame % 12000 == 0 {
        //   let (pitch, yaw) = t.rotation.to_pitch_yaw();
        //   println!(
        //     "[CAMERA] Frame {} | Pos: ({:.6}, {:.6}, {:.6}) | Rot Quat: (w={:.4}, x={:.4}, y={:.4}, z={:.4}) | Euler: (pitch={:.2}°, yaw={:.2}°)",
        //     frame,
        //     t.position.x(),
        //     t.position.y(),
        //     t.position.z(),
        //     t.rotation.0.w(),
        //     t.rotation.0.x(),
        //     t.rotation.0.y(),
        //     t.rotation.0.z(),
        //     pitch.to_degrees(),
        //     yaw.to_degrees()
        //   );
        // }
      }
    }
  }
}

fn main() {
  // force_debug: enable GPU printf shaders before Vulkan instance creation
  #[cfg(feature = "force_debug")]
  {
    aethervk_core_rlib::gpu_backends::vulkan::physics::USE_PRINTF_SHADERS
      .store(true, core::sync::atomic::Ordering::SeqCst);
    println!("[FORCE_DEBUG] USE_PRINTF_SHADERS enabled — shader printf active");
  }

  let _assets_dir = cycle_get_asset_path_from_exe(true);

  //#[cfg(debug_assertions)]
  //{
  //  // A flag to keep the thread spinning
  //  let mut wait: bool = true;
  //  // We use read_volatile so the compiler doesn't optimize the loop away
  //  while unsafe { core::ptr::read_volatile(&wait) } {
  //    // Just spin. (Note: this will pin one CPU core at 100% while waiting)
  //    core::hint::spin_loop();
  //  }
  //}

  let delegate = SpawnCometDelegate {
    camera_ext_entity: 0,
    was_micro: None,
    comet_ext: 0,
    scene_id: 0,
    jet_emitting: false,
    jet_sunlit: false,
    ema_frame_ms: 0.0,
    sub_dt_idx: 0,   // starts at SUB_DT_RING[0] = None (auto 100 s)
  };
  run_simulation_app("AetherVk Comet Spawn Test", delegate);
}