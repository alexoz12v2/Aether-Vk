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
}

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
      false,
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
      Vec3f32::from_components(0.01, 0.0, 0.0), // 0.01 AU along +x
      Quat::identity(),
      1.0,             // radius_km = 1 km
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
      let lat_deg: f32 = 60.0;
      let lon_deg: f32 = 30.0;
      let lat_rad = lat_deg.to_radians();
      let lon_rad = lon_deg.to_radians();

      // Direction vector from center (spherical coordinates: Z=up, lat from XY plane)
      let dir_z = lat_rad.sin();
      let dir_x = lat_rad.cos() * lon_rad.cos();
      let dir_y = lat_rad.cos() * lon_rad.sin();
      let dir = Vec3f32::from_components(dir_x, dir_y, dir_z).normalize();

      // Raycast from outside inward to find surface point
      let ray_orig = dir * (bounding_sphere * 2.0);
      let ray_dir = Vec3f32::from_components(-dir.x(), -dir.y(), -dir.z());

      let (hit_pt, hit_normal) = if let Some(ref bvh) = mesh_arc.bvh {
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
      };

      println!(
        "[JET] Surface hit at ({:.4}, {:.4}, {:.4}), normal ({:.4}, {:.4}, {:.4})",
        hit_pt[0], hit_pt[1], hit_pt[2], hit_normal[0], hit_normal[1], hit_normal[2]
      );

      // Spawn child entity for the emission sphere
      let child_id = scene_ctx.scene.spawn_entity("JetEmissionSphere");
      scene_ctx.scene.set_parent(child_id, Some(comet_int));

      // 100m sphere = 0.1 km. Scale relative to comet's mesh scale so the
      // child entity renders at 0.1 km in micro-frame coordinates.
      let sphere_radius_km = 0.1;
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
      let mut psc = aethervk_core_rlib::scene::particles::ParticleSystemComponent::new(4096);
      psc.color = [0.2, 0.8, 1.0, 1.0]; // cyan
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

      // Add ParticleEmitterCirclesComponent to the comet with one emission circle
      let emitter = aethervk_core_rlib::scene::particles::ParticleEmitterCirclesComponent {
        circles: vec![aethervk_core_rlib::scene::particles::EmissionCircle {
          latitude_rad: lat_rad,
          longitude_rad: lon_rad,
          circle_radius_km: sphere_radius_km,
          mass: 1e-12, // dust-like
          color: [0.2, 0.8, 1.0, 1.0],
          cached_point: Some(hit_pt),
          cached_normal: Some(hit_normal),
          particles_per_tick: 0, // starts OFF, spacebar toggles
          ttl: 300,              // ~5 seconds at 60fps
          mean_velocity: 0.001,  // 1 m/s in km/s
          velocity_std_dev: 0.3,
          child_entity: Some(child_id),
          beta: 2.0, // perfect reflector
          max_particles: 4096,
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

      let cam_pos = Vec3f32::from_components(0.01001, 0.0, 0.0);
      let rot = <Quat as aethervk_oshal_rlib::math::quaternion::Quaternion>::from_vector_and_scalar(
        Vec3f32::from_components(0.0, 0.0, -std::f32::consts::FRAC_1_SQRT_2),
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
          c.focus_distance = 0.00001;
          // Fix for invisible sun: Avalonia default near plane is 0.1 AU, but we spawn at 0.01 AU!
          if let aethervk_core_rlib::scene::CameraProjection::Perspective {
            ref mut near,
            ref mut far,
            ..
          } = c.projection
          {
            *near = 0.00001; // 0.00001 AU = 1500 km
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
              aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(0.01, 0.0, 0.0);
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
      let new_rate: u32 = if self.jet_emitting { 10 } else { 0 };
      let scene = ctx.get_scene(scene_id).unwrap();
      let scene_read = scene.read();
      if let Some(comet_int) = scene_read.get_entity(self.comet_ext) {
        let _ = scene_read.scene.with_component_mut(
          comet_int,
          |emitter: &mut aethervk_core_rlib::scene::particles::ParticleEmitterCirclesComponent| {
            if let Some(circle) = emitter.circles.first_mut() {
              circle.particles_per_tick = new_rate;
            }
          },
        );
      }
      let state_str = if self.jet_emitting { "ON" } else { "OFF" };
      println!(
        "\x1b[1;36m[JET] Emission {} (particles_per_tick={})\x1b[0m",
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

    // ── J key: move jet between sun-facing and dark-side positions ───────
    // Comet at (0.01,0,0) AU, Sun at origin → sun direction = (-1,0,0)
    // Sun-facing: lat=20°, lon=170°  |  Dark side: lat=20°, lon=350°
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

      let ray_orig = dir * (bounding_sphere * 2.0);
      let ray_dir = Vec3f32::from_components(-dir.x(), -dir.y(), -dir.z());
      println!(
        "[JET-J] ray_orig=({:.4},{:.4},{:.4}) ray_dir=({:.4},{:.4},{:.4})",
        ray_orig.x(),
        ray_orig.y(),
        ray_orig.z(),
        ray_dir.x(),
        ray_dir.y(),
        ray_dir.z()
      );

      let (hit_pt, hit_normal, bvh_hit) = if let Some(ref bvh) = mesh_arc.bvh {
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
      };
      println!(
        "[JET-J] BVH hit: {}, hit_pt=({:.6},{:.6},{:.6}), hit_normal=({:.6},{:.6},{:.6})",
        bvh_hit, hit_pt[0], hit_pt[1], hit_pt[2], hit_normal[0], hit_normal[1], hit_normal[2]
      );

      // Update EmissionCircle lat/lon and cached point
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
        (
          Some(Vec3f32::from_components(0.01, 0.0, 0.0)),
          Some(0.00005),
          None,
        ) // 7,500 km away from comet
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
            Some(Vec3f32::from_components(0.01, 0.0, 0.0)),
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
            t.position = aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components((pos.x() + off) as f64, pos.y() as f64, pos.z() as f64);
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
            c.focus_distance = offset.unwrap_or(0.1);
          },
        );
      }
    }
  }

  fn on_about_to_wait(&mut self, ctx: &mut SimulationContext, scene_id: u64, _delta_time: f32) {
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
        let dist = (t_pos_f32 - Vec3f32::from_components(0.01, 0.0, 0.0)).length();
        let is_micro = dist < 0.005;

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
  };
  run_simulation_app("AetherVk Comet Spawn Test", delegate);
}