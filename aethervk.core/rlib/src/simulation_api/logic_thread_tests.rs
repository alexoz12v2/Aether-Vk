use super::*;
use crate::scene::AlmanacPlanet;
use crate::scene::trajectory::TrajectoryComponent;
use crate::simulation_api::external_state::CCometInitialized;
use crate::simulation_api::structs::{KeplerianElements, LogicCommand};
use crate::simulation_api::{SimulationContext, set_external_state_simulation_callback};
use hifitime::{Duration, Epoch};
use std::sync::mpsc;

static MOCK_SENDER: parking_lot::Mutex<Option<mpsc::Sender<CCometInitialized>>> =
  parking_lot::Mutex::new(None);

unsafe extern "C" fn mock_external_state_cb(state_id: u32, data_ptr: *const core::ffi::c_void) {
  if state_id == 4 {
    // CometInitialized
    let comet_init = unsafe { *(data_ptr as *const CCometInitialized) };
    if let Some(sender) = MOCK_SENDER.lock().as_ref() {
      let _ = sender.send(comet_init);
    }
  }
}

#[test]
fn test_reset_simulation_forces_reposition() {
  unsafe { std::env::set_var("ASSET_DIR", "../../assets") };
  let mut ctx = SimulationContext::startup(None).expect("Failed to create SimulationContext");
  {
    let mut logic_state = ctx.logic_state.write();
    logic_state
      .almanac_data
      .load_almanac("../../assets/planets/pck00011.pca")
      .expect("Failed to load PCK");
    logic_state
      .almanac_data
      .load_almanac("../../assets/planets/gm_de431.pca")
      .expect("Failed to load GM");
    logic_state
      .almanac_data
      .load_almanac("../../assets/planets/de442.bsp")
      .expect("Failed to load DE442");
    logic_state
      .almanac_data
      .load_almanac("../../assets/earth_latest_high_prec.bpc")
      .expect("Failed to load BPC");
    println!("LOADED FILES: {:?}", logic_state.almanac_data.file_names);
  }

  let start = Epoch::from_gregorian_utc(2025, 10, 15, 0, 0, 0, 0);
  let end = start + Duration::from_days(10.0);
  let scene_ret = ctx.create_empty_scene2(true, start, end).expect("Failed to create empty scene");
  let scene_id = scene_ret.scene_id;

  let (earth_body, expected_residual) = {
    let scenes = ctx.scenes.read();
    let scene_arc = scenes.get_scene(scene_id).unwrap();
    let scene_guard = scene_arc.read();
    let earth = scene_guard.earth.unwrap();

    // Scramble the transform artificially
    let _ = scene_guard.scene.with_component_mut(
      earth.body,
      |t: &mut crate::scene::TransformComponent| {
        t.position.0[0] += 50000.0;
      },
    );

    let planet = scene_guard
      .scene
      .with_component(earth.body, |p: &crate::scene::AlmanacPlanet| *p)
      .unwrap();

    let logic_state = ctx.logic_state.read();
    let (pos_km, _) = planet.step(start, &logic_state.almanac_data, None).unwrap();

    use aethervk_oshal_rlib::math::vector::{Vector3, vec3::Vec3f32, vec3f64::DVec3};
    let subtree_pos_f32: Vec3f32 = (pos_km * 6.6845871226706e-9_f64).to_f32();
    let subtree_km = DVec3::from_components(
      subtree_pos_f32.x() as f64,
      subtree_pos_f32.y() as f64,
      subtree_pos_f32.z() as f64,
    ) * 149_597_870.7_f64;
    let residual = (pos_km - subtree_km).to_f32();

    (earth.body, residual)
  };

  // Dispatch ResetSimulation command
  let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
  ctx
    .threads
    .logic_thread
    .tx()
    .try_send(LogicCommand::ResetSimulation {
      scene_id,
      done_flag: done.clone(),
    })
    .unwrap();

  // Wait for logic thread completion
  while !done.load(std::sync::atomic::Ordering::Acquire) {
    std::thread::yield_now();
  }

  // Verify
  {
    let scenes = ctx.scenes.read();
    let scene_arc = scenes.get_scene(scene_id).unwrap();
    let scene_guard = scene_arc.read();

    let current_pos = scene_guard
      .scene
      .with_component(earth_body, |t: &crate::scene::TransformComponent| {
        t.position
      })
      .unwrap();

    use aethervk_oshal_rlib::math::vector::Vector3;
    assert!((current_pos.x() - expected_residual.x()).abs() < 1e-4);
    assert!((current_pos.y() - expected_residual.y()).abs() < 1e-4);
    assert!((current_pos.z() - expected_residual.z()).abs() < 1e-4);
  }
}

#[test]
fn test_update_trajectory_command_exists() {
  let _cmd = LogicCommand::UpdateTrajectoryForSpk {
    task_id: 1,
    scene_id: 1,
    entity_id: 1,
    spk_id: 399,
    start_epoch_tai_sec: 0.0,
    end_epoch_tai_sec: 100.0,
    sample_step_days: 1.0,
  };
  // If it compiles, the variant exists and fields are correct.
  assert!(true);
}

#[test]
fn test_two_phase_commit_comet() {
  let mut ctx = SimulationContext::startup(None).expect("Failed to create SimulationContext");

  let (tx, rx) = mpsc::channel();
  *MOCK_SENDER.lock() = Some(tx);
  crate::simulation_api::set_external_state_simulation_callback(Some(mock_external_state_cb));

  fn fetch_spk() -> Option<std::path::PathBuf> {
    use std::io::Write;
    // Query Horizons API for 67P (spk id 1000012)
    let url = "https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND=%2790000703%3B%27&MAKE_EPHEM=%27YES%27&EPHEM_TYPE=%27SPK%27&OBJ_DATA=%27NO%27&START_TIME=%272025-10-01%27&STOP_TIME=%272025-11-02%27";
    let resp = match reqwest::blocking::get(url) {
      Ok(r) => r,
      Err(e) => {
        println!("Skipping test: network unreachable ({e})");
        return None;
      }
    };
    let text = resp.text().unwrap_or_default();
    if text.is_empty() {
      return None;
    }

    let mut base64_clean = String::new();
    let mut marker_seen = false;
    for line in text.lines() {
      if !marker_seen {
        let trimmed = line.trim_start();
        if trimmed.starts_with("REFGL1NQ") {
          marker_seen = true;
          base64_clean.push_str(trimmed.trim_end());
        }
        continue;
      }
      if line.trim().is_empty() {
        break;
      }
      base64_clean.push_str(line.trim());
    }

    use base64::{Engine as _, engine::general_purpose};
    let decoded = general_purpose::STANDARD.decode(&base64_clean).unwrap_or_else(|_| {
      general_purpose::STANDARD_NO_PAD
        .decode(&base64_clean)
        .expect("Failed to decode base64")
    });

    let path = std::env::temp_dir().join("1000012.bsp");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&decoded).unwrap();
    Some(path)
  }

  // Manually load the almanac into logic_state so valid dates pass
  let spk_path = fetch_spk();
  {
    let mut logic_state = ctx.logic_state.write();
    logic_state
      .almanac_data
      .load_almanac("../../assets/planets/pck00011.pca")
      .expect("Failed to load PCK");
    logic_state
      .almanac_data
      .load_almanac("../../assets/planets/gm_de431.pca")
      .expect("Failed to load GM");
    logic_state
      .almanac_data
      .load_almanac("../../assets/planets/de442.bsp")
      .expect("Failed to load DE442");
    logic_state
      .almanac_data
      .load_almanac(spk_path.as_ref().unwrap().to_str().unwrap())
      .expect("Failed to load fetched SPK");
  }

  // Prepare valid scene and dates (within 2025-10-01 and 2025-11-02)
  // 2025-10-15 TDB in seconds since J2000
  let start = Epoch::from_gregorian_utc(2025, 10, 15, 0, 0, 0, 0);
  let end = start + Duration::from_days(10.0);
  let scene_ret = ctx
    .create_empty_scene2(false, start, end)
    .expect("Failed to create empty scene");
  let scene_id = scene_ret.scene_id;

  // --- 1. Test for failure: Out of bounds epoch ---
  let bad_start = Epoch::from_tdb_seconds(31557600000.0); // year 3000
  let bad_end = bad_start + Duration::from_days(10.0);

  ctx
    .threads
    .logic_thread
    .tx()
    .try_send(LogicCommand::TryInitComet {
      scene_id,
      spk_id: 1000012, // 67P
      proposed_start: bad_start,
      proposed_end: bad_end,
      keplerian_elements: KeplerianElements {
        eccentricity: 0.6402,
        perihelion_distance_au: 1.2432,
        inclination_deg: 3.871,
        longitude_of_ascending_node_deg: 36.33,
        argument_of_perihelion_deg: 22.15,
      },
    })
    .unwrap();

  let result = rx
    .recv_timeout(std::time::Duration::from_secs(5))
    .expect("Timeout waiting for failure callback");
  assert_eq!(
    result.success, 0,
    "Expected TryInitComet to fail for out-of-bounds epoch"
  );

  // Verify ECS rollback / lack of attachment on failure
  {
    let scene_ctx = ctx.scenes.read().get(&scene_id).cloned().unwrap();
    let scene_guard = scene_ctx.read();
    let comet = scene_guard.comet.unwrap();
    let has_planet: bool = scene_guard.scene.has_component::<AlmanacPlanet>(comet.body).into();
    let has_traj: bool = scene_guard.scene.has_component::<TrajectoryComponent>(comet.orbit).into();
    assert!(
      !has_planet,
      "AlmanacPlanet should not be attached on failure"
    );
    assert!(
      !has_traj,
      "TrajectoryComponent should not be attached on failure"
    );
  }

  // --- 2. Test for success: Valid epoch ---
  ctx
    .threads
    .logic_thread
    .tx()
    .try_send(LogicCommand::TryInitComet {
      scene_id,
      spk_id: 1000012, // 67P
      proposed_start: start,
      proposed_end: end,
      keplerian_elements: KeplerianElements {
        eccentricity: 0.6402,
        perihelion_distance_au: 1.2432,
        inclination_deg: 3.871,
        longitude_of_ascending_node_deg: 36.33,
        argument_of_perihelion_deg: 22.15,
      },
    })
    .unwrap();

  let result = rx
    .recv_timeout(std::time::Duration::from_secs(15))
    .expect("Timeout waiting for success callback");
  assert_eq!(
    result.success, 1,
    "Expected TryInitComet to succeed for valid epoch"
  );

  // Verify attachment on success
  {
    let scene_ctx = ctx.scenes.read().get(&scene_id).cloned().unwrap();
    let scene_guard = scene_ctx.read();
    let comet = scene_guard.comet.unwrap();
    let has_planet: bool = scene_guard.scene.has_component::<AlmanacPlanet>(comet.body).into();
    let has_traj: bool = scene_guard.scene.has_component::<TrajectoryComponent>(comet.orbit).into();
    assert!(has_planet, "AlmanacPlanet should be attached on success");
    assert!(
      has_traj,
      "TrajectoryComponent should be attached on success"
    );
  }

  ctx.threads.logic_thread.tx().try_send(LogicCommand::Shutdown).unwrap();
}
#[test]
fn test_simulation_start_pause_sync() {
  use aethervk_oshal_rlib::os::time::v2::SimSpeed;

  let mut ctx = SimulationContext::startup(None).expect("Failed to create SimulationContext");

  let start = hifitime::Epoch::from_gregorian_utc(2025, 10, 15, 0, 0, 0, 0);
  let end = start + hifitime::Duration::from_days(10.0);
  let scene_ret = ctx.create_empty_scene2(false, start, end).expect("Failed to create scene");
  let scene_id = scene_ret.scene_id;

  // 1. Initial state should be paused/not running
  let is_running = {
    let scene = ctx.get_scene(scene_id).unwrap();
    scene.read().simulation_running.load(core::sync::atomic::Ordering::Acquire)
  };
  assert!(!is_running, "Simulation should be stopped initially");

  // 2. Start simulation
  let start_ok = ctx.start_simulation(scene_id, SimSpeed::Realtime);
  assert!(start_ok, "start_simulation should succeed");

  let is_running = {
    let scene = ctx.get_scene(scene_id).unwrap();
    scene.read().simulation_running.load(core::sync::atomic::Ordering::Acquire)
  };
  assert!(is_running, "Simulation should be running after start");

  // Mock an active physics task to ensure start fails if we try to restart while running/syncing
  {
    let scene = ctx.get_scene(scene_id).unwrap();
    scene
      .read()
      .active_physics_task
      .store(true, core::sync::atomic::Ordering::Release);
  }

  // 3. Attempting to start again while physics is active should fail
  let start_fail = ctx.start_simulation(scene_id, SimSpeed::Realtime);
  assert!(
    !start_fail,
    "start_simulation should fail if active_physics_task is true"
  );

  // Clean up mock state so pause can resolve (otherwise it spins forever)
  {
    let scene = ctx.get_scene(scene_id).unwrap();
    scene
      .read()
      .active_physics_task
      .store(false, core::sync::atomic::Ordering::Release);
  }

  // 4. Pause simulation
  let pause_ok = ctx.pause_simulation_sync(scene_id);
  assert!(pause_ok, "pause_simulation_sync should succeed");

  let is_running = {
    let scene = ctx.get_scene(scene_id).unwrap();
    scene.read().simulation_running.load(core::sync::atomic::Ordering::Acquire)
  };
  assert!(!is_running, "Simulation should be stopped after pause");

  ctx.threads.logic_thread.tx().try_send(LogicCommand::Shutdown).unwrap();
}

#[test]
fn test_cleanup_and_remove_particle_system() {
  use alloc::sync::Arc;
  use core::sync::atomic::{AtomicBool, Ordering};

  let ctx = SimulationContext::startup(None).expect("Failed to create context");
  let start = hifitime::Epoch::from_gregorian_utc(2025, 10, 15, 0, 0, 0, 0);
  let end = start + hifitime::Duration::from_days(10.0);
  let scene_ret = ctx.create_empty_scene2(false, start, end).unwrap();
  let scene_id = scene_ret.scene_id;

  aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(100));

  let entity_id = scene_ret.comet_body;

  let done_flag = Arc::new(AtomicBool::new(false));

  let send_res = ctx.threads.logic_thread.tx().try_send(LogicCommand::CleanupParticleSystem {
    scene_id,
    entity_id,
    done_flag: done_flag.clone(),
  });
  assert!(
    send_res.is_ok(),
    "Failed to send CleanupParticleSystem command"
  );

  let mut spins = 0;
  while !done_flag.load(Ordering::Acquire) {
    if spins > 500 {
      panic!("Timeout waiting for CleanupParticleSystem command");
    }
    aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(10));
    spins += 1;
  }

  done_flag.store(false, Ordering::Release);

  let send_res = ctx.threads.logic_thread.tx().try_send(LogicCommand::RemoveParticleSystem {
    scene_id,
    entity_id,
    done_flag: done_flag.clone(),
  });
  assert!(
    send_res.is_ok(),
    "Failed to send RemoveParticleSystem command"
  );

  spins = 0;
  while !done_flag.load(Ordering::Acquire) {
    if spins > 500 {
      panic!("Timeout waiting for RemoveParticleSystem command");
    }
    aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(10));
    spins += 1;
  }

  ctx.threads.logic_thread.tx().try_send(LogicCommand::Shutdown).unwrap();
}

#[test]
fn test_pause_simulation_timeout_safety() {
  let mut ctx = SimulationContext::startup(None).expect("Failed to create SimulationContext");
  let start = hifitime::Epoch::from_gregorian_utc(2025, 10, 15, 0, 0, 0, 0);
  let end = start + hifitime::Duration::from_days(10.0);
  let scene_ret = ctx.create_empty_scene2(false, start, end).expect("Failed to create scene");
  let scene_id = scene_ret.scene_id;

  ctx.threads.logic_thread.tx().try_send(LogicCommand::Shutdown).unwrap();
  // Wait for the logic thread to shut down so it doesn't hit a panic when we mock active_physics_task
  std::thread::sleep(std::time::Duration::from_millis(50));

  {
    let scene = ctx.get_scene(scene_id).unwrap();
    let mut scene_write = scene.write();
    scene_write.pending_cross_sync = true;
    scene_write
      .active_physics_task
      .store(true, core::sync::atomic::Ordering::Relaxed);
  }

  let start_time = std::time::Instant::now();
  let pause_ok = ctx.pause_simulation_sync(scene_id);
  let elapsed = start_time.elapsed();

  assert!(
    !pause_ok,
    "pause_simulation_sync should timeout and return false"
  );
  assert!(
    elapsed.as_millis() >= 500,
    "Should have waited for at least 500ms before timeout"
  );

  // Clean up
  {
    let scene = ctx.get_scene(scene_id).unwrap();
    let mut scene_write = scene.write();
    scene_write.pending_cross_sync = false;
    scene_write
      .active_physics_task
      .store(false, core::sync::atomic::Ordering::Relaxed);
  }
}
