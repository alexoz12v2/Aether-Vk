use super::*;
use crate::{
  scene::{AlmanacPlanet, trajectory::TrajectoryComponent},
  simulation_api::{
    SimulationContext,
    external_state::CCometInitialized,
    set_external_state_simulation_callback,
    structs::{KeplerianElements, LogicCommand},
  },
};
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
      succeeded: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
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

  // Manually load the almanac into logic_state so valid dates pass
  let spk_path = fetch_67p_spk("1000012_two_phase.bsp");
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
        time_of_perihelion_jd_tdb: f64::NAN,
      },
      reference_mode: Default::default(),
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
        time_of_perihelion_jd_tdb: f64::NAN,
      },
      reference_mode: Default::default(),
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
    scene_write
      .active_physics_task
      .store(false, core::sync::atomic::Ordering::Relaxed);
  }
}

// ------------------------------------------------------------------------------------------------
// Axes / units / frame shift (no GPU): ANISE SUN_ECLIPJ2000 km -> micro frame AU + body km residual
// ------------------------------------------------------------------------------------------------

fn load_planet_almanac() -> crate::simulation::almanac::AlmanacPackedData {
  let mut data = crate::simulation::almanac::AlmanacPackedData::default();
  for f in [
    "../../assets/planets/pck00011.pca",
    "../../assets/planets/gm_de431.pca",
    "../../assets/planets/de442.bsp",
    "../../assets/earth_latest_high_prec.bpc",
  ] {
    data.load_almanac(f).unwrap_or_else(|e| panic!("load {f}: {e:?}"));
  }
  data
}

/// Scene axes are SUN_ECLIPJ2000 with no swizzle: +z ecliptic north, +x vernal equinox, +y = z × x.
/// At the March equinox the Sun is seen from Earth towards +x (Earth at -x), at the June solstice
/// towards +y (Earth at -y), and the orbit is counter-clockwise seen from +z.
#[test]
fn test_earth_ephemeris_axes_are_ecliptic_j2000() {
  use aethervk_oshal_rlib::math::vector::Vector3;
  let almanac = load_planet_almanac();
  let earth = AlmanacPlanet { naif_id: 399 };

  let march = Epoch::from_gregorian_utc(2025, 3, 20, 9, 1, 0, 0);
  let june = Epoch::from_gregorian_utc(2025, 6, 21, 2, 42, 0, 0);
  let (p_mar, v_mar, _) = earth.step_with_velocity(march, &almanac, None).unwrap();
  let (p_jun, _, _) = earth.step_with_velocity(june, &almanac, None).unwrap();

  // km, not AU nor metres
  let r = p_mar.length();
  assert!((1.45e8..1.53e8).contains(&r), "earth distance {r} km");

  assert!(
    p_mar.x() / r < -0.999,
    "march equinox: earth should be on -x, got {p_mar:?}"
  );
  assert!(
    p_jun.y() / p_jun.length() < -0.999,
    "june solstice: earth should be on -y, got {p_jun:?}"
  );

  // ecliptic plane: z ~ 0 over a whole year
  for d in 0..73 {
    let e = march + Duration::from_days(5.0 * d as f64);
    let (p, _, _) = earth.step_with_velocity(e, &almanac, None).unwrap();
    assert!(
      (p.z() / p.length()).abs() < 1e-4,
      "earth out of ecliptic at day {}: {p:?}",
      5 * d
    );
  }

  // prograde (counter-clockwise from +z)
  assert!(p_mar.cross(v_mar).z() > 0.0);
}

/// After a frame shift, a body that comes back within the threshold of the *old* frame origin must
/// still be expressed relative to the *current* frame origin.
#[test]
fn test_frame_shift_keeps_body_on_spk_position() {
  use crate::simulation_api::{
    reposition::{AU_TO_KM, compute_macro_and_residual},
    structs::SceneEntityId,
  };
  use aethervk_oshal_rlib::math::vector::Vector3;

  let almanac = load_planet_almanac();
  let earth = AlmanacPlanet { naif_id: 399 };
  let scene_id = 7;
  let frame_id = EntityId::from_ffi(1);
  let body_id = EntityId::from_ffi(2);
  let t0 = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);

  let (p0, _, _) = earth.step_with_velocity(t0, &almanac, None).unwrap();
  let (frame_au, residual_km) = compute_macro_and_residual(p0);
  let mut frame_t = TransformComponent::default();
  frame_t.position = frame_au;
  let mut body_t = TransformComponent::default();
  body_t.position = residual_km;

  let cache = dashmap::DashMap::new();
  cache.insert(
    SceneEntityId::new(scene_id, body_id),
    CartesianState::new_comet(body_t, earth, None, frame_id, frame_t),
  );
  cache.insert(
    SceneEntityId::new(scene_id, frame_id),
    CartesianState::new_frame(frame_id, frame_t),
  );
  // an entry of another scene must be left untouched
  let foreign = SceneEntityId::new(scene_id + 1, body_id);
  cache.insert(
    foreign,
    CartesianState::new_comet(body_t, earth, None, frame_id, frame_t),
  );

  let world_km = |cache: &dashmap::DashMap<SceneEntityId, CartesianState>| {
    let frame = cache.get(&SceneEntityId::new(scene_id, frame_id)).unwrap();
    let body = cache.get(&SceneEntityId::new(scene_id, body_id)).unwrap();
    frame.parent_frame_transform.position.to_f64() * AU_TO_KM
      + body.comet_state.as_ref().unwrap().transform.position.to_f64()
  };

  // +8 days: ~0.14 AU away, shifts. +1 day: back within 0.1 AU of the original origin.
  for days in [8.0, 1.0, 20.0, 0.0] {
    let t = t0 + Duration::from_days(days);
    utils::step_cartesian_cache(&cache, scene_id, t, &almanac);
    let (expected, _, _) = earth.step_with_velocity(t, &almanac, None).unwrap();
    let err = (world_km(&cache) - expected).length();
    assert!(
      err < 2.0,
      "day {days}: body off its SPK position by {err} km"
    );
    let body = cache.get(&SceneEntityId::new(scene_id, body_id)).unwrap();
    let frame = cache.get(&SceneEntityId::new(scene_id, frame_id)).unwrap();
    assert_eq!(
      body.parent_frame_transform.position, frame.parent_frame_transform.position,
      "body copy of the frame transform went stale"
    );
  }

  let f = cache.get(&foreign).unwrap();
  assert_eq!(
    f.comet_state.as_ref().unwrap().transform.position,
    residual_km
  );
  assert!(f.comet_state.as_ref().unwrap().helio_state_km.is_none());
}

fn startup_with_planets() -> Box<SimulationContext> {
  unsafe { std::env::set_var("ASSET_DIR", "../../assets") };
  let ctx = SimulationContext::startup(None).expect("Failed to create SimulationContext");
  {
    let mut logic_state = ctx.logic_state.write();
    for f in [
      "../../assets/planets/pck00011.pca",
      "../../assets/planets/gm_de431.pca",
      "../../assets/planets/de442.bsp",
      "../../assets/earth_latest_high_prec.bpc",
    ] {
      logic_state
        .almanac_data
        .load_almanac(f)
        .unwrap_or_else(|e| panic!("load {f}: {e:?}"));
    }
  }
  ctx
}

/// The SPICE step and the commit of the cartesian cache must not depend on the GPU: with no
/// device (stalled compute queue, `physics_done == false`) the planet still follows its SPK.
#[test]
fn test_spk_commit_independent_of_gpu() {
  use aethervk_oshal_rlib::math::vector::Vector3;
  let mut ctx = startup_with_planets();
  let start = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);
  let scene_id = ctx
    .create_empty_scene2(true, start, start + Duration::from_days(60.0))
    .unwrap()
    .scene_id;

  let scenes = ctx.scenes.read();
  let scene_arc = scenes.get_scene(scene_id).unwrap();
  let earth = scene_arc.read().earth.unwrap();
  // holding the shard keeps the logic thread from ticking this clock while we check it
  let mut time_mgr = scenes.time_managers.get_mut(&scene_id).unwrap();
  time_mgr.set_speed(oshal::os::time::v2::SimSpeed::OneDayPerSec);
  {
    let mut st = time_mgr.state.write();
    st.scaled_time = 20 * 86_400 * 1_000_000; // 20 days in
    st.scaled_accumulator = 4 * crate::simulation_api::structs::UNSCALED_FIXED_DELTA_US * 86_400;
  }

  let out = execute_simulation_tick_fixed_update_phase(
    None,
    scene_id,
    scene_arc.upgradable_read(),
    &mut time_mgr,
    crate::simulation_api::structs::UNSCALED_FIXED_DELTA_US,
    &scenes.cartesian_state_cache,
    &ctx.logic_state.read().almanac_data,
  )
  .expect("fixed update without GPU must succeed");
  assert!(
    out.latest_physics_sync.is_none(),
    "no GPU work must have been submitted"
  );

  let epoch = time_mgr.current_epoch();
  let (expected_km, _, _) = AlmanacPlanet { naif_id: 399 }
    .step_with_velocity(epoch, &ctx.logic_state.read().almanac_data, None)
    .unwrap();
  let got = scene_arc.read().scene.global_transform_f64(earth.body).unwrap().position;
  let got_km = got * crate::simulation_api::reposition::AU_TO_KM;
  let err = (got_km - expected_km).length();
  assert!(
    err < 5.0,
    "earth not committed to its SPK position without GPU: off by {err} km"
  );
}

/// `CleanupComet` removes the jets under the comet body but must keep a camera parented there
/// (CometOrbiting mode), moved to root with its world transform, and evict the comet from the
/// cartesian cache.
#[test]
fn test_cleanup_comet_keeps_parented_camera() {
  use aethervk_oshal_rlib::math::vector::{Vector3, vec3f64::DVec3};
  let mut ctx = startup_with_planets();
  let start = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);
  let scene_id = ctx
    .create_empty_scene2(true, start, start + Duration::from_days(10.0))
    .unwrap()
    .scene_id;

  let (cam, comet, world_before) = {
    let scenes = ctx.scenes.read();
    let scene_arc = scenes.get_scene(scene_id).unwrap();
    let g = scene_arc.read();
    let comet = g.comet.unwrap();
    let cam = g.scene.get_entity_by_name("camera").expect("scene camera");
    g.scene.set_parent(cam, Some(comet.body));
    let _ = g
      .scene
      .with_component_mut(cam, |t: &mut crate::scene::HighResTransformComponent| {
        t.position = DVec3::from_components(0.0, 0.0, 1000.0);
      });
    let world = g.scene.global_transform_f64(cam).unwrap().position;
    // as if the comet had been committed and stepped once
    scenes.cartesian_state_cache.insert(
      crate::simulation_api::structs::SceneEntityId::new(scene_id, comet.body),
      CartesianState::new_frame(comet.subtree, TransformComponent::default()),
    );
    (cam, comet, world)
  };

  ctx
    .threads
    .logic_thread
    .tx()
    .try_send(LogicCommand::CleanupComet { scene_id })
    .unwrap();

  let t0 = std::time::Instant::now();
  loop {
    let scenes = ctx.scenes.read();
    let scene_arc = scenes.get_scene(scene_id).unwrap();
    let g = scene_arc.read();
    if g.scene.get_parent(cam) == Some(g.root_entity) {
      let world_after = g.scene.global_transform_f64(cam).expect("camera must survive").position;
      assert!(
        (world_after - world_before).length() < 1e-9,
        "{world_before:?} -> {world_after:?}"
      );
      assert!(!scenes.cartesian_state_cache.contains_key(
        &crate::simulation_api::structs::SceneEntityId::new(scene_id, comet.body)
      ));
      break;
    }
    assert!(
      t0.elapsed().as_secs() < 5,
      "CleanupComet did not reparent the camera"
    );
    drop(g);
    drop(scenes);
    std::thread::sleep(std::time::Duration::from_millis(10));
  }
}

/// Min distance (AU) from `p` to a cubic Bezier track (4 control points per segment).
fn distance_to_bezier_track_au(track: &[[f32; 4]], p: [f64; 3]) -> f64 {
  let mut best = f64::MAX;
  for seg in track.chunks_exact(4) {
    for k in 0..=64 {
      let t = k as f64 / 64.0;
      let u = 1.0 - t;
      let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
      let mut d2 = 0.0;
      for c in 0..3 {
        let x: f64 = (0..4).map(|j| w[j] * seg[j][c] as f64).sum();
        d2 += (x - p[c]).powi(2);
      }
      best = best.min(d2.sqrt());
    }
  }
  best
}

/// The SBDB-style analytical track is in the scene frame (ecliptic J2000). Checked offline with
/// Earth's J2000 mean elements against de442: the old ecliptic->equatorial rotation tilted the
/// track by 23.44 deg (~0.4 AU off), and straight chords sagged by up to ~1e-3 AU.
#[test]
fn test_keplerian_track_matches_spk_in_ecliptic_frame() {
  use crate::simulation_api::reposition::KM_TO_AU;
  use aethervk_oshal_rlib::math::vector::Vector3;
  let almanac = load_planet_almanac();
  let earth_elements = KeplerianElements {
    eccentricity: 0.01671123,
    perihelion_distance_au: 1.00000261 * (1.0 - 0.01671123),
    inclination_deg: 0.0,
    longitude_of_ascending_node_deg: 0.0,
    argument_of_perihelion_deg: 102.93768193,
    time_of_perihelion_jd_tdb: f64::NAN,
  };
  let track = utils::keplerian_track_bezier_au(&earth_elements);
  assert_eq!(track.len(), 4 * utils::KEPLER_TRACK_SEGMENTS);

  let start = Epoch::from_gregorian_utc(2025, 1, 1, 0, 0, 0, 0);
  let mut worst = 0.0f64;
  for d in (0..365).step_by(3) {
    let (p_km, _, _) = AlmanacPlanet { naif_id: 399 }
      .step_with_velocity(start + Duration::from_days(d as f64), &almanac, None)
      .unwrap();
    let p = p_km * KM_TO_AU;
    worst = worst.max(distance_to_bezier_track_au(&track, [p.x(), p.y(), p.z()]));
  }
  // mean elements vs real Earth (Moon wobble, 25 years of perihelion drift): well below 1e-3 AU
  assert!(worst < 1e-3, "track off the SPK path by {worst} AU");
}

/// Hermite handles keep a hyperbolic track on the conic between samples too.
#[test]
fn test_keplerian_track_hyperbola_stays_on_conic() {
  let el = KeplerianElements {
    eccentricity: 1.2,
    perihelion_distance_au: 1.5,
    inclination_deg: 40.0,
    longitude_of_ascending_node_deg: 70.0,
    argument_of_perihelion_deg: 10.0,
    time_of_perihelion_jd_tdb: f64::NAN,
  };
  let track = utils::keplerian_track_bezier_au(&el);
  // with i = 40 deg the orbit normal is tilted; every Bezier midpoint must still satisfy the conic
  // equation r = p / (1 + e cos nu) in its own orbital plane, i.e. |r_mid - r_conic| small.
  let (i, om, w) = (40f64.to_radians(), 70f64.to_radians(), 10f64.to_radians());
  let p_hat = [
    om.cos() * w.cos() - om.sin() * i.cos() * w.sin(),
    om.sin() * w.cos() + om.cos() * i.cos() * w.sin(),
    i.sin() * w.sin(),
  ];
  let q_hat = [
    -om.cos() * w.sin() - om.sin() * i.cos() * w.cos(),
    -om.sin() * w.sin() + om.cos() * i.cos() * w.cos(),
    i.sin() * w.cos(),
  ];
  let p = 1.5 * 2.2;
  for seg in track.chunks_exact(4) {
    let mid: [f64; 3] = core::array::from_fn(|c| {
      0.125 * seg[0][c] as f64
        + 0.375 * seg[1][c] as f64
        + 0.375 * seg[2][c] as f64
        + 0.125 * seg[3][c] as f64
    });
    let x: f64 = (0..3).map(|c| mid[c] * p_hat[c]).sum();
    let y: f64 = (0..3).map(|c| mid[c] * q_hat[c]).sum();
    let r = (x * x + y * y).sqrt();
    let nu = y.atan2(x);
    let r_conic = p / (1.0 + 1.2 * nu.cos());
    assert!(
      (r - r_conic).abs() / r_conic < 1e-4,
      "midpoint off conic: r={r} conic={r_conic}"
    );
  }
}

/// Open orbits span (almost) the whole branch: up to 5° before the asymptote acos(-1/e), capped
/// at `KEPLER_TRACK_MAX_R_AU`. The old acos(1/e) bound drew e = 1.2 over ±33.6° only and a
/// reversed sliver for a parabola.
#[test]
fn test_keplerian_track_open_orbit_extent() {
  for (e, q) in [(1.2, 1.5), (1.0, 1.0), (3.0, 0.5)] {
    let el = KeplerianElements {
      eccentricity: e,
      perihelion_distance_au: q,
      inclination_deg: 0.0,
      longitude_of_ascending_node_deg: 0.0,
      argument_of_perihelion_deg: 0.0,
      time_of_perihelion_jd_tdb: f64::NAN,
    };
    // i = Ω = ω = 0: perifocal == ecliptic, perihelion on +X
    let track = utils::keplerian_track_bezier_au_f64(&el);
    let (first, last) = (track[0], track[track.len() - 1]);
    let nu = |p: [f64; 3]| p[1].atan2(p[0]);
    let r = |p: [f64; 3]| (p[0] * p[0] + p[1] * p[1]).sqrt();
    assert!(
      nu(first) < 0.0 && nu(last) > 0.0,
      "e={e}: track runs from -nu to +nu"
    );
    assert!(
      (nu(first) + nu(last)).abs() < 1e-9,
      "e={e}: track symmetric about perihelion"
    );
    assert!(
      nu(last) > 90f64.to_radians(),
      "e={e}: nu_max {} deg",
      nu(last).to_degrees()
    );
    let asymptote = (-1.0 / e).acos();
    assert!(
      nu(last) <= asymptote - 5f64.to_radians() + 1e-9,
      "e={e}: past asymptote - 5 deg"
    );
    let r_max = track.iter().map(|p| r(*p)).fold(0.0, f64::max);
    assert!(
      r_max <= utils::KEPLER_TRACK_MAX_R_AU * (1.0 + 1e-9),
      "e={e}: r_max {r_max} AU"
    );
  }
}

/// The logic thread records the comet SPK path on `effective_comet_trajectory`; a reset removes
/// the components but keeps the (empty container) entity.
#[test]
fn test_effective_trajectory_recorded_and_cleared_on_reset() {
  use crate::scene::trajectory::{EffectiveTrajectoryComponent, TrajectoryComponent};
  use aethervk_oshal_rlib::math::vector::{Vector3, vec3f64::DVec3};
  let mut ctx = startup_with_planets();
  let start = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);
  let scene_id = ctx
    .create_empty_scene2(true, start, start + Duration::from_days(30.0))
    .unwrap()
    .scene_id;

  let entity = {
    let scenes = ctx.scenes.read();
    let scene_arc = scenes.get_scene(scene_id).unwrap();
    let g = scene_arc.read();
    let entity = g.effective_comet_trajectory.expect("container entity at scene creation");
    assert_eq!(
      g.scene.get_entity_by_name("effective_comet_trajectory"),
      Some(entity)
    );
    assert_eq!(g.scene.get_parent(entity), Some(g.root_entity));

    // a committed comet that the SPICE step moved along +y at 30 km/s
    let comet = g.comet.unwrap();
    let key = crate::simulation_api::structs::SceneEntityId::new(scene_id, comet.body);
    scenes.cartesian_state_cache.insert(
      key,
      CartesianState::new_comet(
        TransformComponent::default(),
        AlmanacPlanet { naif_id: 1000012 },
        None,
        comet.subtree,
        TransformComponent::default(),
      ),
    );
    for k in 0..4 {
      let epoch = start + Duration::from_hours(6.0 * k as f64);
      let t = (epoch - start).to_seconds();
      scenes
        .cartesian_state_cache
        .get_mut(&key)
        .unwrap()
        .comet_state
        .as_mut()
        .unwrap()
        .helio_state_km = Some((
        DVec3::from_components(1.5e8, 30.0 * t, 0.0),
        DVec3::from_components(0.0, 30.0, 0.0),
      ));
      utils::record_effective_trajectory(&g, scene_id, &scenes.cartesian_state_cache, epoch);
    }
    let n = g
      .scene
      .with_component(entity, |c: &EffectiveTrajectoryComponent| c.samples.len());
    assert_eq!(n, Some(4));
    let traj = g
      .scene
      .with_component(entity, |c: &TrajectoryComponent| c.clone())
      .expect("grey track");
    assert_eq!(traj.control_points.len(), 4 * 3);
    assert_eq!(traj.color, EffectiveTrajectoryComponent::COLOR);
    scenes.cartesian_state_cache.remove(&key);
    entity
  };

  let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
  let ok = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
  ctx
    .threads
    .logic_thread
    .tx()
    .try_send(LogicCommand::ResetSimulation {
      scene_id,
      done_flag: done.clone(),
      succeeded: ok.clone(),
    })
    .unwrap();
  let t0 = std::time::Instant::now();
  while !done.load(std::sync::atomic::Ordering::Acquire) {
    assert!(t0.elapsed().as_secs() < 5, "reset did not complete");
    std::thread::yield_now();
  }
  assert!(
    ok.load(std::sync::atomic::Ordering::Relaxed),
    "reset reported failure"
  );

  let scenes = ctx.scenes.read();
  let scene_arc = scenes.get_scene(scene_id).unwrap();
  let g = scene_arc.read();
  assert!(g.scene.with_component(entity, |_: &EffectiveTrajectoryComponent| ()).is_none());
  assert!(g.scene.with_component(entity, |_: &TrajectoryComponent| ()).is_none());
  assert_eq!(
    g.scene.get_entity_by_name("effective_comet_trajectory"),
    Some(entity)
  );
}

/// Downloads the 67P SPK (2025-10-01 .. 2025-11-02) from Horizons; `None` when offline.
fn fetch_67p_spk(file_name: &str) -> Option<std::path::PathBuf> {
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

  let path = std::env::temp_dir().join(file_name);
  let mut file = std::fs::File::create(&path).unwrap();
  file.write_all(&decoded).unwrap();
  Some(path)
}

/// Network-gated: 67P elements re-osculated from its SPK state at the start epoch reproduce the SPK
/// over the 2025-10-01 .. 11-01 window within ~10³ km (planetary perturbations + non-gravitational
/// forces), whereas the SBDB solution (osculating at 2015-10-10) is ~0.12 AU off.
#[test]
fn test_reosculated_67p_reference_tracks_spk() {
  use crate::simulation::orbit_elements::{
    MU_SUN_KM3_S2, elements_from_state, state_from_elements,
  };
  use aethervk_oshal_rlib::math::vector::Vector3;
  let Some(spk) = fetch_67p_spk("1000012_reosculate.bsp") else {
    println!("Skipping: Horizons unreachable");
    return;
  };
  let mut almanac = load_planet_almanac();
  almanac.load_almanac(spk.to_str().unwrap()).expect("load 67P SPK");
  // the small-body id is whatever the downloaded SPK carries (Horizons record 90000703)
  let naif_id = almanac
    .almanac
    .spk_domains()
    .ok()
    .and_then(|d| d.into_iter().map(|(id, _)| id).find(|&id| id != 0 && !(1..=999).contains(&id)))
    .expect("small-body segment in the SPK");
  let comet = AlmanacPlanet { naif_id };
  // the downloaded segment's interpolation data starts a few minutes after 2025-10-01T00:00: a
  // midnight start (what the timeline proposes) must be clamped into the coverage, as the app does
  let midnight = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);
  let start = utils::osculation_epoch(midnight, midnight + Duration::from_days(31.0), |t| {
    comet.step_with_velocity(t, &almanac, None).is_ok()
  })
  .expect("covered epoch");
  println!("re-osculation epoch: {start} (requested {midnight})");
  let rm = crate::scene::BodyRotationalModel::default();
  let (r0, v0, _) = comet.step_with_velocity(start, &almanac, Some(&rm)).unwrap();
  let el = elements_from_state(
    [r0.x(), r0.y(), r0.z()],
    [v0.x(), v0.y(), v0.z()],
    MU_SUN_KM3_S2,
    start.to_jde_tdb_days(),
  );
  let mut worst = 0.0f64;
  for d in 0..=30 {
    let t = start + Duration::from_days(d as f64);
    let (r, _, _) = comet.step_with_velocity(t, &almanac, Some(&rm)).unwrap();
    let (rr, _) = state_from_elements(&el, MU_SUN_KM3_S2, t.to_jde_tdb_days()).unwrap();
    let err = ((rr[0] - r.x()).powi(2) + (rr[1] - r.y()).powi(2) + (rr[2] - r.z()).powi(2)).sqrt();
    worst = worst.max(err);
  }
  println!("re-osculated 67P reference: worst |two-body - SPK| over 31 d = {worst:.0} km");

  // the *drawn* track (Bezier) must pass through the comet at the osculation epoch
  let track = utils::keplerian_track_bezier_au_f64(&el);
  let au = crate::simulation_api::reposition::AU_TO_KM;
  let (_, d_au) = crate::scene::trajectory::closest_point_on_bezier_track(
    &track,
    [r0.x() / au, r0.y() / au, r0.z() / au],
  )
  .unwrap();
  println!(
    "drawn re-osculated track vs comet at start: {:.3} km",
    d_au * au
  );
  assert!(
    d_au * au < 0.1,
    "drawn track misses the comet by {} km",
    d_au * au
  );
  assert!(worst < 1_000.0, "re-osculated reference off by {worst} km");
}

/// The comet label exists from scene creation, follows the comet body, starts hidden and is
/// toggled by `set_comet_indicator_visible` (Earth observer mode).
#[test]
fn test_comet_indicator_spawned_hidden_and_toggles() {
  use crate::scene::{HiddenComponent, ReferentialIndicatorComponent};
  let mut ctx = startup_with_planets();
  let start = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);
  let scene_id = ctx
    .create_empty_scene2(true, start, start + Duration::from_days(10.0))
    .unwrap()
    .scene_id;
  let (e, comet_body) = {
    let scenes = ctx.scenes.read();
    let scene_arc = scenes.get_scene(scene_id).unwrap();
    let g = scene_arc.read();
    (
      g.comet_indicator.expect("comet label spawned"),
      g.comet.unwrap().body,
    )
  };
  let hidden = |ctx: &SimulationContext| {
    let scenes = ctx.scenes.read();
    let scene_arc = scenes.get_scene(scene_id).unwrap();
    let g = scene_arc.read();
    g.scene.with_component(e, |_: &HiddenComponent| ()).is_some()
  };
  {
    let scenes = ctx.scenes.read();
    let scene_arc = scenes.get_scene(scene_id).unwrap();
    let g = scene_arc.read();
    let target = g.scene.with_component(e, |r: &ReferentialIndicatorComponent| r.target_entity);
    assert_eq!(target, Some(comet_body));
  }
  assert!(hidden(&ctx), "hidden outside Earth observer mode");
  assert!(ctx.set_comet_indicator_visible(scene_id, true));
  assert!(!hidden(&ctx));
  assert!(ctx.set_comet_indicator_visible(scene_id, false));
  assert!(hidden(&ctx));
}

/// Reference-position error annotations: shown while enabled and the error exceeds 10 nucleus
/// radii, hidden otherwise; the same-epoch one needs a time of perihelion.
#[test]
fn test_reference_error_annotations_follow_threshold() {
  use crate::{
    scene::{
      HiddenComponent, SphereGizmoComponent,
      trajectory::{ScreenMeasurementComponent, TrajectoryComponent},
    },
    simulation_api::reposition::AU_TO_KM,
  };
  let mut ctx = startup_with_planets();
  let start = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);
  let scene_id = ctx
    .create_empty_scene2(true, start, start + Duration::from_days(10.0))
    .unwrap()
    .scene_id;
  let (comet, [cross, same]) = {
    let scenes = ctx.scenes.read();
    let scene_arc = scenes.get_scene(scene_id).unwrap();
    let g = scene_arc.read();
    (
      g.comet.unwrap(),
      g.reference_error_entities.expect("annotation entities"),
    )
  };
  let place_track = |ctx: &SimulationContext, offset_km: f64| {
    let scenes = ctx.scenes.read();
    let scene_arc = scenes.get_scene(scene_id).unwrap();
    let g = scene_arc.read();
    let p = g.scene.global_transform_f64(comet.body).unwrap().position;
    let x = p.x() + offset_km / AU_TO_KM;
    let cps = alloc::vec![
      [x, p.y() - 0.01, p.z()],
      [x, p.y() - 0.003, p.z()],
      [x, p.y() + 0.003, p.z()],
      [x, p.y() + 0.01, p.z()]
    ];
    let t = TrajectoryComponent::from_f64(cps, [1.0; 4], 2.0, 0, 32);
    if g
      .scene
      .with_component_mut(comet.orbit, |c: &mut TrajectoryComponent| *c = t.clone())
      .is_none()
    {
      let _ = g.scene.add_component(comet.orbit, t);
    }
  };
  let state = |ctx: &SimulationContext, e| {
    let scenes = ctx.scenes.read();
    let scene_arc = scenes.get_scene(scene_id).unwrap();
    let g = scene_arc.read();
    let hidden = g.scene.with_component(e, |_: &HiddenComponent| ()).is_some();
    let label = g
      .scene
      .with_component(e, |m: &ScreenMeasurementComponent| m.label.clone())
      .unwrap();
    (hidden, label)
  };
  {
    let scenes = ctx.scenes.read();
    let scene_arc = scenes.get_scene(scene_id).unwrap();
    let g = scene_arc.read();
    let _ = g.scene.add_component(comet.body, AlmanacPlanet { naif_id: 1000012 });
    let _ = g
      .scene
      .with_component_mut(comet.body, |s: &mut SphereGizmoComponent| s.radius = 4.0); // r = 2 km
  }

  place_track(&ctx, 15_000.0);
  assert!(state(&ctx, cross).0, "hidden while disabled");
  assert!(ctx.set_reference_error_visible(scene_id, true));
  let (hidden, label) = state(&ctx, cross);
  assert!(!hidden, "15 000 km > 10 radii: shown");
  assert!(
    label.starts_with("cross-track 1500") || label.starts_with("cross-track 1499"),
    "{label}"
  );
  assert!(
    state(&ctx, same).0,
    "no time of perihelion: same-epoch hidden"
  );

  place_track(&ctx, 10.0); // 10 km < 20 km threshold
  assert!(ctx.set_reference_error_visible(scene_id, true));
  assert!(state(&ctx, cross).0, "below 10 radii: hidden");

  place_track(&ctx, 15_000.0);
  assert!(ctx.set_reference_error_visible(scene_id, false));
  assert!(state(&ctx, cross).0, "disabled: hidden");
}

/// Seeking while paused moves the clock and puts every almanac body on its SPK position at the
/// target epoch (clamped to the committed range).
#[test]
fn test_seek_epoch_repositions_bodies_while_paused() {
  use crate::simulation_api::reposition::AU_TO_KM;
  use aethervk_oshal_rlib::math::vector::Vector3;
  let mut ctx = startup_with_planets();
  let start = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);
  let end = start + Duration::from_days(30.0);
  let scene_id = ctx.create_empty_scene2(true, start, end).unwrap().scene_id;
  let target = start + Duration::from_days(17.25);

  assert!(ctx.seek_epoch_sync(scene_id, target), "seek not applied");

  let scenes = ctx.scenes.read();
  let now = scenes.time_managers.get(&scene_id).unwrap().current_epoch();
  assert!(
    (now - target).to_seconds().abs() < 1e-3,
    "clock at {now}, wanted {target}"
  );
  let scene_arc = scenes.get_scene(scene_id).unwrap();
  let g = scene_arc.read();
  let earth = g.earth.unwrap();
  let got_km = g.scene.global_transform_f64(earth.body).unwrap().position * AU_TO_KM;
  let (want_km, _, _) = AlmanacPlanet { naif_id: 399 }
    .step_with_velocity(target, &ctx.logic_state.read().almanac_data, None)
    .unwrap();
  let err = (got_km - want_km).length();
  assert!(
    err < 5.0,
    "earth off its SPK position after seek by {err} km"
  );
  drop(g);
  drop(scenes);

  // clamped to the committed range
  assert!(ctx.seek_epoch_sync(scene_id, end + Duration::from_days(3.0)));
  let scenes = ctx.scenes.read();
  let now = scenes.time_managers.get(&scene_id).unwrap().current_epoch();
  assert!((now - end).to_seconds().abs() < 1e-3);
}

/// Dump at one epoch, move elsewhere, restore: the clock returns to the dumped epoch with the
/// bodies on their SPK positions (dust rebuilt deterministically). A dump whose compatibility JSON
/// differs from the live configuration is refused.
#[test]
fn test_dump_restore_returns_to_dumped_epoch_and_checks_compatibility() {
  use crate::simulation_api::{reposition::AU_TO_KM, structs::SceneDump};
  use aethervk_oshal_rlib::math::vector::Vector3;
  let mut ctx = startup_with_planets();
  let start = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);
  let scene_id = ctx
    .create_empty_scene2(true, start, start + Duration::from_days(30.0))
    .unwrap()
    .scene_id;
  let dumped_at = start + Duration::from_days(10.0);
  assert!(ctx.seek_epoch_sync(scene_id, dumped_at));

  let dir = std::env::temp_dir().join(format!("aethervk_dump_test_{}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  ctx
    .threads
    .logic_thread
    .tx()
    .try_send(LogicCommand::DumpScene {
      scene_id,
      base_dir: dir.to_str().unwrap().to_string(),
    })
    .unwrap();
  let file = dir.join(format!("scene_{scene_id}")).join("scene.bin");
  let t0 = std::time::Instant::now();
  while !file.exists() {
    assert!(t0.elapsed().as_secs() < 5, "dump not written");
    std::thread::sleep(std::time::Duration::from_millis(20));
  }
  std::thread::sleep(std::time::Duration::from_millis(50));
  let (dump, _): (SceneDump, usize) =
    bincode::serde::decode_from_slice(&std::fs::read(&file).unwrap(), bincode::config::standard())
      .unwrap();
  assert_eq!(dump.version, SceneDump::CURRENT_VERSION);
  assert!(
    dump.compatibility.contains("\"jets\":["),
    "{}",
    dump.compatibility
  );

  let clock = |ctx: &SimulationContext| {
    ctx.scenes.read().time_managers.get(&scene_id).unwrap().current_epoch()
  };
  let wait_clock = |ctx: &SimulationContext, want: Epoch| {
    let t0 = std::time::Instant::now();
    while (clock(ctx) - want).to_seconds().abs() > 1e-3 {
      if t0.elapsed().as_secs() >= 5 {
        return false;
      }
      std::thread::sleep(std::time::Duration::from_millis(20));
    }
    true
  };

  // incompatible dump: refused, the clock stays where it is
  let moved_to = start + Duration::from_days(3.0);
  assert!(ctx.seek_epoch_sync(scene_id, moved_to));
  let mut bad = dump.clone();
  bad.compatibility = bad.compatibility.replace("\"jets\":[", "\"jets\":[{\"id\":1},");
  ctx
    .threads
    .logic_thread
    .tx()
    .try_send(LogicCommand::RestoreSceneDump {
      scene_id,
      dump: alloc::boxed::Box::new(bad),
    })
    .unwrap();
  std::thread::sleep(std::time::Duration::from_millis(500));
  assert!(
    (clock(&ctx) - moved_to).to_seconds().abs() < 1e-3,
    "incompatible dump was applied"
  );

  // compatible dump: back to the dumped epoch
  ctx
    .threads
    .logic_thread
    .tx()
    .try_send(LogicCommand::RestoreSceneDump {
      scene_id,
      dump: alloc::boxed::Box::new(dump),
    })
    .unwrap();
  assert!(
    wait_clock(&ctx, dumped_at),
    "restore did not return to the dumped epoch"
  );
  let scenes = ctx.scenes.read();
  let scene_arc = scenes.get_scene(scene_id).unwrap();
  let g = scene_arc.read();
  let got_km = g.scene.global_transform_f64(g.earth.unwrap().body).unwrap().position * AU_TO_KM;
  let (want_km, _, _) = AlmanacPlanet { naif_id: 399 }
    .step_with_velocity(dumped_at, &ctx.logic_state.read().almanac_data, None)
    .unwrap();
  assert!((got_km - want_km).length() < 5.0);
  let _ = std::fs::remove_dir_all(&dir);
}

/// Re-osculation happens at the first epoch the comet SPK can be stepped (to within a second),
/// which for Horizons segments is minutes after the requested midnight start.
#[test]
fn test_osculation_epoch_finds_first_covered_instant() {
  let midnight = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);
  let end = midnight + Duration::from_days(31.0);
  let first = midnight + Duration::from_seconds(99.3);
  let got = utils::osculation_epoch(midnight, end, |t| t >= first).unwrap();
  assert!(got >= first && (got - first).to_seconds() <= 1.0, "{got}");
  assert_eq!(
    utils::osculation_epoch(midnight, end, |_| true),
    Some(midnight)
  );
  assert_eq!(utils::osculation_epoch(midnight, end, |_| false), None);
  let late = end - Duration::from_seconds(10.0);
  let got = utils::osculation_epoch(midnight, end, |t| t >= late).unwrap();
  assert!(got >= late && (got - late).to_seconds() <= 1.0, "{got}");
}

/// Earth observer precision: a camera parented to the Earth (f32 body rotation, ~1 AU from the
/// Sun) keeps a world rotation set in f64 to ~1e-15 rad, through `set_global_transform_f64`,
/// `global_transform_f64` and a body spin step (`utils::compensate_body_spin`). With the old f32
/// camera quaternion this was ~1e-7 rad: the comet nucleus fell outside a ±9 km telescope field.
#[test]
fn test_parented_camera_keeps_f64_world_rotation_through_spin() {
  use aethervk_oshal_rlib::math::{
    quaternion::Quaternion,
    vector::{Vector3, vec3::Vec3f32, vec3f64::Vec3f64, vec4::Quat, vec4f64::Quat64},
  };
  let ctx = startup_with_planets();
  let start = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);
  let scene_id = ctx
    .create_empty_scene2(true, start, start + Duration::from_days(10.0))
    .unwrap()
    .scene_id;
  let scenes = ctx.scenes.read();
  let scene_arc = scenes.get_scene(scene_id).unwrap();
  let g = scene_arc.read();
  let earth = g.earth.unwrap().body;
  let cam = g.scene.get_entity_by_name("camera").expect("scene camera");
  let spin = |deg: f32| {
    Quat::from_axis_angle(
      Vec3f32::from_components(0.1, -0.4, 0.9).normalize(),
      deg.to_radians(),
    )
  };
  let _ = g
    .scene
    .with_component_mut(earth, |t: &mut TransformComponent| t.rotation = spin(23.0));
  g.scene.set_parent(cam, Some(earth));

  let angle = |a: Quat64, b: Quat64| {
    let d = a * b.conjugate();
    2.0 * d.vector_part().length().atan2(d.scalar_part().abs())
  };
  // a world rotation f32 cannot represent
  let world = Quat64::from_axis_angle(
    Vec3f64::from_components(0.3, 0.7, -0.2).normalize(),
    1.000_000_123_456,
  );
  let cam_world_pos = g.scene.global_transform_f64(earth).unwrap().position
    + Vec3f64::from_components(4.26e-5, 0.0, 0.0); // on the surface
  g.scene.set_global_transform_f64(cam, cam_world_pos, world).unwrap();
  let got = g.scene.global_transform_f64(cam).unwrap();
  assert!(
    angle(got.rotation, world) < 1e-12,
    "set/get: {} rad",
    angle(got.rotation, world)
  );
  assert!(
    angle(Quat64::from_quat(world.to_quat()), world) > 1e-9,
    "test rotation must not be f32-exact"
  );

  // the Earth spins: its f32 rotation changes, the camera is compensated in its local frame
  let (old_rot, new_rot) = (spin(23.0), spin(23.25));
  let _ = g
    .scene
    .with_component_mut(earth, |t: &mut TransformComponent| t.rotation = new_rot);
  let _ = g
    .scene
    .with_component_mut(cam, |h: &mut crate::scene::HighResTransformComponent| {
      utils::compensate_body_spin(h, old_rot, new_rot)
    });
  let after = g.scene.global_transform_f64(cam).unwrap();
  assert!(
    angle(after.rotation, world) < 1e-12,
    "after spin: {} rad",
    angle(after.rotation, world)
  );
  assert!(
    (after.position - got.position).length() < 1e-12,
    "camera moved by the spin step"
  );
}

/// Dust history before the start epoch needs the comet attitude without its SPK: `rotation_at`
/// alone equals the rotation of a full ephemeris step (67P-like IAU model on Mars' ephemeris).
#[test]
fn test_rotation_at_matches_step_rotation() {
  let almanac = load_planet_almanac();
  let model = crate::scene::BodyRotationalModel {
    pole_ra: 69.3,
    pole_dec: 64.1,
    prime_meridian: 114.2,
    pole_ra_rate: 0.0,
    pole_dec_rate: 0.0,
    rotation_rate: 696.5,
    body_fixed_orientation: false,
  };
  let planet = AlmanacPlanet { naif_id: 499 };
  for days in [0.0, 3.7, 400.0] {
    let epoch = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0) + Duration::from_days(days);
    let (_, _, q_step) = planet.step_with_velocity(epoch, &almanac, Some(&model)).unwrap();
    let q = planet.rotation_at(epoch, &almanac, Some(&model));
    assert_eq!(q, q_step, "day {days}");
  }
  // no model: identity for a small body, also before any ephemeris coverage
  let early = Epoch::from_gregorian_utc(1990, 1, 1, 0, 0, 0, 0);
  let q = AlmanacPlanet { naif_id: 1000012 }.rotation_at(early, &almanac, None);
  assert_eq!(q, aethervk_oshal_rlib::math::vector::vec4::Quat::identity());
}

static ASSET_SENDER: parking_lot::Mutex<
  Option<mpsc::Sender<crate::simulation_api::external_state::CAssetImported>>,
> = parking_lot::Mutex::new(None);

unsafe extern "C" fn asset_imported_cb(state_id: u32, data_ptr: *const core::ffi::c_void) {
  if state_id == 10 {
    let ev = unsafe { *(data_ptr as *const crate::simulation_api::external_state::CAssetImported) };
    if let Some(sender) = ASSET_SENDER.lock().as_ref() {
      let _ = sender.send(ev);
    }
  }
}

/// End to end through the logic thread: `ImportAsset` (thread pool) fills the scene context's
/// asset library with the mesh and its embedded texture, reports completion through the
/// `AssetImported` external state, a second import of the same file adds nothing, and
/// `SetCometAppearance` puts the imported mesh + texture on the comet's visual child.
#[test]
fn test_import_asset_command_fills_library_without_duplicates_and_wires_comet() {
  use crate::simulation_api::comet_appearance::{
    CometAppearanceWiring, CometDisplayMode, CometVisualOffset,
  };
  let glb =
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_assets/BoxTextured.glb");
  let cache_dir = std::env::temp_dir().join(format!("avk_import_cmd_{}", std::process::id()));
  let _ = std::fs::remove_dir_all(&cache_dir);

  let ctx = SimulationContext::startup(None).expect("Failed to create SimulationContext");
  let (tx, rx) = mpsc::channel();
  *ASSET_SENDER.lock() = Some(tx);
  set_external_state_simulation_callback(Some(asset_imported_cb));

  let start = Epoch::from_gregorian_utc(2025, 10, 15, 0, 0, 0, 0);
  let scene_id = ctx
    .create_empty_scene2(false, start, start + Duration::from_days(10.0))
    .expect("scene")
    .scene_id;

  let import = |request_id: u64| {
    ctx
      .threads
      .logic_thread
      .tx()
      .try_send(LogicCommand::ImportAsset {
        request_id,
        path: glb.to_str().unwrap().to_string(),
        cache_dir: cache_dir.to_str().unwrap().to_string(),
      })
      .expect("queue ImportAsset");
    rx.recv_timeout(std::time::Duration::from_secs(20))
      .expect("AssetImported event")
  };

  let first = import(41);
  assert_eq!(first.request_id, 41);
  assert_eq!(first.success, 1);
  assert_eq!(first.added_count, 2, "mesh + embedded base colour texture");
  assert_ne!(first.mesh_id, 0);

  let library = alloc::sync::Arc::clone(&ctx.scenes.read().asset_library);
  let (stats, albedo) = {
    let lib = library.read();
    let mesh = lib.mesh(first.mesh_id).expect("mesh registered");
    let albedo = mesh.bundled_textures[0].expect("albedo registered");
    assert!(lib.texture(albedo).is_some());
    (lib.stats(), albedo)
  };
  assert_eq!((stats.mesh_count, stats.texture_count), (1, 1));

  let second = import(42);
  assert_eq!(second.request_id, 42);
  assert_eq!(second.success, 1);
  assert_eq!(second.added_count, 0, "re-import must not add assets");
  assert_eq!(second.mesh_id, first.mesh_id);
  assert_eq!(library.read().stats(), stats);

  // wire it on the comet (scene is paused)
  ctx
    .threads
    .logic_thread
    .tx()
    .try_send(LogicCommand::SetCometAppearance {
      scene_id,
      wiring: CometAppearanceWiring {
        mode: CometDisplayMode::Custom,
        mesh: Some(first.mesh_id),
        textures: [Some(albedo), None, None, None],
      },
      offset: CometVisualOffset::default(),
    })
    .expect("queue SetCometAppearance");

  let visual = ctx.get_scene(scene_id).unwrap().read().comet.unwrap().visual.unwrap();
  let mut wired = None;
  for _ in 0..200 {
    let scene_arc = ctx.get_scene(scene_id).unwrap();
    let g = scene_arc.read();
    wired = g
      .scene
      .with_component(visual, |m: &crate::scene::StaticMeshComponent| m.clone())
      .filter(|m| m.asset_path.starts_with("asset:"));
    if wired.is_some() {
      break;
    }
    drop(g);
    std::thread::sleep(std::time::Duration::from_millis(10));
  }
  let mesh = wired.expect("custom mesh applied to Comet_visual");
  assert_eq!(mesh.mesh.vertices.len(), 24);
  let tex = mesh.mesh.albedo_map.as_ref().expect("albedo wired");
  assert_eq!((tex.width, tex.height), (256, 256));

  set_external_state_simulation_callback(None);
  *ASSET_SENDER.lock() = None;
  drop(ctx);
  let _ = std::fs::remove_dir_all(&cache_dir);
}

static REMOVED_SENDER: parking_lot::Mutex<
  Option<mpsc::Sender<crate::simulation_api::external_state::CAssetRemoved>>,
> = parking_lot::Mutex::new(None);

unsafe extern "C" fn asset_events_cb(state_id: u32, data_ptr: *const core::ffi::c_void) {
  match state_id {
    10 => unsafe { asset_imported_cb(state_id, data_ptr) },
    11 => {
      let ev =
        unsafe { *(data_ptr as *const crate::simulation_api::external_state::CAssetRemoved) };
      if let Some(sender) = REMOVED_SENDER.lock().as_ref() {
        let _ = sender.send(ev);
      }
    }
    _ => {}
  }
}

/// `RemoveAsset` on a live context: refused while playing; while paused the comet showing the
/// mesh is ejected to the procedural sphere, the mesh leaves the library, its bundled texture
/// stays, and `AssetRemoved` reports it.
#[test]
fn test_remove_asset_command_ejects_comet_and_is_locked_while_playing() {
  use crate::simulation_api::comet_appearance::{
    CometAppearanceComponent, CometAppearanceWiring, CometDisplayMode, CometVisualOffset,
    DEFAULT_COMET_ASSET_PATH,
  };
  let glb =
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_assets/BoxTextured.glb");
  let cache_dir = std::env::temp_dir().join(format!("avk_remove_cmd_{}", std::process::id()));
  let _ = std::fs::remove_dir_all(&cache_dir);

  let ctx = SimulationContext::startup(None).expect("Failed to create SimulationContext");
  let (itx, irx) = mpsc::channel();
  let (rtx, rrx) = mpsc::channel();
  *ASSET_SENDER.lock() = Some(itx);
  *REMOVED_SENDER.lock() = Some(rtx);
  set_external_state_simulation_callback(Some(asset_events_cb));

  let start = Epoch::from_gregorian_utc(2025, 10, 15, 0, 0, 0, 0);
  let scene_id = ctx
    .create_empty_scene2(false, start, start + Duration::from_days(10.0))
    .expect("scene")
    .scene_id;
  let send = |cmd: LogicCommand| ctx.threads.logic_thread.tx().try_send(cmd).expect("queue");
  let timeout = std::time::Duration::from_secs(20);

  send(LogicCommand::ImportAsset {
    request_id: 1,
    path: glb.to_str().unwrap().to_string(),
    cache_dir: cache_dir.to_str().unwrap().to_string(),
  });
  let imported = irx.recv_timeout(timeout).expect("AssetImported");
  let mesh_id = imported.mesh_id;
  let library = alloc::sync::Arc::clone(&ctx.scenes.read().asset_library);
  let albedo = library.read().mesh(mesh_id).unwrap().bundled_textures[0].unwrap();

  send(LogicCommand::SetCometAppearance {
    scene_id,
    wiring: CometAppearanceWiring {
      mode: CometDisplayMode::Custom,
      mesh: Some(mesh_id),
      textures: [Some(albedo), None, None, None],
    },
    offset: CometVisualOffset::default(),
  });
  let visual = ctx.get_scene(scene_id).unwrap().read().comet.unwrap().visual.unwrap();
  let asset_path = || {
    ctx
      .get_scene(scene_id)
      .unwrap()
      .read()
      .scene
      .with_component(visual, |m: &crate::scene::StaticMeshComponent| {
        m.asset_path.clone()
      })
      .unwrap()
  };
  for _ in 0..200 {
    if asset_path().starts_with("asset:") {
      break;
    }
    std::thread::sleep(std::time::Duration::from_millis(10));
  }
  assert!(asset_path().starts_with("asset:"), "custom mesh displayed");

  // playing: refused, nothing changes
  ctx.get_scene(scene_id).unwrap().read().time_state.write().speed =
    aethervk_oshal_rlib::os::time::v2::SimSpeed::Realtime;
  send(LogicCommand::RemoveAsset {
    request_id: 7,
    asset_id: mesh_id,
  });
  let refused = rrx.recv_timeout(timeout).expect("AssetRemoved");
  assert_eq!((refused.request_id, refused.success), (7, 0));
  assert!(library.read().mesh(mesh_id).is_some());
  assert!(asset_path().starts_with("asset:"));

  // paused: ejected + removed
  ctx.get_scene(scene_id).unwrap().read().time_state.write().speed =
    aethervk_oshal_rlib::os::time::v2::SimSpeed::Paused;
  send(LogicCommand::RemoveAsset {
    request_id: 8,
    asset_id: mesh_id,
  });
  let removed = rrx.recv_timeout(timeout).expect("AssetRemoved");
  assert_eq!((removed.request_id, removed.asset_id), (8, mesh_id));
  assert_eq!((removed.success, removed.ejected), (1, 1));
  assert_eq!(asset_path(), DEFAULT_COMET_ASSET_PATH);
  let appearance = ctx
    .get_scene(scene_id)
    .unwrap()
    .read()
    .scene
    .with_component(visual, |a: &CometAppearanceComponent| *a)
    .unwrap();
  assert_eq!(appearance.wiring, CometAppearanceWiring::default());
  let stats = library.read().stats();
  assert_eq!((stats.mesh_count, stats.texture_count), (0, 1));

  set_external_state_simulation_callback(None);
  *ASSET_SENDER.lock() = None;
  *REMOVED_SENDER.lock() = None;
  drop(ctx);
  let _ = std::fs::remove_dir_all(&cache_dir);
}

/// The "dust visibility" softening is per scene, clamped, and read by the render thread.
#[test]
fn test_set_dust_softening_clamps_and_stores() {
  let mut ctx = startup_with_planets();
  let start = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);
  let scene_id = ctx
    .create_empty_scene2(true, start, start + Duration::from_days(10.0))
    .unwrap()
    .scene_id;
  let read = |ctx: &SimulationContext| {
    let scenes = ctx.scenes.read();
    let scene_arc = scenes.get_scene(scene_id).unwrap();
    let g = scene_arc.read();
    f32::from_bits(g.dust_softening.load(core::sync::atomic::Ordering::Relaxed))
  };
  assert_eq!(read(&ctx), crate::scene::dust::DUST_SOFTENING_DEFAULT);
  assert_eq!(ctx.set_dust_softening(scene_id, 1e-3), Some(1e-3));
  assert_eq!(read(&ctx), 1e-3);
  assert_eq!(
    ctx.set_dust_softening(scene_id, 0.0),
    Some(crate::scene::dust::DUST_SOFTENING_MIN)
  );
  assert_eq!(ctx.set_dust_softening(scene_id + 999, 1e-3), None);
}

/// The change stream lists parents before children whatever their entity ids: a camera spawned
/// before the Earth (lower id) used to be streamed first, so C# compared this tick's camera with the
/// previous tick's Earth (0.13 AU apart after a multi-day step: "Earth observer mode invariant
/// broken").
#[test]
fn test_change_stream_lists_parents_before_children() {
  let scene = crate::scene::Scene::new(alloc::sync::Arc::new(parking_lot::RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("parents_first"),
  )));
  let camera = scene.spawn_entity("camera");
  let other = scene.spawn_entity("other");
  let frame = scene.spawn_entity("earth_subtree");
  let earth = scene.spawn_entity("earth");
  scene.set_parent(earth, Some(frame));
  scene.set_parent(camera, Some(earth));
  assert!(
    camera.as_ffi() < earth.as_ffi(),
    "the camera must sort first by id for this test"
  );

  // entity id order, as `changed_entities` (a BTreeMap) yields it
  let mut changes: Vec<(u64, u64, ())> = [camera, other, frame, earth]
    .iter()
    .map(|e| {
      (
        e.as_ffi(),
        ComponentForeignId::HighResTransform.as_u64(),
        (),
      )
    })
    .collect();
  changes.sort_by_key(|c| c.0);
  utils::sort_parents_first(&mut changes, &scene.hierarchy.read());

  let order: Vec<u64> = changes.iter().map(|c| c.0).collect();
  let pos = |e: crate::scene::EntityId| order.iter().position(|&id| id == e.as_ffi()).unwrap();
  assert!(
    pos(frame) < pos(earth) && pos(earth) < pos(camera),
    "{order:?}"
  );
  // roots keep id order among themselves (stable sort)
  assert!(pos(other) < pos(frame), "{order:?}");
}

/// Angle (rad) between the camera's view axis (engine forward = local −Y) and `target − camera`,
/// read back from the scene graph exactly as the renderer does.
fn camera_aim_error(
  g: &crate::simulation_api::structs::SceneContext,
  cam: EntityId,
  target: [f64; 3],
) -> f64 {
  use aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64;
  let t = g.scene.global_transform_f64(cam).unwrap();
  let p: [f64; 3] = t.position.into();
  let fwd: [f64; 3] = t.rotation.rotate_vector(Vec3f64::from_components(0.0, -1.0, 0.0)).into();
  let to = [target[0] - p[0], target[1] - p[1], target[2] - p[2]];
  let c = [
    fwd[1] * to[2] - fwd[2] * to[1],
    fwd[2] * to[0] - fwd[0] * to[2],
    fwd[0] * to[1] - fwd[1] * to[0],
  ];
  let d = fwd[0] * to[0] + fwd[1] * to[1] + fwd[2] * to[2];
  (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().atan2(d)
}

/// The app's real scene (almanac Earth with its BPC rotation, the scene camera parented to the
/// Earth like the C# does): the native Earth observer aims at the Sun when a Sun submode is set
/// (lock-in and tracking) and, tracking, keeps aiming at it in every commit while the simulation
/// plays at 1 h/s (the Earth turns ~15° per second).
#[test]
fn test_earth_observer_aims_at_the_sun_on_the_real_scene() {
  use crate::simulation_api::earth_observer::EarthObserverMode;
  use aethervk_oshal_rlib::os::time::v2::SimSpeed;
  let ctx = startup_with_planets();
  let start = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);
  let scene_id = ctx
    .create_empty_scene2(true, start, start + Duration::from_days(10.0))
    .unwrap()
    .scene_id;
  let (cam, earth) = {
    let scenes = ctx.scenes.read();
    let g = scenes.get_scene(scene_id).unwrap();
    let g = g.read();
    let cam = g.scene.get_entity_by_name("camera").expect("scene camera");
    let earth = g.earth.unwrap().body;
    g.scene.set_parent(cam, Some(earth));
    (cam, earth)
  };
  let sun = [0.0, 0.0, 0.0];
  // observer near the sub-solar point (Sun high above the horizon), from the Earth's real rotation
  let (lat_deg, lon_deg) = {
    let scenes = ctx.scenes.read();
    let g = scenes.get_scene(scene_id).unwrap();
    let g = g.read();
    let e = g.scene.global_transform_f64(earth).unwrap();
    let er = crate::simulation_api::earth_observer::qnormalize([
      e.rotation[0],
      e.rotation[1],
      e.rotation[2],
      e.rotation[3],
    ]);
    let ep: [f64; 3] = e.position.into();
    let n = (ep[0] * ep[0] + ep[1] * ep[1] + ep[2] * ep[2]).sqrt();
    let bf = crate::simulation_api::earth_observer::rotate(
      crate::simulation_api::earth_observer::qinverse(er),
      [-ep[0] / n, -ep[1] / n, -ep[2] / n],
    );
    (bf[2].asin().to_degrees(), bf[1].atan2(bf[0]).to_degrees())
  };
  let read_error = || {
    let scenes = ctx.scenes.read();
    let g = scenes.get_scene(scene_id).unwrap();
    let g = g.read();
    camera_aim_error(&g, cam, sun)
  };

  // lock-in aims when set (the C# computes the body-fixed look the same way)
  for mode in [EarthObserverMode::SunTracking, EarthObserverMode::SunLockIn] {
    let look = {
      let scenes = ctx.scenes.read();
      let g = scenes.get_scene(scene_id).unwrap();
      let g = g.read();
      let e = g.scene.global_transform_f64(earth).unwrap();
      let er = [e.rotation[0], e.rotation[1], e.rotation[2], e.rotation[3]];
      let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());
      let r = crate::simulation_api::earth_observer::EARTH_RADIUS_AU;
      let surf = [
        r * lat.cos() * lon.cos(),
        r * lat.cos() * lon.sin(),
        r * lat.sin(),
      ];
      let ep: [f64; 3] = e.position.into();
      let sw = crate::simulation_api::earth_observer::rotate(
        crate::simulation_api::earth_observer::qnormalize(er),
        surf,
      );
      let cam_pos = [ep[0] + sw[0], ep[1] + sw[1], ep[2] + sw[2]];
      let aimed = crate::simulation_api::earth_observer::look_at(
        [-cam_pos[0], -cam_pos[1], -cam_pos[2]],
        lat_deg,
      );
      crate::simulation_api::earth_observer::qnormalize(
        crate::simulation_api::earth_observer::qmul(
          crate::simulation_api::earth_observer::qinverse(
            crate::simulation_api::earth_observer::qnormalize(er),
          ),
          aimed,
        ),
      )
    };
    assert!(
      ctx.set_earth_observer(
        scene_id,
        cam.as_ffi(),
        Some(mode),
        earth.as_ffi(),
        0,
        lat_deg,
        lon_deg,
        look
      ),
      "{mode:?}: native observer refused"
    );
    let e = read_error();
    assert!(e < 1e-6, "{mode:?} when set: Sun {e} rad off the view axis");
  }

  // tracking while playing: every commit re-aims
  assert!(ctx.set_earth_observer(
    scene_id,
    cam.as_ffi(),
    Some(EarthObserverMode::SunTracking),
    earth.as_ffi(),
    0,
    lat_deg,
    lon_deg,
    [0.0, 0.0, 0.0, 1.0]
  ));
  assert!(ctx.start_simulation(scene_id, SimSpeed::OneHourPerSec));
  let mut worst = 0.0f64;
  let mut samples = 0;
  let t0 = std::time::Instant::now();
  while t0.elapsed() < std::time::Duration::from_millis(1500) {
    std::thread::sleep(std::time::Duration::from_millis(25));
    let e = read_error();
    // below the horizon the observer holds its last aim (the Earth is in the way)
    let above = {
      let scenes = ctx.scenes.read();
      let g = scenes.get_scene(scene_id).unwrap();
      let g = g.read();
      let c: [f64; 3] = g.scene.global_transform_f64(cam).unwrap().position.into();
      let ep: [f64; 3] = g.scene.global_transform_f64(earth).unwrap().position.into();
      let z = [c[0] - ep[0], c[1] - ep[1], c[2] - ep[2]];
      -(z[0] * c[0] + z[1] * c[1] + z[2] * c[2]) > 0.0
    };
    if above {
      worst = worst.max(e);
      samples += 1;
    }
  }
  let _ = ctx.pause_simulation_sync(scene_id);
  ctx.threads.logic_thread.tx().try_send(LogicCommand::Shutdown).unwrap();
  assert!(
    samples > 5,
    "the Sun must be above the horizon for part of the run ({samples})"
  );
  assert!(
    worst < 1e-6,
    "tracking while playing: Sun up to {worst} rad off the view axis"
  );
}
