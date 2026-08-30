use super::*;
use crate::simulation_api::structs::LogicCommand;
use crate::simulation_api::{SimulationContext, set_external_state_simulation_callback};
use crate::simulation_api::external_state::CCometInitialized;
use crate::scene::AlmanacPlanet;
use crate::scene::trajectory::TrajectoryComponent;
use std::sync::mpsc;
use hifitime::{Epoch, Duration};

static MOCK_SENDER: parking_lot::Mutex<Option<mpsc::Sender<CCometInitialized>>> = parking_lot::Mutex::new(None);

unsafe extern "C" fn mock_external_state_cb(state_id: u32, data_ptr: *const core::ffi::c_void) {
    if state_id == 4 { // CometInitialized
        let comet_init = unsafe { *(data_ptr as *const CCometInitialized) };
        if let Some(sender) = MOCK_SENDER.lock().as_ref() {
            let _ = sender.send(comet_init);
        }
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

    fn fetch_spk() -> std::path::PathBuf {
        use std::io::Write;
        // Query Horizons API for 67P (spk id 1000012)
        let url = "https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND=%2790000703%3B%27&MAKE_EPHEM=%27YES%27&EPHEM_TYPE=%27SPK%27&OBJ_DATA=%27NO%27&START_TIME=%272025-10-01%27&STOP_TIME=%272025-11-02%27";
        let resp = reqwest::blocking::get(url).expect("Failed to fetch SPK");
        let text = resp.text().expect("Failed to read text");
        
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
        let decoded = general_purpose::STANDARD.decode(&base64_clean).unwrap_or_else(|_| general_purpose::STANDARD_NO_PAD.decode(&base64_clean).expect("Failed to decode base64"));
        
        let path = std::env::temp_dir().join("1000012.bsp");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&decoded).unwrap();
        path
    }

    // Manually load the almanac into logic_state so valid dates pass
    let spk_path = fetch_spk();
    {
        let mut logic_state = ctx.logic_state.write();
        logic_state.almanac_data.load_almanac("../../assets/planets/pck00011.pca").expect("Failed to load PCK");
        logic_state.almanac_data.load_almanac("../../assets/planets/gm_de431.pca").expect("Failed to load GM");
        logic_state.almanac_data.load_almanac("../../assets/planets/de442.bsp").expect("Failed to load DE442");
        logic_state.almanac_data.load_almanac(spk_path.to_str().unwrap()).expect("Failed to load fetched SPK");
    }

    // Prepare valid scene and dates (within 2025-10-01 and 2025-11-02)
    // 2025-10-15 TDB in seconds since J2000
    let start = Epoch::from_gregorian_utc(2025, 10, 15, 0, 0, 0, 0); 
    let end = start + Duration::from_days(10.0);
    let scene_ret = ctx.create_empty_scene2(false, start, end).expect("Failed to create empty scene");
    let scene_id = scene_ret.scene_id;

    // --- 1. Test for failure: Out of bounds epoch ---
    let bad_start = Epoch::from_tdb_seconds(31557600000.0); // year 3000
    let bad_end = bad_start + Duration::from_days(10.0);

    ctx.threads.logic_thread.tx().try_send(LogicCommand::TryInitComet {
        scene_id,
        spk_id: 1000012, // 67P
        proposed_start: bad_start,
        proposed_end: bad_end,
    }).unwrap();

    let result = rx.recv_timeout(std::time::Duration::from_secs(5)).expect("Timeout waiting for failure callback");
    assert_eq!(result.success, 0, "Expected TryInitComet to fail for out-of-bounds epoch");

    // Verify ECS rollback / lack of attachment on failure
    {
        let scene_ctx = ctx.scenes.read().get(&scene_id).cloned().unwrap();
        let scene_guard = scene_ctx.read();
        let comet = scene_guard.comet.unwrap();
        let has_planet: bool = scene_guard.scene.has_component::<AlmanacPlanet>(comet.body).into();
        let has_traj: bool = scene_guard.scene.has_component::<TrajectoryComponent>(comet.orbit).into();
        assert!(!has_planet, "AlmanacPlanet should not be attached on failure");
        assert!(!has_traj, "TrajectoryComponent should not be attached on failure");
    }

    // --- 2. Test for success: Valid epoch ---
    ctx.threads.logic_thread.tx().try_send(LogicCommand::TryInitComet {
        scene_id,
        spk_id: 1000012, // 67P
        proposed_start: start,
        proposed_end: end,
    }).unwrap();

    let result = rx.recv_timeout(std::time::Duration::from_secs(15)).expect("Timeout waiting for success callback");
    assert_eq!(result.success, 1, "Expected TryInitComet to succeed for valid epoch");

    // Verify attachment on success
    {
        let scene_ctx = ctx.scenes.read().get(&scene_id).cloned().unwrap();
        let scene_guard = scene_ctx.read();
        let comet = scene_guard.comet.unwrap();
        let has_planet: bool = scene_guard.scene.has_component::<AlmanacPlanet>(comet.body).into();
        let has_traj: bool = scene_guard.scene.has_component::<TrajectoryComponent>(comet.orbit).into();
        assert!(has_planet, "AlmanacPlanet should be attached on success");
        assert!(has_traj, "TrajectoryComponent should be attached on success");
    }

    ctx.threads.logic_thread.tx().try_send(LogicCommand::Shutdown).unwrap();
}
