use super::*;
use crate::simulation_api::structs::LogicCommand;

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
