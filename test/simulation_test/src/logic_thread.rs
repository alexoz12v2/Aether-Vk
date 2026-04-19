use aethervk_core_rlib::{
  gpu::PhysicalScene,
  scene::{CameraComponent, EntityId, Scene, SunComponent, TransformComponent},
};
use aethervk_oshal_rlib::math::{
  FloatLike,
  floating::FloatOps,
  matrix::{Matrix4, MatrixVectorMul, SquareMatrix, mat4::Mat4x4f32},
  quaternion::Quaternion,
  vector::{Vector, Vector3, Vector4, vec3::Vec3f32, vec4::Quat},
};
use std::sync::{atomic::AtomicBool, Arc, mpsc};
use std::time::Instant;
use anise::almanac::Almanac;
use anise::prelude::Epoch;
use crate::{constants, utils};

pub enum LogicCommand {
  RotateCamera { delta_x: f32, delta_y: f32 },
  ZoomCamera { amount: f32 },
  PanCursor { delta_x: f32, delta_y: f32 },
  MoveCursor { axis: Vec3f32, amount: f32 },
  RaycastCursor { ndc_x: f32, ndc_y: f32 },
  ResetCursor,
  Resize { width: u32, height: u32 },
  SnapCameraToCursor,
  SnapCursorToSun,
  ToggleGrid,
  ToggleMeasureTool,
  SelectEntity { id: EntityId },
  CycleTimeScale,
  CyclePlanet { forward: bool },
  TogglePlanetOutlines,
  ResetCamera,
  Exit,
}

pub struct LogicState {
  yaw: f32,
  pitch: f32,
  camera_distance: f32,
  physical_scene: PhysicalScene,
}

#[derive(Clone, Copy, PartialEq)]
enum TimeScale {
  Stopped,
  OneDay,
  OneMonth,
  OneYear,
}

impl TimeScale {
  fn cycle(self) -> Self {
    match self {
      TimeScale::OneDay => TimeScale::OneMonth,
      TimeScale::OneMonth => TimeScale::OneYear,
      TimeScale::OneYear => TimeScale::Stopped,
      TimeScale::Stopped => TimeScale::OneDay,
    }
  }
  fn to_days_per_st_second(self) -> f64 {
    match self {
      TimeScale::Stopped => 0.0,
      TimeScale::OneDay => 1.0,
      TimeScale::OneMonth => 30.436875,
      TimeScale::OneYear => 365.25,
    }
  }
  fn label(self) -> &'static str {
    match self {
      TimeScale::Stopped => "Stopped",
      TimeScale::OneDay => "1D = 1STS",
      TimeScale::OneMonth => "1M = 1STS",
      TimeScale::OneYear => "1Y = 1STS",
    }
  }
}

const FIXED_TIME_STEP: f32 = 1.0 / 60.0;

pub fn start_logic_thread(
  rx: mpsc::Receiver<LogicCommand>,
  scene_shared: Arc<Scene>,
  camera_entity: EntityId,
  cursor_entity: EntityId,
  grid_entity: EntityId,
  planets_ids: Vec<(i32, EntityId, f64, f32)>,
  sun_entity: EntityId,
  sun_radius: f32,
  assets_dir: std::path::PathBuf,
  outlines_enabled: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
  std::thread::spawn(move || {
    let mut state = LogicState {
      yaw: std::f32::consts::PI,
      pitch: 0.0,
      camera_distance: 60.0,
      physical_scene: PhysicalScene::new(),
    };

    let mut last_time = Instant::now();
    let mut accumulator = 0.0;

    println!("Starting Almanac load...");
    let start_load = std::time::Instant::now();

    let almanac =
      utils::load_almanac(assets_dir.join("planets").as_path()).expect("couldn't load almanac");
    println!(
      "Finished Almanac load on logic thread! Took {:?}",
      start_load.elapsed()
    );

    let mut current_scale = TimeScale::OneDay;
    let epoch_start = anise::time::Epoch::from_gregorian_utc_at_midnight(2000, 1, 1);
    let epoch_end = anise::time::Epoch::from_gregorian_utc_at_midnight(2100, 1, 1);
    let mut current_epoch = epoch_start;
    let mut st_seconds_elapsed = 0.0_f64;
    let mut following_entity: Option<EntityId> = None;
    let mut focused_planet_idx: Option<usize> = None;

    loop {
      let current_time = Instant::now();
      let delta_time = current_time.duration_since(last_time).as_secs_f32();
      last_time = current_time;

      // Safeguard against spiral of death
      let dt = if delta_time > 0.25 { 0.25 } else { delta_time };
      accumulator += dt;

      // --- Fixed Update Step (Physics/Simulation) ---
      let scale_days = current_scale.to_days_per_st_second();
      let step_days = scale_days * (FIXED_TIME_STEP as f64);

      while accumulator >= FIXED_TIME_STEP {
        logic_fixed_update_step(
          scene_shared.as_ref(),
          camera_entity,
          cursor_entity,
          &planets_ids,
          sun_entity,
          accumulator,
          &almanac.almanac,
          &mut current_scale,
          epoch_start,
          epoch_end,
          &mut current_epoch,
          &mut st_seconds_elapsed,
          &mut following_entity,
          step_days,
        );
        accumulator -= FIXED_TIME_STEP;
      }

      // --- Variable Update Step (Input routing & Commands) ---
      while let Ok(command) = rx.try_recv() {
        if let LogicCommand::Exit = command {
          println!("\nExiting logic thread.");
          return;
        }

        logic_update_command(
          command,
          &mut state,
          scene_shared.as_ref(),
          camera_entity,
          cursor_entity,
          grid_entity,
          &outlines_enabled,
          &mut following_entity,
          &mut focused_planet_idx,
          &planets_ids,
          &mut current_scale,
          &mut current_epoch,
          &mut st_seconds_elapsed,
          epoch_start,
          epoch_end,
        );
      }

      std::thread::sleep(std::time::Duration::from_millis(1));
    }
  })
}

fn logic_fixed_update_step(
  scene_guard: &Scene,
  camera_entity: EntityId,
  cursor_entity: EntityId,
  planets_ids: &Vec<(i32, EntityId, f64, f32)>,
  sun_entity: EntityId,
  accumulator: f32,
  almanac: &Almanac,
  current_scale: &mut TimeScale,
  epoch_start: Epoch,
  epoch_end: Epoch,
  current_epoch: &mut Epoch,
  st_seconds_elapsed: &mut f64,
  following_entity: &mut Option<EntityId>,
  step_days: f64,
) {
  if *current_epoch >= epoch_end && *current_scale != TimeScale::Stopped {
    *current_scale = TimeScale::Stopped;
  }

  for (naif_id, entity, rot_period, planet_radius) in planets_ids.iter() {
    let pos = utils::get_almanac_pos(*naif_id, *current_epoch, &almanac);
    scene_guard.with_component_mut(*entity, |c: &mut TransformComponent| {
      c.position = pos;
      c.scale = Vec3f32::splat(1.0);
      let rotations = if *rot_period != 0.0 {
        step_days * 24.0 / rot_period
      } else {
        0.0
      };
      let radians = (rotations * core::f64::consts::TAU) as f32;
      let rot_delta = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), radians);
      c.rotation = (c.rotation * rot_delta).normalize();
    });
  }

  // Also update Sun
  let pos = utils::get_almanac_pos(constants::PlanetNaifId::SUN, *current_epoch, &almanac);
  scene_guard.with_component_mut(sun_entity, |c: &mut TransformComponent| {
    c.position = pos;
    let rot_period = 25.05; // Sun's equatorial rotation period in days
    let rotations = step_days * 24.0 / rot_period;
    let radians = (rotations * core::f64::consts::TAU) as f32;
    let rot_delta = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), radians);
    c.rotation = (c.rotation * rot_delta).normalize();
  });

  if let Some(target) = *following_entity {
    let mut t_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
    if let Some(t) = scene_guard.global_transform(target) {
      t_pos = t.position;
    }

    let mut cam_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
    let mut cursor_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
    scene_guard.with_component(camera_entity, |c: &TransformComponent| cam_pos = c.position);
    scene_guard.with_component(cursor_entity, |c: &TransformComponent| {
      cursor_pos = c.position
    });

    let offset = cam_pos - cursor_pos;

    scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
      c.position = t_pos;
    });
    scene_guard.with_component_mut(camera_entity, |c: &mut TransformComponent| {
      c.position = t_pos + offset;
    });
  }

  if *current_scale != TimeScale::Stopped || *current_epoch >= epoch_end {
    *st_seconds_elapsed += FIXED_TIME_STEP as f64;
  }

  if accumulator < FIXED_TIME_STEP {
    let et_start = epoch_start.to_tai_seconds();
    let et_end = epoch_end.to_tai_seconds();
    let et_now = current_epoch.to_tai_seconds();

    let progress = ((et_now - et_start) / (et_end - et_start)).clamp(0.0, 1.0);
    let bar_len = 30;
    let filled_len = (progress * bar_len as f64) as usize;
    let bar = format!(
      "{}>{}",
      "=".repeat(filled_len),
      " ".repeat(bar_len - filled_len)
    );

    let et = et_now;
    let st = st_seconds_elapsed;
    let utc = format!("{}", current_epoch);

    print!(
      "\r[{}] ET: {:.0} | ST: {:.1}s | UTC: {} | Scale: {}   ",
      bar,
      et,
      st,
      utc,
      current_scale.label()
    );
    use std::io::Write;
    let _ = std::io::stdout().flush();
  }

  if *current_scale != TimeScale::Stopped {
    *current_epoch += anise::time::Duration::from_days(step_days);
  }
}

fn logic_update_command(
  command: LogicCommand,
  state: &mut LogicState,
  scene_guard: &Scene,
  camera_entity: EntityId,
  cursor_entity: EntityId,
  grid_entity: EntityId,
  outlines_enabled: &AtomicBool,
  following_entity: &mut Option<EntityId>,
  focused_planet_idx: &mut Option<usize>,
  planets_ids: &[(i32, EntityId, f64, f32)],
  current_scale: &mut TimeScale,
  current_epoch: &mut anise::time::Epoch,
  st_seconds_elapsed: &mut f64,
  epoch_start: anise::time::Epoch,
  epoch_end: anise::time::Epoch,
) {
  let mut cam_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
  let mut cam_rot = Quat::identity();
  scene_guard.with_component(camera_entity, |c: &TransformComponent| {
    cam_pos = c.position;
    cam_rot = c.rotation;
  });

  let mut cursor_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
  scene_guard.with_component(cursor_entity, |c: &TransformComponent| {
    cursor_pos = c.position;
  });

  let mut offset = cam_pos - cursor_pos;
  let mut dist = offset.length();
  if dist < 0.1 {
    dist = 0.1;
  }

  match command {
    LogicCommand::Exit => {} // handled above
    LogicCommand::TogglePlanetOutlines => {
      let current = outlines_enabled.load(std::sync::atomic::Ordering::Relaxed);
      outlines_enabled.store(!current, std::sync::atomic::Ordering::Relaxed);
    }
    LogicCommand::ResetCamera => {
      *following_entity = None;
      let ssb = Vec3f32::from_components(0.0, 0.0, 0.0);
      let offset = Vec3f32::from_components(
        0.0,
        -100000.0 / crate::constants::DISTANCE_SCALE_FACTOR as f32, // South
        0.0,
      );

      state.yaw = std::f32::consts::PI;
      state.pitch = 0.0;
      let yaw_quat = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), state.yaw);
      let pitch_quat = Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), state.pitch);
      let new_rot = (yaw_quat * pitch_quat).normalize();

      scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
        c.position = ssb;
      });
      scene_guard.with_component_mut(camera_entity, |c: &mut TransformComponent| {
        c.position = ssb + offset;
        c.rotation = new_rot;
      });
    }
    LogicCommand::RotateCamera { delta_x, delta_y } => {
      let rotation_speed = 0.005;
      state.yaw += delta_x * rotation_speed;
      state.pitch -= delta_y * rotation_speed;

      state.yaw = state.yaw.fmod(<f32 as FloatOps>::PI * 2.0);
      state.pitch = state.pitch.clamp(-1.55, 1.55);

      let yaw_quat = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), state.yaw);
      let pitch_quat = Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), state.pitch);
      let new_rot = yaw_quat * pitch_quat;

      let rot_delta = new_rot * cam_rot.conjugate();
      let new_offset = rot_delta.rotate_vector(offset);

      scene_guard.with_component_mut(camera_entity, |c: &mut TransformComponent| {
        c.position = cursor_pos + new_offset;
        c.rotation = new_rot;
      });
    }
    LogicCommand::ZoomCamera { amount } => {
      let zoom_speed = dist * 0.1;
      let mut new_dist = dist - amount * zoom_speed;
      if new_dist < 0.1 {
        new_dist = 0.1;
      }
      let new_offset = offset.normalize() * new_dist;
      scene_guard.with_component_mut(camera_entity, |c: &mut TransformComponent| {
        c.position = cursor_pos + new_offset;
      });
    }
    LogicCommand::PanCursor { delta_x, delta_y } => {
      *following_entity = None; // break following
      let pan_speed = dist * 0.001;

      let right = cam_rot.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
      let up = cam_rot.rotate_vector(Vec3f32::from_components(0.0, 1.0, 0.0));
      let translation = right * (-delta_x * pan_speed) + up * (delta_y * pan_speed);

      scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
        c.position = c.position + translation;
      });
      scene_guard.with_component_mut(camera_entity, |c: &mut TransformComponent| {
        c.position = c.position + translation;
      });
    }
    LogicCommand::MoveCursor { axis, amount } => {
      *following_entity = None; // break following
      scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
        c.position = c.position + axis * amount;
      });
      scene_guard.with_component_mut(camera_entity, |c: &mut TransformComponent| {
        c.position = c.position + axis * amount;
      });
    }
    LogicCommand::ResetCursor => {
      *following_entity = None; // break following
      scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
        c.position = Vec3f32::from_components(0.0, 0.0, 0.0);
      });
    }
    LogicCommand::Resize { width, height } => {
      scene_guard.with_component_mut(camera_entity, |c: &mut CameraComponent| {
        c.projection = Mat4x4f32::perspective_vk(
          45.0f32.to_radians(),
          width as f32 / height as f32,
          0.1,
          10000.0,
        );
      });
    }
    LogicCommand::RaycastCursor { ndc_x, ndc_y } => {
      *following_entity = None; // break following
      let mut view_proj_inv = Mat4x4f32::identity();
      let mut cam_pos = Vec3f32::from_components(0.0, 0.0, 0.0);

      let mut view = Mat4x4f32::identity();
      scene_guard.with_component(camera_entity, |c: &TransformComponent| {
        cam_pos = c.position;
        // Look along Z axis as fallback if eye==center? No, in Z-up system, looking along Y.
        let mut dir = cursor_pos - cam_pos;
        if dir.length_squared() < 1e-6 {
          dir = Vec3f32::from_components(0.0, 1.0, 0.0);
        }
        view = Mat4x4f32::look_at(
          cam_pos,
          cam_pos + dir,
          Vec3f32::from_components(0.0, 0.0, 1.0),
        );
      });

      scene_guard.with_component(camera_entity, |cam: &CameraComponent| {
        let proj = cam.projection;
        let view_proj = proj * view;
        view_proj_inv = view_proj.inverse().unwrap_or(Mat4x4f32::identity());
      });

      let ndc_near =
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(ndc_x, ndc_y, 0.0, 1.0);
      let ndc_far =
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(ndc_x, ndc_y, 1.0, 1.0);

      let mut world_near = view_proj_inv.mul_vector(ndc_near);
      let mut world_far = view_proj_inv.mul_vector(ndc_far);

      if world_near.w() != 0.0 {
        world_near = world_near / world_near.w();
      }
      if world_far.w() != 0.0 {
        world_far = world_far / world_far.w();
      }

      let ray_origin = Vec3f32::from_components(world_near.x(), world_near.y(), world_near.z());
      let ray_target = Vec3f32::from_components(world_far.x(), world_far.y(), world_far.z());

      let ray_dir = (ray_target - ray_origin).normalize();

      let max_distance = 2.0;
      let mut target_pos = ray_origin + ray_dir * max_distance;

      if ray_dir.z().abs() > 1e-6 {
        let t = -ray_origin.z() / ray_dir.z();
        if t > 0.0 && t <= max_distance {
          target_pos = ray_origin + ray_dir * t;
        }
      }

      scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
        c.position = target_pos;
      });
    }
    LogicCommand::SnapCursorToSun => {
      let mut target_sun_id = None;
      scene_guard.query1::<SunComponent, _>(|sun_id, _sun| {
        target_sun_id = Some(sun_id);
      });

      if let Some(sun_id) = target_sun_id {
        *following_entity = Some(sun_id);
        if let Some(t) = scene_guard.global_transform(sun_id) {
          scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
            c.position = t.position;
          });
        }
      }
    }
    LogicCommand::SnapCameraToCursor => {
      let offset = Vec3f32::from_components(0.0, -60.0, 0.0);
      state.yaw = std::f32::consts::PI;
      state.pitch = 0.0;
      let yaw_quat = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), state.yaw);
      let pitch_quat = Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), state.pitch);
      let new_rot = (yaw_quat * pitch_quat).normalize();
      scene_guard.with_component_mut(camera_entity, |c: &mut TransformComponent| {
        c.position = cursor_pos + offset;
        c.rotation = new_rot;
      });
    }
    LogicCommand::ToggleGrid => {
      let has_grid = scene_guard
        .with_component(
          grid_entity,
          |_c: &aethervk_core_rlib::scene::GridComponent| {},
        )
        .is_some();
      if has_grid {
        let _ =
          scene_guard.remove_component::<aethervk_core_rlib::scene::GridComponent>(grid_entity);
      } else {
        let _ = scene_guard.add_component(grid_entity, aethervk_core_rlib::scene::GridComponent {});
      }
    }
    LogicCommand::ToggleMeasureTool => {}
    LogicCommand::SelectEntity { id: _ } => {}
    LogicCommand::CycleTimeScale => {
      if *current_epoch >= epoch_end {
        *current_epoch = epoch_start;
        *st_seconds_elapsed = 0.0;
        *current_scale = TimeScale::OneDay;
      } else {
        *current_scale = current_scale.cycle();
      }
    }
    LogicCommand::CyclePlanet { forward } => {
      if !planets_ids.is_empty() {
        let current_idx = focused_planet_idx.unwrap_or(planets_ids.len() - 1);
        let new_idx = if forward {
          (current_idx + 1) % planets_ids.len()
        } else {
          (current_idx + planets_ids.len() - 1) % planets_ids.len()
        };
        *focused_planet_idx = Some(new_idx);
        let (_, entity, _, _) = planets_ids[new_idx];

        *following_entity = Some(entity);

        let mut p_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
        scene_guard.with_component(entity, |t: &TransformComponent| {
          p_pos = t.position;
        });

        scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
          c.position = p_pos;
        });

        let offset = Vec3f32::from_components(0.0, -60.0, 0.0);
        state.yaw = std::f32::consts::PI;
        state.pitch = 0.0;
        scene_guard.with_component_mut(camera_entity, |c: &mut TransformComponent| {
          c.position = p_pos + offset;
          c.rotation = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), std::f32::consts::PI);
        });
      }
    }
  }

  let new_cam_pos = scene_guard
    .with_component(camera_entity, |c: &TransformComponent| c.position)
    .unwrap();
  let new_cursor_pos = scene_guard
    .with_component(cursor_entity, |c: &TransformComponent| c.position)
    .unwrap();
  let new_dist = (new_cam_pos - new_cursor_pos).length();

  let scale_factor = (new_dist * 0.01).clamp(0.02, 0.05);
  scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
    c.scale = Vec3f32::from_components(scale_factor, scale_factor, scale_factor);
  });
}
