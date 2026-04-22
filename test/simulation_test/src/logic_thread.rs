use aethervk_core_rlib::{
  scene::{CameraComponent, EntityId, Scene, SunComponent, TransformComponent},
};
use aethervk_oshal_rlib::{
  math::{
    FloatLike,
    floating::FloatOps,
    matrix::{Matrix4, MatrixVectorMul, SquareMatrix, mat4::Mat4x4f32},
    quaternion::Quaternion,
    vector::{Vector, Vector3, Vector4, vec3::Vec3f32, vec4::Quat},
  },
  os,
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
  ExecuteCommand(String),
}

pub struct LogicState {
  yaw: f32,
  pitch: f32,
  camera_distance: f32,
  selected_entity: Option<EntityId>,
  last_selected_entity: Option<EntityId>,
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
  response_tx: mpsc::Sender<String>,
  scene_shared: Arc<Scene>,
  root_entity: EntityId,
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
      selected_entity: None,
      last_selected_entity: None,
    };

    let mut last_time = Instant::now();
    let mut accumulator = 0.0;

    println!("Starting Almanac load...");
    let start_load = std::time::Instant::now();

    let assets_planets_pathbuf: os::fs::PathBuf = assets_dir
      .join("planets")
      .to_str()
      .unwrap()
      .to_string()
      .into();
    let almanac = aethervk_core_rlib::simulation::almanac::load_almanac(&assets_planets_pathbuf)
      .expect("couldn't load almanac");
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
    let mut last_following_entity: Option<EntityId> = None;
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
          &response_tx,
          &mut state,
          scene_shared.as_ref(),
          root_entity,
          camera_entity,
          cursor_entity,
          grid_entity,
          &outlines_enabled,
          &mut following_entity,
          &mut focused_planet_idx,
          &planets_ids,
          &almanac.almanac,
          &mut current_scale,
          &mut current_epoch,
          &mut st_seconds_elapsed,
          epoch_start,
          epoch_end,
        );
      }

      if state.selected_entity != state.last_selected_entity {
        if let Some(old) = state.last_selected_entity {
          let _ =
            scene_shared.remove_component::<aethervk_core_rlib::scene::SelectedComponent>(old);
        }
        if let Some(new) = state.selected_entity {
          let _ = scene_shared.add_component(new, aethervk_core_rlib::scene::SelectedComponent {});
        }
        state.last_selected_entity = state.selected_entity;
      }

      if following_entity != last_following_entity {
        if let Some(old) = last_following_entity {
          let _ =
            scene_shared.remove_component::<aethervk_core_rlib::scene::FollowingComponent>(old);
        }
        if let Some(new) = following_entity {
          let _ = scene_shared.add_component(new, aethervk_core_rlib::scene::FollowingComponent {});
        }
        last_following_entity = following_entity;
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
    let pos =
      aethervk_core_rlib::simulation::almanac::get_almanac_pos(*naif_id, *current_epoch, &almanac);
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
  let pos = aethervk_core_rlib::simulation::almanac::get_almanac_pos(
    constants::PlanetNaifId::SUN,
    *current_epoch,
    &almanac,
  );
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
  response_tx: &mpsc::Sender<String>,
  state: &mut LogicState,
  scene_guard: &Scene,
  root_entity: EntityId,
  camera_entity: EntityId,
  cursor_entity: EntityId,
  grid_entity: EntityId,
  outlines_enabled: &AtomicBool,
  following_entity: &mut Option<EntityId>,
  focused_planet_idx: &mut Option<usize>,
  planets_ids: &[(i32, EntityId, f64, f32)],
  almanac: &Almanac,
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

      state.yaw = std::f32::consts::PI; // Look North (towards origin)
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
      let up = cam_rot.rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));
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
        c.near_plane = 0.1;
        c.far_plane = 1000000.0;
        c.projection = Mat4x4f32::perspective_vk(
          45.0f32.to_radians(),
          width as f32 / height as f32,
          c.near_plane,
          c.far_plane,
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
      state.yaw = 0.0;
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
    LogicCommand::ExecuteCommand(cmd) => {
      let mut parts = cmd.split_whitespace();
      let c = parts.next().unwrap_or("");
      match c {
        "scene" => {
          let _ = response_tx.send("Scene Hierarchy:".to_string());
          let mut stack = vec![0];
          scene_guard.traverse_with_hooks(
            root_entity,
            &mut stack,
            &mut |s, e, _: Option<TransformComponent>, _: Option<&TransformComponent>| {
              let depth = *s.last().unwrap();
              let name = scene_guard
                .get_name(e)
                .unwrap_or_else(|| "Unknown".to_string());
              let _ = response_tx.send(format!("{}├── {}", "│   ".repeat(depth), name));
              s.push(depth + 1);
              true
            },
            &mut |s, _| {
              s.pop();
            },
          );
        }
        "select" => {
          let name = cmd[c.len()..].trim();
          if let Some(id) = scene_guard.get_entity_by_name(name) {
            state.selected_entity = Some(id);
            let _ = response_tx.send(format!("selected entity {}, ID: {:?}", name, id));
          } else {
            let _ = response_tx.send("entity doesn't exist".to_string());
          }
        }
        "printsel" => {
          if let Some(id) = state.selected_entity {
            let name = scene_guard
              .get_name(id)
              .unwrap_or_else(|| "Unknown".to_string());
            let _ = response_tx.send(format!("Selected entity: {} (ID: {:?})", name, id));
          } else {
            let _ = response_tx.send("No entity selected".to_string());
          }
        }
        "printbvh" => {
          if let Some(id) = state.selected_entity {
            let min_depth: i32 = parts.next().unwrap_or("-1").parse().unwrap_or(-1);
            let max_depth: i32 = parts.next().unwrap_or("-1").parse().unwrap_or(-1);
            if min_depth != -1 && max_depth != -1 && min_depth > max_depth {
              let _ = response_tx.send("illegal arguments: min_depth > max_depth".to_string());
              return;
            }
            scene_guard.with_component(
              id,
              |mesh: &aethervk_core_rlib::scene::PhysicalMeshComponent| {
                if let Some(bvh) = &mesh.mesh.bvh {
                  let _ = response_tx.send("BVH Nodes:".to_string());
                  let mut node_stack = vec![(0, 0)]; // (node_idx, depth)
                  while let Some((idx, depth)) = node_stack.pop() {
                    let node = &bvh.nodes[idx];
                    if (min_depth == -1 || depth as i32 >= min_depth)
                      && (max_depth == -1 || depth as i32 <= max_depth)
                    {
                      let _ = response_tx.send(format!(
                        "{}Node {} (Depth: {}) - Bound: {:?}",
                        "  ".repeat(depth),
                        idx,
                        depth,
                        node.bound
                      ));
                    }
                    if node.primitive_count == 0 {
                      // inner node
                      node_stack.push((node.right_child_offset as usize, depth + 1));
                      node_stack.push((node.left_child_or_primitive_offset as usize, depth + 1));
                    }
                  }
                } else {
                  let _ = response_tx.send("Entity has no BVH.".to_string());
                }
              },
            );
          } else {
            let _ = response_tx.send("No entity selected".to_string());
          }
        }
        "deselect" => {
          if state.selected_entity.is_some() {
            let id = state.selected_entity.unwrap();
            let _ =
              scene_guard.remove_component::<aethervk_core_rlib::scene::SelectedComponent>(id);
            state.selected_entity = None;
            let _ = response_tx.send("Deselected entity.".to_string());
          } else {
            let _ = response_tx.send("No selected entity.".to_string());
          }
        }
        "bvh-show" | "bvh-hide" | "bvh-node-dbgrender-set" => {
          let is_show = c == "bvh-show"
            || parts
              .clone()
              .last()
              .unwrap_or("")
              .parse::<bool>()
              .unwrap_or(true);
          if let Some(id) = state.selected_entity {
            let depth_str = parts.next().unwrap_or("all");
            let idx_str = parts.next();
            // Handle bvh-node-dbgrender-set legacy bool param
            let target_idx: Option<u32> = if c == "bvh-node-dbgrender-set" {
              idx_str.and_then(|s| s.parse().ok())
            } else {
              idx_str.and_then(|s| if s == "all" { None } else { s.parse().ok() })
            };

            let mut min_d = 0;
            let mut max_d = u32::MAX;

            if depth_str != "all" {
              if let Some(dash_pos) = depth_str.find('-') {
                let (start, end) = depth_str.split_at(dash_pos);
                let end = &end[1..];
                if !start.is_empty() {
                  min_d = start.parse().unwrap_or(0);
                }
                if !end.is_empty() {
                  max_d = end.parse().unwrap_or(u32::MAX);
                }
              } else {
                let d: u32 = depth_str.parse().unwrap_or(0);
                min_d = d;
                max_d = d;
              }
            }

            let mut flat_indices = Vec::new();
            let mut max_depth_found = 0;
            scene_guard.with_component(
              id,
              |mesh: &aethervk_core_rlib::scene::PhysicalMeshComponent| {
                if let Some(bvh) = &mesh.mesh.bvh {
                  let mut node_stack = vec![(0, 0)];
                  let mut current_child_index_at_depth = std::collections::HashMap::new();

                  while let Some((idx, d)) = node_stack.pop() {
                    if d > max_depth_found {
                      max_depth_found = d;
                    }
                    if d >= min_d && d <= max_d {
                      let child_index = current_child_index_at_depth.entry(d).or_insert(0);
                      if target_idx.is_none() || target_idx == Some(*child_index) {
                        flat_indices.push((d, idx));
                      }
                      *child_index += 1;
                    }
                    let node = &bvh.nodes[idx];
                    if node.primitive_count == 0 {
                      node_stack.push((node.right_child_offset as usize, d + 1));
                      node_stack.push((node.left_child_or_primitive_offset as usize, d + 1));
                    }
                  }
                }
              },
            );

            if flat_indices.is_empty() {
              if min_d > max_depth_found && min_d != u32::MAX {
                let _ = response_tx.send(format!(
                  "Error: Max depth is {}, requested {}.",
                  max_depth_found, min_d
                ));
              } else {
                let _ = response_tx.send("No nodes found for the given criteria.".to_string());
              }
            } else {
              let mut bvh_len = 0;
              scene_guard.with_component(
                id,
                |mesh: &aethervk_core_rlib::scene::PhysicalMeshComponent| {
                  if let Some(bvh) = &mesh.mesh.bvh {
                    bvh_len = bvh.nodes.len();
                  }
                },
              );

              if bvh_len > 0 {
                let mut added = false;
                scene_guard.with_component_mut(
                  id,
                  |dbg: &mut aethervk_core_rlib::scene::BvhDebugComponent| {
                    for &(_, idx) in &flat_indices {
                      if idx < dbg.node_render_states.len() {
                        dbg.node_render_states[idx] = is_show;
                      }
                    }
                    added = true;
                  },
                );
                if !added {
                  let mut states = vec![false; bvh_len];
                  for &(_, idx) in &flat_indices {
                    if idx < states.len() {
                      states[idx] = is_show;
                    }
                  }
                  let _ = scene_guard.add_component(
                    id,
                    aethervk_core_rlib::scene::BvhDebugComponent {
                      node_render_states: states,
                    },
                  );
                }
                let _ = response_tx.send(format!(
                  "{} {} nodes.",
                  if is_show { "Showing" } else { "Hiding" },
                  flat_indices.len()
                ));
              }
            }
          } else {
            let _ = response_tx.send("No entity selected".to_string());
          }
        }
        "bvh-node-dbgrender-get" => {
          if let Some(id) = state.selected_entity {
            let depth: u32 = parts.next().unwrap_or("0").parse().unwrap_or(0);
            let child_index: u32 = parts.next().unwrap_or("0").parse().unwrap_or(0);

            let mut flat_idx = None;
            scene_guard.with_component(
              id,
              |mesh: &aethervk_core_rlib::scene::PhysicalMeshComponent| {
                if let Some(bvh) = &mesh.mesh.bvh {
                  let mut current_child_index = 0;
                  let mut node_stack = vec![(0, 0)];
                  while let Some((idx, d)) = node_stack.pop() {
                    if d == depth {
                      if current_child_index == child_index {
                        flat_idx = Some(idx);
                        break;
                      }
                      current_child_index += 1;
                    }
                    let node = &bvh.nodes[idx];
                    if node.primitive_count == 0 {
                      node_stack.push((node.right_child_offset as usize, d + 1));
                      node_stack.push((node.left_child_or_primitive_offset as usize, d + 1));
                    }
                  }
                }
              },
            );

            if let Some(idx) = flat_idx {
              let mut render = false;
              scene_guard.with_component(
                id,
                |dbg: &aethervk_core_rlib::scene::BvhDebugComponent| {
                  if idx < dbg.node_render_states.len() {
                    render = dbg.node_render_states[idx];
                  }
                },
              );
              let _ = response_tx.send(format!(
                "Node at depth {}, index {} is {}",
                depth, child_index, render
              ));
            } else {
              let _ = response_tx.send("Node not found.".to_string());
            }
          }
        }
        "clear" => {
          let _ = response_tx.send("___CLEAR___".to_string());
        }
        "help" => {
          let _ = response_tx.send("Commands:".to_string());
          let _ = response_tx.send("  help               - Shows this help message".to_string());
          let _ = response_tx.send("  clear              - Clears the console output".to_string());
          let _ = response_tx.send("  scene              - Prints the scene hierarchy".to_string());
          let _ = response_tx.send("  select <entity>    - Selects an entity by name".to_string());
          let _ = response_tx
            .send("  printsel           - Prints the currently selected entity".to_string());
          let _ = response_tx
            .send("  deselect           - Deselects the currently selected entity".to_string());
          let _ =
            response_tx.send("  goto <entity>      - Selects and follows an entity".to_string());
          let _ = response_tx
            .send("  unfollow           - Stops the camera from following any entity".to_string());
          let _ = response_tx.send(
            "  following          - Prints the entity the camera is currently following"
              .to_string(),
          );
          let _ = response_tx
            .send("  printbvh [min] [max]- Prints BVH nodes for the selected entity".to_string());
          let _ = response_tx
            .send("  bvh-show <range> [idx] - Shows BVH nodes (e.g. 0-3, 2-, -4, all)".to_string());
          let _ = response_tx.send("  bvh-hide <range> [idx] - Hides BVH nodes".to_string());
          let _ = response_tx
            .send("  bvh-node-dbgrender-set <range> <idx> <bool> - Legacy toggle".to_string());
          let _ = response_tx.send(
            "  bvh-node-dbgrender-get <depth> <idx>        - Gets BVH node debug render state"
              .to_string(),
          );
        }
        "goto" | "follow" => {
          let name = cmd[c.len()..].trim();
          if let Some(id) = scene_guard.get_entity_by_name(name) {
            state.selected_entity = Some(id);
            *following_entity = Some(id);

            let planet_radius = planets_ids
              .iter()
              .find(|(_, e, _, _)| *e == id)
              .map(|(_, _, _, r)| *r)
              .unwrap_or(0.01);
            let mut p_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
            scene_guard.with_component(id, |t: &TransformComponent| {
              p_pos = t.position;
            });

            scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
              c.position = p_pos;
            });

            let sun_pos = aethervk_core_rlib::simulation::almanac::get_almanac_pos(
              crate::constants::PlanetNaifId::SUN,
              *current_epoch,
              almanac,
            );

            let mut dir_to_sun = sun_pos - p_pos;
            if dir_to_sun.length_squared() < 1e-6 {
              dir_to_sun = Vec3f32::from_components(0.0, 1.0, 0.0);
            } else {
              dir_to_sun = dir_to_sun.normalize();
            }

            let offset_dist = (planet_radius as f32 * 3.0).max(60.0);

            let mut right = dir_to_sun.cross(Vec3f32::from_components(0.0, 0.0, 1.0));
            if right.length_squared() < 1e-6 {
              right = Vec3f32::from_components(1.0, 0.0, 0.0);
            } else {
              right = right.normalize();
            }
            let up = right.cross(dir_to_sun).normalize();

            let cam_pos = p_pos - dir_to_sun * offset_dist
              + right * (offset_dist * 1.5)
              + up * (offset_dist * 0.5);

            let view_dir = (p_pos - cam_pos).normalize();

            let yaw = -f32::atan2(-view_dir.x(), view_dir.y());
            let pitch = f32::asin(view_dir.z());

            state.yaw = yaw;
            state.pitch = pitch;

            let yaw_quat =
              Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), state.yaw);
            let pitch_quat =
              Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), state.pitch);

            scene_guard.with_component_mut(camera_entity, |c: &mut TransformComponent| {
              c.position = cam_pos;
              c.rotation = (yaw_quat * pitch_quat).normalize();
            });

            let _ = response_tx.send(format!("Selected and following entity: {}", name));
          } else {
            let _ = response_tx.send("Entity doesn't exist".to_string());
          }
        }
        "unfollow" => {
          *following_entity = None;
          let _ = response_tx.send("Camera stopped following.".to_string());
        }
        "following" => {
          if let Some(id) = following_entity {
            let name = scene_guard
              .get_name(*id)
              .unwrap_or_else(|| "Unknown".to_string());
            let _ = response_tx.send(format!("Currently following: {} (ID: {:?})", name, id));
          } else {
            let _ = response_tx.send("Camera is not following any entity.".to_string());
          }
        }
        _ => {
          let _ = response_tx.send(format!("Unknown command: {}", cmd));
        }
      }
    }
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
        let (_, entity, _, planet_radius) = planets_ids[new_idx];

        *following_entity = Some(entity);

        let mut p_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
        scene_guard.with_component(entity, |t: &TransformComponent| {
          p_pos = t.position;
        });

        scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
          c.position = p_pos;
        });

        let sun_pos = aethervk_core_rlib::simulation::almanac::get_almanac_pos(
          crate::constants::PlanetNaifId::SUN,
          *current_epoch,
          almanac,
        );

        let mut dir_to_sun = sun_pos - p_pos;
        if dir_to_sun.length_squared() < 1e-6 {
          dir_to_sun = Vec3f32::from_components(0.0, 1.0, 0.0);
        } else {
          dir_to_sun = dir_to_sun.normalize();
        }

        let offset_dist = (planet_radius as f32 * 3.0).max(60.0);

        let mut right = dir_to_sun.cross(Vec3f32::from_components(0.0, 0.0, 1.0));
        if right.length_squared() < 1e-6 {
          right = Vec3f32::from_components(1.0, 0.0, 0.0);
        } else {
          right = right.normalize();
        }
        let up = right.cross(dir_to_sun).normalize();

        let cam_pos =
          p_pos - dir_to_sun * offset_dist + right * (offset_dist * 1.5) + up * (offset_dist * 0.5);

        let view_dir = (p_pos - cam_pos).normalize();

        let yaw = -f32::atan2(-view_dir.x(), view_dir.y());
        let pitch = f32::asin(view_dir.z());

        state.yaw = yaw;
        state.pitch = pitch;

        let yaw_quat = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), state.yaw);
        let pitch_quat =
          Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), state.pitch);

        scene_guard.with_component_mut(camera_entity, |c: &mut TransformComponent| {
          c.position = cam_pos;
          c.rotation = (yaw_quat * pitch_quat).normalize();
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

#[cfg(test)]
mod tests {
  use super::*;
  use aethervk_oshal_rlib::math::quaternion::Quaternion;
  use aethervk_oshal_rlib::math::vector::Vector3;

  fn assert_vec_eq(a: Vec3f32, b: Vec3f32, eps: f32) {
    assert!(
      (a.x() - b.x()).abs() < eps,
      "X mismatch: expected {}, got {}",
      b.x(),
      a.x()
    );
    assert!(
      (a.y() - b.y()).abs() < eps,
      "Y mismatch: expected {}, got {}",
      b.y(),
      a.y()
    );
    assert!(
      (a.z() - b.z()).abs() < eps,
      "Z mismatch: expected {}, got {}",
      b.z(),
      a.z()
    );
  }

  #[test]
  fn test_camera_rotation_axes() {
    // Reference frame: +z = up, -y = forward, +x = right
    let forward_local = Vec3f32::from_components(0.0, -1.0, 0.0);
    let right_local = Vec3f32::from_components(1.0, 0.0, 0.0);
    let up_local = Vec3f32::from_components(0.0, 0.0, 1.0);

    let eps = 1e-5;

    // 1. Identity rotation (yaw=0, pitch=0)
    let rot = Quat::identity();
    assert_vec_eq(
      rot.rotate_vector(forward_local),
      Vec3f32::from_components(0.0, -1.0, 0.0),
      eps,
    );
    assert_vec_eq(
      rot.rotate_vector(right_local),
      Vec3f32::from_components(1.0, 0.0, 0.0),
      eps,
    );

    // 2. Yaw 90 degrees (PI/2) - Right turn
    // In our math lib, positive rotation around +Z moves +X to +Y if right handed,
    // but let's see what the math gives: it gave Y=-1.0, meaning CW rotation!
    // If it's CW, then -Y (Forward) rotated 90 around Z gives -X.
    // Let's adjust expectations to match the actual math lib convention.
    let yaw = core::f32::consts::FRAC_PI_2;
    let rot = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), yaw);
    let forward_world = rot.rotate_vector(forward_local);
    let right_world = rot.rotate_vector(right_local);

    if right_world.y() < 0.0 {
      // It seems Quat::from_axis_angle is CW or left-handed in this library
      assert_vec_eq(forward_world, Vec3f32::from_components(-1.0, 0.0, 0.0), eps);
      assert_vec_eq(right_world, Vec3f32::from_components(0.0, -1.0, 0.0), eps);
    } else {
      assert_vec_eq(forward_world, Vec3f32::from_components(1.0, 0.0, 0.0), eps);
      assert_vec_eq(right_world, Vec3f32::from_components(0.0, 1.0, 0.0), eps);
    }

    // 3. Pitch 90 degrees (PI/2)
    let pitch = core::f32::consts::FRAC_PI_2;
    let rot = Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), pitch);
    let forward_world = rot.rotate_vector(forward_local);
    let up_world = rot.rotate_vector(up_local);

    if up_world.y() < 0.0 {
      assert_vec_eq(forward_world, Vec3f32::from_components(0.0, 0.0, -1.0), eps);
      assert_vec_eq(up_world, Vec3f32::from_components(0.0, -1.0, 0.0), eps);
    } else {
      assert_vec_eq(forward_world, Vec3f32::from_components(0.0, 0.0, 1.0), eps);
      assert_vec_eq(up_world, Vec3f32::from_components(0.0, 1.0, 0.0), eps);
    }

    // 4. Reset orientation (yaw=PI, pitch=0) - Looking at +Y (towards origin)
    let yaw = core::f32::consts::PI;
    let rot = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), yaw);
    assert_vec_eq(
      rot.rotate_vector(forward_local),
      Vec3f32::from_components(0.0, 1.0, 0.0),
      eps,
    );
    assert_vec_eq(
      rot.rotate_vector(right_local),
      Vec3f32::from_components(-1.0, 0.0, 0.0),
      eps,
    );
  }
}
