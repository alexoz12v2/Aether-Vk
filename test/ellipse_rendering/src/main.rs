use aethervk_core_rlib::scene::trajectory::TrajectoryComponent;
use aethervk_core_rlib::simulation_api::components_api::CameraParams;
use aethervk_core_rlib::simulation_api::SimulationContext;
use aethervk_core_rlib::types::GpuResult;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::{Vector3, vec3::Vec3f32, vec4::Quat};
use test_utils::cycle_get_asset_path_from_exe;
use test_utils::sim_app::{run_simulation_app, SimulationDelegate};
use winit::window::Window;

fn fetch_earth_orbit() -> (f32, f32, f32, f32, f32) {
  // a, e, i, Omega, omega
  let url = "https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='399'&CENTER='@10'&OBJ_DATA='NO'&MAKE_EPHEM='YES'&EPHEM_TYPE='ELEMENTS'&START_TIME='2024-01-01'&STOP_TIME='2024-01-02'&STEP_SIZE='1d'";
  let client = reqwest::blocking::Client::new();
  let resp = client.get(url).send().expect("Failed to query Horizons API");
  let json: serde_json::Value = resp.json().expect("Failed to parse JSON");
  let result = json["result"].as_str().expect("No result field");

  let mut a = 1.496e8;
  let mut e = 0.0167;
  let mut i = 0.0;
  let mut omega_node = 0.0;
  let mut omega_peri = 0.0;

  if let Some(ec_idx) = result.find("EC=") {
    let part = &result[ec_idx + 3..];
    if let Some(val) = part.split_whitespace().next() {
      e = val.parse().unwrap_or(e);
    }
  }
  if let Some(a_idx) = result.find("A =") {
    let part = &result[a_idx + 3..];
    if let Some(val) = part.split_whitespace().next() {
      a = val.parse().unwrap_or(a);
    }
  }
  if let Some(in_idx) = result.find("IN=") {
    let part = &result[in_idx + 3..];
    if let Some(val) = part.split_whitespace().next() {
      i = val.parse().unwrap_or(i);
    }
  }
  if let Some(om_idx) = result.find("OM=") {
    let part = &result[om_idx + 3..];
    if let Some(val) = part.split_whitespace().next() {
      omega_node = val.parse().unwrap_or(omega_node);
    }
  }
  if let Some(w_idx) = result.find("W =") {
    let part = &result[w_idx + 3..];
    if let Some(val) = part.split_whitespace().next() {
      omega_peri = val.parse().unwrap_or(omega_peri);
    }
  }

  (a, e, i, omega_node, omega_peri)
}

fn build_ellipse_trajectory(
  a: f32,
  e: f32,
  i: f32,
  omega_node: f32,
  omega_peri: f32,
) -> TrajectoryComponent {
  let rot = Quat::from_axis_angle(
    Vec3f32::from_components(0.0, 0.0, 1.0),
    omega_node.to_radians(),
  ) * Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), i.to_radians())
    * Quat::from_axis_angle(
      Vec3f32::from_components(0.0, 0.0, 1.0),
      omega_peri.to_radians(),
    );

  let transform_point = |x: f32, y: f32| -> [f32; 4] {
    let b = a * f32::sqrt(1.0 - e * e);
    let peri_x = a * x - a * e;
    let peri_y = b * y;

    let p = Vec3f32::from_components(peri_x, peri_y, 0.0);
    let p_rot = rot.rotate_vector(p);

    [p_rot.x(), p_rot.y(), p_rot.z(), 1.0]
  };

  let mut control_points = Vec::new();
  let k = 4.0 / 3.0 * (f32::sqrt(2.0) - 1.0);

  // Arc 1 (0 to 90 deg)
  control_points.push(transform_point(1.0, 0.0));
  control_points.push(transform_point(1.0, k));
  control_points.push(transform_point(k, 1.0));
  control_points.push(transform_point(0.0, 1.0));

  // Arc 2 (90 to 180 deg)
  control_points.push(transform_point(0.0, 1.0));
  control_points.push(transform_point(-k, 1.0));
  control_points.push(transform_point(-1.0, k));
  control_points.push(transform_point(-1.0, 0.0));

  // Arc 3 (180 to 270 deg)
  control_points.push(transform_point(-1.0, 0.0));
  control_points.push(transform_point(-1.0, -k));
  control_points.push(transform_point(-k, -1.0));
  control_points.push(transform_point(0.0, -1.0));

  // Arc 4 (270 to 360 deg)
  control_points.push(transform_point(0.0, -1.0));
  control_points.push(transform_point(k, -1.0));
  control_points.push(transform_point(1.0, -k));
  control_points.push(transform_point(1.0, 0.0));

  TrajectoryComponent::new(control_points, [0.0, 1.0, 0.0, 1.0], 5.0, 0, 64)
}

struct EllipseDelegate;

impl SimulationDelegate for EllipseDelegate {
  fn on_setup(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    window: &Window,
  ) -> GpuResult<()> {
    // Root entity
    let root_entity = ctx.spawn_entity(scene_id, "root").unwrap();
    ctx.add_transform_component(
      scene_id,
      root_entity,
      Vec3f32::zero(),
      Quat::identity(),
      Vec3f32::one(),
    )
    .unwrap();

    let (a, e, i, omega_node, omega_peri) = fetch_earth_orbit();
    println!(
      "Earth orbit: a={:.2e} km, e={:.4}, i={:.2}°, Omega={:.2}°, w={:.2}°",
      a, e, i, omega_node, omega_peri
    );

    let visual_scale = 1.0 / 1e7;
    let visual_a = a * visual_scale;

    let traj_comp = build_ellipse_trajectory(visual_a, e, i, omega_node, omega_peri);

    let traj_entity = ctx.spawn_entity(scene_id, "earth_orbit").unwrap();
    ctx.set_parent(scene_id, traj_entity, Some(root_entity)).unwrap();
    ctx.add_transform_component(
      scene_id,
      traj_entity,
      Vec3f32::zero(),
      Quat::identity(),
      Vec3f32::one(),
    )
    .unwrap();
    ctx.add_trajectory_component(scene_id, traj_entity, traj_comp).unwrap();

    let cam_entity = ctx.add_perspective_camera(
      scene_id,
      pe_handle,
      "camera",
      45.0f32.to_radians(),
      0.1,
      1000.0,
    ).unwrap().get();
    ctx.set_parent(scene_id, cam_entity, Some(root_entity)).unwrap();

    let cam_pos = Vec3f32::from_components(0.0, visual_a * 1.5, visual_a * 1.5);
    let target = Vec3f32::zero();
    let up = Vec3f32::from_components(0.0, 0.0, 1.0);
    let view_dir = (target - cam_pos).normalized();
    let right = view_dir.cross(up).normalized();
    let actual_up = right.cross(view_dir).normalized();

    // Rotate camera to look at target
    let rot = Quat::look_at(view_dir, actual_up);

    ctx.set_transform_component(
      scene_id,
      cam_entity,
      cam_pos,
      rot,
      Vec3f32::one(),
    )
    .unwrap();

    let _ = ctx.threads.logic_thread.tx().try_send(
      aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id },
    );

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
    if let Some(scene) = ctx.get_scene(scene_id) {
      if let Some(camera_entity) = scene.read().active_camera_entity {
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
  }
}

fn main() {
  let _assets_dir = cycle_get_asset_path_from_exe(true);
  run_simulation_app("Ellipse Rendering Playground", EllipseDelegate);
}
          shift_down,
          ctrl_down,
          camera_entity,
          scene.clone(),
        ) {
          let _ = ctx.threads.logic_thread.tx().try_send(logic_command);
        }
      }
    }
  }
}

fn main() {
  let _assets_dir = cycle_get_asset_path_from_exe(true);
  run_simulation_app("Ellipse Rendering Playground", EllipseDelegate);
}