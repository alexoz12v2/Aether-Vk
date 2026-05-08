use aethervk_core_rlib::gpu;
use aethervk_core_rlib::simulation_api::components_api::CameraParams;
use aethervk_core_rlib::scene::trajectory::TrajectoryComponent;
use aethervk_core_rlib::simulation_api::SimulationContext;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::{Vector3, vec3::Vec3f32, vec4::Quat};
use std::sync::Arc;
use test_utils::{
  create_winit_window_and_event_loop, cycle_get_asset_path_from_exe, get_handle_and_window_info,
};
use test_utils::sim_app::{CustomRenderData, SimApp};

fn panic_error_callback(msg: &str) {
  panic!("Vulkan Error: {}", msg);
}

fn fetch_earth_orbit() -> (f32, f32, f32, f32, f32) { // a, e, i, Omega, omega
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
      if let Some(val) = part.split_whitespace().next() { e = val.parse().unwrap_or(e); }
  }
  if let Some(a_idx) = result.find("A =") {
      let part = &result[a_idx + 3..];
      if let Some(val) = part.split_whitespace().next() { a = val.parse().unwrap_or(a); }
  }
  if let Some(in_idx) = result.find("IN=") {
      let part = &result[in_idx + 3..];
      if let Some(val) = part.split_whitespace().next() { i = val.parse().unwrap_or(i); }
  }
  if let Some(om_idx) = result.find("OM=") {
      let part = &result[om_idx + 3..];
      if let Some(val) = part.split_whitespace().next() { omega_node = val.parse().unwrap_or(omega_node); }
  }
  if let Some(w_idx) = result.find("W =") {
      let part = &result[w_idx + 3..];
      if let Some(val) = part.split_whitespace().next() { omega_peri = val.parse().unwrap_or(omega_peri); }
  }

  (a, e, i, omega_node, omega_peri)
}

fn build_ellipse_trajectory(a: f32, e: f32, i: f32, omega_node: f32, omega_peri: f32) -> TrajectoryComponent {
  let rot = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), omega_node.to_radians()) *
            Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), i.to_radians()) *
            Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), omega_peri.to_radians());

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

fn main() {
  let _assets_dir = cycle_get_asset_path_from_exe(true);
  
  let (window, event_loop) = create_winit_window_and_event_loop("Ellipse Rendering Playground");

  let simulation_context =
    SimulationContext::startup(gpu::VULKAN_RENDER_BACKEND, Some(panic_error_callback)).unwrap();

  let (native_handles, window_info) = {
    let render_frontend = simulation_context.render_frontend().unwrap();
    let render_device_handle = simulation_context.render_device_handle();
    get_handle_and_window_info(&render_frontend, render_device_handle, &window)
  };

  let width = window.inner_size().width;
  let height = window.inner_size().height;

  let scene_id = simulation_context.create_empty_scene().unwrap();

  let presentation_engine = simulation_context
    .create_presentation_engine_windowed(scene_id, width, height, native_handles)
    .unwrap();

  // Root entity
  let root_entity = simulation_context.spawn_entity(scene_id, "root").unwrap();
  simulation_context.add_transform_component(
      scene_id, root_entity, Vec3f32::from_components(0.0, 0.0, 0.0), Quat::identity(), Vec3f32::from_components(1.0, 1.0, 1.0)
  ).unwrap();

  // Camera
  let camera_entity = simulation_context.spawn_entity(scene_id, "camera").unwrap();
  simulation_context.add_transform_component(
      scene_id, camera_entity, Vec3f32::from_components(0.0, -40.0, 10.0), Quat::identity(), Vec3f32::from_components(1.0, 1.0, 1.0)
  ).unwrap();
  simulation_context.add_camera_component(
      scene_id, camera_entity, CameraParams::new_perspective(60.0f32.to_radians(), width as f32 / height as f32, 0.1, 1000.0)
  ).unwrap();
  simulation_context.set_active_camera(scene_id, camera_entity).unwrap();

  // Cursor for orbital camera center
  let cursor_entity = simulation_context.spawn_entity(scene_id, "cursor").unwrap();
  simulation_context.add_transform_component(
      scene_id, cursor_entity, Vec3f32::from_components(0.0, 0.0, 0.0), Quat::identity(), Vec3f32::from_components(1.0, 1.0, 1.0)
  ).unwrap();
  simulation_context.add_cursor_component(scene_id, cursor_entity).unwrap();
  {
      let scene_ctx = simulation_context.get_scene(scene_id).unwrap();
      let mut scene_write = scene_ctx.write();
      let internal_id = scene_write.get_entity(cursor_entity).unwrap();
      scene_write.cursor_entity = Some(internal_id);
  }
  
  // Grid
  let grid_entity = simulation_context.spawn_entity(scene_id, "grid").unwrap();
  simulation_context.add_grid_component(scene_id, grid_entity).unwrap();

  // Trajectory
  let (a, e, i, omega_node, omega_peri) = fetch_earth_orbit();
  // Scale down the semi-major axis by 10 million to fit the view
  let scaled_a = a / 10_000_000.0;
  
  let trajectory_entity = simulation_context.spawn_entity(scene_id, "orbit").unwrap();
  let traj_comp = build_ellipse_trajectory(scaled_a, e, i, omega_node, omega_peri);
  
  {
      let scene_ctx = simulation_context.get_scene(scene_id).unwrap();
      let mut scene_write = scene_ctx.write();
      let internal_id = scene_write.get_entity(trajectory_entity).unwrap();
      scene_write.scene.add_component(internal_id, traj_comp).unwrap();
  }

  let custom_render_data = CustomRenderData::default();
  let custom_data = Arc::new(std::sync::Mutex::new(custom_render_data));

  let sim_app = SimApp::new(
    simulation_context,
    custom_data,
    scene_id,
    presentation_engine,
    camera_entity,
    window,
    window_info
  );

  let _ = sim_app.app_state.ctx.threads.logic_thread.tx().try_send(aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id });

  test_utils::app::run_app(sim_app, event_loop);
}