//! ffi module.

use aethervk_core_rlib::{
  gpu,
  math::collision::{linear_bvh, linear_bvh::LinearBVHNode},
  scene::Marker,
  simulation_api::{
    comet_api::CometApi,
    components_api::{CameraParams, OrthographicCameraParams, PerspectiveCameraParams},
    structs::*,
    *,
  },
  types::EngineError,
};
use aethervk_oshal_rlib as oshal;
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{Vector3, Vector4, vec3::Vec3f32, vec4::Quat},
};
use alloc::{boxed::Box, string::ToString};
use core::{
  ffi::{CStr, c_char},
  str::FromStr,
};

pub type PanicCallback = extern "C" fn(*const u8, usize);
pub static mut PANIC_CALLBACK: Option<PanicCallback> = None;

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_registerPanicCallback(cb: PanicCallback) {
  PANIC_CALLBACK = Some(cb);
}

// -------------------- C Exposed API (Async & Stateless) ----------------------------

fn backend_id_from_str(backend_std: &str) -> Option<gpu::RenderBackendId> {
  const VULKAN_BACKEND_STR: &str = "Vulkan";
  match backend_std {
    VULKAN_BACKEND_STR => Some(gpu::VULKAN_RENDER_BACKEND),
    _ => None,
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_startup(
  backend: *const c_char,
) -> *mut SimulationContext {
  let backend_str = if backend.is_null() {
    ""
  } else {
    unsafe { CStr::from_ptr(backend).to_str().unwrap_or("") }
  };
  if let Some(backend) = backend_id_from_str(backend_str) {
    SimulationContext::startup(backend, None).map(|boxed| Box::into_raw(boxed)).unwrap_or_else(
      |e| {
        oshal::log!("avkSimulationContext_startup failed: {}", e.to_string());
        emit_breadcrumb(1, &alloc::format!("Startup failed: {}", e.to_string()));
        core::ptr::null_mut()
      },
    )
  } else {
    oshal::log!("Unsupported backend: {}", backend_str);
    core::ptr::null_mut()
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_shutdown(ctx: *mut SimulationContext) {
  if !ctx.is_null() {
    let ctx_box = unsafe { Box::from_raw(ctx) };
    let _ = ctx_box
      .threads
      .logic_thread
      .tx()
      .try_send(aethervk_core_rlib::simulation_api::structs::LogicCommand::Shutdown);
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTaskStatus(
  ctx: *mut SimulationContext,
  task_id: u64,
) -> i32 {
  if ctx.is_null() {
    return -1;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_task_status(task_id) as i32
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTaskResultU64(
  ctx: *mut SimulationContext,
  task_id: u64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_task_result_u64(task_id)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTaskResultBool(
  ctx: *mut SimulationContext,
  task_id: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_task_result_bool(task_id)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTaskResultRaycast(
  ctx: *mut SimulationContext,
  task_id: u64,
  out_hit: *mut FfiRaycastResult,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_task_result_raycast(task_id, out_hit)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTaskResultKinematicState(
  ctx: *mut SimulationContext,
  task_id: u64,
  out_state: *mut FfiKinematicState,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_task_result_kinematic_state(task_id, out_state)
}

// --- Scene Management ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_destroyScene(
  ctx: *mut SimulationContext,
  scene_id: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.destroy_scene(scene_id);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_createEmptyScene(ctx: *mut SimulationContext) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.create_empty_scene().unwrap_or_else(|e| {
    oshal::log!("create_empty_scene failed: {}", e);
    0
  })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addPerspectiveCamera(
  ctx: *mut SimulationContext,
  scene_id: u64,
  presentation_engine: u64,
  name: *const c_char,
  fov: f32,
  near: f32,
  far: f32,
) -> u64 {
  if ctx.is_null() || name.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  unsafe { core::ffi::CStr::from_ptr(name) }
    .to_str()
    .map_err(|_| EngineError::InvalidNullArgument)
    .and_then(|name_str| {
      ctx_ref
        .add_perspective_camera(
          scene_id,
          gpu::PresentationEngineHandle(presentation_engine),
          name_str,
          fov.to_radians(),
          near,
          far,
        )
        .map(|id| id.get())
    })
    .unwrap_or_else(|e| {
      oshal::log!("add_perspective_camera failed: {}", e);
      0
    })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setCameraForPresentationEngine(
  ctx: *mut SimulationContext,
  scene_id: u64,
  presentation_engine: u64,
  camera_entity_id: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .set_camera_for_presentation_engine(
      scene_id,
      gpu::PresentationEngineHandle(presentation_engine),
      camera_entity_id,
    )
    .map(|_| true)
    .unwrap_or_else(|e| {
      oshal::log!("set_camera_for_presentation_engine failed: {}", e);
      false
    })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addOrthographicCamera(
  ctx: *mut SimulationContext,
  scene_id: u64,
  presentation_engine: u64,
  name: *const c_char,
  left: f32,
  bottom: f32,
  near: f32,
  far: f32,
) -> u64 {
  if ctx.is_null() || name.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  unsafe { core::ffi::CStr::from_ptr(name) }
    .to_str()
    .map_err(|_| EngineError::InvalidNullArgument)
    .and_then(|name_str| {
      ctx_ref
        .add_orthographic_camera(
          scene_id,
          gpu::PresentationEngineHandle(presentation_engine),
          name_str,
          left,
          bottom,
          near,
          far,
        )
        .map(|id| id.get())
    })
    .unwrap_or_else(|e| {
      oshal::log!("add_orthographic_camera failed: {}", e);
      0
    })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_createPresentationEngine(
  ctx: *mut SimulationContext,
  width: u32,
  height: u32,
  scene_id: u64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.create_presentation_engine(scene_id, width, height).map(|h| h.0).unwrap_or(0)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_destroyPresentationEngine(
  ctx: *mut SimulationContext,
  scene_id: u64,
  handle: u64,
) {
  if ctx.is_null() || handle == 0 {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.destroy_presentation_engine(scene_id, gpu::PresentationEngineHandle(handle));
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_resize(
  ctx: *mut SimulationContext,
  scene_id: u64,
  handle: u64,
  width: u32,
  height: u32,
) {
  if ctx.is_null() || handle == 0 {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  let _ = ctx_ref.resize(
    scene_id,
    gpu::PresentationEngineHandle(handle),
    width,
    height,
  );
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_createDefaultScene(
  ctx: *mut SimulationContext,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  match ctx_ref.create_default_scene() {
    Ok(id) => {
      oshal::log!("create_default_scene SUCCESS: {}", id);
      id
    }
    Err(e) => {
      oshal::log!("create_default_scene failed: {}", e);
      0
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setSceneDebugName(
  ctx: *mut SimulationContext,
  scene_id: u64,
  name: *const c_char,
) {
  if ctx.is_null() || name.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let name_str = unsafe { core::ffi::CStr::from_ptr(name).to_str().unwrap_or("") };
  if let Some(scene) = ctx_ref.get_scene(scene_id) {
    scene.write().debug_name = alloc::string::String::from(name_str);
  }
}

// --- Entity Management (Async) ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnEntity(
  ctx: *mut SimulationContext,
  scene_id: u64,
  name: *const c_char,
) -> u64 {
  if ctx.is_null() || name.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let name_str = unsafe { CStr::from_ptr(name).to_str().unwrap_or("Entity") };
  ctx_ref.spawn_entity(scene_id, name_str).unwrap_or_else(|e| {
    oshal::log!("spawn_entity failed: {}", e);
    0
  })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_removeEntity(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.remove_entity(scene_id, entity).is_ok()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setParent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  parent: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.set_parent(scene_id, entity, parent).is_ok()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnBillboard(
  ctx: *mut SimulationContext,
  scene_id: u64,
  image_path: *const c_char,
  out_entity_id: *mut u64,
) -> bool {
  if ctx.is_null() || image_path.is_null() || out_entity_id.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let image_path_str = unsafe { CStr::from_ptr(image_path).to_str().unwrap_or("") };

  // Create a basic billboard entity with a BillboardComponent
  if let Ok(entity_id) = ctx_ref.spawn_entity(scene_id, "Billboard") {
    // We add a component to mark it as a billboard. The renderer should handle it.
    // For now we just add a TransformComponent so it exists in space.
    let _ = ctx_ref.add_transform_component(
      scene_id,
      entity_id,
      Vec3f32::from_components(0.0, 0.0, 0.0),
      Quat::from_components(1.0, 0.0, 0.0, 0.0),
      Vec3f32::from_components(1.0, 1.0, 1.0),
    );
    let _ = ctx_ref.add_image_billboard_component(
      scene_id, entity_id, true, // is_screen_space
      1.0,  // width
      1.0,  // height
    );
    unsafe {
      *out_entity_id = entity_id;
    }
    return true;
  }
  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnComet(
  ctx: *mut SimulationContext,
  scene_id: u64,
  model_id: u64,
  name: *const c_char,
  pos_x: f32,
  pos_y: f32,
  pos_z: f32,
  rot_w: f32,
  rot_x: f32,
  rot_y: f32,
  rot_z: f32,
  radius_km: f32,
  physics_type: u32,
  out_micro_id: *mut u64,
  out_comet_id: *mut u64,
) -> bool {
  if ctx.is_null() || name.is_null() || out_micro_id.is_null() || out_comet_id.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let name_str = unsafe { CStr::from_ptr(name).to_str().unwrap_or("Comet") };

  match ctx_ref.spawn_comet_internal(
    scene_id,
    model_id,
    name_str,
    Vec3f32::from_components(pos_x, pos_y, pos_z),
    Quat::from_components(rot_w, rot_x, rot_y, rot_z),
    radius_km,
    physics_type,
  ) {
    Ok((micro_id, comet_id)) => {
      unsafe {
        *out_micro_id = micro_id;
        *out_comet_id = comet_id;
      }
      true
    }
    Err(e) => {
      oshal::log!("spawn_comet_hierarchy failed: {}", e);
      false
    }
  }
}

// --- Components ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addTransformComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  pos_x: f32,
  pos_y: f32,
  pos_z: f32,
  rot_w: f32,
  rot_x: f32,
  rot_y: f32,
  rot_z: f32,
  scale_x: f32,
  scale_y: f32,
  scale_z: f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .add_transform_component(
      scene_id,
      entity,
      Vec3f32::from_components(pos_x, pos_y, pos_z),
      Quat::from_components(rot_x, rot_y, rot_z, rot_w),
      Vec3f32::from_components(scale_x, scale_y, scale_z),
    )
    .is_ok()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setTransformComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  transform: *const FfiTransform,
) -> bool {
  if ctx.is_null() || transform.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let t = unsafe { &*transform };
  if !t.px.is_finite()
    || !t.py.is_finite()
    || !t.pz.is_finite()
    || !t.rw.is_finite()
    || !t.rx.is_finite()
    || !t.ry.is_finite()
    || !t.rz.is_finite()
    || !t.sx.is_finite()
    || !t.sy.is_finite()
    || !t.sz.is_finite()
  {
    return false;
  }
  ctx_ref
    .set_transform_component(
      scene_id,
      entity,
      Vec3f32::from_components(t.px, t.py, t.pz),
      Quat::from_components(t.rx, t.ry, t.rz, t.rw),
      Vec3f32::from_components(t.sx, t.sy, t.sz),
    )
    .is_ok()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTransformComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  out_transform: *mut FfiTransform,
) -> bool {
  if ctx.is_null() || out_transform.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };

  let mut pos_x = 0.0;
  let mut pos_y = 0.0;
  let mut pos_z = 0.0;
  let mut rot_w = 0.0;
  let mut rot_x = 0.0;
  let mut rot_y = 0.0;
  let mut rot_z = 0.0;
  let mut scale_x = 0.0;
  let mut scale_y = 0.0;
  let mut scale_z = 0.0;

  if ctx_ref
    .get_transform_component(
      scene_id,
      entity,
      &mut pos_x,
      &mut pos_y,
      &mut pos_z,
      &mut rot_w,
      &mut rot_x,
      &mut rot_y,
      &mut rot_z,
      &mut scale_x,
      &mut scale_y,
      &mut scale_z,
    )
    .is_ok()
  {
    unsafe {
      *out_transform = FfiTransform {
        px: pos_x,
        py: pos_y,
        pz: pos_z,
        rw: rot_w,
        rx: rot_x,
        ry: rot_y,
        rz: rot_z,
        sx: scale_x,
        sy: scale_y,
        sz: scale_z,
      };
    }
    true
  } else {
    false
  }
}

#[repr(C)]
pub struct FfiEmissionCircle {
  pub latitude_rad: f32,
  pub longitude_rad: f32,
  pub circle_radius_frac: f32,
  pub mass: f32,
  pub color_r: f32,
  pub color_g: f32,
  pub color_b: f32,
  pub color_a: f32,
  pub particles_per_tick: u32,
  pub ttl: u64,
  pub mean_velocity: f32,
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setParticleEmitterCirclesComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  circles: *const FfiEmissionCircle,
  count: u32,
) -> bool {
  if ctx.is_null() || (count > 0 && circles.is_null()) {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };

  let slice = if count > 0 {
    unsafe { core::slice::from_raw_parts(circles, count as usize) }
  } else {
    &[]
  };

  let mut rust_circles = alloc::vec::Vec::with_capacity(slice.len());
  for c in slice {
    rust_circles.push(aethervk_core_rlib::scene::EmissionCircle {
      latitude_rad: c.latitude_rad,
      longitude_rad: c.longitude_rad,
      circle_radius_frac: c.circle_radius_frac,
      mass: c.mass,
      color: [c.color_r, c.color_g, c.color_b, c.color_a],

      cached_point: None,
      cached_normal: None,
      particles_per_tick: c.particles_per_tick,
      ttl: c.ttl,
      mean_velocity: c.mean_velocity,
    });
  }

  ctx_ref.set_particle_emitter_circles_component(scene_id, entity, rust_circles).is_ok()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_recalculateJetPoints(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
) -> bool {
  use aethervk_core_rlib::math::collision::intersection::intersect_ray_triangle;
  use aethervk_oshal_rlib::math::vector::{Vector, Vector3, vec3::Vec3f32};

  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  let scene_ctx_lock = match scenes.get_scene(scene_id) {
    Some(s) => s,
    None => return false,
  };
  let scene_ctx = scene_ctx_lock.read();
  let internal_id = match scene_ctx.entity_map.get(&entity_id).copied() {
    Some(id) => id,
    None => return false,
  };

  // We need to fetch both components, but Rust's borrow checker prevents mutably borrowing one while immutably borrowing another from the same struct directly if not split.
  // Actually, we can get them sequentially or use a BVH/Mesh clone if needed, but `mesh` is Arc<Comet> so we can just clone the Arc.
  let mut mesh_arc_opt = None;
  let _ = scene_ctx.scene.with_component(
    internal_id,
    |c: &aethervk_core_rlib::scene::PhysicalMeshComponent| {
      mesh_arc_opt = Some(c.mesh.clone());
    },
  );
  let mesh_arc = match mesh_arc_opt {
    Some(m) => m,
    None => return false,
  };

  let r = 10.0;

  let _ = scene_ctx.scene.with_component_mut(
    internal_id,
    |circles_comp: &mut aethervk_core_rlib::scene::ParticleEmitterCirclesComponent| {
      for circle in &mut circles_comp.circles {
        let lat = circle.latitude_rad;
        let lon = circle.longitude_rad;

        // Direction vector from center (spherical coordinates relative to +Z)
        // In our convention: Z is up. Latitude is from XY plane to Z. Longitude is around Z from X.
        let dir_z = lat.sin();
        let dir_x = lat.cos() * lon.cos();
        let dir_y = lat.cos() * lon.sin();
        let dir = Vec3f32::from_components(dir_x, dir_y, dir_z);

        // Cast a ray from far away towards the origin.
        let start_dist = r * 2.0;
        let origin = dir * start_dist;
        let ray_dir = dir * -1.0;
        let ray = aethervk_core_rlib::math::collision::intersection::Ray {
          origin,
          direction: ray_dir,
          length: start_dist * 2.0,
        };

        let mut closest_t = f32::MAX;
        let mut hit_normal = dir;

        // Brute force raycast against all triangles
        for tri in mesh_arc.iter_triangles() {
          if intersect_ray_triangle(&ray, &tri) {
            // Compute precise intersection point to get distance
            // Simplified: intersect_ray_triangle doesn't return t, so we just use the triangle center for approx, or we do a quick Möller–Trumbore here.
            // For simplicity and since intersect_ray_triangle is already returning bool, we can compute MT:
            let e1 = tri.vertices[1] - tri.vertices[0];
            let e2 = tri.vertices[2] - tri.vertices[0];
            let h = ray_dir.cross(e2);
            let a = e1.dot(h);
            if a.abs() > 1e-6 {
              let f = 1.0 / a;
              let s = origin - tri.vertices[0];
              let u = f * s.dot(h);
              if u >= 0.0 && u <= 1.0 {
                let q = s.cross(e1);
                let v = f * ray_dir.dot(q);
                if v >= 0.0 && u + v <= 1.0 {
                  let t = f * e2.dot(q);
                  if t > 0.0 && t < closest_t {
                    closest_t = t;
                    hit_normal = e1.cross(e2).normalize();
                  }
                }
              }
            }
          }
        }

        if closest_t < f32::MAX {
          let hit_pt = origin + ray_dir * closest_t;
          circle.cached_point = Some([hit_pt.x(), hit_pt.y(), hit_pt.z()]);
          circle.cached_normal = Some([hit_normal.x(), hit_normal.y(), hit_normal.z()]);
        } else {
          // Fallback to bounding sphere
          let hit_pt = dir * r;
          circle.cached_point = Some([hit_pt.x(), hit_pt.y(), hit_pt.z()]);
          circle.cached_normal = Some([dir.x(), dir.y(), dir.z()]);
        }
      }
    },
  );

  true
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getParticleEmitterCirclesComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  out_circles: *mut FfiEmissionCircle,
  max_count: u32,
  out_actual_count: *mut u32,
) -> bool {
  if ctx.is_null() || out_actual_count.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };

  if let Ok(circles) = ctx_ref.get_particle_emitter_circles_component(scene_id, entity) {
    let actual_count = circles.len() as u32;
    unsafe {
      *out_actual_count = actual_count;
    }

    if !out_circles.is_null() && max_count > 0 {
      let copy_count = core::cmp::min(max_count, actual_count) as usize;
      let out_slice = unsafe { core::slice::from_raw_parts_mut(out_circles, copy_count) };
      for i in 0..copy_count {
        let c = &circles[i];
        out_slice[i] = FfiEmissionCircle {
          latitude_rad: c.latitude_rad,
          longitude_rad: c.longitude_rad,
          circle_radius_frac: c.circle_radius_frac,
          mass: c.mass,
          color_r: c.color[0],
          color_g: c.color[1],
          color_b: c.color[2],
          color_a: c.color[3],
          particles_per_tick: c.particles_per_tick,
          ttl: c.ttl,
          mean_velocity: c.mean_velocity,
        };
      }
    }
    true
  } else {
    false
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setSphereGizmoVisibility(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  is_visible: bool,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  if let Some(mut scenes) = ctx_ref.scenes.try_write() {
    if let Some(scene_ctx_lock) = scenes.scenes.get(&scene_id) {
      let mut scene_ctx = scene_ctx_lock.write();
      if let Some(entity) = scene_ctx.entity_map.get(&entity).copied() {
        let mut found = false;
        let _ = scene_ctx.scene.with_component_mut(
          entity,
          |gizmo: &mut aethervk_core_rlib::scene::SphereGizmoComponent| {
            gizmo.is_visible = is_visible;
            found = true;
          },
        );
        if found {
          return true;
        }
      }
    }
  }
  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getSphereGizmoVisibility(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  out_is_visible: *mut bool,
) -> bool {
  if ctx.is_null() || out_is_visible.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  if let Some(scenes) = ctx_ref.scenes.try_read() {
    if let Some(scene_ctx_lock) = scenes.scenes.get(&scene_id) {
      let scene_ctx = scene_ctx_lock.read();
      if let Some(entity) = scene_ctx.entity_map.get(&entity).copied() {
        let mut found = false;
        let _ = scene_ctx.scene.with_component(
          entity,
          |gizmo: &aethervk_core_rlib::scene::SphereGizmoComponent| {
            unsafe {
              *out_is_visible = gizmo.is_visible;
            }
            found = true;
          },
        );
        if found {
          return true;
        }
      }
    }
  }
  false
}

// --- Queries ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityCount(
  ctx: *mut SimulationContext,
  scene_id: u64,
) -> u32 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref
    .get_entity_count(scene_id)
    .map(|c| {
      oshal::log!("getEntityCount for scene_id {} returned {}", scene_id, c);
      c
    })
    .unwrap_or_else(|e| {
      oshal::log!("getEntityCount failed for scene_id {}: {}", scene_id, e);
      0
    })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityIds(
  ctx: *mut SimulationContext,
  scene_id: u64,
  out_ids: *mut u64,
  max_count: u32,
) -> u32 {
  if ctx.is_null() || out_ids.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let ids = unsafe { core::slice::from_raw_parts_mut(out_ids, max_count as usize) };
  ctx_ref.get_entity_ids(scene_id, ids).map(|(n, _)| n).unwrap_or(0)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityName(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  out_name: *mut c_char,
  max_len: u32,
) -> bool {
  if ctx.is_null() || out_name.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let name = unsafe { core::slice::from_raw_parts_mut(out_name, max_len as usize) };
  ctx_ref.get_entity_name(scene_id, entity, name).is_ok()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityParent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_entity_parent(scene_id, entity).unwrap_or(0)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setEntityVisibility(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  visible: bool,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.set_entity_visibility(scene_id, entity, visible);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setEntityFollowing(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  following: bool,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.set_entity_following(scene_id, entity, following);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setEntitySelected(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  selected: bool,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.set_entity_selected(scene_id, entity, selected);
}

// --- Async Heavy Operations ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_importModel(
  ctx: *mut SimulationContext,
  path: *const c_char,
) -> u64 {
  if ctx.is_null() || path.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("").to_string() };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::ImportModel {
    task_id,
    path: path_str,
  });
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getImportedModelsCount(
  ctx: *mut SimulationContext,
) -> u32 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  scenes.model_registry.len() as u32
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
pub struct FfiMat3 {
  pub m00: f32,
  pub m10: f32,
  pub m20: f32,
  pub m01: f32,
  pub m11: f32,
  pub m21: f32,
  pub m02: f32,
  pub m12: f32,
  pub m22: f32,
}

impl Default for FfiMat3 {
  fn default() -> Self {
    Self {
      m00: 1.0,
      m10: 0.0,
      m20: 0.0,
      m01: 0.0,
      m11: 1.0,
      m21: 0.0,
      m02: 0.0,
      m12: 0.0,
      m22: 1.0,
    }
  }
}

impl FfiMat3 {
  pub fn from_mat3(mat: oshal::math::matrix::mat3::Mat3f32) -> Self {
    Self {
      m00: mat.x.x(),
      m10: mat.x.y(),
      m20: mat.x.z(),
      m01: mat.y.x(),
      m11: mat.y.y(),
      m21: mat.y.z(),
      m02: mat.z.x(),
      m12: mat.z.y(),
      m22: mat.z.z(),
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getModelLocalFrames(
  ctx: *mut SimulationContext,
  model_id: u64,
  out_user_frame: *mut FfiMat3,
  out_sim_frame: *mut FfiMat3,
) -> bool {
  if ctx.is_null() || out_user_frame.is_null() || out_sim_frame.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };

  if let Some((user_frame, sim_frame)) = ctx_ref.get_model_local_frames(model_id) {
    unsafe {
      *out_user_frame = FfiMat3::from_mat3(user_frame);
      *out_sim_frame = FfiMat3::from_mat3(sim_frame);
    }
    return true;
  }
  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_overrideModelSpherical(
  ctx: *mut SimulationContext,
  model_id: u64,
  radius_km: f32,
  mass_kg: f32,
  user_frame: *const FfiMat3,
) -> bool {
  if ctx.is_null() || user_frame.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };

  // For a spherical or externally defined object, the principal axes are isotropic.
  // Therefore, the simulation frame aligns perfectly with the user's defined local frame.
  let uf = unsafe { &*user_frame };
  let mat = oshal::math::matrix::mat3::Mat3f32 {
    x: oshal::math::vector::vec3::Vec3f32::from_components(uf.m00, uf.m10, uf.m20),
    y: oshal::math::vector::vec3::Vec3f32::from_components(uf.m01, uf.m11, uf.m21),
    z: oshal::math::vector::vec3::Vec3f32::from_components(uf.m02, uf.m12, uf.m22),
  };

  ctx_ref.override_model_spherical(model_id, radius_km, mass_kg, mat)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getImportedModels(
  ctx: *mut SimulationContext,
  out_ids: *mut u64,
  out_paths: *mut *const c_char,
  capacity: u32,
) -> u32 {
  if ctx.is_null() || out_ids.is_null() || out_paths.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();

  let mut count = 0;
  for (id, path) in &scenes.model_registry {
    if count >= capacity {
      break;
    }
    unsafe {
      *out_ids.add(count as usize) = *id;
      let c_str = alloc::ffi::CString::new(path.clone()).unwrap();
      *out_paths.add(count as usize) = c_str.into_raw();
    }
    count += 1;
  }
  count
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_unloadModel(
  ctx: *mut SimulationContext,
  model_id: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.unload_model(model_id);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_loadAlmanacFile(
  ctx: *mut SimulationContext,
  path: *const c_char,
) -> u64 {
  if ctx.is_null() || path.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("").to_string() };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::LoadAlmanac {
    task_id,
    path: path_str,
  });
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_unloadAlmanacFile(
  ctx: *mut SimulationContext,
  path: *const c_char,
) -> u64 {
  if ctx.is_null() || path.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("").to_string() };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::UnloadAlmanac {
    task_id,
    path: path_str,
  });
  task_id
}

// TODO C# side: builder/helper functions for strings suppoerted by `anise::time::Epoch::from_str("2024-03-24 12:00:00 TDB")`
// alternative (nah): Epoch::from_gregorian_utc_at_midnight(2000, 1, 1). But string is more precise
// TODO C# side: update
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_parseEpochToTaiSec(
  epoch_raw: *const core::ffi::c_char,
  out_tai_sec: *mut f64,
) -> bool {
  if epoch_raw.is_null() || out_tai_sec.is_null() {
    return false;
  }
  let epoch_opt = unsafe { core::ffi::CStr::from_ptr(epoch_raw) }
    .to_str()
    .ok()
    .and_then(|epoch_str| anise::time::Epoch::from_str(epoch_str).ok());

  if let Some(epoch) = epoch_opt {
    unsafe { *out_tai_sec = epoch.to_tai_seconds() };
    true
  } else {
    false
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_loadCometSpk(
  ctx: *mut SimulationContext,
  spk_id: i32,
  epoch_raw: *const c_char,
) -> u64 {
  if ctx.is_null() || epoch_raw.is_null() {
    return 0;
  }
  let epoch_opt = unsafe { CStr::from_ptr(epoch_raw) }
    .to_str()
    .ok()
    .and_then(|epoch_str| anise::time::Epoch::from_str(epoch_str).ok());
  if epoch_opt.is_none() {
    // TODO log/breadcrumb?
    return 0;
  }
  let epoch = unsafe { epoch_opt.unwrap_unchecked() };
  let ctx_ref = unsafe { &*ctx };

  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::LoadCometSpk {
    task_id,
    spk_id,
    frame: anise::constants::frames::SUN_J2000,
    epoch,
  });
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEphemerisPosition(
  ctx: *mut SimulationContext,
  spk_id: i32,
  epoch_tai_sec: f64,
  out_pos: *mut FfiKinematicState,
) -> bool {
  if ctx.is_null() || out_pos.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let frame = anise::constants::frames::SUN_J2000;
  let epoch = anise::time::Epoch::from_tai_seconds(epoch_tai_sec);

  if let Ok(state) =
    ctx_ref.logic_state.read().almanac_data.get_ephem_full(spk_id, frame, epoch, true, false)
  {
    unsafe {
      *out_pos = FfiKinematicState::from(state);
    }
    true
  } else {
    false
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_updateTrajectoryForSpk(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  spk_id: i32,
  start_epoch_tai_sec: f64,
  end_epoch_tai_sec: f64,
  sample_step_days: f64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };

  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ =
    ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::UpdateTrajectoryForSpk {
      task_id,
      scene_id,
      entity_id,
      spk_id,
      start_epoch_tai_sec,
      end_epoch_tai_sec,
      sample_step_days,
    });
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnModelInstance(
  ctx: *mut SimulationContext,
  scene_id: u64,
  model_id: u64,
  name: *const c_char,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let name_str = if name.is_null() {
    "ModelInstance".to_string()
  } else {
    unsafe { CStr::from_ptr(name).to_str().unwrap_or("ModelInstance").to_string() }
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::SpawnModelInstance {
    task_id,
    scene_id,
    model_id,
    name: name_str,
  });
  task_id
}

/// Result type written to the `out_result` pointer by `avkSimulationContext_spawnComet`.
#[repr(C)]
pub struct FfiSpawnCometResult {
  /// External entity id of the LCA micro-frame parent entity.
  pub lca_frame_id: u64,
  /// External entity id of the comet mesh child entity.
  pub comet_entity_id: u64,
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
/// Synchronously spawns a static mesh entity hierarchy (LCA micro-frame + static mesh).
///
/// Returns `true` on success; `false` on any error. On success `*out_result` is
/// populated with both entity ids.
pub unsafe extern "C" fn avkSimulationContext_spawnStaticMesh(
  ctx: *mut SimulationContext,
  scene_id: u64,
  model_id: u64,
  entity_name: *const c_char,
  pos_x: f32,
  pos_y: f32,
  pos_z: f32,
  rot_w: f32,
  rot_x: f32,
  rot_y: f32,
  rot_z: f32,
  radius_km: f32,
  out_result: *mut FfiSpawnCometResult,
) -> bool {
  use aethervk_oshal_rlib::math::{
    quaternion::Quaternion,
    vector::{Vector3, vec3::Vec3f32, vec4::Quat},
  };

  if ctx.is_null() || entity_name.is_null() || out_result.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let name = unsafe { CStr::from_ptr(entity_name).to_str().unwrap_or("StaticMesh") };
  let pos = Vec3f32::from_components(pos_x, pos_y, pos_z);
  let rot = Quat::from_components(rot_w, rot_x, rot_y, rot_z);
  let scale = Vec3f32::from_components(radius_km, radius_km, radius_km);

  match ctx_ref.spawn_static_mesh_internal(scene_id, model_id, name, pos, rot, scale) {
    Ok((lca_id, mesh_id)) => {
      unsafe {
        (*out_result).lca_frame_id = lca_id;
        (*out_result).comet_entity_id = mesh_id;
      }
      true
    }
    Err(e) => {
      oshal::log!("avkSimulationContext_spawnStaticMesh failed: {:?}", e);
      false
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_raycastNdc(
  ctx: *mut SimulationContext,
  scene_id: u64,
  camera_id: u64,
  ndc_x: f32,
  ndc_y: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.raycast_ndc(scene_id, camera_id, ndc_x, ndc_y).map(|id| id.get()).unwrap_or(0)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnProceduralSphere(
  ctx: *mut SimulationContext,
  scene_id: u64,
  name: *const c_char,
  radius: f32,
  mass: f32,
) -> u64 {
  if ctx.is_null() || mass <= 0.0 || name.is_null() || radius <= 0.0 {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.spawn_procedural_sphere(scene_id, name, radius, mass).unwrap_or_else(|e| {
    oshal::log!("spawn_procedural_sphere failed: {}", e);
    0
  })
}

// --- Camera & Cursor Commands ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_rotateCamera(
  ctx: *mut SimulationContext,
  scene_id: u64,
  camera_entity: u64,
  delta_x: f32,
  delta_y: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let internal_entity = match scene.read().get_entity(camera_entity) {
    Some(e) => e,
    None => return 0,
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::RotateCamera(
    structs::RotateCamera {
      camera_entity: internal_entity,
      scene,
      delta_x,
      delta_y,
    },
  ));
  task_id
}

// TODO pass the task_id to the command so that logic thread can give feedback. apply this to all functions
// TODO: since this try_send functions are used in test/simulation_test and tests of simulation_api, refactor these commands into methods
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_zoomCamera(
  ctx: *mut SimulationContext,
  scene_id: u64,
  camera_entity: u64,
  amount: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let internal_entity = match scene.read().get_entity(camera_entity) {
    Some(e) => e,
    None => return 0,
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::ZoomCamera(
    structs::ZoomCamera {
      camera_entity: internal_entity,
      scene,
      amount,
    },
  ));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_resetCamera(
  ctx: *mut SimulationContext,
  scene_id: u64,
  camera_entity: u64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let internal_entity = match scene.read().get_entity(camera_entity) {
    Some(e) => e,
    None => return 0,
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::ResetCamera(
    structs::ResetCamera {
      camera_entity: internal_entity,
      scene,
    },
  ));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_panCamera(
  ctx: *mut SimulationContext,
  scene_id: u64,
  camera_entity: u64,
  delta_x: f32,
  delta_y: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let internal_entity = match scene.read().get_entity(camera_entity) {
    Some(e) => e,
    None => return 0,
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::PanCamera(
    structs::PanCamera {
      camera_entity: internal_entity,
      scene,
      delta_x,
      delta_y,
    },
  ));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_panCursor(
  ctx: *mut SimulationContext,
  scene_id: u64,
  delta_x: f32,
  delta_y: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::PanCursor(
    structs::PanCursor {
      scene,
      delta_x,
      delta_y,
    },
  ));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_moveCursor(
  ctx: *mut SimulationContext,
  scene_id: u64,
  delta_x: f32,
  delta_y: f32,
  delta_z: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  // TODO pass the task_id to the command so that logic thread can give feedback. apply this to all functions
  // TODO: since this try_send functions are used in test/simulation_test and tests of simulation_api, refactor these commands into methods
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::MoveCursor(
    structs::MoveCursor {
      scene,
      delta_x,
      delta_y,
      delta_z,
    },
  ));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_snapToEntity(
  ctx: *mut SimulationContext,
  scene_id: u64,
  snap_entity: u64,
  target_entity: u64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let internal_snap = match scene.read().get_entity(snap_entity) {
    Some(e) => e,
    None => return 0,
  };
  let internal_target = match scene.read().get_entity(target_entity) {
    Some(e) => e,
    None => return 0,
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::SnapToEntity(
    structs::SnapToEntity {
      snap_entity: internal_snap,
      target_entity: internal_target,
      scene,
    },
  ));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_followEntity(
  ctx: *mut SimulationContext,
  scene_id: u64,
  snap_entity: u64,
  target_entity: u64,
  unfollow_other: bool,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let internal_snap = match scene.read().get_entity(snap_entity) {
    Some(e) => e,
    None => return 0,
  };
  let internal_target = match scene.read().get_entity(target_entity) {
    Some(e) => e,
    None => return 0,
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::FollowEntity(
    structs::FollowEntity {
      snap_entity: internal_snap,
      entity_id: internal_target,
      scene,
      unfollow_other,
    },
  ));
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_unfollowEntity(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scene = match ctx_ref.get_scene(scene_id) {
    Some(s) => s,
    None => return 0,
  };
  let internal_entity = match scene.read().get_entity(entity_id) {
    Some(e) => e,
    None => return 0,
  };
  let task_id = ctx_ref.task_manager.write().create_task().get();
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::UnfollowEntity(
    structs::UnfollowEntity {
      entity_id: internal_entity,
      scene,
    },
  ));
  task_id
}

// --- Change Tracking ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getChangedEntityCount(
  ctx: *mut SimulationContext,
  scene_id: u64,
) -> u32 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx) = scenes.get(&scene_id) {
    let len = scene_ctx.read().changed_entities.read().len();
    return len as u32;
  }
  0
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getChangedEntityIds(
  ctx: *mut SimulationContext,
  scene_id: u64,
  out_ids: *mut u64,
  max_count: u32,
) -> u32 {
  if ctx.is_null() || out_ids.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx) = scenes.get(&scene_id) {
    let read_ctx = scene_ctx.read();
    let changed = read_ctx.changed_entities.read();
    let mut count = 0;
    for (id, _) in changed.iter() {
      if count >= max_count {
        break;
      }
      unsafe { *out_ids.add(count as usize) = *id };
      count += 1;
    }
    return count;
  }
  0
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getChangedComponentCount(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
) -> u32 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx) = scenes.get(&scene_id) {
    let read_ctx = scene_ctx.read();
    let changed = read_ctx.changed_entities.read();
    if let Some(components) = changed.get(&entity_id) {
      return components.len() as u32;
    }
  }
  0
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getChangedComponentNames(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_id: u64,
  out_names: *mut *const c_char,
  max_count: u32,
) -> u32 {
  if ctx.is_null() || out_names.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  let scenes = ctx_ref.scenes.read();
  if let Some(scene_ctx) = scenes.get(&scene_id) {
    let read_ctx = scene_ctx.read();
    let changed = read_ctx.changed_entities.read();
    if let Some(components) = changed.get(&entity_id) {
      let mut count = 0;
      for name in components.iter() {
        if count >= max_count {
          break;
        }
        // Safely pass strings to C#, assuming components names are static or leaked
        // For now, we return the internal pointer. C# must not free it.
        // Actually, name.as_ptr() is not guaranteed to be null-terminated if it's just String.
        // But since these are strings we insert, let's create a CString and leak it, or just copy it.
        // Better: use a pre-allocated buffer per FFI call or leak CString for now.
        // To be safe, we will assume C# copies the string immediately.
        if let Ok(c_str) = alloc::ffi::CString::new(alloc::format!("{}", name)) {
          unsafe { *out_names.add(count as usize) = c_str.into_raw() };
        }
        count += 1;
      }
      return count;
    }
  }
  0
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_freeComponentNames(
  names: *mut *const c_char,
  count: u32,
) {
  if names.is_null() {
    return;
  }
  for i in 0..count {
    let ptr = unsafe { *names.add(i as usize) };
    if !ptr.is_null() {
      let _ = unsafe { alloc::ffi::CString::from_raw(ptr as *mut c_char) };
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityComponentNames(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity_ext_id: u64,
  out_names: *mut *const c_char,
  max_count: u32,
) -> u32 {
  if ctx.is_null() || out_names.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };

  if let Some(scene_ctx) = ctx_ref.get_scene(scene_id) {
    let s = scene_ctx.read();
    if let Some(internal_entity) = s.get_entity(entity_ext_id) {
      // Get the archetype signature for the entity
      let names = s.scene.get_entity_component_names(internal_entity);

      let mut count = 0;
      for name in names {
        if count >= max_count {
          break;
        }
        // CString allocates, C# must call your existing avkSimulationContext_freeComponentNames
        if let Ok(c_str) = alloc::ffi::CString::new(name) {
          unsafe { *out_names.add(count as usize) = c_str.into_raw() };
        }
        count += 1;
      }
      return count;
    }
  }
  0
}

// --- Callbacks & Asset Path ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setAssetPath(path: *const c_char) {
  if path.is_null() {
    return;
  }

  if let Ok(path_str) = unsafe { core::ffi::CStr::from_ptr(path) }.to_str() {
    SimulationContext::set_asset_path(path_str)
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getBodyRadius(body_id: i32) -> f32 {
  if let Some(asset_dir) = aethervk_core_rlib::gpu::ASSET_DIR.read().as_ref() {
    let pck_path = alloc::format!("{}/planets/pck00011.tpc", asset_dir);
    if let Some(radii) = aethervk_core_rlib::simulation::pck::read_body_radii(&pck_path, body_id) {
      return (radii[0] / aethervk_core_rlib::simulation::almanac::DISTANCE_SCALE_FACTOR) as f32;
    }
  }
  0.0
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setLoggerCallback(
  cb: Option<extern "C" fn(*const c_char)>,
) {
  SimulationContext::set_logger_callback(cb)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setBreadcrumbCallback(
  cb: Option<extern "C" fn(u32, *const c_char)>,
) {
  SimulationContext::set_breadcrumb_callback(cb)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setSimulationCallback(
  cb: Option<extern "C" fn(u64, u64, u64, *const core::ffi::c_void)>,
) {
  SimulationContext::set_simulation_callback(cb)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setRenderCallback(
  cb: Option<extern "C" fn(u64, u64, u64)>,
) {
  SimulationContext::set_render_callback(cb)
}

// --- Tick & Time ---

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_playScene(
  ctx: *mut SimulationContext,
  scene_id: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::PlayScene { scene_id });
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_snapshotScene(
  ctx: *mut SimulationContext,
  scene_id: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ =
    ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::SnapshotScene { scene_id });
  true
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_restoreSnapshot(
  ctx: *mut SimulationContext,
  scene_id: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ =
    ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::RestoreSnapshot { scene_id });
  true
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_pauseScene(
  ctx: *mut SimulationContext,
  scene_id: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ =
    ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::PauseScene { scene_id });
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setTimeScale(
  ctx: *mut SimulationContext,
  scene_id: u64,
  scale: u32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let time_scale = match scale {
    1 => structs::TimeScale::OneDay,
    2 => structs::TimeScale::OneWeek,
    3 => structs::TimeScale::OneMonth,
    _ => structs::TimeScale::Stopped,
  };
  let _ = ctx_ref.threads.logic_thread.tx().try_send(structs::LogicCommand::SetSceneTimeScale {
    scene_id,
    scale: time_scale,
  });
}

/// the C# caller should be responsible for polling getTaskStatus and only calling download_image once the status is completed.
/// Given the architecture of the C# async API and the fact that Vulkan windowless downloads are placed in persistently mapped memory buffers, bouncing the download request through the RenderCommand channel was actually an anti-pattern. Here is why:
/// 1. Thread Safety: is_task_completed and read_windowless_download only acquire read/write locks (self.res.read() and pending_downloads.write()). They don't actually need to execute on the Render Thread.
/// 2. No Render Thread Blocking: By having C# call download_image synchronously after polling confirms it's ready, the memory copy happens on the caller thread (e.g., C#'s worker pool), freeing the Render Thread to continue pushing Vulkan commands without stalling
///   during the memory copy.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_downloadImage(
  ctx: *mut SimulationContext,
  task_id: u64,
  buffer_ptr: *mut u8,
  buffer_size: usize,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.download_image(task_id, buffer_ptr, buffer_size)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addSunComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  res_x: u32,
  res_y: u32,
  res_z: u32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.add_sun_component(scene_id, entity, (res_x, res_y, res_z), 0.6);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addCursorComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.add_cursor_component(scene_id, entity);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addSkyComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.add_sky_component(scene_id, entity);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addGridComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.add_grid_component(scene_id, entity);
}

/// fov is in radians
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addCameraComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  is_orthographic: bool,
  fov: f32,
  aspect: f32,
  near: f32,
  far: f32,
  ortho_left: f32,
  ortho_right: f32,
  ortho_bottom: f32,
  ortho_top: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let params = if is_orthographic {
    CameraParams::Orthographic(OrthographicCameraParams {
      left: ortho_left,
      right: ortho_right,
      bottom: ortho_bottom,
      top: ortho_top,
      near,
      far,
    })
  } else {
    CameraParams::Perspective(PerspectiveCameraParams {
      fov: fov.to_radians(),
      aspect_ratio: aspect,
      near_plane: near,
      far_plane: far,
    })
  };
  let _ = ctx_ref.add_camera_component(scene_id, entity, params);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setCameraComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  camera: *const FfiCamera,
) -> bool {
  if ctx.is_null() || camera.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  let c = unsafe { &*camera };
  let params = if c.is_orthographic {
    CameraParams::Orthographic(OrthographicCameraParams {
      left: c.ortho_left,
      right: c.ortho_right,
      bottom: c.ortho_bottom,
      top: c.ortho_top,
      near: c.near,
      far: c.far,
    })
  } else {
    CameraParams::Perspective(PerspectiveCameraParams {
      fov: c.fov.to_radians(),
      aspect_ratio: c.aspect,
      near_plane: c.near,
      far_plane: c.far,
    })
  };
  ctx_ref.set_camera_component(scene_id, entity, params).is_ok()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getCameraComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  out_camera: *mut FfiCamera,
) -> bool {
  if ctx.is_null() || out_camera.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &*ctx };
  if let Ok(params) = ctx_ref.get_camera_component(scene_id, entity) {
    let mut arr = [0.0; 16];
    // We can also retrieve the projection matrix separately if needed
    // However, we just need to return the struct for now.
    // Wait, let's actually just get the projection matrix manually via `get_projection_matrix()` if needed.
    // I can get the matrix using `with_component` here or just return an empty array if not requested.
    // Actually, `get_camera_component` returns `CameraParams`. The projection matrix is generated dynamically by the component.
    // To include the projection matrix we should probably query it. Let's do that.

    let _ = ctx_ref.get_scene(scene_id).map(|s| {
      if let Some(e) = s.read().get_entity(entity) {
        let _ =
          s.read().scene.with_component(e, |c: &aethervk_core_rlib::scene::CameraComponent| {
            arr = c.get_projection_matrix().into();
          });
      }
    });

    unsafe {
      *out_camera = match params {
        CameraParams::Perspective(p) => FfiCamera {
          is_orthographic: false,
          fov: p.fov.to_degrees(),
          aspect: p.aspect_ratio,
          near: p.near_plane,
          far: p.far_plane,
          ortho_left: 0.0,
          ortho_right: 0.0,
          ortho_bottom: 0.0,
          ortho_top: 0.0,
          proj: arr,
        },
        CameraParams::Orthographic(o) => FfiCamera {
          is_orthographic: true,
          fov: 45.0,   // Default or ignored
          aspect: 1.0, // Default or ignored
          near: o.near,
          far: o.far,
          ortho_left: o.left,
          ortho_right: o.right,
          ortho_bottom: o.bottom,
          ortho_top: o.top,
          proj: arr,
        },
      };
    }
    true
  } else {
    false
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addMeasurementComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  p1x: f32,
  p1y: f32,
  p1z: f32,
  p2x: f32,
  p2y: f32,
  p2z: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
  let _ = ctx_ref.add_measurement_component(
    scene_id,
    entity,
    Vec3f32::from_components(p1x, p1y, p1z),
    Vec3f32::from_components(p2x, p2y, p2z),
  );
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addImageBillboardComponent(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  is_screen_space: bool,
  width: f32,
  height: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.add_image_billboard_component(scene_id, entity, is_screen_space, width, height);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setMarkers(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  count: u32,
  px: *const f32,
  py: *const f32,
  pz: *const f32,
  cr: *const f32,
  cg: *const f32,
  cb: *const f32,
  sizes: *const f32,
) {
  if ctx.is_null()
    || px.is_null()
    || py.is_null()
    || pz.is_null()
    || cr.is_null()
    || cg.is_null()
    || cb.is_null()
    || sizes.is_null()
  {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  use alloc::vec::Vec;

  let mut markers = Vec::with_capacity(count as usize);
  for i in 0..(count as usize) {
    markers.push(FfiMarker {
      position: [unsafe { *px.add(i) }, unsafe { *py.add(i) }, unsafe {
        *pz.add(i)
      }],
      color: [unsafe { *cr.add(i) }, unsafe { *cg.add(i) }, unsafe {
        *cb.add(i)
      }],
      size: unsafe { *sizes.add(i) },
    });
  }

  let _ = ctx_ref.set_markers(scene_id, entity, &markers);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getBvhNodes(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  count: *mut u32,
) -> *mut FfiBvhNode {
  if ctx.is_null() || count.is_null() {
    return core::ptr::null_mut();
  }
  let ctx_ref = unsafe { &*ctx };
  ctx_ref.get_bvh_nodes(scene_id, entity, count).unwrap_or(core::ptr::null_mut())
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_freeBvhNodes(ptr: *mut FfiBvhNode, count: u32) {
  if !ptr.is_null() && count > 0 {
    let _ = unsafe { alloc::vec::Vec::from_raw_parts(ptr, count as usize, count as usize) };
  }
}

impl From<LinearBVHNode<f32>> for FfiBvhNode {
  fn from(node: LinearBVHNode<f32>) -> Self {
    let mut ffi_node = FfiBvhNode::from_offsets(
      node.left_child_or_primitive_offset,
      node.right_child_offset,
      node.primitive_count,
    );

    match &node.bound {
      linear_bvh::LinearBound::AABB(aabb) => {
        ffi_node.node_type = FfiNodeType::AABB;
        ffi_node.min_x = aabb.min::<Vec3f32>().x();
        ffi_node.min_y = aabb.min::<Vec3f32>().y();
        ffi_node.min_z = aabb.min::<Vec3f32>().z();
        ffi_node.max_x = aabb.max::<Vec3f32>().x();
        ffi_node.max_y = aabb.max::<Vec3f32>().y();
        ffi_node.max_z = aabb.max::<Vec3f32>().z();
      }
      linear_bvh::LinearBound::OBB(obb) => {
        ffi_node.node_type = FfiNodeType::OBB;
        let t: Vec3f32 = obb.translation();
        let ext: Vec3f32 = obb.half_extent();
        ffi_node.center_x = t.x();
        ffi_node.center_y = t.y();
        ffi_node.center_z = t.z();
        ffi_node.extents_x = ext.x();
        ffi_node.extents_y = ext.y();
        ffi_node.extents_z = ext.z();
      }
    }
    ffi_node
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setBvhNodeVisibility(
  ctx: *mut SimulationContext,
  scene_id: u64,
  entity: u64,
  node_index: u32,
  visible: bool,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &*ctx };
  let _ = ctx_ref.set_bvh_node_visibility(scene_id, entity, node_index, visible);
}

// --------------------- FFI Types ---------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfiTransform {
  pub px: f32,
  pub py: f32,
  pub pz: f32,
  pub rw: f32,
  pub rx: f32,
  pub ry: f32,
  pub rz: f32,
  pub sx: f32,
  pub sy: f32,
  pub sz: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfiCamera {
  pub is_orthographic: bool,
  pub fov: f32,
  pub aspect: f32,
  pub near: f32,
  pub far: f32,
  pub ortho_left: f32,
  pub ortho_right: f32,
  pub ortho_bottom: f32,
  pub ortho_top: f32,
  pub proj: [f32; 16],
}

#[repr(u32)]
/// TODO: Document this item
pub enum FfiRenderTaskStatus {
  Completed = 0,
  Pending = 1,
  Error = 2,
}

impl From<RenderTaskStatus> for FfiRenderTaskStatus {
  fn from(value: RenderTaskStatus) -> Self {
    match value {
      RenderTaskStatus::Completed => FfiRenderTaskStatus::Completed,
      RenderTaskStatus::Pending => FfiRenderTaskStatus::Pending,
      RenderTaskStatus::Error(_) => FfiRenderTaskStatus::Error,
    }
  }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
/// TODO: Document this item
pub struct FfiRaycastResult {
  pub hit: bool,
  pub entity: u64,
  pub px: f32,
  pub py: f32,
  pub pz: f32,
}

impl From<RaycastResult> for FfiRaycastResult {
  fn from(value: RaycastResult) -> Self {
    let mut res = Self {
      hit: false,
      entity: 0,
      px: 0.0,
      py: 0.0,
      pz: 0.0,
    };

    if let Some(hit) = value {
      res.hit = true;
      res.entity = hit.entity_ext_id;
      res.px = hit.p.x();
      res.py = hit.p.y();
      res.pz = hit.p.z();
    }

    res
  }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
/// TODO: Document this item
pub struct FfiKinematicState {
  pub pos_x: f32,
  pub pos_y: f32,
  pub pos_z: f32,
  pub vel_x: f32,
  pub vel_y: f32,
  pub vel_z: f32,
  pub has_rotation: bool,
  pub rot_w: f32,
  pub rot_x: f32,
  pub rot_y: f32,
  pub rot_z: f32,
  pub has_angular_velocity: bool,
  pub ang_vel_x: f32,
  pub ang_vel_y: f32,
  pub ang_vel_z: f32,
}

impl From<aethervk_core_rlib::simulation::almanac::KinematicState> for FfiKinematicState {
  fn from(value: aethervk_core_rlib::simulation::almanac::KinematicState) -> Self {
    let mut res = Self {
      pos_x: value.position.x(),
      pos_y: value.position.y(),
      pos_z: value.position.z(),
      vel_x: value.velocity.x(),
      vel_y: value.velocity.y(),
      vel_z: value.velocity.z(),
      has_rotation: false,
      rot_w: 1.0,
      rot_x: 0.0,
      rot_y: 0.0,
      rot_z: 0.0,
      has_angular_velocity: false,
      ang_vel_x: 0.0,
      ang_vel_y: 0.0,
      ang_vel_z: 0.0,
    };

    if let Some(rot) = value.rotation {
      res.has_rotation = true;
      res.rot_w = rot.0.w();
      res.rot_x = rot.0.x();
      res.rot_y = rot.0.y();
      res.rot_z = rot.0.z();
    }

    if let Some(ang_vel) = value.angular_velocity {
      res.has_angular_velocity = true;
      res.ang_vel_x = ang_vel.x();
      res.ang_vel_y = ang_vel.y();
      res.ang_vel_z = ang_vel.z();
    }

    res
  }
}

// TODO sync with C#
#[repr(u32)]
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
/// TODO: Document this item
pub enum FfiLogicCommandType {
  #[default]
  Shutdown = 0,

  RotateCamera = 1,
  ZoomCamera = 2,
  ResetCamera = 3,
  PanCamera = 4,

  PanCursor = 5,
  MoveCursor = 6,

  SnapToEntity = 7,
  FollowEntity = 8,
  UnfollowEntity = 9,

  FeedbackGetTimeScale = 10,
  FeedbackGetDateTimeUTC = 11,
  FeedbackGetDateTimeLimitsUTC = 12,
  // TODO: probably we'll need Ephemeris duration
}

// TODO modify in C#
#[repr(C, align(4))]
#[derive(Default, Clone, Copy, Debug)]
/// TODO: Document this item
pub struct FfiLogicCommand {
  pub cmd_type: FfiLogicCommandType,
  pub payload: [u8; 28],
}

impl FfiLogicCommand {
  /// TODO: Document this item
  pub fn get_u32_u64x3_at_start(&self) -> Option<(u32, u64, u64, u64)> {
    let first = self.get_u32_at_offset(0)?;
    let second = self.get_u64_at_offset(4)?;
    let third = self.get_u64_at_offset(12)?;
    let fourth = self.get_u64_at_offset(20)?;
    Some((first, second, third, fourth))
  }

  /// TODO: Document this item
  pub fn get_u64_f32x3_at_start(&self) -> Option<(u64, f32, f32, f32)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_f32_at_offset(12)?;
    let third = self.get_f32_at_offset(16)?;
    let fourth = self.get_f32_at_offset(20)?;
    Some((first, second, third, fourth))
  }

  /// TODO: Document this item
  pub fn get_u64_f32x2_at_start(&self) -> Option<(u64, f32, f32)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_f32_at_offset(12)?;
    let third = self.get_f32_at_offset(16)?;
    Some((first, second, third))
  }

  /// TODO: Document this item
  pub fn get_u64x2_f32_at_start(&self) -> Option<(u64, u64, f32)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_u64_at_offset(12)?;
    let third = self.get_f32_at_offset(20)?;
    Some((first, second, third))
  }

  /// TODO: Document this item
  pub fn get_u64x2_f32x2_at_start(&self) -> Option<(u64, u64, f32, f32)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_u64_at_offset(12)?;
    let third = self.get_f32_at_offset(20)?;
    let fourth = self.get_f32_at_offset(24)?;
    Some((first, second, third, fourth))
  }

  /// Utility for commands which take 2 u64 from start, aligned to 8 (eg 2 entities and a scene id)
  pub fn get_u64x2_at_start(&self) -> Option<(u64, u64)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_u64_at_offset(12)?;
    Some((first, second))
  }

  /// Utility for commands which take 3 u64 from start, aligned to 8 (eg 2 entities and a scene id)
  pub fn get_u64x3_at_start(&self) -> Option<(u64, u64, u64)> {
    let first = self.get_u64_at_offset(4)?;
    let second = self.get_u64_at_offset(12)?;
    let third = self.get_u64_at_offset(20)?;
    Some((first, second, third))
  }

  /// Utility for commands which takes 3 floats from start (eg delta_x and delta_y)
  pub fn get_f32x3_at_start(&self) -> Option<(f32, f32, f32)> {
    let first = self.get_f32_at_offset(0)?;
    let second = self.get_f32_at_offset(4)?;
    let third = self.get_f32_at_offset(8)?;
    Some((first, second, third))
  }

  /// Utility for commands which takes 2 floats from start (eg delta_x and delta_y)
  pub fn get_f32x2_at_start(&self) -> Option<(f32, f32)> {
    let first = self.get_f32_at_offset(0)?;
    let second = self.get_f32_at_offset(4)?;
    Some((first, second))
  }

  /// Safely reads a u32 from the payload.
  /// Payload starts at struct offset 4, so payload offset must be a multiple of 4.
  pub fn get_u32_at_offset(&self, offset: usize) -> Option<u32> {
    let size = core::mem::size_of::<u32>();

    // (offset + 4) % 4 == 0 simplifies down to offset % 4 == 0
    if offset % 4 != 0 || offset + size > self.payload.len() {
      return None;
    }

    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&self.payload[offset..offset + size]);
    Some(u32::from_ne_bytes(bytes))
  }

  /// Safely reads an f32 from the payload.
  /// Payload starts at struct offset 4, so payload offset must be a multiple of 4.
  pub fn get_f32_at_offset(&self, offset: usize) -> Option<f32> {
    let size = core::mem::size_of::<f32>();

    if offset % 4 != 0 || offset + size > self.payload.len() {
      return None;
    }

    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&self.payload[offset..offset + size]);
    Some(f32::from_ne_bytes(bytes))
  }

  /// Safely reads a u64 from the payload.
  /// Because the payload itself starts at struct offset 4,
  /// the absolute memory offset is (offset + 4).
  /// This absolute offset must be a multiple of 8.
  pub fn get_u64_at_offset(&self, offset: usize) -> Option<u64> {
    let size = core::mem::size_of::<u64>();

    // Ensure the *absolute* struct offset is 8-byte aligned.
    // Valid payload offsets for u64: 4, 12, 20.
    if (offset + 4) % 8 != 0 || offset + size > self.payload.len() {
      return None;
    }

    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&self.payload[offset..offset + size]);
    Some(u64::from_ne_bytes(bytes))
  }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)] // Optional: Keeps memory layout predictable if interfacing with C
/// TODO: Document this item
pub struct FfiMarker {
  pub position: [f32; 3], // [x, y, z]
  pub color: [f32; 3],    // [r, g, b]
  pub size: f32,
}

impl From<FfiMarker> for Marker {
  fn from(value: FfiMarker) -> Self {
    Self {
      local_pos: value.position,
      color: value.color,
      size: value.size,
    }
  }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// TODO: Document this item
pub enum FfiNodeType {
  AABB = 0,
  OBB = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
pub struct FfiBvhNode {
  pub node_type: FfiNodeType,
  pub min_x: f32,
  pub min_y: f32,
  pub min_z: f32,
  pub max_x: f32,
  pub max_y: f32,
  pub max_z: f32,
  pub center_x: f32,
  pub center_y: f32,
  pub center_z: f32,
  pub extents_x: f32,
  pub extents_y: f32,
  pub extents_z: f32,
  pub left_child: u32,
  pub right_child: u32,
  pub primitive_count: u32,
}

impl FfiBvhNode {
  /// TODO: Document this item
  pub fn from_offsets(left_child: u32, right_child: u32, primitive_count: u32) -> Self {
    Self {
      node_type: FfiNodeType::AABB,
      min_x: 0.0,
      min_y: 0.0,
      min_z: 0.0,
      max_x: 0.0,
      max_y: 0.0,
      max_z: 0.0,
      center_x: 0.0,
      center_y: 0.0,
      center_z: 0.0,
      extents_x: 0.0,
      extents_y: 0.0,
      extents_z: 0.0,
      left_child,
      right_child,
      primitive_count,
    }
  }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn avkSimulationContext_getSimulationTime(
  ctx: *mut SimulationContext,
  scene_id: u64,
) -> f64 {
  let ctx = unsafe { &*ctx };
  ctx.get_simulation_time(scene_id)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn avkSimulationContext_getSimulationTimeUtc(
  ctx: *mut SimulationContext,
  scene_id: u64,
  buffer: *mut core::ffi::c_char,
  buffer_len: u32,
) -> bool {
  let ctx = unsafe { &*ctx };
  ctx.get_simulation_time_utc(scene_id, buffer, buffer_len)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn avkSimulationContext_setSimulationTime(
  ctx: *mut SimulationContext,
  scene_id: u64,
  time_tai: f64,
) {
  let ctx = unsafe { &*ctx };
  ctx.set_simulation_time(scene_id, time_tai);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn avkSimulationContext_getEpochLimits(
  ctx: *mut SimulationContext,
  scene_id: u64,
  start_tai: *mut f64,
  end_tai: *mut f64,
) -> bool {
  let ctx = unsafe { &*ctx };
  ctx.get_epoch_limits(scene_id, start_tai, end_tai)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_ephemeris_position_ffi_signature() {
    // We just test if the symbol exists and compiles.
    let f: unsafe extern "C" fn(*mut SimulationContext, i32, f64, *mut FfiKinematicState) -> bool =
      avkSimulationContext_getEphemerisPosition;
    assert!(f as usize > 0);
  }

  #[test]
  fn test_update_trajectory_ffi_signature() {
    let f: unsafe extern "C" fn(*mut SimulationContext, u64, u64, i32, f64, f64, f64) -> u64 =
      avkSimulationContext_updateTrajectoryForSpk;
    assert!(f as usize > 0);
  }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn avkSimulationContext_addJet(
  ctx: *mut SimulationContext,
  scene_id: u64,
  comet_id: u64,
  radius_km: f32,
  lat: f32,
  lon: f32,
  color_r: f32,
  color_g: f32,
  color_b: f32,
  mass: f32,
  particles_per_tick: u32,
  ttl: f32,
  mean_velocity: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &*ctx };
  match ctx_ref.add_jet(
    scene_id,
    comet_id,
    radius_km,
    lat,
    lon,
    color_r,
    color_g,
    color_b,
    mass,
    particles_per_tick,
    ttl,
    mean_velocity,
  ) {
    Ok(id) => id,
    Err(e) => {
      aethervk_oshal_rlib::log!("Error adding jet: {:?}", e);
      0
    }
  }
}
