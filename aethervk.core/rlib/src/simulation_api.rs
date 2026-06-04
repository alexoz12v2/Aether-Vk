//! simulation_api module.

use crate::{
  gpu::{RenderDevice, RenderDeviceHandle, WeakRenderFrontend, WeakRenderFrontendExt},
  physics,
  scene::{
    CameraComponent, CursorComponent, GridComponent, HighResTransformComponent,
    PhysicalMeshComponent, Scene, SkyComponent, SunComponent, TransformComponent,
  },
  simulation,
  simulation::texture_cache::TextureCache,
  simulation_api::structs::{SceneContext, SimulationSceneData, SimulationTaskManager},
  types::{EngineError, EngineResult, GpuResult},
};
use aethervk_oshal_rlib as oshal;
use alloc::{string::ToString, sync::Arc};
use core::ffi::c_char;
use oshal::{
  math::{
    matrix::Matrix4,
    quaternion::Quaternion,
    vector::{Vector3, vec3::Vec3f32, vec4::Quat},
  },
  os,
  os::pool::WorkloadStatus,
};
use parking_lot::RwLock;

pub mod comet_api;
pub mod components_api;
pub mod core_api;
pub mod logic_thread;
pub mod misc_api;
pub mod models_api;
pub mod render_thread;
pub mod scene_api;
pub mod structs;
pub mod time_api;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod integration_test_large;
/// Holds State for the whole simulation. For now, default drop order (from first to last)
/// is fine. Threads are first shut down, then data is deallocated.
/// if this grows more complicated, implement `Drop`
/// creation in [`core_api::SimulationContext::startup`] is in reverse
pub struct SimulationContext {
  /// Necessary evil to let the render thread own a strong reference to render frontend, but
  /// still having the possibility for FFI caller threads to create presentation engines
  pub render_proxy: (WeakRenderFrontend, RenderDeviceHandle),
  pub threads: structs::SimulationThreads,
  pub scenes: Arc<RwLock<SimulationSceneData>>,
  pub task_manager: Arc<RwLock<SimulationTaskManager>>,
  pub logic_state: Arc<RwLock<structs::LogicState>>,
  pub kernels: Arc<RwLock<structs::KernelsEnum>>,
  texture_cache: Arc<RwLock<TextureCache>>,
  pub audio_mixer: Arc<RwLock<crate::audio::AudioMixer>>,
}

impl Drop for SimulationContext {
  fn drop(&mut self) {
    oshal::log!("SimulationContext drop started");
  }
}

unsafe impl Sync for SimulationContext {}

impl SimulationContext {
  /// Necessary evil function so that FFI caller threads can create a presentation engine on demand
  /// even though render device is owned by render thread
  pub fn with_device<F, R>(&self, f: F) -> EngineResult<R>
  where
    F: FnOnce(&dyn RenderDevice) -> GpuResult<R>,
  {
    let render_device_handle = self.render_proxy.1;
    let render_frontend =
      self.render_proxy.0.as_frontend().ok_or(EngineError::InvalidOperation(
        "SimulationContext::with_device | couldn't upgrade weak pointer to render context",
      ))?;
    render_frontend
      .with_device(render_device_handle, f)
      .map_err(|e| EngineError::from(e))
  }

  /// TODO: Document this item
  pub fn get_scene(&self, scene_id: u64) -> Option<Arc<RwLock<SceneContext>>> {
    self.scenes.read().get(&scene_id).cloned()
  }
}

/// Macro to quickly extract scene, convert option to result and produce a `&'static str` error message
#[macro_export]
macro_rules! expect_scene {
  ($expr:expr, $context:expr) => {
    $expr.ok_or(EngineError::InvalidOperation(concat!(
      $context,
      " | scene not found"
    )))?
  };
}
/// Macro to quickly extract an entity from a scene, convert option to result and produce a `&'static str` error message
#[macro_export]
macro_rules! expect_entity {
  ($scene:expr, $entity:expr, $context:expr) => {
    $scene.get_entity($entity).ok_or(EngineError::InvalidOperation(concat!(
      $context,
      " | child entity not found"
    )))?
  };
}
/// Macro to quickly extract both a scene and an entity, returning a tuple (scene, entity_id),
/// while converting options to results with `&'static str` error messages.
#[macro_export]
macro_rules! expect_scene_and_entity {
  ($scene_expr:expr, $entity_expr:expr, $context:expr) => {{
    // Re-use your existing scene macro
    let scene = $crate::expect_scene!($scene_expr, $context);

    // Extract the entity
    let entity =
      scene
        .read()
        .get_entity($entity_expr)
        .ok_or(EngineError::InvalidOperation(concat!(
          $context,
          " | child entity not found"
        )))?;

    // Return both so they can be destructured
    (scene, entity)
  }};
}

struct PhysicsRebuildWorkload {
  scene: Arc<Scene>,
  physics_scene: Arc<RwLock<physics::physics_scene::PhysicsScene>>,
}

impl os::pool::Workload for PhysicsRebuildWorkload {
  fn execute(&mut self) -> WorkloadStatus {
    let new_physics = physics::physics_scene::PhysicsScene::build_from_scene(&self.scene, 0.016);
    let mut guard = self.physics_scene.write();
    *guard = new_physics;
    WorkloadStatus::Complete
  }
}

/// TODO: Document this item
pub type BreadcrumbCallback = unsafe extern "C" fn(u32, *const core::ffi::c_char);
pub static BREADCRUMB_CALLBACK: parking_lot::RwLock<Option<BreadcrumbCallback>> = parking_lot::RwLock::new(None);

pub type SimulationCallback = unsafe extern "C" fn(u64, u64, u64, *const core::ffi::c_void);
pub static SIMULATION_CALLBACK: parking_lot::RwLock<Option<SimulationCallback>> = parking_lot::RwLock::new(None);

pub type RenderCallback = unsafe extern "C" fn(u64, u64, u64);
pub static RENDER_CALLBACK: parking_lot::RwLock<Option<RenderCallback>> = parking_lot::RwLock::new(None);

/// TODO: Document this item
pub fn emit_breadcrumb(status: u32, msg: &str) {
  if let Some(cb) = *BREADCRUMB_CALLBACK.read() {
    if let Ok(c_msg) = alloc::ffi::CString::new(msg) {
      unsafe { cb(status, c_msg.as_ptr()) };
    }
  }
}
pub mod test_composite_render;
pub mod test_lca_render;
