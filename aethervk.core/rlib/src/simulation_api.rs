//! simulation_api module.

use crate::{
  gpu::{RenderDevice, RenderDeviceHandle, WeakRenderFrontend, WeakRenderFrontendExt},
  gpu_backends::vulkan,
  scene::{
    CameraComponent, CursorComponent, GridComponent, HighResTransformComponent, SkyComponent,
    SunComponent, TransformComponent,
  },
  simulation::{self, texture_cache::TextureCache},
  simulation_api::structs::{SceneContext, SimulationSceneData},
  types::{EngineError, EngineResult, GpuResult},
};
use aethervk_oshal_rlib as oshal;
use alloc::{string::ToString, sync::Arc};
use oshal::math::{
  quaternion::Quaternion,
  vector::{Vector3, vec3::Vec3f32, vec4::Quat},
};
use parking_lot::RwLock;

pub mod components_api;
pub mod core_api;
pub mod logic_thread;
pub mod misc_api;
pub mod render_thread;
pub mod scene_api;
pub mod structs;
pub mod time_api;
const MAX_UNSCALED_DELTA_MS: u32 = 500_u32;

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
  pub logic_state: Arc<RwLock<structs::LogicState>>,
  texture_cache: Arc<RwLock<TextureCache>>,
  pub audio_mixer: Arc<RwLock<crate::audio::AudioMixer>>,
}

impl Drop for SimulationContext {
  fn drop(&mut self) {
    oshal::log!("SimulationContext drop started");
    // Now drop from top to bottom all members
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

  pub fn get_scene(&self, scene_id: u64) -> Option<Arc<RwLock<SceneContext>>> {
    self.scenes.read().get(&scene_id).cloned()
  }

  pub fn unload_model(&self, model_id: u64) {
    // TODO: all [`crate::scene::StaticMeshComponent`] using this mesh should have their mesh
    // swapped with a procedurally generated sphere
    if self.scenes.write().model_registry.remove(&model_id).is_some() {
      emit_breadcrumb(1, &alloc::format!("Unloaded model {}", model_id));
    }
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

/// Callback towards Managed side of the application to communicate errors arising in the logic
/// layer or below
pub type BreadcrumbCallback = unsafe extern "C" fn(u32, *const core::ffi::c_char);
pub static BREADCRUMB_CALLBACK: parking_lot::RwLock<Option<BreadcrumbCallback>> =
  parking_lot::RwLock::new(None);

/// Callback towards Managed side of the application to communicate updates to a given scene. Its
/// format is the following
/// - first argument: scene_id
/// - second argument: external entity id
/// - third argument: component foreign id
/// - fourth argument: component specific DTO buffer (length and format knowledge is implicit on the
///   foreign id. Holds also computed properties which are not strictly stored)
pub type SimulationCallback = unsafe extern "C" fn(u64, u64, u64, *const core::ffi::c_void);
pub static SIMULATION_CALLBACK: parking_lot::RwLock<Option<SimulationCallback>> =
  parking_lot::RwLock::new(None);

pub type RenderCallback = unsafe extern "C" fn(u64, u64, u64);
pub static RENDER_CALLBACK: parking_lot::RwLock<Option<RenderCallback>> =
  parking_lot::RwLock::new(None);

/// Callback towards Managed side of the application to communicate updates for simulation state
/// which are not part of any scene.
/// - first argument: state identifier (documented in ExternalState enum)
/// - second argument: state buffer, whose length and format is implicitly known given state
///   identifier
pub type ExternalStateSimulationCallback = unsafe extern "C" fn(u32, *const core::ffi::c_void);
pub static EXTERNAL_STATE_SIMULATION_CALLBACK: parking_lot::RwLock<
  Option<ExternalStateSimulationCallback>,
> = parking_lot::RwLock::new(None);

/// Function to set a delegate callback for external state propagation
pub fn set_external_state_simulation_callback(cb: Option<ExternalStateSimulationCallback>) {
  *EXTERNAL_STATE_SIMULATION_CALLBACK.write() = cb;
}

/// Callback signature for functions which should be executed from the main thread
/// Made so that Render Device can do its periodic and shutdown cleanup
/// u32 is the type of action to perform.
/// 1 -> `vulkan_device.process_main_thread_cleanup_queue()`
/// 2 -> `vulkan_device.flush_main_thread_cleanup_queue()`
/// Second argument is an optional pointer to an atomic bool so that, if necessary, caller can
/// spin-wait on the atomic variable
///
/// Native runtime guarantees [`vulkan::device::Device`] won't be dropped if and only if
/// - when called with `2` (called during the drop) we also give a non null signal done
///   To be safe, we always pass a `signal_done`, so that when dropping we first wait on the previous
///   callback execution
pub type MainThreadDispatchCallback = unsafe extern "C" fn(
  device_ptr: *const vulkan::device::Device,
  command: u32,
  signal_done: *const core::sync::atomic::AtomicBool,
);
pub static MAIN_THREAD_DISPATCH_CALLBACK: parking_lot::RwLock<Option<MainThreadDispatchCallback>> =
  parking_lot::RwLock::new(None);

/// Function to be executed when registering a delegate function pointer for main thread affinity
/// render related cleanup function
///
/// To be wrapped into a no_mangle "C" calling convetion function at FFI Layer
pub fn register_main_thread_dispatcher(cb: Option<MainThreadDispatchCallback>) {
  *MAIN_THREAD_DISPATCH_CALLBACK.write() = cb
}

/// This function should be pointed by the C# managed code to be called when the `MAIN_THREAD_DISPATCH_CALLBACK`
/// is invoked
///
/// This should be reexpoted with C ABI towards C#, and it should be called
///
/// # Safety
/// - `device_ptr` should be not null and pointing to a valid vulkan device abstraction
pub unsafe fn execute_main_thread_cleanup(
  device_ptr: *const vulkan::device::Device,
  command: u32,
  signal_done: *const core::sync::atomic::AtomicBool,
) {
  const PROCESS_MAIN_THREAD_CLEANUP_QUEUE: u32 = 1;
  const FLUSH_MAIN_THREAD_CLEANUP_QUEUE: u32 = 2;
  let vulkan_device = unsafe { device_ptr.as_ref().unwrap() };
  #[cfg(debug_assertions)]
  if !signal_done.is_null() {
    let atomic_ref = unsafe { signal_done.as_ref().unwrap() };
    assert!(!atomic_ref.load(core::sync::atomic::Ordering::Relaxed));
  }

  // TODO add logging on error
  if command == PROCESS_MAIN_THREAD_CLEANUP_QUEUE {
    let _ = vulkan_device.process_main_thread_cleanup_queue();
  } else if command == FLUSH_MAIN_THREAD_CLEANUP_QUEUE {
    let _ = vulkan_device.flush_main_thread_cleanup_queue();
  }

  if !signal_done.is_null() {
    let atomic_ref = unsafe { signal_done.as_ref().unwrap() };
    atomic_ref.store(true, core::sync::atomic::Ordering::Release);
  }
}

/// To be invoked periodically.
///
/// # Safety
/// - If [`MAIN_THREAD_DISPATCH_CALLBACK`] was correctly populated, and if
///   C# managed callback towards [`execute_main_thread_cleanup`] was correctly wired up, then this
///   function will ensure that `vulkan_device.process_main_thread_cleanup_queue()` will be
///   asynchronously executed on the main thread
/// - `signal_done` should be false
pub(crate) unsafe fn invoke_main_thread_process_cleanup(
  vulkan_device: &vulkan::device::Device,
  signal_done: &core::sync::atomic::AtomicBool,
) {
  if let Some(cb) = *MAIN_THREAD_DISPATCH_CALLBACK.read() {
    unsafe { cb(vulkan_device as *const _, 1, signal_done as *const _) };
  }
}

/// To be invoked during cleanup, eg `RenderCommand::Shutdown`
///
/// SAFETY:
/// - If [`MAIN_THREAD_DISPATCH_CALLBACK`] was correctly populated, and if
///   C# managed callback towards [`execute_main_thread_cleanup`] was correctly wired up, then this
///   function will ensure that `vulkan_device.process_main_thread_cleanup_queue()` will be
///   asynchronously executed on the main thread
/// - `signal_done` should be false
pub(crate) unsafe fn invoke_main_thread_flush_cleanup(
  vulkan_device: &vulkan::device::Device,
  signal_done: &core::sync::atomic::AtomicBool,
) {
  if let Some(cb) = *MAIN_THREAD_DISPATCH_CALLBACK.read() {
    unsafe { cb(vulkan_device as *const _, 2, signal_done as *const _) };
  }
}

pub mod external_state {
  #[repr(C)]
  #[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod)]
  pub struct CTimeRange {
    /// Index 0: Start, Index 1: End
    pub nanoseconds: [u64; 2],
    pub centuries: [i16; 2],
    pub _padding: [u8; 4],
  }
  impl CTimeRange {
    pub fn new(start: hifitime::Epoch, end: hifitime::Epoch) -> Self {
      let (start_c, start_ns) = start.to_tai_parts();
      let (end_c, end_ns) = end.to_tai_parts();
      Self {
        nanoseconds: [start_ns, end_ns],
        centuries: [start_c, end_c],
        _padding: [0; 4],
      }
    }
  }

  #[repr(C)]
  #[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Zeroable, bytemuck::Pod)]
  pub struct CModelImported {
    pub model_id: u64,
    /// Fixed size buffer for the mesh name. small
    pub path_bytes: [u8; 32],
  }
  impl CModelImported {
    pub fn new(model_id: u64, path: &str) -> Self {
      let mut path_bytes = [0u8; 32];
      let bytes = path.rfind(['/', '\\']).map(|idx| &path[idx + 1..]).unwrap_or(path).as_bytes();
      // truncate to 31 to null-terminate the string
      let len = bytes.len().min(31);
      path_bytes[..len].copy_from_slice(&bytes[..len]);
      Self {
        model_id,
        path_bytes,
      }
    }
  }

  pub enum ExternalState {
    /// Signifies a change in the epoch range state
    TimeRange(CTimeRange),
    /// Signifies a new model was successfully imported
    ModelImported(CModelImported),
  }

  impl ExternalState {
    pub fn state_identifier(&self) -> u32 {
      match self {
        Self::TimeRange(_) => 1,
        Self::ModelImported(_) => 2,
      }
    }
  }
}

pub fn emit_external_state_change(external_state: &external_state::ExternalState) {
  use external_state::ExternalState;
  if let Some(cb) = *EXTERNAL_STATE_SIMULATION_CALLBACK.read() {
    let id = external_state.state_identifier();
    let bytes_ptr = match external_state {
      ExternalState::TimeRange(time_range) => {
        bytemuck::bytes_of(time_range).as_ptr().cast::<core::ffi::c_void>()
      }
      ExternalState::ModelImported(model_imported) => {
        bytemuck::bytes_of(model_imported).as_ptr().cast()
      }
    };
    unsafe { cb(id, bytes_ptr) };
  }
}

/// Wrapper function to invoke the [`BREADCRUMB_CALLBACK`], if it was set by C# side
pub fn emit_breadcrumb(status: u32, msg: &str) {
  if let Some(cb) = *BREADCRUMB_CALLBACK.read()
    && let Ok(c_msg) = alloc::ffi::CString::new(msg)
  {
    unsafe { cb(status, c_msg.as_ptr()) };
  }
}

/// Prints the full scene hierarchy to the log in a tree format (debug builds only).
#[cfg(debug_assertions)]
pub fn debug_print_scene_hierarchy(ctx_ref: &SimulationContext, scene_id: u64) {
  let scene_arc = match ctx_ref.scenes.read().get_scene(scene_id) {
    Some(s) => s,
    None => {
      oshal::log!("[Scene Hierarchy] scene {} not found", scene_id);
      return;
    }
  };
  let scene_ctx = scene_arc.read();

  // Find root entities (those with no parent).
  let all_ext_ids: alloc::vec::Vec<u64> = scene_ctx.entity_map.keys().copied().collect();
  let mut roots: alloc::vec::Vec<u64> = alloc::vec::Vec::new();
  for &ext_id in &all_ext_ids {
    if let Some(int_id) = scene_ctx.get_entity(ext_id) {
      let parent = scene_ctx.scene.get_parent(int_id);
      if parent.is_none() {
        roots.push(ext_id);
      }
    }
  }

  oshal::log!(
    "[Scene Hierarchy] scene={} total_entities={}",
    scene_id,
    all_ext_ids.len()
  );

  // DFS print helper using an explicit stack to avoid recursion in no_std.
  // Stack entries: (ext_id, depth)
  let mut stack: alloc::vec::Vec<(u64, usize)> = roots.iter().map(|&id| (id, 0)).collect();
  // Reverse so first root prints first.
  stack.reverse();

  while let Some((ext_id, depth)) = stack.pop() {
    let int_id = match scene_ctx.get_entity(ext_id) {
      Some(id) => id,
      None => continue,
    };
    let name = scene_ctx
      .scene
      .get_name(int_id)
      .unwrap_or_else(|| alloc::string::String::from("<unnamed>"));
    let components = scene_ctx.scene.get_entity_component_names(int_id);
    let indent: alloc::string::String = (0..depth).map(|_| "  ").collect();
    let comp_str = if components.is_empty() {
      alloc::string::String::from("(no components)")
    } else {
      components.join(", ")
    };
    oshal::log!(
      "{}├─ \"{}\" [ext={}] {{{}}}",
      indent,
      name,
      ext_id,
      comp_str
    );

    // Push children in reverse order so first child prints first.
    if let Some(children) = scene_ctx.scene.get_children(int_id) {
      // Map internal ids back to external ids.
      let mut child_ext: alloc::vec::Vec<u64> = alloc::vec::Vec::new();
      for child_int in &children {
        if let Some((&child_ext_id, _)) =
          scene_ctx.entity_map.iter().find(|&(_, &v)| v == *child_int)
        {
          child_ext.push(child_ext_id);
        }
      }
      child_ext.reverse();
      for cext in child_ext {
        stack.push((cext, depth + 1));
      }
    }
  }
}

#[cfg(test)]
pub mod test_composite_render;

#[cfg(test)]
pub mod test_lca_render;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod integration_test_large;
