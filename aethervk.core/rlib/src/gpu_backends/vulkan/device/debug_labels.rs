//! Complete implementation of the VK_EXT_debug_utils extension.
//!
//! Covers all 9 functions defined in the Vulkan spec debugging chapter:
//!
//! ## Object annotation
//! - [`set_object_name`]  — `vkSetDebugUtilsObjectNameEXT`
//! - [`set_object_tag`]   — `vkSetDebugUtilsObjectTagEXT`
//!
//! ## Command-buffer labels
//! - [`DebugLabelScope`]  — RAII begin/end (`vkCmdBeginDebugUtilsLabelEXT` + `vkCmdEndDebugUtilsLabelEXT`)
//! - [`cmd_insert`]       — `vkCmdInsertDebugUtilsLabelEXT`
//!
//! ## Queue labels
//! - [`QueueLabelScope`]  — RAII begin/end (`vkQueueBeginDebugUtilsLabelEXT` + `vkQueueEndDebugUtilsLabelEXT`)
//! - [`queue_insert`]     — `vkQueueInsertDebugUtilsLabelEXT`
//!
//! ## Synthetic messages
//! - [`submit_message`]   — `vkSubmitDebugUtilsMessageEXT`
//!
//! ## Debug messenger lifecycle
//! `vkCreateDebugUtilsMessengerEXT` and `vkDestroyDebugUtilsMessengerEXT` are managed at the
//! instance level in `instance.rs` and are not repeated here.
//!
//! ## Release builds
//! Every function is a `#[cfg(debug_assertions)]` no-op in release builds.
//! The compiler eliminates all calls at zero overhead.

use ash::vk;

// ── Object annotation ────────────────────────────────────────────────────────

/// Assign a human-readable name to any Vulkan object.
///
/// Wraps `vkSetDebugUtilsObjectNameEXT`. The name appears in RenderDoc,
/// validation layer messages, and GPU debuggers.
///
/// Prefer using the existing [`super::VulkanDebugNameExt`] / [`super::VmaDebugNameExt`]
/// ergonomic helpers over calling this directly.
#[inline]
pub fn set_object_name<H: vk::Handle>(
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  debug_utils: &ash::ext::debug_utils::Device,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  handle: H,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  name: &core::ffi::CStr,
) {
  #[cfg(debug_assertions)]
  {
    // ash 0.38: .object_handle<T: Handle>() sets both object_type and object_handle
    // atomically from the T::TYPE associated constant — no separate .object_type() call.
    let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
      .object_handle(handle)
      .object_name(name);
    // Naming is always best-effort — ignore errors.
    let _ = unsafe { debug_utils.set_debug_utils_object_name(&name_info) };
  }
}

/// Attach an arbitrary binary blob to a Vulkan object.
///
/// Wraps `vkSetDebugUtilsObjectTagEXT`. Useful for attaching offline shader
/// debugging data or CPU-side metadata. `tag_name` is a u64 identifier for
/// the tag type — the meaning is tool-defined.
#[inline]
pub fn set_object_tag<H: vk::Handle>(
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  debug_utils: &ash::ext::debug_utils::Device,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  handle: H,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  tag_name: u64,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  tag_data: &[u8],
) {
  #[cfg(debug_assertions)]
  {
    // ash 0.38: .object_handle<T: Handle>() sets both object_type and object_handle.
    let tag_info = vk::DebugUtilsObjectTagInfoEXT::default()
      .object_handle(handle)
      .tag_name(tag_name)
      .tag(tag_data);
    let _ = unsafe { debug_utils.set_debug_utils_object_tag(&tag_info) };
  }
}

// ── Command-buffer labels ────────────────────────────────────────────────────

/// RAII guard for a named command-buffer debug region.
///
/// Calls `vkCmdBeginDebugUtilsLabelEXT` on construction and
/// `vkCmdEndDebugUtilsLabelEXT` on drop. Drop order must match nesting order —
/// Vulkan requires paired begin/end calls within the same command buffer.
///
/// In release builds this type is zero-sized and all methods compile away.
pub struct DebugLabelScope<'d> {
  #[cfg(debug_assertions)]
  debug_utils: &'d ash::ext::debug_utils::Device,
  #[cfg(debug_assertions)]
  cmd: vk::CommandBuffer,
  /// Zero-size marker so the lifetime is well-formed in release builds too.
  _phantom: core::marker::PhantomData<&'d ()>,
}

impl<'d> DebugLabelScope<'d> {
  /// Begin a named debug region on a command buffer.
  ///
  /// `color` is RGBA in \[0.0, 1.0\] — cosmetic only, used by RenderDoc to
  /// colour the event timeline entries.
  #[inline]
  pub fn begin(
    #[cfg_attr(not(debug_assertions), allow(unused_variables))]
    debug_utils: &'d ash::ext::debug_utils::Device,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))]
    cmd: vk::CommandBuffer,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))]
    name: &core::ffi::CStr,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))]
    color: [f32; 4],
  ) -> Self {
    #[cfg(debug_assertions)]
    {
      let label = vk::DebugUtilsLabelEXT::default().label_name(name).color(color);
      unsafe { debug_utils.cmd_begin_debug_utils_label(cmd, &label) };
    }
    Self {
      #[cfg(debug_assertions)]
      debug_utils,
      #[cfg(debug_assertions)]
      cmd,
      _phantom: core::marker::PhantomData,
    }
  }
}

impl Drop for DebugLabelScope<'_> {
  #[inline]
  fn drop(&mut self) {
    #[cfg(debug_assertions)]
    unsafe {
      self.debug_utils.cmd_end_debug_utils_label(self.cmd);
    }
  }
}

/// Insert a single-point label into a command buffer.
///
/// Wraps `vkCmdInsertDebugUtilsLabelEXT`. No matching end required.
/// Useful for marking individual draw calls that don't need a begin/end scope.
#[inline]
pub fn cmd_insert(
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  debug_utils: &ash::ext::debug_utils::Device,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  cmd: vk::CommandBuffer,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  name: &core::ffi::CStr,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  color: [f32; 4],
) {
  #[cfg(debug_assertions)]
  {
    let label = vk::DebugUtilsLabelEXT::default().label_name(name).color(color);
    unsafe { debug_utils.cmd_insert_debug_utils_label(cmd, &label) };
  }
}

// ── Queue labels ─────────────────────────────────────────────────────────────

/// RAII guard for a named queue debug region.
///
/// Calls `vkQueueBeginDebugUtilsLabelEXT` on construction and
/// `vkQueueEndDebugUtilsLabelEXT` on drop. Visible in the RenderDoc queue
/// timeline — distinct from command-buffer labels, which appear inside the
/// frame's command-buffer view.
pub struct QueueLabelScope<'d> {
  #[cfg(debug_assertions)]
  debug_utils: &'d ash::ext::debug_utils::Device,
  #[cfg(debug_assertions)]
  queue: vk::Queue,
  _phantom: core::marker::PhantomData<&'d ()>,
}

impl<'d> QueueLabelScope<'d> {
  /// Begin a named debug region on a queue.
  #[inline]
  pub fn begin(
    #[cfg_attr(not(debug_assertions), allow(unused_variables))]
    debug_utils: &'d ash::ext::debug_utils::Device,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))]
    queue: vk::Queue,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))]
    name: &core::ffi::CStr,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))]
    color: [f32; 4],
  ) -> Self {
    #[cfg(debug_assertions)]
    {
      let label = vk::DebugUtilsLabelEXT::default().label_name(name).color(color);
      unsafe { debug_utils.queue_begin_debug_utils_label(queue, &label) };
    }
    Self {
      #[cfg(debug_assertions)]
      debug_utils,
      #[cfg(debug_assertions)]
      queue,
      _phantom: core::marker::PhantomData,
    }
  }
}

impl Drop for QueueLabelScope<'_> {
  #[inline]
  fn drop(&mut self) {
    #[cfg(debug_assertions)]
    unsafe {
      self.debug_utils.queue_end_debug_utils_label(self.queue);
    }
  }
}

/// Insert a single-point label into a queue's debug timeline.
///
/// Wraps `vkQueueInsertDebugUtilsLabelEXT`.
#[inline]
pub fn queue_insert(
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  debug_utils: &ash::ext::debug_utils::Device,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  queue: vk::Queue,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  name: &core::ffi::CStr,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  color: [f32; 4],
) {
  #[cfg(debug_assertions)]
  {
    let label = vk::DebugUtilsLabelEXT::default().label_name(name).color(color);
    unsafe { debug_utils.queue_insert_debug_utils_label(queue, &label) };
  }
}

// ── Synthetic messages ────────────────────────────────────────────────────────

/// Inject a synthetic message into the debug messenger callback chain.
///
/// Wraps `vkSubmitDebugUtilsMessageEXT` (instance-level). Useful for:
/// - Inserting application-defined checkpoints into validation layer logs.
/// - Unit-testing message callback handlers.
/// - Marking frame boundaries in external profiling tools.
///
/// `objects` may be empty (`&[]`). `severity` and `types` must each have at
/// least one bit set (Vulkan spec VUID requirement).
#[inline]
pub fn submit_message(
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  instance_debug_utils: &ash::ext::debug_utils::Instance,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  severity: vk::DebugUtilsMessageSeverityFlagsEXT,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  types: vk::DebugUtilsMessageTypeFlagsEXT,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  message_id_name: &core::ffi::CStr,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  message: &core::ffi::CStr,
  #[cfg_attr(not(debug_assertions), allow(unused_variables))]
  objects: &[vk::DebugUtilsObjectNameInfoEXT<'_>],
) {
  #[cfg(debug_assertions)]
  {
    let callback_data = vk::DebugUtilsMessengerCallbackDataEXT::default()
      .message_id_name(message_id_name)
      .message_id_number(0)
      .message(message)
      .objects(objects);
    unsafe {
      instance_debug_utils.submit_debug_utils_message(severity, types, &callback_data);
    }
  }
}



