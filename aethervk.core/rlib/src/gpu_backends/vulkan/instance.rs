use super::utils;
use crate::types::{GpuError, GpuResult};

use ash::vk;
use core::{
  result::Result::{Err, Ok},
  ffi::CStr,
};
use alloc::{string::ToString, vec::Vec};

// -------------------------------- Instance --------------------------------

pub(super) struct Instance {
  pub instance: ash::Instance,
  #[cfg(debug_assertions)]
  debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl Instance {
  /// ## Safety
  /// See `utils::vk_entry` and `ash::Entry::create_instance`
  pub(super) unsafe fn new() -> GpuResult<Self> {
    let vk_entry = unsafe { utils::vk_entry() };
    if !utils::is_vk_entry_valid(&vk_entry) {
      return Err(GpuError::BackendSpecific(
        "Invalid Vulkan Loader".to_string(),
      ));
    }
    // 1. Declare extensions, layer and settings and debug messenger if debug
    let app_info = vk::ApplicationInfo::default()
      .application_name(c"AetherVk")
      .application_version(vk::make_api_version(0, 1, 1, 0))
      .engine_version(vk::make_api_version(0, 1, 0, 0))
      .api_version(vk::API_VERSION_1_1); // TODO check support to update to 1.3?

    let mut desired_instance_extensions = Vec::<&CStr>::with_capacity(64);
    desired_instance_extensions.extend_from_slice(utils::required_instance_extensions());
    let instance_extensions_properties =
      unsafe { vk_entry.enumerate_instance_extension_properties(None) }?;
    utils::first_unsupported_extension(
      &desired_instance_extensions,
      &instance_extensions_properties,
    )
    .map_or(Ok(()), |unsupported| {
      Err(GpuError::UnsupportedFeatureNamed(
        unsupported.to_str().unwrap().to_string(),
      ))
    })?;
    #[cfg(debug_assertions)]
    let has_layer_settings = desired_instance_extensions
      .iter()
      .find(|&name| *name == ash::ext::layer_settings::NAME)
      .is_some();

    #[cfg(debug_assertions)]
    const LAYER_NAMES: [&CStr; 2] = [
      c"VK_LAYER_KHRONOS_validation",
      c"VK_LAYER_KHRONOS_synchronization2",
    ];
    #[cfg(debug_assertions)]
    let mut has_khronos_validation = false;
    let mut desired_layer_names: Vec<&CStr> = if cfg!(debug_assertions) {
      Vec::with_capacity(4)
    } else {
      Vec::new()
    };
    let layer_properties = unsafe { vk_entry.enumerate_instance_layer_properties() }?;
    #[cfg(debug_assertions)]
    for desired_layer_name in &LAYER_NAMES {
      if layer_properties
        .iter()
        .find(|&p| p.layer_name_as_c_str().unwrap() == *desired_layer_name)
        .is_some()
      {
        desired_layer_names.push(desired_layer_name);
        if *desired_layer_name == c"VK_LAYER_KHRONOS_validation" {
          has_khronos_validation = true;
        }
      }
    }
    #[cfg(debug_assertions)]
    let layer_settings = Vec::<vk::LayerSettingEXT>::with_capacity(16);
    #[cfg(debug_assertions)]
    if has_khronos_validation && has_layer_settings {}

    Err(GpuError::DeviceLost)
  }
}

// -------------------------------- Details --------------------------------

// --------------------------- Unit Testing --------------------------------
// TODO
#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
  use super::*;
}
