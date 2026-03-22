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

  pub entry_wrapper: utils::EntryWrapper,
}

impl Instance {
  /// ## Safety
  /// See `utils::vk_entry` and `ash::Entry::create_instance`
  pub(super) unsafe fn new(base_path_override: Option<&CStr>) -> GpuResult<Self> {
    let entry_wrapper = utils::EntryWrapper::new(base_path_override)?;
    let vk_entry = entry_wrapper
      .weak_entry()
      .upgrade()
      .ok_or(GpuError::UnsupportedFeature)?;

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
    #[cfg(debug_assertions)]
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
    let mut layer_settings = Vec::<vk::LayerSettingEXT>::with_capacity(16);
    #[cfg(debug_assertions)]
    let mut layer_settings_create_info = vk::LayerSettingsCreateInfoEXT::default();
    #[cfg(debug_assertions)]
    let validation_layer_enables_values =
      [c"VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT".as_ptr()];
    #[cfg(debug_assertions)]
    if has_khronos_validation && has_layer_settings {
      layer_settings.push({
        let mut l = vk::LayerSettingEXT::default()
          .layer_name(c"VK_LAYER_KHRONOS_validation")
          .setting_name(c"enables")
          .ty(vk::LayerSettingTypeEXT::STRING);
        l.value_count = validation_layer_enables_values.len() as u32;
        l.p_values = validation_layer_enables_values.as_ptr().cast();

        l
      });

      layer_settings_create_info = layer_settings_create_info.settings(&layer_settings);
    }
    #[cfg(debug_assertions)]
    let mut msg_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
      .message_severity(
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
          | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
          | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
          | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
      )
      .message_type(
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
          | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
          | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
      )
      .pfn_user_callback(Some(utils::debug_utils_messenger_user_callback));

    let instance_extensions = Vec::from_iter(
      desired_instance_extensions
        .iter()
        .map(|&c_str| c_str.as_ptr()),
    );

    // Setup Instance
    let mut instance_create_info = vk::InstanceCreateInfo::default()
      .application_info(&app_info)
      .enabled_extension_names(&instance_extensions);
    #[cfg(target_vendor = "apple")]
    {
      instance_create_info =
        instance_create_info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
    }
    #[cfg(debug_assertions)]
    let enabled_layers = Vec::from_iter(desired_layer_names.iter().map(|&c_str| c_str.as_ptr()));
    #[cfg(debug_assertions)]
    {
      instance_create_info = instance_create_info
        .enabled_layer_names(&enabled_layers)
        .push_next(&mut msg_create_info);
      if !layer_settings.is_empty() {
        instance_create_info = instance_create_info.push_next(&mut layer_settings_create_info);
      }
    }

    let instance = unsafe { vk_entry.create_instance(&instance_create_info, None) }?;
    #[cfg(debug_assertions)]
    {
      // fetch PFN_vkCreateDebugUtilsMessengerEXT
      let dbg_instance = ash::ext::debug_utils::Instance::new(vk_entry.as_ref(), &instance);
      let debug_messenger =
        unsafe { dbg_instance.create_debug_utils_messenger(&msg_create_info, None) }?;
      Ok(Self {
        instance,
        debug_messenger,
        entry_wrapper,
      })
    }
    #[cfg(not(debug_assertions))]
    {
      Ok(Self { instance })
    }
  }

  pub(super) fn api_version(&self) -> u32 {
    // TODO if changed
    vk::API_VERSION_1_1
  }

  pub(super) fn get_eligible_devices(
    &self,
    query_input: &utils::PhysicalDeviceQueryInput,
  ) -> GpuResult<Vec<utils::PhysicalDeviceQueryResult>> {
    let entry = self
      .entry_wrapper
      .weak_entry()
      .upgrade()
      .ok_or(GpuError::DeviceLost)?;
    // 1. enumerate vulkan capable devices
    let physical_devices = unsafe { self.instance.enumerate_physical_devices() }?;
    if physical_devices.is_empty() {
      return Err(GpuError::UnsupportedFeatureNamed(
        "No Vulkan Capable Devices Found".to_string(),
      ));
    }
    // 2. filter those which are eligible and map to query result
    let mut eligible_devices =
      Vec::from_iter(physical_devices.iter().filter_map(|&physical_device| {
        // a. properties (TODO: Subgroup information)
        let mut props = vk::PhysicalDeviceProperties2::default();
        unsafe {
          self
            .instance
            .get_physical_device_properties2(physical_device, &mut props)
        };
        // TODO log

        // b. supported queue families
        let queue_family_properties_len = unsafe {
          self
            .instance
            .get_physical_device_queue_family_properties2_len(physical_device)
        };
        let mut queue_family_properties: Vec<_> =
          core::iter::repeat_with(|| vk::QueueFamilyProperties2::default())
            .take(queue_family_properties_len)
            .collect();
        unsafe {
          self.instance.get_physical_device_queue_family_properties2(
            physical_device,
            &mut queue_family_properties,
          )
        };

        let graphics_queue_family_index = queue_family_properties.iter().enumerate().position(
          |(queue_family_index, queue_props)| {
            // first queue family supporting graphics and presentation
            queue_props
              .queue_family_properties
              .queue_flags
              .contains(vk::QueueFlags::GRAPHICS)
              && query_input.supports_presentation(
                entry.as_ref(),
                physical_device,
                self.instance.handle(),
                queue_family_index as u32,
              )
          },
        )? as u32;

        let compute_queue_family_index = queue_family_properties
          .iter()
          .position(|queue_props| {
            // try to find async compute ...
            let flags = queue_props.queue_family_properties.queue_flags;
            flags.contains(vk::QueueFlags::COMPUTE) && !flags.contains(vk::QueueFlags::GRAPHICS)
          })
          .or_else(|| {
            queue_family_properties.iter().position(|queue_props| {
              // ... otherwise mixed graphics is fine
              let flags = queue_props.queue_family_properties.queue_flags;
              flags.contains(vk::QueueFlags::COMPUTE) && flags.contains(vk::QueueFlags::GRAPHICS)
            })
          })? as u32;

        let transfer_queue_family_index = queue_family_properties
          .iter()
          .position(|queue_props| {
            // try to find a dedicated DMA engine (no graphics, no compute)
            let flags = queue_props.queue_family_properties.queue_flags;
            flags.contains(vk::QueueFlags::TRANSFER)
              && !flags.contains(vk::QueueFlags::GRAPHICS)
              && !flags.contains(vk::QueueFlags::COMPUTE)
          })
          .or_else(|| {
            queue_family_properties.iter().position(|queue_props| {
              // otherwise compute and transfer is fine
              let flags = queue_props.queue_family_properties.queue_flags;
              flags.contains(vk::QueueFlags::TRANSFER) && flags.contains(vk::QueueFlags::COMPUTE)
            })
          })
          .or_else(|| {
            queue_family_properties.iter().position(|queue_props| {
              // otherwise any transfer is fine
              let flags = queue_props.queue_family_properties.queue_flags;
              flags.contains(vk::QueueFlags::TRANSFER)
            })
          })? as u32;

        // c. required device extensions and optional device extensions (TODO)
        let mut desired_device_extensions = Vec::with_capacity(64);
        desired_device_extensions.extend_from_slice(utils::required_device_extensions());
        let device_extension_properties = unsafe {
          self
            .instance
            .enumerate_device_extension_properties(physical_device)
        }
        .ok()?;

        if utils::first_unsupported_extension(
          &desired_device_extensions,
          &device_extension_properties,
        )
        .is_some()
        {
          // TODO log
          return None;
        }

        // d. device features
        let mut required_features = utils::RequiredFeatures::new();
        let mut features2 = required_features.as_features2();
        unsafe {
          self
            .instance
            .get_physical_device_features2(physical_device, &mut features2)
        };
        if required_features.any_missing().is_some() {
          // TODO Log
          return None;
        }

        // e. device is valid, compute its score
        let score: i32 = match props.properties.device_type {
          vk::PhysicalDeviceType::DISCRETE_GPU => 100,
          vk::PhysicalDeviceType::INTEGRATED_GPU => 50,
          vk::PhysicalDeviceType::VIRTUAL_GPU => 20,
          _ => 1,
        };

        // f. TODO: optional extension and features bookkeeping and score increase/decrease

        Some(utils::PhysicalDeviceQueryResult {
          physical_device,
          physical_device_properties: props.properties,
          family_count: queue_family_properties_len,
          optional_extensions: utils::OptionalExtensionSupportFlags::NONE, // TODO
          graphics_queue_family_index,
          compute_queue_family_index,
          transfer_queue_family_index,
          score,
        })
      }));

    if eligible_devices.is_empty() {
      return Err(GpuError::UnsupportedFeatureNamed("Vulkan Capable Devices were found, but none of them have the featureset required by our application".to_string()));
    }

    // 3. sort on score and return
    eligible_devices.sort_by(|a, b| a.score.cmp(&b.score));
    Ok(eligible_devices)
  }
}

impl Drop for Instance {
  fn drop(&mut self) {
    let vk_entry = self.entry_wrapper.weak_entry().upgrade().unwrap();
    #[cfg(debug_assertions)]
    {
      let dbg_instance = ash::ext::debug_utils::Instance::new(vk_entry.as_ref(), &self.instance);
      unsafe { dbg_instance.destroy_debug_utils_messenger(self.debug_messenger, None) };
    }
    unsafe { self.instance.destroy_instance(None) };
  }
}

// --------------------------- Unit Testing --------------------------------
// TODO
#[cfg(test)]
mod tests {
  use super::*;
}
