//! instance module.

use super::utils;
use crate::types::{GpuError, GpuResult};

use aethervk_oshal_rlib as oshal;
use alloc::{string::ToString, vec::Vec};
use ash::vk;
use core::{
  ffi::CStr,
  result::Result::{Err, Ok},
};

// -------------------------------- Instance --------------------------------

/// TODO: Document this item
pub struct Instance {
  pub instance: ash::Instance,
  #[cfg(debug_assertions)]
  debug_messenger: vk::DebugUtilsMessengerEXT,

  pub entry_wrapper: utils::EntryWrapper,
  pub has_surface_maintenance1: bool,
  pub has_headless_surface: bool,
}

impl Instance {
  /// ## Safety
  /// See `utils::vk_entry` and `ash::Entry::create_instance`
  pub unsafe fn new(
    base_path_override: Option<&CStr>,
    validation_error_callback: Option<fn(&str)>,
  ) -> GpuResult<Self> {
    let entry_wrapper = utils::EntryWrapper::new(base_path_override)?;
    let vk_entry = entry_wrapper.weak_entry().upgrade().ok_or(GpuError::UnsupportedFeature)?;

    let app_info = vk::ApplicationInfo::default()
      .application_name(c"AetherVk")
      .application_version(vk::make_api_version(0, 1, 1, 0))
      .engine_version(vk::make_api_version(0, 1, 0, 0))
      .api_version(vk::API_VERSION_1_1);

    // =========================================================================
    // 1. Resolve Layers First
    // =========================================================================
    #[cfg(debug_assertions)]
    let mut layer_names = alloc::vec![c"VK_LAYER_KHRONOS_validation"];
    
    #[cfg(debug_assertions)]
    {
      if aethervk_oshal_rlib::os::env::var("AETHERVK_DISABLE_SYNC_VAL").is_none() {
        layer_names.push(c"VK_LAYER_KHRONOS_synchronization2");
      }
    }

    let mut desired_layer_names: Vec<&CStr> = if cfg!(debug_assertions) {
      Vec::with_capacity(4)
    } else {
      Vec::new()
    };

    #[cfg(debug_assertions)]
    let mut has_khronos_validation = false;

    #[cfg(debug_assertions)]
    {
      let layer_properties = unsafe { vk_entry.enumerate_instance_layer_properties() }?;
      for desired_layer_name in &layer_names {
        if layer_properties
          .iter()
          .any(|p| p.layer_name_as_c_str().unwrap() == *desired_layer_name)
        {
          desired_layer_names.push(desired_layer_name);
          if *desired_layer_name == c"VK_LAYER_KHRONOS_validation" {
            has_khronos_validation = true;
          }
        }
      }
    }

    // =========================================================================
    // 2. Resolve and Validate Extensions
    // =========================================================================
    let mut desired_instance_extensions = Vec::<&CStr>::with_capacity(64);
    desired_instance_extensions.extend_from_slice(utils::required_instance_extensions());

    // Get global extensions (None)
    let mut available_extensions =
      unsafe { vk_entry.enumerate_instance_extension_properties(None) }?;

    // Get layer-specific extensions and pool them together
    #[cfg(debug_assertions)]
    {
      for layer in &desired_layer_names {
        if let Ok(layer_exts) =
          unsafe { vk_entry.enumerate_instance_extension_properties(Some(*layer)) }
        {
          available_extensions.extend(layer_exts);
        }
      }
    }

    // --- Dynamically check for Validation Features ---
    #[cfg(debug_assertions)]
    let mut has_validation_features = false;

    #[cfg(debug_assertions)]
    {
      let features_ext = ash::ext::validation_features::NAME;
      if has_khronos_validation
        && available_extensions
          .iter()
          .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == features_ext)
      {
        desired_instance_extensions.push(features_ext);
        has_validation_features = true;
      } else {
        // If we hit this, RenderDoc is likely hiding the extension.
        // We log it and move on without crashing.
        oshal::log!("VK_EXT_validation_features not found. Programmatic layer settings disabled.");
      }
    }

    // --- Dynamically check for surface_maintenance1 ---
    let mut has_surface_maintenance1 = false;
    let surface_maint_ext = ash::ext::surface_maintenance1::NAME;
    if available_extensions
      .iter()
      .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == surface_maint_ext)
    {
      desired_instance_extensions.push(surface_maint_ext);
      has_surface_maintenance1 = true;
    }

    // --- Dynamically check for headless_surface ---
    let mut has_headless_surface = false;
    #[cfg(test)]
    {
      let headless_surface_ext = ash::ext::headless_surface::NAME;
      if available_extensions
        .iter()
        .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == headless_surface_ext)
      {
        desired_instance_extensions.push(headless_surface_ext);
        has_headless_surface = true;
      }
    }

    // Now check support against the merged pool
    if let Some(unsupported) =
      utils::first_unsupported_extension(&desired_instance_extensions, &available_extensions)
    {
      return Err(GpuError::UnsupportedFeatureNamed(
        unsupported.to_str().unwrap().to_string(),
      ));
    }

    // =========================================================================
    // 3. Setup Validation Features & Debug Messenger
    // =========================================================================
    #[cfg(debug_assertions)]
    let mut printf_features = alloc::vec![];

    #[cfg(debug_assertions)]
    let disable_gpu_av = aethervk_oshal_rlib::os::env::var("AETHERVK_DISABLE_GPU_AV").is_some();

    #[cfg(debug_assertions)]
    if cfg!(target_vendor = "apple") || disable_gpu_av {
      aethervk_oshal_rlib::log!("Disabling GPU-Assisted/Printf Validation.");
    } else {
      #[cfg(test)]
      {
        #[cfg(all(test, not(target_vendor = "apple")))]
        if crate::gpu_backends::vulkan::physics::USE_PRINTF_SHADERS
          .load(core::sync::atomic::Ordering::Relaxed)
        {
          printf_features.push(vk::ValidationFeatureEnableEXT::DEBUG_PRINTF);
        } else {
          printf_features.push(vk::ValidationFeatureEnableEXT::GPU_ASSISTED);
          printf_features.push(vk::ValidationFeatureEnableEXT::GPU_ASSISTED_RESERVE_BINDING_SLOT);
        }
        #[cfg(not(all(test, not(target_vendor = "apple"))))]
        {
          printf_features.push(vk::ValidationFeatureEnableEXT::GPU_ASSISTED);
          printf_features.push(vk::ValidationFeatureEnableEXT::GPU_ASSISTED_RESERVE_BINDING_SLOT);
        }
      }
      #[cfg(not(test))]
      {
        printf_features.push(vk::ValidationFeatureEnableEXT::DEBUG_PRINTF);
      }
    }

    #[cfg(debug_assertions)]
    let mut validation_features =
      vk::ValidationFeaturesEXT::default().enabled_validation_features(&printf_features);

    #[cfg(debug_assertions)]
    let p_user_data = validation_error_callback
      .map(|f| f as *mut core::ffi::c_void)
      .unwrap_or(core::ptr::null_mut());

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
      .user_data(p_user_data)
      .pfn_user_callback(Some(utils::debug_utils_messenger_user_callback));

    // =========================================================================
    // 4. Create Instance
    // =========================================================================
    let instance_extensions =
      Vec::from_iter(desired_instance_extensions.iter().map(|&c_str| c_str.as_ptr()));

    let mut instance_create_info = vk::InstanceCreateInfo::default()
      .application_info(&app_info)
      .enabled_extension_names(&instance_extensions);

    #[cfg(target_vendor = "apple")]
    let mut export_metal_objects = vk::ExportMetalObjectCreateInfoEXT::default()
      .export_object_type(vk::ExportMetalObjectTypeFlagsEXT::METAL_DEVICE);
    #[cfg(target_vendor = "apple")]
    {
      instance_create_info = instance_create_info
        .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
        .push_next(&mut export_metal_objects);
    }

    #[cfg(debug_assertions)]
    let enabled_layers = Vec::from_iter(desired_layer_names.iter().map(|&c_str| c_str.as_ptr()));

    #[cfg(debug_assertions)]
    {
      instance_create_info = instance_create_info
        .enabled_layer_names(&enabled_layers)
        .push_next(&mut msg_create_info);

      // Only attach the features struct if the extension is actually supported/visible
      if has_validation_features {
        instance_create_info = instance_create_info.push_next(&mut validation_features);
      }
    }

    let instance = unsafe { vk_entry.create_instance(&instance_create_info, None) }?;

    #[cfg(debug_assertions)]
    {
      let dbg_instance = ash::ext::debug_utils::Instance::new(vk_entry.as_ref(), &instance);
      let debug_messenger =
        unsafe { dbg_instance.create_debug_utils_messenger(&msg_create_info, None) }?;
      Ok(Self {
        instance,
        debug_messenger,
        entry_wrapper,
        has_surface_maintenance1,
        has_headless_surface,
      })
    }
    #[cfg(not(debug_assertions))]
    Ok(Self {
      instance,
      entry_wrapper,
      has_surface_maintenance1,
      has_headless_surface,
    })
  }

  /// TODO: Document this item
  pub fn api_version(&self) -> u32 {
    // TODO if changed
    vk::API_VERSION_1_1
  }

  /// TODO: Document this item
  pub fn get_eligible_devices(
    &self,
    query_input: &utils::PhysicalDeviceQueryInput,
  ) -> GpuResult<Vec<utils::PhysicalDeviceQueryResult>> {
    let entry = self.entry_wrapper.weak_entry().upgrade().ok_or(GpuError::DeviceLost)?;
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
        // a. properties (TODO: Subgroup Information)
        let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
        let mut descriptor_indexing_props = vk::PhysicalDeviceDescriptorIndexingProperties::default();
        let (subgroup_size, is_cpu, max_per_stage_samplers, max_set_samplers) = {
          // `props` mutably borrows `subgroup_props` via push_next().  The block
          // scope ends the borrow so we can read `subgroup_props.subgroup_size` next.
          let mut props = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut subgroup_props)
            .push_next(&mut descriptor_indexing_props);
          unsafe { self.instance.get_physical_device_properties2(physical_device, &mut props) };
          // Cache device type; block scope ends props borrow here.
          let dev_type = props.properties.device_type;
          (
            subgroup_props.subgroup_size,
            dev_type == vk::PhysicalDeviceType::CPU,
            descriptor_indexing_props.max_per_stage_descriptor_update_after_bind_samplers,
            descriptor_indexing_props.max_descriptor_set_update_after_bind_samplers,
          )
        };
        // TODO log

        // b. supported queue families
        let queue_family_properties_len = unsafe {
          self.instance.get_physical_device_queue_family_properties2_len(physical_device)
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
                &self.instance,
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
        let device_extension_properties =
          unsafe { self.instance.enumerate_device_extension_properties(physical_device) }.ok()?;

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
        unsafe { self.instance.get_physical_device_features2(physical_device, &mut features2) };
        required_features.features = features2.features;
        if required_features.any_missing().is_some() {
          // TODO Log
          return None;
        }

        // e. device is valid, compute its score
        let score: i32 = if is_cpu {
          1 // CPU device (Lavapipe) — lowest preference
        } else {
          // Query properties again using the non-pNext variant (safe: props was dropped above)
          let props2 = unsafe { self.instance.get_physical_device_properties(physical_device) };
          match props2.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 100,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 50,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 20,
            _ => 1,
          }
        };

        // f. TODO: optional extension and features bookkeeping and score increase/decrease
        let mut optional_extensions = utils::OptionalExtensionSupportFlags::NONE;

        let supports_swapchain_maintenance1 = device_extension_properties.iter().any(|prop| {
          prop.extension_name_as_c_str().unwrap() == ash::ext::swapchain_maintenance1::NAME
        });

        if supports_swapchain_maintenance1 && self.has_surface_maintenance1 {
          optional_extensions.insert(utils::OptionalExtensionSupportFlags::SWAPCHAIN_MAINTENANCE1);
        }

        Some(utils::PhysicalDeviceQueryResult {
          physical_device,
          physical_device_properties: unsafe {
            self.instance.get_physical_device_properties(physical_device)
          },
          family_count: queue_family_properties_len,
          optional_extensions,
          graphics_queue_family_index,
          compute_queue_family_index,
          transfer_queue_family_index,
          subgroup_size,
          is_cpu,
          score,
          debug_shaders: query_input.debug_shaders,
          max_per_stage_descriptor_update_after_bind_samplers: max_per_stage_samplers,
          max_descriptor_set_update_after_bind_samplers: max_set_samplers,
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
