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

/// Which Linux Vulkan surface extensions were successfully enabled for this instance.
/// Populated after filtering the desired extensions against what the driver/layers expose.
#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct LinuxSurfaceSupport {
  pub wayland: bool,
  pub xcb: bool,
  pub xlib: bool,
}

/// TODO: Document this item
pub struct Instance {
  pub instance: ash::Instance,
  #[cfg(debug_assertions)]
  debug_messenger: vk::DebugUtilsMessengerEXT,

  pub entry_wrapper: utils::EntryWrapper,
  pub has_surface_maintenance1: bool,
  pub has_headless_surface: bool,
  #[cfg(target_os = "linux")]
  pub linux_surface_support: LinuxSurfaceSupport,
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

    // Now check support against the merged pool and filter out unsupported extensions
    // This allows fallback (e.g. Wayland missing under RenderDoc, but X11 available)
    desired_instance_extensions.retain(|&ext| {
      let is_supported = available_extensions
        .iter()
        .any(|avail| unsafe { CStr::from_ptr(avail.extension_name.as_ptr()) } == ext);
      if !is_supported {
        aethervk_oshal_rlib::log!(
          "Warning: Requested instance extension {:?} is not supported. Ignoring.",
          ext
        );
      }
      is_supported
    });

    // Record which Linux surface extensions survived the filter
    #[cfg(target_os = "linux")]
    let mut linux_surface_support = LinuxSurfaceSupport {
      wayland: desired_instance_extensions.contains(&ash::khr::wayland_surface::NAME),
      xcb: desired_instance_extensions.contains(&ash::khr::xcb_surface::NAME),
      xlib: desired_instance_extensions.contains(&ash::khr::xlib_surface::NAME),
    };
    #[cfg(target_os = "linux")]
    aethervk_oshal_rlib::log!(
      "Linux surface support: wayland={} xcb={} xlib={}",
      linux_surface_support.wayland,
      linux_surface_support.xcb,
      linux_surface_support.xlib
    );

    // =========================================================================
    // 3. Setup Validation Features & Debug Messenger
    // =========================================================================
    #[cfg(debug_assertions)]
    let mut printf_features = alloc::vec![];

    #[cfg(any(debug_assertions, test))]
    {
      let is_apple = cfg!(target_vendor = "apple");
      let disable_requested =
        aethervk_oshal_rlib::os::env::var("AETHERVK_DISABLE_GPU_AV").is_some();
      let enable_requested = aethervk_oshal_rlib::os::env::var("AETHERVK_ENABLE_GPU_AV").is_some();

      #[cfg(not(target_vendor = "apple"))]
      let use_printf = crate::gpu_backends::vulkan::physics::USE_PRINTF_SHADERS
        .load(core::sync::atomic::Ordering::Relaxed);
      #[cfg(target_vendor = "apple")]
      let use_printf = false;

      #[cfg(test)]
      let is_test = true;
      #[cfg(not(test))]
      let is_test = false;

      // By default, tests use GPU-AV (unless USE_PRINTF is on). Non-tests don't.
      let mut wants_gpu_av = (is_test || enable_requested) && !use_printf;
      let wants_printf = (!is_test && !wants_gpu_av) || use_printf;

      if is_apple || disable_requested {
        aethervk_oshal_rlib::log!("Disabling GPU-Assisted/Printf Validation.");
        wants_gpu_av = false;
      }

      // Filter Lavapipe / CPU-only out of GPU-AV automatically
      if wants_gpu_av {
        let dummy_ci = vk::InstanceCreateInfo::default().application_info(&app_info);
        if let Ok(dummy_inst) = unsafe { vk_entry.create_instance(&dummy_ci, None) } {
          if let Ok(pdevs) = unsafe { dummy_inst.enumerate_physical_devices() }
            && !pdevs.is_empty()
          {
            let mut all_cpu = true;
            for pdev in pdevs {
              let props = unsafe { dummy_inst.get_physical_device_properties(pdev) };
              if props.device_type != vk::PhysicalDeviceType::CPU
                && props.device_type != vk::PhysicalDeviceType::OTHER
              {
                all_cpu = false;
                break;
              }
            }
            if all_cpu {
              aethervk_oshal_rlib::log!(
                "Detected only CPU devices (e.g. Lavapipe). Force-disabling GPU-AV."
              );
              wants_gpu_av = false;
            }
          }

          unsafe { dummy_inst.destroy_instance(None) };
        }
      }

      if wants_gpu_av {
        printf_features.push(vk::ValidationFeatureEnableEXT::GPU_ASSISTED);
        printf_features.push(vk::ValidationFeatureEnableEXT::GPU_ASSISTED_RESERVE_BINDING_SLOT);
      } else if wants_printf && !is_apple && !disable_requested {
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
    // We attach a SAFE creation-phase debug messenger via pNext (see
    // `utils::creation_phase_debug_callback` for why it is separate from the real one).
    // It logs extension-compat messages from tools like RenderDoc without panicking
    // through FFI, and stores genuine validation errors so we can panic AFTER
    // vkCreateInstance returns to Rust. The real panicking messenger is created
    // separately below, after the instance exists.

    #[cfg(debug_assertions)]
    let enabled_layers = Vec::from_iter(desired_layer_names.iter().map(|&c_str| c_str.as_ptr()));

    // State for the creation-phase callback — lives on the stack for the duration
    // of all vkCreateInstance attempts.
    #[cfg(debug_assertions)]
    let mut creation_state = utils::InstanceCreationState {
      user_error_callback: validation_error_callback,
      had_validation_error: false,
      validation_error_message: alloc::string::String::new(),
    };

    // -------------------------------------------------------------------------
    // Retry loop: if vkCreateInstance fails with VK_ERROR_EXTENSION_NOT_PRESENT,
    // strip Linux surface extensions one by one (Wayland → XCB → Xlib) and retry.
    // This handles tools like RenderDoc that don't support Wayland surfaces but
    // don't correctly filter the extension from vkEnumerateInstanceExtensionProperties.
    // -------------------------------------------------------------------------
    #[cfg(target_os = "linux")]
    let linux_surface_retry_order: &[&CStr] = &[
      ash::khr::wayland_surface::NAME,
      ash::khr::xcb_surface::NAME,
      ash::khr::xlib_surface::NAME,
    ];

    // =========================================================================
    // RENDERDOC DEADLOCK WORKAROUND
    // =========================================================================
    // If RenderDoc fails an instance creation, it corrupts the loader's mutexes.
    // We cannot rely on the retry loop. We must forcefully strip Wayland upfront.
    let is_renderdoc = aethervk_oshal_rlib::os::env::var("VK_INSTANCE_LAYERS")
      .map(|v| v.contains("RENDERDOC"))
      .unwrap_or(false)
      || desired_layer_names.iter().any(|&l| l == c"VK_LAYER_RENDERDOC_Capture");

    if is_renderdoc {
      aethervk_oshal_rlib::log!(
        "RenderDoc detected! Proactively stripping Wayland to prevent loader deadlock."
      );
      desired_instance_extensions.retain(|&ext| ext != ash::khr::wayland_surface::NAME);
      #[cfg(target_os = "linux")]
      {
        linux_surface_support.wayland = false;
      }
    }

    let instance = 'create: {
      loop {
        // Reset per-attempt state for the creation-phase callback.
        #[cfg(debug_assertions)]
        {
          creation_state.had_validation_error = false;
          creation_state.validation_error_message = alloc::string::String::new();
        }

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

        // debug_assertions: attach the safe creation-phase messenger via pNext.
        // The `creation_msg_info` must be declared here (inside the loop scope) so
        // that ash's lifetime-tied pNext builder can borrow it for the create call.
        // The borrow ends when `vk_entry.create_instance` returns.
        #[cfg(debug_assertions)]
        let result = {
          let mut creation_msg_info = msg_create_info
            .user_data(&mut creation_state as *mut _ as *mut core::ffi::c_void)
            .pfn_user_callback(Some(utils::creation_phase_debug_callback));

          let mut ci = instance_create_info
            .enabled_layer_names(&enabled_layers)
            .push_next(&mut creation_msg_info);
          if has_validation_features {
            ci = ci.push_next(&mut validation_features);
          }

          unsafe { vk_entry.create_instance(&ci, None) }
        };

        #[cfg(not(debug_assertions))]
        let result = unsafe { vk_entry.create_instance(&instance_create_info, None) };

        match result {
          Ok(inst) => {
            // If the creation-phase callback captured a real (non-extension-compat)
            // validation error, call the user callback NOW — safely, in Rust, not through FFI.
            #[cfg(debug_assertions)]
            if creation_state.had_validation_error {
              if let Some(cb) = creation_state.user_error_callback {
                cb(&creation_state.validation_error_message);
              }
            }
            break 'create inst;
          }

          #[cfg(target_os = "linux")]
          Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT) => {
            // Strip the first Linux surface extension that's still in the list and retry.
            let mut stripped = false;
            for &candidate in linux_surface_retry_order {
              if desired_instance_extensions.contains(&candidate) {
                aethervk_oshal_rlib::log!(
                  "vkCreateInstance: VK_ERROR_EXTENSION_NOT_PRESENT — retrying without {:?}",
                  candidate
                );
                desired_instance_extensions.retain(|&e| e != candidate);
                stripped = true;
                break;
              }
            }
            if !stripped {
              return Err(GpuError::UnsupportedFeatureNamed(
                "vkCreateInstance failed with VK_ERROR_EXTENSION_NOT_PRESENT after exhausting \
                 all Linux surface extension fallbacks"
                  .to_string(),
              ));
            }
            // Update linux_surface_support to reflect what's still enabled.
            linux_surface_support = LinuxSurfaceSupport {
              wayland: desired_instance_extensions.contains(&ash::khr::wayland_surface::NAME),
              xcb: desired_instance_extensions.contains(&ash::khr::xcb_surface::NAME),
              xlib: desired_instance_extensions.contains(&ash::khr::xlib_surface::NAME),
            };
            aethervk_oshal_rlib::log!(
              "Linux surface support after retry: wayland={} xcb={} xlib={}",
              linux_surface_support.wayland,
              linux_surface_support.xcb,
              linux_surface_support.xlib
            );
            // continue loop with reduced extension list
          }

          Err(e) => return Err(e.into()),
        }
      }
    };

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
        #[cfg(target_os = "linux")]
        linux_surface_support,
      })
    }
    #[cfg(not(debug_assertions))]
    Ok(Self {
      instance,
      entry_wrapper,
      has_surface_maintenance1,
      has_headless_surface,
      #[cfg(target_os = "linux")]
      linux_surface_support,
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

    let mut active_query = query_input.clone();
    #[cfg(target_os = "linux")]
    {
      active_query.linux_surface_support = self.linux_surface_support;
    }

    // 2. filter those which are eligible and map to query result
    let mut eligible_devices =
      Vec::from_iter(physical_devices.iter().filter_map(|&physical_device| {
        // a. properties (TODO: Subgroup Information)
        let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
        let mut descriptor_indexing_props =
          vk::PhysicalDeviceDescriptorIndexingProperties::default();
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
          core::iter::repeat_with(vk::QueueFamilyProperties2::default)
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
              && active_query.supports_presentation(
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
        if let Some(missing) = required_features.any_missing() {
          aethervk_oshal_rlib::log!("Device missing features: {:?}", missing);
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

        // f. Optional extension and features bookkeeping
        let mut optional_extensions = utils::OptionalExtensionSupportFlags::NONE;

        let supports_swapchain_maintenance1 = device_extension_properties.iter().any(|prop| {
          prop.extension_name_as_c_str().unwrap() == ash::ext::swapchain_maintenance1::NAME
        });

        if supports_swapchain_maintenance1 && self.has_surface_maintenance1 {
          optional_extensions.insert(utils::OptionalExtensionSupportFlags::SWAPCHAIN_MAINTENANCE1);
        }

        // shaderFloat16 is queried (not requested) — absent on Pascal/GTX10xx.
        // required_features was already populated by get_physical_device_features2 above.
        if required_features.shader_float16_int8.shader_float16 == ash::vk::TRUE {
          optional_extensions.insert(utils::OptionalExtensionSupportFlags::NATIVE_FLOAT16);
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

    // 3. sort on score and return (descending, highest score first)
    eligible_devices.sort_by(|a, b| b.score.cmp(&a.score));
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