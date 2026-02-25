extern crate core;
extern crate alloc;

use core::{ffi::{CStr, c_void}, mem, ptr};
use alloc::{vec, string};

use ash::{
  Entry, vk::{self, PFN_vkGetInstanceProcAddr}
};

#[cfg(windows)]
use windows::{
  core::{w, s},
  Win32::System::LibraryLoader::{LoadLibraryExW, GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32},
  core::PCSTR,
};

// -------------------------------- Debug essenger --------------------------
// TODO: copy from mac
// TODO: Printer
#[cfg(debug_assertions)]
pub(super) unsafe extern "system" fn debug_utils_messenger_user_callback(_message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,  _message_types: vk::DebugUtilsMessageTypeFlagsEXT, p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>, _p_user_data: *mut c_void) -> vk::Bool32 {
  #[cfg(windows)]
  {
    use windows::Win32::System::Console::{ GetStdHandle, STD_OUTPUT_HANDLE };

    let h_stdout=  unsafe { GetStdHandle(STD_OUTPUT_HANDLE) }.unwrap();

  }
  #[cfg(not(windows))]
  {
    todo!();
  }

  // don't abort Vulkan call
  vk::FALSE
}

// -------------------------------- Startup Functions --------------------------

// TODO Runtime Callback on crash
/// ## Safety:
/// - as long as we can guarantee that the vulkan loader exists and its loaded counterpart is not tampered with, the returned pointer is non null and working entry
pub(super) unsafe fn vk_entry() -> &'static Entry {
  static ENTRY: spin::Once<Entry> = spin::Once::new();
  ENTRY.call_once(|| {
    let static_fn: ash::StaticFn = {
      let get_instance_proc_addr: PFN_vkGetInstanceProcAddr;
      #[cfg(windows)]
      {
        const VK_GET_INSTANCE_PROC_ADDR_NAME: PCSTR = s!("vkGetInstanceProcAddr");
        let h_vulkan =
          unsafe { LoadLibraryExW(w!("vulkan-1.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32) }
            .unwrap();
        let vk_get_instance_proc_addr =
          unsafe { GetProcAddress(h_vulkan, VK_GET_INSTANCE_PROC_ADDR_NAME) }
            .expect("vkGetInstanceProcAddr not found");
        get_instance_proc_addr = unsafe { mem::transmute(vk_get_instance_proc_addr) };
        ash::StaticFn {
          get_instance_proc_addr,
        }
      }
      #[cfg(target_os = "linux")]
      {
        todo!()
      }
      #[cfg(target_os = "macos")]
      {
        todo!()
      }
    };

    unsafe { Entry::from_static_fn(static_fn) }
  })
}

pub(super) fn required_instance_extensions() -> &'static vec::Vec<&'static CStr> {
  static INSTANCE_EXTENSIONS: spin::Once<vec::Vec<&'static CStr>> = spin::Once::new();
  INSTANCE_EXTENSIONS.call_once(|| {
    let mut the_vec = vec::Vec::with_capacity(64);
    #[cfg(debug_assertions)]
    {
      the_vec.push(ash::ext::debug_utils::NAME);
    }
    // surface
    the_vec.push(ash::khr::surface::NAME);
    the_vec.push(ash::khr::get_surface_capabilities2::NAME);
    #[cfg(windows)]
    {
      the_vec.push(ash::khr::win32_surface::NAME);
    }
    #[cfg(not(windows))]
    {
      todo!();
    }
    // colorspaces
    the_vec.push(ash::ext::swapchain_colorspace::NAME);

    the_vec
  })
}

pub(super) fn required_device_extensions() -> &'static vec::Vec<&'static CStr> {
  static DEVICE_EXTENSIONS: spin::Once<vec::Vec<&'static CStr>> = spin::Once::new();
  DEVICE_EXTENSIONS.call_once(|| {
    let mut the_vec = vec::Vec::with_capacity(64);
    // basic and shader stuff
    the_vec.push(ash::khr::timeline_semaphore::NAME);
    the_vec.push(ash::khr::buffer_device_address::NAME);
    the_vec.push(ash::khr::vulkan_memory_model::NAME);
    // maintenance4, not promoted to Vulkan 1.1, adds SPIR-V 1.2
    the_vec.push(ash::khr::maintenance4::NAME);
    the_vec.push(ash::khr::shader_float_controls::NAME); // req. for SPIR-V 1.4
    the_vec.push(ash::khr::spirv_1_4::NAME);
    // more fine grained synchronization stages. Also necessary for sync2 layer
    the_vec.push(ash::khr::synchronization2::NAME);

    #[cfg(windows)]
    {
      // external `HANDLE` stuff
      the_vec.push(ash::khr::external_fence_win32::NAME);
      the_vec.push(ash::khr::external_memory_win32::NAME);
      the_vec.push(ash::khr::external_semaphore_win32::NAME);
      // TODO: keyed mutex?
      // TODO: if not Windows 11, need fullscreen exclusive extension
    }
    #[cfg(not(windows))]
    {
      todo!();
    }

    // extensions for VMA (memory budget also important for out-of-core/streaming)
    the_vec.push(ash::ext::memory_budget::NAME);
    the_vec.push(ash::khr::dedicated_allocation::NAME);

    // https://docs.vulkan.org/samples/latest/samples/extensions/descriptor_indexing/README.html
    // flexibility in update after bind and non-uniform indexing
    the_vec.push(ash::ext::descriptor_indexing::NAME);

    // TODO: pipeline extensions after sampling for desktop device support

    the_vec
  })
}

// -------------------------------- Device Features Handling -------------------
#[derive(Copy, Clone, Debug)]
pub(super) struct RequiredFeatures<'a> {
  pub buffer_device_address: vk::PhysicalDeviceBufferDeviceAddressFeatures<'a>,
  pub vulkan_memory_model: vk::PhysicalDeviceVulkanMemoryModelFeatures<'a>,
  pub timeline_semaphore: vk::PhysicalDeviceTimelineSemaphoreFeatures<'a>,
  // TODO add VK_KHR_variable_pointers (promoted to 1.1)
}

impl RequiredFeatures<'_> {
  pub fn new() -> Self {
    let buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
    let vulkan_memory_model = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
    let timeline_semaphore = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();

    Self {
      buffer_device_address,
      vulkan_memory_model,
      timeline_semaphore,
    }
  }

  pub fn as_features2(&mut self) -> vk::PhysicalDeviceFeatures2<'_> {
    vk::PhysicalDeviceFeatures2::default()
      .push_next(&mut self.buffer_device_address)
      .push_next(&mut self.vulkan_memory_model)
      .push_next(&mut self.timeline_semaphore)
  }

  pub fn populate(&mut self) -> &mut Self {
    self.buffer_device_address.buffer_device_address = vk::TRUE;
    self.vulkan_memory_model.vulkan_memory_model = vk::TRUE;
    self.timeline_semaphore.timeline_semaphore = vk::TRUE;

    self
  }

  pub fn any_missing(&self) -> Option<vec::Vec<string::String>> {
    use string::ToString;
    let mut the_vec = vec::Vec::with_capacity(64);
    if self.buffer_device_address.buffer_device_address != vk::TRUE {
      the_vec.push("buffer_device_address".to_string());
    }
    if self.vulkan_memory_model.vulkan_memory_model != vk::TRUE {
      the_vec.push("vulkan_memory_model".to_string());
    }
    if self.timeline_semaphore.timeline_semaphore != vk::TRUE {
      the_vec.push("timeline_semaphore".to_string());
    }

    if the_vec.is_empty() { None } else { Some(the_vec) }
  }
}

// -------------------------------- Unit Testing -------------------------------

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_vulkan_entry() {
    let the_ptr = unsafe { vk_entry() };
    let proc_addr_value = unsafe {
      mem::transmute::<PFN_vkGetInstanceProcAddr, *const core::ffi::c_void>(
        (&the_ptr).static_fn().get_instance_proc_addr,
      )
    };
    assert!(!proc_addr_value.is_null());
    assert!(unsafe { the_ptr.enumerate_instance_layer_properties() }.is_ok());
  }
}
