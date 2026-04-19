use core::{
  char::MAX,
  ffi::{CStr, c_char, c_void},
  mem, ops, ptr,
};
use alloc::{
  string::{self, ToString},
  sync::{Arc, Weak},
  vec::Vec,
};

use ash::{
  Entry,
  vk::{self, PFN_vkGetInstanceProcAddr},
};
use bitflags::bitflags;

use itertools::Itertools;
use vk_mem::Alloc;
#[cfg(windows)]
use windows::{
  core::{w, s},
  Win32::System::LibraryLoader::{LoadLibraryExW, GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32},
  core::PCSTR,
};
use crate::{
  gpu::{DeviceAdditionalParams, OpaqueNativeHandleInfo},
  types::{EngineError, EngineResult, GpuError, GpuResult},
};

// -------------------------------- Helper Types -----------------------------
bitflags! {
  #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub(super) struct OptionalExtensionSupportFlags: u64 {
    const NONE = 0;
    // TODO
    const SOME_EXTENSION = 1 << 0;
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PhysicalDeviceQueryInput {
  #[cfg(all(target_os = "linux", feature = "linux_wayland"))]
  wl_display: core::ptr::NonNull<vk::wl_display>,
  #[cfg(all(target_os = "linux", feature = "linux_xcb"))]
  xcb_connection: core::ptr::NonNull<vk::xcb_connection_t>,
  #[cfg(all(target_os = "linux", feature = "linux_xcb"))]
  xcb_visualid: vk::xcb_visualid_t,
  #[cfg(all(target_os = "linux", feature = "linux_xlib"))]
  dpy: core::ptr::NonNull<vk::Display>,
  #[cfg(all(target_os = "linux", feature = "linux_xlib"))]
  visual_id: vk::VisualID,
}
impl PhysicalDeviceQueryInput {
  pub(super) fn from_params(_value: &DeviceAdditionalParams) -> Option<Self> {
    #[cfg(all(target_os = "linux", feature = "linux_wayland"))]
    let wl_display: ptr::NonNull<vk::wl_display> = _value
      .get(&super::DEVICE_ADDIDITIONAL_PARAM_WL_DISPLAY)
      .and_then(|intptr| ptr::NonNull::new((*intptr) as *mut _))?;
    #[cfg(all(target_os = "linux", feature = "linux_xcb"))]
    let xcb_connection: ptr::NonNull<vk::xcb_connection_t> = _value
      .get(&super::DEVICE_ADDIDITIONAL_PARAM_XCB_CONNECTION)
      .and_then(|intptr| ptr::NonNull::new((*intptr) as *mut _))?;
    #[cfg(all(target_os = "linux", feature = "linux_xcb"))]
    let xcb_visualid: vk::xcb_visualid_t =
      *_value.get(&super::DEVICE_ADDIDITIONAL_PARAM_XCB_VISUALID)? as _;
    #[cfg(all(target_os = "linux", feature = "linux_xlib"))]
    let dpy: ptr::NonNull<vk::Display> = _value
      .get(&super::DEVICE_ADDIDITIONAL_PARAM_DPY)
      .and_then(|intptr| ptr::NonNull::new((*intptr) as *mut _))?;
    #[cfg(all(target_os = "linux", feature = "linux_xlib"))]
    let visual_id: vk::VisualID = *_value.get(&super::DEVICE_ADDIDITIONAL_PARAM_VISUAL_ID)? as _;

    Some(Self {
      #[cfg(all(target_os = "linux", feature = "linux_wayland"))]
      wl_display,
      #[cfg(all(target_os = "linux", feature = "linux_xcb"))]
      xcb_connection,
      #[cfg(all(target_os = "linux", feature = "linux_xcb"))]
      xcb_visualid,
      #[cfg(all(target_os = "linux", feature = "linux_xlib"))]
      dpy,
      #[cfg(all(target_os = "linux", feature = "linux_xlib"))]
      visual_id,
    })
  }

  pub(super) fn supports_presentation(
    &self,
    _entry: &ash::Entry,
    _physical_device: vk::PhysicalDevice,
    _instance: &ash::Instance,
    _queue_family_index: u32,
  ) -> bool {
    let mut _supported = false;

    #[cfg(any(target_os = "android", target_vendor = "apple"))]
    {
      _supported = true;
    }

    #[cfg(all(target_os = "linux", feature = "linux_wayland"))]
    unsafe {
      _supported = ash::khr::wayland_surface::Instance::new(_entry, &_instance)
        .get_physical_device_wayland_presentation_support(
          _physical_device,
          _queue_family_index,
          self.wl_display,
        );
    }

    #[cfg(all(target_os = "linux", feature = "linux_xcb"))]
    unsafe {
      _supported = ash::khr::xcb_surface::Instance::new(_entry, &_instance)
        .get_physical_device_xcb_presentation_support(
          _physical_device,
          _queue_family_index,
          self.xcb_connection,
          self.xcb_visualid,
        );
    }

    #[cfg(all(target_os = "linux", feature = "linux_xlib"))]
    unsafe {
      _supported = ash::khr::xlib_surface::Instance::new(_entry, &_instance)
        .get_physical_device_xlib_presentation_support(
          _physical_device,
          _queue_family_index,
          self.dpy,
          self.visual_id,
        );
    }

    #[cfg(windows)]
    unsafe {
      _supported = ash::khr::win32_surface::Instance::new(_entry, &_instance)
        .get_physical_device_win32_presentation_support(_physical_device, _queue_family_index);
    }

    _supported
  }
}

/// queue families we are interested in: GRAPHICS, COMPUTE, TRANSFER.
/// best case, they are all different
pub(super) const MAX_QUEUE_FAMILY_COUNT: usize = 4;

#[derive(Debug, Clone, Copy)]
pub(super) struct PhysicalDeviceQueryResult {
  pub physical_device: vk::PhysicalDevice,
  pub physical_device_properties: vk::PhysicalDeviceProperties,
  pub family_count: usize,
  pub optional_extensions: OptionalExtensionSupportFlags,
  pub graphics_queue_family_index: u32,
  pub compute_queue_family_index: u32,
  pub transfer_queue_family_index: u32,
  pub score: i32,
}

impl PhysicalDeviceQueryResult {
  pub(super) fn has_valid_score(&self) -> bool {
    self.score > 0
  }

  pub(super) fn family_count(&self) -> usize {
    self.family_count
  }

  pub(super) fn unique_family_indices_set(
    &self,
  ) -> heapless::index_set::FnvIndexSet<u32, MAX_QUEUE_FAMILY_COUNT> {
    let mut unique_queue_families = heapless::index_set::FnvIndexSet::new();
    unique_queue_families
      .insert(self.graphics_queue_family_index)
      .unwrap();
    unique_queue_families
      .insert(self.compute_queue_family_index)
      .unwrap();
    unique_queue_families
      .insert(self.transfer_queue_family_index)
      .unwrap();

    unique_queue_families
  }

  pub(super) fn used_family_count(&self) -> usize {
    let mut families = [
      self.graphics_queue_family_index,
      self.compute_queue_family_index,
      self.transfer_queue_family_index,
    ];

    families.sort_unstable();
    families.into_iter().dedup().count()
  }

  pub(super) fn enabled_extension_names(&self) -> Vec<*const c_char> {
    let the_vec = required_device_extensions()
      .iter()
      .map(|cstr| cstr.as_ptr())
      .collect();
    // TODO optional extensions if needed

    the_vec
  }
}

// -------------------------------- Debug Messenger --------------------------
// TODO: copy from mac
// TODO: Printer
#[cfg(debug_assertions)]
pub(super) unsafe extern "system" fn debug_utils_messenger_user_callback(
  _message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
  _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
  p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
  _p_user_data: *mut c_void,
) -> vk::Bool32 {
  #[cfg(windows)]
  {
    use aethervk_oshal_rlib::log;

    let p_msg = unsafe { (*p_callback_data).p_message };
    let msg = unsafe { core::ffi::CStr::from_ptr(p_msg) };
    log!("[Vulkan Messenger]: {:?}", msg);
  }
  #[cfg(target_family = "unix")]
  {
    let p_msg = unsafe { (*p_callback_data).p_message };
    let count = unsafe { libc::strlen(p_msg) };
    unsafe { libc::write(libc::STDOUT_FILENO, p_msg.cast(), count) };
  }

  // don't abort Vulkan call
  vk::FALSE
}

// -------------------------------- Startup Functions --------------------------
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) struct VkLibHandle(pub ptr::NonNull<c_void>);
// safety: Ensure thread safety on this, achieved through checking ref count in Drop
unsafe impl Send for VkLibHandle {}
unsafe impl Sync for VkLibHandle {}

#[derive(Clone)]
pub(super) struct EntryWrapper {
  vk_entry: Arc<ash::Entry>,
  vulkan_loader_module: VkLibHandle,
}

impl EntryWrapper {
  pub(super) fn weak_entry(&self) -> Weak<ash::Entry> {
    Arc::downgrade(&self.vk_entry)
  }

  pub(super) fn new(base_path_override: Option<&CStr>) -> GpuResult<Self> {
    let mut vulkan_loader_module = ptr::NonNull::dangling();
    let static_fn: ash::StaticFn = {
      let get_instance_proc_addr: PFN_vkGetInstanceProcAddr;
      #[cfg(windows)]
      {
        use windows::Win32::System::LibraryLoader::{
          GetModuleHandleExW, GetProcAddress, LoadLibraryExW,
          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, LOAD_LIBRARY_SEARCH_SYSTEM32,
        };
        use windows::Win32::Foundation::HMODULE;
        use windows::core::{s, w, PCSTR, PCWSTR};

        const VK_GET_INSTANCE_PROC_ADDR_NAME: PCSTR = s!("vkGetInstanceProcAddr");
        const VULKAN_DLL_NAME: PCWSTR = w!("vulkan-1.dll");

        let mut h_vulkan = HMODULE::default();

        // 1. Try to get the handle without incrementing the reference count if it's already loaded
        let handle_exists = unsafe {
          GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            VULKAN_DLL_NAME,
            &mut h_vulkan,
          )
        };

        // 2. If it's not already in the process, load it explicitly from System32
        if handle_exists.is_err() {
          h_vulkan = unsafe { LoadLibraryExW(VULKAN_DLL_NAME, None, LOAD_LIBRARY_SEARCH_SYSTEM32) }
            .map_err(|_| {
              GpuError::BackendSpecific("Failed to load vulkan-1.dll from System32".into())
            })?;
        }

        // 3. Extract the function pointer
        let vk_get_instance_proc_addr_ptr =
          unsafe { GetProcAddress(h_vulkan, VK_GET_INSTANCE_PROC_ADDR_NAME) }.ok_or(
            GpuError::BackendSpecific("vkGetInstanceProcAddr not found in vulkan-1.dll".into()),
          )?;

        get_instance_proc_addr = unsafe { core::mem::transmute(vk_get_instance_proc_addr_ptr) };

        // 4. Store the module handle as a NonNull pointer
        // Note: The cast `as *mut core::ffi::c_void` handles the inner type of HMODULE (.0),
        // which varies slightly depending on your exact `windows` crate version (isize vs *mut c_void).
        vulkan_loader_module = core::ptr::NonNull::new(h_vulkan.0 as *mut core::ffi::c_void)
          .expect("Vulkan module handle was null");
      }
      #[cfg(target_os = "linux")]
      {
        todo!()
      }
      #[cfg(target_os = "macos")]
      {
        static ZERO: isize = 0;
        // find the current cdylib directory, assuming the packaged vulkan stuff is in there.
        // how: take this function and query which shared module is it contained in
        let mut info = unsafe { core::mem::zeroed::<libc::Dl_info>() };
        let func_addr = Self::new as *const core::ffi::c_void;
        get_instance_proc_addr = if let Some(path) = base_path_override {
          // TODO check if exists?
          Some(path)
        } else if unsafe { libc::dladdr(func_addr, &mut info) } != 0 && !info.dli_fname.is_null() {
          // TODO: move elsewhere and see how to fix the addition of "vulkan"
          let cstr_name_ref = unsafe { core::ffi::CStr::from_ptr(info.dli_fname.cast::<i8>()) };
          Some(cstr_name_ref)
        } else {
          None
        }
        .and_then(|base_path| {
          use aethervk_oshal_rlib::os::fs::{FileSystemObject, Path};

          // 1. Extract base directory bytes, retaining the trailing slash
          let base_bytes = base_path.to_bytes();
          let dir_bytes = match base_bytes.iter().rposition(|&b| b == b'/') {
            Some(pos) => &base_bytes[..=pos],
            None => base_bytes,
          };

          // 2. Build local paths natively via slice concatenation
          let local_layer =
            alloc::ffi::CString::new([dir_bytes, b"vulkan/explicit_layer"].concat()).unwrap();
          let local_icd = alloc::ffi::CString::new([dir_bytes, b"vulkan/icd"].concat()).unwrap();
          let local_loader =
            alloc::ffi::CString::new([dir_bytes, b"vulkan/lib/libvulkan.dylib"].concat()).unwrap();

          let local_layer_obj = unsafe {
            Path::from_slice(core::slice::from_raw_parts(
              local_layer.as_ptr().cast::<core::ffi::c_char>(),
              local_layer.as_bytes().len(),
            ))
          };

          let local_icd_obj = unsafe {
            Path::from_slice(core::slice::from_raw_parts(
              local_icd.as_ptr().cast::<core::ffi::c_char>(),
              local_icd.as_bytes().len(),
            ))
          };

          // 3. Determine final paths (Bundle vs VULKAN_SDK Fallback)
          let paths_opt = if local_layer_obj.is_dir() && local_icd_obj.is_dir() {
            Some((local_layer, local_icd, local_loader))
          } else {
            #[cfg(debug_assertions)]
            {
              aethervk_oshal_rlib::log!("Trying to load vulkan from VULKAN_SDK");
              let env_ptr = unsafe { libc::getenv(b"VULKAN_SDK\0".as_ptr().cast()) };
              core::ptr::NonNull::new(env_ptr).map(|ptr| {
                let sdk_cstr = unsafe { core::ffi::CStr::from_ptr(ptr.as_ptr()) };
                let sdk_bytes = sdk_cstr.to_bytes();

                // Ensure single slash separation
                let slash = if sdk_bytes.last() == Some(&b'/') {
                  b"".as_slice()
                } else {
                  b"/".as_slice()
                };

                // Typical LunarG Vulkan SDK layout on macOS
                let sdk_layer = alloc::ffi::CString::new(
                  [sdk_bytes, slash, b"share/vulkan/explicit_layer.d"].concat(),
                )
                .unwrap();
                let sdk_icd =
                  alloc::ffi::CString::new([sdk_bytes, slash, b"share/vulkan/icd.d"].concat())
                    .unwrap();
                let sdk_loader =
                  alloc::ffi::CString::new([sdk_bytes, slash, b"lib/libvulkan.dylib"].concat())
                    .unwrap();

                (sdk_layer, sdk_icd, sdk_loader)
              })
            }

            #[cfg(not(debug_assertions))]
            {
              None
            }
          };
          #[cfg(debug_assertions)]
          {
            aethervk_oshal_rlib::log!("paths found {:?}", paths_opt);
          }

          // Early out (return None from the .and_then closure) if neither paths exist
          let (vk_layer_path, vk_icd_path, vk_loader_path) = paths_opt?;

          // 4. Set environment variables
          unsafe {
            libc::setenv(
              b"VK_LAYER_PATH\0".as_ptr().cast(),
              vk_layer_path.as_ptr(),
              1,
            );
            libc::setenv(
              b"VK_DRIVER_FILES\0".as_ptr().cast(),
              vk_icd_path.as_ptr(),
              1,
            );
          };

          // 5. Load vulkan function
          let mut lib =
            unsafe { libc::dlopen(vk_loader_path.as_ptr(), libc::RTLD_LAZY | libc::RTLD_NOLOAD) };
          if lib.is_null() {
            lib =
              unsafe { libc::dlopen(vk_loader_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
          }
          core::ptr::NonNull::new(lib)
        })
        .and_then(|module_ptr| unsafe {
          vulkan_loader_module = module_ptr;
          core::ptr::NonNull::new(libc::dlsym(
            module_ptr.as_ptr(),
            b"vkGetInstanceProcAddr\0".as_ptr().cast(),
          ))
        })
        .map(|func_addr| unsafe { core::mem::transmute(func_addr) })
        .unwrap_or(unsafe { core::mem::transmute(ZERO) });
      }

      ash::StaticFn {
        get_instance_proc_addr,
      }
    };

    let vk_entry = if 0usize
      == unsafe {
        mem::transmute::<PFN_vkGetInstanceProcAddr, usize>(static_fn.get_instance_proc_addr)
      } {
      Err(GpuError::BackendSpecific(
        "vulkan loader couldn't be loaded".to_string(),
      ))
    } else {
      Ok(unsafe { Entry::from_static_fn(static_fn) })
    }?;

    Ok(Self {
      vk_entry: Arc::new(vk_entry),
      vulkan_loader_module: VkLibHandle(vulkan_loader_module),
    })
  }
}

impl Drop for EntryWrapper {
  fn drop(&mut self) {
    if 1 == Arc::strong_count(&self.vk_entry) {
      #[cfg(windows)]
      {
        unsafe {
          use windows::Win32::Foundation::{FreeLibrary, HMODULE};
          // TODO: Debug logging on failure?
          let _ = FreeLibrary(HMODULE(self.vulkan_loader_module.0.as_ptr()));
        };
      }
      #[cfg(target_family = "unix")]
      {
        unsafe { libc::dlclose(self.vulkan_loader_module.0.as_ptr()) };
      }
    }
  }
}

pub(super) fn required_instance_extensions() -> &'static Vec<&'static CStr> {
  static INSTANCE_EXTENSIONS: spin::Once<Vec<&'static CStr>> = spin::Once::new();
  INSTANCE_EXTENSIONS.call_once(|| {
    let mut the_vec = Vec::with_capacity(64);
    #[cfg(debug_assertions)]
    {
      the_vec.push(ash::ext::debug_utils::NAME);
      // Renderdoc intercepts this layer extension (from VVL) hence make it optional?
      // the_vec.push(ash::ext::layer_settings::NAME);
    }

    // surface
    the_vec.push(ash::khr::surface::NAME);
    the_vec.push(ash::khr::get_surface_capabilities2::NAME);
    #[cfg(windows)]
    {
      the_vec.push(ash::khr::win32_surface::NAME);
    }
    #[cfg(all(target_vendor = "apple", target_family = "unix"))]
    {
      the_vec.push(ash::khr::portability_enumeration::NAME);
      the_vec.push(ash::ext::metal_surface::NAME);
    }
    #[cfg(target_os = "linux")]
    {
      todo!();
    }
    // colorspaces
    the_vec.push(ash::ext::swapchain_colorspace::NAME);

    the_vec
  })
}

pub(super) fn required_device_extensions() -> &'static Vec<&'static CStr> {
  static DEVICE_EXTENSIONS: spin::Once<Vec<&'static CStr>> = spin::Once::new();
  DEVICE_EXTENSIONS.call_once(|| {
    let mut the_vec = Vec::with_capacity(64);
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
    the_vec.push(ash::khr::create_renderpass2::NAME);

    the_vec.push(ash::khr::swapchain::NAME);

    // maintenance1 -> promoted to vk::API_VERSION_1_1
    // allows for vk::Viewport negative height, flipping +Y axis to point upwards

    // promoted to vk::API_VERSION_1_3
    // This extension allows the use of the SPV_KHR_non_semantic_info extension in SPIR-V shader modules. (eg printf)
    the_vec.push(ash::khr::shader_non_semantic_info::NAME);

    // TODO: put this into an optional extension, as the `nullDescriptors` feature is not supported
    // by everybody (namely, Apple M4)
    // This extension also adds support for “null descriptors”, where VK_NULL_HANDLE can be used
    // instead of a valid handle. Accesses to null descriptors have well-defined behavior, and do not rely on robustness.
    // promoted to vk::API_VERSION_1_3
    // the_vec.push(ash::ext::robustness2::NAME);

    #[cfg(windows)]
    {
      // external `HANDLE` stuff
      the_vec.push(ash::khr::external_fence_win32::NAME);
      the_vec.push(ash::khr::external_memory_win32::NAME);
      the_vec.push(ash::khr::external_semaphore_win32::NAME);
      // TODO: keyed mutex?
      // TODO: if not Windows 11, need fullscreen exclusive extension
    }
    #[cfg(all(target_vendor = "apple", target_family = "unix"))]
    {
      // Metal is not 100% Vulkan Spec conformant
      the_vec.push(ash::khr::portability_subset::NAME);
      the_vec.push(ash::ext::metal_objects::NAME);
    }
    #[cfg(target_os = "linux")]
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

// -------------------------------- Extensions Handling ------------------------
pub(super) fn first_unsupported_extension<'a>(
  desired_names: &'a [&'_ CStr],
  properties: &'a [vk::ExtensionProperties],
) -> Option<&'a CStr> {
  for desired_name in desired_names {
    if let None = properties
      .iter()
      .find(|&prop| *desired_name == prop.extension_name_as_c_str().unwrap())
    {
      return Some(desired_name);
    }
  }
  None
}

// -------------------------------- Device Features Handling -------------------
#[derive(Copy, Clone, Debug)]
pub(super) struct RequiredFeatures<'a> {
  pub features: vk::PhysicalDeviceFeatures,
  pub buffer_device_address: vk::PhysicalDeviceBufferDeviceAddressFeatures<'a>,
  pub vulkan_memory_model: vk::PhysicalDeviceVulkanMemoryModelFeatures<'a>,
  pub timeline_semaphore: vk::PhysicalDeviceTimelineSemaphoreFeatures<'a>,
  pub synchronization2: vk::PhysicalDeviceSynchronization2Features<'a>,
  // TODO add VK_KHR_variable_pointers (promoted to 1.1)
}

impl RequiredFeatures<'_> {
  pub fn new() -> Self {
    let features = vk::PhysicalDeviceFeatures::default();
    let buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
    let vulkan_memory_model = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
    let timeline_semaphore = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
    let synchronization2 = vk::PhysicalDeviceSynchronization2Features::default();

    Self {
      features,
      buffer_device_address,
      vulkan_memory_model,
      timeline_semaphore,
      synchronization2,
    }
  }

  pub fn as_features2(&mut self) -> vk::PhysicalDeviceFeatures2<'_> {
    vk::PhysicalDeviceFeatures2::default()
      .features(self.features)
      .push_next(&mut self.buffer_device_address)
      .push_next(&mut self.vulkan_memory_model)
      .push_next(&mut self.timeline_semaphore)
      .push_next(&mut self.synchronization2)
  }

  pub fn populate(&mut self) -> &mut Self {
    self.features.fill_mode_non_solid = vk::TRUE;
    self.buffer_device_address.buffer_device_address = vk::TRUE;
    self.vulkan_memory_model.vulkan_memory_model = vk::TRUE;
    self.timeline_semaphore.timeline_semaphore = vk::TRUE;
    self.synchronization2.synchronization2 = vk::TRUE;

    self
  }

  pub fn any_missing(&self) -> Option<Vec<string::String>> {
    use string::ToString;
    let mut the_vec = Vec::with_capacity(64);
    if self.features.fill_mode_non_solid != vk::TRUE {
      the_vec.push("fill_mode_non_solid".to_string());
    }
    if self.buffer_device_address.buffer_device_address != vk::TRUE {
      the_vec.push("buffer_device_address".to_string());
    }
    if self.vulkan_memory_model.vulkan_memory_model != vk::TRUE {
      the_vec.push("vulkan_memory_model".to_string());
    }
    if self.timeline_semaphore.timeline_semaphore != vk::TRUE {
      the_vec.push("timeline_semaphore".to_string());
    }
    if self.synchronization2.synchronization2 != vk::TRUE {
      the_vec.push("synchronization2".to_string());
    }

    if the_vec.is_empty() {
      None
    } else {
      Some(the_vec)
    }
  }
}

// -------------------------------- Extension: Result Mapping ------------------

impl From<vk::Result> for GpuError {
  fn from(err: vk::Result) -> Self {
    match err {
      vk::Result::ERROR_DEVICE_LOST => GpuError::DeviceLost,
      vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => GpuError::OutOfMemory,
      // TODO more as needed
      _ => GpuError::BackendSpecific(err.to_string()),
    }
  }
}

/// Necessary wrapper struct as `ptr::NonNull` cannot be used with ash's
/// implementation of Vulkan's non dispatchable handles
pub(super) struct NonZeroHandle<T>
where
  T: ash::vk::Handle + Copy,
{
  handle: T,
}

impl<T> NonZeroHandle<T>
where
  T: ash::vk::Handle + Copy,
{
  #[inline(always)]
  pub(super) unsafe fn new_unchecked(value: T) -> Self {
    Self { handle: value }
  }

  #[inline(always)]
  pub(super) fn dangling() -> Self {
    Self {
      handle: <T as ash::vk::Handle>::from_raw(u64::MAX),
    }
  }

  #[inline(always)]
  pub(super) fn new(value: T) -> Option<Self> {
    if value.is_null() {
      None
    } else {
      Some(unsafe { Self::new_unchecked(value) })
    }
  }

  #[inline(always)]
  pub(super) fn get(&self) -> T {
    self.handle
  }
}

impl<T> Clone for NonZeroHandle<T>
where
  T: ash::vk::Handle + Copy,
{
  fn clone(&self) -> Self {
    Self {
      handle: self.handle.clone(),
    }
  }
}

impl<T> Copy for NonZeroHandle<T> where T: ash::vk::Handle + Copy {}

/// Allow compiler to implicitly coerce &NonZeroHandle<Handle> to &Handle
impl<T> ops::Deref for NonZeroHandle<T>
where
  T: ash::vk::Handle + Copy,
{
  type Target = T;
  #[inline(always)]
  fn deref(&self) -> &Self::Target {
    &self.handle
  }
}

pub(super) fn create_transient_attachment(
  allocator: &vk_mem::Allocator,
  extent: vk::Extent2D,
  format: vk::Format,
  usage: vk::ImageUsageFlags,
  samples: vk::SampleCountFlags,
) -> GpuResult<(NonZeroHandle<vk::Image>, vk_mem::Allocation)> {
  // Note: Trying transient on Apple gives `Metal validation error: residency sets do not support memoryless resources``
  let image_create_info = vk::ImageCreateInfo::default()
    .extent(vk::Extent3D {
      width: extent.width,
      height: extent.height,
      depth: 1,
    })
    .format(format)
    .image_type(vk::ImageType::TYPE_2D)
    .mip_levels(1)
    .array_layers(1)
    .samples(samples)
    .usage(
      usage
        | if cfg!(not(target_os = "macos")) {
          vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
        } else {
          vk::ImageUsageFlags::empty()
        },
    )
    .sharing_mode(vk::SharingMode::EXCLUSIVE);

  let allocation_info = {
    let mut x = vk_mem::AllocationCreateInfo::default();
    x.usage = vk_mem::MemoryUsage::AutoPreferDevice;
    x.required_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    // prefer lazily allocated (transient -> tile cacheable) on Tile GPUs
    if cfg!(not(target_os = "macos")) {
      x.preferred_flags = vk::MemoryPropertyFlags::LAZILY_ALLOCATED;
    }
    x.priority = 1.0;
    x
  };

  unsafe { allocator.create_image(&image_create_info, &allocation_info) }
    .map(|(i, a)| (unsafe { NonZeroHandle::new_unchecked(i) }, a))
    .map_err(|e| e.into())
}

// -------------------------------- Unit Testing -------------------------------
#[cfg(test)]
mod tests {
  use super::*;
  // TODO: Test that vulkan loader library is dropped only when arc count is zero
}
