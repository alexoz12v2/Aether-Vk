//! utils module.

use crate::{
  gpu::DeviceAdditionalParams,
  gpu_backends::vulkan::device::LogicalDevice,
  types::{GpuError, GpuResult},
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
use core::{
  ffi::{CStr, c_char, c_void},
  mem, ops, ptr,
};
use itertools::Itertools;
use vk_mem::Alloc;
#[cfg(windows)]
use windows::{
  Win32::System::LibraryLoader::{GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW},
  core::PCSTR,
  core::{s, w},
};

// -------------------------------- Helper Types -----------------------------
bitflags! {
  #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub(super) struct OptionalExtensionSupportFlags: u64 {
    const NONE = 0;
    const SOME_EXTENSION = 1 << 0;
    const SWAPCHAIN_MAINTENANCE1 = 1 << 1;
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalDeviceQueryInput {
  /// Which Linux surface extensions were actually enabled on the instance.
  #[cfg(target_os = "linux")]
  pub linux_surface_support: super::instance::LinuxSurfaceSupport,
  #[cfg(target_os = "linux")]
  pub wl_display: Option<core::ptr::NonNull<vk::wl_display>>,
  #[cfg(target_os = "linux")]
  pub xcb_connection: Option<core::ptr::NonNull<vk::xcb_connection_t>>,
  #[cfg(target_os = "linux")]
  pub xcb_visualid: Option<vk::xcb_visualid_t>,
  #[cfg(target_os = "linux")]
  pub dpy: Option<core::ptr::NonNull<vk::Display>>,
  #[cfg(target_os = "linux")]
  pub visual_id: Option<vk::VisualID>,
  pub debug_shaders: bool,
}
impl PhysicalDeviceQueryInput {
  pub(super) fn from_params(_value: &DeviceAdditionalParams) -> Option<Self> {
    #[cfg(target_os = "linux")]
    let wl_display = _value
      .get(&super::DEVICE_ADDIDITIONAL_PARAM_WL_DISPLAY)
      .and_then(|intptr| core::ptr::NonNull::new((*intptr) as *mut _));
    #[cfg(target_os = "linux")]
    let xcb_connection = _value
      .get(&super::DEVICE_ADDIDITIONAL_PARAM_XCB_CONNECTION)
      .and_then(|intptr| core::ptr::NonNull::new((*intptr) as *mut _));
    #[cfg(target_os = "linux")]
    let xcb_visualid = _value.get(&super::DEVICE_ADDIDITIONAL_PARAM_XCB_VISUALID).map(|v| *v as _);
    #[cfg(target_os = "linux")]
    let dpy = _value
      .get(&super::DEVICE_ADDIDITIONAL_PARAM_DPY)
      .and_then(|intptr| core::ptr::NonNull::new((*intptr) as *mut _));
    #[cfg(target_os = "linux")]
    let visual_id = _value.get(&super::DEVICE_ADDIDITIONAL_PARAM_VISUAL_ID).map(|v| *v as _);

    let debug_shaders = _value
      .get(&super::DEVICE_ADDIDITIONAL_PARAM_DEBUG_SHADERS)
      .map_or(false, |v| *v != 0);

    Some(Self {
      #[cfg(target_os = "linux")]
      // Will be populated by the caller once the Instance knows what's available
      linux_surface_support: Default::default(),
      #[cfg(target_os = "linux")]
      wl_display,
      #[cfg(target_os = "linux")]
      xcb_connection,
      #[cfg(target_os = "linux")]
      xcb_visualid,
      #[cfg(target_os = "linux")]
      dpy,
      #[cfg(target_os = "linux")]
      visual_id,
      debug_shaders,
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

    #[cfg(target_os = "linux")]
    {
      if self.linux_surface_support.wayland {
        if let Some(wl) = self.wl_display {
          unsafe {
            let ptr_copy = wl.as_ptr();
            _supported = ash::khr::wayland_surface::Instance::new(_entry, _instance)
              .get_physical_device_wayland_presentation_support(
                _physical_device,
                _queue_family_index,
                ptr_copy.as_mut().unwrap(),
              );
          }
        } else {
          _supported = true;
        }
      } else if self.linux_surface_support.xcb {
        if let (Some(conn), Some(vis)) = (self.xcb_connection, self.xcb_visualid) {
          unsafe {
            _supported = ash::khr::xcb_surface::Instance::new(_entry, _instance)
              .get_physical_device_xcb_presentation_support(
                _physical_device,
                _queue_family_index,
                &mut *conn.as_ptr(),
                vis,
              );
          }
        } else {
          _supported = true;
        }
      } else if self.linux_surface_support.xlib {
        if let (Some(dpy), Some(vis)) = (self.dpy, self.visual_id) {
          unsafe {
            _supported = ash::khr::xlib_surface::Instance::new(_entry, _instance)
              .get_physical_device_xlib_presentation_support(
                _physical_device,
                _queue_family_index,
                dpy.as_ptr(),
                vis,
              );
          }
        } else {
          _supported = true;
        }
      } else {
        // No recognised surface extension: assume presentation is possible
        // (headless / unknown environment)
        _supported = true;
      }
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
pub struct PhysicalDeviceQueryResult {
  pub physical_device: vk::PhysicalDevice,
  pub physical_device_properties: vk::PhysicalDeviceProperties,
  pub family_count: usize,
  pub optional_extensions: OptionalExtensionSupportFlags,
  pub graphics_queue_family_index: u32,
  pub compute_queue_family_index: u32,
  pub transfer_queue_family_index: u32,
  pub subgroup_size: u32,
  /// True when the physical device is a CPU (e.g. Lavapipe / llvmpipe).
  /// Used to select CPU-optimised SPIR-V variants and reduced workgroup sizes.
  pub is_cpu: bool,
  pub score: i32,
  pub debug_shaders: bool,
  pub max_per_stage_descriptor_update_after_bind_samplers: u32,
  pub max_descriptor_set_update_after_bind_samplers: u32,
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
    unique_queue_families.insert(self.graphics_queue_family_index).unwrap();
    unique_queue_families.insert(self.compute_queue_family_index).unwrap();
    unique_queue_families.insert(self.transfer_queue_family_index).unwrap();

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
    let mut the_vec: Vec<*const c_char> =
      required_device_extensions().iter().map(|cstr| cstr.as_ptr()).collect();

    if self
      .optional_extensions
      .contains(OptionalExtensionSupportFlags::SWAPCHAIN_MAINTENANCE1)
    {
      the_vec.push(ash::ext::swapchain_maintenance1::NAME.as_ptr());
    }

    the_vec
  }
}

// -------------------------------- Debug Messenger --------------------------
#[cfg(test)]
pub static VULKAN_ERROR_MESSAGES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

#[cfg(debug_assertions)]
pub(super) unsafe extern "system" fn debug_utils_messenger_user_callback(
  message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
  _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
  p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
  _p_user_data: *mut c_void,
) -> vk::Bool32 {
  let p_msg = unsafe { (*p_callback_data).p_message };
  let msg = unsafe { core::ffi::CStr::from_ptr(p_msg) };
  let s = msg.to_string_lossy();

  if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
    #[cfg(test)]
    {
      extern crate std;
      let _ = std::fs::write("VULKAN_ERROR_DUMP.txt", s.as_ref());
      std::eprintln!("VULKAN ERROR: {}", s);
    }
    if !_p_user_data.is_null() {
      let cb: fn(&str) = unsafe { core::mem::transmute(_p_user_data) };
      cb(&s);
    }
  } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
    #[cfg(test)]
    {
      extern crate std;
      std::eprintln!("VULKAN WARNING: {}", s);
    }
    #[cfg(not(test))]
    {
      aethervk_oshal_rlib::log!("VULKAN WARNING: {}", s);
    }
  } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
    #[cfg(test)]
    {
      extern crate std;
      std::println!("VULKAN INFO: {}", s);
    }
    #[cfg(not(test))]
    {
      aethervk_oshal_rlib::log!("VULKAN INFO: {}", s);
    }
  }

  vk::FALSE
}

/// Safe debug callback for use **only** during `vkCreateInstance` (via `pNext`).
///
/// ## Why a separate callback?
///
/// The real [`debug_utils_messenger_user_callback`] transmutes `user_data` into a Rust
/// `fn(&str)` and calls it. Panicking from inside a C `vkCreateInstance` call stack is
/// **undefined behaviour** (even if it works by accident on Linux via libunwind).
///
/// This callback never panics through FFI. Instead:
/// - **Extension-compatibility messages** (e.g. RenderDoc saying `VK_KHR_wayland_surface`
///   is unsupported): logged only — the retry loop strips that extension and retries.
/// - **Genuine validation errors**: stored in [`InstanceCreationState`] so the retry loop
///   can call the user panic-callback *after* `vkCreateInstance` returns to Rust.

/// State passed via `user_data` to [`creation_phase_debug_callback`].
#[cfg(debug_assertions)]
pub(super) struct InstanceCreationState {
  /// User-supplied callback that may panic. Called safely AFTER `vkCreateInstance`
  /// returns to Rust — never from within the C FFI call chain.
  pub user_error_callback: Option<fn(&str)>,
  /// Set when a genuine (non-extension-compat) validation error was captured.
  pub had_validation_error: bool,
  /// The first real validation error message received in this attempt.
  pub validation_error_message: alloc::string::String,
}

#[cfg(debug_assertions)]
pub(super) unsafe extern "system" fn creation_phase_debug_callback(
  message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
  _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
  p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
  p_user_data: *mut c_void,
) -> vk::Bool32 {
  let p_msg = unsafe { (*p_callback_data).p_message };
  let msg = unsafe { core::ffi::CStr::from_ptr(p_msg) };
  let s = msg.to_string_lossy();

  if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
    // Heuristic: extension-compatibility messages name a surface extension or use a
    // known phrase. They are expected during our retry loop and must NOT panic.
    let is_ext_compat = s.contains("does not support requested instance extension")
      || s.contains("does not support Wayland")
      || (s.contains("VK_KHR_") && s.contains("surface"));

    if is_ext_compat {
      aethervk_oshal_rlib::log!("[vkCreateInstance] Extension compat (will retry): {}", s);
    } else {
      // Real validation error — store it; caller panics after vkCreateInstance returns.
      aethervk_oshal_rlib::log!("[vkCreateInstance] VULKAN ERROR: {}", s);
      #[cfg(test)]
      {
        extern crate std;
        std::eprintln!("[vkCreateInstance] VULKAN ERROR: {}", s);
      }
      if !p_user_data.is_null() {
        let state = unsafe { &mut *(p_user_data as *mut InstanceCreationState) };
        if !state.had_validation_error {
          state.had_validation_error = true;
          state.validation_error_message = s.into_owned();
        }
      }
    }
  } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
    aethervk_oshal_rlib::log!("[vkCreateInstance] VULKAN WARNING: {}", s);
  }
  // INFO / VERBOSE suppressed during creation to reduce noise.

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
    // MoltenVK on macOS requires a massive amount of stack space during its C++ static initialization
    // and Metal driver boot sequence via dlopen. To prevent stack overflows on background worker threads
    // (like those spawned by cargo test), we isolate the entire Vulkan loader initialization in a dedicated
    // throwaway thread with an 8MB stack.
    let base_path_override_owned = base_path_override.map(|p| alloc::ffi::CString::from(p));
    let result_arc = alloc::sync::Arc::new(spin::Mutex::new(None));
    let result_arc_clone = result_arc.clone();

    let th = aethervk_oshal_rlib::os::thread::Builder::new()
      .name("vulkan_loader".into())
      .stack_size(8 * 1024 * 1024)
      .spawn(move || {
        let base_path_override_ref = base_path_override_owned.as_deref();
        let res = Self::new_internal(base_path_override_ref);
        *result_arc_clone.lock() = Some(res);
      })
      .map_err(|_| GpuError::InvalidState("Couldn't start vulkan_loader thread".to_string()))?;

    th.join();
    let mut guard = result_arc.lock();
    guard.take().unwrap()
  }

  fn new_internal(base_path_override: Option<&CStr>) -> GpuResult<Self> {
    let mut vulkan_loader_module = ptr::NonNull::dangling();
    let static_fn: ash::StaticFn = {
      let get_instance_proc_addr: PFN_vkGetInstanceProcAddr;
      #[cfg(windows)]
      {
        use windows::{
          Win32::{
            Foundation::HMODULE,
            System::LibraryLoader::{
              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleHandleExW, GetProcAddress,
              LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
            },
          },
          core::{PCSTR, PCWSTR, s, w},
        };

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
        // On Linux, the Vulkan loader is libvulkan.so.1 provided by the system or SDK.
        // The ICD and layer paths are typically managed by the system (mesa, nvidia, etc.)
        // or by environment variables (VK_ICD_FILENAMES, VK_LAYER_PATH) already set by the
        // caller / CI environment.

        // If a base_path_override or VULKAN_SDK env var is available, try to configure
        // layer/ICD paths and load the loader from there first (debug builds).
        // Otherwise fall back to the system libvulkan.so.1.
        let sdk_loader: Option<alloc::ffi::CString> = base_path_override
          .and_then(|base_path| {
            use aethervk_oshal_rlib::os::fs::{FileSystemObject, Path};

            let base_bytes = base_path.to_bytes();
            let dir_bytes = match base_bytes.iter().rposition(|&b| b == b'/') {
              Some(pos) => &base_bytes[..=pos],
              None => base_bytes,
            };

            let local_layer = alloc::ffi::CString::new(
              [dir_bytes, b"vulkan/share/vulkan/explicit_layer.d"].concat(),
            )
            .unwrap();
            let local_icd =
              alloc::ffi::CString::new([dir_bytes, b"vulkan/share/vulkan/icd.d"].concat()).unwrap();
            let local_loader =
              alloc::ffi::CString::new([dir_bytes, b"vulkan/lib/libvulkan.so.1"].concat()).unwrap();

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

            if local_layer_obj.is_dir() && local_icd_obj.is_dir() {
              unsafe {
                libc::setenv(b"VK_LAYER_PATH\0".as_ptr().cast(), local_layer.as_ptr(), 1);
                libc::setenv(b"VK_ICD_FILENAMES\0".as_ptr().cast(), local_icd.as_ptr(), 1);
                libc::setenv(b"VK_DRIVER_FILES\0".as_ptr().cast(), local_icd.as_ptr(), 1);
              }
              Some(local_loader)
            } else {
              None
            }
          })
          .or_else(|| {
            #[cfg(debug_assertions)]
            {
              let env_ptr = unsafe { libc::getenv(b"VULKAN_SDK\0".as_ptr().cast()) };
              core::ptr::NonNull::new(env_ptr).and_then(|ptr| {
                let sdk_cstr = unsafe { core::ffi::CStr::from_ptr(ptr.as_ptr()) };
                let sdk_bytes = sdk_cstr.to_bytes();

                let slash = if sdk_bytes.last() == Some(&b'/') {
                  b"".as_slice()
                } else {
                  b"/".as_slice()
                };

                let sdk_layer = alloc::ffi::CString::new(
                  [sdk_bytes, slash, b"share/vulkan/explicit_layer.d"].concat(),
                )
                .unwrap();
                let sdk_icd =
                  alloc::ffi::CString::new([sdk_bytes, slash, b"share/vulkan/icd.d"].concat())
                    .unwrap();
                let sdk_loader =
                  alloc::ffi::CString::new([sdk_bytes, slash, b"lib/libvulkan.so.1"].concat())
                    .unwrap();

                unsafe {
                  libc::setenv(b"VK_LAYER_PATH\0".as_ptr().cast(), sdk_layer.as_ptr(), 1);
                }

                // Check if the SDK loader actually exists before committing
                let sdk_loader_path = unsafe {
                  aethervk_oshal_rlib::os::fs::Path::from_slice(core::slice::from_raw_parts(
                    sdk_loader.as_ptr().cast::<core::ffi::c_char>(),
                    sdk_loader.as_bytes().len(),
                  ))
                };
                use aethervk_oshal_rlib::os::fs::FileSystemObject;
                if sdk_loader_path.is_file() {
                  Some(sdk_loader)
                } else {
                  None
                }
              })
            }
            #[cfg(not(debug_assertions))]
            {
              None
            }
          });

        // Load the Vulkan shared library
        let loader_name = sdk_loader
          .as_deref()
          .map(|c| c.as_ptr())
          .unwrap_or(b"libvulkan.so.1\0".as_ptr().cast());

        // Try RTLD_NOLOAD first (already mapped into the process)
        let mut lib = unsafe { libc::dlopen(loader_name, libc::RTLD_LAZY | libc::RTLD_NOLOAD) };
        if lib.is_null() {
          lib = unsafe { libc::dlopen(loader_name, libc::RTLD_NOW | libc::RTLD_GLOBAL) };
        }

        if let Some(module_ptr) = core::ptr::NonNull::new(lib) {
          vulkan_loader_module = module_ptr;
          let sym = unsafe {
            libc::dlsym(
              module_ptr.as_ptr(),
              b"vkGetInstanceProcAddr\0".as_ptr().cast(),
            )
          };
          get_instance_proc_addr = if let Some(func_addr) = core::ptr::NonNull::new(sym) {
            unsafe { core::mem::transmute(func_addr) }
          } else {
            return Err(GpuError::BackendSpecific(
              "vkGetInstanceProcAddr not found in libvulkan.so.1".into(),
            ));
          };
        } else {
          let err_ptr = unsafe { libc::dlerror() };
          let err_msg = if !err_ptr.is_null() {
            unsafe { core::ffi::CStr::from_ptr(err_ptr) }
              .to_str()
              .unwrap_or("unknown dlerror")
          } else {
            "unknown error"
          };
          return Err(GpuError::BackendSpecific(alloc::format!(
            "Failed to load libvulkan.so.1: {}",
            err_msg
          )));
        }
      }
      #[cfg(target_os = "macos")]
      {
        static ZERO: isize = 0;
        // find the current cdylib directory, assuming the packaged vulkan stuff is in there.
        // how: take this function and query which shared module is it contained in
        let mut info = unsafe { core::mem::zeroed::<libc::Dl_info>() };
        let func_addr = Self::new as *const core::ffi::c_void;
        get_instance_proc_addr = if let Some(path) = base_path_override {
          Some(path)
        } else if unsafe { libc::dladdr(func_addr, &mut info) } != 0 && !info.dli_fname.is_null() {
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
            alloc::ffi::CString::new([dir_bytes, b"vulkan/share/vulkan/explicit_layer.d"].concat())
              .unwrap();
          let local_icd =
            alloc::ffi::CString::new([dir_bytes, b"vulkan/share/vulkan/icd.d"].concat()).unwrap();
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
                let sdk_icd = alloc::ffi::CString::new(
                  [sdk_bytes, slash, b"share/vulkan/icd.d/MoltenVK_icd.json"].concat(),
                )
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
              b"VK_ICD_FILENAMES\0".as_ptr().cast(),
              vk_icd_path.as_ptr(),
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

          if lib.is_null() {
            let err_ptr = unsafe { libc::dlerror() };
            if !err_ptr.is_null() {
              let err_msg = unsafe { core::ffi::CStr::from_ptr(err_ptr) };
              aethervk_oshal_rlib::log!("dlopen error: {:?}", err_msg);
            }
          }

          core::ptr::NonNull::new(lib)
        })
        .and_then(|module_ptr| unsafe {
          vulkan_loader_module = module_ptr;
          let sym = libc::dlsym(
            module_ptr.as_ptr(),
            b"vkGetInstanceProcAddr\0".as_ptr().cast(),
          );

          if sym.is_null() {
            let err_ptr = libc::dlerror();
            if !err_ptr.is_null() {
              let err_msg = core::ffi::CStr::from_ptr(err_ptr);
              aethervk_oshal_rlib::log!("dlsym error: {:?}", err_msg);
            } else {
              aethervk_oshal_rlib::log!("dlsym error: unknown (null symbol but no dlerror)");
            }
          }

          core::ptr::NonNull::new(sym)
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

    // ash::ext::headless_surface::NAME is now added dynamically in instance.rs for testing

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
      the_vec.push(ash::khr::wayland_surface::NAME);
      the_vec.push(ash::khr::xcb_surface::NAME);
      the_vec.push(ash::khr::xlib_surface::NAME);
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
    #[cfg(any(debug_assertions, test))]
    {
      #[cfg(not(target_vendor = "apple"))]
      if crate::gpu_backends::vulkan::physics::USE_PRINTF_SHADERS
        .load(core::sync::atomic::Ordering::Relaxed)
      {
        the_vec.push(ash::khr::shader_non_semantic_info::NAME);
      }
    }

    the_vec.push(ash::ext::scalar_block_layout::NAME);

    // by everybody (namely, Apple M4)
    #[cfg(test)]
    {
      if aethervk_oshal_rlib::os::env::var("AETHERVK_ROBUST_ACCESS").as_deref() == Some("1") {
        the_vec.push(ash::ext::robustness2::NAME);
      }
    }

    #[cfg(windows)]
    {
      // external `HANDLE` stuff
      the_vec.push(ash::khr::external_fence_win32::NAME);
      the_vec.push(ash::khr::external_memory_win32::NAME);
      the_vec.push(ash::khr::external_semaphore_win32::NAME);
    }
    #[cfg(all(target_vendor = "apple", target_family = "unix"))]
    {
      // Metal is not 100% Vulkan Spec conformant
      the_vec.push(ash::khr::portability_subset::NAME);
      the_vec.push(ash::ext::metal_objects::NAME);
    }
    #[cfg(target_os = "linux")]
    {}

    // extensions for VMA (memory budget also important for out-of-core/streaming)
    the_vec.push(ash::ext::memory_budget::NAME);
    the_vec.push(ash::khr::dedicated_allocation::NAME);

    // https://docs.vulkan.org/samples/latest/samples/extensions/descriptor_indexing/README.html
    // flexibility in update after bind and non-uniform indexing
    the_vec.push(ash::ext::descriptor_indexing::NAME);

    // float16 and int8 support (with features)
    the_vec.push(ash::khr::shader_float16_int8::NAME);

    // 8-bit storage buffer access (required by physics compute shaders)
    the_vec.push(ash::khr::_8bit_storage::NAME);

    // Atomic float add (required by barnes_hut.comp: OpAtomicFAddEXT)
    the_vec.push(ash::ext::shader_atomic_float::NAME);

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
  /// promoted to 1.2
  pub buffer_device_address: vk::PhysicalDeviceBufferDeviceAddressFeatures<'a>,
  /// promoted to 1.2
  pub vulkan_memory_model: vk::PhysicalDeviceVulkanMemoryModelFeatures<'a>,
  /// promoted to 1.2
  pub timeline_semaphore: vk::PhysicalDeviceTimelineSemaphoreFeatures<'a>,
  /// promoted to 1.2
  pub synchronization2: vk::PhysicalDeviceSynchronization2Features<'a>,
  /// promoted to 1.2
  pub descriptor_indexing: vk::PhysicalDeviceDescriptorIndexingFeatures<'a>,
  pub scalar_block_layout: vk::PhysicalDeviceScalarBlockLayoutFeatures<'a>,
  pub shader_float16_int8: vk::PhysicalDeviceShaderFloat16Int8Features<'a>,
  /// Required for physics compute shaders that use StorageBuffer8BitAccess
  pub storage_8bit: vk::PhysicalDevice8BitStorageFeatures<'a>,
  pub storage_16bit: vk::PhysicalDevice16BitStorageFeatures<'a>,
  #[cfg(test)]
  pub robustness2: vk::PhysicalDeviceRobustness2FeaturesEXT<'a>,
}

impl RequiredFeatures<'_> {
  pub fn new() -> Self {
    let features = vk::PhysicalDeviceFeatures::default();
    let buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
    let vulkan_memory_model = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
    let timeline_semaphore = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
    let synchronization2 = vk::PhysicalDeviceSynchronization2Features::default();
    let descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
    let scalar_block_layout = vk::PhysicalDeviceScalarBlockLayoutFeatures::default();
    let shader_float16_int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let storage_16bit = vk::PhysicalDevice16BitStorageFeatures::default();

    Self {
      features,
      buffer_device_address,
      vulkan_memory_model,
      timeline_semaphore,
      synchronization2,
      descriptor_indexing,
      scalar_block_layout,
      shader_float16_int8,
      storage_16bit,
      storage_8bit: vk::PhysicalDevice8BitStorageFeatures::default(),
      #[cfg(test)]
      robustness2: vk::PhysicalDeviceRobustness2FeaturesEXT::default(),
    }
  }

  pub fn as_features2(&mut self) -> vk::PhysicalDeviceFeatures2<'_> {
    let mut f = vk::PhysicalDeviceFeatures2::default()
      .features(self.features)
      .push_next(&mut self.buffer_device_address)
      .push_next(&mut self.vulkan_memory_model)
      .push_next(&mut self.timeline_semaphore)
      .push_next(&mut self.synchronization2)
      .push_next(&mut self.descriptor_indexing)
      .push_next(&mut self.scalar_block_layout)
      .push_next(&mut self.shader_float16_int8)
      .push_next(&mut self.storage_16bit)
      .push_next(&mut self.storage_8bit);

    #[cfg(test)]
    {
      if aethervk_oshal_rlib::os::env::var("AETHERVK_ROBUST_ACCESS").as_deref() == Some("1") {
        f = f.push_next(&mut self.robustness2);
      }
    }

    f
  }

  pub fn populate(&mut self) -> &mut Self {
    self.features.fill_mode_non_solid = vk::TRUE;
    self.features.shader_int64 = vk::TRUE;
    self.buffer_device_address.buffer_device_address = vk::TRUE;
    self.vulkan_memory_model.vulkan_memory_model = vk::TRUE;
    self.vulkan_memory_model.vulkan_memory_model_device_scope = vk::TRUE;
    self.timeline_semaphore.timeline_semaphore = vk::TRUE;
    self.synchronization2.synchronization2 = vk::TRUE;
    self.scalar_block_layout.scalar_block_layout = vk::TRUE;
    self.descriptor_indexing.runtime_descriptor_array = vk::TRUE;
    self.descriptor_indexing.shader_sampled_image_array_non_uniform_indexing = vk::TRUE;
    self.descriptor_indexing.shader_storage_buffer_array_non_uniform_indexing = vk::TRUE;
    self.descriptor_indexing.descriptor_binding_partially_bound = vk::TRUE;
    self.descriptor_indexing.descriptor_binding_sampled_image_update_after_bind = vk::TRUE;
    self.descriptor_indexing.descriptor_binding_storage_buffer_update_after_bind = vk::TRUE;
    self.shader_float16_int8.shader_int8 = vk::TRUE;
    self.storage_16bit.storage_buffer16_bit_access = vk::TRUE;
    self.shader_float16_int8.shader_float16 = vk::TRUE;
    // Required for SPIR-V shaders that use StorageBuffer8BitAccess capability
    self.storage_8bit.storage_buffer8_bit_access = vk::TRUE;

    self.features.large_points = vk::TRUE;

    // [TEST ONLY] Enable robustBufferAccess when AETHERVK_ROBUST_ACCESS=1.
    // When enabled, OOB GPU reads return 0 and OOB writes are discarded instead of
    // killing the driver, allowing debugPrintfEXT output to survive shader bugs.
    // This is gated behind an env var so we can test both with and without it.
    #[cfg(test)]
    {
      if aethervk_oshal_rlib::os::env::var("AETHERVK_ROBUST_ACCESS").as_deref() == Some("1") {
        aethervk_oshal_rlib::log!("[ROBUST] robustBufferAccess ENABLED (AETHERVK_ROBUST_ACCESS=1)");
        self.features.robust_buffer_access = vk::TRUE;
        self.robustness2.robust_buffer_access2 = vk::TRUE;
      }
    }

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
    if self.vulkan_memory_model.vulkan_memory_model_device_scope != vk::TRUE {
      the_vec.push("vulkan_memory_model_device_scope".to_string());
    }
    if self.timeline_semaphore.timeline_semaphore != vk::TRUE {
      the_vec.push("timeline_semaphore".to_string());
    }
    if self.synchronization2.synchronization2 != vk::TRUE {
      the_vec.push("synchronization2".to_string());
    }
    if self.scalar_block_layout.scalar_block_layout != vk::TRUE {
      the_vec.push("scalar_block_layout".to_string());
    }
    if self.descriptor_indexing.runtime_descriptor_array != vk::TRUE {
      the_vec.push("descriptor_indexing".to_string());
    }
    if self.descriptor_indexing.shader_sampled_image_array_non_uniform_indexing != vk::TRUE {
      the_vec.push("descriptor_indexing_non_uniform_indexing".to_string());
    }
    if self.descriptor_indexing.shader_storage_buffer_array_non_uniform_indexing != vk::TRUE {
      the_vec.push("descriptor_indexing_storage_buffer_non_uniform_indexing".to_string());
    }
    if self.descriptor_indexing.descriptor_binding_partially_bound != vk::TRUE {
      the_vec.push("descriptor_binding_partially_bound".to_string());
    }
    if self.descriptor_indexing.descriptor_binding_sampled_image_update_after_bind != vk::TRUE {
      the_vec.push("descriptor_binding_sampled_image_update_after_bind".to_string());
    }
    if self.shader_float16_int8.shader_int8 != vk::TRUE {
      the_vec.push("shader_float16_int8".to_string());
    }
    if self.storage_16bit.storage_buffer16_bit_access != vk::TRUE {
      the_vec.push("storage_buffer16_bit_access".to_string());
    }
    if self.shader_float16_int8.shader_float16 != vk::TRUE {
      the_vec.push("shader_float16_float16".to_string());
    }
    if self.storage_8bit.storage_buffer8_bit_access != vk::TRUE {
      the_vec.push("storage_buffer_8_bit_access".to_string());
    }
    if self.features.large_points != vk::TRUE {
      the_vec.push("large_points".to_string());
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
      _ => GpuError::BackendSpecific(err.to_string()),
    }
  }
}

/// Necessary wrapper struct as `ptr::NonNull` cannot be used with ash's
/// implementation of Vulkan's non dispatchable handles
#[repr(transparent)]
pub struct NonZeroHandle<T>
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
  pub fn get(&self) -> T {
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
  allocator: vk_mem::AllocatorView,
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
    crate::apply_test_dedicated_alloc!(x);
    x
  };

  unsafe { allocator.create_image(&image_create_info, &allocation_info) }
    .map(|(i, a)| (unsafe { NonZeroHandle::new_unchecked(i) }, a))
    .map_err(|e| e.into())
}

#[cfg(test)]
pub(super) fn create_test_attachment(
  allocator: vk_mem::AllocatorView,
  extent: vk::Extent2D,
  format: vk::Format,
  usage: vk::ImageUsageFlags,
  samples: vk::SampleCountFlags,
) -> GpuResult<(NonZeroHandle<vk::Image>, vk_mem::Allocation)> {
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
    .usage(usage)
    .sharing_mode(vk::SharingMode::EXCLUSIVE);

  let allocation_info = {
    let mut x = vk_mem::AllocationCreateInfo::default();
    x.usage = vk_mem::MemoryUsage::AutoPreferDevice;
    x.required_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    x.priority = 1.0;
    crate::apply_test_dedicated_alloc!(x);
    x
  };

  unsafe { allocator.create_image(&image_create_info, &allocation_info) }
    .map(|(i, a)| (unsafe { NonZeroHandle::new_unchecked(i) }, a))
    .map_err(|e| e.into())
}

// -------------------------------- Vulkan Transaction --------------------------

/// Tracks Vulkan resources allocated during lock-free execution.
/// Destroys them in LIFO order if the transaction is aborted.
pub struct RollbackContext<'a> {
  pub device: &'a LogicalDevice,
  rollbacks: alloc::vec::Vec<alloc::boxed::Box<dyn FnOnce(&LogicalDevice) + 'a>>,
  defused: bool,
}

impl<'a> RollbackContext<'a> {
  pub fn new(device: &'a LogicalDevice) -> Self {
    Self {
      device,
      rollbacks: alloc::vec::Vec::new(),
      defused: false,
    }
  }

  /// Schedule a cleanup closure for a Vulkan resource created during execution.
  pub fn defer<F: FnOnce(&LogicalDevice) + 'a>(&mut self, f: F) {
    self.rollbacks.push(alloc::boxed::Box::new(f));
  }

  /// Internal method called on successful commit to prevent rollbacks.
  pub fn defuse(&mut self) {
    self.defused = true;
  }
}

impl<'a> Drop for RollbackContext<'a> {
  fn drop(&mut self) {
    if !self.defused {
      // LIFO order destruction guarantees correct Vulkan cleanup dependencies
      // (e.g. image views are destroyed before their images)
      while let Some(rollback) = self.rollbacks.pop() {
        rollback(self.device);
      }
    }
  }
}

pub trait RwLockable {
  // Define the target state as an associated type
  type State;

  type RwWriteGuard<'a>: core::ops::DerefMut<Target = Self::State> + Drop
  where
    Self: 'a;
  type RwReadGuard<'a>: core::ops::Deref<Target = Self::State> + Drop
  where
    Self: 'a;

  fn write(&self) -> Self::RwWriteGuard<'_>;
  fn read(&self) -> Self::RwReadGuard<'_>;
}

pub trait RwLockableTuple {
  type TupleReadGuards<'a>
  where
    Self: 'a;
  type TupleWriteGuards<'a>
  where
    Self: 'a;

  fn read_both(&self) -> Self::TupleReadGuards<'_>;
  fn write_both(&self) -> Self::TupleWriteGuards<'_>;
}

// No T1 or T2 needed at all!
impl<L1, L2> RwLockableTuple for (&L1, &L2)
where
  L1: RwLockable,
  L2: RwLockable,
{
  type TupleReadGuards<'a>
    = (L1::RwReadGuard<'a>, L2::RwReadGuard<'a>)
  where
    Self: 'a;
  type TupleWriteGuards<'a>
    = (L1::RwWriteGuard<'a>, L2::RwWriteGuard<'a>)
  where
    Self: 'a;

  fn read_both(&self) -> Self::TupleReadGuards<'_> {
    (self.0.read(), self.1.read())
  }

  fn write_both(&self) -> Self::TupleWriteGuards<'_> {
    (self.0.write(), self.1.write())
  }
}

pub struct VulkanTransaction<'a, State, Lock: RwLockable<State = State>> {
  lock: &'a Lock,
  device: &'a LogicalDevice,
  _marker: core::marker::PhantomData<fn() -> State>,
}

pub struct ChainedPreparedTransaction<'a, State, Lock: RwLockable<State = State>, Prepared, Error> {
  lock: &'a Lock,
  rollback: RollbackContext<'a>,
  prepared: Result<Prepared, Error>,
  _marker: core::marker::PhantomData<fn() -> State>,
}

impl<'a, State, Lock: RwLockable<State = State>, Prepared, Error>
  ChainedPreparedTransaction<'a, State, Lock, Prepared, Error>
{
  pub fn execute<Output, F>(mut self, f: F) -> ExecutedTransaction<'a, State, Lock, Output, Error>
  where
    // F is only invoked if all previous steps (and this preparation) succeeded.
    F: FnOnce(Prepared, &mut RollbackContext<'a>) -> Result<Output, Error>,
  {
    let result = match self.prepared {
      Ok(prepared) => f(prepared, &mut self.rollback),
      Err(e) => Err(e), // Short-circuit: skip execution, pass the error forward
    };

    ExecutedTransaction {
      lock: self.lock,
      result,
      rollback: self.rollback,
      _marker: core::marker::PhantomData,
    }
  }
}

impl<'a, State, Lock: RwLockable<State = State>> VulkanTransaction<'a, State, Lock> {
  pub fn new(lock: &'a Lock, device: &'a LogicalDevice) -> Self {
    Self {
      lock,
      device,
      _marker: core::marker::PhantomData,
    }
  }

  pub fn prepare_read<Args, Prepared, Error, F>(
    self,
    args: Args,
    mut f: F,
  ) -> Result<PreparedTransaction<'a, State, Lock, Prepared, Error>, Error>
  where
    F: FnMut(&State, Args) -> Result<Prepared, Error>,
  {
    let prepared = {
      let state = self.lock.read(); // Read lock acquired
      f(&state, args)?
    }; // Read lock dropped
    Ok(PreparedTransaction {
      lock: self.lock,
      device: self.device,
      prepared,
      _marker: core::marker::PhantomData,
    })
  }

  pub fn prepare_write<Args, Prepared, Error, F>(
    self,
    args: Args,
    mut f: F,
  ) -> Result<PreparedTransaction<'a, State, Lock, Prepared, Error>, Error>
  where
    F: FnMut(&mut State, Args) -> Result<Prepared, Error>,
  {
    let prepared = {
      let mut state = self.lock.write(); // Write lock acquired
      f(&mut state, args)?
    }; // Write lock dropped
    Ok(PreparedTransaction {
      lock: self.lock,
      device: self.device,
      prepared,
      _marker: core::marker::PhantomData,
    })
  }
}

pub struct PreparedTransaction<'a, State, Lock: RwLockable<State = State>, Prepared, Error> {
  lock: &'a Lock,
  device: &'a LogicalDevice,
  prepared: Prepared,
  _marker: core::marker::PhantomData<fn() -> (State, Error)>,
}

impl<'a, State, Lock: RwLockable<State = State>, Prepared, Error>
  PreparedTransaction<'a, State, Lock, Prepared, Error>
{
  pub fn execute<Output, F>(self, f: F) -> ExecutedTransaction<'a, State, Lock, Output, Error>
  where
    // RollbackContext injected securely into the execute closure
    F: FnOnce(Prepared, &mut RollbackContext<'a>) -> Result<Output, Error>,
  {
    let mut rollback = RollbackContext::new(self.device);

    // Execute heavy OS/Vulkan API calls WITHOUT locks.
    let result = f(self.prepared, &mut rollback);

    ExecutedTransaction {
      lock: self.lock,
      result,
      rollback,
      _marker: core::marker::PhantomData,
    }
  }
}

pub struct ExecutedTransaction<'a, State, Lock: RwLockable<State = State>, Output, Error> {
  lock: &'a Lock,
  result: Result<Output, Error>,
  rollback: RollbackContext<'a>,
  _marker: core::marker::PhantomData<fn() -> State>,
}

impl<'a, State, Lock: RwLockable<State = State>, Output, Error>
  ExecutedTransaction<'a, State, Lock, Output, Error>
{
  pub fn commit<NewOutput, F>(mut self, mut f: F) -> Result<NewOutput, Error>
  where
    F: FnMut(&mut State, Result<Output, Error>) -> Result<NewOutput, Error>,
  {
    let final_result = {
      let mut state = self.lock.write(); // Write lock acquired
      f(&mut state, self.result)
    }; // Write lock dropped

    // If the final outcome is Ok, defuse the rollbacks so resources are kept!
    if final_result.is_ok() {
      self.rollback.defuse();
    }

    // If final_result is Err, `self.rollback` drops here and runs cleanup automatically!
    final_result
  }

  /// Optimized commit for concurrent data structures (like DashMap)
  pub fn commit_read<NewOutput, F>(mut self, mut f: F) -> Result<NewOutput, Error>
  where
    F: FnMut(&State, Result<Output, Error>) -> Result<NewOutput, Error>,
  {
    let final_result = {
      let state = self.lock.read(); // Read lock acquired
      f(&state, self.result)
    }; // Read lock dropped

    if final_result.is_ok() {
      self.rollback.defuse();
    }
    final_result
  }

  /// Chains a new read preparation phase using the output of the previous execution.
  pub fn and_then_prepare_read<Args, NewPrepared, F>(
    self,
    args: Args,
    mut f: F,
  ) -> ChainedPreparedTransaction<'a, State, Lock, NewPrepared, Error>
  where
    F: FnMut(&State, Output, Args) -> Result<NewPrepared, Error>,
  {
    let prepared = match self.result {
      Ok(output) => {
        let state = self.lock.read(); // Read lock acquired
        f(&state, output, args)
      } // Read lock dropped
      Err(e) => Err(e),
    };

    ChainedPreparedTransaction {
      lock: self.lock,
      rollback: self.rollback,
      prepared,
      _marker: core::marker::PhantomData,
    }
  }

  /// Chains a new write preparation phase using the output of the previous execution.
  pub fn and_then_prepare_write<Args, NewPrepared, F>(
    self,
    args: Args,
    mut f: F,
  ) -> ChainedPreparedTransaction<'a, State, Lock, NewPrepared, Error>
  where
    F: FnMut(&mut State, Output, Args) -> Result<NewPrepared, Error>,
  {
    let prepared = match self.result {
      Ok(output) => {
        let mut state = self.lock.write(); // Write lock acquired
        f(&mut state, output, args)
      } // Write lock dropped
      Err(e) => Err(e),
    };

    ChainedPreparedTransaction {
      lock: self.lock,
      rollback: self.rollback,
      prepared,
      _marker: core::marker::PhantomData,
    }
  }
}

pub struct VulkanTupleTransaction<'a, Lock: RwLockableTuple> {
  lock: &'a Lock,
  device: &'a LogicalDevice,
}

impl<'a, Lock: RwLockableTuple> VulkanTupleTransaction<'a, Lock> {
  pub fn new(lock: &'a Lock, device: &'a LogicalDevice) -> Self {
    Self { lock, device }
  }

  pub fn prepare_read<Args, Prepared, Error, F>(
    self,
    args: Args,
    mut f: F,
  ) -> Result<PreparedTupleTransaction<'a, Lock, Prepared>, Error>
  where
    // The closure now takes a reference to the TupleReadGuards
    F: FnMut(&Lock::TupleReadGuards<'_>, Args) -> Result<Prepared, Error>,
  {
    let prepared = {
      let guards = self.lock.read_both(); // Tuple of read locks acquired
      f(&guards, args)?
    }; // Read locks dropped

    Ok(PreparedTupleTransaction {
      lock: self.lock,
      device: self.device,
      prepared,
    })
  }

  pub fn prepare_write<Args, Prepared, Error, F>(
    self,
    args: Args,
    mut f: F,
  ) -> Result<PreparedTupleTransaction<'a, Lock, Prepared>, Error>
  where
    // The closure takes a mutable reference to the TupleWriteGuards
    F: FnMut(&mut Lock::TupleWriteGuards<'_>, Args) -> Result<Prepared, Error>,
  {
    let prepared = {
      let mut guards = self.lock.write_both(); // Tuple of write locks acquired
      f(&mut guards, args)?
    }; // Write locks dropped

    Ok(PreparedTupleTransaction {
      lock: self.lock,
      device: self.device,
      prepared,
    })
  }
}

pub struct PreparedTupleTransaction<'a, Lock: RwLockableTuple, Prepared> {
  lock: &'a Lock,
  device: &'a LogicalDevice,
  prepared: Prepared,
}

impl<'a, Lock: RwLockableTuple, Prepared> PreparedTupleTransaction<'a, Lock, Prepared> {
  pub fn execute<Output, Error, F>(self, f: F) -> ExecutedTupleTransaction<'a, Lock, Output, Error>
  where
    F: FnOnce(Prepared, &mut RollbackContext<'a>) -> Result<Output, Error>,
  {
    let mut rollback = RollbackContext::new(self.device);

    // Execute heavy OS/Vulkan API calls WITHOUT locks.
    let result = f(self.prepared, &mut rollback);

    ExecutedTupleTransaction {
      lock: self.lock,
      result,
      rollback,
    }
  }
}

pub struct ExecutedTupleTransaction<'a, Lock: RwLockableTuple, Output, Error> {
  lock: &'a Lock,
  result: Result<Output, Error>,
  rollback: RollbackContext<'a>,
}

impl<'a, Lock: RwLockableTuple, Output, Error> ExecutedTupleTransaction<'a, Lock, Output, Error> {
  pub fn commit<NewOutput, F>(mut self, mut f: F) -> Result<NewOutput, Error>
  where
    F: FnMut(&mut Lock::TupleWriteGuards<'_>, Result<Output, Error>) -> Result<NewOutput, Error>,
  {
    let final_result = {
      let mut guards = self.lock.write_both();
      f(&mut guards, self.result)
    };

    if final_result.is_ok() {
      self.rollback.defuse();
    }

    final_result
  }

  // commit_read, and_then_prepare_read, etc., follow the exact same pattern
  // substituting `&State` for `&Lock::TupleReadGuards<'_>`
}

// --- Nested Transaction ---

pub struct NestedVulkanTransaction<'a, OuterLock: RwLockable, InnerLock: RwLockable, P> {
  outer_lock: &'a OuterLock,
  device: &'a LogicalDevice,
  project: P,
  _marker: core::marker::PhantomData<fn() -> InnerLock>,
}

impl<'a, OuterLock, InnerLock, P> NestedVulkanTransaction<'a, OuterLock, InnerLock, P>
where
  OuterLock: RwLockable,
  InnerLock: RwLockable,
  // The closure maps from the outer state reference to the inner lock reference
  P: Fn(&OuterLock::State) -> &InnerLock,
{
  pub fn new(outer_lock: &'a OuterLock, device: &'a LogicalDevice, project: P) -> Self {
    Self {
      outer_lock,
      device,
      project,
      _marker: core::marker::PhantomData,
    }
  }

  pub fn prepare_read<Args, Prepared, Error, F>(
    self,
    args: Args,
    mut f: F,
  ) -> Result<PreparedNestedTransaction<'a, OuterLock, InnerLock, P, Prepared>, Error>
  where
    F: FnMut(&OuterLock::State, &InnerLock::State, Args) -> Result<Prepared, Error>,
  {
    let prepared = {
      let outer_guard = self.outer_lock.read();
      let inner_lock = (self.project)(&*outer_guard);
      let inner_guard = inner_lock.read();
      f(&*outer_guard, &*inner_guard, args)?
    };

    Ok(PreparedNestedTransaction {
      outer_lock: self.outer_lock,
      device: self.device,
      project: self.project,
      prepared,
      _marker: core::marker::PhantomData,
    })
  }

  pub fn prepare_write<Args, Prepared, Error, F>(
    self,
    args: Args,
    mut f: F,
  ) -> Result<PreparedNestedTransaction<'a, OuterLock, InnerLock, P, Prepared>, Error>
  where
    F: FnMut(&OuterLock::State, &mut InnerLock::State, Args) -> Result<Prepared, Error>,
  {
    let prepared = {
      let outer_guard = self.outer_lock.read();
      let inner_lock = (self.project)(&*outer_guard);
      let mut inner_guard = inner_lock.write();
      f(&*outer_guard, &mut *inner_guard, args)?
    };

    Ok(PreparedNestedTransaction {
      outer_lock: self.outer_lock,
      device: self.device,
      project: self.project,
      prepared,
      _marker: core::marker::PhantomData,
    })
  }
}

// --- Prepared Nested Transaction ---

pub struct PreparedNestedTransaction<'a, OuterLock: RwLockable, InnerLock: RwLockable, P, Prepared>
{
  outer_lock: &'a OuterLock,
  device: &'a LogicalDevice,
  project: P,
  prepared: Prepared,
  _marker: core::marker::PhantomData<fn() -> InnerLock>,
}

impl<'a, OuterLock, InnerLock, P, Prepared>
  PreparedNestedTransaction<'a, OuterLock, InnerLock, P, Prepared>
where
  OuterLock: RwLockable,
  InnerLock: RwLockable,
  P: Fn(&OuterLock::State) -> &InnerLock,
{
  pub fn execute<Output, Error, F>(
    self,
    f: F,
  ) -> ExecutedNestedTransaction<'a, OuterLock, InnerLock, P, Output, Error>
  where
    F: FnOnce(Prepared, &mut RollbackContext<'a>) -> Result<Output, Error>,
  {
    let mut rollback = RollbackContext::new(self.device);
    let result = f(self.prepared, &mut rollback);

    ExecutedNestedTransaction {
      outer_lock: self.outer_lock,
      project: self.project,
      result,
      rollback,
      _marker: core::marker::PhantomData,
    }
  }
}

// --- Executed Nested Transaction ---

pub struct ExecutedNestedTransaction<
  'a,
  OuterLock: RwLockable,
  InnerLock: RwLockable,
  P,
  Output,
  Error,
> {
  outer_lock: &'a OuterLock,
  project: P,
  result: Result<Output, Error>,
  rollback: RollbackContext<'a>,
  _marker: core::marker::PhantomData<fn() -> InnerLock>,
}

impl<'a, OuterLock, InnerLock, P, Output, Error>
  ExecutedNestedTransaction<'a, OuterLock, InnerLock, P, Output, Error>
where
  OuterLock: RwLockable,
  InnerLock: RwLockable,
  P: Fn(&OuterLock::State) -> &InnerLock,
{
  pub fn commit<NewOutput, F>(mut self, mut f: F) -> Result<NewOutput, Error>
  where
    F: FnMut(
      &OuterLock::State,
      &mut InnerLock::State,
      Result<Output, Error>,
    ) -> Result<NewOutput, Error>,
  {
    let final_result = {
      let outer_guard = self.outer_lock.read();
      let inner_lock = (self.project)(&*outer_guard);
      let mut inner_guard = inner_lock.write();
      f(&*outer_guard, &mut *inner_guard, self.result)
    };

    if final_result.is_ok() {
      self.rollback.defuse();
    }

    final_result
  }

  pub fn commit_read<NewOutput, F>(mut self, mut f: F) -> Result<NewOutput, Error>
  where
    F: FnMut(
      &OuterLock::State,
      &InnerLock::State,
      Result<Output, Error>,
    ) -> Result<NewOutput, Error>,
  {
    let final_result = {
      let outer_guard = self.outer_lock.read();
      let inner_lock = (self.project)(&*outer_guard);
      let inner_guard = inner_lock.read();
      f(&*outer_guard, &*inner_guard, self.result)
    };

    if final_result.is_ok() {
      self.rollback.defuse();
    }

    final_result
  }

  // extend `and_then_prepare_read/write` here following the exact same projection pattern.
}
