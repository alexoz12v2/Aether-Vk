//! renderdoc.rs — RenderDoc In-Application API integration (debug builds only).
//!
//! Uses `dlopen`/`GetModuleHandle` with the *no-load* flag so we never force-inject
//! the library ourselves.  If the process was not launched under RenderDoc the
//! `RDOC_API` cell stays `None` and every public function is a silent no-op.
//!
//! Compiles only when `debug_assertions` is enabled (dev / test profiles).

#[allow(non_camel_case_types, dead_code)]
mod ffi {
  // ── RENDERDOC_Version enum equivalent ───────────────────────────────────
  pub const RENDERDOC_API_VERSION_1_6_0: u32 = 10600;

  // ── Vtable layout for RENDERDOC_API_1_6_0 ───────────────────────────────
  // Only the fields we actually call need to be correct.  Unused fields are
  // represented as opaque function-pointer-sized words so the struct layout
  // matches the C definition even if we never call them.
  #[repr(C)]
  pub struct RENDERDOC_API_1_6_0 {
    pub GetAPIVersion: unsafe extern "C" fn(*mut i32, *mut i32, *mut i32),
    pub SetCaptureOptionU32: unsafe extern "C" fn(u32, u32) -> i32,
    pub SetCaptureOptionF32: unsafe extern "C" fn(u32, f32) -> i32,
    pub GetCaptureOptionU32: unsafe extern "C" fn(u32) -> u32,
    pub GetCaptureOptionF32: unsafe extern "C" fn(u32) -> f32,
    pub SetFocusToggleKeys: unsafe extern "C" fn(*const i32, i32),
    pub SetCaptureKeys: unsafe extern "C" fn(*const i32, i32),
    pub GetOverlayBits: unsafe extern "C" fn() -> u32,
    pub MaskOverlayBits: unsafe extern "C" fn(u32, u32),
    pub RemoveHooks: unsafe extern "C" fn(),
    pub UnloadCrashHandler: unsafe extern "C" fn(),
    pub SetCaptureFilePathTemplate: unsafe extern "C" fn(*const core::ffi::c_char),
    pub GetCaptureFilePathTemplate: unsafe extern "C" fn() -> *const core::ffi::c_char,
    pub GetNumCaptures: unsafe extern "C" fn() -> u32,
    pub GetCapture: unsafe extern "C" fn(u32, *mut core::ffi::c_char, *mut u32, *mut u64) -> u32,
    pub TriggerCapture: unsafe extern "C" fn(),
    pub IsTargetControlConnected: unsafe extern "C" fn() -> u32,
    pub LaunchReplayUI: unsafe extern "C" fn(u32, *const core::ffi::c_char) -> u32,
    pub SetActiveWindow: unsafe extern "C" fn(*mut core::ffi::c_void, *mut core::ffi::c_void),
    pub StartFrameCapture: unsafe extern "C" fn(*mut core::ffi::c_void, *mut core::ffi::c_void),
    pub IsFrameCapturing: unsafe extern "C" fn() -> u32,
    pub EndFrameCapture: unsafe extern "C" fn(*mut core::ffi::c_void, *mut core::ffi::c_void),
    pub TriggerMultiFrameCapture: unsafe extern "C" fn(u32),
    pub SetCaptureFileComments:
      unsafe extern "C" fn(*const core::ffi::c_char, *const core::ffi::c_char),
    pub DiscardFrameCapture:
      unsafe extern "C" fn(*mut core::ffi::c_void, *mut core::ffi::c_void) -> u32,
    pub ShowReplayUI: unsafe extern "C" fn(u32),
    pub SetCaptureTitle: unsafe extern "C" fn(*const core::ffi::c_char),
  }

  /// Signature of the single export that RenderDoc exposes from its shared library.
  pub type RENDERDOC_GetAPI_fn =
    unsafe extern "C" fn(version: u32, out_api: *mut *mut core::ffi::c_void) -> i32;
}

use core::ptr;
use spin::Once;

#[derive(Clone, Copy)]
struct RdocApiPtr(*mut ffi::RENDERDOC_API_1_6_0);
// SAFETY: The API vtable pointer is immutable and points to static function pointers.
// It is safe to share and use it across multiple threads.
unsafe impl Send for RdocApiPtr {}
unsafe impl Sync for RdocApiPtr {}

/// Process-wide cache: `None` = RenderDoc not present, `Some(ptr)` = valid vtable.
///
/// # Safety
/// The pointer is obtained once at first use and never freed.  The RenderDoc
/// shared library stays loaded for the lifetime of the process when it is
/// injected, so the pointer remains valid.
static RDOC_API: Once<Option<RdocApiPtr>> = Once::new();

/// Internal: get the cached (or freshly loaded) vtable reference.
fn api() -> Option<&'static ffi::RENDERDOC_API_1_6_0> {
  RDOC_API.call_once(|| try_load().map(RdocApiPtr));
  // SAFETY: `Once` ensures `try_load` ran exactly once and the result is stable.
  unsafe { (*RDOC_API.get()?).map(|p| &*p.0) }
}

/// Try to resolve the RenderDoc vtable from an already-injected library.
/// Returns `None` if RenderDoc is not present in this process.
fn try_load() -> Option<*mut ffi::RENDERDOC_API_1_6_0> {
  let handle = get_rdoc_handle()?;
  resolve_api(handle)
}

// ── Platform-specific handle acquisition ─────────────────────────────────────

#[cfg(target_os = "linux")]
fn get_rdoc_handle() -> Option<*mut core::ffi::c_void> {
  // RTLD_NOLOAD (4): succeed only if already loaded; never map a new library.
  const RTLD_NOW: i32 = 2;
  const RTLD_NOLOAD: i32 = 4;
  let handle =
    unsafe { libc::dlopen(b"librenderdoc.so\0".as_ptr().cast(), RTLD_NOW | RTLD_NOLOAD) };
  if handle.is_null() {
    None
  } else {
    Some(handle.cast())
  }
}

#[cfg(windows)]
fn get_rdoc_handle() -> Option<*mut core::ffi::c_void> {
  use windows::Win32::System::LibraryLoader::GetModuleHandleA;
  // SAFETY: windows crate, safe wrapper.
  unsafe { GetModuleHandleA(windows::core::s!("renderdoc.dll")).ok() }
    .map(|h| h.0 as *mut core::ffi::c_void)
}

#[cfg(not(any(target_os = "linux", windows)))]
fn get_rdoc_handle() -> Option<*mut core::ffi::c_void> {
  None
}

// ── Symbol resolution ─────────────────────────────────────────────────────────

fn resolve_api(handle: *mut core::ffi::c_void) -> Option<*mut ffi::RENDERDOC_API_1_6_0> {
  let sym = get_symbol(handle, b"RENDERDOC_GetAPI\0")?;

  // SAFETY: we verified the symbol is non-null and has the expected C signature.
  let get_api: ffi::RENDERDOC_GetAPI_fn = unsafe { core::mem::transmute(sym) };

  let mut api_ptr: *mut core::ffi::c_void = ptr::null_mut();
  let ok = unsafe { get_api(ffi::RENDERDOC_API_VERSION_1_6_0, &mut api_ptr) };

  if ok != 1 || api_ptr.is_null() {
    aethervk_oshal_rlib::log!(
      "[RenderDoc] RENDERDOC_GetAPI returned {} — API unavailable",
      ok
    );
    None
  } else {
    aethervk_oshal_rlib::log!("[RenderDoc] API acquired (version 1.6.0)");
    Some(api_ptr.cast())
  }
}

#[cfg(target_os = "linux")]
fn get_symbol(handle: *mut core::ffi::c_void, name: &[u8]) -> Option<*mut core::ffi::c_void> {
  let sym = unsafe { libc::dlsym(handle.cast(), name.as_ptr().cast()) };
  if sym.is_null() {
    None
  } else {
    Some(sym.cast())
  }
}

#[cfg(windows)]
fn get_symbol(handle: *mut core::ffi::c_void, name: &[u8]) -> Option<*mut core::ffi::c_void> {
  use windows::Win32::Foundation::HMODULE;
  use windows::Win32::System::LibraryLoader::GetProcAddress;
  let name_cstr = unsafe { core::ffi::CStr::from_bytes_with_nul_unchecked(name) };
  unsafe {
    GetProcAddress(
      HMODULE(handle as _),
      windows::core::PCSTR(name_cstr.as_ptr().cast()),
    )
    .map(|f| f as *mut core::ffi::c_void)
  }
}

#[cfg(not(any(target_os = "linux", windows)))]
fn get_symbol(_handle: *mut core::ffi::c_void, _name: &[u8]) -> Option<*mut core::ffi::c_void> {
  None
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Returns `true` if the process was launched under RenderDoc and the in-app
/// API was successfully acquired.
///
/// Calling this function triggers the one-time library probe via [`Once`].
pub fn is_available() -> bool {
  api().is_some()
}

/// Requests RenderDoc to capture the **next presented frame** on any active
/// swapchain (equivalent to pressing F12 in the RenderDoc UI).
///
/// No-op if RenderDoc is not loaded.
pub fn trigger_capture() {
  if let Some(a) = api() {
    unsafe { (a.TriggerCapture)() };
    aethervk_oshal_rlib::log!("[RenderDoc] TriggerCapture() dispatched — next frame will be saved");
  } else {
    aethervk_oshal_rlib::log!("[RenderDoc] trigger_capture() called but API is unavailable");
  }
}

/// Begins a manually-scoped capture limited to one `device + window` pair.
///
/// - `device`: The `VkDevice` handle cast to `*mut c_void`.
/// - `wnd`:    The OS window handle — XCB: `xcb_window_t` as pointer; Wayland: `wl_surface *`; Win32: `HWND`.
///
/// Must be called from the render thread, before `vkAcquireNextImageKHR`.
/// Pair every call with [`end_frame_capture`] using the same arguments.
/// No-op if RenderDoc is not loaded.
pub unsafe fn start_frame_capture(device: *mut core::ffi::c_void, wnd: *mut core::ffi::c_void) {
  if let Some(a) = api() {
    unsafe { (a.StartFrameCapture)(device, wnd) };
    aethervk_oshal_rlib::log!("[RenderDoc] StartFrameCapture dispatched");
  }
}

/// Ends the scoped capture started by [`start_frame_capture`].
///
/// Call after `vkQueuePresentKHR` for the **same** `device + window` pair.
/// No-op if RenderDoc is not loaded.
pub unsafe fn end_frame_capture(device: *mut core::ffi::c_void, wnd: *mut core::ffi::c_void) {
  if let Some(a) = api() {
    unsafe { (a.EndFrameCapture)(device, wnd) };
    aethervk_oshal_rlib::log!("[RenderDoc] EndFrameCapture dispatched — .rdc file written");
  }
}