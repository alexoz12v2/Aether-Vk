use aethervk_core_rlib::gpu::{OpaqueNativeHandleInfo, RenderDeviceHandle, RenderFrontend};

#[cfg(target_os = "macos")]
use objc2_app_kit::NSView;
#[cfg(target_os = "macos")]
use objc2_quartz_core::CAAutoresizingMask;
#[cfg(target_os = "macos")]
use raw_window_handle::RawWindowHandle;
#[cfg(all(target_os = "linux", feature = "linux_xcb"))]
use raw_window_handle::RawWindowHandle;
#[cfg(target_os = "linux")]
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
#[cfg(target_os = "linux")]
use core::ffi;
#[cfg(windows)]
use raw_window_handle::RawWindowHandle;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
#[cfg(windows)]
use core::ffi;
use winit::{event_loop::EventLoopProxy, window::Window};

/// Custom event type to handle resizing start and stop
pub enum AppEvent {
  ResizeStarted,
  ResizeEnded,
}

#[cfg(target_os = "macos")]
pub unsafe fn setup_metal_layer(
  window: &Window,
  device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> objc2::rc::Retained<objc2_quartz_core::CAMetalLayer> {
  use objc2_app_kit::NSView;
  use objc2_quartz_core::CAMetalLayer;
  use objc2_metal::MTLPixelFormat;
  use objc2_core_foundation::CGSize;

  let raw_handle = window.window_handle().unwrap().as_raw();
  let view_ptr = match raw_handle {
    RawWindowHandle::AppKit(w) => w.ns_view.as_ptr(),
    _ => panic!("Expected an AppKit window handle"),
  };

  let view: &NSView = unsafe { (view_ptr as *const NSView).as_ref() }.unwrap();

  let layer = CAMetalLayer::new();
  layer.setDevice(Some(device));

  // REMOVED: setPixelFormat and setDrawableSize. MoltenVK MUST own these.
  layer.setPresentsWithTransaction(false);

  let scale_factor = window.scale_factor();
  layer.setContentsScale(scale_factor);

  // 1. Give the layer a physical UI dimension by matching the View's bounds
  let view_bounds = view.bounds();
  layer.setFrame(view_bounds);

  // 2. Ensure the layer resizes when the window/view resizes
  // CAAutoresizingMask: WidthSizable (1 << 1) | HeightSizable (1 << 4) = 18
  layer.setAutoresizingMask(
    CAAutoresizingMask::LayerHeightSizable | CAAutoresizingMask::LayerWidthSizable,
  );

  // Attach to NSView (creating a layer-hosting view)
  view.setLayer(Some(&layer));
  view.setWantsLayer(true);

  layer
}

pub struct WindowPlatformData {
  #[cfg(target_os = "macos")]
  pub metal_layer: objc2::rc::Retained<objc2_quartz_core::CAMetalLayer>,
}

impl WindowPlatformData {
  #[cfg(target_os = "macos")]
  pub fn new_macos(metal_layer: objc2::rc::Retained<objc2_quartz_core::CAMetalLayer>) -> Self {
    Self { metal_layer }
  }
}

pub struct WindowExtractHandlesParams {
  #[cfg(target_os = "macos")]
  pub mtl_device: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
}

impl WindowExtractHandlesParams {
  #[cfg(target_os = "macos")]
  pub fn new_macos(
    mtl_device: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
  ) -> Self {
    Self { mtl_device }
  }
}

/// Utility to extract native handles from [`winit::Window`]
pub fn extract_native_handles(
  window: &Window,
  _params: &WindowExtractHandlesParams,
) -> (OpaqueNativeHandleInfo, WindowPlatformData) {
  // extract raw handles from winit window
  let window_handle = window.window_handle().unwrap().as_raw();
  let display_handle = window.display_handle().unwrap().as_raw();

  match (window_handle, display_handle) {
    #[cfg(windows)]
    (RawWindowHandle::Win32(w), _) => (
      OpaqueNativeHandleInfo {
        ptr0: w.hinstance.map(|h| h.get()).unwrap_or(0) as *mut ffi::c_void,
        ptr1: w.hwnd.get() as *mut ffi::c_void,
      },
      WindowPlatformData {},
    ),

    #[cfg(all(target_os = "linux", feature = "linux_wayland"))]
    (RawWindowHandle::Wayland(w), RawDisplayHandle::Wayland(d)) => (
      OpaqueNativeHandleInfo {
        ptr0: d.display.as_ptr() as *mut ffi::c_void,
        ptr1: w.surface.as_ptr() as *mut ffi::c_void,
      },
      WindowPlatformData {},
    ),

    #[cfg(all(target_os = "linux", feature = "linux_xlib"))]
    (RawWindowHandle::Xlib(w), RawDisplayHandle::Xlib(d)) => (
      OpaqueNativeHandleInfo {
        ptr0: d
          .display
          .map(|d| d.as_ptr())
          .unwrap_or(std::ptr::null_mut()) as *mut ffi::c_void,
        ptr1: w.window as usize as *mut ffi::c_void,
      },
      WindowPlatformData {},
    ),

    #[cfg(all(target_os = "linux", feature = "linux_xcb"))]
    (RawWindowHandle::Xcb(w), RawDisplayHandle::Xcb(d)) => (
      OpaqueNativeHandleInfo {
        ptr0: d
          .connection
          .map(|c| c.as_ptr())
          .unwrap_or(std::ptr::null_mut()) as *mut ffi::c_void,
        ptr1: w.window.get() as usize as *mut ffi::c_void,
      },
      WindowPlatformData {},
    ),

    #[cfg(target_os = "macos")]
    (RawWindowHandle::AppKit(w), _) => {
      let layer = unsafe { setup_metal_layer(window, &_params.mtl_device) };

      let info = OpaqueNativeHandleInfo {
        ptr0: core::ptr::from_ref::<objc2_quartz_core::CALayer>(layer.as_ref())
          as *mut core::ffi::c_void,
        ptr1: std::ptr::null_mut(),
      };

      (info, WindowPlatformData::new_macos(layer))
    }

    _ => panic!("unsupported platform or handle mismatch"),
  }
}

/// We need to intercept `WM_ENTERSIZEMOVE` and `WM_EXITSIZEMOVE` so that we can pause and resume
/// rendering during live resize on Windows.
#[cfg(windows)]
pub fn setup_windows_resize_hook(
  window: &Window,
  proxy_ptr: std::ptr::NonNull<EventLoopProxy<AppEvent>>,
) {
  use windows::Win32::{
    Foundation::{HWND, LPARAM, LRESULT, WPARAM},
    UI::Shell::{SetWindowSubclass, DefSubclassProc},
    UI::WindowsAndMessaging::{WM_ENTERSIZEMOVE, WM_EXITSIZEMOVE},
  };

  unsafe extern "system" fn subclass_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    _id_subclass: usize,
    _ref_data: usize,
  ) -> LRESULT {
    match msg {
      WM_ENTERSIZEMOVE => {
        let proxy =
          unsafe { std::ptr::NonNull::new_unchecked(_ref_data as *mut EventLoopProxy<AppEvent>) };
        let _ = unsafe { proxy.as_ref() }.send_event(AppEvent::ResizeStarted);
      }
      WM_EXITSIZEMOVE => {
        let proxy =
          unsafe { std::ptr::NonNull::new_unchecked(_ref_data as *mut EventLoopProxy<AppEvent>) };
        let _ = unsafe { proxy.as_ref() }.send_event(AppEvent::ResizeEnded);
      }
      _ => {}
    }

    unsafe { DefSubclassProc(hwnd, msg, wparam, lparam) }
  }

  let handle = window.window_handle().unwrap().as_raw();
  if let RawWindowHandle::Win32(win32_handle) = handle {
    unsafe {
      let hwnd = HWND(win32_handle.hwnd.get() as *mut _);
      let _ = SetWindowSubclass(hwnd, Some(subclass_proc), 1, proxy_ptr.as_ptr() as _);
    }
  }
}

#[cfg(target_os = "macos")]
use objc2::{ClassType, DeclaredClass, declare_class, msg_send, msg_send_id, rc::Retained, sel};
#[cfg(target_os = "macos")]
use objc2_foundation::{ns_string, NSNotification, NSNotificationCenter, NSObject};
#[cfg(target_os = "macos")]
use std::cell::Cell;
use std::path::PathBuf;
use aethervk_core_rlib::gpu;
use aethervk_core_rlib::types::GpuResult;

#[cfg(target_os = "macos")]
pub struct ResizeObserverIvars {
  proxy_ptr: Cell<std::ptr::NonNull<EventLoopProxy<AppEvent>>>,
}

#[cfg(target_os = "macos")]
objc2::define_class!(
  #[unsafe(super(NSObject))]
  #[ivars = ResizeObserverIvars]
  pub struct ResizeObserver;

  impl ResizeObserver {
    #[unsafe(method(windowWillStartLiveResize:))]
    fn will_start_resize(&self, _notif: &NSNotification) {
      let ivars = self.ivars();
      let ptr = ivars.proxy_ptr.get();
      let _ = unsafe { ptr.as_ref() }.send_event(AppEvent::ResizeStarted);
    }

    #[unsafe(method(windowDidEndLiveResize:))]
    fn did_end_resize(&self, _notif: &NSNotification) {
      let ivars = self.ivars();
      let ptr = ivars.proxy_ptr.get();
      let _ = unsafe { ptr.as_ref() }.send_event(AppEvent::ResizeEnded);
    }
  }
);

#[cfg(target_os = "macos")]
impl ResizeObserver {
  pub fn new(proxy_ptr: std::ptr::NonNull<EventLoopProxy<AppEvent>>) -> Retained<Self> {
    let this: Retained<Self> = unsafe {
      let some = msg_send![Self::class(), alloc];
      msg_send![some, init]
    };
    this.ivars().proxy_ptr.set(proxy_ptr);
    this
  }
}

#[cfg(target_os = "macos")]
pub fn setup_macos_resize_hook(
  window: &Window,
  proxy_ptr: std::ptr::NonNull<EventLoopProxy<AppEvent>>,
) {
  let handle = window.window_handle().unwrap().as_raw();
  if let RawWindowHandle::AppKit(appkit_handle) = handle {
    let observer = ResizeObserver::new(proxy_ptr);
    let observer_raw = Retained::into_raw(observer);
    unsafe {
      let center = NSNotificationCenter::defaultCenter();
      let view = appkit_handle.ns_view.cast::<NSView>();
      let window_obj = unsafe { view.as_ref() }.window().unwrap();
      center.addObserver_selector_name_object(
        unsafe { observer_raw.as_ref().unwrap_unchecked() },
        sel!(windowWillStartLiveResize:),
        Some(ns_string!("NSWindowWillStartLiveResizeNotification")),
        Some(&window_obj),
      );
      center.addObserver_selector_name_object(
        unsafe { observer_raw.as_ref().unwrap_unchecked() },
        sel!(windowDidEndLiveResize:),
        Some(ns_string!("NSWindowDidEndLiveResizeNotification")),
        Some(&window_obj),
      );
    }
  }
}

pub fn setup_resize_hook(window: &Window, proxy_ptr: std::ptr::NonNull<EventLoopProxy<AppEvent>>) {
  #[cfg(windows)]
  {
    setup_windows_resize_hook(&window, proxy_ptr);
  }
  #[cfg(target_os = "macos")]
  {
    setup_macos_resize_hook(&window, proxy_ptr);
  }
  #[cfg(target_os = "linux")]
  {
    // todo!();
  }
}

pub fn cycle_get_asset_path_from_exe(use_args: bool) -> PathBuf {
  let asset_dir = {
    let mut args = std::env::args();
    if use_args && args.len() > 1 {
      let _ = args.next().unwrap();
      std::path::PathBuf::from(args.next().unwrap())
    } else {
      let mut path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();
      while !path.join("assets").exists() {
        path = path.parent().unwrap().to_owned();
      }
      path.join("assets")
    }
  };
  assert!(asset_dir.is_dir());
  let mut guard = aethervk_core_rlib::gpu::ASSET_DIR.write();
  *guard = Some(asset_dir.to_str().unwrap().to_string());
  drop(guard);
  asset_dir
}

pub fn get_handle_and_window_info(
  render_frontend: &RenderFrontend,
  render_device_handle: RenderDeviceHandle,
  window: &Window,
) -> (OpaqueNativeHandleInfo, WindowPlatformData) {
  let params: WindowExtractHandlesParams;
  #[cfg(not(target_os = "macos"))]
  {
    params = WindowExtractHandlesParams {};
  }
  #[cfg(target_os = "macos")]
  {
    let mtl_device_id = render_frontend
      .with_device(render_device_handle, |device| {
        let mtl_device_id = device
          .get_native_prop(gpu::NativeGpuProperty::VulkanMetalDeviceId)
          .unwrap();
        let dev_ptr =
          mtl_device_id as *mut objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>;
        let metal_device = unsafe { objc2::rc::Retained::retain(dev_ptr).unwrap() };
        Ok(metal_device)
      })
      .unwrap();
    params = WindowExtractHandlesParams::new_macos(mtl_device_id);
  }
  extract_native_handles(&window, &params)
}
