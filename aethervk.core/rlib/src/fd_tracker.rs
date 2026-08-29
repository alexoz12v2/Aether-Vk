#[cfg(all(test, target_os = "linux"))]
pub mod tracker {
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::sync::Mutex;
    use std::collections::HashMap;

    pub static NVIDIA_OPEN_COUNT: AtomicI32 = AtomicI32::new(0);
    pub static MAX_FD_SEEN: AtomicI32 = AtomicI32::new(0);

    lazy_static::lazy_static! {
        pub static ref OPEN_TRACES: Mutex<HashMap<i32, String>> = Mutex::new(HashMap::new());
    }

    thread_local! {
        pub static IN_HOOK: std::cell::Cell<bool> = std::cell::Cell::new(false);
    }
}

#[cfg(all(test, target_os = "linux"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn open64(path: *const libc::c_char, oflag: libc::c_int, mode: libc::mode_t) -> libc::c_int {
    type Open64Func = unsafe extern "C" fn(*const libc::c_char, libc::c_int, libc::mode_t) -> libc::c_int;
    lazy_static::lazy_static! {
        static ref REAL_OPEN64: Open64Func = unsafe {
            let handle = libc::dlsym(libc::RTLD_NEXT, b"open64\0".as_ptr() as *const _);
            std::mem::transmute(handle)
        };
    }
    
    let fd = unsafe { REAL_OPEN64(path, oflag, mode) };
    
    if fd >= 0 {
        tracker::MAX_FD_SEEN.fetch_max(fd as i32, std::sync::atomic::Ordering::SeqCst);
        if !path.is_null() {
            let c_str = unsafe { std::ffi::CStr::from_ptr(path) };
            if let Ok(s) = c_str.to_str() {
                if s.contains("nvidia") {
                    tracker::IN_HOOK.with(|in_hook| {
                        if !in_hook.get() {
                            in_hook.set(true);
                            let count = tracker::NVIDIA_OPEN_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                            let backtrace = std::backtrace::Backtrace::force_capture().to_string();
                            if count > 50 || fd > 500 {
                                println!("[FD_TRACKER] Warning: open64('{}') returned fd {}. Nvidia open count: {}.\n{}", s, fd, count, backtrace);
                            }
                            tracker::OPEN_TRACES.lock().unwrap().insert(fd, backtrace);
                            in_hook.set(false);
                        }
                    });
                }
            }
        }
    }
    
    fd
}

#[cfg(all(test, target_os = "linux"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn open(path: *const libc::c_char, oflag: libc::c_int, mode: libc::mode_t) -> libc::c_int {
    type OpenFunc = unsafe extern "C" fn(*const libc::c_char, libc::c_int, libc::mode_t) -> libc::c_int;
    lazy_static::lazy_static! {
        static ref REAL_OPEN: OpenFunc = unsafe {
            let handle = libc::dlsym(libc::RTLD_NEXT, b"open\0".as_ptr() as *const _);
            std::mem::transmute(handle)
        };
    }
    
    let fd = unsafe { REAL_OPEN(path, oflag, mode) };
    
    if fd >= 0 {
        tracker::MAX_FD_SEEN.fetch_max(fd as i32, std::sync::atomic::Ordering::SeqCst);
        if !path.is_null() {
            let c_str = unsafe { std::ffi::CStr::from_ptr(path) };
            if let Ok(s) = c_str.to_str() {
                if s.contains("nvidia") {
                    tracker::IN_HOOK.with(|in_hook| {
                        if !in_hook.get() {
                            in_hook.set(true);
                            let count = tracker::NVIDIA_OPEN_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                            let backtrace = std::backtrace::Backtrace::force_capture().to_string();
                            if count > 50 || fd > 500 {
                                println!("[FD_TRACKER] Warning: open('{}') returned fd {}. Nvidia open count: {}.\n{}", s, fd, count, backtrace);
                            }
                            tracker::OPEN_TRACES.lock().unwrap().insert(fd, backtrace);
                            in_hook.set(false);
                        }
                    });
                }
            }
        }
    }
    
    fd
}

#[cfg(all(test, target_os = "linux"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn close(fd: libc::c_int) -> libc::c_int {
    type CloseFunc = unsafe extern "C" fn(libc::c_int) -> libc::c_int;
    lazy_static::lazy_static! {
        static ref REAL_CLOSE: CloseFunc = unsafe {
            let handle = libc::dlsym(libc::RTLD_NEXT, b"close\0".as_ptr() as *const _);
            std::mem::transmute(handle)
        };
    }
    
    let ret = unsafe { REAL_CLOSE(fd) };
    
    if ret == 0 {
        let mut traces = tracker::OPEN_TRACES.lock().unwrap();
        if traces.remove(&fd).is_some() {
            tracker::NVIDIA_OPEN_COUNT.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    
    ret
}
