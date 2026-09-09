#[cfg(all(test, target_os = "linux"))]
pub mod tracker {
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::sync::Mutex;
    use std::collections::HashMap;

    pub static NVIDIA_OPEN_COUNT: AtomicI32 = AtomicI32::new(0);
    pub static MAX_FD_SEEN: AtomicI32 = AtomicI32::new(0);

    lazy_static::lazy_static! {
        pub static ref OPEN_TRACES: Mutex<HashMap<i32, String>> = Mutex::new(HashMap::new());
        pub static ref CONFIG: (bool, i32) = {
            let disabled = std::env::var("FD_TRACKER_DISABLE").map(|v| v == "1" || v == "true").unwrap_or(false);
            let delta = std::env::var("FD_TRACKER_DELTA").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(1);
            (disabled, delta)
        };
    }

    thread_local! {
        pub static IN_HOOK: std::cell::Cell<bool> = std::cell::Cell::new(false);
    }

    pub fn compress_backtrace(bt: &str) -> String {
        let mut compressed = String::new();
        let mut count = 0;
        let mut unknown_count = 0;
        
        let mut iter = bt.lines().peekable();
        
        while let Some(line) = iter.next() {
            let trimmed = line.trim_start();
            
            if trimmed.contains("<unknown>") {
                unknown_count += 1;
                continue;
            }
            
            if trimmed.chars().next().map_or(false, |c| c.is_digit(10)) {
                if unknown_count > 0 {
                    compressed.push_str(&format!("      ... [{} unknown frames]\n", unknown_count));
                    unknown_count = 0;
                }
                
                let func_line = line;
                let mut has_file_line = false;
                let mut file_line = "";
                
                if let Some(next) = iter.peek() {
                    if next.trim_start().starts_with("at ") {
                        file_line = next;
                        has_file_line = true;
                    }
                }
                
                if has_file_line {
                    iter.next(); // consume file line
                    if file_line.contains("/rustc/") || file_line.contains(".cargo/registry") || 
                       func_line.contains("std::") || func_line.contains("core::") || 
                       func_line.contains("backtrace::") || func_line.contains("libc") {
                        continue;
                    }
                    compressed.push_str(func_line);
                    compressed.push('\n');
                    compressed.push_str(file_line);
                    compressed.push('\n');
                } else {
                    if func_line.contains("libc") || func_line.contains("pthread") {
                        continue;
                    }
                    compressed.push_str(func_line);
                    compressed.push('\n');
                }
                
                count += 1;
                if count >= 8 {
                    compressed.push_str("      ... [truncated]\n");
                    break;
                }
            }
        }
        
        if unknown_count > 0 && count < 8 {
            compressed.push_str(&format!("      ... [{} unknown frames]\n", unknown_count));
        }
        
        if compressed.is_empty() {
            bt.lines().take(5).collect::<Vec<_>>().join("\n")
        } else {
            compressed
        }
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
                            if !tracker::CONFIG.0 {
                                if count > 50 || fd > 500 {
                                    if count % tracker::CONFIG.1 == 0 {
                                        let compressed = tracker::compress_backtrace(&backtrace);
                                        println!("[FD_TRACKER] Warning: open64('{}') returned fd {}. Nvidia open count: {}.\n{}", s, fd, count, compressed);
                                    }
                                }
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
                            if !tracker::CONFIG.0 {
                                if count > 50 || fd > 500 {
                                    if count % tracker::CONFIG.1 == 0 {
                                        let compressed = tracker::compress_backtrace(&backtrace);
                                        println!("[FD_TRACKER] Warning: open('{}') returned fd {}. Nvidia open count: {}.\n{}", s, fd, count, compressed);
                                    }
                                }
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
