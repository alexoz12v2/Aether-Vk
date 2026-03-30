use aethervk_oshal_rlib as r#impl;
pub use r#impl::{AvkSystemInfo};

extern crate core;
extern crate alloc;

use core::{ptr, mem};

// -------------------- Static Storage (From rlib) ------------------
pub use r#impl::{SYSTEM_INFO};

// -------------------- C Exposed API -------------------------------
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSystemInfo_get(system_info: *mut AvkSystemInfo) {
  if !system_info.is_null() {
    unsafe {
      ptr::copy_nonoverlapping(
        SYSTEM_INFO.as_mut_ptr(),
        system_info,
        mem::size_of::<AvkSystemInfo>(),
      );
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "aarch64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasNEON(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_neon()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasSSE(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_sse()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasSSE2(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_sse2()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasSSE3(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_sse3()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasSSSE3(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_ssse3()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasSSE4_1(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_sse4_1()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasSSE4_2(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_sse4_2()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasFMA(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_fma()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasAVX(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_avx()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasAVX2(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_avx2()
}
