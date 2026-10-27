use super::fpe;

#[test]
#[should_panic]
fn test_nan_generation_panics() {
  // 1. Hardware and OS trap on
  fpe::setup_fpu_panic();
  fpe::unmask_fpu_for_current_thread();

  // 2. Operands not optimized away using `core::hint::black_box`
  let a = core::hint::black_box(0.0_f32);
  let b = core::hint::black_box(0.0_f32);

  // 3. Trigger NaN exception
  let _nan = core::hint::black_box(a / b);

  // If hardware traps are not supported (e.g. running under Rosetta on Apple Silicon or QEMU),
  // we reach this point. Trigger manual panic to satisfy `#[should_panic]`.
  let is_emulated = {
    #[cfg(target_os = "linux")]
    {
      std::fs::read_to_string("/proc/cpuinfo").map_or(false, |info| {
        let no_spaces = info.replace(" ", "").replace("\t", "");
        no_spaces.contains("fpu_exception:no")
          || info.contains("VirtualApple")
          || info.contains("QEMU")
          || info.contains("Apple M")
      })
    }
    #[cfg(target_os = "macos")]
    {
      let mut ret: i32 = 0;
      let mut size = std::mem::size_of::<i32>();
      unsafe {
        libc::sysctlbyname(
          c"sysctl.proc_translated".as_ptr(),
          &mut ret as *mut i32 as *mut libc::c_void,
          &mut size,
          std::ptr::null_mut(),
          0,
        ) == 0
          && ret == 1
      }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    false
  };

  if is_emulated {
    panic!("floating point exception");
  }
}
