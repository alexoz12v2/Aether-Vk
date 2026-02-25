// disable std only for non tests
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

// crash compilation on 32 bit machines
#[cfg(not(target_pointer_width = "64"))]
compile_error!("Target Must be a 64-bit machine");

#[cfg(any(test, feature = "std"))]
extern crate std;

// panic_handler and global allocator declared for non tests at cdylib/FFI level
extern crate core;
extern crate alloc;

#[cfg(windows)]
use windows::{
  Win32::Foundation::HANDLE,
  Win32::System::Threading::{GetCurrentProcess, TerminateProcess},
};

// --------------- Centralized Panic Handler Implementation ---------
#[inline]
pub fn panic_handler_impl() -> ! {
  #[cfg(windows)]
  {
    // TODO: on debug_assertions, use WinDbg to print stacktrace
    unsafe {
      let handle: HANDLE = GetCurrentProcess();
      let exit_code: u32 = 1;
      let _ = TerminateProcess(handle, exit_code);
    }
  }
  #[cfg(target_family = "unix")]
  {
    unsafe { libc::exit(libc::EXIT_FAILURE) };
  }

  #[allow(unreachable_code)]
  loop {}
}

// -------------------- Runtime Struct ------------------------------

// -------------------- Initialization Types ------------------------
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AvkSystemInfo {
  pub arch_features: u32,
}

#[cfg(target_arch = "x86_64")]
const SSE_MASK: u32 = 1 << 0;
#[cfg(target_arch = "x86_64")]
const SSE2_MASK: u32 = 1 << 1;
#[cfg(target_arch = "x86_64")]
const SSE3_MASK: u32 = 1 << 2;
#[cfg(target_arch = "x86_64")]
const SSSE3_MASK: u32 = 1 << 3;
#[cfg(target_arch = "x86_64")]
const SSE4_1_MASK: u32 = 1 << 4;
#[cfg(target_arch = "x86_64")]
const SSE4_2_MASK: u32 = 1 << 5;
#[cfg(target_arch = "x86_64")]
const AVX_MASK: u32 = 1 << 6;
#[cfg(target_arch = "x86_64")]
const AVX2_MASK: u32 = 1 << 7;
#[cfg(target_arch = "x86_64")]
const FMA_MASK: u32 = 1 << 8;

#[cfg(target_arch = "aarch64")]
const NEON_MASK: u32 = 1 << 0;

// -------------------- Initialization Impl ------------------------
impl AvkSystemInfo {
  pub fn new() -> AvkSystemInfo {
    #[cfg(target_arch = "x86_64")]
    {
      use core::arch::x86_64::{__cpuid, __cpuid_count};
      let mut arch_features: u32 = 0;

      // It's safe to call CPUID, but wrapped in an unsafe block because
      // core::arch intrinsics require it.
      unsafe {
        // First, check the maximum supported leaf to avoid querying invalid leaves
        let max_leaf = __cpuid(0).eax;

        if max_leaf >= 1 {
          // Leaf 1 gives us ECX and EDX feature flags
          let leaf_1 = __cpuid(1);

          if (leaf_1.edx & (1 << 25)) != 0 {
            arch_features |= SSE_MASK;
          }
          if (leaf_1.edx & (1 << 26)) != 0 {
            arch_features |= SSE2_MASK;
          }

          if (leaf_1.ecx & (1 << 0)) != 0 {
            arch_features |= SSE3_MASK;
          }
          if (leaf_1.ecx & (1 << 9)) != 0 {
            arch_features |= SSSE3_MASK;
          }
          if (leaf_1.ecx & (1 << 12)) != 0 {
            arch_features |= FMA_MASK;
          }
          if (leaf_1.ecx & (1 << 19)) != 0 {
            arch_features |= SSE4_1_MASK;
          }
          if (leaf_1.ecx & (1 << 20)) != 0 {
            arch_features |= SSE4_2_MASK;
          }
          if (leaf_1.ecx & (1 << 28)) != 0 {
            arch_features |= AVX_MASK;
          }
        }

        if max_leaf >= 7 {
          // Leaf 7, Sub-leaf 0 gives us extended EBX feature flags like AVX2
          let leaf_7 = __cpuid_count(7, 0);

          if (leaf_7.ebx & (1 << 5)) != 0 {
            arch_features |= AVX2_MASK;
          }
        }
      }

      AvkSystemInfo { arch_features }
    }
    #[cfg(target_arch = "aarch64")]
    {
      let mut arch_features: u32 = 0;
      // aarch64 = alias for ARMv8-A 64-bit, for which NEON support is mandatory
      // - macOS terminal: `sysctl "hw.optional.neon"`. For more, `sysctl "hw.optional"`
      arch_features |= NEON_MASK;

      // Apple Silicon (M1 - M4) doesn't support SVE extensions, while on linux it relies on either
      // - reading register `ID_AA64PFR_EL1`, which is only accessible in EL1 (which is kernel mode, while EL0 is user mode)
      //   - linux kernel 4.something should detect you want to read SIMD features, hence traps to kernel and returns to you the value, but sketchy.
      // - linux API: either parse pseudofile `/proc/self/auxv` or use `getauxval`
      // TODO: Linux

      AvkSystemInfo { arch_features }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      panic!("Unsupported architecture");
    }
  }

  #[inline]
  #[cfg(target_arch = "x86_64")]
  pub fn has_sse(&self) -> bool {
    self.arch_features & SSE_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  pub fn has_sse2(&self) -> bool {
    self.arch_features & SSE2_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  pub fn has_sse3(&self) -> bool {
    self.arch_features & SSE3_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  pub fn has_ssse3(&self) -> bool {
    self.arch_features & SSSE3_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  pub fn has_sse4_1(&self) -> bool {
    self.arch_features & SSE4_1_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  pub fn has_sse4_2(&self) -> bool {
    self.arch_features & SSE4_2_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  pub fn has_fma(&self) -> bool {
    self.arch_features & FMA_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  pub fn has_avx(&self) -> bool {
    self.arch_features & AVX_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  pub fn has_avx2(&self) -> bool {
    self.arch_features & AVX2_MASK != 0
  }

  #[inline]
  #[cfg(target_arch = "aarch64")]
  pub fn has_neon(&self) -> bool {
    self.arch_features & NEON_MASK != 0
  }
}

// -------------------- Unit Testing Implementation ------------------------
#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn should_support_baseline_on_any_reasonable_machine() {
    let system_info = AvkSystemInfo::new();
    #[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
    {
      println!(
        concat!(
          "Compare the following with CPU-Z\n",
          "Thing: {:x}\n",
          "SSE: {}\n",
          "SSE2: {}\n",
          "SSE3: {}\n",
          "SSSE3: {}\n",
          "SSE4.1: {}\n",
          "SSE4.2: {}\n",
          "FMA: {}\n",
          "AVX: {}\n",
          "AVX2: {}\n",
        ),
        system_info.arch_features,
        (&system_info).has_sse(),
        (&system_info).has_sse2(),
        (&system_info).has_sse3(),
        (&system_info).has_ssse3(),
        (&system_info).has_sse4_1(),
        (&system_info).has_sse4_2(),
        (&system_info).has_fma(),
        (&system_info).has_avx(),
        (&system_info).has_avx2()
      );
    }
    #[cfg(all(target_arch = "aarch64", not(target_os = "none")))]
    {
      println!("Compare the following with sysctl on macOS:\n  feat: {:x}\n NEON: {}", system_info.arch_features, system_info.has_neon());
    }
  }
}
