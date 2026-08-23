//! lib module.

// disable std only for non tests
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

// crash compilation on 32 bit machines
#[cfg(not(target_pointer_width = "64"))]
compile_error!("Target Must be a 64-bit machine");

#[cfg(any(test, feature = "std"))]
extern crate std;

// panic_handler and global allocator declared for non tests at cdylib/FFI level
extern crate alloc;
extern crate core;

use spin::once;

use crate::os::debug;
#[cfg(windows)]
use windows::{
  Win32::Foundation::HANDLE,
  Win32::System::Threading::{GetCurrentProcess, TerminateProcess},
};

// --------------- Centralized Panic Handler Implementation ---------
#[inline]
/// TODO: Document this item
pub fn panic_handler_impl() -> ! {
  #[cfg(debug_assertions)]
  {
    debug::print_stacktrace();
  }
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

// -------------------- Static Storage ------------------------------
/// TODO: Document this item
pub static SYSTEM_INFO: once::Once<AvkSystemInfo> = once::Once::new();

pub mod math;

// -------------------- Initialization Types ------------------------
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// TODO: Document this item
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
impl Default for AvkSystemInfo {
  fn default() -> Self {
    Self::new()
  }
}

impl AvkSystemInfo {
  /// TODO: Document this item
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
  /// TODO: Document this item
  pub fn has_sse(&self) -> bool {
    self.arch_features & SSE_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  /// TODO: Document this item
  pub fn has_sse2(&self) -> bool {
    self.arch_features & SSE2_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  /// TODO: Document this item
  pub fn has_sse3(&self) -> bool {
    self.arch_features & SSE3_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  /// TODO: Document this item
  pub fn has_ssse3(&self) -> bool {
    self.arch_features & SSSE3_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  /// TODO: Document this item
  pub fn has_sse4_1(&self) -> bool {
    self.arch_features & SSE4_1_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  /// TODO: Document this item
  pub fn has_sse4_2(&self) -> bool {
    self.arch_features & SSE4_2_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  /// TODO: Document this item
  pub fn has_fma(&self) -> bool {
    self.arch_features & FMA_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  /// TODO: Document this item
  pub fn has_avx(&self) -> bool {
    self.arch_features & AVX_MASK != 0
  }
  #[inline]
  #[cfg(target_arch = "x86_64")]
  /// TODO: Document this item
  pub fn has_avx2(&self) -> bool {
    self.arch_features & AVX2_MASK != 0
  }

  #[inline]
  #[cfg(target_arch = "aarch64")]
  /// TODO: Document this item
  pub fn has_neon(&self) -> bool {
    self.arch_features & NEON_MASK != 0
  }
}

// -------------------- Modules ------------------------
pub mod os;

pub mod hash {
  use core::{
    hash::{Hash, Hasher},
    marker,
  };

  /// TODO: Document this item
  pub struct FnvHasher {
    hash: u64,
  }

  impl Default for FnvHasher {
    fn default() -> Self {
      Self::new()
    }
  }

  impl FnvHasher {
    /// TODO: Document this item
    pub const fn new() -> Self {
      Self {
        hash: 0xcbf29ce484222325, // FNV offset basis
      }
    }
  }

  impl Hasher for FnvHasher {
    fn write(&mut self, bytes: &[u8]) {
      for b in bytes {
        self.hash ^= *b as u64;
        self.hash = self.hash.wrapping_mul(0x100000001b3);
      }
    }

    fn finish(&self) -> u64 {
      self.hash
    }
  }

  /// TODO: Document this item
  pub struct Key<T>
  where
    T: Hash,
  {
    v: u64,
    _marker: marker::PhantomData<T>,
  }

  impl<T> Key<T>
  where
    T: Hash,
  {
    fn new(value: &T) -> Self {
      let mut hasher = FnvHasher::new();
      value.hash(&mut hasher);
      Self {
        v: hasher.finish(),
        _marker: marker::PhantomData,
      }
    }
  }
}

// -------------------- Unit Testing Implementation ------------------------
#[cfg(test)]
mod tests;

