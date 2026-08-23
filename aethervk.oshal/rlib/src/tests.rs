
use super::*;
use std::println;

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
    println!(
      "Compare the following with sysctl on macOS:\n  feat: {:x}\n NEON: {}",
      system_info.arch_features,
      system_info.has_neon()
    );
  }
}
