#pragma once

#include <atomic>

namespace avk {

template <typename T>
inline constexpr T max(T t0, T t1) {
  return t0 > t1 ? t0 : t1;
}

template <typename T>
inline constexpr T min(T t0, T t1) {
  return t0 < t1 ? t0 : t1;
}

/// Busy-wait until `flag` becomes `value`
inline void waitAtomicCpuIntensive(std::atomic_bool& flag,
                                   bool value) noexcept {
  bool expected = value;
  // loop: on failure, use relaxed memory barrier, ie no synchronization among
  // CPU caches. on success repeat test with acquire semantics
  // more info on "sync among caches" also called cache coherency
  // https://developer.arm.com/documentation/den0013/0400/Multi-core-processors/Cache-coherency/MESI-and-MOESI-protocols
  while (!flag.compare_exchange_weak(
      expected, !value, std::memory_order_acquire, std::memory_order_relaxed)) {
    // handle spurious failures
    expected = value;
    // CPU-intensive spin: doesn't yield CPU, doesn't sleep. Assumes it should
    // be quick
  }
}

}  // namespace avk