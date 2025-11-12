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

}  // namespace avk