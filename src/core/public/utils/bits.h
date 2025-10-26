#pragma once

#include <cstdint>

namespace avk {

// WARNING: only for POT
template <size_t Base>
inline constexpr size_t nextMultipleOf(size_t x) {
  return (x + Base - 1) & (~size_t(Base - 1));
}

}  // namespace avk